from __future__ import annotations

import json
import numpy as np
from pydantic import BaseModel, ConfigDict

from ..llm import Llm
from ..models import FunctionRegistry, FunctionCallResult, Prompt
from .vocabulary_compiler import VocabularyCompiler
from .grammar_compiler import GrammarCompiler
from .json_function_call_dfa import JsonFunctionCallDFA
from ..utils import timer


class ConstrainedDecoder(BaseModel):
    """
    Grammar-constrained decoding orchestrator.

    Guarantees 100% valid JSON outputs conforming strictly to the
    defined schema without runtime parsing overhead during token generation.

    Attributes:
        llm: Wrapper instance around the target Large Language Model SDK.
        registry: Function registry containing available function definitions.
        vocab_compiler: Vocabulary compiler for pre-analyzing token metadata.
        grammar_compiler: Compiler responsible for generating token-level DFAs.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    llm: Llm
    registry: FunctionRegistry
    vocab_compiler: VocabularyCompiler = VocabularyCompiler()
    grammar_compiler: GrammarCompiler = GrammarCompiler()

    @timer
    def run(self, prompts: list[Prompt]) -> list[FunctionCallResult]:
        """
        Process a batch of prompts, compiling the DFA once
        and logging progress.

        Args:
            prompts: List of prompt objects to process.

        Returns:
            List of FunctionCallResult objects containing
            extracted function calls.
        """
        results: list[FunctionCallResult] = []

        print("[INFO] Compiling token-level graph and DFA...")
        compiled_vocab = self.vocab_compiler.compile(
            self.llm.normalized_vocabulary
        )
        dfa = self.grammar_compiler.compile(
            vocabulary=compiled_vocab,
            allowed_functions=self.registry.function_names(),
        )

        total = len(prompts)
        for index, prompt_model in enumerate(prompts, start=1):
            prompt_text = prompt_model.prompt
            print(f"[{index}/{total}] Processing: '{prompt_text}'")

            result = self._decode_single_prompt(
                prompt_text=prompt_text,
                dfa=dfa,
            )

            print(f"   └─ Function Selected: {result.name}")
            results.append(result)

        return results

    def _decode_single_prompt(
        self,
        prompt_text: str,
        dfa: JsonFunctionCallDFA,
        max_new_tokens: int = 128,
    ) -> FunctionCallResult:
        """
        Internal token-by-token generation loop guided by the precomputed DFA.

        Args:
            prompt_text: Input prompt text from the user.
            dfa: Precomputed token-level deterministic finite automaton.
            max_new_tokens: Maximum number of tokens to generate.

        Returns:
            A FunctionCallResult object containing decoded function details.
        """
        # 1. Build context prompt and instruct the JSON start
        system_context = self.registry.build_context()
        full_prompt = (
            f"{system_context}\n\n"
            f"User request: {prompt_text}\n"
            "JSON Output: "
        )

        # 2. Encode prompt text into input IDs
        input_ids = self.llm.encode(full_prompt)

        # 3. Generation loop guided by the state machine
        generated_token_ids: list[int] = []
        current_state = dfa.start_state

        for _ in range(max_new_tokens):
            if dfa.is_accept_state(current_state):
                break

            # A. Compute next-token logits from LLM
            current_input = input_ids + generated_token_ids
            logits = self.llm.get_logits(current_input)

            # B. Mask invalid logits in O(1) using the state's NumPy mask
            mask = dfa.get_mask(current_state)
            masked_logits = logits + mask

            # C. Greedy selection (Argmax)
            selected_token_id = int(np.argmax(masked_logits))

            # D. Advance state transition in automaton
            try:
                current_state = dfa.next_state(
                    current_state,
                    selected_token_id
                )
            except ValueError:
                # Disallowed transition / End of valid path
                break

            generated_token_ids.append(selected_token_id)

        # 4. Decode generated token IDs into text and structure output result
        generated_json_str = self.llm.decode(generated_token_ids)

        return self._parse_output(
            prompt=prompt_text,
            raw_json=generated_json_str,
        )

    def _parse_output(
        self,
        prompt: str,
        raw_json: str,
    ) -> FunctionCallResult:
        """
        Parse generated JSON string and enforce correct parameter type casting
        (float vs int)
        according to the formal function specification in the registry.

        Args:
            prompt: Original prompt text.
            raw_json: Raw JSON output string generated by the model.

        Returns:
            The parsed and type-casted FunctionCallResult instance.
        """
        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError:
            data = {}

        func_name = data.get("name", "")
        parameters = data.get("parameters", {})

        # If the function exists in registry, perform dynamic type casting
        if self.registry.exists(func_name):
            func_def = self.registry.get(func_name)

            for param_name, param_val in parameters.items():
                if param_name in func_def.parameters:
                    expected_type = func_def.parameters[param_name].type

                    # Rule: "number" -> float
                    if expected_type == "number" and isinstance(param_val,
                                                                (int, float)):
                        parameters[param_name] = float(param_val)

                    # Rule: "integer" -> int
                    elif expected_type == "integer" and isinstance(
                            param_val, (int, float)):
                        parameters[param_name] = int(param_val)

        return FunctionCallResult(
            prompt=prompt,
            name=func_name,
            parameters=parameters,
        )
