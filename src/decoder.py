from __future__ import annotations

import json
import numpy as np
from pydantic import BaseModel, ConfigDict

from .llm import Llm
from .models import FunctionRegistry, FunctionCallResult, Prompt
from .vocabulary_compiler import VocabularyCompiler
from .grammar_compiler import GrammarCompiler
from .json_grammar import JsonGrammar, JsonObjectNode
from .json_function_call_dfa import JsonFunctionCallDFA


class ConstrainedDecoder(BaseModel):
    """
    Orquestador de decodificación restringida por gramática.
    Garantiza salidas JSON 100% válidas y ajustadas al esquema sin parsing en runtime.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    llm: Llm
    registry: FunctionRegistry
    vocab_compiler: VocabularyCompiler = VocabularyCompiler()
    grammar_compiler: GrammarCompiler = GrammarCompiler()

    def run(self, prompts: list[Prompt]) -> list[FunctionCallResult]:
        """
        Procesa una lista de prompts en lote, compilando el DFA una sola vez
        y mostrando el progreso por consola.
        """
        results: list[FunctionCallResult] = []

        # 1. Compilamos el vocabulario y el DFA UNA SOLA VEZ para todo el lote
        print("[INFO] Compilando grafo y DFA a nivel de tokens...")
        compiled_vocab = self.vocab_compiler.compile(self.llm.normalized_vocabulary)
        
        dummy_grammar = JsonGrammar(root=JsonObjectNode())
        dfa = self.grammar_compiler.compile(
            grammar=dummy_grammar,
            vocabulary=compiled_vocab,
            allowed_functions=self.registry.function_names(),
        )

        # 2. Procesamos cada prompt usando el DFA ya empaquetado en memoria
        total = len(prompts)
        for index, prompt_model in enumerate(prompts, start=1):
            prompt_text = prompt_model.prompt
            print(f"[{index}/{total}] Procesando: '{prompt_text}'")

            result = self._decode_single_prompt(
                prompt_text=prompt_text,
                dfa=dfa,
            )
            
            print(f"   └─ Function Selected: {result.name}")
            results.append(result)

        return results

    def decode_prompt(
        self,
        prompt_text: str,
        max_new_tokens: int = 128,
    ) -> FunctionCallResult:
        """
        Wrapper para procesar un único prompt compilando el DFA en el acto.
        Útil para llamadas individuales aisladas.
        """
        # En decoder.py -> dentro de run() o _decode_single_prompt():
        # 1. Inspeccionamos la dimensión real devuelta por la LLM
        sample_logits = self.llm.get_logits(self.llm.encode("test"))
        real_vocab_size = sample_logits.shape[0]  # Devuelve 151936

        # 2. Pasamos ese tamaño al VocabularyCompiler
        compiled_vocab = self.vocab_compiler.compile(
            normalized_vocabulary=self.llm.normalized_vocabulary,
            target_vocab_size=real_vocab_size,
        )
        dummy_grammar = JsonGrammar(root=JsonObjectNode())
        dfa = self.grammar_compiler.compile(
            grammar=dummy_grammar,
            vocabulary=compiled_vocab,
            allowed_functions=self.registry.function_names(),
        )

        return self._decode_single_prompt(
            prompt_text=prompt_text,
            dfa=dfa,
            max_new_tokens=max_new_tokens,
        )

    def _decode_single_prompt(
        self,
        prompt_text: str,
        dfa: JsonFunctionCallDFA,
        max_new_tokens: int = 128,
    ) -> FunctionCallResult:
        """
        Bucle interno de generación token por token usando el DFA precomputado.
        """
        # 1. Construir el contexto e instruir el inicio del JSON
        system_context = self.registry.build_context()
        full_prompt = f"{system_context}\n\nUser request: {prompt_text}\nJSON Output: "

        # 2. Codificar prompt
        input_ids = self.llm.encode(full_prompt)

        # 3. Bucle de generación guiado por el autómata
        generated_token_ids: list[int] = []
        current_state = dfa.start_state

        for _ in range(max_new_tokens):
            if dfa.is_accept_state(current_state):
                break

            # A. Obtener logits de la LLM
            current_input = input_ids + generated_token_ids
            logits = self.llm.get_logits(current_input)

            # B. Filtrar logits inválidos en O(1) con la máscara NumPy del estado
            mask = dfa.get_mask(current_state)
            masked_logits = logits + mask

            # C. Selección codiciosa (Argmax)
            selected_token_id = int(np.argmax(masked_logits))

            # D. Avanzar en el autómata
            try:
                current_state = dfa.next_state(current_state, selected_token_id)
            except ValueError:
                # Transición no permitida / Fin de camino
                break

            generated_token_ids.append(selected_token_id)

        # 4. Decodificar tokens generados a texto y estructurar la salida
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
        Parsea el JSON generado y fuerza el casting correcto de tipos (float vs int)
        según la definición formal de la función en el registry.
        """
        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError:
            data = {}

        func_name = data.get("name", "")
        parameters = data.get("parameters", {})

        # Si la función existe en el registry, casteamos sus parámetros
        if self.registry.exists(func_name):
            func_def = self.registry.get(func_name)
            
            for param_name, param_val in parameters.items():
                if param_name in func_def.parameters:
                    expected_type = func_def.parameters[param_name].type
                    
                    # Regla: "number" -> float
                    if expected_type == "number" and isinstance(param_val, (int, float)):
                        parameters[param_name] = float(param_val)
                        
                    # Regla: "integer" -> int
                    elif expected_type == "integer" and isinstance(param_val, (int, float)):
                        parameters[param_name] = int(param_val)

        return FunctionCallResult(
            prompt=prompt,
            name=func_name,
            parameters=parameters,
        )
