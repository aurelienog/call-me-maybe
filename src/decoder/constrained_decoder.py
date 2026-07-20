from __future__ import annotations

import json

import numpy as np

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from ..llm import Llm
from ..models import (
    FunctionCallResult,
    FunctionRegistry,
    Prompt,
)

from .consume_result import ConsumptionStatus
from .consumption_context import ConsumptionContext
from .state import DecoderState
from .state_consumer import StateConsumer
from .token_selector import TokenSelector
from .vocabulary_cache import VocabularyCache
from .dfa import DFA


class ConstrainedDecoder(BaseModel):

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    #
    # External components
    #

    llm: Llm = Field(
        default_factory=Llm,
    )

    registry: FunctionRegistry

    #
    # Cached decoding structures
    #

    vocabulary_cache: VocabularyCache = Field(
        default_factory=VocabularyCache,
    )

    state_consumer: StateConsumer = Field(
        default_factory=StateConsumer,
    )

    #
    # Built after the vocabulary cache.
    #

    token_selector: TokenSelector | None = None

    def model_post_init(
            self,
            __context,
        ) -> None:
        """
        Build every cached structure required by the decoder.

        This is executed once after Pydantic has created the model.
        """

        #
        # Build tokenizer trie, literal DFAs, function DFAs,
        # parameter DFAs and primitive DFAs.
        #

        self.vocabulary_cache.build(
            registry=self.registry,
            vocabulary=self.llm.normalized_vocabulary,
        )

        #
        # Build the component responsible for selecting
        # the active DFA during decoding.
        #

        self.token_selector = TokenSelector(
            llm=self.llm,
            registry=self.registry,
            vocabulary=self.vocabulary_cache,
        )

    def run(
            self,
            prompts: list[Prompt],
        ) -> list[FunctionCallResult]:

        context = self.registry.build_context()

        context_ids = self.llm.encode(
            context,
        )

        return [
            self.process_single_prompt(
                prompt,
                context_ids,
            )
            for prompt in prompts
        ]

    def process_single_prompt(
            self,
            prompt: Prompt,
            context_ids: list[int],
        ) -> FunctionCallResult:

        prompt_ids = self.llm.encode(
            (
                "\nUser request:\n"
                f"{prompt.prompt}\n\n"
                "Generate the function call:\n"
            )
        )

        output_ids = self.generate(
            context_ids + prompt_ids,
        )

        data = json.loads(
            self.llm.decode(
                output_ids,
            )
        )

        return FunctionCallResult(
            prompt=prompt.prompt,
            name=data["name"],
            parameters=data["parameters"],
        )

    def generate(
            self,
            input_ids: list[int],
        ) -> list[int]:

        assert self.token_selector is not None

        context = ConsumptionContext(
            registry=self.registry,
        )

        decoder_state = DecoderState.EXPECT_OPEN_BRACE

        machine = self.token_selector.machine(
            decoder_state,
            context,
        )

        machine_state = machine.start()

        tokens = list(input_ids)
        generated: list[int] = []

        while decoder_state != DecoderState.FINISHED:

            print()
            print("=" * 80)
            print("Decoder state :", decoder_state)
            print("Machine       :", type(machine).__name__)
            print("Machine state :", machine_state)

            logits = self.llm.get_logits(tokens)

            allowed = self.token_selector.get_allowed_tokens(
                machine,
                machine_state,
            )

            print(f"Allowed tokens: {len(allowed)}")

            if not allowed:
                raise RuntimeError(
                    f"No valid tokenizer token for {decoder_state}"
                )

            masked = self._mask_logits(
                logits,
                allowed,
            )

            token_id = self._select_next_token(masked)

            token = self.llm.normalized_token(token_id)

            print()
            print("Selected token")
            print(" id   :", token_id)
            print(" text :", repr(token))

            tokens.append(token_id)
            generated.append(token_id)

            offset = 0

            while offset < len(token):

                result = machine.consume(
                    machine_state,
                    token,
                    offset,
                )

                print(
                    "consume ->",
                    result.status.name,
                    "offset=",
                    result.offset,
                    "state=",
                    result.state,
                )
                match result.status:

                    #
                    # Token incompatible with DFA.
                    #

                    case ConsumptionStatus.ERROR:

                        raise RuntimeError(
                            f"Unexpected token {token!r}"
                        )

                    #
                    # DFA needs another tokenizer token.
                    #

                    case ConsumptionStatus.IN_PROGRESS:

                        machine_state = result.state
                        break

                    #
                    # DFA finished successfully.
                    #

                    case ConsumptionStatus.ACCEPTED:

                        machine_state = result.state
                        offset = result.offset

                        print("ACCEPTED -> switching DFA")

                        decoder_state = self.state_consumer.consume(
                            decoder_state=decoder_state,
                            machine=machine,
                            machine_state=machine_state,
                            context=context,
                        )

                        print("Next decoder state:", decoder_state)

                        if decoder_state == DecoderState.FINISHED:

                            if offset != len(token):
                                raise RuntimeError(
                                    "Grammar finished before tokenizer token ended."
                                )

                            return generated

                        machine = self.token_selector.machine(
                            decoder_state,
                            context,
                        )

                        machine_state = machine.start()

                        print(
                            "Next DFA:",
                            type(machine).__name__,
                        )

                        #
                        # IMPORTANT:
                        # offset IS NOT reset.
                        #

                    #
                    # Character belongs to another grammar element
                    # but current DFA is not in a final state.
                    #

                    case ConsumptionStatus.BLOCKED:

                        raise RuntimeError(
                            f"DFA blocked while decoding "
                            f"{decoder_state}. Remaining={token[offset:]!r}"
                        )

        return generated

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _mask_logits(
            self,
            logits: np.ndarray,
            allowed_tokens: frozenset[int],
        ) -> np.ndarray:

        masked = np.full_like(
            logits,
            -np.inf,
        )

        token_ids = np.fromiter(
            allowed_tokens,
            dtype=np.int64,
        )

        masked[token_ids] = logits[token_ids]

        return masked

    def _select_next_token(
            self,
            logits: np.ndarray,
        ) -> int:

        return int(np.argmax(logits))