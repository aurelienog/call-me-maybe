"""Token selection for constrained decoding."""

from __future__ import annotations

from ..llm import Llm
from ..models import FunctionRegistry

from .consumption_context import ConsumptionContext
from .dfa import DFA
from .state import DecoderState
from .vocabulary_cache import VocabularyCache


class TokenSelector:
    """
    Maps the current grammar state to its corresponding DFA and computes
    the tokenizer vocabulary compatible with the current DFA state.
    """

    def __init__(
        self,
        llm: Llm,
        registry: FunctionRegistry,
        vocabulary: VocabularyCache,
    ) -> None:

        self.llm = llm
        self.registry = registry
        self.vocabulary = vocabulary

    #
    # Vocabulary filtering
    #

    def get_allowed_tokens(
        self,
        machine: DFA,
        machine_state: object,
    ) -> frozenset[int]:
        """
        Return every tokenizer token that is compatible with the current
        DFA state.
        """

        return self.vocabulary.explorer.allowed_tokens(
            machine=machine,
            state=machine_state,
        )

    #
    # DFA selection
    #

    def machine(
        self,
        decoder_state: DecoderState,
        context: ConsumptionContext,
    ) -> DFA:

        match decoder_state:

            #
            # JSON structure
            #

            case DecoderState.EXPECT_OPEN_BRACE:
                return self.vocabulary.literal("{")

            case DecoderState.EXPECT_NAME_KEY:
                return self.vocabulary.literal('"name"')

            case DecoderState.EXPECT_NAME_COLON:
                return self.vocabulary.literal(":")

            case DecoderState.EXPECT_FUNCTION_NAME:
                return self.vocabulary.function_machine()

            case DecoderState.EXPECT_COMMA:
                return self.vocabulary.literal(",")

            case DecoderState.EXPECT_PARAMETERS_KEY:
                return self.vocabulary.literal('"parameters"')

            case DecoderState.EXPECT_PARAMETERS_COLON:
                return self.vocabulary.literal(":")

            case DecoderState.EXPECT_PARAMETERS_OPEN:
                return self.vocabulary.literal("{")

            #
            # Parameter names
            #

            case DecoderState.EXPECT_PARAMETER_NAME:

                assert context.current_function is not None

                return self.vocabulary.parameter_machine(
                    context.current_function,
                )

            case DecoderState.EXPECT_PARAMETER_COLON:
                return self.vocabulary.literal(":")

            #
            # Parameter values
            #

            case DecoderState.EXPECT_PARAMETER_VALUE:

                assert context.current_function is not None
                assert context.current_parameter is not None

                parameter_type = self.registry.parameter_type(
                    context.current_function,
                    context.current_parameter,
                )

                match parameter_type:

                    case "string":
                        return self.vocabulary.string_dfa

                    case "number":
                        return self.vocabulary.number_dfa

                    case "boolean":
                        return self.vocabulary.boolean_dfa

                raise RuntimeError(
                    f"Unsupported parameter type: {parameter_type}"
                )

            #
            # Between parameters
            #

            case DecoderState.EXPECT_PARAMETER_SEPARATOR:

                if context.has_remaining_parameters():
                    return self.vocabulary.literal(",")

                return self.vocabulary.literal("}")

            #
            # Closing braces
            #

            case DecoderState.EXPECT_CLOSE_PARAMETERS:
                return self.vocabulary.literal("}")

            case DecoderState.EXPECT_CLOSE_OBJECT:
                return self.vocabulary.literal("}")

            case DecoderState.FINISHED:

                raise RuntimeError(
                    "No DFA exists for the FINISHED state."
                )

        raise RuntimeError(
            f"Unhandled decoder state: {decoder_state}"
        )