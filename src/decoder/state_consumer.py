from __future__ import annotations

from .choice_dfa import ChoiceDFA
from .consumption_context import ConsumptionContext
from .dfa import DFA
from .state import DecoderState


class StateConsumer:
    """
    Advances the semantic decoding state after a DFA has been
    successfully recognized.

    This class never consumes tokenizer characters.
    It only updates the grammar and semantic context.
    """

    def consume(
        self,
        decoder_state: DecoderState,
        machine: DFA,
        machine_state: object,
        context: ConsumptionContext,
    ) -> DecoderState:

        match decoder_state:

            #
            # {
            #

            case DecoderState.EXPECT_OPEN_BRACE:
                return DecoderState.EXPECT_NAME_KEY

            #
            # "name"
            #

            case DecoderState.EXPECT_NAME_KEY:
                return DecoderState.EXPECT_NAME_COLON

            #
            # :
            #

            case DecoderState.EXPECT_NAME_COLON:
                return DecoderState.EXPECT_FUNCTION_NAME

            #
            # Function name
            #

            case DecoderState.EXPECT_FUNCTION_NAME:

                assert isinstance(machine, ChoiceDFA)

                function_name = (
                    machine.accepted_literal(
                        machine_state,
                    )
                    .strip('"')
                )

                context.begin_function(
                    function_name,
                )

                return DecoderState.EXPECT_COMMA

            #
            # ,
            #

            case DecoderState.EXPECT_COMMA:
                return DecoderState.EXPECT_PARAMETERS_KEY

            #
            # "parameters"
            #

            case DecoderState.EXPECT_PARAMETERS_KEY:
                return DecoderState.EXPECT_PARAMETERS_COLON

            #
            # :
            #

            case DecoderState.EXPECT_PARAMETERS_COLON:
                return DecoderState.EXPECT_PARAMETERS_OPEN

            #
            # {
            #

            case DecoderState.EXPECT_PARAMETERS_OPEN:

                if context.has_remaining_parameters():
                    return DecoderState.EXPECT_PARAMETER_NAME

                return DecoderState.EXPECT_CLOSE_PARAMETERS

            #
            # Parameter name
            #

            case DecoderState.EXPECT_PARAMETER_NAME:

                assert isinstance(machine, ChoiceDFA)

                parameter_name = (
                    machine.accepted_literal(
                        machine_state,
                    )
                    .strip('"')
                )

                context.begin_parameter(
                    parameter_name,
                )

                return DecoderState.EXPECT_PARAMETER_COLON

            #
            # :
            #

            case DecoderState.EXPECT_PARAMETER_COLON:
                return DecoderState.EXPECT_PARAMETER_VALUE

            #
            # Parameter value
            #

            case DecoderState.EXPECT_PARAMETER_VALUE:

                context.finish_parameter()

                if context.has_remaining_parameters():
                    return DecoderState.EXPECT_PARAMETER_SEPARATOR

                return DecoderState.EXPECT_CLOSE_PARAMETERS

            #
            # ,
            #

            case DecoderState.EXPECT_PARAMETER_SEPARATOR:
                return DecoderState.EXPECT_PARAMETER_NAME

            #
            # }
            #

            case DecoderState.EXPECT_CLOSE_PARAMETERS:
                return DecoderState.EXPECT_CLOSE_OBJECT

            #
            # }
            #

            case DecoderState.EXPECT_CLOSE_OBJECT:
                return DecoderState.FINISHED

            #
            # Finished
            #

            case DecoderState.FINISHED:
                return DecoderState.FINISHED

        raise RuntimeError(
            f"Unhandled decoder state: {decoder_state}"
        )