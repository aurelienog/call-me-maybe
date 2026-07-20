from __future__ import annotations

from dataclasses import dataclass, field

from ..models import FunctionRegistry


@dataclass
class ConsumptionContext:
    """
    Semantic state of the constrained decoder.

    This object is updated only when a DFA has been completely accepted.
    It never stores tokenizer state.
    """

    registry: FunctionRegistry

    #
    # Current function call.
    #

    current_function: str | None = None

    #
    # Current parameter being decoded.
    #

    current_parameter: str | None = None

    #
    # Parameters already emitted.
    #

    written_parameters: set[str] = field(
        default_factory=set,
    )

    def reset(self) -> None:
        """
        Reset the semantic decoding context.
        """

        self.current_function = None
        self.current_parameter = None
        self.written_parameters.clear()

    def begin_function(
        self,
        function_name: str,
    ) -> None:
        """
        Start decoding a new function call.
        """

        self.current_function = function_name
        self.current_parameter = None
        self.written_parameters.clear()

    def begin_parameter(
        self,
        parameter_name: str,
    ) -> None:
        """
        Start decoding a parameter.
        """

        self.current_parameter = parameter_name

    def finish_parameter(self) -> None:
        """
        Mark the current parameter as completed.
        """

        if self.current_parameter is None:
            return

        self.written_parameters.add(
            self.current_parameter,
        )

        self.current_parameter = None

    def remaining_parameters(self) -> tuple[str, ...]:
        """
        Parameters that have not yet been emitted.
        """

        if self.current_function is None:
            return ()

        return tuple(
            parameter
            for parameter in self.registry.parameters(
                self.current_function,
            )
            if parameter not in self.written_parameters
        )

    def has_remaining_parameters(self) -> bool:
        """
        Whether there are still parameters to emit.
        """

        return bool(
            self.remaining_parameters()
        )

    def parameter_completed(
        self,
        parameter_name: str,
    ) -> bool:
        """
        Whether a parameter has already been emitted.
        """

        return (
            parameter_name in self.written_parameters
        )
