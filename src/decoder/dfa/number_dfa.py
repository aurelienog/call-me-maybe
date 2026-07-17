from enum import IntEnum
from collections.abc import Iterable
from typing import ClassVar

from .dfa import DFA, DFATransition


DELIMITERS = {",", "}", "]", " ", "\t", "\n"}


class NumberState(IntEnum):
    START = 0

    MINUS = 1

    ZERO = 2
    INTEGER = 3

    DOT = 4
    FRACTION = 5

    EXPONENT = 6
    EXP_SIGN = 7
    EXP_NUMBER = 8


class NumberDFA(DFA):

    FINAL_STATES: ClassVar = {
        NumberState.ZERO,
        NumberState.INTEGER,
        NumberState.FRACTION,
        NumberState.EXP_NUMBER,
    }

    def states(self) -> Iterable[int]:
        return NumberState

    def _next_state(
        self,
        state: NumberState,
        c: str,
    ) -> NumberState | None:

        match state:

            case NumberState.START:

                if c == "-":
                    return NumberState.MINUS

                if c == "0":
                    return NumberState.ZERO

                if c.isdigit() and c != "0":
                    return NumberState.INTEGER

            case NumberState.MINUS:

                if c == "0":
                    return NumberState.ZERO

                if c.isdigit() and c != "0":
                    return NumberState.INTEGER

            case NumberState.ZERO:

                if c == ".":
                    return NumberState.DOT

                if c in "eE":
                    return NumberState.EXPONENT

            case NumberState.INTEGER:

                if c.isdigit():
                    return NumberState.INTEGER

                if c == ".":
                    return NumberState.DOT

                if c in "eE":
                    return NumberState.EXPONENT

            case NumberState.DOT:

                if c.isdigit():
                    return NumberState.FRACTION

            case NumberState.FRACTION:

                if c.isdigit():
                    return NumberState.FRACTION

                if c in "eE":
                    return NumberState.EXPONENT

            case NumberState.EXPONENT:

                if c in "+-":
                    return NumberState.EXP_SIGN

                if c.isdigit():
                    return NumberState.EXP_NUMBER

            case NumberState.EXP_SIGN:

                if c.isdigit():
                    return NumberState.EXP_NUMBER

            case NumberState.EXP_NUMBER:

                if c.isdigit():
                    return NumberState.EXP_NUMBER

        return None

    def _transition(
        self,
        state: int,
        token: str,
    ) -> DFATransition:

        if not token:
            return self.invalid_transition

        current = NumberState(state)

        for i, c in enumerate(token):

            next_state = self._next_state(current, c)

            if next_state is not None:
                current = next_state
                continue

            # el número ya estaba completo y el resto pertenece
            # al siguiente símbolo JSON
            if (
                current in self.FINAL_STATES
                and c in DELIMITERS
            ):
                return DFATransition(
                    next_state=current,
                    valid=True,
                    complete=True,
                    consumed=i,
                )

            return self.invalid_transition

        return DFATransition(
            next_state=current,
            valid=True,
            complete=current in self.FINAL_STATES,
            consumed=len(token),
        )
