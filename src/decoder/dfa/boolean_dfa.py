from enum import IntEnum
from collections.abc import Iterable
from typing import ClassVar

from .dfa import DFA, DFATransition


class BooleanState(IntEnum):
    START = 0

    T = 1
    TR = 2
    TRU = 3
    TRUE = 4

    F = 5
    FA = 6
    FAL = 7
    FALS = 8
    FALSE = 9


DELIMITERS = {",", "}", "]", " ", "\t", "\n"}


class BooleanDFA(DFA):

    FINAL_STATES: ClassVar = {
        BooleanState.TRUE,
        BooleanState.FALSE,
    }

    def states(self) -> Iterable[int]:
        return BooleanState

    def _next_state(
        self,
        state: BooleanState,
        c: str,
    ) -> BooleanState | None:

        match state:

            case BooleanState.START:
                if c == "t":
                    return BooleanState.T
                if c == "f":
                    return BooleanState.F

            case BooleanState.T:
                if c == "r":
                    return BooleanState.TR

            case BooleanState.TR:
                if c == "u":
                    return BooleanState.TRU

            case BooleanState.TRU:
                if c == "e":
                    return BooleanState.TRUE

            case BooleanState.F:
                if c == "a":
                    return BooleanState.FA

            case BooleanState.FA:
                if c == "l":
                    return BooleanState.FAL

            case BooleanState.FAL:
                if c == "s":
                    return BooleanState.FALS

            case BooleanState.FALS:
                if c == "e":
                    return BooleanState.FALSE

        return None

    def _transition(
        self,
        state: int,
        token: str,
    ) -> DFATransition:

        if not token:
            return self.invalid_transition

        current = BooleanState(state)

        for i, c in enumerate(token):

            next_state = self._next_state(current, c)

            if next_state is not None:
                current = next_state
                continue

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
