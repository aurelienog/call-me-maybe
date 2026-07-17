from enum import IntEnum
from collections.abc import Iterable
from typing import ClassVar

from .dfa import DFA, DFATransition


class StringState(IntEnum):
    START = 0       # aún no hemos leído la "
    STRING = 1      # dentro del string
    ESCAPE = 2      # acabamos de leer '\'
    COMPLETE = 3    # hemos leído la comilla final


class StringDFA(DFA):

    FINAL_STATES: ClassVar = {
        StringState.COMPLETE,
    }

    def states(self) -> Iterable[int]:
        return StringState

    def _next_state(
        self,
        state: StringState,
        c: str,
    ) -> StringState | None:

        match state:

            case StringState.START:
                if c == '"':
                    return StringState.STRING

            case StringState.STRING:
                if c == "\\":
                    return StringState.ESCAPE

                if c == '"':
                    return StringState.COMPLETE

                if ord(c) < 0x20:
                    return None

                # cualquier otro carácter sigue dentro del string
                return StringState.STRING

            case StringState.ESCAPE:
                # después de '\' cualquier carácter vuelve al string
                return StringState.STRING

        return None

    def _transition(
        self,
        state: int,
        token: str,
    ) -> DFATransition:

        if not token:
            return self.invalid_transition

        current = StringState(state)

        for i, c in enumerate(token):

            next_state = self._next_state(current, c)

            if next_state is not None:
                current = next_state
                continue

            if current in self.FINAL_STATES:
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
