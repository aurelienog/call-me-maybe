from __future__ import annotations

from enum import IntEnum
from typing import ClassVar

from pydantic import ConfigDict

from .consume_result import (
    ConsumeResult,
    ConsumptionStatus,
)
from .dfa import DFA


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


class BooleanDFA(DFA):

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    FINAL_STATES: ClassVar[frozenset[BooleanState]] = frozenset(
        {
            BooleanState.TRUE,
            BooleanState.FALSE,
        }
    )

    def start(
        self,
    ) -> BooleanState:

        return BooleanState.START

    def is_final(
        self,
        state: BooleanState,
    ) -> bool:

        return state in self.FINAL_STATES

    def consume(
        self,
        state: BooleanState,
        token: str,
        offset: int,
    ) -> ConsumeResult:

        current = state

        while True:

            #
            # Ignore JSON whitespace.
            #

            offset = self._skip_whitespace(
                token,
                offset,
            )

            if offset >= len(token):
                break

            c = token[offset]

            match current:

                #
                # Start
                #

                case BooleanState.START:

                    if c == "t":
                        current = BooleanState.T

                    elif c == "f":
                        current = BooleanState.F

                    else:
                        return ConsumeResult(
                            state=current,
                            offset=offset,
                            status=ConsumptionStatus.ERROR,
                        )

                #
                # true
                #

                case BooleanState.T:

                    if c == "r":
                        current = BooleanState.TR

                    else:
                        return ConsumeResult(
                            state=current,
                            offset=offset,
                            status=ConsumptionStatus.ERROR,
                        )

                case BooleanState.TR:

                    if c == "u":
                        current = BooleanState.TRU

                    else:
                        return ConsumeResult(
                            state=current,
                            offset=offset,
                            status=ConsumptionStatus.ERROR,
                        )

                case BooleanState.TRU:

                    if c == "e":
                        current = BooleanState.TRUE

                    else:
                        return ConsumeResult(
                            state=current,
                            offset=offset,
                            status=ConsumptionStatus.ERROR,
                        )

                #
                # false
                #

                case BooleanState.F:

                    if c == "a":
                        current = BooleanState.FA

                    else:
                        return ConsumeResult(
                            state=current,
                            offset=offset,
                            status=ConsumptionStatus.ERROR,
                        )

                case BooleanState.FA:

                    if c == "l":
                        current = BooleanState.FAL

                    else:
                        return ConsumeResult(
                            state=current,
                            offset=offset,
                            status=ConsumptionStatus.ERROR,
                        )

                case BooleanState.FAL:

                    if c == "s":
                        current = BooleanState.FALS

                    else:
                        return ConsumeResult(
                            state=current,
                            offset=offset,
                            status=ConsumptionStatus.ERROR,
                        )

                case BooleanState.FALS:

                    if c == "e":
                        current = BooleanState.FALSE

                    else:
                        return ConsumeResult(
                            state=current,
                            offset=offset,
                            status=ConsumptionStatus.ERROR,
                        )

                #
                # Already finished.
                #

                case BooleanState.TRUE | BooleanState.FALSE:

                    return ConsumeResult(
                        state=current,
                        offset=offset,
                        status=ConsumptionStatus.BLOCKED,
                    )

            offset += 1

        #
        # End of tokenizer token.
        #

        if self.is_final(current):

            return ConsumeResult(
                state=current,
                offset=offset,
                status=ConsumptionStatus.ACCEPTED,
            )

        return ConsumeResult(
            state=current,
            offset=offset,
            status=ConsumptionStatus.IN_PROGRESS,
        )