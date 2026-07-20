from __future__ import annotations

from enum import IntEnum
from typing import ClassVar

from pydantic import ConfigDict

from .consume_result import (
    ConsumeResult,
    ConsumptionStatus,
)
from .dfa import DFA


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

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    FINAL_STATES: ClassVar[frozenset[NumberState]] = frozenset(
        {
            NumberState.ZERO,
            NumberState.INTEGER,
            NumberState.FRACTION,
            NumberState.EXP_NUMBER,
        }
    )

    def start(
        self,
    ) -> NumberState:

        return NumberState.START

    def is_final(
        self,
        state: NumberState,
    ) -> bool:

        return state in self.FINAL_STATES

    def consume(
        self,
        state: NumberState,
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
                # Beginning
                #

                case NumberState.START:

                    if c == "-":
                        current = NumberState.MINUS

                    elif c == "0":
                        current = NumberState.ZERO

                    elif "1" <= c <= "9":
                        current = NumberState.INTEGER

                    else:
                        return ConsumeResult(
                            state=current,
                            offset=offset,
                            status=ConsumptionStatus.ERROR,
                        )

                #
                # After '-'
                #

                case NumberState.MINUS:

                    if c == "0":
                        current = NumberState.ZERO

                    elif "1" <= c <= "9":
                        current = NumberState.INTEGER

                    else:
                        return ConsumeResult(
                            state=current,
                            offset=offset,
                            status=ConsumptionStatus.ERROR,
                        )

                #
                # Zero
                #

                case NumberState.ZERO:

                    if c == ".":
                        current = NumberState.DOT

                    elif c in "eE":
                        current = NumberState.EXPONENT

                    else:

                        return ConsumeResult(
                            state=current,
                            offset=offset,
                            status=ConsumptionStatus.BLOCKED,
                        )

                #
                # Integer
                #

                case NumberState.INTEGER:

                    if c.isdigit():
                        pass

                    elif c == ".":
                        current = NumberState.DOT

                    elif c in "eE":
                        current = NumberState.EXPONENT

                    else:

                        return ConsumeResult(
                            state=current,
                            offset=offset,
                            status=ConsumptionStatus.BLOCKED,
                        )

                #
                # Decimal point
                #

                case NumberState.DOT:

                    if c.isdigit():
                        current = NumberState.FRACTION

                    else:

                        return ConsumeResult(
                            state=current,
                            offset=offset,
                            status=ConsumptionStatus.ERROR,
                        )

                #
                # Fraction
                #

                case NumberState.FRACTION:

                    if c.isdigit():
                        pass

                    elif c in "eE":
                        current = NumberState.EXPONENT

                    else:

                        return ConsumeResult(
                            state=current,
                            offset=offset,
                            status=ConsumptionStatus.BLOCKED,
                        )

                #
                # Exponent marker
                #

                case NumberState.EXPONENT:

                    if c in "+-":
                        current = NumberState.EXP_SIGN

                    elif c.isdigit():
                        current = NumberState.EXP_NUMBER

                    else:

                        return ConsumeResult(
                            state=current,
                            offset=offset,
                            status=ConsumptionStatus.ERROR,
                        )

                #
                # Exponent sign
                #

                case NumberState.EXP_SIGN:

                    if c.isdigit():
                        current = NumberState.EXP_NUMBER

                    else:

                        return ConsumeResult(
                            state=current,
                            offset=offset,
                            status=ConsumptionStatus.ERROR,
                        )

                #
                # Exponent digits
                #

                case NumberState.EXP_NUMBER:

                    if c.isdigit():
                        pass

                    else:

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