from __future__ import annotations

from enum import IntEnum
from typing import ClassVar

from pydantic import ConfigDict

from .consume_result import (
    ConsumeResult,
    ConsumptionStatus,
)
from .dfa import DFA


class StringState(IntEnum):
    START = 0
    STRING = 1
    ESCAPE = 2
    UNICODE_1 = 3
    UNICODE_2 = 4
    UNICODE_3 = 5
    UNICODE_4 = 6
    COMPLETE = 7


class StringDFA(DFA):

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    FINAL_STATES: ClassVar[frozenset[StringState]] = frozenset(
        {
            StringState.COMPLETE,
        }
    )

    def start(
        self,
    ) -> StringState:

        return StringState.START

    def is_final(
        self,
        state: StringState,
    ) -> bool:

        return state == StringState.COMPLETE

    def consume(
        self,
        state: StringState,
        token: str,
        offset: int,
    ) -> ConsumeResult:

        current = state

        while True:

            #
            # Once the string is complete, ignore any JSON whitespace
            # before the next grammar element.
            #

            if current == StringState.COMPLETE:

                offset = self._skip_whitespace(
                    token,
                    offset,
                )

                if offset >= len(token):

                    return ConsumeResult(
                        state=current,
                        offset=offset,
                        status=ConsumptionStatus.ACCEPTED,
                    )

                return ConsumeResult(
                    state=current,
                    offset=offset,
                    status=ConsumptionStatus.BLOCKED,
                )

            #
            # End of tokenizer token.
            #

            if offset >= len(token):
                break

            c = token[offset]

            match current:

                #
                # Opening quote.
                #

                case StringState.START:

                    if c != '"':
                        return ConsumeResult(
                            state=current,
                            offset=offset,
                            status=ConsumptionStatus.ERROR,
                        )

                    current = StringState.STRING

                #
                # Inside string.
                #

                case StringState.STRING:

                    if c == '"':
                        current = StringState.COMPLETE

                    elif c == "\\":
                        current = StringState.ESCAPE

                    elif ord(c) < 0x20:
                        return ConsumeResult(
                            state=current,
                            offset=offset,
                            status=ConsumptionStatus.ERROR,
                        )

                    #
                    # Ordinary character.
                    #

                #
                # Escape sequence.
                #

                case StringState.ESCAPE:

                    if c in '"\\/bfnrt':
                        current = StringState.STRING

                    elif c == "u":
                        current = StringState.UNICODE_1

                    else:
                        return ConsumeResult(
                            state=current,
                            offset=offset,
                            status=ConsumptionStatus.ERROR,
                        )

                #
                # Unicode escape.
                #

                case (
                    StringState.UNICODE_1
                    | StringState.UNICODE_2
                    | StringState.UNICODE_3
                    | StringState.UNICODE_4
                ):

                    if c.lower() not in "0123456789abcdef":

                        return ConsumeResult(
                            state=current,
                            offset=offset,
                            status=ConsumptionStatus.ERROR,
                        )

                    match current:

                        case StringState.UNICODE_1:
                            current = StringState.UNICODE_2

                        case StringState.UNICODE_2:
                            current = StringState.UNICODE_3

                        case StringState.UNICODE_3:
                            current = StringState.UNICODE_4

                        case StringState.UNICODE_4:
                            current = StringState.STRING

            offset += 1

        #
        # End of tokenizer token.
        #

        if current == StringState.COMPLETE:

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