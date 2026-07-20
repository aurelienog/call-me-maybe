from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from .consume_result import (
    ConsumeResult,
    ConsumptionStatus,
)
from .dfa import DFA


class ChoiceState(BaseModel):

    model_config = ConfigDict(
        frozen=True,
    )

    consumed: int

    candidates: frozenset[int]


class ChoiceDFA(DFA):
    """
    DFA accepting one of several fixed literals.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    literals: tuple[str, ...] = Field(
        default_factory=tuple,
    )

    final_lengths: frozenset[int] = Field(
        default_factory=frozenset,
    )

    def model_post_init(
        self,
        __context,
    ) -> None:

        object.__setattr__(
            self,
            "final_lengths",
            frozenset(
                len(literal)
                for literal in self.literals
            ),
        )

    def start(
        self,
    ) -> ChoiceState:

        return ChoiceState(
            consumed=0,
            candidates=frozenset(
                range(len(self.literals))
            ),
        )

    def is_final(
        self,
        state: ChoiceState,
    ) -> bool:

        return state.consumed in self.final_lengths

    def accepted_literal(
        self,
        state: ChoiceState,
    ) -> str:

        if not self.is_final(state):
            raise RuntimeError(
                "ChoiceDFA is not in a final state."
            )

        for i in state.candidates:

            literal = self.literals[i]

            if len(literal) == state.consumed:
                return literal

        raise RuntimeError(
            "No accepted literal found."
        )

    def consume(
        self,
        state: ChoiceState,
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

            next_candidates = {
                i
                for i in current.candidates
                if (
                    current.consumed < len(self.literals[i])
                    and self.literals[i][current.consumed] == c
                )
            }

            #
            # No literal matches.
            #

            if not next_candidates:

                if self.is_final(current):
                    return ConsumeResult(
                        state=current,
                        offset=offset,
                        status=ConsumptionStatus.ACCEPTED,
                    )

                return ConsumeResult(
                    state=current,
                    offset=offset,
                    status=ConsumptionStatus.ERROR,
                )

            current = ChoiceState(
                consumed=current.consumed + 1,
                candidates=frozenset(
                    next_candidates,
                ),
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