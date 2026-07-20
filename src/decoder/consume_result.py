from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any


class ConsumptionStatus(Enum):
    """
    Result of attempting to consume characters from a tokenizer token.
    """

    #
    # The tokenizer token ended before the DFA could finish.
    # Continue with the next tokenizer token.
    #
    IN_PROGRESS = auto()

    #
    # The DFA finished exactly at the end of the tokenizer token.
    # The next grammar element starts in the next tokenizer token.
    #
    ACCEPTED = auto()

    #
    # The DFA finished before the tokenizer token ended.
    # The returned offset points to the first unconsumed character,
    # which belongs to the next grammar element.
    #
    BLOCKED = auto()

    #
    # The tokenizer token cannot be consumed by this DFA.
    #
    ERROR = auto()


@dataclass(frozen=True)
class ConsumeResult:
    """
    Result of consuming characters from a tokenizer token.

    Attributes
    ----------
    state
        DFA state after consuming every accepted character.

    offset
        Index of the first character that has NOT been consumed.

    status
        Outcome of the consumption.
    """

    state: Any

    #
    # First character not consumed.
    #
    offset: int

    status: ConsumptionStatus

    @property
    def in_progress(self) -> bool:
        return self.status is ConsumptionStatus.IN_PROGRESS

    @property
    def accepted(self) -> bool:
        return self.status is ConsumptionStatus.ACCEPTED

    @property
    def blocked(self) -> bool:
        return self.status is ConsumptionStatus.BLOCKED

    @property
    def error(self) -> bool:
        return self.status is ConsumptionStatus.ERROR