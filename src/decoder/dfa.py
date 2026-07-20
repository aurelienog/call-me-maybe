from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict

from .consume_result import ConsumeResult


class DFA(
    BaseModel,
    ABC,
):
    """
    Base class for every deterministic finite automaton used by the
    constrained decoder.

    A DFA only knows how to consume characters.

    It never knows:
        - tokenizer boundaries
        - JSON grammar
        - decoder states
        - function registry
        - semantic context

    Those responsibilities belong to ConstrainedDecoder.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    @abstractmethod
    def start(
        self,
    ) -> Any:
        """
        Return the initial DFA state.
        """
        raise NotImplementedError

    @abstractmethod
    def is_final(
        self,
        state: Any,
    ) -> bool:
        """
        Return whether the supplied state is accepting.
        """
        raise NotImplementedError

    @abstractmethod
    def consume(
        self,
        state: Any,
        token: str,
        offset: int,
    ) -> ConsumeResult:
        """
        Consume characters from a tokenizer token.

        Parameters
        ----------
        state:
            Current DFA state.

        token:
            Complete tokenizer token.

        offset:
            First character that has not yet been consumed.

        Returns
        -------
        ConsumeResult
            New DFA state, next offset inside the tokenizer token,
            and the consumption status.
        """
        raise NotImplementedError


    def _skip_whitespace(
            self,
            token: str,
            offset: int,
        ) -> int:

        while offset < len(token) and token[offset] in " \t\r\n":
            offset += 1

        return offset