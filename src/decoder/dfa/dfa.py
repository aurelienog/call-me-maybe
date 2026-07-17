from pydantic import BaseModel, ConfigDict, Field
from typing import ClassVar

from collections.abc import Iterable
from abc import ABC, abstractmethod


class DFATransition(BaseModel):
    model_config = ConfigDict(frozen=True)

    next_state: int
    valid: bool
    complete: bool
    consumed: int


class DFA(BaseModel, ABC):
    INVALID_STATE: ClassVar[int] = -1

    model_config = ConfigDict(arbitrary_types_allowed=True)

    transitions: dict[int, dict[int, DFATransition]] = Field(
        default_factory=dict
    )

    invalid_transition: ClassVar[DFATransition] = DFATransition(
            next_state=-1,
            valid=False,
            complete=False,
            consumed=0,
    )

    @abstractmethod
    def states(self) -> Iterable[int]:
        ...

    @abstractmethod
    def _transition(
        self,
        state: int,
        token: str,
    ) -> DFATransition:
        ...

    def _add_transition(
        self,
        state: int,
        token_id: int,
        next_state: int,
        consumed: int,
        complete: bool = False,
    ) -> None:

        self.transitions.setdefault(state, {})[token_id] = DFATransition(
            next_state=next_state,
            valid=True,
            complete=complete,
            consumed=consumed,
        )

    def step(
        self,
        state: int,
        token_id: int,
    ) -> DFATransition:
        return (
            self.transitions
            .get(state, {})
            .get(token_id, self.invalid_transition)
        )

    def build(self, vocabulary: dict[int, str]) -> None:

        for state in self.states():

            for token_id, token in vocabulary.items():

                transition = self._transition(state, token)

                if transition.valid:

                    self._add_transition(
                        state,
                        token_id,
                        transition.next_state,
                        transition.consumed,
                        transition.complete,
                    )
