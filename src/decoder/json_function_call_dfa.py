from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .char_dfa import CharState
from typing import cast


class JsonFunctionCallDFA(BaseModel):
    """
    Immutable token-level packaged DFA for grammar-constrained decoding.

    Provides O(1) state transitions and fast logit masking during generation.

    Attributes:
        logit_masks: Matrix (num_states, vocab_size) of type float32
        containing 0.0 for valid tokens and -inf for invalid tokens.

        transitions: Matrix (num_states, vocab_size) of type int32
        containing the next state ID (-1 for invalid transitions).

        start_state: The initial state ID of the DFA.
        accept_states: Set of state IDs that represent
        valid final/accepting states.
        idx_to_state: Mapping from integer state index to its corresponding
        CharState enum.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    # Matrix (num_states, vocab_size) of type float32
    # (contains 0.0 for valid, -inf for invalid)
    logit_masks: np.typing.NDArray[np.float32]

    # Matrix (num_states, vocab_size) of type int32
    # containing the next state ID
    transitions: np.typing.NDArray[np.int32]

    start_state: int
    accept_states: set[int]
    idx_to_state: dict[int, CharState] = Field(default_factory=dict)

    def is_accept_state(self, state: int) -> bool:
        """
        Check whether the given state is an accepting final state.

        Args:
            state: The state ID to check.

        Returns:
            True if the state is in the accept set, False otherwise.
        """
        return state in self.accept_states

    def get_mask(self, state: int) -> np.typing.NDArray[np.float32]:
        """
        Return the logit mask vector for the given state in O(1) time.

        Args:
            state: The state ID whose mask is requested.

        Returns:
            A 1D numpy array of shape (vocab_size,) with logit additive masks.
        """
        return cast(np.typing.NDArray[np.float32], self.logit_masks[state])

    def next_state(self, current_state: int, token_id: int) -> int:
        """
        Perform a state transition for a given token ID in O(1) time.

        Args:
            current_state: The current state ID.
            token_id: The selected token ID.

        Returns:
            The next state ID.

        Raises:
            ValueError: If the transition is invalid (i.e., evaluates to -1).
        """
        next_s = int(self.transitions[current_state, token_id])
        if next_s == -1:
            raise ValueError(
                f"Invalid transition from state {current_state} with token"
                f" {token_id}"
            )
        return next_s
