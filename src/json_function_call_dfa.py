from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .char_dfa import CharState


class JsonFunctionCallDFA(BaseModel):
    """
    DFA inmutable empaquetado a nivel de tokens para la decodificación restringida.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    # Matriz (num_states, vocab_size) de tipo float32 (contiene 0.0 para válidos, -inf para inválidos)
    logit_masks: np.ndarray

    # Matriz (num_states, vocab_size) de tipo int32 con el id del siguiente estado
    transitions: np.ndarray

    start_state: int
    accept_states: set[int]
    idx_to_state: dict[int, CharState] = Field(default_factory=dict)

    def is_accept_state(self, state: int) -> bool:
        """Comprueba si el estado actual es de aceptación."""
        return state in self.accept_states

    def get_mask(self, state: int) -> np.ndarray:
        """
        Retorna el vector de máscara para los logits en O(1).
        """
        return self.logit_masks[state]

    def next_state(self, current_state: int, token_id: int) -> int:
        """
        Transición de estado por token_id en O(1).
        """
        next_s = int(self.transitions[current_state, token_id])
        if next_s == -1:
            raise ValueError(
                f"Transición inválida desde el estado {current_state} con el token {token_id}"
            )
        return next_s