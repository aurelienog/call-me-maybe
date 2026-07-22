from __future__ import annotations

from enum import Enum, auto
from typing import Any, Optional


class CharState(Enum):
    """Estados del autómata a nivel de carácter."""

    START = auto()
    OBJ_OPEN = auto()
    
    # Key: "name"
    KEY_NAME_QUOTE1 = auto()
    KEY_NAME_TEXT = auto()
    KEY_NAME_QUOTE2 = auto()
    COLON_NAME = auto()
    VAL_NAME_QUOTE1 = auto()
    VAL_NAME_TEXT = auto()
    VAL_NAME_QUOTE2 = auto()
    COMMA_AFTER_NAME = auto()
    
    # Key: "parameters"
    KEY_PARAMS_QUOTE1 = auto()
    KEY_PARAMS_TEXT = auto()
    KEY_PARAMS_QUOTE2 = auto()
    COLON_PARAMS = auto()
    PARAMS_OBJ_OPEN = auto()
    
    # Dentro de los parámetros
    PARAM_KEY_QUOTE1 = auto()
    PARAM_KEY_TEXT = auto()
    PARAM_KEY_QUOTE2 = auto()
    PARAM_COLON = auto()
    
    # Valores de parámetros
    VAL_STRING_QUOTE1 = auto()
    VAL_STRING_TEXT = auto()
    VAL_STRING_QUOTE2 = auto()
    VAL_NUMBER = auto()
    VAL_BOOL = auto()
    
    PARAM_COMMA = auto()
    PARAMS_OBJ_CLOSE = auto()
    OBJ_CLOSE = auto()
    REJECT = auto()


class CharDFA:
    """
    Autómata Determinista Finito (DFA) a nivel de carácter.
    Valida la sintaxis carácter a carácter y simula la ingesta de subcadenas/tokens.
    """

    def __init__(self, allowed_function_names: list[str]) -> None:
        self.allowed_function_names = allowed_function_names
        self.start_state = CharState.START
        self.accept_states = {CharState.OBJ_CLOSE}

    def step(self, current_state: CharState, char: str) -> CharState:
        """
        Función de transición delta: (Estado, Carácter) -> Nuevo Estado.
        Maneja espacios en blanco opcionales de forma transparente.
        """
        # Permite espacios en blanco opcionales en transiciones no literales
        if char in " \t\n\r" and not self._is_literal_state(current_state):
            return current_state

        match current_state:
            case CharState.START:
                return CharState.OBJ_OPEN if char == "{" else CharState.REJECT

            case CharState.OBJ_OPEN:
                return CharState.KEY_NAME_QUOTE1 if char == '"' else CharState.REJECT

            case CharState.KEY_NAME_QUOTE1:
                # Transición simplificada de la clave "name"
                return CharState.KEY_NAME_QUOTE2 if char == '"' else current_state

            case CharState.KEY_NAME_QUOTE2:
                return CharState.COLON_NAME if char == ":" else CharState.REJECT

            case CharState.COLON_NAME:
                return CharState.VAL_NAME_QUOTE1 if char == '"' else CharState.REJECT

            case CharState.VAL_NAME_QUOTE1:
                return CharState.VAL_NAME_QUOTE2 if char == '"' else current_state

            case CharState.VAL_NAME_QUOTE2:
                return CharState.COMMA_AFTER_NAME if char == "," else CharState.REJECT

            case CharState.COMMA_AFTER_NAME:
                return CharState.KEY_PARAMS_QUOTE1 if char == '"' else CharState.REJECT

            case CharState.KEY_PARAMS_QUOTE1:
                return CharState.KEY_PARAMS_QUOTE2 if char == '"' else current_state

            case CharState.KEY_PARAMS_QUOTE2:
                return CharState.COLON_PARAMS if char == ":" else CharState.REJECT

            case CharState.COLON_PARAMS:
                return CharState.PARAMS_OBJ_OPEN if char == "{" else CharState.REJECT

            case CharState.PARAMS_OBJ_OPEN:
                if char == "}":
                    return CharState.PARAMS_OBJ_CLOSE
                return CharState.PARAM_KEY_QUOTE1 if char == '"' else CharState.REJECT

            case CharState.PARAM_KEY_QUOTE1:
                return CharState.PARAM_KEY_QUOTE2 if char == '"' else current_state

            case CharState.PARAM_KEY_QUOTE2:
                return CharState.PARAM_COLON if char == ":" else CharState.REJECT

            case CharState.PARAM_COLON:
                if char == '"':
                    return CharState.VAL_STRING_QUOTE1
                if char.isdigit() or char in "-+":
                    return CharState.VAL_NUMBER
                if char in "tf":
                    return CharState.VAL_BOOL
                return CharState.REJECT

            case CharState.VAL_STRING_QUOTE1:
                return CharState.VAL_STRING_QUOTE2 if char == '"' else current_state

            case CharState.VAL_STRING_QUOTE2 | CharState.VAL_NUMBER | CharState.VAL_BOOL:
                if char == ",":
                    return CharState.PARAM_COMMA
                if char == "}":
                    return CharState.PARAMS_OBJ_CLOSE
                if current_state == CharState.VAL_NUMBER and (char.isdigit() or char in ".eE-+"):
                    return CharState.VAL_NUMBER
                return CharState.REJECT

            case CharState.PARAM_COMMA:
                return CharState.PARAM_KEY_QUOTE1 if char == '"' else CharState.REJECT

            case CharState.PARAMS_OBJ_CLOSE:
                return CharState.OBJ_CLOSE if char == "}" else CharState.REJECT

            case _:
                return CharState.REJECT

    def simulate_string(self, start_state: CharState, text: str) -> Optional[CharState]:
        """
        Consume una cadena completa (un token BPE) desde 'start_state'.
        Si la cadena es válida, retorna el estado final. Si rompe la regla, retorna None.
        """
        state = start_state
        for char in text:
            state = self.step(state, char)
            if state == CharState.REJECT:
                return None
        return state

    @staticmethod
    def _is_literal_state(state: CharState) -> bool:
        """Determina si un estado se encuentra procesando el interior de un String literal."""
        return state in (
            CharState.KEY_NAME_QUOTE1,
            CharState.VAL_NAME_QUOTE1,
            CharState.KEY_PARAMS_QUOTE1,
            CharState.PARAM_KEY_QUOTE1,
            CharState.VAL_STRING_QUOTE1,
        )