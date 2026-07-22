from __future__ import annotations

from enum import IntEnum
from pydantic import BaseModel, ConfigDict, Field


class CharState(IntEnum):
    START = 0
    OBJ_OPEN = 1
    KEY_NAME_Q1 = 2
    KEY_NAME_Q2 = 3
    COLON_NAME = 4
    VAL_NAME_Q1 = 5
    VAL_NAME_Q2 = 6
    COMMA_AFTER_NAME = 7
    KEY_PARAMS_Q1 = 8
    KEY_PARAMS_Q2 = 9
    COLON_PARAMS = 10
    PARAMS_OBJ_OPEN = 11
    PARAM_KEY_Q1 = 12
    PARAM_KEY_Q2 = 13
    PARAM_COLON = 14
    VAL_STRING_Q1 = 15
    VAL_STRING_Q2 = 16
    VAL_NUMBER = 17
    VAL_BOOL = 18
    PARAM_COMMA = 19
    PARAMS_OBJ_CLOSE = 20
    OBJ_CLOSE = 21
    REJECT = -1


NUM_STATES = 22


class CharDFA(BaseModel):
    """
    Autómata determinista a nivel de caracteres validado por Pydantic.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    allowed_function_names: list[str] = Field(default_factory=list)
    start_state: CharState = CharState.START
    accept_states: set[CharState] = Field(default_factory=lambda: {CharState.OBJ_CLOSE})

    def get_allowed_first_chars(self, state: int) -> set[str]:
        match state:
            case CharState.START:
                return {"{", " "}
            case CharState.OBJ_OPEN:
                return {'"', " "}
            case CharState.KEY_NAME_Q1:
                return set('abcdefghijklmnopqrstuvwxyz"_')
            case CharState.KEY_NAME_Q2:
                return {":", " "}
            case CharState.COLON_NAME:
                return {'"', " "}
            case CharState.VAL_NAME_Q1:
                return set('abcdefghijklmnopqrstuvwxyz"_')
            case CharState.VAL_NAME_Q2:
                return {",", " "}
            case CharState.COMMA_AFTER_NAME:
                return {'"', " "}
            case CharState.KEY_PARAMS_Q1:
                return set('abcdefghijklmnopqrstuvwxyz"_')
            case CharState.KEY_PARAMS_Q2:
                return {":", " "}
            case CharState.COLON_PARAMS:
                return {"{", " "}
            case CharState.PARAMS_OBJ_OPEN:
                return {'"', "}", " "}
            case CharState.PARAM_KEY_Q1:
                return set('abcdefghijklmnopqrstuvwxyz"_')
            case CharState.PARAM_KEY_Q2:
                return {":", " "}
            case CharState.PARAM_COLON:
                return {'"', "t", "f", "-", "+", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", " "}
            case CharState.VAL_STRING_Q1:
                return set('abcdefghijklmnopqrstuvwxyz0123456789_ -/\\":')
            case CharState.VAL_STRING_Q2 | CharState.VAL_NUMBER | CharState.VAL_BOOL:
                return {",", "}", " "}
            case CharState.PARAM_COMMA:
                return {'"', " "}
            case CharState.PARAMS_OBJ_CLOSE:
                return {"}", " "}
            case _:
                return set()

    def step(self, state: int, char: str) -> int:
        if state == CharState.REJECT:
            return CharState.REJECT

        if char in " \t\n\r" and state not in (
            CharState.KEY_NAME_Q1,
            CharState.VAL_NAME_Q1,
            CharState.KEY_PARAMS_Q1,
            CharState.PARAM_KEY_Q1,
            CharState.VAL_STRING_Q1,
        ):
            return state

        match state:
            case CharState.START: return CharState.OBJ_OPEN if char == "{" else CharState.REJECT
            case CharState.OBJ_OPEN: return CharState.KEY_NAME_Q1 if char == '"' else CharState.REJECT
            case CharState.KEY_NAME_Q1: return CharState.KEY_NAME_Q2 if char == '"' else CharState.KEY_NAME_Q1
            case CharState.KEY_NAME_Q2: return CharState.COLON_NAME if char == ":" else CharState.REJECT
            case CharState.COLON_NAME: return CharState.VAL_NAME_Q1 if char == '"' else CharState.REJECT
            case CharState.VAL_NAME_Q1: return CharState.VAL_NAME_Q2 if char == '"' else CharState.VAL_NAME_Q1
            case CharState.VAL_NAME_Q2: return CharState.COMMA_AFTER_NAME if char == "," else CharState.REJECT
            case CharState.COMMA_AFTER_NAME: return CharState.KEY_PARAMS_Q1 if char == '"' else CharState.REJECT
            case CharState.KEY_PARAMS_Q1: return CharState.KEY_PARAMS_Q2 if char == '"' else CharState.KEY_PARAMS_Q1
            case CharState.KEY_PARAMS_Q2: return CharState.COLON_PARAMS if char == ":" else CharState.REJECT
            case CharState.COLON_PARAMS: return CharState.PARAMS_OBJ_OPEN if char == "{" else CharState.REJECT
            case CharState.PARAMS_OBJ_OPEN:
                if char == "}": return CharState.PARAMS_OBJ_CLOSE
                return CharState.PARAM_KEY_Q1 if char == '"' else CharState.REJECT
            case CharState.PARAM_KEY_Q1: return CharState.PARAM_KEY_Q2 if char == '"' else CharState.PARAM_KEY_Q1
            case CharState.PARAM_KEY_Q2: return CharState.PARAM_COLON if char == ":" else CharState.REJECT
            case CharState.PARAM_COLON:
                if char == '"': return CharState.VAL_STRING_Q1
                if char.isdigit() or char in "-+": return CharState.VAL_NUMBER
                if char in "tf": return CharState.VAL_BOOL
                return CharState.REJECT
            case CharState.VAL_STRING_Q1: return CharState.VAL_STRING_Q2 if char == '"' else CharState.VAL_STRING_Q1
            case CharState.VAL_STRING_Q2 | CharState.VAL_NUMBER | CharState.VAL_BOOL:
                if char == ",": return CharState.PARAM_COMMA
                if char == "}": return CharState.PARAMS_OBJ_CLOSE
                if state == CharState.VAL_NUMBER and (char.isdigit() or char in ".eE-+"): return CharState.VAL_NUMBER
                return CharState.REJECT
            case CharState.PARAM_COMMA: return CharState.PARAM_KEY_Q1 if char == '"' else CharState.REJECT
            case CharState.PARAMS_OBJ_CLOSE: return CharState.OBJ_CLOSE if char == "}" else CharState.REJECT
            case _: return CharState.REJECT

    def simulate_string(self, start_state: int, text: str) -> int:
        state = start_state
        for char in text:
            state = self.step(state, char)
            if state == CharState.REJECT:
                return CharState.REJECT
        return state
