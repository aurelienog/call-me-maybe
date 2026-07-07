from pydantic import BaseModel, ConfigDict    # type: ignore

from .types import ParameterValue


class FunctionCallResult(BaseModel):
    """
    Represents the predicted function call for a prompt.

    A function call result contains the original natural
    language prompt, the selected function name, and the
    arguments extracted from the prompt.

    Example
    -------
    {
        "prompt": "Add 5 and 10.",
        "name": "fn_add_numbers",
        "parameters": {
            "a": 5,
            "b": 10
        }
    }
    """
    model_config = ConfigDict(extra="forbid")

    prompt: str
    name: str
    parameters: dict[str, ParameterValue]
