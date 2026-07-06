from pydantic import (BaseModel, ConfigDict)   # type: ignore

from .types import JsonType


class ParameterDefinition(BaseModel):
    """
    Represents the definition of a function parameter.

    A parameter definition specifies the expected JSON type
    of a function argument.

    Example
    -------
    {
        "type": "number"
    }
    """
    model_config = ConfigDict(extra='forbid')
    type: JsonType

class ReturnDefinition(ParameterDefinition):
    """
    Represents the definition of a function return value.

    A return definition specifies the JSON type produced
    by a function. It shares the same schema as a parameter
    definition but is modeled separately to distinguish
    input parameters from return values.
    """
    pass