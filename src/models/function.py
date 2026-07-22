from pydantic import BaseModel, ConfigDict

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


class FunctionDefinition(BaseModel):
    """
    Represents the definition of a callable function.

    A function definition describes the function name, its purpose,
    the expected input parameters, and the type of value it returns.

    Example
    -------
    {
        "name": "fn_add_numbers",
        "description": "Add two numbers together.",
        "parameters": {
            "a": {
                "type": "number"
            },
            "b": {
                "type": "number"
            }
        },
        "returns": {
            "type": "number"
        }
    }
    """
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: dict[str, ParameterDefinition]
    returns: ReturnDefinition

    @classmethod
    def validate_many(cls, data: list[dict]) -> list["FunctionDefinition"]:
        """
        Validate and convert multiple function definitions.

        Args:
            data: List of dictionaries loaded from the JSON
                functions definition file.

        Returns:
            A list of validated FunctionDefinition objects.

        Raises:
            ValidationError: If one or more function definitions
                do not match the expected schema.
        """
        return [cls.model_validate(item) for item in data]
