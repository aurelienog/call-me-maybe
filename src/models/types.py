from typing import Literal

# Supported JSON primitive and composite types used in
# function definitions and parameter validation.
JsonType = Literal[
    "string",
    "number",
    "integer",
    "boolean",
    "object",
    "array",
]

# Python values accepted as function arguments.
ParameterValue = str | int | float | bool
