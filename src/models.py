from pydantic import (BaseModel, Field, model_validator,    # type: ignore
                      ConfigDict, ValidationError)

class FunctionRegistry(BaseModel):

model_config = ConfigDict(extra='forbid')

@model_validator(mode="after")
    def validate_business_rules(self) -> "SpaceMission":
    
    def exists("fn_add_numbers")

    def get_parameters("fn_add_numbers")

    def get_parameter_type(
    "fn_add_numbers",
    "a"
)

    def required_parameters(
    "fn_add_numbers"
)

    def function_names()

Devuélveme la definición completa de una función.


class FunctionDefinition(BaseModel):
    
class ParameterDefinition(BaseModel):
    
class ConstrainedDecoder(BaseModel):