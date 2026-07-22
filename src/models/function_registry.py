from . import FunctionDefinition, ParameterDefinition, JsonType

from pydantic import BaseModel, ConfigDict, Field    # type: ignore

# @model_validator(mode="after")
# def validate_business_rules(self) -> "FunctionRegistry":
#     ... que no tenga string vacios

# registry.model_dump_json() para comprobar?


class FunctionRegistry(BaseModel):
    """
    Registry of available function definitions.

    Stores validated FunctionDefinition objects indexed by their
    unique names and provides a high-level API to query function
    metadata, including parameters, parameter types, and return
    types.

    The registry is used by the constrained decoder to validate
    function calls against the available function definitions.
    """
    model_config = ConfigDict(extra="forbid")

    functions: dict[str, FunctionDefinition] = Field(default_factory=dict)

    def load(self, functions: list[FunctionDefinition]) -> None:
        """
        Load multiple function definitions into the registry.

        Each function definition is indexed by its name, allowing
        fast lookups during constrained decoding.

        Args:
            functions: List of validated FunctionDefinition objects.

        Raises:
            ValueError: If duplicate function names are found.
        """
        for function in functions:
            self.add(function)

    def add(self, function: FunctionDefinition) -> None:
        """
        Add a single function definition to the registry.

        Args:
            function: A validated FunctionDefinition instance.

        Raises:
            ValueError: If another function with the same name
                already exists in the registry.
        """
        if self.exists(function.name):
            raise ValueError(f"Function '{function.name}' already exists.")

        self.functions[function.name] = function

    def get(self, name: str) -> FunctionDefinition:
        """
        Return the complete definition of a function.

        Args:
            name: Function name.

        Returns:
            The corresponding FunctionDefinition.

        Raises:
            KeyError: If the function does not exist.
        """
        return self.functions[name]

    def exists(self, name: str) -> bool:
        """
        Check whether a function exists in the registry.

        Args:
            name: Function name.

        Returns:
            True if the function exists, False otherwise.
        """
        return name in self.functions

    def function_names(self) -> list[str]:
        """
        Return the names of all registered functions.

        Returns:
            A list containing the name of every registered function.
        """
        return list(self.functions.keys())

    def parameters(self, name: str) -> dict[str, ParameterDefinition]:
        """
        Return all parameters defined for a function.

        Args:
            name: Function name.

        Returns:
            A mapping from parameter names to ParameterDefinition objects.

        Raises:
            KeyError: If the function does not exist.
        """
        return self.get(name).parameters

    def parameter_type(self,
                       function_name: str,
                       parameter_name: str
                       ) -> JsonType:
        """
        Return the expected type of a function parameter.

        Args:
            function_name: Function name.
            parameter_name: Parameter name.

        Returns:
            The expected JSON type of the parameter.

        Raises:
            KeyError: If the function or parameter does not exist.
        """
        return self.parameters(function_name)[parameter_name].type

    def return_type(self, name: str) -> JsonType:
        """
        Return the declared return type of a function.

        Args:
            name: Function name.

        Returns:
            The JSON type returned by the function.

        Raises:
            KeyError: If the function does not exist.
        """
        return self.get(name).returns.type

    def build_context(self) -> str:
        """
        Build the prompt context describing available functions and output rules.
        """
        lines = ["Available functions:\n"]

        for function in self.functions.values():
            lines.append(f"- {function.name}")
            lines.append(f"  Description: {function.description}")
            lines.append("  Parameters:")

            for name, parameter in function.parameters.items():
                lines.append(f"    - {name}: {parameter.type}")
            lines.append("")

        lines.append("Instructions:")
        lines.append("- Output strictly valid JSON with keys 'name' and 'parameters'.")
        lines.append("- For regex patterns, avoid unnecessary capture groups e.g. use [0-9]+ instead of ([0-9]+).")

        return "\n".join(lines)