from pydantic import BaseModel, ConfigDict, field_validator


class Prompt(BaseModel):
    """
    Represents a single natural language prompt.

    A prompt is a user request that the system must analyze
    and convert into a structured function call.

    Example
    -------
    {
        "prompt": "Add 5 and 10."
    }
    """
    model_config = ConfigDict(extra="forbid")
    prompt: str

    @field_validator("prompt", mode="after")
    @classmethod
    def validate_non_empty_prompt(cls, value: str) -> str:
        """
        Validate that the prompt text is non-empty and non-whitespace.

        Args:
            value: The prompt text string.

        Returns:
            The validated prompt text.

        Raises:
            ValueError: If the prompt is empty or consists only of whitespace.
        """
        if not value.strip():
            raise ValueError("Prompt text cannot be empty or consist only "
                             "of whitespace.")
        return value

    @classmethod
    def validate_many(cls, data: list[dict[str, 'Prompt']]) -> list["Prompt"]:
        """
        Validate and convert multiple prompts.

        Args:
            data: List of dictionaries loaded from the input
                prompts JSON file.

        Returns:
            A list of validated Prompt objects.

        Raises:
            ValidationError: If one or more prompts do not
                match the expected schema.
        """
        return [cls.model_validate(item) for item in data]
