from pydantic import BaseModel, ConfigDict    # type: ignore


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

    @classmethod
    def validate_many(cls, data: list[dict]) -> list["Prompt"]:
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