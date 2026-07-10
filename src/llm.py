from llm_sdk.llm_sdk import Small_LLM_Model     # type: ignore
import numpy as np    # type: ignore
from pydantic import BaseModel, Field, ConfigDict

from .utils import load_json


class Llm(BaseModel):
    """
    Wrapper around the provided LLM SDK.

    Exposes a simplified interface for encoding text into token IDs,
    decoding token IDs back into text, and obtaining the logits for
    the next generated token. The rest of the project interacts only
    with standard Python types and NumPy arrays, without depending on
    the SDK's internal tensor representation.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    model: Small_LLM_Model = Field(default_factory=Small_LLM_Model)
    token_to_id: dict[str, int] = Field(default_factory=dict)
    id_to_token: dict[int, str] = Field(default_factory=dict)

    def model_post_init(self, _: object) -> None:
        """Load the tokenizer vocabulary once."""
        vocab_path = self.model.get_path_to_vocab_file()
        self.token_to_id = load_json(vocab_path)

        self.id_to_token = {
            token_id: token
            for token, token_id in self.token_to_id.items()
        }

    def encode(self, text: str) -> list[int]:
        """
        Encode a text prompt into a list of token IDs.

        The SDK returns a tensor with shape (1, sequence_length).
        This method converts the first (and only) batch into a plain
        Python list so the rest of the project remains independent
        from the underlying tensor implementation.

        Args:
            text: Input text to tokenize.

        Returns:
            A list of integer token IDs representing the input text.
        """
        ids = self.model.encode(text)
        return [int(token_id) for token_id in ids[0]]

    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs back into text.

        Args:
            ids: Token IDs to decode.

        Returns:
            The decoded text corresponding to the provided token IDs.
        """
        return self.model.decode(ids)

    def get_logits(self, input_ids: list[int]) -> np.ndarray:
        """
        Compute the logits for the next token.

        Given the current sequence of input token IDs, queries the LLM
        and returns the raw logits associated with every token in the
        vocabulary. These logits are later filtered by the constrained
        decoder before selecting the next token.

        Args:
            input_ids: Current sequence of input token IDs.

        Returns:
            A NumPy array containing the raw logits for every token in
            the model vocabulary.
        """
        logits = self.model.get_logits_from_input_ids(input_ids)
        return np.asarray(logits, dtype=np.float32)

    def token_from_id(self, token_id: int) -> str:
        """
        Return the string corresponding to a token ID.
        """
        return self.id_to_token[token_id]

    @staticmethod
    def normalize(token: str) -> str:
        return (
            token
            .replace("▁", " ")
            .replace("Ġ", " ")
            .lstrip()
        )
