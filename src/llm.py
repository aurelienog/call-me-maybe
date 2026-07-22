from llm_sdk.llm_sdk import Small_LLM_Model
import numpy as np

from pydantic import BaseModel, ConfigDict, Field
from typing import Any
from .utils import load_json


class Llm(BaseModel):
    """
    Thin wrapper around the provided LLM SDK.

    Exposes a simplified interface for tokenization,
    decoding, and next-token prediction while caching
    the tokenizer vocabulary for efficient constrained
    decoding.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    model: Small_LLM_Model = Field(
        default_factory=Small_LLM_Model,
    )

    token_to_id: dict[str, int] = Field(
        default_factory=dict,
    )

    id_to_token: dict[int, str] = Field(
        default_factory=dict,
    )

    normalized_vocabulary: dict[int, str] = Field(
        default_factory=dict,
    )

    def model_post_init(
        self,
        __context: Any,
    ) -> None:
        """
        Initialize tokenizer lookup tables.

        Loads the tokenizer vocabulary and builds mappings
        between token ids, raw tokenizer tokens, and their
        normalized representations.
        """
        super().model_post_init(__context)

        vocab_path = self.model.get_path_to_vocab_file()

        self.token_to_id = load_json(vocab_path)

        self.id_to_token = {
            token_id: token
            for token, token_id in self.token_to_id.items()
        }

        self.normalized_vocabulary = {
            token_id: self.normalize(token)
            for token_id, token in self.id_to_token.items()
        }

    def encode(
        self,
        text: str,
    ) -> list[int]:
        """
        Encode text into token ids.

        Args:
            text: Input text.

        Returns:
            A list of token ids.
        """

        ids = self.model.encode(text)

        return [
            int(token_id)
            for token_id in ids[0]
        ]

    def decode(
        self,
        ids: list[int],
    ) -> str:
        """
        Decode token ids into text.

        Args:
            ids: Token ids to decode.

        Returns:
            The decoded text.
        """
        decoded_text = self.model.decode(ids)
        return str(decoded_text)

    def get_logits(
        self,
        input_ids: list[int],
    ) -> np.typing.NDArray[np.float32]:
        """
        Compute next-token logits.

        Args:
            input_ids: Input token ids.

        Returns:
            A NumPy array containing the logits for the next
            predicted token.
        """

        logits = self.model.get_logits_from_input_ids(
            input_ids,
        )

        return np.asarray(
            logits,
            dtype=np.float32,
        )

    def token_from_id(
        self,
        token_id: int,
    ) -> str:
        """
        Return the raw tokenizer token for a token id.

        Args:
            token_id: Token identifier.

        Returns:
            The corresponding tokenizer token.
        """

        return self.id_to_token[token_id]

    def normalized_token(
        self,
        token_id: int,
    ) -> str:
        """
        Return the normalized token for a token id.

        Args:
            token_id: Token identifier.

        Returns:
            The normalized token representation.
        """

        return self.normalized_vocabulary[token_id]

    @staticmethod
    def normalize(
        token: str,
    ) -> str:
        """
        Normalize a tokenizer token.

        Replaces tokenizer-specific whitespace markers with
        their corresponding standard whitespace characters.

        Args:
            token: Raw tokenizer token.

        Returns:
            The normalized token.
        """

        return (
            token
            .replace("▁", " ")
            .replace("Ġ", " ")
            .replace("Ċ", "\n")
            .replace("ĉ", "\t")
        )
