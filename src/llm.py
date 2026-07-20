from llm_sdk.llm_sdk import Small_LLM_Model      # type: ignore
import numpy as np                              # type: ignore
from pydantic import BaseModel, ConfigDict, Field

from .utils import load_json


class Llm(BaseModel):
    """
    Thin wrapper around the provided LLM SDK.
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
        __context,
    ) -> None:
        """
        Load the tokenizer vocabulary once.
        """

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
        """

        return self.model.decode(ids)

    def get_logits(
        self,
        input_ids: list[int],
    ) -> np.ndarray:
        """
        Compute next-token logits.
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
        Raw tokenizer token.
        """

        return self.id_to_token[token_id]

    def normalized_token(
        self,
        token_id: int,
    ) -> str:
        """
        Normalized tokenizer token.
        """

        return self.normalized_vocabulary[token_id]

    @staticmethod
    def normalize(
        token: str,
    ) -> str:
        """
        Replace tokenizer-specific whitespace markers by ordinary spaces.
        """

        return (
            token
            .replace("▁", " ")
            .replace("Ġ", " ")
            .replace("Ċ", "\n")
            .replace("ĉ", "\t")
        )