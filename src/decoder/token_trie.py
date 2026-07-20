from __future__ import annotations

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


class TokenTrieNode(BaseModel):

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    #
    # Outgoing transitions.
    #

    children: dict[str, "TokenTrieNode"] = Field(
        default_factory=dict,
    )

    #
    # Tokenizer token ending at this node.
    #

    token_id: int | None = None


class TokenTrie(BaseModel):

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    root: TokenTrieNode = Field(
        default_factory=TokenTrieNode,
    )

    def clear(
        self,
    ) -> None:
        """
        Remove every tokenizer token.
        """

        self.root = TokenTrieNode()

    def insert(
        self,
        token_id: int,
        token: str,
    ) -> None:
        """
        Insert one tokenizer token into the trie.
        """

        node = self.root

        for character in token:

            child = node.children.get(character)

            if child is None:
                child = TokenTrieNode()
                node.children[character] = child

            node = child

        node.token_id = token_id

    def build(
        self,
        vocabulary: dict[int, str],
    ) -> None:
        """
        Build the trie from the tokenizer vocabulary.
        """

        self.clear()

        #
        # Deterministic insertion order improves reproducibility.
        #

        for token_id in sorted(vocabulary):

            self.insert(
                token_id=token_id,
                token=vocabulary[token_id],
            )


TokenTrieNode.model_rebuild()