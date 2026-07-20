from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .consume_result import ConsumptionStatus
from .dfa import DFA
from .token_trie import TokenTrie, TokenTrieNode


class TokenTrieExplorer(BaseModel):

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    trie: TokenTrie

    cache: dict[
        tuple[int, int, Any],
        frozenset[int],
    ] = Field(
        default_factory=dict,
    )

    def allowed_tokens(
        self,
        machine: DFA,
        state: Any,
    ) -> frozenset[int]:
        """
        Return every tokenizer token that can be consumed by the DFA
        without entering ERROR.

        The explorer does NOT know anything about grammar boundaries.
        It simply answers whether a tokenizer token is compatible with
        the current DFA state.
        """

        return self._explore(
            machine,
            self.trie.root,
            state,
        )

    def _explore(
        self,
        machine: DFA,
        node: TokenTrieNode,
        state: Any,
    ) -> frozenset[int]:

        key = (
            id(machine),
            id(node),
            state,
        )

        cached = self.cache.get(key)
        if cached is not None:
            return cached

        allowed: set[int] = set()

        for character, child in node.children.items():

            result = machine.consume(
                state=state,
                token=character,
                offset=0,
            )

            #
            # Character rejected.
            #

            if result.status is ConsumptionStatus.ERROR:
                continue

            #
            # Character accepted.
            #

            if child.token_id is not None:
                allowed.add(
                    child.token_id,
                )

            #
            # Continue exploring the tokenizer trie from the DFA state
            # reached after consuming this character.
            #

            allowed.update(
                self._explore(
                    machine=machine,
                    node=child,
                    state=result.state,
                )
            )

        result = frozenset(
            allowed,
        )

        self.cache[key] = result

        return result

    def clear(
        self,
    ) -> None:

        self.cache.clear()