from __future__ import annotations

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from ..models import FunctionRegistry

from .boolean_dfa import BooleanDFA
from .choice_dfa import ChoiceDFA
from .number_dfa import NumberDFA
from .string_dfa import StringDFA
from .token_trie import TokenTrie
from .token_trie_explorer import TokenTrieExplorer


class VocabularyCache(BaseModel):

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    #
    # Tokenizer vocabulary.
    #

    vocabulary_trie: TokenTrie = Field(
        default_factory=TokenTrie,
    )

    explorer: TokenTrieExplorer = Field(
        default_factory=lambda: TokenTrieExplorer(
            trie=TokenTrie(),
        ),
    )

    #
    # Primitive DFAs.
    #

    string_dfa: StringDFA = Field(
        default_factory=StringDFA,
    )

    number_dfa: NumberDFA = Field(
        default_factory=NumberDFA,
    )

    boolean_dfa: BooleanDFA = Field(
        default_factory=BooleanDFA,
    )

    #
    # Cached literal DFAs.
    #

    literal_dfas: dict[str, ChoiceDFA] = Field(
        default_factory=dict,
    )

    #
    # Cached function names.
    #

    function_dfa: ChoiceDFA | None = None

    #
    # Cached parameter names.
    #

    parameter_dfas: dict[str, ChoiceDFA] = Field(
        default_factory=dict,
    )

    #
    # Helpers
    #

    def literal(
        self,
        literal: str,
    ) -> ChoiceDFA:

        return self.literal_dfas[literal]

    def function_machine(
        self,
    ) -> ChoiceDFA:

        assert self.function_dfa is not None
        return self.function_dfa

    def parameter_machine(
        self,
        function_name: str,
    ) -> ChoiceDFA:

        return self.parameter_dfas[function_name]

    #
    # Build
    #

    def build(
        self,
        registry: FunctionRegistry,
        vocabulary: dict[int, str],
    ) -> None:
        """
        Build every cached DFA and the tokenizer trie.

        This method is executed only once during decoder initialization.
        """

        #
        # Build tokenizer trie.
        #

        self.vocabulary_trie.build(
            vocabulary,
        )

        self.explorer.trie = self.vocabulary_trie
        self.explorer.clear()

        #
        # Clear previous caches.
        #

        self.literal_dfas.clear()
        self.parameter_dfas.clear()

        #
        # JSON literals.
        #

        for literal in (
            "{",
            "}",
            ":",
            ",",
            '"name"',
            '"parameters"',
        ):

            self.literal_dfas[literal] = ChoiceDFA(
                literals=(literal,),
            )

        #
        # Function names.
        #

        function_names = tuple(
            registry.function_names()
        )

        self.function_dfa = ChoiceDFA(
            literals=tuple(
                f'"{name}"'
                for name in function_names
            ),
        )

        #
        # Parameter names.
        #

        for function_name in function_names:

            parameters = tuple(
                f'"{parameter}"'
                for parameter in registry.parameters(
                    function_name,
                )
            )

            self.parameter_dfas[function_name] = ChoiceDFA(
                literals=parameters,
            )