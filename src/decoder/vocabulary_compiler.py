from __future__ import annotations

from enum import Enum, auto
import re

from pydantic import BaseModel, ConfigDict


class LexicalKind(Enum):
    """
    Elementary lexical categories for BPE tokens.
    """

    WHITESPACE = auto()
    PUNCTUATION = auto()
    BOOLEAN = auto()
    NULL = auto()
    NUMBER_CHUNK = auto()
    STRING_CHUNK = auto()
    MIXED = auto()


class CompiledToken(BaseModel):
    """
    Metadata for an individual BPE token analyzed at the lexical level.

    Attributes:
        token_id: The unique integer ID of the token in the vocabulary.
        raw_text: The original raw string representation of the token.
        normalized_text: The normalized string representation of the token.
        kinds: A tuple of lexical categories matched by this token.
        is_pure_whitespace: Fast flag indicating if the token contains only
            whitespace.
        is_valid_in_string: Fast flag indicating if the token can safely
            appear inside a JSON string.
        is_valid_in_number: Fast flag indicating if the token represents a
            valid JSON number chunk.
    """

    model_config = ConfigDict(frozen=True)

    token_id: int
    raw_text: str
    normalized_text: str
    kinds: tuple[LexicalKind, ...]

    # Fast flags for shortcuts during compilation
    is_pure_whitespace: bool
    is_valid_in_string: bool
    is_valid_in_number: bool


class CompiledVocabulary(BaseModel):
    """
    Analyzed and categorized vocabulary used to accelerate DFA compilation.

    Attributes:
        tokens: Mapping from token ID to its compiled metadata.
        vocab_size: The total size of the target vocabulary tensor dimension.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    tokens: dict[int, CompiledToken]
    vocab_size: int

    def get(self, token_id: int) -> CompiledToken:
        """
        Return the compiled token metadata by its ID.

        Args:
            token_id: The vocabulary integer index.

        Returns:
            The associated CompiledToken object.
        """
        return self.tokens[token_id]


class VocabularyCompiler(BaseModel):
    """
    Compiler responsible for pre-processing the raw tokenizer vocabulary into
    lexical tokens.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    _num_pattern: re.Pattern[str] = re.compile(
        r"^[-+]?\d*\.?\d*([eE][-+]?\d*)?$")

    def compile(
        self,
        normalized_vocabulary: dict[int, str],
        target_vocab_size: int = 151936,  # <--- Tamaño de logits real
    ) -> CompiledVocabulary:
        """
        Analyze and classify all tokens in the vocabulary.

        Args:
            normalized_vocabulary: Mapping of token IDs to normalized text
                strings.
            target_vocab_size: Minimum expected vocabulary size for
                model tensor alignment.

        Returns:
            A populated CompiledVocabulary container.
        """
        compiled_tokens: dict[int, CompiledToken] = {}

        for token_id, norm_text in normalized_vocabulary.items():
            compiled_tokens[token_id] = self._analyze_token(token_id,
                                                            norm_text)

        # If there are padding or special IDs undefined in the JSON vocabulary,
        # create neutral entries to avoid out-of-bound errors in tensors/arrays
        max_id = max(max(normalized_vocabulary.keys()), target_vocab_size - 1)
        vocab_size = max_id + 1

        return CompiledVocabulary(
            tokens=compiled_tokens,
            vocab_size=vocab_size,
        )

    def _analyze_token(self, token_id: int, text: str) -> CompiledToken:
        """
        Classify a BPE token according to the syntactic fragments it can
        satisfy.

        Args:
            token_id: The integer ID of the token.
            text: The normalized string content of the token.

        Returns:
            The analyzed CompiledToken metadata instance.
        """
        kinds: list[LexicalKind] = []

        is_pure_ws = len(text) > 0 and text.isspace()
        if is_pure_ws:
            kinds.append(LexicalKind.WHITESPACE)

        # 1. JSON structural punctuation
        if any(char in text for char in "{}[]:,"):
            kinds.append(LexicalKind.PUNCTUATION)

        # 2. Boolean / Null literals
        if "true" in text or "false" in text:
            kinds.append(LexicalKind.BOOLEAN)
        if "null" in text:
            kinds.append(LexicalKind.NULL)

        # 3. Numeric fragment inspection (e.g., "12", ".5", "e-3", "-")
        is_num = False
        if text and not is_pure_ws:
            if self._num_pattern.match(text.strip()):
                kinds.append(LexicalKind.NUMBER_CHUNK)
                is_num = True

        # 4. JSON string contents inclusion (basic quote / control
        #  character escaping)
        # A substring inside a JSON string must not prematurely terminate or
        # corrupt the string
        is_str = False
        if '"' not in text and "\\" not in text:
            kinds.append(LexicalKind.STRING_CHUNK)
            is_str = True

        if not kinds:
            kinds.append(LexicalKind.MIXED)

        return CompiledToken(
            token_id=token_id,
            raw_text=text,
            normalized_text=text,
            kinds=tuple(kinds),
            is_pure_whitespace=is_pure_ws,
            is_valid_in_string=is_str,
            is_valid_in_number=is_num,
        )
