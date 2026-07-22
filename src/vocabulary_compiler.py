from __future__ import annotations

from enum import Enum, auto
import re
from typing import NamedTuple

from pydantic import BaseModel, ConfigDict, Field


class LexicalKind(Enum):
    """Categorías léxicas elementales para tokens BPE."""

    WHITESPACE = auto()
    PUNCTUATION = auto()
    BOOLEAN = auto()
    NULL = auto()
    NUMBER_CHUNK = auto()
    STRING_CHUNK = auto()
    MIXED = auto()


class CompiledToken(BaseModel):
    """
    Metadata de un token BPE individual analizado a nivel léxico.
    """

    model_config = ConfigDict(frozen=True)

    token_id: int
    raw_text: str
    normalized_text: str
    kinds: tuple[LexicalKind, ...]
    
    # Banderas rápidas para atajos en compilación
    is_pure_whitespace: bool
    is_valid_in_string: bool
    is_valid_in_number: bool


class CompiledVocabulary(BaseModel):
    """
    Vocabulario analizado y categorizado para acelerar la compilación del DFA.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    tokens: dict[int, CompiledToken]
    vocab_size: int

    def get(self, token_id: int) -> CompiledToken:
        """Retorna el token compilado por id."""
        return self.tokens[token_id]


class VocabularyCompiler(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    _num_pattern: re.Pattern[str] = re.compile(r"^[-+]?\d*\.?\d*([eE][-+]?\d*)?$")

    def compile(
        self,
        normalized_vocabulary: dict[int, str],
        target_vocab_size: int = 151936,  # <--- Tamaño de logits real
    ) -> CompiledVocabulary:
        compiled_tokens: dict[int, CompiledToken] = {}

        for token_id, norm_text in normalized_vocabulary.items():
            compiled_tokens[token_id] = self._analyze_token(token_id, norm_text)

        # Si hay IDs de padding/especiales sin definir en el JSON,
        # creamos entradas neutras para no romper el array
        max_id = max(max(normalized_vocabulary.keys()), target_vocab_size - 1)
        vocab_size = max_id + 1

        return CompiledVocabulary(
            tokens=compiled_tokens,
            vocab_size=vocab_size,
        )

    def _analyze_token(self, token_id: int, text: str) -> CompiledToken:
        """
        Clasifica un token BPE según los fragmentos sintácticos que puede cubrir.
        """
        kinds: list[LexicalKind] = []

        is_pure_ws = len(text) > 0 and text.isspace()
        if is_pure_ws:
            kinds.append(LexicalKind.WHITESPACE)

        # 1. Puntuación estructural JSON
        if any(char in text for char in "{}[]:,"):
            kinds.append(LexicalKind.PUNCTUATION)

        # 2. Literales booleanos / Null
        if "true" in text or "false" in text:
            kinds.append(LexicalKind.BOOLEAN)
        if "null" in text:
            kinds.append(LexicalKind.NULL)

        # 3. Inspección de fragmentos numéricos (ej. "12", ".5", "e-3", "-")
        is_num = False
        if text and not is_pure_ws:
            if self._num_pattern.match(text.strip()):
                kinds.append(LexicalKind.NUMBER_CHUNK)
                is_num = True

        # 4. Inclusión en cadenas JSON (escapado básico de comillas / control)
        # Una subcadena dentro de un string JSON no debe romper el string prematuramente
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