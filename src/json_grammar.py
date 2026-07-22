from __future__ import annotations

from abc import ABC
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


class JsonGrammarNode(
    BaseModel,
    ABC,
):
    """
    Clase base abstracta e inmutable para todos los nodos del árbol de la gramática JSON.
    """

    model_config = ConfigDict(
        frozen=True,
    )


class JsonPropertyNode(
    BaseModel,
):
    """
    Representa una propiedad dentro de un objeto JSON (un par clave-valor).
    """

    model_config = ConfigDict(
        frozen=True,
    )

    name: str
    required: bool = True
    value: JsonGrammarNode


class JsonObjectNode(
    JsonGrammarNode,
):
    """
    Representa un tipo de dato Objeto JSON ({ ... }).
    """

    properties: tuple[JsonPropertyNode, ...] = Field(
        default_factory=tuple,
    )


class JsonArrayNode(
    JsonGrammarNode,
):
    """
    Representa un tipo de dato Array JSON ([ ... ]).
    """

    items: JsonGrammarNode


class JsonStringNode(
    JsonGrammarNode,
):
    """
    Representa un tipo de dato String JSON.
    """

    pass


class JsonIntegerNode(
    JsonGrammarNode,
):
    """
    Representa un tipo de dato Entero JSON.
    """

    pass


class JsonNumberNode(
    JsonGrammarNode,
):
    """
    Representa un tipo de dato Numérico JSON (float o int).
    """

    pass


class JsonBooleanNode(
    JsonGrammarNode,
):
    """
    Representa un tipo de dato Booleano JSON (true / false).
    """

    pass


class JsonNullNode(
    JsonGrammarNode,
):
    """
    Representa el valor nulo en JSON (null).
    """

    pass


class JsonEnumNode(
    JsonGrammarNode,
):
    """
    Representa un valor restringido a un conjunto específico de opciones (Enum).
    """

    values: tuple[str, ...]


class JsonAnyOfNode(
    JsonGrammarNode,
):
    """
    Permite definir alternativas sintácticas (ej. un valor que puede ser string o number).
    """

    choices: tuple[JsonGrammarNode, ...] = Field(
        default_factory=tuple,
    )


class JsonGrammar(BaseModel):
    """
    Representa la gramática JSON completa cuya raíz es un objeto JSON.
    """

    model_config = ConfigDict(
        frozen=True,
    )

    root: JsonObjectNode