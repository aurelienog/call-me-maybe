from .state import ConsumeResult


def parse_string(text: str, is_generating: bool = False) -> tuple[ConsumeResult, str]:
    """
    Parse a JSON string.

    Returns:
        (result, remaining)

        INVALID:
            `text` cannot become a valid JSON string.

        PREFIX:
            `text` is a valid prefix of a JSON string but is not complete.

        COMPLETE:
            A complete JSON string was parsed. The returned `remaining`
            contains the unconsumed suffix.
    """
    if not text.startswith('"'):
        return ConsumeResult.INVALID, text

    escaped = False

    for i in range(1, len(text)):
        c = text[i]

        if escaped:
            escaped = False
            continue

        if c == "\\":
            escaped = True
            continue

        if c == '"':
            if is_generating and i == 1:
                return ConsumeResult.PREFIX, text # Fuerza a que siga escribiendo
            return (
                ConsumeResult.COMPLETE,
                text[i + 1:],
            )

    return ConsumeResult.PREFIX, text


def parse_number(
        text: str,
        is_generating: bool = False) -> tuple[ConsumeResult, str]:
    """
    Parse a JSON number.

    Returns:
        (result, remaining)
    """
    end = 0

    while (
        end < len(text)
        and text[end] not in ", \t\n}"
    ):
        end += 1

    number = text[:end]
    remaining = text[end:]

    if not number:
        return ConsumeResult.INVALID, text

    try:
        float(number)

        if number.endswith((".", "e", "E", "+", "-")):
            return ConsumeResult.PREFIX, text

        if is_generating and not remaining:
            return ConsumeResult.PREFIX, text

        return ConsumeResult.COMPLETE, remaining

    except ValueError:

        if (
            number == "-"
            or number.endswith((".", "e", "E", "e+", "e-", "E+", "E-"))
        ):
            return ConsumeResult.PREFIX, text

    return ConsumeResult.INVALID, text


def parse_boolean(
    text: str,
    is_generating: bool = False
) -> tuple[ConsumeResult, str]:

    for literal in ("true", "false"):

        if literal.startswith(text):
            if text == literal:
                if is_generating:
                    return ConsumeResult.PREFIX, text
                return ConsumeResult.COMPLETE, ""
            return ConsumeResult.PREFIX, text

        if text.startswith(literal):
            return ConsumeResult.COMPLETE, text[len(literal):]

    return ConsumeResult.INVALID, text
