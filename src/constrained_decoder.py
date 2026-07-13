from .models import Prompt, FunctionRegistry, FunctionCallResult
from .llm import Llm

import json
from pydantic import BaseModel, Field
import numpy as np


from enum import Enum


class ConsumeResult(Enum):
    INVALID = 0
    PREFIX = 1
    COMPLETE = 2


class DecoderState(Enum):
    START = 0

    EXPECT_OPEN_BRACE = 1

    EXPECT_NAME_KEY = 2
    EXPECT_NAME_COLON = 3
    EXPECT_FUNCTION_NAME = 4

    EXPECT_COMMA = 5

    EXPECT_PARAMETERS_KEY = 6
    EXPECT_PARAMETERS_COLON = 7
    EXPECT_PARAMETERS_OPEN = 8

    EXPECT_PARAMETER_NAME = 9
    EXPECT_PARAMETER_COLON = 10
    EXPECT_PARAMETER_VALUE = 11

    EXPECT_PARAMETER_SEPARATOR = 12

    EXPECT_CLOSE_PARAMETERS = 13
    EXPECT_CLOSE_OBJECT = 14

    FINISHED = 15


class ConstrainedDecoder(BaseModel):
    llm: Llm = Field(default_factory=Llm)
    registry: FunctionRegistry
    current_function: str | None = None
    current_parameter: str | None = None
    written_parameters: set[str] = Field(default_factory=set)

    def run(self, prompts: list[Prompt]) -> list[FunctionCallResult]:
        """
        Process a batch of prompts using constrained decoding.

        Args:
            prompts: Prompts to process.

        Returns:
            A FunctionCallResult for each input prompt.
        """
        results = []
        for prompt in prompts:
            result = self.process_prompt(prompt)
            results.append(result)
        return results

    def process_prompt(self, prompt: Prompt) -> FunctionCallResult:
        """
        Generate a function call for a single prompt.

        Args:
            prompt: User prompt.

        Returns:
            The generated FunctionCallResult.
        """
        input_ids = self.llm.encode(prompt.prompt)
        output_ids = self.generate(input_ids)
        output = self.llm.decode(output_ids)
        data = json.loads(output)
        return FunctionCallResult(
                prompt=prompt.prompt,
                name=data["name"],
                parameters=data["parameters"]
                )

    def generate(self, input_ids: list[int]) -> list[int]:
        """
        Generate output tokens using constrained decoding.

        Args:
            input_ids: Tokenized prompt.

        Returns:
            The generated token IDs.
        """
        self.current_function = None
        self.current_parameter = None
        self.written_parameters.clear()

        state = DecoderState.EXPECT_OPEN_BRACE
        tokens = input_ids.copy()
        generated_tokens: list[int] = []
        partial_text = ""

        while state != DecoderState.FINISHED:
            logits = self.llm.get_logits(tokens)
            allowed = self._allowed_tokens(state, partial_text)
            masked = self._mask_logits(logits, list(allowed))
            next_token = self._select_next_token(masked)
            tokens.append(next_token)
            generated_tokens.append(next_token)
            partial_text += self.llm.normalize(self.llm.token(next_token))

            while True:
                new_state, partial_text = self._consume(
                    state,
                    partial_text,
                )
                if new_state == state:
                    break
                state = new_state

        return generated_tokens

    def _candidate_tokens(self, partial_text):

        for token_id, token in self.llm.id_to_token.items():
            token = self.llm.normalize(token)
            yield token_id, partial_text + token

    def _allowed_tokens(
        self,
        state: DecoderState,
        partial_text: str,
    ) -> set[int]:
    
        match state:

            case DecoderState.EXPECT_OPEN_BRACE:
                return self._allowed_literal_tokens(partial_text, "{")

            case DecoderState.EXPECT_NAME_KEY:
                return self._allowed_literal_tokens(partial_text, '"name"')

            case DecoderState.EXPECT_NAME_COLON:
                return self._allowed_literal_tokens(partial_text, ":")

            case DecoderState.EXPECT_FUNCTION_NAME:
                return self._allowed_function_name_tokens(partial_text)

            case DecoderState.EXPECT_COMMA:
                return self._allowed_literal_tokens(partial_text, ",")

            case DecoderState.EXPECT_PARAMETERS_KEY:
                return self._allowed_literal_tokens(
                    partial_text,
                    '"parameters"',
                )

            case DecoderState.EXPECT_PARAMETERS_COLON:
                return self._allowed_literal_tokens(partial_text, ":")

            case DecoderState.EXPECT_PARAMETERS_OPEN:
                return self._allowed_literal_tokens(partial_text, "{")

            case DecoderState.EXPECT_PARAMETER_NAME:
                return self._allowed_parameter_name_tokens(partial_text)

            case DecoderState.EXPECT_PARAMETER_COLON:
                return self._allowed_literal_tokens(partial_text, ":")

            case DecoderState.EXPECT_PARAMETER_VALUE:
                return self._allowed_parameter_value_tokens(partial_text)

            case DecoderState.EXPECT_PARAMETER_SEPARATOR:
                return self._allowed_parameter_separator_tokens(partial_text)

            case DecoderState.EXPECT_CLOSE_PARAMETERS:
                return self._allowed_literal_tokens(partial_text, "}")

            case DecoderState.EXPECT_CLOSE_OBJECT:
                return self._allowed_literal_tokens(partial_text, "}")

            case DecoderState.FINISHED:
                return set()

        raise ValueError(f"Unexpected decoder state: {state}")

    def _mask_logits(self, logits: np.ndarray, allowed_tokens: list[int]
                     ) -> np.ndarray:
        masked = logits.copy()

        for i in range(len(masked)):
            if i not in allowed_tokens:
                masked[i] = -np.inf
        return masked

    def _select_next_token(self, logits: np.ndarray) -> int:
        return int(np.argmax(logits))

    def _consume(
        self,
        state: DecoderState,
        remaining: str,
        ) -> tuple[DecoderState, str]:
        """
        Consume as much text as possible from `remaining`
        according to the current decoder state.

        Returns:
            (next_state, remaining_text)
        """

        match state:

            case DecoderState.EXPECT_OPEN_BRACE:
                if remaining.startswith("{"):
                    return (DecoderState.EXPECT_NAME_KEY,
                            remaining[1:])
                return state, remaining

            case DecoderState.EXPECT_NAME_KEY:
                literal = '"name"'
                if remaining.startswith(literal):
                    return (DecoderState.EXPECT_NAME_COLON,
                            remaining[len(literal):])
                return state, remaining

            case DecoderState.EXPECT_NAME_COLON:
                if remaining.startswith(':'):
                    return (DecoderState.EXPECT_FUNCTION_NAME,
                            remaining[1:])
                return state, remaining

            case DecoderState.EXPECT_FUNCTION_NAME:
                for function in self.registry.functions():
                    literal = f'"{function}"'
                    if remaining.startswith(literal):
                        self.current_function = function
                        return (
                            DecoderState.EXPECT_COMMA,
                            remaining[len(literal):],
                        )
                return state, remaining

            case DecoderState.EXPECT_COMMA:
                if remaining.startswith(','):
                    return (DecoderState.EXPECT_PARAMETERS_KEY,
                            remaining[1:])
                return state, remaining

            case DecoderState.EXPECT_PARAMETERS_KEY:
                literal = '"parameters"'

                if remaining.startswith(literal):
                    return (DecoderState.EXPECT_PARAMETERS_COLON,
                            remaining[len(literal):])
                return state, remaining

            case DecoderState.EXPECT_PARAMETERS_COLON:
                if remaining.startswith(':'):
                    return (DecoderState.EXPECT_PARAMETERS_OPEN,
                            remaining[1:])
                return state, remaining

            case DecoderState.EXPECT_PARAMETERS_OPEN:
                if remaining.startswith("{"):
                    parameters = self.registry.parameters(
                        self.current_function)
                    if parameters:
                        return (DecoderState.EXPECT_PARAMETER_NAME,
                                remaining[1:])
                    return (DecoderState.EXPECT_CLOSE_PARAMETERS,
                            remaining[1:])
                return state, remaining

            case DecoderState.EXPECT_PARAMETER_NAME:
                for parameter in self.registry.parameters(self.current_function):
                    if parameter in self.written_parameters:
                        continue
                    literal = f'"{parameter}"'
                    if remaining.startswith(literal):
                        self.current_parameter = parameter
                        self.written_parameters.add(parameter)
                        return (DecoderState.EXPECT_PARAMETER_COLON,
                                remaining[len(literal):])
                return state, remaining

            case DecoderState.EXPECT_PARAMETER_COLON:
                if remaining.startswith(":"):
                    return (DecoderState.EXPECT_PARAMETER_VALUE,
                            remaining[1:])
                return state, remaining

            case DecoderState.EXPECT_PARAMETER_VALUE:

                result, remaining = self._parse_parameter_value(remaining)

                if result != ConsumeResult.COMPLETE:
                    return state, remaining

                self.current_parameter = None

                return (
                    DecoderState.EXPECT_PARAMETER_SEPARATOR,
                    remaining,
                )

            case DecoderState.EXPECT_PARAMETER_SEPARATOR:
                if remaining.startswith(","):
                    return (DecoderState.EXPECT_PARAMETER_NAME,
                            remaining[1:])

                if remaining.startswith("}"):
                    return (DecoderState.EXPECT_CLOSE_PARAMETERS,
                            remaining[1:])
                return state, remaining

            case DecoderState.EXPECT_CLOSE_PARAMETERS:
                if remaining.startswith("}"):
                    return (DecoderState.EXPECT_CLOSE_OBJECT,
                            remaining[1:])
                return state, remaining

            case DecoderState.EXPECT_CLOSE_OBJECT:
                if remaining.startswith("}"):
                    return (DecoderState.FINISHED,
                            remaining[1:])
                return state, remaining

            case DecoderState.FINISHED:
                return DecoderState.FINISHED, remaining

        raise ValueError(f"Unexpected decoder state: {state}")

    def _parse_string(
        self,
        text: str,
    ) -> tuple[ConsumeResult, str]:
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
                return (
                    ConsumeResult.COMPLETE,
                    text[i + 1:],
                )

        return ConsumeResult.PREFIX, text

    def _parse_number(
        self,
        text: str,
    ) -> tuple[ConsumeResult, str]:
        """
        Parse a JSON number.

        Returns:
            (result, remaining)
        """
        end = 0

        while (
            end < len(text)
            and text[end] not in ",}"
        ):
            end += 1

        number = text[:end]
        remaining = text[end:]

        if not number:
            return ConsumeResult.INVALID, text

        try:
            float(number)

            # Puede ser un prefijo aunque float() lo acepte
            if number.endswith((".", "e", "E", "+", "-")):
                return ConsumeResult.PREFIX, text

            return ConsumeResult.COMPLETE, remaining

        except ValueError:

            # Prefijos válidos de un número JSON
            if (
                number == "-"
                or number.endswith((".", "e", "E", "e+", "e-", "E+", "E-"))
            ):
                return ConsumeResult.PREFIX, text

            return ConsumeResult.INVALID, text

    def _parse_boolean(
        self,
        text: str,
    ) -> tuple[ConsumeResult, str]:

        for literal in ("true", "false"):

            if literal.startswith(text):
                if text == literal:
                    return ConsumeResult.COMPLETE, ""
                return ConsumeResult.PREFIX, text

            if text.startswith(literal):
                return ConsumeResult.COMPLETE, text[len(literal):]

        return ConsumeResult.INVALID, text

    def _parse_parameter_value(
        self,
        remaining: str,
    ) -> tuple[ConsumeResult, str]:
        """
        Parse the current parameter value.

        Returns:
            (result, remaining)
        """
        parameter = self.registry.parameter(
            self.current_function,
            self.current_parameter,
        )

        match parameter.type:

            case "string":
                return self._parse_string(remaining)

            case "number":
                return self._parse_number(remaining)

            case "boolean":
                return self._parse_boolean(remaining)

        raise ValueError(f"Unknown parameter type: {parameter.type}")

    def _allowed_from_parser(
        self,
        partial_text: str,
        parser,
    ) -> set[int]:

        allowed = set()

        for token_id, candidate in self._candidate_tokens(partial_text):
            result, _ = parser(candidate)

            if result != ConsumeResult.INVALID:
                allowed.add(token_id)

        return allowed

    def _allowed_literal_tokens(
            self,
            partial_text: str,
            literal: str,
        ) -> set[int]:
            """
            Return the token IDs that can legally continue a fixed JSON literal.

            The target literal is known in advance (for example "{", ":",
            ",", '"name"', or '"parameters"'). Given the text generated so
            far for the current state, only tokens that keep matching the
            target literal are allowed.

            Examples:
                literal = '"name"'
                partial_text = ""      -> tokens starting '"'
                partial_text = '"'     -> tokens continuing 'name'
                partial_text = '"na'   -> tokens continuing 'me"'
                partial_text = '"name"' -> no continuation (state changes)
            """
            allowed: set[int] = set()

            for token_id, candidate in self._candidate_tokens(partial_text):

                if literal.startswith(candidate):
                    allowed.add(token_id)

            return allowed

    def _allowed_function_name_tokens(
            self,
            partial_text: str,
        ) -> set[int]:
            """
            Return the token IDs that can legally continue a function name.

            Only function names registered in the FunctionRegistry are valid.
            At every generation step, a token is allowed only if appending its
            normalized text keeps matching at least one registered function
            name. Once a complete function name has been written, only the
            closing quote is allowed.

            Examples:
                available = ["fn_add", "fn_subtract"]

                partial_text = '"'          -> tokens continuing 'fn_'
                partial_text = '"fn_'       -> tokens continuing 'add' or 'subtract'
                partial_text = '"fn_add'    -> only the closing quote is valid
            """
            allowed: set[int] = set()

            functions = [
                f'"{function}"'
                for function in self.registry.functions()
            ]

            for token_id, candidate in self._candidate_tokens(partial_text):

                for function in functions:
                    if function.startswith(candidate):
                        allowed.add(token_id)
                        break

            return allowed

    def _allowed_parameter_name_tokens(
            self,
            partial_text: str,
        ) -> set[int]:
            """
            Return the token IDs that can legally continue a parameter name.

            Only parameters belonging to the currently selected function are
            considered valid. Parameters already generated earlier in the JSON
            object are excluded.

            Examples:
                function parameters = ["a", "b"]

                partial_text = '"'      -> tokens starting 'a' or 'b'
                partial_text = '"a'     -> only the closing quote is valid
                written_parameters = {"a"} -> only parameter 'b' is allowed
            """
            allowed: set[int] = set()

            assert self.current_function is not None

            parameters = [
                f'"{parameter}"'
                for parameter in self.registry.parameters(self.current_function)
                if parameter not in self.written_parameters
            ]

            for token_id, candidate in self._candidate_tokens(partial_text):

                for parameter in parameters:
                    if parameter.startswith(candidate):
                        allowed.add(token_id)
                        break

            return allowed

    def _allowed_parameter_value_tokens(
            self,
            partial_text: str,
        ) -> set[int]:
            """
            Return the token IDs that can legally continue the current
            parameter value.

            The allowed tokens depend on the declared type of the current
            parameter (string, number or boolean). Only tokens that keep
            producing a valid JSON value of that type are allowed.

            Examples:
                type = string
                    partial_text = '"'      -> printable characters or closing quote

                type = number
                    partial_text = ""       -> '-', digits
                    partial_text = "12"     -> digits, '.', 'e', 'E'

                type = boolean
                    partial_text = ""       -> tokens continuing 'true' or 'false'
                    partial_text = "tr"     -> tokens continuing 'ue'
            """
            assert self.current_function is not None
            assert self.current_parameter is not None

            parameter_type = self.registry.parameter_type(
                self.current_function,
                self.current_parameter
            )

            match parameter_type:
                case "string":
                    parser = self._parse_string
                case "number":
                    parser = self._parse_number
                case "boolean":
                    parser = self._parse_boolean
                case _:
                    raise ValueError(
                        f"Unsupported parameter type: {parameter_type}"
                    )

            return self._allowed_from_parser(
                partial_text,
                parser,
            )

    def _allowed_parameter_separator_tokens(
            self,
            partial_text: str,
        ) -> set[int]:
            """
            Return the token IDs that can legally follow a completed parameter.

            If there are remaining parameters to generate, only the comma is
            allowed. Otherwise, only the closing brace of the parameters object
            is allowed.
            """
            assert self.current_function is not None

            parameters = set(self.registry.parameters(self.current_function))
            remaining = parameters - self.written_parameters

            if remaining:
                return self._allowed_literal_tokens(partial_text, ",")

            return self._allowed_literal_tokens(partial_text, "}")

    def _allowed_string_tokens(
        self,
        partial_text: str,
    ) -> set[int]:
        return self._allowed_from_parser(
        partial_text,
        self._parse_string,
    )

    def _allowed_number_tokens(
            self,
            partial_text: str,
        ) -> set[int]:
            """
            Return the token IDs that can legally continue a JSON number.
            """
            return self._allowed_from_parser(
                partial_text,
                self._parse_number,
            )

    def _allowed_boolean_tokens(
            self,
            partial_text: str,
        ) -> set[int]:
            """
            Return the token IDs that can legally continue a JSON boolean.
            """
            return self._allowed_from_parser(
                partial_text,
                self._parse_boolean,
            )
