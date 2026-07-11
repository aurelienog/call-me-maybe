from .models import Prompt, FunctionRegistry, FunctionCallResult
from .llm import Llm

import json
from pydantic import BaseModel, Field
import numpy as np


from enum import Enum


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
            token_text = self.llm.normalize(self.llm.token(next_token))
            partial_text += token_text

            next_state = self._next_state(state, partial_text)
            if next_state != state:
                partial_text = ""
            state = next_state

            tokens.append(next_token)
            generated_tokens.append(next_token)

        return generated_tokens

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

    def _next_state(
        self,
        state: DecoderState,
        partial_text: str,
    ) -> DecoderState:
        """
        Return the next decoder state.

        Args:
            state: Current decoder state.
            partial_text: Text generated while remaining in the current state.

        Returns:
            The next decoder state.
        """
        match state:

            case DecoderState.EXPECT_OPEN_BRACE:
                if partial_text == "{":
                    return DecoderState.EXPECT_NAME_KEY
                return state

            case DecoderState.EXPECT_NAME_KEY:
                if partial_text == '"name"':
                    return DecoderState.EXPECT_NAME_COLON
                return state

            case DecoderState.EXPECT_NAME_COLON:
                if partial_text == ':':
                    return DecoderState.EXPECT_FUNCTION_NAME
                return state

            case DecoderState.EXPECT_FUNCTION_NAME:
                if partial_text.startswith('"') and partial_text.endswith('"'):
                    function_name = partial_text[1:-1]
                    if self.registry.exists(function_name):
                        self.current_function = function_name
                        return DecoderState.EXPECT_COMMA
                return state

            case DecoderState.EXPECT_COMMA:
                if partial_text == ',':
                    return DecoderState.EXPECT_PARAMETERS_KEY
                return state

            case DecoderState.EXPECT_PARAMETERS_KEY:
                if partial_text == '"parameters"':
                    return DecoderState.EXPECT_PARAMETERS_COLON
                return state

            case DecoderState.EXPECT_PARAMETERS_COLON:
                if partial_text == ':':
                    return DecoderState.EXPECT_PARAMETERS_OPEN
                return state

            case DecoderState.EXPECT_PARAMETERS_OPEN:
                if partial_text == "{":
                    parameters = self.registry.parameters(
                        self.current_function)
                    if parameters:
                        return DecoderState.EXPECT_PARAMETER_NAME
                    return DecoderState.EXPECT_CLOSE_PARAMETERS
                return state

            case DecoderState.EXPECT_PARAMETER_NAME:
                if (
                    partial_text.startswith('"')
                    and partial_text.endswith('"')
                ):
                    parameter = partial_text[1:-1]
                    if parameter in self.registry.parameters(self.current_function):
                        self.current_parameter = parameter
                        self.written_parameters.add(parameter)
                        return DecoderState.EXPECT_PARAMETER_COLON
                return state

            case DecoderState.EXPECT_PARAMETER_COLON:
                if partial_text == ":":
                    return DecoderState.EXPECT_PARAMETER_VALUE
                return state

            case DecoderState.EXPECT_PARAMETER_VALUE:
                if self._parameter_value_complete(partial_text):
                    self.current_parameter = None
                    return DecoderState.EXPECT_PARAMETER_SEPARATOR
                return state

            case DecoderState.EXPECT_PARAMETER_SEPARATOR:
                if partial_text == ",":
                    return DecoderState.EXPECT_PARAMETER_NAME

                if partial_text == "}":
                    return DecoderState.EXPECT_CLOSE_PARAMETERS
                return state

            case DecoderState.EXPECT_CLOSE_PARAMETERS:
                if partial_text == "}":
                    return DecoderState.EXPECT_CLOSE_OBJECT
                return state

            case DecoderState.EXPECT_CLOSE_OBJECT:
                if partial_text == "}":
                    return DecoderState.FINISHED
                return state

            case DecoderState.FINISHED:
                return DecoderState.FINISHED

        raise ValueError(f"Unexpected decoder state: {state}")

    def _parameter_value_complete(self, partial_text: str) -> bool:
        parameter = self.registry.parameter(
            self.current_function,
            self.current_parameter,
        )

        match parameter.type:

            case "string":
                return (
                    len(partial_text) >= 2
                    and partial_text.startswith('"')
                    and partial_text.endswith('"')
                )

            case "number":
                try:
                    float(partial_text)
                    return True
                except ValueError:
                    return False

            case "boolean":
                return partial_text in {"true", "false"}

        raise ValueError(f"Unknown parameter type: {parameter.type}")

# _allowed_* implementa la gramática (qué puede venir después).
# _next_state() implementa el autómata.
# _parameter_value_complete() solo indica el momento de cambiar de estado. Es una función pequeña porque toda la complejidad ya está resuelta por _allowed_parameter_value_tokens()

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

        for token_id, token in self.llm.id_to_token.items():
            token = self.llm.normalize(token)

            candidate = partial_text + token

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

        for token_id, token in self.llm.id_to_token.items():
            token = self.llm.normalize(token)

            candidate = partial_text + token

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

        for token_id, token in self.llm.id_to_token.items():
            token = self.llm.normalize(token)

            candidate = partial_text + token

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
                return self._allowed_string_tokens(partial_text)

            case "number":
                return self._allowed_number_tokens(partial_text)

            case "boolean":
                return self._allowed_boolean_tokens(partial_text)

        raise ValueError(f"Unsupported parameter type: {parameter_type}")

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

# No es un parser JSON completo (no maneja escapes como \"), pero suele ser suficiente para este proyecto.
    def _allowed_string_tokens(
        self,
        partial_text: str,
    ) -> set[int]:
        """
        Return the token IDs that can legally continue a JSON string.
        """
        allowed: set[int] = set()

        for token_id, token in self.llm.id_to_token.items():
            token = self.llm.normalize(token)

            candidate = partial_text + token

            # string not opened
            if not partial_text:
                if candidate == '"':
                    allowed.add(token_id)
                continue

            # string finished
            if (len(candidate) >= 2
                    and candidate.startswith('"')
                    and candidate.endswith('"')):
                if '"' not in candidate[1:-1]:
                    allowed.add(token_id)
                continue

            # still inside the string
            if '"' not in token:
                allowed.add(token_id)

        return allowed

    def _allowed_number_tokens(
        self,
        partial_text: str,
    ) -> set[int]:
        """
        Return the token IDs that can legally continue a JSON number.
        """
        allowed: set[int] = set()

        valid_chars = set("0123456789-+.eE")

        for token_id, token in self.llm.id_to_token.items():
            token = self.llm.normalize(token)

            candidate = partial_text + token

            # sólo caracteres válidos
            if not all(c in valid_chars for c in candidate):
                continue

            # un único '-'
            if candidate.count("-") > 1:
                continue

            # un único punto
            if candidate.count(".") > 1:
                continue

            # un único exponente
            if candidate.lower().count("e") > 1:
                continue

            allowed.add(token_id)

        return allowed

    def _allowed_boolean_tokens(
        self,
        partial_text: str,
    ) -> set[int]:
        """
        Return the token IDs that can legally continue a JSON boolean.
        """
        allowed: set[int] = set()

        for token_id, token in self.llm.id_to_token.items():
            token = self.llm.normalize(token)

            candidate = partial_text + token

            if "true".startswith(candidate) or "false".startswith(candidate):
                allowed.add(token_id)

        return allowed
