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
                prompt = prompt.prompt,
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
        state = DecoderState.EXPECT_OPEN_BRACE
        tokens = input_ids.copy()
        generated_tokens: list[int]  = []
        while state != DecoderState.FINISHED:
            logits = self.llm.get_logits(tokens)
            allowed = self._allowed_tokens(state, generated_tokens)
            masked = self._mask_logits(logits, allowed)
            next_token = self._select_next_token(masked)
            state = self._next_state(state, next_token, generated_tokens)
            tokens.append(next_token)
            generated_tokens.append(next_token)
        return generated_tokens

    def _allowed_tokens(
        self,
        state: DecoderState,
        generated_tokens: list[int],
    ) -> set[int]:
        """
        Return the token IDs that are valid in the current decoder state.
        """

        match state:

            case DecoderState.EXPECT_OPEN_BRACE:
                return set(self.llm.token_ids("{"))

            case DecoderState.EXPECT_NAME_KEY:
                return set(self.llm.token_ids('"name"'))

            case DecoderState.EXPECT_NAME_COLON:
                return set(self.llm.token_ids(":"))

            case DecoderState.EXPECT_FUNCTION_NAME:
                return self._allowed_function_name_tokens(generated_tokens)

            case DecoderState.EXPECT_COMMA:
                return set(self.llm.token_ids(","))

            case DecoderState.EXPECT_PARAMETERS_KEY:
                return set(self.llm.token_ids('"parameters"'))

            case DecoderState.EXPECT_PARAMETERS_COLON:
                return set(self.llm.token_ids(":"))

            case DecoderState.EXPECT_PARAMETERS_OPEN:
                return set(self.llm.token_ids("{"))

            case DecoderState.EXPECT_PARAMETER_NAME:
                return self._allowed_parameter_name_tokens(generated_tokens)

            case DecoderState.EXPECT_PARAMETER_COLON:
                return set(self.llm.token_ids(":"))

            case DecoderState.EXPECT_PARAMETER_VALUE:
                return self._allowed_parameter_value_tokens(generated_tokens)

            case DecoderState.EXPECT_PARAMETER_SEPARATOR:
                return self._allowed_parameter_separator_tokens(generated_tokens)

            case DecoderState.EXPECT_CLOSE_PARAMETERS:
                return set(self.llm.token_ids("}"))

            case DecoderState.EXPECT_CLOSE_OBJECT:
                return set(self.llm.token_ids("}"))

            case DecoderState.FINISHED:
                return set()

        raise ValueError(f"Unexpected decoder state: {state}")
                

    def _mask_logits(self, logits: np.ndarray, allowed_tokens: list[int]) -> np.ndarray:
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
        next_token: int,
        generated_tokens: list[int],
    ) -> DecoderState:
        """
        Return the next decoder state.

        Args:
            state: Current decoder state.
            next_token: Newly generated token ID.
            generated_tokens: Tokens generated so far.

        Returns:
            The next decoder state.
        """

        match state:

            case DecoderState.EXPECT_OPEN_BRACE:
                return DecoderState.EXPECT_NAME_KEY

            case DecoderState.EXPECT_NAME_KEY:
                return DecoderState.EXPECT_NAME_COLON

            case DecoderState.EXPECT_NAME_COLON:
                return DecoderState.EXPECT_FUNCTION_NAME

            case DecoderState.EXPECT_FUNCTION_NAME:
                return DecoderState.EXPECT_COMMA

            case DecoderState.EXPECT_COMMA:
                return DecoderState.EXPECT_PARAMETERS_KEY

            case DecoderState.EXPECT_PARAMETERS_KEY:
                return DecoderState.EXPECT_PARAMETERS_COLON

            case DecoderState.EXPECT_PARAMETERS_COLON:
                return DecoderState.EXPECT_PARAMETERS_OPEN

            case DecoderState.EXPECT_PARAMETERS_OPEN:
                # Si la función no tiene parámetros,
                # aquí en el futuro irás directamente a EXPECT_CLOSE_PARAMETERS.
                return DecoderState.EXPECT_PARAMETER_NAME

            case DecoderState.EXPECT_PARAMETER_NAME:
                return DecoderState.EXPECT_PARAMETER_COLON

            case DecoderState.EXPECT_PARAMETER_COLON:
                return DecoderState.EXPECT_PARAMETER_VALUE

            case DecoderState.EXPECT_PARAMETER_VALUE:
                # Más adelante decidirás si queda otro parámetro
                # o hay que cerrar el objeto.
                return DecoderState.EXPECT_PARAMETER_SEPARATOR

            case DecoderState.EXPECT_PARAMETER_SEPARATOR:
                # Placeholder.
                return DecoderState.EXPECT_CLOSE_PARAMETERS

            case DecoderState.EXPECT_CLOSE_PARAMETERS:
                return DecoderState.EXPECT_CLOSE_OBJECT

            case DecoderState.EXPECT_CLOSE_OBJECT:
                return DecoderState.FINISHED

            case DecoderState.FINISHED:
                return DecoderState.FINISHED

        raise ValueError(f"Unexpected decoder state: {state}")