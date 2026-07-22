from __future__ import annotations

import numpy as np  # type: ignore
from pydantic import BaseModel, ConfigDict

from .vocabulary_compiler import CompiledVocabulary
from .json_function_call_dfa import JsonFunctionCallDFA
from .char_dfa import CharDFA, CharState


class GrammarCompiler(BaseModel):
    """
    Compiler that transforms a JSON grammar and vocabulary
    into a token-level DFA.

    Fuses the character-level state machine with the model's BPE vocabulary
    to produce dense numpy matrices optimized for fast logit masking
    and O(1) state transitions.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    def compile(
        self,
        vocabulary: CompiledVocabulary,
        allowed_functions: list[str],
    ) -> JsonFunctionCallDFA:
        """
        Convert the character-level CharDFA into a token-level
        JsonFunctionCallDFA.

        For each state in the CharDFA and each token_id in the vocabulary,
        simulates whether the token's character sequence can be consumed
        without landing in the REJECT state.

        Args:
            grammar: The input JSON grammar specification.
            vocabulary: The pre-analyzed compiled vocabulary container.
            allowed_functions: List of function names permitted
            by the registry.

        Returns:
            An immutable JsonFunctionCallDFA instance with precomputed mask
            and transition matrices.
        """
        char_dfa = CharDFA(allowed_function_names=allowed_functions)

        # Retrieve all valid CharDFA states (excluding the reject state)
        valid_char_states = [state for state in CharState
                             if state != CharState.REJECT]

        # Bijective mapping from CharState to a dense numeric ID (0..N-1)
        # for fast indexing
        state_to_idx: dict[CharState, int] = {
            state: i for i, state in enumerate(valid_char_states)
        }
        idx_to_state: dict[int, CharState] = {
            i: state for state, i in state_to_idx.items()
        }

        num_states = len(valid_char_states)
        vocab_size = vocabulary.vocab_size

        # 1. Logit validity mask matrix: (num_states, vocab_size)
        # Initialize with -inf (invalid by default)
        logit_masks = np.full(
            (num_states, vocab_size),
            fill_value=-np.inf,
            dtype=np.float32,
        )

        # 2. Token transition table: (num_states, vocab_size) -> next_state_idx
        # Initialize with -1 to indicate impossible transitions
        transitions = np.full(
            (num_states, vocab_size),
            fill_value=-1,
            dtype=np.int32,
        )

        # --- FOLDING PHASE (TOKEN-BY-TOKEN SIMULATION) ---
        for char_state in valid_char_states:
            state_idx = state_to_idx[char_state]

            for token_id, compiled_token in vocabulary.tokens.items():
                token_text = compiled_token.normalized_text

                # Edge case: Empty tokens do not alter state
                if not token_text:
                    continue

                # Simulation: Consume token string character by character
                # starting from char_state
                end_state = char_dfa.simulate_string(
                    start_state=char_state,
                    text=token_text,
                )

                # CASE B (VALID): String consumed completely and landed in a
                # non-REJECT state
                if end_state is not None and end_state != CharState.REJECT:
                    end_state_idx = state_to_idx[end_state]

                    # Enable token by adding 0.0 to logit during inference
                    logit_masks[state_idx, token_id] = 0.0

                    # Record destination state
                    transitions[state_idx, token_id] = end_state_idx

        # CASE A (INVALID): end_state is None or REJECT -> Retains -inf and -1

        # Map initial and accept states to their dense numeric IDs
        start_state_idx = state_to_idx[char_dfa.start_state]
        accept_state_indices = {
            state_to_idx[state]
            for state in char_dfa.accept_states
            if state in state_to_idx
        }

        # Package everything into an immutable JsonFunctionCallDFA
        return JsonFunctionCallDFA(
            logit_masks=logit_masks,
            transitions=transitions,
            start_state=start_state_idx,
            accept_states=accept_state_indices,
            idx_to_state=idx_to_state,
        )
