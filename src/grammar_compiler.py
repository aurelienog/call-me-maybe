from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict

from .vocabulary_compiler import CompiledVocabulary
from .json_function_call_dfa import JsonFunctionCallDFA
from .json_grammar import JsonGrammar
from .char_dfa import CharDFA, CharState


class GrammarCompiler(BaseModel):
    """
    Compila la gramática JSON y el vocabulario en un DFA optimizado a nivel de tokens.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    def compile(
        self,
        grammar: JsonGrammar,
        vocabulary: CompiledVocabulary,
        allowed_functions: list[str],
    ) -> JsonFunctionCallDFA:
        """
        Paso B (Fase 2): Convierte el CharDFA a nivel de caracteres en un TokenDFA.
        
        Para cada estado del CharDFA y cada token_id del vocabulario, simula si la
        cadena del token se consume completamente sin caer en el estado REJECT.
        """
        char_dfa = CharDFA(allowed_function_names=allowed_functions)
        
        # Obtenemos todos los estados válidos del CharDFA (excluyendo el de rechazo)
        valid_char_states = [state for state in CharState if state != CharState.REJECT]
        
        # Mapeo biyectivo de CharState a un ID numérico denso (0..N-1) para indexación rápida
        state_to_idx: dict[CharState, int] = {
            state: i for i, state in enumerate(valid_char_states)
        }
        idx_to_state: dict[int, CharState] = {
            i: state for state, i in state_to_idx.items()
        }

        num_states = len(valid_char_states)
        vocab_size = vocabulary.vocab_size

        # 1. Matriz de máscaras de validez para Logits: (num_states, vocab_size)
        # Inicializamos con -inf (inválido por defecto)
        logit_masks = np.full(
            (num_states, vocab_size),
            fill_value=-np.inf,
            dtype=np.float32,
        )

        # 2. Tabla de Transiciones por Token: (num_states, vocab_size) -> next_state_idx
        # Inicializamos con -1 para indicar transiciones imposibles
        transitions = np.full(
            (num_states, vocab_size),
            fill_value=-1,
            dtype=np.int32,
        )

        # --- FASE DE PLEGADO (SIMULACIÓN TOKEN POR TOKEN) ---
        for char_state in valid_char_states:
            state_idx = state_to_idx[char_state]

            for token_id, compiled_token in vocabulary.tokens.items():
                token_text = compiled_token.normalized_text

                # Caso límite: Tokens vacíos no alteran el estado
                if not token_text:
                    continue

                # Simulación: Consumir la cadena del token carácter a carácter desde char_state
                end_state = char_dfa.simulate_string(
                    start_state=char_state,
                    text=token_text,
                )

                # CASO B (VÁLIDO): La cadena se consumió completa y aterrizó en un estado no-REJECT
                if end_state is not None and end_state != CharState.REJECT:
                    end_state_idx = state_to_idx[end_state]

                    # Habilitamos el token sumando 0.0 al logit en la inferencia
                    logit_masks[state_idx, token_id] = 0.0

                    # Registramos el estado de destino
                    transitions[state_idx, token_id] = end_state_idx

                # CASO A (INVÁLIDO): end_state es None o REJECT -> Se mantiene en -inf y -1

        # Mapeamos los estados iniciales y finales a sus IDs numéricos
        start_state_idx = state_to_idx[char_dfa.start_state]
        accept_state_indices = {
            state_to_idx[state]
            for state in char_dfa.accept_states
            if state in state_to_idx
        }

        # Empaquetamos todo en el JsonFunctionCallDFA inmutable
        return JsonFunctionCallDFA(
            logit_masks=logit_masks,
            transitions=transitions,
            start_state=start_state_idx,
            accept_states=accept_state_indices,
            idx_to_state=idx_to_state,
        )