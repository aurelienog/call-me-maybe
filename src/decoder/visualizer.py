from __future__ import annotations

from typing import Callable
import numpy as np
import numpy.typing as npt


class GenerationVisualizer:
    """Renders real-time internal mechanics of grammar-constrained decoding."""

    @staticmethod
    def print_header(prompt: str) -> None:
        """Print the trace header for a single prompt decoding session."""
        print("\n" + "=" * 80)
        print(f" 🚀 CONSTRAINED DECODING TRACE | Prompt: {prompt!r}")
        print("=" * 80)

    @staticmethod
    def print_step(
        step: int,
        current_state: int,
        next_state: int,
        raw_logits: npt.NDArray[np.float32],
        masked_logits: npt.NDArray[np.float32],
        mask: npt.NDArray[np.float32],
        selected_token_id: int,
        normalized_token: str,
        current_stream_text: str,
        token_lookup: Callable[[int], str],
        top_k: int = 3,
    ) -> None:
        """
        Print detailed step execution metrics showing LLM top intent vs
        DFA constraints.

        Args:
            step: Current decoding step index.
            current_state: DFA state prior to transition.
            next_state: DFA state after transition.
            raw_logits: Original unconstrained LLM logits.
            masked_logits: Final logits after adding DFA mask.
            mask: DFA binary logit mask array (0.0 for allowed, -inf for
            blocked).
            selected_token_id: ID of the chosen token (argmax of
            masked_logits).
            normalized_token: Clean human-readable token text representation.
            current_stream_text: Accumulated JSON string generated so far.
            token_lookup: Callable to fetch token string representations by ID.
            top_k: Number of top unconstrained intentions to display.
        """
        # A. Top-K candidates the LLM wanted BEFORE grammar masking
        raw_top_ids = np.argsort(raw_logits)[-top_k:][::-1]

        # B. Count how many tokens are legally permitted in this DFA state
        allowed_count = int(np.sum(mask == 0.0))

        # C. Format the LLM's raw intentions with status indicators
        intentions: list[str] = []
        for tid in raw_top_ids:
            tok_str = token_lookup(int(tid))
            status = "✅" if mask[tid] == 0.0 else "❌ (Blocked)"
            intentions.append(f"{tok_str!r} {status}")

        print(f"\n🔹 [STEP {step:02d}] DFA State:"
              f" {current_state} -> {next_state}")
        print(
            f"   ├─ Grammar constraint : {allowed_count} / "
            f"{len(raw_logits)} tokens allowed"
        )
        print(f"   ├─ LLM Top Intentions : {', '.join(intentions)}")
        print(
            f"   ├─ Selected Token     : {normalized_token!r} "
            f"(ID: {selected_token_id})"
        )
        print(
            f"   └─ Current Stream     : \033[92m{current_stream_text!r}"
            "\033[0m"
        )

    @staticmethod
    def print_footer(json_result: str) -> None:
        """Print the final decoded JSON string."""
        print("\n" + "─" * 80)
        print(f" ✨ FINAL VALID JSON OUTPUT:\n\033[96m{json_result}\033[0m")
        print("=" * 80 + "\n")
