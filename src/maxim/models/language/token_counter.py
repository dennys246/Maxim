from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class TokenCounter(Protocol):
    """Protocol for counting tokens in text."""

    def count_tokens(self, text: str) -> int: ...


class CharEstimateCounter:
    """Fallback token counter using ~3 chars per token estimate.

    Structured prompts (JSON, headers, formatting) average ~3.0-3.5 chars/token
    on common tokenizers. Using //3 is intentionally conservative to prevent
    context overflow when the real tokenizer isn't available.
    """

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 3)


class LlamaCppTokenCounter:
    """Token counter backed by llama-cpp-python's actual tokenizer."""

    def __init__(self, llm: Any) -> None:
        self._llm = llm

    def count_tokens(self, text: str) -> int:
        try:
            return len(self._llm.tokenize(text.encode("utf-8")))
        except Exception:
            return max(1, len(text) // 3)


class _LazyTokenCounter:
    """Token counter that upgrades to the real tokenizer when the model loads.

    At construction time, the LLM model may still be warming up in a background
    thread. This counter starts with CharEstimateCounter and transparently
    upgrades to LlamaCppTokenCounter once the backend model is loaded.
    """

    def __init__(self, router: Any) -> None:
        self._router = router
        self._real_counter: TokenCounter | None = None

    def _try_upgrade(self) -> TokenCounter:
        if self._real_counter is not None:
            return self._real_counter
        # Check if backend model has loaded since construction
        backend = self._router._get_tokenizer_backend()
        if backend is not None and hasattr(backend, "_llm") and backend._llm is not None:
            self._real_counter = LlamaCppTokenCounter(backend._llm)
            return self._real_counter
        return CharEstimateCounter()

    def count_tokens(self, text: str) -> int:
        return self._try_upgrade().count_tokens(text)
