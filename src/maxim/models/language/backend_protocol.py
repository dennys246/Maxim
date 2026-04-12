"""LLMBackend Protocol — formal contract for LLM backend implementations.

S2 Step 2A of simulator_upgrades_plan. Reverse-engineered from
router.py dispatch sites and the four existing backends (_Anthropic,
_OpenAI, _LlamaCpp, _PyTorchTransformers).

This is a structural Protocol — existing backends satisfy it implicitly
without inheriting from it. The Protocol exists so that:
  1. MockLLMBackend (tests/substrate/) can implement against a real contract
  2. mypy catches interface drift between backends and the router
  3. Future backends have a clear spec to implement

The router uses two dispatch paths:
  Path A (requires_prompt_formatting=True): calls complete() with a pre-formatted
    prompt string. Used by local backends (llama-cpp, transformers).
  Path B (requires_prompt_formatting=False): calls complete_with_usage() with
    structured system/user messages. Used by cloud backends (Anthropic, OpenAI).
  Path C (fallback): calls complete() if complete_with_usage is missing.

All backends MUST implement complete(). Cloud-style backends SHOULD also
implement complete_with_usage() for structured responses with token counts.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from maxim.models.language.types import LLMResponse


@runtime_checkable
class LLMBackend(Protocol):
    """Structural protocol for LLM backend implementations.

    Required attributes:
        requires_prompt_formatting: If True, the router pre-formats the
            prompt before calling complete(). If False, the router calls
            complete_with_usage() with separate system/user messages.

    Optional attributes (checked via getattr with defaults):
        supports_model_override: Backend accepts model_override kwarg
        supports_tool_use: Backend accepts tools kwarg (native tool calling)
        supports_streaming: Backend accepts stream kwarg

    Required methods:
        complete(): Text-in, text-out completion. Every backend must have this.

    Optional methods (checked via hasattr):
        complete_with_usage(): Structured completion returning LLMResponse
        warmup(): Pre-load model at startup, return True if ready
    """

    requires_prompt_formatting: bool

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        stop: tuple[str, ...],
    ) -> str:
        """Generate a text completion from a formatted prompt string.

        This is the universal entry point — all backends must implement it.
        Local backends (llama-cpp, transformers) do their work here.
        Cloud backends typically delegate to complete_with_usage().
        """
        ...

    def complete_with_usage(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
        stop: tuple[str, ...] = (),
        model_override: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        thinking: dict[str, Any] | None = None,
        stream: bool = False,
    ) -> LLMResponse:
        """Structured completion with token usage and metadata.

        Cloud-style backends implement this for richer response data.
        The router only calls this when requires_prompt_formatting is False.

        Optional kwargs are only passed when the corresponding
        supports_* flag is True on the backend instance.
        """
        ...

    def warmup(self) -> bool:
        """Pre-load model at startup. Return True if ready.

        Optional — the router checks hasattr before calling.
        """
        ...
