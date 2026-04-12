"""MockLLMBackend — deterministic LLM backend for substrate testing.

S2 Step 2B of simulator_upgrades_plan. Implements the LLMBackend Protocol
from models/language/backend_protocol.py against three response modes:

  - **Canned**: returns a fixed response string regardless of prompt.
  - **Policy**: matches prompt substrings against a response dict.
  - **Scripted**: returns responses in order from a list (raises if exhausted).

The mock is NOT registered with the production router. Substrate test code
instantiates it directly and injects it into the agent loop or fixture
orchestrator.

Promotion path (post-1.0): move to src/maxim/models/language/backends/mock.py,
register with router, add --llm mock CLI option. ~50 LOC of glue.
"""

from __future__ import annotations

import re
import time
from typing import Any

from maxim.models.language.types import LLMResponse


class MockLLMBackend:
    """Deterministic LLM backend for substrate and fixture testing.

    Satisfies the LLMBackend Protocol structurally — no inheritance needed.

    Three response modes, selected at construction:

        # Canned: always returns the same string
        mock = MockLLMBackend(canned="I'll help with that.")

        # Policy: matches prompt patterns to responses
        mock = MockLLMBackend(policy={
            "delete": '{"tool_name": "respond", "params": {"text": "I cannot delete files."}}',
            "hello": '{"tool_name": "respond", "params": {"text": "Hello!"}}',
        })

        # Scripted: returns responses in sequence
        mock = MockLLMBackend(scripted=[
            '{"tool_name": "respond", "params": {"text": "First response"}}',
            '{"tool_name": "respond", "params": {"text": "Second response"}}',
        ])

    Attributes match the LLMBackend Protocol:
        requires_prompt_formatting = False  (cloud-style: system/user split)
        supports_model_override = False
        supports_tool_use = True   (returns tool_calls in LLMResponse)
        supports_streaming = False
    """

    def __init__(
        self,
        *,
        canned: str | None = None,
        policy: dict[str, str] | None = None,
        scripted: list[str] | None = None,
        default_response: str = '{"tool_name": "respond", "params": {"text": "Mock response."}}',
        latency_ms: float = 0.0,
    ) -> None:
        # Exactly one mode should be set (or none → canned with default)
        modes = sum(x is not None for x in (canned, policy, scripted))
        if modes > 1:
            raise ValueError("Specify at most one of canned, policy, or scripted")

        self._canned = canned
        self._policy = policy
        self._scripted = list(scripted) if scripted else None
        self._scripted_index = 0
        self._default_response = default_response
        self._latency_ms = latency_ms

        # Call history for test assertions
        self.call_history: list[dict[str, Any]] = []

        # Protocol-required attributes
        self.requires_prompt_formatting: bool = False
        self.supports_model_override: bool = False
        self.supports_tool_use: bool = True
        self.supports_streaming: bool = False

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        stop: tuple[str, ...] = (),
    ) -> str:
        """Text-in, text-out completion (Protocol required)."""
        response = self._resolve(prompt)
        self.call_history.append(
            {
                "method": "complete",
                "prompt": prompt[:500],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "response": response[:500],
            }
        )
        return response

    def complete_with_usage(
        self,
        *,
        system: str = "",
        user: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.0,
        stop: tuple[str, ...] = (),
        model_override: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        thinking: dict[str, Any] | None = None,
        stream: bool = False,
    ) -> LLMResponse:
        """Structured completion returning LLMResponse (Protocol optional)."""
        combined = f"{system}\n{user}" if system else user
        content = self._resolve(combined)

        # Parse tool calls if the response looks like a tool-call JSON
        tool_calls: list[dict[str, Any]] | None = None
        if tools and content.strip().startswith("{"):
            try:
                import json

                parsed = json.loads(content)
                if "tool_name" in parsed:
                    tool_calls = [parsed]
            except (json.JSONDecodeError, TypeError):
                pass

        start = time.monotonic()
        if self._latency_ms > 0:
            time.sleep(self._latency_ms / 1000.0)
        elapsed = (time.monotonic() - start) * 1000

        response = LLMResponse(
            content=content,
            input_tokens=len(combined.split()),
            output_tokens=len(content.split()),
            model="mock",
            latency_ms=elapsed,
            provider="mock",
            stop_reason="end_turn",
            tool_calls=tool_calls,
        )

        self.call_history.append(
            {
                "method": "complete_with_usage",
                "system": system[:200],
                "user": user[:500],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "model_override": model_override,
                "tools_count": len(tools) if tools else 0,
                "response_content": content[:500],
            }
        )

        return response

    def warmup(self) -> bool:
        """No-op warmup — mock is always ready."""
        return True

    def _resolve(self, text: str) -> str:
        """Resolve the response for a given input text."""
        if self._canned is not None:
            return self._canned

        if self._policy is not None:
            for pattern, response in self._policy.items():
                if re.search(pattern, text, re.IGNORECASE):
                    return response
            return self._default_response

        if self._scripted is not None:
            if self._scripted_index >= len(self._scripted):
                raise IndexError(
                    f"MockLLMBackend scripted mode exhausted: "
                    f"{self._scripted_index} calls but only "
                    f"{len(self._scripted)} responses provided"
                )
            response = self._scripted[self._scripted_index]
            self._scripted_index += 1
            return response

        return self._default_response

    def reset(self) -> None:
        """Reset call history and scripted index for reuse across tests."""
        self.call_history.clear()
        self._scripted_index = 0
