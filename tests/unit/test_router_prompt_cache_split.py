"""Router system/user split for prompt caching (prompt_caching_for_cloud_backends.md Phase 1).

``LLMRouter._generate_tool_response`` splits a ``TOOL_PROMPT`` payload on
``PROMPT_SEGMENT_DELIMITER`` and routes the byte-stable prefix into the system
message (where cloud cache_control lives) and the dynamic remainder into the
user message. Payloads without the delimiter (followup prompts, legacy callers)
keep everything in the user message.
"""

from __future__ import annotations

import dataclasses
from unittest.mock import MagicMock, patch

from maxim.agents.prompt_builder import PROMPT_SEGMENT_DELIMITER
from maxim.models.language.config import LLMConfig
from maxim.models.language.cloud_dispatch import SYSTEM_TOOL_RESPONSE
from maxim.models.language.router import LLMRouter


def _make_router() -> LLMRouter:
    cfg = dataclasses.replace(
        LLMConfig(),
        enabled=True,
        providers={
            "fake-peer": {
                "type": "maxim_peer",
                "base_url": "http://127.0.0.1:9999/v1",
                "api_key_env": "FAKE_KEY",
                "model": "fake-model",
                "allow_local_endpoints": True,
                "pricing_required": False,
            }
        },
    )
    return LLMRouter(cfg)


def _capture_complete_text(router: LLMRouter, tool_prompt: str) -> dict[str, str]:
    captured: dict[str, str] = {}

    def fake(system, user, **kw):  # noqa: ANN001
        captured["system"] = system
        captured["user"] = user
        return '{"action":{"tool_name":"respond","params":{}},"confidence":0.9,"reasoning":"x"}', None

    with patch.object(router, "_complete_text", side_effect=fake):
        router._generate_tool_response(MagicMock(), tool_prompt, 0.0, 64)
    return captured


def test_stable_prefix_routed_to_system():
    router = _make_router()
    cap = _capture_complete_text(router, f"STABLE-PREFIX-XYZ{PROMPT_SEGMENT_DELIMITER}DYNAMIC-BODY-ABC")
    assert "STABLE-PREFIX-XYZ" in cap["system"]
    assert "STABLE-PREFIX-XYZ" not in cap["user"]
    assert "DYNAMIC-BODY-ABC" in cap["user"]
    assert "DYNAMIC-BODY-ABC" not in cap["system"]
    # Base JSON-only constant precedes the appended prefix → whole system block
    # is byte-stable and ends at the cache breakpoint.
    assert cap["system"].startswith(SYSTEM_TOOL_RESPONSE)


def test_no_delimiter_keeps_everything_in_user():
    """Followup prompts / legacy callers (no delimiter) behave exactly as
    before: system is the bare constant, all content stays in user."""
    router = _make_router()
    cap = _capture_complete_text(router, "PLAIN-FOLLOWUP-PROMPT-NO-DELIMITER")
    assert cap["system"] == SYSTEM_TOOL_RESPONSE
    assert "PLAIN-FOLLOWUP-PROMPT-NO-DELIMITER" in cap["user"]


def test_empty_stable_prefix_leaves_system_constant():
    """An empty stable segment (delimiter present, prefix blank) must not append
    a trailing separator to the system constant."""
    router = _make_router()
    cap = _capture_complete_text(router, f"{PROMPT_SEGMENT_DELIMITER}ONLY-DYNAMIC")
    assert cap["system"] == SYSTEM_TOOL_RESPONSE
    assert "ONLY-DYNAMIC" in cap["user"]


def test_planning_mode_detected_across_segments():
    """PLANNING-mode detection runs on the full payload before the split — the
    banner lives in the stable prefix, so detecting on the dynamic remainder
    alone would miss it."""
    router = _make_router()
    prefix = "PLANNING MODE - YOU MUST ASK PERMISSION ... APPROVAL"
    cap = _capture_complete_text(router, f"{prefix}{PROMPT_SEGMENT_DELIMITER}the dynamic turn body")
    # Planning system prompt mentions PLANNING; the stable prefix is appended.
    assert "PLANNING" in cap["system"]
    assert prefix in cap["system"]
    assert "the dynamic turn body" in cap["user"]
