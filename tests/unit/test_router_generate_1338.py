"""#1338 (advisor item 6): ``LLMRouter.generate`` returns text or raises a typed error, and its callers reach it.

Every test drives a REAL ``LLMRouter`` (the full ``_complete_text`` dispatch: lock, provider loop, accounting) over a
stub backend object, never a spec-less ``MagicMock`` router: a caller that names a method the router does not have
fails here (the defect the wire-integrity review found in d362b481).
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from maxim.exceptions import ConfigurationError, LeaderRefusal, ModelError, ModelInferenceError
from maxim.models.language.config import LLMConfig
from maxim.models.language.router import LLMRouter
from maxim.models.language.types import LLMResponse


class _StubBackend:
    """The backend surface ``_invoke_backend`` uses for a non-prompt-formatting backend."""

    requires_prompt_formatting = False

    def __init__(self, content: str = "", *, raises: Exception | None = None) -> None:
        self.content = content
        self.raises = raises
        self.calls: list[dict[str, Any]] = []

    def complete_with_usage(self, **kwargs: Any) -> LLMResponse:
        self.calls.append(kwargs)
        if self.raises is not None:
            raise self.raises
        return LLMResponse(content=self.content, input_tokens=3, output_tokens=2, model="fake-model", provider="p")


def _router(backend: _StubBackend, *, enabled: bool = True) -> LLMRouter:
    cfg = dataclasses.replace(
        LLMConfig(),
        enabled=enabled,
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
    router = LLMRouter(cfg)
    router._backends["fake-peer"] = backend
    return router


def _texts(call: dict[str, Any]) -> str:
    return repr(call)


def test_generate_returns_the_backend_text_and_sends_the_system_prompt() -> None:
    backend = _StubBackend("  Halt, stranger.  ")
    out = _router(backend).generate("A stranger approaches.", system="You are a guard.", max_tokens=50)
    assert out == "Halt, stranger."
    assert len(backend.calls) == 1
    sent = _texts(backend.calls[0])
    assert "A stranger approaches." in sent and "You are a guard." in sent


def test_generate_empty_text_is_a_typed_error_not_a_stub() -> None:
    backend = _StubBackend("   ")
    with pytest.raises(ModelInferenceError, match="no text"):
        _router(backend).generate("hello")
    assert backend.calls, "the backend was never called"


def test_generate_backend_failure_is_a_typed_error() -> None:
    backend = _StubBackend(raises=RuntimeError("backend down"))
    with pytest.raises(ModelError):
        _router(backend).generate("hello")
    assert backend.calls


def test_generate_on_a_disabled_router_raises() -> None:
    with pytest.raises(ModelInferenceError, match="disabled"):
        _router(_StubBackend("x"), enabled=False).generate("hello")


def test_generate_wraps_a_dispatch_exception_but_lets_a_refusal_through(monkeypatch: pytest.MonkeyPatch) -> None:
    router = _router(_StubBackend("x"))

    def _boom(*a: Any, **k: Any) -> Any:
        raise ValueError("dispatch broke")

    monkeypatch.setattr(router, "_complete_text", _boom)
    with pytest.raises(ModelInferenceError, match="dispatch broke"):
        router.generate("hello")

    def _refuse(*a: Any, **k: Any) -> Any:
        raise LeaderRefusal("refused")

    monkeypatch.setattr(router, "_complete_text", _refuse)
    with pytest.raises(LeaderRefusal):
        router.generate("hello")


def _guard(personality: str | None) -> Any:
    from types import SimpleNamespace

    inst = SimpleNamespace(agent_id="guard", role="npc", personality=personality, hippocampus=None, nac=None)
    inst.shutdown = lambda: None
    return inst


def test_create_pool_run_turn_reaches_the_backend_through_a_real_router() -> None:
    import maxim

    backend = _StubBackend('"Halt, stranger."')
    pool = maxim.create.pool(llm_router=_router(backend))
    pool.add(_guard("Stern guard. Watches the gate."))
    try:
        result = pool.run_turn("guard", "A stranger approaches.")
    finally:
        pool.shutdown()
    assert result.error is None, result.error
    assert result.response == "Halt, stranger."
    assert len(backend.calls) == 1
    assert "Stern guard" in _texts(backend.calls[0])


def test_run_turn_without_a_personality_still_calls_the_llm() -> None:
    """Arch S2: ``create.agent``'s default ``personality=None`` must not skip the LLM and return a silent None."""
    import maxim

    backend = _StubBackend("Who goes there?")
    pool = maxim.create.pool(llm_router=_router(backend))
    pool.add(_guard(None))
    try:
        result = pool.run_turn("guard", "A stranger approaches.")
    finally:
        pool.shutdown()
    assert len(backend.calls) == 1
    assert result.response == "Who goes there?" and result.error is None


def test_run_turn_llm_failure_through_a_real_router_is_typed() -> None:
    import maxim

    backend = _StubBackend("")
    pool = maxim.create.pool(llm_router=_router(backend))
    pool.add(_guard("Stern guard."))
    try:
        result = pool.run_turn("guard", "A stranger approaches.")
    finally:
        pool.shutdown()
    assert backend.calls
    assert result.response is None
    assert result.error and result.error_type == "ModelInferenceError"


def test_tool_pain_bridge_reflection_uses_the_real_router() -> None:
    from types import SimpleNamespace

    from maxim.bridges.tool_pain_bridge import ToolPainBridge

    backend = _StubBackend("Try a smaller file next time.")
    bridge = ToolPainBridge(nac=None, pain_detector=None, scn=None, llm=_router(backend))
    link = SimpleNamespace(last_rpe=0.9, predicted_value=0.8)
    out = bridge._generate_reflection([link], {"tool_name": "read_file"}, "too large", {"path": "x"})
    assert out == "Try a smaller file next time."
    assert len(backend.calls) == 1


class TestCreateRouterRaises:
    def _no_router(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import maxim.runtime.lane_backends as lb

        monkeypatch.setattr(lb, "build_primary_router", lambda **kw: (None, None))

    def test_a_cloud_profile_without_its_key_names_the_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import maxim

        self._no_router(monkeypatch)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(ConfigurationError, match="ANTHROPIC_API_KEY"):
            maxim.create.router("claude-sonnet-4-6")

    def test_a_local_profile_names_the_local_profile(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import maxim

        self._no_router(monkeypatch)
        with pytest.raises(ConfigurationError, match="local LLM profile"):
            maxim.create.router("mistral-7b")

    def test_a_built_router_is_returned(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import maxim
        import maxim.runtime.lane_backends as lb

        sentinel = object()
        monkeypatch.setattr(lb, "build_primary_router", lambda **kw: (sentinel, None))
        assert maxim.create.router("mistral-7b") is sentinel


def test_generate_llm_response_without_a_router_is_a_typed_error() -> None:
    """The NPC LLM path never falls back to the persona stub: with no router it raises ``ModelInferenceError``."""
    from maxim.runtime.agent_pool import AgentPool

    pool = AgentPool()
    assert pool._llm_router is None
    with pytest.raises(ModelInferenceError, match="no LLM router"):
        pool._generate_llm_response("npc_a", "gruff", "hello")
