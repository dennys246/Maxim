"""Router typed-exception integration tests (Plan 3 R2.5).

Covers :meth:`LLMRouter._try_provider` and
:meth:`LLMRouter._complete_text_locked` for the new typed-exception
branches added in Plan 3 R2.5. Each test:

- Mocks ``_invoke_backend`` to raise one :class:`BackendError` subclass.
- Asserts the matching backoff helper was called
  (``_note_provider_overload``, ``_set_long_backoff``,
  ``_set_short_backoff``, ``_note_provider_failure``).
- Asserts the recorded attempt outcome.
- For the exhausted-dispatch path, asserts the aggregated
  ``dispatch_exhausted`` WARN is emitted once with all attempts.

Critical regression guards:

- **Auth must NOT be classified as inference_broken.** This is the same
  bug the R2 stage-2 probe review round caught; the router path has its
  own ``except`` ordering and needs its own test.
- **Unclassified (non-BackendError) exceptions bump
  ``backend_unclassified_errors_total``.** Non-zero count means the typed
  hierarchy has a gap.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from maxim.models.language.config import LLMConfig
from maxim.models.language.lane_metrics import (
    metrics_snapshot,
    reset_backend_unclassified_errors,
)
from maxim.models.language.router import LLMRouter
from maxim.models.language.types import (
    INFERENCE_BROKEN_BACKOFF_S,
    BackendAuthFailed,
    BackendDown,
    BackendError,
    BackendInferenceBroken,
    BackendModelMissing,
    BackendOverloaded,
    BackendTimeout,
    ProviderState,
)


@pytest.fixture(autouse=True)
def _reset_unclassified():
    reset_backend_unclassified_errors()
    yield
    reset_backend_unclassified_errors()


def _make_router() -> LLMRouter:
    """Minimal LLMRouter with one fake provider. No real backend init."""
    import dataclasses

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
    router = LLMRouter(cfg)
    # Seed the ProviderState so the backoff helpers find something to
    # update (LLMRouter.__init__ already creates states for all
    # providers, but be defensive).
    router._provider_states.setdefault("fake-peer", ProviderState())
    return router


def _call_try_provider(router: LLMRouter, side_effect: Exception):
    """Invoke ``_try_provider`` with ``_invoke_backend`` raising.

    Acquires ``router._inference_lock`` before the call because
    R3 review fix added an assertion in ``_record_attempt_outcome``
    that the dispatch-attempt buffer is only mutated under the lock.
    Production code holds the lock via ``_complete_text`` around a
    full dispatch; the unit tests here exercise ``_try_provider`` in
    isolation so they must acquire the lock explicitly.
    """
    # Fake backend passes the _get_backend_for_provider branch without
    # hitting a real import. Use _backends cache to short-circuit the
    # lookup.
    fake_backend = MagicMock()
    fake_backend.complete = MagicMock()
    fake_backend.complete_with_usage = MagicMock()
    router._backends["fake-peer"] = fake_backend

    with router._inference_lock:
        with patch.object(router, "_invoke_backend", side_effect=side_effect):
            return router._try_provider(
                provider_key="fake-peer",
                system="",
                user="hi",
                temperature=0.0,
                max_tokens=1,
                prompt_tokens=1,
                budget_tier="local_only",
                tools=None,
                thinking=None,
                stream=False,
                request_context=None,
                now=0.0,
            )


class TestTypedExceptionHandlers:
    def test_backend_overloaded_calls_note_provider_overload(self):
        router = _make_router()
        exc = BackendOverloaded(
            "fake-peer",
            status=429,
            retry_after_s=5.0,
            suggested_peer="rtx-leader",
            queue_depth=3,
        )
        with patch.object(router, "_note_provider_overload", wraps=router._note_provider_overload) as spy:
            result = _call_try_provider(router, exc)
        assert result == ("", None, "failed")
        spy.assert_called_once()
        assert spy.call_args.kwargs["retry_after_s"] == 5.0
        # Attempt log carries outcome + retry_after_s extra
        assert len(router._dispatch_attempts) == 1
        attempt = router._dispatch_attempts[0]
        assert attempt["outcome"] == "overloaded"
        assert attempt["retry_after_s"] == 5.0

    def test_backend_overloaded_records_suggested_peer_hint(self):
        router = _make_router()
        exc = BackendOverloaded(
            "fake-peer",
            retry_after_s=0.0,
            suggested_peer="rtx-leader",
        )
        _call_try_provider(router, exc)
        assert router._last_suggested_peer == "rtx-leader"

    def test_backend_auth_failed_applies_300s_long_backoff(self):
        router = _make_router()
        import time

        t0 = time.time()
        exc = BackendAuthFailed("fake-peer", status=401)
        _call_try_provider(router, exc)
        state = router._provider_states["fake-peer"]
        # 300s long cooldown — room for loose float comparison
        assert state.backoff_until >= t0 + 290
        assert state.backoff_until <= t0 + 310
        assert router._dispatch_attempts[0]["outcome"] == "auth_failed"

    def test_backend_auth_failed_is_not_inference_broken(self):
        """Regression guard: auth 401/403 must not mis-classify as
        inference_broken (the 2026-04-12 stage-2 probe bug). Router's
        except ordering is specific-before-general; this test locks it
        in."""
        router = _make_router()
        exc = BackendAuthFailed("fake-peer", status=401)
        _call_try_provider(router, exc)
        # If routed through the inference_broken branch, the backoff
        # would be 15s. We expect ~300s.
        state = router._provider_states["fake-peer"]
        import time

        assert state.backoff_until - time.time() > 30.0

    def test_backend_model_missing_applies_60s_long_backoff(self):
        router = _make_router()
        import time

        t0 = time.time()
        exc = BackendModelMissing(
            "fake-peer",
            status=404,
            requested_model="qwen2.5-14b-instruct",
        )
        _call_try_provider(router, exc)
        state = router._provider_states["fake-peer"]
        assert state.backoff_until >= t0 + 55
        assert state.backoff_until <= t0 + 65
        attempt = router._dispatch_attempts[0]
        assert attempt["outcome"] == "model_missing"
        assert attempt["requested_model"] == "qwen2.5-14b-instruct"

    def test_backend_inference_broken_applies_15s_short_backoff(self):
        router = _make_router()
        import time

        t0 = time.time()
        exc = BackendInferenceBroken("fake-peer")
        _call_try_provider(router, exc)
        state = router._provider_states["fake-peer"]
        # Matches INFERENCE_BROKEN_BACKOFF_S = 15.0, loose float comparison
        elapsed = state.backoff_until - t0
        assert 14.0 <= elapsed <= 16.0
        assert elapsed == pytest.approx(INFERENCE_BROKEN_BACKOFF_S, abs=1.0)
        assert router._dispatch_attempts[0]["outcome"] == "inference_broken"

    def test_backend_timeout_records_elapsed_s(self):
        router = _make_router()
        exc = BackendTimeout("fake-peer", elapsed_s=2.3)
        _call_try_provider(router, exc)
        attempt = router._dispatch_attempts[0]
        assert attempt["outcome"] == "timeout"
        assert attempt["elapsed_s"] == 2.3

    def test_backend_down_records_http_status(self):
        router = _make_router()
        exc = BackendDown("fake-peer", status=502)
        _call_try_provider(router, exc)
        attempt = router._dispatch_attempts[0]
        assert attempt["outcome"] == "down"
        assert attempt["http_status"] == 502

    def test_backend_error_base_class_records_generic_outcome(self):
        """Unmapped BackendError subclass falls into the
        ``except BackendError`` branch and records
        ``generic_backend_error``."""
        router = _make_router()
        exc = BackendError("fake-peer", status=418, fix_hint="teapot")
        _call_try_provider(router, exc)
        attempt = router._dispatch_attempts[0]
        assert attempt["outcome"] == "generic_backend_error"


class TestUnclassifiedSafetyNet:
    def test_non_backend_error_bumps_unclassified_counter(self):
        """Non-BackendError exception trips the safety net and bumps
        ``backend_unclassified_errors_total``. Non-zero count is a bug
        signal."""
        router = _make_router()

        class _RogueError(RuntimeError):
            pass

        _call_try_provider(router, _RogueError("rogue"))
        snap = metrics_snapshot()["backend_unclassified_errors_total"]
        assert snap["count"] == 1
        assert snap["by_provider"].get("fake-peer") == 1
        assert router._dispatch_attempts[0]["outcome"] == "unclassified"
        assert router._dispatch_attempts[0]["error"] == "_RogueError"

    def test_typed_exception_does_not_bump_counter(self):
        """Regression guard: BackendError subclasses must NOT land in the
        safety net. The counter stays at zero."""
        router = _make_router()
        _call_try_provider(router, BackendDown("fake-peer"))
        snap = metrics_snapshot()["backend_unclassified_errors_total"]
        assert snap["count"] == 0


class TestDispatchExhausted:
    def test_dispatch_exhausted_emits_one_warn_with_all_attempts(self, caplog):
        """Wires up the aggregated failure log. One WARN line per failed
        dispatch, with all per-provider attempts listed inline.

        **R3 review fix:** beefed up to assert the full payload reaches
        the structured log record (agent_id, session_id, request_id,
        lane, total_elapsed_ms, attempts). The original test only
        asserted the event substring was present in the message, which
        would have let a regression dropping the multi-agent fields
        sail through. R3 also locked ``_emit_dispatch_exhausted`` to
        use ``_normalize_request_context`` instead of reading
        ``ctx.get("agent")`` inline — this test locks in the
        canonical-shim delegation.
        """
        import dataclasses
        import logging

        cfg = dataclasses.replace(
            LLMConfig(),
            enabled=True,
            providers={
                "a": {"type": "maxim_peer", "base_url": "http://127.0.0.1:1/v1", "model": "m"},
                "b": {"type": "maxim_peer", "base_url": "http://127.0.0.1:2/v1", "model": "m"},
            },
        )
        router = LLMRouter(cfg)

        # Seed two different failure modes for A and B
        def fake_try_provider(*, provider_key, **kwargs):
            if provider_key == "a":
                router._record_attempt_outcome("a", outcome="overloaded", extra={"retry_after_s": 5.0})
                return "", None, "failed"
            router._record_attempt_outcome("b", outcome="timeout", extra={"elapsed_s": 2.1})
            return "", None, "failed"

        with caplog.at_level(logging.WARNING):
            with patch.object(router, "_try_provider", side_effect=fake_try_provider):
                with patch.object(
                    router,
                    "_candidate_providers",
                    return_value=(["a", "b"], "local_only", {}),
                ):
                    with router._inference_lock:
                        text, usage = router._complete_text_locked(
                            "",
                            "hi",
                            temperature=0.0,
                            max_tokens=1,
                            request_context={
                                "agent_id": "npc-mother",
                                "session_id": "sim-42",
                                "request_id": "req-abc",
                                "lane": "large",
                            },
                        )
        assert text == "" and usage is None
        # Find the dispatch_exhausted record
        exhausted = [r for r in caplog.records if "dispatch_exhausted" in r.getMessage()]
        assert exhausted, "expected one dispatch_exhausted WARN log line"
        # R3 review fix: assert the actual payload fields. Structured
        # logging attaches the data dict as attributes on the LogRecord
        # via StructuredFormatter's flattening — read the "event_data"
        # attribute (or equivalent) that log_structured populates.
        rec = exhausted[-1]
        # Walk the record's attributes looking for the multi-agent
        # fields. log_structured attaches them as top-level record
        # attributes so the StructuredFormatter can flatten them.
        rec_dict = vars(rec)
        # At minimum, agent_id / session_id / request_id / lane / attempts
        # must all have reached the logging layer (either as direct
        # attributes or nested in a data dict). Accept either shape.
        serialized = str(rec_dict) + str(getattr(rec, "event_data", {}))
        assert "npc-mother" in serialized, f"agent_id missing from log: {serialized[:400]}"
        assert "sim-42" in serialized, f"session_id missing from log: {serialized[:400]}"
        assert "req-abc" in serialized, f"request_id missing from log: {serialized[:400]}"
        assert "large" in serialized, f"lane missing from log: {serialized[:400]}"
        # Per-attempt breakdown must include both providers' outcomes
        assert "overloaded" in serialized, "provider A outcome missing"
        assert "timeout" in serialized, "provider B outcome missing"

    def test_successful_dispatch_does_not_emit_exhausted_log(self, caplog):
        """Sanity: a happy-path dispatch must not emit the aggregated
        failure log."""
        import dataclasses
        import logging

        cfg = dataclasses.replace(
            LLMConfig(),
            enabled=True,
            providers={
                "a": {"type": "maxim_peer", "base_url": "http://127.0.0.1:1/v1", "model": "m"},
            },
        )
        router = LLMRouter(cfg)

        def fake_try_provider(*, provider_key, **kwargs):
            return "ok", {"input_tokens": 1, "output_tokens": 1}, "success"

        with caplog.at_level(logging.WARNING):
            with patch.object(router, "_try_provider", side_effect=fake_try_provider):
                with patch.object(
                    router,
                    "_candidate_providers",
                    return_value=(["a"], "local_only", {}),
                ):
                    with router._inference_lock:
                        text, usage = router._complete_text_locked("", "hi", temperature=0.0, max_tokens=1)
        assert text == "ok"
        exhausted = [r for r in caplog.records if "dispatch_exhausted" in r.getMessage()]
        assert not exhausted, "dispatch_exhausted should not fire on success"


# ─── BACKEND_CLASSES dispatch integration (R3 review fix) ──────────────
#
# Architecture-lens reviewer finding #6: the ``"maxim_peer"`` provider
# type branch in ``LLMRouter._get_backend_for_provider`` was never
# exercised by the unit tests — they stuffed a MagicMock into
# ``router._backends`` directly to bypass the lookup. A typo in either
# ``BACKEND_CLASSES`` or the router's lookup would slip through. These
# tests build a real LLMConfig with a ``"type": "maxim_peer"`` provider
# and assert the router instantiates the right class.


class TestBackendClassesDispatch:
    def test_maxim_peer_type_instantiates_maxim_peer_backend(self):
        """The dispatch table must resolve ``"maxim_peer"`` to
        :class:`_MaximPeerBackend`. Regression guard for the
        two-hop dispatch drift the architecture reviewer flagged."""
        import dataclasses

        from maxim.models.language.maxim_peer_backend import _MaximPeerBackend

        cfg = dataclasses.replace(
            LLMConfig(),
            enabled=True,
            providers={
                "peer-x": {
                    "type": "maxim_peer",
                    "base_url": "http://127.0.0.1:9995/v1",
                    "model": "m",
                    "allow_local_endpoints": True,
                },
            },
        )
        router = LLMRouter(cfg)
        backend = router._get_backend_for_provider("peer-x")
        assert backend is not None
        assert isinstance(backend, _MaximPeerBackend), f"expected _MaximPeerBackend, got {type(backend).__name__}"

    def test_hyphenated_maxim_peer_type_also_dispatches(self):
        """``"maxim-peer"`` (hyphenated operator spelling) must resolve
        to the same class — ``resolve_backend_class`` normalises the
        identifier by stripping hyphens."""
        import dataclasses

        from maxim.models.language.maxim_peer_backend import _MaximPeerBackend

        cfg = dataclasses.replace(
            LLMConfig(),
            enabled=True,
            providers={
                "peer-y": {
                    "type": "maxim-peer",
                    "base_url": "http://127.0.0.1:9994/v1",
                    "model": "m",
                    "allow_local_endpoints": True,
                },
            },
        )
        router = LLMRouter(cfg)
        backend = router._get_backend_for_provider("peer-y")
        assert backend is not None
        assert isinstance(backend, _MaximPeerBackend)

    def test_resolve_backend_class_unknown_returns_none(self):
        """An unknown identifier must return None so the router falls
        through to its "unknown provider type" warning path."""
        from maxim.runtime.lane_backends import resolve_backend_class

        assert resolve_backend_class("not-a-real-backend") is None
        assert resolve_backend_class("") is None

    def test_resolve_backend_class_maxim_peer(self):
        """Direct lookup through the dispatch table returns the
        correct class."""
        from maxim.models.language.maxim_peer_backend import _MaximPeerBackend
        from maxim.runtime.lane_backends import resolve_backend_class

        assert resolve_backend_class("maxim_peer") is _MaximPeerBackend
        assert resolve_backend_class("maxim-peer") is _MaximPeerBackend


# ─── Plan 4 A.1: request_context capability-flag forwarding ───────────────


class TestRequestContextForwarding:
    """Regression guards for Plan 4 A.1: _invoke_backend must forward
    ``request_context`` to backends that declare
    ``accepts_request_context = True``, and must NOT forward it to
    backends that don't (e.g., _OpenAIBackend which has no **kwargs
    catch-all and would crash with TypeError).

    This was the Phase D observability gap root cause: the kwarg was
    being dropped on the floor during the kwargs dict construction in
    _invoke_backend, so every peer_backend_call event logged
    agent_id=null.
    """

    def _setup_router_with_mock_backend(self, backend_cls_attrs: dict):
        """Build a router and wire a mock backend with configurable
        class-level capability flags (supports_model_override,
        accepts_request_context, ...)."""
        from maxim.models.language.types import LLMResponse

        router = _make_router()
        mock_backend = MagicMock()
        # Mimic the class-attribute lookup pattern the router uses via
        # getattr(backend, "name", default)
        for k, v in backend_cls_attrs.items():
            setattr(mock_backend, k, v)
        mock_backend.requires_prompt_formatting = False
        mock_backend.complete_with_usage = MagicMock(
            return_value=LLMResponse(content="ok", provider="fake-peer", model="m"),
        )
        router._backends["fake-peer"] = mock_backend
        return router, mock_backend

    def test_kwarg_forwarded_when_capability_flag_set(self):
        """When ``accepts_request_context = True`` and request_context
        is not None, the dict must appear in the complete_with_usage
        kwargs. Without this, peer_backend_call emits agent_id=null."""
        router, backend = self._setup_router_with_mock_backend(
            {"accepts_request_context": True},
        )
        request_ctx = {
            "agent_id": "npc-mother",
            "session_id": "sim-42",
            "request_id": "r-abc",
            "lane": "large",
        }
        with router._inference_lock:
            router._invoke_backend(
                backend=backend,
                provider_key="fake-peer",
                redacted_system="",
                redacted_user="hi",
                model="m",
                model_override=None,
                temperature=0.0,
                max_tokens=1,
                tools=None,
                thinking=None,
                stream=False,
                redaction_result=None,
                request_context=request_ctx,
                now=0.0,
            )
        call = backend.complete_with_usage.call_args
        assert call.kwargs.get("request_context") == request_ctx

    def test_kwarg_NOT_forwarded_when_capability_flag_absent(self):
        """Cloud backends (_OpenAIBackend, _AnthropicBackend) don't accept
        request_context. Forwarding to them would crash with TypeError.
        The capability-flag check prevents this. This test locks in the
        invariant — removing the flag check must also update this test."""
        router, backend = self._setup_router_with_mock_backend(
            {"accepts_request_context": False},
        )
        with router._inference_lock:
            router._invoke_backend(
                backend=backend,
                provider_key="fake-peer",
                redacted_system="",
                redacted_user="hi",
                model="m",
                model_override=None,
                temperature=0.0,
                max_tokens=1,
                tools=None,
                thinking=None,
                stream=False,
                redaction_result=None,
                request_context={"agent_id": "should-not-reach-backend"},
                now=0.0,
            )
        call = backend.complete_with_usage.call_args
        assert "request_context" not in call.kwargs

    def test_kwarg_NOT_forwarded_when_request_context_is_none(self):
        """Even with the capability flag set, a None request_context
        must not be forwarded. (Keeps the backend's default-arg path
        live for backends that branch on None.)"""
        router, backend = self._setup_router_with_mock_backend(
            {"accepts_request_context": True},
        )
        with router._inference_lock:
            router._invoke_backend(
                backend=backend,
                provider_key="fake-peer",
                redacted_system="",
                redacted_user="hi",
                model="m",
                model_override=None,
                temperature=0.0,
                max_tokens=1,
                tools=None,
                thinking=None,
                stream=False,
                redaction_result=None,
                request_context=None,
                now=0.0,
            )
        call = backend.complete_with_usage.call_args
        assert "request_context" not in call.kwargs
