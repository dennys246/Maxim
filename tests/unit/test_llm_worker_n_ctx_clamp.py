"""Regression guard for the LLMWorker n_ctx clamp (1.1 cut-line item 3).

THE BUG THIS PINS (CLAUDE.md "run-config single source" known bug): the
pre-fix constructor took ``max()`` across provider contexts — and only when a
CLOUD provider was configured — so a mixed lane (local primary 16k + cloud
fallback 200k) budgeted prompts at 200k that the local server rejects with a
raw HTTP 500 (``down_500`` → provider cooldown → every subsequent call
``_llm_unavailable`` → the agent silently takes 0 real actions). The whole
thing sat inside a bare ``except: pass``, and local-only runs never clamped
at all.

Post-fix: the budget clamps to the SMALLEST declared provider context,
lower-only, cloud gate removed, exceptions narrowed + logged.
"""

from __future__ import annotations

from maxim.agents.llm_worker import LLMWorker


class _RouterStub:
    def __init__(self, provider_ctxs: dict[str, object]) -> None:
        self._provider_ctxs = provider_ctxs

    def get_provider_configs(self):
        return {name: ({"n_ctx": ctx} if ctx is not None else {}) for name, ctx in self._provider_ctxs.items()}

    def cloud_allowed(self) -> bool:
        # Review fold: the pre-fix code gated on cloud_allowed(); without
        # this method the old max()-raise path could never execute against
        # the stub, so test_clamp_is_lower_only pinned the new invariant
        # but could not DISCRIMINATE the old raise behavior. With it, the
        # lower-only test is an honest before/after witness (fails on
        # pre-fix code, which raises 16384 → 200000).
        return True


def _make_worker(llm, n_ctx: int) -> LLMWorker:
    return LLMWorker(llm=llm, n_ctx=n_ctx)


class TestNCtxClamp:
    def test_mixed_lane_clamps_to_smallest_declared(self):
        """local 16k + cloud 200k must budget at 16k — the pre-fix max()
        budgeted 200k and the local server 500'd every oversize prompt."""
        worker = _make_worker(_RouterStub({"local": 16384, "cloud": 200000}), n_ctx=32768)
        assert worker._n_ctx == 16384

    def test_local_only_lane_clamps_too(self):
        """The pre-fix cloud gate meant local-only runs NEVER clamped."""
        worker = _make_worker(_RouterStub({"local": 8192}), n_ctx=32768)
        assert worker._n_ctx == 8192

    def test_clamp_is_lower_only(self):
        """A provider declaring MORE context than the constructor budget must
        not raise the budget (the constructor value reflects the resolved
        run config; raising it re-introduces the overflow the clamp kills)."""
        worker = _make_worker(_RouterStub({"cloud": 200000}), n_ctx=16384)
        assert worker._n_ctx == 16384

    def test_undeclared_n_ctx_is_skipped_not_treated_as_small(self):
        worker = _make_worker(_RouterStub({"mystery": None, "local": 16384}), n_ctx=32768)
        assert worker._n_ctx == 16384

    def test_no_declared_contexts_keeps_constructor_value(self):
        worker = _make_worker(_RouterStub({"mystery": None}), n_ctx=4096)
        assert worker._n_ctx == 4096

    def test_router_without_provider_configs_keeps_constructor_value(self):
        class _Bare:
            pass

        worker = _make_worker(_Bare(), n_ctx=4096)
        assert worker._n_ctx == 4096

    def test_malformed_declared_value_logs_and_keeps_budget(self, caplog):
        """The pre-fix bare `except: pass` swallowed everything silently."""
        import logging

        with caplog.at_level(logging.WARNING):
            worker = _make_worker(_RouterStub({"bad": "not-a-number"}), n_ctx=4096)
        assert worker._n_ctx == 4096
        assert any("n_ctx scan failed" in r.message for r in caplog.records)

    def test_nonpositive_declared_value_cannot_poison_the_budget(self):
        """Review fold: with min(), a bogus n_ctx of -5 or 0 would clamp the
        WHOLE budget to garbage (the pre-fix max() made bogus-small harmless;
        min() inverts the blast radius). Nonpositive declarations are skipped
        as malformed, not believed."""
        worker = _make_worker(_RouterStub({"bogus": -5, "zero": 0, "local": 16384}), n_ctx=32768)
        assert worker._n_ctx == 16384

    def test_non_dict_provider_cfg_warns_and_keeps_budget(self, caplog):
        """Review fold: a router returning a non-dict cfg raises
        AttributeError — the scan must degrade to a warning, not crash the
        constructor."""
        import logging

        class _WeirdRouter:
            def get_provider_configs(self):
                return {"weird": "not-a-dict"}

        with caplog.at_level(logging.WARNING):
            worker = _make_worker(_WeirdRouter(), n_ctx=4096)
        assert worker._n_ctx == 4096
        assert any("n_ctx scan failed" in r.message for r in caplog.records)

    def test_bool_n_ctx_is_skipped_not_believed_as_one(self):
        """Review fold NIT: n_ctx: true would int() to 1 and clamp the whole
        budget to a single token."""
        worker = _make_worker(_RouterStub({"weird": True, "local": 16384}), n_ctx=32768)
        assert worker._n_ctx == 16384

    def test_provider_scan_raising_arbitrary_error_degrades_to_warning(self, caplog):
        """Review fold: the scan now runs for EVERY router exposing
        get_provider_configs — a third-party backend raising RuntimeError
        must not crash the constructor."""
        import logging

        class _ExplodingRouter:
            def get_provider_configs(self):
                raise RuntimeError("backend exploded")

        with caplog.at_level(logging.WARNING):
            worker = _make_worker(_ExplodingRouter(), n_ctx=4096)
        assert worker._n_ctx == 4096
        assert any("n_ctx scan failed" in r.message for r in caplog.records)
