"""Regression guard for n_ctx leg 3 (auto-spawn served-context propagation).

THE GAP THIS PINS (CLAUDE.md "Running simulations" known-bug, leg 3): the
auto-spawn path launched llama-cpp-server with a RESOLVED n_ctx but never
propagated it into the lane's provider config — only the hot-swap path did
(``router.update_provider_n_ctx``). A provider whose config declared the
profile default while the server spawned smaller budgeted too large →
oversize prompt → llama-cpp HTTP 500 → provider cooldown → every action
``_llm_unavailable`` (the Exp 44 blocker class).

The fix: ``LaneConfig.served_n_ctx`` — set ONLY by code that knows what the
server actually serves (auto-spawn fresh spawn stamps its launch value;
singleton-reuse stamps the owned spawner's value; unknown external servers
stay unset). ``_build_remote_backend`` threads it into the lane provider
entry (the llm_worker min-clamp input) and the lane llm_config's ``n_ctx``
(the budgeter's direct belief). A remote peer lane's LOCAL tier estimate
(``LaneConfig.n_ctx``) must NEVER be used as a stand-in — it describes this
machine's hardware, not the remote server.
"""

from __future__ import annotations

from unittest import mock

import pytest

from maxim.runtime.worker_pool import LaneConfig


class TestLaneConfigField:
    def test_served_n_ctx_defaults_to_none(self):
        cfg = LaneConfig(name="large", max_workers=1)
        assert cfg.served_n_ctx is None

    def test_served_n_ctx_is_independent_of_tier_estimate(self):
        cfg = LaneConfig(name="large", max_workers=1, n_ctx=32768, served_n_ctx=8192)
        assert cfg.n_ctx == 32768
        assert cfg.served_n_ctx == 8192


def _build_remote(cfg: LaneConfig):
    """Run LaneBackendManager._build_remote_backend far enough to capture the
    llm_config it constructs (the router build is mocked out)."""
    from maxim.runtime import lane_backends as lb

    captured: dict = {}

    class _FakeRouter:
        def __init__(self, llm_config):
            captured["cfg"] = llm_config

    mgr = lb.LaneBackendManager.__new__(lb.LaneBackendManager)
    mgr._peer_owned = False
    mgr._logger = None
    with mock.patch.object(lb, "LLMRouter", _FakeRouter, create=True):
        with mock.patch("maxim.models.language.router.LLMRouter", _FakeRouter):
            mgr._build_remote_backend(cfg)
    return captured.get("cfg")


class TestServedNctxThreading:
    def _lane(self, **kw) -> LaneConfig:
        base = dict(
            name="large",
            max_workers=1,
            model_profile="mistral-7b",
            remote_url="http://127.0.0.1:8100/v1",
            remote_model="mistral-7b",
            remote_api_key="k",
        )
        base.update(kw)
        return LaneConfig(**base)

    def test_served_n_ctx_reaches_provider_entry_and_config(self):
        cfg = self._lane(served_n_ctx=8192, n_ctx=32768)
        remote_cfg = _build_remote(cfg)
        assert remote_cfg is not None, "router construction was not reached"
        entry = remote_cfg.providers["lane-large"]
        assert entry.get("n_ctx") == 8192, (
            "the served context must be declared on the lane provider entry — "
            "the llm_worker min-clamp reads it; without it the budgeter "
            "believes the profile default and oversizes prompts (leg 3)."
        )
        assert remote_cfg.n_ctx == 8192, (
            "router.n_ctx (the budgeter's direct belief) must track the SERVED "
            "window, not the profile default, when the served value is known."
        )

    def test_unknown_served_n_ctx_declares_nothing(self):
        """A remote peer lane: LOCAL tier estimate must NOT masquerade as the
        remote server's window."""
        cfg = self._lane(served_n_ctx=None, n_ctx=4096)
        remote_cfg = _build_remote(cfg)
        assert remote_cfg is not None
        entry = remote_cfg.providers["lane-large"]
        assert "n_ctx" not in entry, (
            "cfg.n_ctx is a LOCAL hardware estimate (detect_tiers) — declaring "
            "it for a remote server silently distorts the min-clamp."
        )

    def test_nonpositive_served_n_ctx_ignored(self):
        cfg = self._lane(served_n_ctx=0)
        remote_cfg = _build_remote(cfg)
        assert remote_cfg is not None
        assert "n_ctx" not in remote_cfg.providers["lane-large"]


class TestAutoSpawnStamp:
    def test_fresh_spawn_rewrite_carries_resolved_n_ctx(self):
        """The fresh-spawn lane rewrite must stamp served_n_ctx with the value
        the server was launched with (source-level pin: the dataclasses.replace
        call in _maybe_auto_spawn_server passes served_n_ctx=resolved_n_ctx)."""
        import inspect

        from maxim.runtime import lane_backends as lb

        src = inspect.getsource(lb._maybe_auto_spawn_server)
        assert "served_n_ctx=resolved_n_ctx" in src, (
            "_maybe_auto_spawn_server's fresh-spawn rewrite no longer stamps "
            "served_n_ctx=resolved_n_ctx — leg 3 regresses to profile-default "
            "budgeting on first spawn."
        )
        assert "served_n_ctx=_reused_served_n_ctx or infer_cfg.served_n_ctx" in src, (
            "the singleton-reuse rewrite must stamp the owned spawner's n_ctx "
            "(and leave unknown external servers unset)."
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
