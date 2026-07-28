"""Guards for the launcher-completion seams (maxim-pulse asks, 2026-07-28).

Covers the four items the pulse repo was waiting on besides talk mode:
  * ``GET /api/campaigns`` — discovery so the picker is a dropdown, not a
    pasted absolute path;
  * ``RunRequest.input`` — "describe an adventure and let Maxim imagine it"
    (generative flavor) instead of a 422;
  * display auto-revert — gives the EVENT seam's ``display``/revert event a
    production producer (``revert_display_to_floor`` had zero callers);
  * ``placement_resolvable()`` — the SETUP read-side, so callers stop
    re-implementing config vocabulary.
"""

from __future__ import annotations

import pytest

from maxim.simulation.sim_logger import (
    DisplayTier,
    agent_escalate_display,
    disable_sim_logging,
    enable_sim_logging,
    get_display_tier,
    get_sim_records,
    maybe_auto_revert_display,
    reset_sim_display_state,
    set_display_tier,
)


@pytest.fixture()
def sim_logging():
    enable_sim_logging(use_color=False)
    try:
        yield
    finally:
        disable_sim_logging()
        reset_sim_display_state()


class TestDisplayAutoRevert:
    def test_escalation_expires_and_emits_revert(self, sim_logging):
        set_display_tier(DisplayTier.CLEAN)
        # hold_s=0 → already expired on the next tick.
        assert agent_escalate_display(DisplayTier.BIO, reason="something happened", hold_s=0.0) is True
        assert get_display_tier() is DisplayTier.BIO
        assert maybe_auto_revert_display() is True
        assert get_display_tier() is DisplayTier.CLEAN
        actions = [r["data"]["action"] for r in get_sim_records() if r["subsystem"] == "DISPLAY"]
        assert actions == ["escalate", "revert"]

    def test_unexpired_escalation_is_held(self, sim_logging):
        set_display_tier(DisplayTier.CLEAN)
        agent_escalate_display(DisplayTier.BIO, hold_s=300.0)
        assert maybe_auto_revert_display() is False
        assert get_display_tier() is DisplayTier.BIO

    def test_tick_is_a_noop_without_escalation(self, sim_logging):
        set_display_tier(DisplayTier.CLEAN)
        assert maybe_auto_revert_display() is False
        assert not [r for r in get_sim_records() if r["subsystem"] == "DISPLAY"]

    def test_agent_loop_tick_wired(self):
        # The producer only exists if the loop actually ticks it.
        import inspect

        from maxim.runtime import agent_loop

        src = inspect.getsource(agent_loop.run_agentic_loop)
        assert "_maybe_auto_revert_display()" in src


class TestPlacementResolvable:
    def test_reports_kind_and_never_raises(self):
        from maxim.runtime.config_writer import placement_resolvable

        ok, kind, detail = placement_resolvable()
        assert isinstance(ok, bool)
        assert kind in {"mesh", "cloud", "local", "none"}
        assert detail  # always explains itself

    def test_unknown_tier_answers_instead_of_raising(self):
        from maxim.runtime.config_writer import placement_resolvable

        ok, kind, detail = placement_resolvable("not_a_tier")
        assert (ok, kind) == (False, "none")
        assert "unknown tier" in detail

    def test_mesh_setup_becomes_resolvable(self, tmp_path, monkeypatch):
        # Round-trip against the SETUP write-side: what apply_mesh_setup
        # writes, placement_resolvable must read back as a mesh placement.
        import maxim.runtime.config_writer as cw

        cfg = tmp_path / "config.json"
        monkeypatch.setattr(cw, "config_path", lambda: cfg)
        monkeypatch.setattr("maxim.runtime.config_loader.config_path", lambda: cfg)
        cw.apply_mesh_setup("https://leader.example.com", "sk-test", path=cfg)
        ok, kind, detail = cw.placement_resolvable()
        assert (ok, kind) == (True, "mesh")
        assert "leader.example.com" in detail
        assert "sk-test" not in detail  # never leak the key into UI text


fastapi = pytest.importorskip("fastapi", reason="requires the `console` extra (fastapi/uvicorn)")

from fastapi.testclient import TestClient  # noqa: E402

from maxim.console.server import build_app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    return TestClient(build_app(None))


class TestCampaignDiscovery:
    def test_lists_campaigns_with_run_ready_paths(self, client):
        body = client.get("/api/campaigns").json()
        assert body["searched"], "discovery must report where it looked"
        for c in body["campaigns"]:
            assert c["path"].endswith((".yaml", ".yml"))
            assert c["source"] in {"user", "repo"}
            assert c["name"]

    def test_user_campaigns_dir_is_searched_first(self, client):
        from maxim.console.server import campaign_search_roots

        roots = campaign_search_roots()
        # (root, source) PAIRS — a parallel label tuple at the call site would
        # silently truncate if a third root were added (review fold).
        assert all(len(entry) == 2 for entry in roots)
        first_root, first_source = roots[0]
        assert str(first_root).endswith("campaigns")
        assert ".maxim" in str(first_root)  # user dir wins over the repo dir
        assert first_source == "user"

    def test_malformed_yaml_still_lists(self, tmp_path):
        # A campaign that cannot be parsed must still appear (by filename) —
        # hiding it would make the picker silently incomplete; the RUN call
        # surfaces the real validation error.
        from maxim.console.server import _campaign_info

        bad = tmp_path / "broken.yaml"
        bad.write_text("campaign: [this is not a mapping")
        info = _campaign_info(bad, "user")
        assert info.name == "broken"
        assert info.goal is None


class TestAdventurePremise:
    def test_premise_or_campaign_but_not_both(self, client):
        both = client.post("/api/run", json={"mode": "adventure", "campaign": "x.yaml", "input": "a heist"})
        assert both.status_code == 422
        assert "EXACTLY ONE" in both.json()["detail"]

    def test_neither_is_rejected(self, client):
        assert client.post("/api/run", json={"mode": "adventure"}).status_code == 422

    def test_blank_premise_is_not_a_premise(self, client):
        # Whitespace-only input must not be mistaken for a premise (it would
        # start a generative run with an empty goal).
        r = client.post("/api/run", json={"mode": "adventure", "input": "   "})
        assert r.status_code == 422

    def test_premise_reaches_the_handle(self, monkeypatch):
        # The whole point of the ask: body.input was read zero times.
        from maxim.console import server as srv

        seen: dict = {}

        class FakeHandle:
            agent_id = "console_agent"

            def play_premise(self, premise, **kw):
                seen["premise"] = premise
                raise RuntimeError("stop here — reaching play_premise is the assertion")

            def play_campaign(self, path, **kw):  # pragma: no cover
                seen["campaign"] = path
                raise AssertionError("premise run must not take the campaign path")

        monkeypatch.setattr(srv, "_get_handle", lambda: FakeHandle())
        with TestClient(build_app(None)) as c:
            r = c.post("/api/run", json={"mode": "adventure", "input": "a heist on a moon base"})
            assert r.status_code == 200
            assert r.json()["status"] == "started"
            srv._active_run["thread"].join(timeout=10)
        assert seen.get("premise") == "a heist on a moon base"
        assert "campaign" not in seen

    def test_handle_rejects_empty_premise(self):
        from maxim.console.handle import MaximHandle

        handle = MaximHandle.__new__(MaximHandle)  # no agent construction
        handle._stopped = False
        with pytest.raises(ValueError, match="non-empty"):
            handle.play_premise("   ")
