"""``console.agent_id`` — which ``agents/<id>/`` home ``maxim serve`` fronts.

Pins the Exec B1 property (recall reads the SAME home the handle writes) for
the configurable id, lazy resolution (so ``build_app()`` embedders see the
config, not only ``run_serve``), and loud validation at serve start instead
of a 500 on the first Talk request. Skips cleanly without the console extra.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="requires the `console` extra (fastapi/uvicorn)")

from maxim.console import server as srv  # noqa: E402
from maxim.runtime.config_loader import ConfigurationError, coerce_agent_id  # noqa: E402

_TOKEN = "mxc_" + "t" * 43
_AUTH = {"Authorization": f"Bearer {_TOKEN}"}


@pytest.fixture(autouse=True)
def _fresh_handle_state(monkeypatch, tmp_path):
    monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setattr(srv, "_handle", None)
    yield
    monkeypatch.setattr(srv, "_handle", None)


class _FakeHandle:
    built: list[str] = []

    def __init__(self, *, agent_id: str):
        self.agent_id = agent_id
        type(self).built.append(agent_id)


@pytest.fixture()
def fake_handle_class(monkeypatch):
    import maxim.console.handle as handle_mod

    _FakeHandle.built = []
    monkeypatch.setattr(handle_mod, "MaximHandle", _FakeHandle)
    return _FakeHandle


class TestResolution:
    def test_default_is_console_agent(self):
        assert srv._console_agent_id() == "console_agent"

    def test_env_overrides(self, monkeypatch):
        monkeypatch.setenv("MAXIM_CONSOLE_AGENT_ID", "knight_seed")
        assert srv._console_agent_id() == "knight_seed"

    def test_blank_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("MAXIM_CONSOLE_AGENT_ID", "   ")
        assert srv._console_agent_id() == "console_agent"

    def test_handle_is_built_with_the_configured_id(self, monkeypatch, fake_handle_class):
        monkeypatch.setenv("MAXIM_CONSOLE_AGENT_ID", "knight_seed")
        h = srv._get_handle()
        assert h.agent_id == "knight_seed" and fake_handle_class.built == ["knight_seed"]
        assert srv._get_handle() is h  # still one handle per process

    def test_recall_reads_the_same_home_before_any_handle_exists(self, monkeypatch):
        # Exec B1: the MemoryView must never read a different home than the
        # handle writes — including before the first Talk builds the handle.
        monkeypatch.setenv("MAXIM_CONSOLE_AGENT_ID", "knight_seed")
        seen: list[str] = []
        from maxim.integration.recall import CuratedRecall

        def _recall(*, agent_id):
            seen.append(agent_id)
            return CuratedRecall(name="", player_model=[], story_memories=[], preferences=[])

        monkeypatch.setattr("maxim.recall", _recall)
        from fastapi.testclient import TestClient

        app = srv.build_app(None, auth_token=_TOKEN)
        r = TestClient(app).get("/api/recall", headers=_AUTH)
        assert r.status_code in (200, 501)
        assert seen == ["knight_seed"]


class TestValidation:
    """The validator is the config loader's (`coerce_agent_id`), so
    `maxim config set`, the env form and serve-start resolution all refuse
    the same values."""

    @pytest.mark.parametrize("bad", ["sim_aut", "../escape", "a b", "with/slash", "dot.dot", "", "a\nb"])
    def test_bad_ids_are_rejected(self, bad):
        with pytest.raises(ConfigurationError):
            coerce_agent_id(bad, "console.agent_id")

    @pytest.mark.parametrize("good", ["console_agent", "knight-seed", "Exp53_seed42", "a"])
    def test_good_ids_pass(self, good):
        assert coerce_agent_id(good, "console.agent_id") == good

    def test_env_form_is_validated_at_resolution(self, monkeypatch):
        monkeypatch.setenv("MAXIM_CONSOLE_AGENT_ID", "../escape")
        with pytest.raises(ConfigurationError):
            srv._console_agent_id()

    def test_run_serve_fails_at_start_not_on_first_talk(self, monkeypatch, capsys):
        monkeypatch.setenv("MAXIM_CONSOLE_AGENT_ID", "sim_aut")
        import uvicorn

        monkeypatch.setattr(uvicorn, "run", lambda *a, **k: pytest.fail("uvicorn.run must not be reached"))
        rc = srv.run_serve([])
        assert rc == 2
        assert "sim_aut" in capsys.readouterr().out
