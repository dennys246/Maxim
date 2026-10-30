"""``cli._handle_foundry`` (the foundry dispatch, moved out of ``_main_impl`` for rule T in #1315 step 0a) and
``cli.main``'s rendering of a refused ``profiles.yml``, in-process.

The end-to-end refusal runs in a fresh process in ``test_profile_readers_v9.py``; these pin the two code paths
directly: the foundry run's wiring (with and without an LLM profile, including the template fallback when the LLM
cannot be built) and ``main()`` turning ``UserProfilesInvalid`` into exit 2 with the file and the fix.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from maxim import cli


class _Score(SimpleNamespace):
    pass


class _FakeRunner:
    """Stands in for ``maxim.simulation.foundry.FoundryRunner``: records its arguments, returns a fixed result."""

    calls: list[dict[str, Any]] = []

    def __init__(self, **kwargs: Any) -> None:
        _FakeRunner.calls.append(kwargs)

    def run(self, *, count: int) -> Any:
        _FakeRunner.calls[-1]["count"] = count
        return SimpleNamespace(
            output_dir="/tmp/foundry-out",
            generated=3,
            validated=2,
            tested=2,
            promoted=[_Score(candidate_name="lantern", total_score=0.91)],
            review=[],
            rejected=[object()],
        )


@pytest.fixture
def fake_runner(monkeypatch: pytest.MonkeyPatch) -> type[_FakeRunner]:
    import maxim.simulation.foundry as foundry

    _FakeRunner.calls = []
    monkeypatch.setattr(foundry, "FoundryRunner", _FakeRunner)
    return _FakeRunner


def test_foundry_without_an_llm_profile_runs_the_templates(fake_runner: type[_FakeRunner], capsys) -> None:
    args = SimpleNamespace(foundry_genre="scifi", foundry_category="props", foundry_dry_run=True, foundry_count=4)
    assert cli._handle_foundry(args, "space station") == 0
    (call,) = fake_runner.calls
    assert call == {
        "theme": "space station",
        "genre": "scifi",
        "category": "props",
        "llm_router": None,
        "dry_run": True,
        "count": 4,
    }
    out = capsys.readouterr().out
    assert "Foundry complete: /tmp/foundry-out" in out
    assert "Promoted: 1 | Review: 0 | Rejected: 1" in out
    assert "lantern (0.91)" in out


def test_foundry_falls_back_to_templates_when_its_llm_cannot_be_built(
    fake_runner: type[_FakeRunner], monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    from maxim.models.language import config

    def _boom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("no such model")

    monkeypatch.setattr(config, "load_llm_config", _boom)
    args = SimpleNamespace(language_model="some-profile", foundry_count=None)
    assert cli._handle_foundry(args, "forest") == 0
    (call,) = fake_runner.calls
    assert call["llm_router"] is None and call["genre"] == "fantasy" and call["count"] == 10
    assert "LLM init failed (no such model)" in capsys.readouterr().out


def test_foundry_uses_its_llm_once_it_is_ready(fake_runner: type[_FakeRunner], monkeypatch: pytest.MonkeyPatch) -> None:
    from maxim.models.language import config, router

    class _Router:
        def __init__(self, cfg: Any) -> None:
            self.cfg = cfg

        def warmup(self) -> None:
            pass

        def wait_ready(self, timeout: float) -> bool:
            return True

    monkeypatch.setattr(config, "load_llm_config", lambda profile_override: f"cfg:{profile_override}")
    monkeypatch.setattr(router, "LLMRouter", _Router)
    assert cli._handle_foundry(SimpleNamespace(language_model="p1"), "desert") == 0
    (call,) = fake_runner.calls
    assert isinstance(call["llm_router"], _Router) and call["llm_router"].cfg == "cfg:p1"


def test_main_renders_a_refused_profiles_file_as_exit_2_with_the_fix(monkeypatch: pytest.MonkeyPatch) -> None:
    from maxim.exceptions import UserProfilesInvalid

    seen: dict[str, Any] = {}

    def _raise(_argv: Any) -> int:
        raise UserProfilesInvalid("cannot load /x/profiles.yml: bad yaml")

    def _warn(message: str, **kwargs: Any) -> None:
        seen["message"], seen["kwargs"] = message, kwargs

    monkeypatch.setattr(cli, "_main_impl", _raise)
    monkeypatch.setattr("maxim.utils.logging.user_warn", _warn)
    assert cli.main(["--mode", "agentic"]) == 2
    assert "UserProfilesInvalid" in seen["message"] and "/x/profiles.yml" in seen["message"]
    assert "maxim doctor" in seen["message"]
    assert seen["kwargs"].get("event") == "user_profiles_invalid"
