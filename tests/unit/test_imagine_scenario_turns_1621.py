"""#1621: ``maxim.imagine(scenario=)`` delivers the scenario's ``cli_input`` lines, or refuses the scenario.

``imagine`` hand-parsed ``percepts[]`` reading ``p.get("text", "")``, but scenario YAMLs carry their lines in
``cli_input``, so every scripted turn reached the bridge as an empty string, silently, and the run still produced a
report. The verb now loads turns through the one scripted-turn loader, ``experiment.load_campaign_turns``, STRICTLY
(owner decision 2026-11-21): every percept must be a ``cli_input`` line, and anything else is refused before any run
starts (delivering pain/vision percepts here is #1393's). Research mode's campaign goes through the same strict load
and raises instead of falling back to a free-form sim. The lenient mode (``run_campaign``) skips and names every
percept that is not a line, by the same test.
"""

from __future__ import annotations

import logging
import textwrap
from pathlib import Path
from typing import Any

import pytest

from maxim.simulation.experiment import load_campaign_turns

LINES = ["remember the red key is under the mat", "what did I tell you about the key?"]


class _Stop(Exception):
    """Ends a run at the sim boundary."""


@pytest.fixture
def no_model_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("maxim.api._resolve_model", lambda model: model)
    monkeypatch.setattr("maxim.api._validate_model", lambda model: None)


def _scenario(tmp_path: Path, body: str) -> str:
    path = tmp_path / "s.yaml"
    path.write_text(textwrap.dedent(body))
    return str(path)


def _cli_scenario(tmp_path: Path) -> str:
    return _scenario(
        tmp_path,
        f"""\
        name: s1621
        percepts:
          - at: 0
            source: cli
            cli_input: "{LINES[0]}"
            metadata: {{phase: encode}}
          - at: 1
            source: cli
            cli_input: "{LINES[1]}"
            salience: 0.5
            metadata: {{phase: recall}}
        """,
    )


PAIN_PERCEPT = "  - at: 1\n    source: proprioception\n    content: pain_signal\n"


class _Bridge:
    def __init__(self) -> None:
        self.received: list[str] = []

    def send_and_wait(self, text: str, **_kw: Any) -> dict[str, Any]:
        self.received.append(text)
        return {"actions": [], "blocked": [], "response": ""}

    def finish(self) -> None:
        pass


class _Records(logging.Handler):
    """The maxim loggers do not propagate to caplog, so the test listens on the logger itself."""

    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def _warnings_while(fn: Any) -> list[str]:
    records = _Records()
    experiment_logger = logging.getLogger("maxim.simulation.experiment")
    experiment_logger.addHandler(records)
    try:
        fn()
    finally:
        experiment_logger.removeHandler(records)
    return records.messages


def test_imagine_delivers_the_scenarios_cli_input_lines_to_the_bridge(
    no_model_checks: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from maxim.api import imagine
    from maxim.simulation.campaign_runner import run_precampaign_turns

    path = _cli_scenario(tmp_path)
    seen: dict[str, Any] = {}

    def _sim(**kwargs: Any) -> Any:
        seen["turns"] = kwargs["pre_campaign_turns"]
        raise _Stop

    monkeypatch.setattr("maxim.simulation.orchestrator.start_simulation_mode", _sim)
    monkeypatch.setattr("maxim.simulation.campaign_runner.time.sleep", lambda _s: None)
    monkeypatch.setattr("maxim.utils.paths.sim_reports", lambda: tmp_path)
    with pytest.raises(_Stop):
        imagine("g", model="m", scenario=path)

    assert seen["turns"] == load_campaign_turns(path, strict=True)  # the one loader's dicts (provenance hashes them)
    bridge = _Bridge()
    run_precampaign_turns(turns=seen["turns"], bridge=bridge, introspector=None)  # what the sim does with them
    assert bridge.received == LINES
    assert [t["phase"] for t in seen["turns"]] == ["encode", "recall"]
    assert seen["turns"][1]["salience"] == 0.5


@pytest.mark.parametrize(
    ("body", "names"),
    [
        ("name: empty\npercepts: []\n", "non-empty 'percepts'"),
        ("name: none\n", "non-empty 'percepts'"),
        ("", "non-empty 'percepts'"),
        ("name: pain_only\npercepts:\n" + PAIN_PERCEPT, "percept 0 has no cli_input line"),
        (
            "name: mixed\npercepts:\n  - at: 0\n    cli_input: hello\n" + PAIN_PERCEPT,
            "percept 1 has no cli_input line",
        ),
        ("name: text_key\npercepts:\n  - at: 0\n    text: the old key imagine read\n", "percept 0 has no cli_input"),
        ("name: phase_key\npercepts:\n  - cli_input: hi\n    phase: encode\n", "percept 0 carries a percept-level"),
        ("name: blank\npercepts:\n  - cli_input: '   '\n", "percept 0 has no cli_input line"),
    ],
    ids=["empty_list", "no_percepts", "empty_file", "pain_only", "mixed", "text_key", "phase_key", "blank_line"],
)
def test_imagine_refuses_a_scenario_that_is_not_all_cli_input_lines_before_any_run(
    no_model_checks: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, body: str, names: str
) -> None:
    from maxim.api import imagine

    def _never(**_kw: Any) -> Any:
        raise AssertionError("no run may start for a scenario imagine cannot deliver")

    monkeypatch.setattr("maxim.simulation.orchestrator.start_simulation_mode", _never)
    with pytest.raises(ValueError, match="imagine\\(scenario=\\) delivers only cli_input lines") as info:
        imagine("g", model="m", scenario=_scenario(tmp_path, body))
    assert names in str(info.value)
    assert "#1393" in str(info.value) and "maxim --sim <scenario>" in str(info.value)


def test_the_repo_pain_scenario_is_refused_by_imagine(no_model_checks: None, monkeypatch: pytest.MonkeyPatch) -> None:
    from maxim.api import imagine

    monkeypatch.setattr("maxim.simulation.orchestrator.start_simulation_mode", lambda **_k: pytest.fail("ran"))
    path = Path(__file__).resolve().parents[2] / "scenarios" / "malware_with_pain.yaml"
    with pytest.raises(ValueError, match="percept 1 has no cli_input line"):
        imagine("g", model="m", scenario=str(path))


# --- the loader's malformed-file branches (both modes) -----------------------------------------------------------


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize(
    ("body", "match"),
    [
        ("- cli_input: a\n- cli_input: b\n", "not a mapping"),
        ("percepts: abc\n", "'percepts' is not a list"),
        ("percepts: [unclosed\n", "YAML"),
        ("expectations:\n  survived: true\npercepts:\n  - cli_input: a\n", "'expectations' is not a list"),
        ("expectations: [just a string]\npercepts:\n  - cli_input: a\n", "expectation 0 is not a mapping"),
        ("expectations:\n  - type: x\n    params: 3\npercepts:\n  - cli_input: a\n", "params is not a mapping"),
        ("benchmark:\n  scoring: [a]\npercepts:\n  - cli_input: a\n", "benchmark.scoring is not a mapping"),
        ("estimated_duration_s: [1]\npercepts:\n  - cli_input: a\n", "estimated_duration_s"),
    ],
    ids=[
        "list_top_level",
        "percepts_string",
        "malformed_yaml",
        "dm_expectations_mapping",
        "expectation_string",
        "params_not_mapping",
        "scoring_not_mapping",
        "duration_not_number",
    ],
)
def test_a_malformed_file_raises_value_error_in_both_modes(tmp_path: Path, body: str, match: str, strict: bool) -> None:
    with pytest.raises(ValueError, match=match):
        load_campaign_turns(_scenario(tmp_path, body), strict=strict)


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize(
    "campaign", sorted((Path(__file__).resolve().parents[2] / "scenarios" / "campaigns").glob("*.yaml")), ids=str
)
def test_every_dm_campaign_is_a_typed_refusal_naming_its_path(campaign: Path, strict: bool) -> None:
    """A DM campaign (``acts``/``encounters``, ``expectations:`` as a mapping) is not scripted turns: a
    ``ValueError`` naming the file, never an untyped ``AttributeError`` (#1621 delta review)."""
    with pytest.raises(ValueError, match=str(campaign.name)):
        load_campaign_turns(str(campaign), strict=strict)


def test_null_expectations_read_as_none(tmp_path: Path) -> None:
    path = _scenario(tmp_path, "expectations: null\npercepts:\n  - cli_input: a\n")
    assert [t["text"] for t in load_campaign_turns(path, strict=True)] == ["a"]


# --- the lenient mode: skip and name, by the strict mode's test ---------------------------------------------------


def test_lenient_mode_skips_and_names_every_percept_that_is_not_a_line(tmp_path: Path) -> None:
    from maxim.simulation.campaign_runner import run_precampaign_turns

    path = _scenario(
        tmp_path,
        """\
        percepts:
          - cli_input: first
          - just a string
          - cli_input: "   "
          - cli_input: 42
          - source: proprioception
            content: pain_signal
          - cli_input: bad metadata
            metadata: []
          - cli_input: zero metadata
            metadata: 0
          - cli_input: old phase
            phase: encode
          - cli_input: last
            metadata: null
        """,
    )
    turns: list[dict[str, Any]] = []
    messages = _warnings_while(lambda: turns.extend(load_campaign_turns(path)))
    assert [t["text"] for t in turns] == ["first", "old phase", "last"]
    assert turns[1]["phase"] == ""  # the percept-level phase is not read ...
    assert any("percept 7's percept-level 'phase' is ignored" in m for m in messages)  # ... and says so
    assert turns[2]["metadata"] == {} and turns[2]["phase"] == ""  # only null reads as no metadata
    (warning,) = [m for m in messages if "not delivered as turns" in m]
    assert "percept 1 (is not a mapping)" in warning
    for index in (2, 3, 4):
        assert f"{index} (has no cli_input line" in warning
    for index in (5, 6):  # metadata [] and 0: skipped and named, not silently {}
        assert f"{index} (has a metadata that is not a mapping)" in warning
    bridge = _Bridge()
    run_precampaign_turns(turns=turns, bridge=bridge, introspector=None)  # 42 no longer reaches len(text)
    assert bridge.received == ["first", "old phase", "last"]


# --- research mode ------------------------------------------------------------------------------------------------


def test_research_campaign_turns_come_from_the_same_strict_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from maxim.simulation.research_orchestrator import _load_campaign_turns

    monkeypatch.setattr("maxim.simulation.sim_logger.display_status", lambda _m: None)
    path = _cli_scenario(tmp_path)
    assert _load_campaign_turns(path) == load_campaign_turns(path, strict=True)


@pytest.mark.parametrize(
    "body", ["name: none\n", "name: pain_only\npercepts:\n" + PAIN_PERCEPT], ids=["no_lines", "pain_only"]
)
def test_research_refuses_a_campaign_with_no_deliverable_line_before_any_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: str
) -> None:
    from maxim.simulation import research_orchestrator

    monkeypatch.setattr(research_orchestrator, "_build_research_llm", lambda *_a: pytest.fail("built an LLM"))
    monkeypatch.setattr("maxim.simulation.orchestrator.start_simulation_mode", lambda **_k: pytest.fail("ran"))
    monkeypatch.setattr("maxim.utils.paths.sim_reports", lambda: tmp_path / "reports")
    from maxim.exceptions import CampaignUnloadable

    with pytest.raises(CampaignUnloadable) as info:
        research_orchestrator.start_research_mode(goal="g", campaign=_scenario(tmp_path, body))
    assert isinstance(info.value, ValueError)  # maxim.research() callers catching ValueError keep working
    assert info.value.campaign.endswith("s.yaml") and info.value.reason
    assert not (tmp_path / "reports").exists()  # no session dir for a refused campaign


def test_research_refuses_a_missing_campaign(tmp_path: Path) -> None:
    from maxim.simulation.research_orchestrator import _load_campaign_turns

    from maxim.exceptions import CampaignUnloadable

    with pytest.raises(CampaignUnloadable, match="not found") as info:
        _load_campaign_turns(str(tmp_path / "nope.yaml"))
    assert isinstance(info.value.__cause__, FileNotFoundError)


def test_the_cli_refuses_an_unloadable_research_campaign_with_exit_2_not_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--research --campaign`` reaches ``start_research_mode``; its typed refusal is rendered by
    ``cli._main_guarded`` (``user_warn`` + exit 2), not a traceback (#1621 delta review)."""
    from maxim import cli
    from maxim.simulation import research_orchestrator

    campaign = str(Path(__file__).resolve().parents[2] / "scenarios" / "campaigns" / "heist_v1.yaml")
    monkeypatch.setattr(research_orchestrator, "_build_research_llm", lambda *_a: pytest.fail("built an LLM"))

    def _impl(argv: Any = None) -> int:
        return research_orchestrator.start_research_mode(goal="g", campaign=campaign)

    warned: list[str] = []
    monkeypatch.setattr(cli, "_main_impl", _impl)
    monkeypatch.setattr("maxim.utils.logging.user_warn", lambda message, **_k: warned.append(message))
    assert cli._main_guarded([]) == 2
    assert len(warned) == 1 and "heist_v1.yaml" in warned[0] and "cli_input" in warned[0]
