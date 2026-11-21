"""#1621: ``maxim.imagine(scenario=)`` delivers the scenario's ``cli_input`` lines, or refuses the scenario.

``imagine`` hand-parsed ``percepts[]`` reading ``p.get("text", "")``, but scenario YAMLs carry their lines in
``cli_input``, so every scripted turn reached the bridge as an empty string, silently, and the run still produced a
report. The verb now loads turns through the one scripted-turn loader, ``experiment.load_campaign_turns``, and a
scenario with no deliverable line is refused before any run starts. The research path's second hand parser
(``research_orchestrator._load_campaign_turns``) goes through the same loader.
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
            source: proprioception
            content: pain_signal
          - at: 2
            source: cli
            cli_input: "{LINES[1]}"
            salience: 0.5
            metadata: {{phase: recall}}
        """,
    )


class _Bridge:
    def __init__(self) -> None:
        self.received: list[str] = []

    def send_and_wait(self, text: str, **_kw: Any) -> dict[str, Any]:
        self.received.append(text)
        return {"actions": [], "blocked": [], "response": ""}

    def finish(self) -> None:
        pass


class _Records(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


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
    records = _Records()
    experiment_logger = logging.getLogger("maxim.simulation.experiment")
    experiment_logger.addHandler(records)
    try:
        with pytest.raises(_Stop):
            imagine("g", model="m", scenario=path)
    finally:
        experiment_logger.removeHandler(records)

    assert seen["turns"] == load_campaign_turns(path)  # the one loader's dicts (the sim's provenance hashes them)
    bridge = _Bridge()
    run_precampaign_turns(turns=seen["turns"], bridge=bridge, introspector=None)  # what the sim does with them
    assert bridge.received == LINES
    assert [t["phase"] for t in seen["turns"]] == ["encode", "recall"]
    assert seen["turns"][1]["salience"] == 0.5
    # the percept that carries no line is named, not dropped silently
    assert any("percept 1 " in m and "no cli_input" in m for m in records.messages)


@pytest.mark.parametrize(
    "body",
    [
        "name: empty\npercepts: []\n",
        "name: none\n",
        "name: vision_only\npercepts:\n  - at: 0\n    source: proprioception\n    content: pain_signal\n",
        "name: text_key\npercepts:\n  - at: 0\n    text: the old key imagine used to read\n",
        "",
    ],
    ids=["empty_list", "no_percepts", "no_cli_line", "text_key_only", "empty_file"],
)
def test_imagine_refuses_a_scenario_with_no_deliverable_line_before_any_run(
    no_model_checks: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, body: str
) -> None:
    from maxim.api import imagine

    def _never(**_kw: Any) -> Any:
        raise AssertionError("no run may start for a scenario with no deliverable line")

    monkeypatch.setattr("maxim.simulation.orchestrator.start_simulation_mode", _never)
    path = _scenario(tmp_path, body)
    with pytest.raises(ValueError, match="no percept carries a cli_input line"):
        imagine("g", model="m", scenario=path)


def test_research_campaign_turns_come_from_the_same_loader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from maxim.simulation.research_orchestrator import _load_campaign_turns

    monkeypatch.setattr("maxim.simulation.sim_logger.display_status", lambda _m: None)
    path = _cli_scenario(tmp_path)
    assert _load_campaign_turns(path) == load_campaign_turns(path)
