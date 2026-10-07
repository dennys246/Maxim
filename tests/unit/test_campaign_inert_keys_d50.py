"""D50: campaign-YAML keys that nothing reads are not accepted silently (owner decision 2026-10-07).

``party_mode`` loads onto ``CampaignDef`` but there is no party runtime, and ``choice_resolution`` has no
reader at all. A campaign author who sets ``party_mode: true`` gets no party and, until now, no word. The
tests load the shipped ``broken_database_v1`` campaign with only those keys changed.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml

from maxim.simulation.dm_schema import load_campaign

SHIPPED = Path(__file__).resolve().parents[2] / "scenarios" / "campaigns" / "broken_database_v1.yaml"


def _campaign(tmp_path: Path, **campaign_keys) -> Path:
    raw = yaml.safe_load(SHIPPED.read_text())
    raw["campaign"].update(campaign_keys)
    path = tmp_path / "campaign.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_party_mode_true_warns_that_no_party_runtime_exists(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="maxim.simulation.dm_schema"):
        load_campaign(_campaign(tmp_path, party_mode=True))
    assert any("party_mode" in r.getMessage() for r in caplog.records)


def test_choice_resolution_is_not_a_campaign_field(tmp_path):
    assert not hasattr(load_campaign(_campaign(tmp_path)), "choice_resolution")


def test_a_non_default_choice_resolution_warns_as_inert(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="maxim.simulation.dm_schema"):
        load_campaign(_campaign(tmp_path, choice_resolution="vote"))
    assert any("choice_resolution" in r.getMessage() for r in caplog.records)


def test_the_shipped_campaign_loads_without_an_inert_key_warning(caplog):
    """It sets ``party_mode: false`` and ``choice_resolution: pc_decides``, the defaults: no warning."""
    with caplog.at_level(logging.WARNING, logger="maxim.simulation.dm_schema"):
        load_campaign(SHIPPED)
    assert not [r for r in caplog.records if "party_mode" in r.getMessage() or "choice_resolution" in r.getMessage()]


def test_the_api_party_mode_override_warns_too(monkeypatch):
    """``maxim.campaign(party_mode=True)`` goes around the YAML load, so it warns through the same helper.
    (``api.campaign`` reconfigures logging, so this spies on the helper; the load tests pin its log line.)"""
    import maxim.simulation.dm_schema as dm_schema
    import maxim.simulation.orchestrator as orchestrator
    from maxim import api

    class _Stop(Exception):
        pass

    def _stop(*_a, **_k):
        raise _Stop

    warned: list[str] = []
    monkeypatch.setattr(dm_schema, "warn_party_mode_unsupported", warned.append)
    monkeypatch.setattr(orchestrator, "start_simulation_mode", _stop)
    with pytest.raises(_Stop):
        api.campaign(str(SHIPPED), party_mode=True)
    assert warned == ["maxim.campaign(party_mode=True)"]


def test_the_api_override_does_not_warn_twice_when_the_yaml_already_set_it(monkeypatch, tmp_path):
    import maxim.simulation.dm_schema as dm_schema
    import maxim.simulation.orchestrator as orchestrator
    from maxim import api

    class _Stop(Exception):
        pass

    def _stop(*_a, **_k):
        raise _Stop

    warned: list[str] = []
    monkeypatch.setattr(dm_schema, "warn_party_mode_unsupported", warned.append)
    monkeypatch.setattr(orchestrator, "start_simulation_mode", _stop)
    with pytest.raises(_Stop):
        api.campaign(str(_campaign(tmp_path, party_mode=True)), party_mode=True)
    assert warned == ["campaign campaign.yaml"]  # the load's warning only
