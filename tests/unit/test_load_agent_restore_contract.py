"""D17 guard — ``maxim.load.agent()`` restores fully or fails loudly.

Two defects, both silent, both fixed together:

* ``_create_atl`` constructed the ATL with a persistence path but never read
  it. The restore was left to ``MemoryHub.on_session_start()``, which
  ``load.agent()`` returns without calling — so the documented "restores
  Hippocampus, NAc, ATL" promise was false for ATL.
* Corrupt Hippocampus and SCN files were caught by ``except Exception: pass``
  and replaced with fresh state. For an API named ``load``, silently handing
  back an empty agent risks overwriting recoverable data on the next save.

Every test below fails against the pre-fix code.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import maxim.load as load_mod
from maxim.exceptions import MemoryCorruptionError
from maxim.memory.atl import ATL, ATLConfig
from maxim.memory.semantic_types import SemanticMemory


@pytest.fixture
def agent_home(tmp_path: Path) -> Path:
    """A persisted-agent directory containing a real, loadable ATL."""
    d = tmp_path / "scout"
    d.mkdir()
    atl = ATL(ATLConfig(persistence_path=str(d / "atl.json")))
    atl.store(
        SemanticMemory(
            id="d17-lantern",
            timestamp=0.0,
            name="lantern",
            definition="a light source",
            category="object",
        )
    )
    atl.save()
    assert (d / "atl.json").exists()
    return d


def _load(tmp_path: Path, **kw):
    return load_mod.agent("scout", base_dir=str(tmp_path), **kw)


def test_atl_is_restored_before_load_agent_returns(agent_home, tmp_path):
    """The core D17 defect — ATL used to load only at a later session start."""
    inst = _load(tmp_path)
    assert inst.atl is not None
    names = {getattr(c, "name", None) for c in inst.atl.recall(limit=50)}
    assert "lantern" in names, "ATL was not restored by load.agent() (D17)"


def test_corrupt_hippocampus_raises_by_default(agent_home, tmp_path):
    (agent_home / "hippocampus.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(MemoryCorruptionError) as ei:
        _load(tmp_path)
    assert "hippocampus" in str(ei.value)
    assert str(agent_home / "hippocampus.json") in str(ei.value)


def test_corrupt_scn_raises_by_default(agent_home, tmp_path):
    (agent_home / "scn.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(MemoryCorruptionError) as ei:
        _load(tmp_path)
    assert "scn" in str(ei.value)


def test_corrupt_atl_raises_by_default(agent_home, tmp_path):
    (agent_home / "atl.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(MemoryCorruptionError) as ei:
        _load(tmp_path)
    assert "atl" in str(ei.value)


def test_every_corrupt_file_is_reported_not_just_the_first(agent_home, tmp_path):
    for f in ("hippocampus.json", "scn.json", "atl.json"):
        (agent_home / f).write_text("{not json", encoding="utf-8")
    with pytest.raises(MemoryCorruptionError) as ei:
        _load(tmp_path)
    reported = {c["subsystem"] for c in ei.value.context["corrupt"]}
    assert {"hippocampus", "scn", "atl"} <= reported, f"only reported {reported}"


def test_error_is_actionable(agent_home, tmp_path):
    (agent_home / "hippocampus.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(MemoryCorruptionError) as ei:
        _load(tmp_path)
    msg = str(ei.value)
    assert 'on_corrupt="fresh"' in msg, "error must name the recovery path"
    assert ei.value.context["agent_id"] == "scout"


def test_fresh_is_an_explicit_opt_in(agent_home, tmp_path):
    (agent_home / "hippocampus.json").write_text("{not json", encoding="utf-8")
    inst = _load(tmp_path, on_corrupt="fresh")
    assert inst is not None


def test_fresh_does_not_delete_the_corrupt_file(agent_home, tmp_path):
    """Choosing 'fresh' must not destroy recoverable bytes."""
    p = agent_home / "hippocampus.json"
    p.write_text("{not json", encoding="utf-8")
    _load(tmp_path, on_corrupt="fresh")
    assert p.read_text(encoding="utf-8") == "{not json"


def test_invalid_on_corrupt_rejected(agent_home, tmp_path):
    with pytest.raises(ValueError, match="on_corrupt"):
        _load(tmp_path, on_corrupt="ignore")


def test_healthy_agent_still_loads_clean(agent_home, tmp_path):
    inst = _load(tmp_path)
    assert inst.agent_id == "scout"


def test_create_path_is_unaffected_by_corrupt_files(agent_home, tmp_path):
    """Blast-radius guard: only the stable load path opts into raising.

    ``_create_memory_hub`` restores SCN for EVERY caller, so a raising default
    would reach far outside ``load.agent()``.
    """
    from maxim.runtime.agent_factory import AgentConfig, AgentFactory

    (agent_home / "scn.json").write_text("{not json", encoding="utf-8")
    (agent_home / "hippocampus.json").write_text("{not json", encoding="utf-8")

    factory = AgentFactory(base_data_dir=str(tmp_path))
    inst = factory.create_agent(AgentConfig(agent_id="scout", persistence_dir=str(agent_home)))
    assert inst is not None, "create.agent() must not inherit load.agent()'s strictness"


def test_corruption_is_logged_even_when_not_raising(agent_home, tmp_path, caplog):
    """The pre-fix code swallowed silently; 'warn' mode must still be loud."""
    (agent_home / "hippocampus.json").write_text("{not json", encoding="utf-8")
    with caplog.at_level("WARNING"):
        _load(tmp_path, on_corrupt="fresh")
    assert any("Corrupt hippocampus" in r.getMessage() for r in caplog.records), "corruption must be logged"


def test_nac_corruption_is_reported(agent_home, tmp_path):
    """load_safe recovers internally and reports via its return value."""
    (agent_home / "nac.json").write_text(json.dumps({"links": "not-a-dict"}), encoding="utf-8")
    with pytest.raises(MemoryCorruptionError) as ei:
        _load(tmp_path)
    assert "nac" in {c["subsystem"] for c in ei.value.context["corrupt"]}
