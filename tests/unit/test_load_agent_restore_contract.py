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


def test_create_path_is_unaffected_by_corrupt_files(agent_home, tmp_path, caplog):
    """Blast-radius guard: only the stable load path opts into raising.

    Post-D28 (1.2 gate 8(c)) the FRESH create path reads NOTHING — SCN's
    restore is gated on ``auto_load`` like every other subsystem, so a fresh
    agent cannot even notice a corrupt file. The reading create path
    (``auto_load=True``) parses it, warns, and continues — raising remains
    ``load.agent()``'s own opt-in. (The pre-D28 version of this test pinned
    the OPPOSITE: that SCN was parsed regardless of ``auto_load`` — that pin
    was the D28 defect made load-bearing.)
    """
    from maxim.runtime.agent_factory import AgentConfig, AgentFactory

    (agent_home / "scn.json").write_text("{not json", encoding="utf-8")
    (agent_home / "hippocampus.json").write_text("{not json", encoding="utf-8")

    factory = AgentFactory(base_data_dir=str(tmp_path))
    # Fresh path: succeeds AND reads no persisted file at all.
    with caplog.at_level("WARNING"):
        inst = factory.create_agent(AgentConfig(agent_id="scout", persistence_dir=str(agent_home)))
    assert inst is not None, "create.agent() must not inherit load.agent()'s strictness"
    assert not any("Corrupt" in r.getMessage() for r in caplog.records), (
        "the fresh create path read a persisted file — fresh is not fresh (D28)"
    )

    caplog.clear()
    # Reading path: the corrupt files are genuinely parsed (anti-vacuity for
    # the blast-radius claim), warned about, and construction still succeeds.
    with caplog.at_level("WARNING"):
        inst2 = factory.create_agent(AgentConfig(agent_id="scout", persistence_dir=str(agent_home)), auto_load=True)
    assert inst2 is not None, "auto_load create must warn-and-continue, not raise"
    assert any("Corrupt scn" in r.getMessage() for r in caplog.records), (
        "no corrupt file was actually read — the blast-radius assertion is vacuous"
    )


def test_create_is_fresh_scn_does_not_leak_previous_temporal_state(agent_home, tmp_path):
    """D28 (1.2 gate 8(c)): ``create``'s documented contract is "always start
    fresh" — until 2026-09-05, SCN alone leaked the previous session's temporal
    state through it. Fresh must not restore; ``auto_load=True`` must."""
    from maxim.runtime.agent_factory import AgentConfig, AgentFactory
    from maxim.time.scn import SCN
    from maxim.time.temporal_signature import TemporalSignature

    prior = SCN(persistence_path=str(agent_home / "scn.json"))
    prior.register("d28_probe", TemporalSignature.now(), significance=0.9)
    prior.save(str(agent_home / "scn.json"))
    assert (agent_home / "scn.json").exists()

    factory = AgentFactory(base_data_dir=str(tmp_path))
    fresh = factory.create_agent(AgentConfig(agent_id="scout", persistence_dir=str(agent_home)))
    assert fresh.memory_hub.scn.get_signature("d28_probe") is None, (
        "a FRESH agent inherited the previous session's temporal state (D28)"
    )
    # The write side stays bound: a fresh agent persists at session end.
    assert fresh.memory_hub.scn.persistence_path == str(agent_home / "scn.json")

    loaded = factory.create_agent(AgentConfig(agent_id="scout", persistence_dir=str(agent_home)), auto_load=True)
    assert loaded.memory_hub.scn.get_signature("d28_probe") is not None, (
        "auto_load=True stopped restoring SCN — the fix over-rotated"
    )


def test_auto_load_true_on_factory_still_raises_when_asked(tmp_path, agent_home):
    """The strictness follows on_corrupt, not the load.agent() wrapper."""
    from maxim.runtime.agent_factory import AgentConfig, AgentFactory

    (agent_home / "hippocampus.json").write_text("{not json", encoding="utf-8")
    factory = AgentFactory(base_data_dir=str(tmp_path))
    cfg = AgentConfig(agent_id="scout", persistence_dir=str(agent_home), on_corrupt="raise")
    with pytest.raises(MemoryCorruptionError):
        factory.create_agent(cfg, auto_load=True)


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


# ---------------------------------------------------------------------------
# Review-fold guards (two-lens round, 2026-08-23)
# ---------------------------------------------------------------------------


def test_on_corrupt_typo_is_a_hard_error_not_a_silent_downgrade(tmp_path):
    """A safety switch compared with `== "raise"` must not accept near-misses."""
    from maxim.runtime.agent_factory import AgentConfig

    for good in ("warn", "raise"):
        AgentConfig(agent_id="x", on_corrupt=good)
    for bad in ("Raise", "RAISE", "fail", "error", ""):
        with pytest.raises(ValueError, match="on_corrupt"):
            AgentConfig(agent_id="x", on_corrupt=bad)


def test_failed_load_does_not_leak_the_memory_hub_worker(agent_home, tmp_path):
    """The raise path must release the hub; a retrying caller would leak per attempt."""
    import threading

    (agent_home / "hippocampus.json").write_text("{not json", encoding="utf-8")
    before = {t.ident for t in threading.enumerate()}
    for _ in range(3):
        with pytest.raises(MemoryCorruptionError):
            _load(tmp_path)
    leaked = [t for t in threading.enumerate() if t.ident not in before and t.is_alive()]
    extractor = [t for t in leaked if "concept-extractor" in t.name]
    assert not extractor, f"leaked {len(extractor)} ConceptExtractor worker(s) across 3 failed loads"


def test_oserror_family_is_reported_not_swallowed(agent_home, tmp_path):
    """NAc.load_safe deliberately does not catch OSError; the factory must still report."""
    (agent_home / "nac.json").mkdir()  # IsADirectoryError on read
    with pytest.raises(MemoryCorruptionError) as ei:
        _load(tmp_path)
    assert "nac" in {c["subsystem"] for c in ei.value.context["corrupt"]}


def test_subsystem_that_fails_to_build_is_never_silently_absent(agent_home, tmp_path):
    """A None subsystem must not be handed back as a successful 'full restore'."""
    (agent_home / "nac.json").mkdir()
    with pytest.raises(MemoryCorruptionError) as ei:
        _load(tmp_path)
    reported = {c["subsystem"] for c in ei.value.context["corrupt"]}
    assert "memory_hub" in reported, f"a null MemoryHub went unreported: {reported}"


def test_ec_is_restored_alongside_nac(agent_home, tmp_path):
    """D2's invariant at the LOAD site: a restored NAc may not get a fresh EC."""
    from maxim.similarity.ec import ECConfig, EntorhinalCortex

    ec = EntorhinalCortex(config=ECConfig(persistence_path=str(agent_home / "ec.json")))
    ec.save(str(agent_home / "ec.json"))
    (agent_home / "nac.json").write_text("{}", encoding="utf-8")

    inst = _load(tmp_path, on_corrupt="fresh")
    hub_ec = getattr(inst.memory_hub, "ec", None)
    assert hub_ec is not None
    assert getattr(hub_ec.config, "persistence_path", None) == str(agent_home / "ec.json"), (
        "EC was constructed without persistence — restored NAc biases would dangle (D2)"
    )


def test_half_present_nac_ec_pair_warns_but_still_loads(agent_home, tmp_path, caplog):
    """nac.json without ec.json is the dangling-bias state D2 describes — but it
    is also the NORMAL state of every agent saved before EC was persisted on
    this path, so it must be loud, not fatal. Refusing would make every
    pre-existing agent unloadable, and re-saving (the fix) requires loading."""
    (agent_home / "nac.json").write_text("{}", encoding="utf-8")
    assert not (agent_home / "ec.json").exists()
    with caplog.at_level("WARNING"):
        inst = _load(tmp_path)
    assert inst is not None, "a legacy half-pair must still load"
    assert any("Half-present NAc/EC pair" in r.getMessage() for r in caplog.records), "the half-pair must be reported"


def test_complete_pair_is_not_flagged(agent_home, tmp_path):
    from maxim.similarity.ec import ECConfig, EntorhinalCortex

    EntorhinalCortex(config=ECConfig(persistence_path=str(agent_home / "ec.json"))).save(str(agent_home / "ec.json"))
    (agent_home / "nac.json").write_text("{}", encoding="utf-8")
    inst = _load(tmp_path)  # must not raise
    assert inst is not None


def test_session_start_does_not_wipe_writes_made_after_load(agent_home, tmp_path):
    """ATL.load_state clears before restoring — a second read would discard writes."""
    inst = _load(tmp_path)
    inst.atl.store(SemanticMemory(id="d17-rope", timestamp=0.0, name="rope", definition="cordage", category="object"))
    inst.memory_hub.on_session_start()
    names = {getattr(c, "name", None) for c in inst.atl.recall(limit=50)}
    assert {"lantern", "rope"} <= names, f"session start discarded a post-load write: {names}"


def test_memory_corruption_error_is_importable_from_maxim():
    """stable_api.md tells callers to catch this by name."""
    import maxim

    assert maxim.MemoryCorruptionError is MemoryCorruptionError
