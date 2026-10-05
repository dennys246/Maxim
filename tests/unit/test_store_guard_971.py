"""#971 -- the #939 save guard on every other persisted store: NAc, EC, SCN, AngularGyrus, cross-layer.

#939 made Hippocampus and ATL refuse to save over a file they neither read nor created. The other
per-agent stores still started empty and saved over one: `AgentFactory.create_agent(auto_load=False)`
overwrote a readable home at shutdown, and NAc (`load_safe`), EC (the empty rebuild), SCN (the factory)
and AngularGyrus (`load` swallowed the error) each saved an empty store over an UNREADABLE file, keeping
no copy.

Owner decisions 2026-09-28: the same guard on each store; an unreadable file is kept as
`<name>.corrupt-<UTC>` and the store saves fresh in its place, uniformly (SCN's pathless special case
retired); and because NAc biases are keyed on EC node ids, an unreadable ec.json resets NAc too.
"""

from __future__ import annotations

import json
import logging
import time as _real_time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pytest

from maxim.exceptions import StoreOverwriteRefused


# ── one spec per store ───────────────────────────────────────────────────


@dataclass
class Spec:
    name: str
    filename: str
    make: Callable[[str | None], Any]
    populate: Callable[[Any], None]
    content: Callable[[Any], Any]
    break_partway: Callable[[dict], None]  # parses as JSON, fails after some surfaces are loaded


def _nac_break(d):
    d["percept_valences"] = "not-a-container"


def _ec_break(d):
    d["substrate_nodes"]["zz-bad"] = "not-a-node"


def _scn_break(d):
    d["signatures"]["zz-bad"] = "not-a-signature"


def _ag_break(d):
    d["records"]["zz-bad"] = "not-a-record"


def _cl_break(d):
    d["edges"].append({"edge_type": d["edges"][0]["edge_type"]})  # after a good edge, missing its ends


def _nac(path):
    from maxim.decisions.nac import NAc, NACConfig

    return NAc(NACConfig(persistence_path=path))


def _nac_fill(n):
    n._reward_bias[("act", "node-a")] = 0.5


def _ec(path):
    from maxim.similarity.ec import ECConfig, EntorhinalCortex

    return EntorhinalCortex(ECConfig(persistence_path=path))


def _ec_fill(ec):
    from maxim.similarity.ec import SituationSignature

    ec.ingest_substrate_nodes({"node-a": {"embedding": [0.1, 0.2, 0.3], "modality": "text"}})
    # Signatures and encoder stamps too: load sets them BEFORE the substrate nodes a break lands in, so
    # a partial load leaves them behind unless the reset list covers them.
    ec.register(
        "mem-1",
        signature=SituationSignature(
            semantic_hash=(1, 2, 3),
            structural_hash=42,
            temporal_hash=(9, 3, 1, 8),
            context_hash=7,
            tool_name="probe_tool",
            outcome_type="success",
            mode="test",
            goal_keywords=("probe",),
        ),
    )
    ec.record_encoder_provenance("text", {"model": "probe"})


def _scn(path):
    from maxim.time.scn import SCN

    return SCN(persistence_path=path)


def _scn_fill(scn):
    from maxim.time.temporal_signature import TemporalSignature

    scn.register("mem-1", TemporalSignature.from_timestamp(1_700_000_000.0))


def _ag(path):
    from maxim.math.angular_gyrus import AngularGyrus, AngularGyrusConfig

    # Unseeded, so a partial load (which would re-add the built-ins) differs from a fresh store.
    return AngularGyrus(AngularGyrusConfig(persistence_path=path, seed_built_ins=False))


def _cl(path):
    from maxim.memory.cross_layer import CrossLayerGraph

    return CrossLayerGraph(persistence_path=path)


def _cl_fill(cl):
    from maxim.memory.cross_layer import CrossLayerEdgeType

    cl.add_edge("hippocampus", "m1", "atl", "c1", next(iter(CrossLayerEdgeType)))


SPECS = [
    # Exact: the frozen clock below makes NAc.load()'s wall-clock decay-on-load a no-op.
    Spec("nac", "nac.json", _nac, _nac_fill, lambda s: s.dump()["reward_bias"], _nac_break),
    Spec("ec", "ec.json", _ec, _ec_fill, lambda s: sorted(s._substrate_nodes), _ec_break),
    Spec("scn", "scn.json", _scn, _scn_fill, lambda s: s.dump()["signatures"], _scn_break),
    Spec(
        "angular_gyrus",
        "angular_gyrus.json",
        _ag,
        lambda s: s._seed_built_ins(),
        lambda s: sorted(s._records),
        _ag_break,
    ),
    Spec("cross_layer", "cross_layer_graph.json", _cl, _cl_fill, lambda s: s.dump(), _cl_break),
]


class _FrozenWallClock:
    """``time`` as NAc sees it, with ``time()`` stopped: every other attribute is the real module's."""

    def __init__(self, now: float) -> None:
        self._now = now

    def time(self) -> float:
        return self._now

    def __getattr__(self, name: str) -> Any:
        return getattr(_real_time, name)


@pytest.fixture(autouse=True)
def _nac_wall_clock_stopped(monkeypatch):
    """NAc decays its biases on load by ``time.time() - saved_at`` (#818). These tests are about which file a
    store may write, not about decay, so no wall-clock time passes between a save and a load: the round trip
    is exact on any runner. (Rounding the compared value only moved the threshold a slow CI box crossed.)"""
    import maxim.decisions.nac as nac_module

    monkeypatch.setattr(nac_module, "time", _FrozenWallClock(1_800_000_000.0))


@pytest.fixture(params=SPECS, ids=lambda s: s.name)
def spec(request) -> Spec:
    return request.param


def _write_populated(spec: Spec, path: Path) -> Any:
    store = spec.make(str(path))
    spec.populate(store)
    store.save()
    return store


# ── the guard, per store ─────────────────────────────────────────────────


def test_an_empty_store_refuses_to_save_over_a_file_it_never_read(spec: Spec, tmp_path: Path) -> None:
    path = tmp_path / spec.filename
    _write_populated(spec, path)
    before = path.read_bytes()
    fresh = spec.make(str(path))
    with pytest.raises(StoreOverwriteRefused) as exc:
        fresh.save()
    assert exc.value.store == spec.name
    assert path.read_bytes() == before


def test_a_store_that_read_its_file_saves_back_to_it(spec: Spec, tmp_path: Path) -> None:
    path = tmp_path / spec.filename
    writer = _write_populated(spec, path)
    reader = spec.make(str(path))
    reader.load()
    reader.save()
    again = spec.make(str(path))
    again.load()
    assert spec.content(again) == spec.content(writer)


def test_a_store_that_created_its_file_keeps_saving_to_it(spec: Spec, tmp_path: Path) -> None:
    path = tmp_path / spec.filename
    store = _write_populated(spec, path)
    store.save()  # its own file: no refusal


def test_overwrite_is_an_explicit_choice(spec: Spec, tmp_path: Path) -> None:
    path = tmp_path / spec.filename
    _write_populated(spec, path)
    spec.make(str(path)).save(overwrite=True)
    other = spec.make(str(path))
    other.allow_overwrite()
    other.save()


def test_a_tilde_path_saves_under_home(spec: Spec, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    monkeypatch.chdir(tmp_path)
    _write_populated(spec, f"~/{spec.filename}")  # type: ignore[arg-type]
    assert (tmp_path / "home" / spec.filename).exists()
    assert not (tmp_path / "~").exists()


def _recover(spec: Spec, store: Any) -> None:
    """The store's own recovering read: load_safe where it has one, else load (AG, cross-layer)."""
    if hasattr(store, "load_safe"):
        store.load_safe()
    else:
        store.load()


def test_an_unreadable_file_is_kept_as_a_copy_and_the_store_saves_fresh(spec: Spec, tmp_path: Path) -> None:
    path = tmp_path / spec.filename
    path.write_text('{"not": "valid"')
    store = spec.make(str(path))
    _recover(spec, store)
    copies = list(tmp_path.glob(f"{spec.filename}.corrupt-*"))
    assert len(copies) == 1 and copies[0].read_text() == '{"not": "valid"'
    store.save()  # claimed: the fresh store replaces the unreadable file
    json.loads(path.read_text())


def test_a_partly_loaded_store_is_emptied_before_it_saves(spec: Spec, tmp_path: Path) -> None:
    """A load that fails partway must not leave part of the file in the store it then saves."""
    path = tmp_path / spec.filename
    _write_populated(spec, path)
    data = json.loads(path.read_text())
    spec.break_partway(data)
    path.write_text(json.dumps(data))
    store = spec.make(str(path))
    _recover(spec, store)
    assert len(list(tmp_path.glob(f"{spec.filename}.corrupt-*"))) == 1
    assert spec.content(store) == spec.content(spec.make(None))
    # Every container the store holds, not a chosen few: a hand-kept reset list that misses one fails.
    assert _containers(store) == _containers(spec.make(None))


def _containers(store: Any) -> dict[str, Any]:
    out = {k: v for k, v in vars(store).items() if isinstance(v, (dict, set, list)) and k != "_store_owned_files"}
    if hasattr(store, "_inverted"):
        out["_inverted"] = store._inverted.to_dict()
    return out


def test_an_unreachable_file_is_never_licensed_for_replacement(tmp_path: Path, monkeypatch) -> None:
    """An OSError is not a data error: the store keeps no copy, and a save over the file refuses."""
    from maxim.decisions import nac as nac_mod

    path = tmp_path / "nac.json"
    _write_populated(SPECS[0], path)
    store = _nac(str(path))
    real_open = open

    def denied(p, *a, **k):
        if str(p) == str(path):
            raise PermissionError("denied")
        return real_open(p, *a, **k)

    monkeypatch.setattr(nac_mod, "open", denied, raising=False)
    with pytest.raises(PermissionError):
        store.load_safe()
    monkeypatch.undo()
    assert not list(tmp_path.glob("nac.json.corrupt-*"))
    with pytest.raises(StoreOverwriteRefused):
        store.save()


def test_nac_recovery_empties_every_surface_load_state_sets(tmp_path: Path) -> None:
    """load_safe's hand-kept reset list had drifted (cluster_fear, cluster_reward_source, inherent keys)."""
    from maxim.decisions.nac import NAc

    path = tmp_path / "nac.json"
    writer = _nac(str(path))
    writer._cluster_fear[("act", "ctx", "node-a")] = -0.9
    writer.save()
    data = json.loads(path.read_text())
    data["percept_valences"] = "not-a-container"
    path.write_text(json.dumps(data))
    store = _nac(str(path))
    ok, _ = store.load_safe()
    assert not ok
    assert store.dump() == NAc().dump()


def test_the_angular_gyrus_refusal_is_not_swallowed_by_its_own_save(tmp_path: Path) -> None:
    """AngularGyrus.save catches OSError; the refusal is a FileExistsError and must still surface."""
    path = tmp_path / "angular_gyrus.json"
    _write_populated(SPECS[3], path)
    with pytest.raises(StoreOverwriteRefused):
        _ag(str(path)).save()


# ── compositions ─────────────────────────────────────────────────────────


def _seed_home(home: Path) -> dict[str, bytes]:
    """A readable agent home with NAc, EC, SCN and AngularGyrus state; returns each file's bytes."""
    home.mkdir(parents=True, exist_ok=True)
    for spec in SPECS[:4]:
        _write_populated(spec, home / spec.filename)
    return {s.filename: (home / s.filename).read_bytes() for s in SPECS[:4]}


def test_create_agent_without_loading_never_overwrites_an_existing_home(tmp_path: Path, caplog) -> None:
    """The factory's fresh agent over a readable home: its shutdown saves are refused, loudly."""
    from maxim.runtime.agent_factory import AgentConfig, AgentFactory

    home = tmp_path / "agents" / "a1"
    before = _seed_home(home)
    factory = AgentFactory(base_data_dir=tmp_path / "agents")
    agent = factory.create_agent(AgentConfig(agent_id="a1"), auto_load=False)
    with caplog.at_level(logging.ERROR):
        agent.shutdown()
    for name, data in before.items():
        assert (home / name).read_bytes() == data, name
    refused = [r for r in caplog.records if r.levelno >= logging.ERROR and "never read" in r.getMessage()]
    assert {n for n in ("nac", "ec", "scn") if any(f"{n}: refusing" in r.getMessage() for r in refused)} == {
        "nac",
        "ec",
        "scn",
    }


def test_the_write_but_dont_read_agent_declares_every_store(tmp_path: Path) -> None:
    from maxim.runtime.bio_stack import build_bio_stack

    _seed_home(tmp_path)
    stack = build_bio_stack(persistence_dir=str(tmp_path), agent_id="default_agent", load_persisted=False)
    try:
        for store in (stack.nac, stack.ec, stack.scn, stack.angular_gyrus):
            assert store.may_write(), type(store).__name__
    finally:
        stack.memory_hub.shutdown()


def _corrupt(path: Path) -> None:
    path.write_text('{"broken": ')


def test_an_unreadable_ec_resets_its_nac_too_in_the_bio_stack(tmp_path: Path) -> None:
    """NAc biases are keyed on EC node ids: a fresh EC beside a restored NAc leaves them dangling."""
    from maxim.runtime.bio_stack import build_bio_stack

    _seed_home(tmp_path)
    nac_bytes = (tmp_path / "nac.json").read_bytes()
    _corrupt(tmp_path / "ec.json")
    stack = build_bio_stack(persistence_dir=str(tmp_path), agent_id="default_agent")
    try:
        assert stack.nac.dump()["reward_bias"] == {}
        assert len(list(tmp_path.glob("ec.json.corrupt-*"))) == 1
        nac_copies = list(tmp_path.glob("nac.json.corrupt-*"))
        assert len(nac_copies) == 1 and nac_copies[0].read_bytes() == nac_bytes
        assert stack.nac.may_write() and stack.ec.may_write()
    finally:
        stack.memory_hub.shutdown()


def test_an_unreadable_nac_alone_keeps_its_ec(tmp_path: Path) -> None:
    from maxim.runtime.bio_stack import build_bio_stack

    _seed_home(tmp_path)
    _corrupt(tmp_path / "nac.json")
    stack = build_bio_stack(persistence_dir=str(tmp_path), agent_id="default_agent")
    try:
        assert "node-a" in stack.ec._substrate_nodes
        assert not list(tmp_path.glob("ec.json.corrupt-*"))
        assert len(list(tmp_path.glob("nac.json.corrupt-*"))) == 1
    finally:
        stack.memory_hub.shutdown()


def test_an_unreadable_scn_is_kept_and_the_agent_persists_scn_again(tmp_path: Path) -> None:
    """Retires SCN's pathless special case (owner decision): a copy, then SCN saves like the rest."""
    from maxim.runtime.bio_stack import build_bio_stack

    _seed_home(tmp_path)
    _corrupt(tmp_path / "scn.json")
    stack = build_bio_stack(persistence_dir=str(tmp_path), agent_id="default_agent")
    try:
        assert stack.scn.persistence_path == str(tmp_path / "scn.json")
        assert len(list(tmp_path.glob("scn.json.corrupt-*"))) == 1
        assert stack.scn.may_write()
    finally:
        stack.memory_hub.shutdown()


def test_load_agent_fresh_on_an_unreadable_ec_keeps_copies_and_saves(tmp_path: Path) -> None:
    from maxim.runtime.agent_factory import AgentConfig, AgentFactory

    home = tmp_path / "agents" / "a1"
    _seed_home(home)
    _corrupt(home / "ec.json")
    factory = AgentFactory(base_data_dir=tmp_path / "agents")
    agent = factory.create_agent(AgentConfig(agent_id="a1", on_corrupt="warn"), auto_load=True)
    assert agent.nac.dump()["reward_bias"] == {}
    agent.shutdown()
    assert len(list(home.glob("ec.json.corrupt-*"))) == 1
    assert len(list(home.glob("nac.json.corrupt-*"))) == 1
    json.loads((home / "ec.json").read_text())  # saved fresh in place
    assert json.loads((home / "nac.json").read_text())["reward_bias"] == {}


@pytest.mark.parametrize("which", ["nac", "ec", "scn"])
def test_load_agent_raise_mode_makes_no_copies(tmp_path: Path, which: str) -> None:
    from maxim.exceptions import MemoryCorruptionError
    from maxim.runtime.agent_factory import AgentConfig, AgentFactory

    home = tmp_path / "agents" / "a1"
    _seed_home(home)
    _corrupt(home / f"{which}.json")
    factory = AgentFactory(base_data_dir=tmp_path / "agents")
    with pytest.raises(MemoryCorruptionError):
        factory.create_agent(AgentConfig(agent_id="a1", on_corrupt="raise"), auto_load=True)
    assert not list(home.glob("*.corrupt-*"))


@pytest.mark.parametrize("lightweight", [False, True], ids=["full", "lightweight"])
@pytest.mark.parametrize("store", ["nac", "ec", "scn", "angular_gyrus", "cross_layer"])
def test_the_hub_logs_a_session_end_refusal_at_error(tmp_path: Path, store: str, lightweight: bool, caplog) -> None:
    from maxim.integration.memory_hub import build_memory_hub

    spec = next(s for s in SPECS if s.name == store)
    path = tmp_path / spec.filename
    _write_populated(spec, path)
    from maxim.memory.hippocampus import Hippocampus, HippocampusConfig

    kwargs: dict[str, Any] = {
        "hippocampus": Hippocampus(HippocampusConfig(auto_save_after_sleep=False)),
        "scn": _scn(None),
        "nac": _nac(None),
        "ec": _ec(None),
        "load_persisted": True,
    }
    if store != "cross_layer":
        kwargs[store] = spec.make(str(path))
    hub = build_memory_hub(agent_id="t", **kwargs)
    if store == "cross_layer":
        # The hub builds its graph only beside ATL/AngularGyrus, and never with a path; give it one.
        hub._cross_layer = spec.make(str(path))
    for name in ("angular_gyrus", "_cross_layer"):
        # The hub READS these at session start; keep that read off the file under test.
        if getattr(hub, name, None) is not None:
            getattr(hub, name).load = lambda *a, **k: None
    hub.on_session_start()
    with caplog.at_level(logging.ERROR):
        (hub.on_session_end_lightweight if lightweight else hub.on_session_end)()
    assert any(r.levelno >= logging.ERROR and "refusing to save" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("store", ["nac", "ec", "scn"])
def test_save_aut_state_logs_a_refusal_at_error(tmp_path: Path, store: str, caplog) -> None:
    """The sim's per-session save: a refusal is an ERROR there too, never the DEBUG other failures get."""
    from maxim.simulation.report import save_aut_state

    spec = next(s for s in SPECS if s.name == store)
    session_dir = tmp_path / "sim_reports" / "s1"
    session_dir.mkdir(parents=True)
    _write_populated(spec, session_dir / f"aut_{spec.filename}")  # a file this AUT store never read
    stores: dict[str, Any] = {"nac": None, "ec": None, "scn": None}
    stores[store] = spec.make(None)
    with caplog.at_level(logging.DEBUG):
        save_aut_state(hippocampus=None, base_dir=str(tmp_path / "sim_reports"), session_id="s1", **stores)
    assert any(r.levelno >= logging.ERROR and "not saved" in r.getMessage() for r in caplog.records)


def test_a_nac_that_cannot_keep_its_copy_stops_owning_its_readable_file(tmp_path: Path, monkeypatch) -> None:
    """Emptied beside an unreadable EC, a NAc whose copy failed must not save its emptiness over the
    readable file -- then the file is the only copy of what it held."""
    import shutil

    from maxim.runtime.agent_factory import reset_nac_beside_unreadable_ec

    path = tmp_path / "nac.json"
    _write_populated(SPECS[0], path)
    before = path.read_bytes()
    nac = _nac(str(path))
    nac.load()
    ec_path = tmp_path / "ec.json"
    ec = _write_populated(SPECS[1], ec_path)  # stands in for an EC that already saved fresh
    monkeypatch.setattr(shutil, "copy2", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    reset_nac_beside_unreadable_ec(nac, ec=ec)
    monkeypatch.undo()
    assert nac.dump()["reward_bias"] == {}
    with pytest.raises(StoreOverwriteRefused):
        nac.save()
    assert path.read_bytes() == before
    # The pair stays together on disk: the EC does not save over its file either (review round 2).
    with pytest.raises(StoreOverwriteRefused):
        ec.save()


@pytest.mark.parametrize("which", ["nac", "scn"])
def test_load_agent_fresh_on_an_unreadable_store_keeps_a_copy_and_saves(tmp_path: Path, which: str) -> None:
    from maxim.runtime.agent_factory import AgentConfig, AgentFactory

    home = tmp_path / "agents" / "a1"
    _seed_home(home)
    _corrupt(home / f"{which}.json")
    factory = AgentFactory(base_data_dir=tmp_path / "agents")
    agent = factory.create_agent(AgentConfig(agent_id="a1", on_corrupt="warn"), auto_load=True)
    agent.shutdown()
    assert len(list(home.glob(f"{which}.json.corrupt-*"))) == 1
    json.loads((home / f"{which}.json").read_text())  # saved fresh in place
    assert "node-a" in json.loads((home / "ec.json").read_text())["substrate_nodes"]  # EC kept


def test_ec_recovery_restores_the_config_the_file_overrode(tmp_path: Path) -> None:
    """EC.load adopts the file's index settings before it fails; the recovered EC keeps its own."""
    from maxim.similarity.ec import ECConfig, EntorhinalCortex

    path = tmp_path / "ec.json"
    _write_populated(SPECS[1], path)
    data = json.loads(path.read_text())
    _ec_break(data)
    path.write_text(json.dumps(data))
    ec = EntorhinalCortex(ECConfig(persistence_path=str(path), default_k=7))
    ok, _ = ec.load_safe()
    assert not ok and ec.config.default_k == 7


def test_shutdown_logs_its_own_nac_refusal_at_error(tmp_path: Path, caplog) -> None:
    from maxim.runtime.agent_factory import AgentConfig, AgentFactory

    home = tmp_path / "agents" / "a1"
    _seed_home(home)
    agent = AgentFactory(base_data_dir=tmp_path / "agents").create_agent(AgentConfig(agent_id="a1"))
    agent.memory_hub.shutdown()
    agent.memory_hub = None  # only shutdown's own NAc save runs
    with caplog.at_level(logging.ERROR):
        agent.shutdown()
    assert any(r.levelno >= logging.ERROR and "NAc not saved" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize(
    "module",
    [
        "maxim.decisions.nac",
        "maxim.similarity.ec",
        "maxim.time.scn",
        "maxim.math.angular_gyrus",
        "maxim.memory.cross_layer",
        "maxim.utils.store_ownership",
    ],
)
def test_each_guarded_store_imports_first_in_a_fresh_interpreter(module: str) -> None:
    """The mixin lives in a leaf module: importing maxim.memory runs its __init__, which imports the
    Hippocampus, which imports NAc -- so NAc inheriting from anything in that package is a cycle that
    only shows when NAc is imported FIRST. A single pytest process imports in an order that hides it."""
    import os
    import subprocess
    import sys

    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-c", f"import {module}"], capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr[-2000:]


# ── review round 1 folds ─────────────────────────────────────────────────

_APPLY_STEP = {  # the method that rebuilds the store from what was read
    "nac": "load_state",
    "ec": "_load_file",
    "scn": "load_state",
    "angular_gyrus": "_load_payload",
    "cross_layer": "load_state",
}


@pytest.mark.parametrize("error", [IndexError, RecursionError, OverflowError, ZeroDivisionError])
def test_the_added_data_error_types_mean_unreadable(spec: Spec, tmp_path: Path, monkeypatch, error) -> None:
    """IndexError, RecursionError and OverflowError are shapes bad content raises; outside the old list
    of four they escaped recovery and crashed construction (#971 review)."""
    path = tmp_path / spec.filename
    _write_populated(spec, path)
    store = spec.make(str(path))
    real = getattr(store, _APPLY_STEP[spec.name])
    calls = []

    def broken_once(*a, **k):  # the reset reuses load_state on some stores: fail the load only
        calls.append(1)
        if len(calls) == 1:
            raise error("bad content")
        return real(*a, **k)

    monkeypatch.setattr(store, _APPLY_STEP[spec.name], broken_once)
    _recover(spec, store)
    monkeypatch.undo()
    assert len(list(tmp_path.glob(f"{spec.filename}.corrupt-*"))) == 1
    store.save()


def test_the_bio_stack_survives_an_odd_corruption(tmp_path: Path, monkeypatch) -> None:
    from maxim.runtime.bio_stack import build_bio_stack
    from maxim.similarity.ec import EntorhinalCortex

    _seed_home(tmp_path)

    def broken(self, path):
        raise IndexError("bad content")

    monkeypatch.setattr(EntorhinalCortex, "_load_file", broken)
    stack = build_bio_stack(persistence_dir=str(tmp_path), agent_id="default_agent")
    try:
        assert len(list(tmp_path.glob("ec.json.corrupt-*"))) == 1
        assert stack.nac.dump()["reward_bias"] == {}  # the pair rule fired too
    finally:
        stack.memory_hub.shutdown()


def test_load_agent_fresh_survives_a_wrongly_typed_ec_config(tmp_path: Path) -> None:
    """EC.load adopts the file's index settings first; recovery must not rebuild from them (reproduced:
    a string num_lsh_tables crashed load.agent(on_corrupt="fresh") with a TypeError)."""
    from maxim.runtime.agent_factory import AgentConfig, AgentFactory

    home = tmp_path / "agents" / "a1"
    _seed_home(home)
    data = json.loads((home / "ec.json").read_text())
    data["config"]["num_lsh_tables"] = "4"
    (home / "ec.json").write_text(json.dumps(data))
    agent = AgentFactory(base_data_dir=tmp_path / "agents").create_agent(
        AgentConfig(agent_id="a1", on_corrupt="warn"), auto_load=True
    )
    assert isinstance(agent.memory_hub.ec.config.num_lsh_tables, int)
    agent.shutdown()
    assert len(list(home.glob("ec.json.corrupt-*"))) == 1


def test_a_store_without_a_way_to_empty_itself_is_refused_at_class_definition() -> None:
    from maxim.utils.store_ownership import StoreFileOwnership

    with pytest.raises(TypeError, match="_reset_store_state"):

        class Leaky(StoreFileOwnership):  # no dump/load_state, no reset override
            _store_name = "leaky"

    class Fine(StoreFileOwnership):
        _store_name = "fine"

        def _reset_store_state(self) -> None:
            pass


@pytest.mark.parametrize("which", ["ec", "scn"])
def test_an_unreachable_file_propagates_from_load_safe_and_stays_unowned(
    tmp_path: Path, which: str, monkeypatch
) -> None:
    import builtins

    spec = next(s for s in SPECS if s.name == which)
    path = tmp_path / spec.filename
    _write_populated(spec, path)
    store = spec.make(str(path))
    real_open = builtins.open

    def denied(p, *a, **k):
        if str(p) == str(path):
            raise PermissionError("denied")
        return real_open(p, *a, **k)

    monkeypatch.setattr(builtins, "open", denied)
    with pytest.raises(PermissionError):
        store.load_safe()
    monkeypatch.undo()
    assert not list(tmp_path.glob(f"{spec.filename}.corrupt-*"))
    with pytest.raises(StoreOverwriteRefused):
        store.save()


def test_an_unreachable_angular_gyrus_file_is_left_alone(tmp_path: Path, monkeypatch, caplog) -> None:
    import builtins

    path = tmp_path / "angular_gyrus.json"
    _write_populated(SPECS[3], path)
    store = _ag(str(path))
    real_open = builtins.open

    def denied(p, *a, **k):
        if str(p) == str(path):
            raise PermissionError("denied")
        return real_open(p, *a, **k)

    monkeypatch.setattr(builtins, "open", denied)
    with caplog.at_level(logging.WARNING):
        store.load()  # the hub calls this unconditionally: it reports, it does not raise
    monkeypatch.undo()
    assert any("left untouched" in r.getMessage() for r in caplog.records)
    assert not list(tmp_path.glob("angular_gyrus.json.corrupt-*"))
    with pytest.raises(StoreOverwriteRefused):
        store.save()


@pytest.mark.parametrize("error", [NameError, ImportError, AssertionError, NotImplementedError, RuntimeError])
def test_a_code_defect_while_loading_propagates_and_leaves_the_file_alone(
    spec: Spec, tmp_path: Path, monkeypatch, error
) -> None:
    """Owner decision 2026-09-28: bad content is an explicit list of data-error types. A loader bug is
    not corruption -- it must fail loudly, never set the memories aside and start empty."""
    path = tmp_path / spec.filename
    _write_populated(spec, path)
    before = path.read_bytes()
    store = spec.make(str(path))

    def defect(*a, **k):
        raise error("a bug in the loader")

    monkeypatch.setattr(store, _APPLY_STEP[spec.name], defect)
    with pytest.raises(error):
        _recover(spec, store)
    monkeypatch.undo()
    assert not list(tmp_path.glob(f"{spec.filename}.corrupt-*"))
    assert path.read_bytes() == before
    with pytest.raises(StoreOverwriteRefused):
        store.save()


def test_the_factory_under_warn_empties_a_store_a_defect_left_half_loaded(tmp_path: Path, monkeypatch, caplog) -> None:
    """A loader defect is not bad content: no copy, the file untouched and unowned -- and the session does
    not run on whatever the failed load put in the store (review round 3)."""
    from maxim.runtime.agent_factory import AgentConfig, AgentFactory
    from maxim.similarity.ec import EntorhinalCortex

    home = tmp_path / "agents" / "a1"
    _seed_home(home)
    before = (home / "ec.json").read_bytes()
    real = EntorhinalCortex._load_file

    def half_then_defect(self, path):
        real(self, path)  # the whole file lands in the store...
        raise NameError("...then a bug in the loader")

    monkeypatch.setattr(EntorhinalCortex, "_load_file", half_then_defect)
    with caplog.at_level(logging.ERROR):
        agent = AgentFactory(base_data_dir=tmp_path / "agents").create_agent(
            AgentConfig(agent_id="a1", on_corrupt="warn"), auto_load=True
        )
    monkeypatch.undo()
    ec = agent.memory_hub.ec
    assert ec._substrate_nodes == {}  # not the half-loaded state
    assert any("reason other than bad content" in r.getMessage() for r in caplog.records)
    with pytest.raises(StoreOverwriteRefused):
        ec.save()
    agent.shutdown()
    assert (home / "ec.json").read_bytes() == before
    assert not list(home.glob("ec.json.corrupt-*"))


def test_a_defect_after_the_nac_claimed_its_file_still_leaves_the_file_alone(tmp_path: Path, monkeypatch) -> None:
    """NAc.load claims its file BEFORE decay-on-load: a defect there leaves it owned. Emptied in memory,
    the NAc must still stop owning the file, or its next save replaces good memories with nothing."""
    from maxim.decisions.nac import NAc
    from maxim.runtime.agent_factory import AgentConfig, AgentFactory

    home = tmp_path / "agents" / "a1"
    _seed_home(home)
    data = json.loads((home / "nac.json").read_text())
    data["saved_at"] = 1.0  # old enough that decay-on-load runs
    (home / "nac.json").write_text(json.dumps(data))
    before = (home / "nac.json").read_bytes()

    def defect(self, elapsed_s):
        raise NameError("a bug in decay-on-load")

    monkeypatch.setattr(NAc, "apply_wall_clock_decay", defect)
    agent = AgentFactory(base_data_dir=tmp_path / "agents").create_agent(
        AgentConfig(agent_id="a1", on_corrupt="warn"), auto_load=True
    )
    monkeypatch.undo()
    assert agent.nac.dump()["reward_bias"] == {}
    with pytest.raises(StoreOverwriteRefused):
        agent.nac.save()
    agent.shutdown()
    assert (home / "nac.json").read_bytes() == before


def test_an_ec_that_fails_otherwise_empties_its_nac_in_the_factory(tmp_path: Path, monkeypatch) -> None:
    """Round 4: the pair rule also holds when the EC fails for a reason other than bad content."""
    from maxim.runtime.agent_factory import AgentConfig, AgentFactory
    from maxim.similarity.ec import EntorhinalCortex

    home = tmp_path / "agents" / "a1"
    before = _seed_home(home)

    def defect(self, path):
        raise NameError("a bug in the loader")

    monkeypatch.setattr(EntorhinalCortex, "_load_file", defect)
    agent = AgentFactory(base_data_dir=tmp_path / "agents").create_agent(
        AgentConfig(agent_id="a1", on_corrupt="warn"), auto_load=True
    )
    monkeypatch.undo()
    assert agent.nac.dump()["reward_bias"] == {}
    agent.shutdown()
    for name in ("nac.json", "ec.json"):
        assert (home / name).read_bytes() == before[name], name
    assert not list(home.glob("*.corrupt-*"))


def test_an_unreachable_ec_empties_its_nac_in_the_bio_stack(tmp_path: Path, monkeypatch) -> None:
    from maxim.runtime.bio_stack import build_bio_stack
    from maxim.similarity.ec import EntorhinalCortex

    before = _seed_home(tmp_path)

    def unreachable(self, path):
        raise PermissionError("denied")

    monkeypatch.setattr(EntorhinalCortex, "_load_file", unreachable)
    stack = build_bio_stack(persistence_dir=str(tmp_path), agent_id="default_agent")
    monkeypatch.undo()
    try:
        assert stack.nac.dump()["reward_bias"] == {}
        with pytest.raises(StoreOverwriteRefused):
            stack.nac.save()
    finally:
        stack.memory_hub.shutdown()
    assert (tmp_path / "nac.json").read_bytes() == before["nac.json"]


def test_an_atl_emptied_in_memory_is_not_re_read_by_the_session_start(tmp_path: Path, monkeypatch) -> None:
    """Round 4: the hub's session start re-read atl.json, half-loading the ATL the factory had emptied."""
    from maxim.memory.atl import ATL, ATLConfig
    from maxim.runtime.agent_factory import AgentConfig, AgentFactory

    home = tmp_path / "agents" / "a1"
    home.mkdir(parents=True)
    atl = ATL(ATLConfig(persistence_path=str(home / "atl.json")))
    atl.find_or_create("wolf", "object")
    atl.save()
    real = ATL.load_state
    calls = []

    def half_then_defect(self, state):  # the reset reuses load_state: only the first load fails
        real(self, state)
        calls.append(1)
        if len(calls) == 1:
            raise NameError("a bug after the restore")

    monkeypatch.setattr(ATL, "load_state", half_then_defect)
    agent = AgentFactory(base_data_dir=tmp_path / "agents").create_agent(
        AgentConfig(agent_id="a1", on_corrupt="warn"), auto_load=True
    )
    # create_agent opens the session (the hub's restore runs there): it must not have re-read atl.json.
    assert len(agent.atl._concepts) == 0
    monkeypatch.undo()
    agent.shutdown()
