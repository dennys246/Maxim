"""Every persistent bio-system must get a persistence path from build_bio_stack.

NAc was the ONLY one left without one: hippocampus, ATL and angular-gyrus all
received `str(p / "<name>.json")`, NAc did not. MemoryHub's save is guarded by
`if nac_path:` — so it silently skipped, and causal links + reward biases were
NEVER written to disk. Episodes survived a restart; the LEARNING did not, which
is exactly the substrate the cross-session claim rests on.

It went unnoticed because the failure is silent at every layer: no exception,
no warning, and the agent home still fills with plausible-looking JSON.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from maxim.runtime.bio_stack import build_bio_stack


@pytest.fixture(scope="module")
def stack():
    return build_bio_stack(agent_id="persistence_probe", persistence_dir=tempfile.mkdtemp())


@pytest.mark.parametrize("system,attr", [("hippocampus", "config"), ("nac", "config")])
def test_core_systems_have_a_persistence_path(stack, system, attr):
    obj = getattr(stack, system, None)
    assert obj is not None, f"{system} not built"
    path = getattr(getattr(obj, attr, None), "persistence_path", None)
    assert path, f"{system} has no persistence_path — its state will silently never be saved"
    assert str(path).endswith(".json")


def test_nac_path_lands_in_the_agent_home(stack):
    # Not just non-None: it must be IN the agent's directory, beside the others.
    nac_path = Path(stack.nac.config.persistence_path)
    hippo_path = Path(stack.hippocampus.config.persistence_path)
    assert nac_path.parent == hippo_path.parent
    assert nac_path.name == "nac.json"


def test_nac_actually_round_trips_to_disk():
    # A path alone proves nothing — the whole bug was a path-shaped hole. Drive
    # the real save path and assert the file appears.
    from maxim.decisions.nac import Valence

    d = Path(tempfile.mkdtemp())
    bs = build_bio_stack(agent_id="roundtrip_probe", persistence_dir=str(d))
    bs.memory_hub._session_active = True
    bs.nac.observe(
        event_type="tool_use",
        event_signature="tool:respond",
        outcome_type="result",
        outcome_signature="ok",
        outcome_valence=Valence.POSITIVE,
        delta_seconds=0.0,
    )
    bs.memory_hub.on_session_end_lightweight()
    assert (d / "nac.json").is_file(), "NAc state was not written — the save is being skipped again"


def test_no_persistent_system_is_silently_pathless(stack):
    # Generic coverage: if a future bio-system persists state but gets no path,
    # it fails HERE rather than by quietly losing a session's learning.
    missing = []
    for name in ("hippocampus", "nac", "atl", "angular_gyrus"):
        obj = getattr(stack, name, None)
        if obj is None:
            continue  # optional systems may be absent by config
        cfg = getattr(obj, "config", obj)
        path = getattr(cfg, "persistence_path", None) or getattr(obj, "persistence_path", None)
        if not path:
            missing.append(name)
    assert not missing, f"bio-systems built with a persistence_dir but no path: {missing}"
