"""Exp 61 — the RED GATE for fear transport (1.3 Phase 2), written to fail.

`docs/experiments/exp61_shared_fear_prereg.md` §Mechanism: a learned Wire-4 `cluster_fear` must
travel from donor A to an independent receiver B through the SHIPPED bundle path — real
`compose_bundle` → real `ingest_bundle` (journal, `receiver_agent_id`) → B loads the report — and
be READ by the real consumer (`anticipatory_threat_need` on the node B's own reading completes
into, then `recommend_action`). On `main` today the scrub pops `cluster_fear`, ingest strips it
from a hand-built bundle, and the receiver reads 0.0: the two arms marked `xfail(strict=True)`
below are RED for that reason and for no other.

Discipline (CLAUDE.md "a fix ships with a CALLER"; D44's precedent in
`test_d44_merge_behavioural_delta.py`): these arms call what consumers call — never a hand-composed
sequence. The mechanism PR removes the markers IN the same PR; a marker that does not flip is
data, and re-pointing an arm at a recipe is forbidden. The unmarked arms are the negative controls
that must hold both BEFORE and AFTER the fix (an ABLATED donor transfers nothing; an ingest without
`receiver_agent_id` reads nothing — the agent-id rewrite is load-bearing).

Independence is D44's: distinct `agent_id`, a separate `EntorhinalCortex` + `SensorEncoder` per
agent (disjoint cluster ids by construction), no shared objects. The world modality is used as the
live body uses it — GAINED (`SensorEncoderConfig.gain_modalities`), range-aware, geometry-stamped,
so the ingest runs the STRICT path (no unstamped-geometry override).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from maxim.decisions.nac import NAc, NACConfig
from maxim.hivemind.bundle import compose_bundle
from maxim.hivemind.ingest import IngestionJournal, ingest_bundle
from maxim.similarity.ec import ECConfig, EntorhinalCortex
from maxim.similarity.encoder import SensorEncoder

BODY = "minecraft_player"
# A slice of the live body's world roster with its DECLARED ranges (rest at the midpoint under the
# A4 gain). "shore" moves one place sensor off rest so it does not encode to the zero vector (D2);
# "water" is the Exp 60 cue: the binary flag ON plus the air-hunger swing.
RANGES = {"is_in_water": (-1.0, 1.0), "oxygen": (0.0, 40.0), "y_altitude": (0.0, 128.0), "light_level": (0.0, 15.0)}
STATES = {
    "shore": {"is_in_water": 0.0, "oxygen": 20.0, "y_altitude": 64.0, "light_level": 14.0},
    "water": {"is_in_water": 1.0, "oxygen": 12.0, "y_altitude": 60.0, "light_level": 9.0},
}
ESCAPE_TOOLS = ("minecraft_player_escape_water", "minecraft_player_flee")
ROSTER = [
    "minecraft_player_move_to",
    "minecraft_player_turn",
    "minecraft_player_mine_block",
    "minecraft_player_place_block",
    "minecraft_player_eat",
    "minecraft_player_attack_nearest",
    *ESCAPE_TOOLS,
]
FAILURE_MODE = "drive:oxygen"


class Agent:
    """An agent with its OWN encoder, EC and NAc — no shared objects (D44 barrier 2)."""

    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self.nac = NAc(NACConfig())
        self.ec = EntorhinalCortex(ECConfig())
        self.encoder = SensorEncoder(ec=self.ec, atl=None, nac=self.nac)

    def cluster_for(self, state: str) -> str:
        node = self.encoder.encode_sensors(
            agent_id=self.agent_id, sensors=STATES[state], modality="world", ranges=RANGES
        )
        assert node is not None, state
        return str(node)

    def ec_nodes(self) -> dict:
        """The `substrate_nodes` slice in the shape export/ingest consume — geometry-STAMPED."""
        return {
            nid: {
                "embedding": emb,
                "modality": mod,
                "count": self.ec._substrate_node_counts.get(nid, 1),
                "source": self.ec._substrate_node_sources.get(nid, "local"),
                "domain": self.ec._substrate_node_domains.get(nid),
                "geometry": self.ec._substrate_node_geometries.get(nid),
            }
            for nid, (emb, mod) in self.ec._substrate_nodes.items()
        }

    def learn_drowning_fear(self, episodes: int = 2) -> str:
        """Exp 60's write, in miniature: saturating `drive:oxygen` pain while the water cluster is
        the noted situation → fear converges to the cap (−1.0 at α 0.5 after two 1.0 writes)."""
        cid = self.cluster_for("water")
        self.cluster_for("shore")  # the shore node exists too — specificity has something to read
        for _ in range(episodes):
            self.nac.record_cluster_fear(self.agent_id, cid, FAILURE_MODE, 1.0)
        return cid


def _export(donor: Agent, out: Path, *, ec: bool = True) -> Path:
    compose_bundle(
        nac_state=donor.nac.dump(),
        ec_substrate_nodes=donor.ec_nodes() if ec else None,
        output_path=out,
        contributor_id=donor.agent_id,
        body_ref=BODY,
    )
    return out


def _ingest_into(receiver: Agent, bundle: Path, tmp: Path, *, rewrite_agent_id: bool = True):
    report = ingest_bundle(
        bundle,
        receiver_nac=receiver.nac.dump(),
        receiver_ec_nodes=receiver.ec_nodes(),
        receiver_body=BODY,
        trusted_sources=frozenset({bundle.stem.split("__")[0]}),
        journal=IngestionJournal(tmp / f"journal_{receiver.agent_id}.json"),
        receiver_agent_id=receiver.agent_id if rewrite_agent_id else None,
    )
    receiver.nac.load_state(report.nac)
    receiver.ec.ingest_substrate_nodes(report.ec_nodes)
    return report


def _fresh_receiver(agent_id: str) -> Agent:
    b = Agent(agent_id)
    assert not b.ec._substrate_nodes, "a fresh receiver holds no world nodes (D44 independence)"
    return b


def _bundle_path(tmp: Path, donor: Agent, tag: str) -> Path:
    return tmp / f"{donor.agent_id}__{tag}.zip"


# ── sanity: the donor really learned, and independence is real ────────────────────────────────


def test_donor_learns_fear_to_the_cap_on_its_water_cluster_only() -> None:
    a = Agent("donor-A")
    water = a.learn_drowning_fear()
    assert a.nac.cluster_fear(a.agent_id, water) == -1.0
    assert a.nac.cluster_fear(a.agent_id, a.cluster_for("shore")) == 0.0
    assert a.nac.anticipatory_threat_need(a.agent_id, {"world": water}) >= 0.9
    # Export-before-probe, as a file fact: nothing was ever executed, so the link channel is empty.
    dump = a.nac.dump()
    assert dump.get("links") in (None, {}, []) and dump.get("cluster_reward_bias") in (None, {})


def test_independence_is_real_not_assumed() -> None:
    a, b = Agent("donor-A"), Agent("recv-B")
    assert a.cluster_for("water") != b.cluster_for("water")
    assert a.ec is not b.ec and a.nac is not b.nac and a.encoder is not b.encoder


# ── the gate ──────────────────────────────────────────────────────────────────────────────────


@pytest.mark.xfail(strict=True, reason="Exp 61 red gate: the bundle scrub pops cluster_fear; the receiver reads 0.0")
def test_transferred_fear_is_read_on_the_receiver_and_selects_escape(tmp_path: Path) -> None:
    """The claim's composition: real export → real ingest (agent id rewritten, strict geometry) →
    B's OWN submerged reading completes into the imported water node → the transferred fear is
    read there above the strict floor → the real consumer picks an escape affordance."""
    a = Agent("donor-A")
    a_water = a.learn_drowning_fear()
    b = _fresh_receiver("recv-B")
    report = _ingest_into(b, _export(a, _bundle_path(tmp_path, a, "fear")), tmp_path)

    # The representation half: B's reading lands on the imported node (a fresh receiver inserts
    # donor nodes under their own ids — asserted, not assumed).
    b_water = b.cluster_for("water")
    assert b_water == a_water, "B's submerged reading must complete into the transferred node"
    assert all(k.split("\x1f")[0] == b.agent_id for k in report.nac.get("cluster_fear", {})), (
        "every transported fear key must carry the RECEIVER's agent id after ingest"
    )

    # The credit half, at the real consumer.
    need = b.nac.anticipatory_threat_need(b.agent_id, {"world": b_water})
    assert need > NACConfig().cluster_fear_threshold, f"transferred fear must clear the strict floor, read {need}"
    assert b.nac.cluster_fear(b.agent_id, b.cluster_for("shore")) == 0.0, "specificity: no fear on the shore"
    rec = b.nac.recommend_action(
        agent_id=b.agent_id,
        available_tools=ROSTER,
        current_drives={"threat": need},
        current_clusters=None,
        min_confidence=0.0,
    )
    assert rec is not None and rec["tool_name"] in ESCAPE_TOOLS, rec


@pytest.mark.xfail(strict=True, reason="Exp 61 red gate: the ingest report carries no fear counters yet")
def test_dangling_half_drops_the_fear_loudly(tmp_path: Path) -> None:
    """The falsifier's accounting (the representation half): a nac-only bundle (no `ec.json`) must
    drop every fear key — counted, never faked — and the receiver must read nothing."""
    a = Agent("donor-A")
    a.learn_drowning_fear()
    shipped = len(a.nac.dump()["cluster_fear"])
    assert shipped >= 1
    b = _fresh_receiver("recv-B")
    report = _ingest_into(b, _export(a, _bundle_path(tmp_path, a, "dangling"), ec=False), tmp_path)
    assert report.fear_dropped == shipped and report.fear_rekeyed == 0 and report.fear_below_floor == 0
    assert b.nac.anticipatory_threat_need(b.agent_id, {"world": b.cluster_for("water")}) == 0.0


# ── negative controls: true before AND after the fix ──────────────────────────────────────────


def test_ablated_donor_transfers_the_cluster_but_no_fear(tmp_path: Path) -> None:
    """Arm 3's shape: a donor that saw the same water (its EC carries the node) but booked no fear
    (subscriber detached) changes nothing on the receiver — the arrival of a cluster is not the
    arrival of fear."""
    a = Agent("donor-A-ablated")
    a_water = a.cluster_for("water")
    a.cluster_for("shore")
    assert a.nac.cluster_fear(a.agent_id, a_water) == 0.0
    b = _fresh_receiver("recv-B")
    _ingest_into(b, _export(a, _bundle_path(tmp_path, a, "nofear")), tmp_path)
    b_water = b.cluster_for("water")
    assert b_water == a_water, "the CLUSTER transfers (representation half) even without fear"
    assert b.nac.anticipatory_threat_need(b.agent_id, {"world": b_water}) == 0.0
    rec = b.nac.recommend_action(agent_id=b.agent_id, available_tools=ROSTER, current_drives={}, min_confidence=0.0)
    assert rec is None or rec["tool_name"] not in ESCAPE_TOOLS


def test_ingest_without_receiver_agent_id_reads_nothing(tmp_path: Path) -> None:
    """Why the agent-id rewrite is in the path: `NAc.cluster_fear` filters on agent id, so a key
    left under the donor's id is invisible to the receiver — silently, which is the point."""
    a = Agent("donor-A")
    a.learn_drowning_fear()
    b = _fresh_receiver("recv-B")
    _ingest_into(b, _export(a, _bundle_path(tmp_path, a, "noid")), tmp_path, rewrite_agent_id=False)
    assert b.nac.anticipatory_threat_need(b.agent_id, {"world": b.cluster_for("water")}) == 0.0
