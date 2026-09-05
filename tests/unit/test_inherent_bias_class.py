"""Guards for the inherent bias class (1.2 poison-resistance slice, NAc side).

coding_habits_oasis.md §4 / docs/plans/oasis_ingestion_contract.md §6: the
class marker persists through dump/load, is decay-EXEMPT (with the
anti-vacuity arm: a non-inherent bias under the same conditions decays and
prunes — a guard that cannot fail is not a guard), enters only through the
curation surface (``mark_inherent_bias`` has no production caller — pinned
by grep here, so the day a learning path calls it, this test names the
self-promotion), and survives the bundle scrub only where its bias does.

The merge-transport guards live in test_hivemind_merge.py; the adapter's
Queen-gated entry guards in test_hivemind_ingest.py.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from maxim.decisions.nac import NAc, NACConfig

AID, CID, TSIG = "agent1", "cluster-a", "tool:touch"
OTHER_CID = "cluster-b"


def _nac_with_two_biases() -> NAc:
    nac = NAc(config=NACConfig())
    # Drive both biases to a clearly-negative learned aversion.
    for _ in range(20):
        nac.update_cluster_reward(AID, CID, TSIG, -1.0)
        nac.update_cluster_reward(AID, OTHER_CID, TSIG, -1.0)
    return nac


class TestDecayExemption:
    def test_inherent_bias_neither_decays_nor_prunes(self) -> None:
        nac = _nac_with_two_biases()
        nac.mark_inherent_bias(AID, CID, TSIG)
        held = nac.cluster_reward_bias(AID, CID, TSIG)
        assert held < 0.0
        # Far past the prune horizon for a learned bias at tau=300:
        # |bias| * (1 - 1/300)^N < 0.001 needs N ≈ 2000 ticks.
        for _ in range(5000):
            nac.decay_cluster_reward_biases()
        # The inherent bias is EXACTLY where it was — no decay, no prune.
        assert nac.cluster_reward_bias(AID, CID, TSIG) == held
        # Anti-vacuity arm: the non-inherent twin, identical in every other
        # respect, decayed to deletion under the same ticks.
        assert nac.cluster_reward_bias(AID, OTHER_CID, TSIG) == 0.0

    def test_marker_requires_existing_bias(self) -> None:
        nac = NAc(config=NACConfig())
        with pytest.raises(KeyError):
            nac.mark_inherent_bias(AID, "no-such-cluster", TSIG)


class TestPersistence:
    def test_marker_round_trips_through_dump_and_load(self) -> None:
        nac = _nac_with_two_biases()
        nac.mark_inherent_bias(AID, CID, TSIG)
        state = nac.dump()
        assert state["inherent_bias_keys"] == [f"{AID}\x1f{CID}\x1f{TSIG}"]

        nac2 = NAc(config=NACConfig())
        nac2.load_state(state)
        assert nac2.inherent_bias_keys == frozenset({(AID, CID, TSIG)})
        # And the exemption is live post-load, not just the marker.
        held = nac2.cluster_reward_bias(AID, CID, TSIG)
        for _ in range(5000):
            nac2.decay_cluster_reward_biases()
        assert nac2.cluster_reward_bias(AID, CID, TSIG) == held

    def test_pre_1_2_files_load_with_empty_class(self) -> None:
        nac = _nac_with_two_biases()
        state = nac.dump()
        del state["inherent_bias_keys"]
        nac2 = NAc(config=NACConfig())
        nac2.load_state(state)
        assert nac2.inherent_bias_keys == frozenset()

    def test_dangling_marker_is_dropped_at_load(self) -> None:
        nac = _nac_with_two_biases()
        state = nac.dump()
        state["inherent_bias_keys"] = [f"{AID}\x1fgone\x1f{TSIG}"]
        nac2 = NAc(config=NACConfig())
        nac2.load_state(state)
        assert nac2.inherent_bias_keys == frozenset()


class TestScrubTransport:
    def test_scrub_keeps_marker_whose_bias_survives_and_drops_the_rest(self) -> None:
        from maxim.hivemind.bundle import scrub_nac_state_for_bundle

        nac = _nac_with_two_biases()
        nac.mark_inherent_bias(AID, CID, TSIG)
        state = nac.dump()
        state["inherent_bias_keys"] = [
            f"{AID}\x1f{CID}\x1f{TSIG}",
            f"{AID}\x1fgone\x1f{TSIG}",  # no surviving bias entry
        ]
        scrubbed = scrub_nac_state_for_bundle(state)
        assert scrubbed["inherent_bias_keys"] == [f"{AID}\x1f{CID}\x1f{TSIG}"]

    def test_scrub_rekeys_marker_tsig_with_its_bias(self) -> None:
        from maxim.hivemind.bundle import scrub_nac_state_for_bundle

        free_text_sig = "tool:use:walk towards the shiny thing"
        key = f"{AID}\x1f{CID}\x1f{free_text_sig}"
        state = {
            "cluster_reward_bias": {key: -0.5},
            "inherent_bias_keys": [key],
            "links": {},
            "event_outcome_welford": {},
            "percept_valences": {},
        }
        scrubbed = scrub_nac_state_for_bundle(state)
        scrubbed_key = f"{AID}\x1f{CID}\x1ftool:use"
        assert scrubbed_key in scrubbed["cluster_reward_bias"]
        assert scrubbed["inherent_bias_keys"] == [scrubbed_key]


class TestNoSelfPromotionPath:
    def test_mark_inherent_bias_has_no_production_caller(self) -> None:
        """A locally-learned bias never self-promotes into the safety floor.

        Entry to the class is Queen provenance at the ingestion adapter
        (which transports already-marked bundles — it never CALLS the
        marker), plus future Queen-curation tooling. The day a learning
        path calls ``mark_inherent_bias``, this test fails and names the
        privilege-escalation path; the curation tooling that legitimately
        calls it must then be added to the allowlist HERE, in review.
        """
        src_root = Path(__file__).resolve().parents[2] / "src" / "maxim"
        allowlist: set[str] = set()  # no sanctioned production caller yet
        pattern = re.compile(r"\.mark_inherent_bias\(")
        offenders: list[str] = []
        for py in src_root.rglob("*.py"):
            rel = py.relative_to(src_root).as_posix()
            if rel in allowlist:
                continue
            if pattern.search(py.read_text(encoding="utf-8")):
                offenders.append(rel)
        assert offenders == [], f"mark_inherent_bias gained production callers: {offenders}"
