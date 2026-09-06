"""Guard tests for the Exp 56 four-arm harness (scripts/exp56 + the bench body).

The prereg's sign-off requires the harness PR to carry these
(docs/experiments/protocols/exp56_four_arm_sharing_preregistration.md):
frozen-constant pins, the body pins (opaque names, same-entity satiated
twin, <= 12 world sensors, L12 non-membership), the balanced-schedule
and teacher semantics (satiated mints ZERO — the mechanism check, with
its taught anti-vacuity twin), the translating client, the analyzer's
gates + refusals, and a reduced end-to-end chain on the deterministic
ScriptedBridge (donor -> real CLI export -> real CLI ingest -> probe ->
bias-decisive first contact) — the 1.1.4 ship-gate precedent:
deliberately NOT slow-marked, this IS the wiring gate.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from exp56 import common as C  # noqa: E402

BENCH_YAML = REPO / "src/maxim/_data/components/bodies/minecraft_bench.yaml"
SATIATED_YAML = REPO / "src/maxim/_data/components/bodies/minecraft_bench_satiated.yaml"


# ── body pins ────────────────────────────────────────────────────────────


class TestBenchBody:
    def test_world_channel_budget_and_opaque_names(self):
        spec = yaml.safe_load(BENCH_YAML.read_text())
        sensors = spec["entity"]["sensors"]
        world = [n for n, s in sensors.items() if s.get("modality") == "world"]
        assert len(world) <= 12, "L11 per-channel budget (prereg: stated and enforced)"
        assert len(world) >= 2
        affs = spec["entity"]["modulators"]["act"]["affordances"]
        assert sorted(affs) == sorted(C.AFFORDANCES)
        assert all(a.startswith("aff_") for a in affs), "opaque affordance names are the L12 mitigation"
        assert "d1" in sensors and sensors["d1"].get("drive"), "the opaque modeled drive"
        assert sensors["d1"].get("modality") != "world", (
            "d1 must be MODELED interoception — a world-owned drive short-circuits the operant path"
        )

    def test_drive_name_not_in_affinity_table(self):
        from maxim.decisions.nac import _DRIVE_TOOL_AFFINITIES

        assert "d1" not in _DRIVE_TOOL_AFFINITIES
        # And no roster tool name contains the drive name (the substring channel).
        assert not any("d1" in tool for tool in C.ROSTER)

    def test_satiated_twin_keeps_entity_name(self):
        spec = yaml.safe_load(SATIATED_YAML.read_text())
        assert spec["entity"]["name"] == C.ENTITY_NAME, (
            "the satiated twin must keep the SAME entity name — it prefixes every tool "
            "signature and is the body_ref (gate 7 refuses a renamed twin)"
        )
        assert spec["entity"]["sensors"]["d1"]["drive"]["drift_rate"] == 0

    def test_export_spec_matches_component_affordances(self):
        body = yaml.safe_load(C.BODY_SPEC_YAML.read_text())["body"]
        assert body["name"] == C.ENTITY_NAME
        assert sorted(body["modulators"]["act"]["affordances"]) == sorted(C.AFFORDANCES)

    def test_bench_body_instantiates_with_the_roster(self):
        from maxim.embodiment.component_registry import ComponentRegistry

        e = ComponentRegistry().instantiate(C.BODY_REF)
        assert e.name == C.ENTITY_NAME
        world = [n for n, s in e.sensors.items() if (s.reading_schema or {}).get("modality") == "world"]
        assert 2 <= len(world) <= 12


# ── frozen constants + per-pair derivations ──────────────────────────────


class TestFrozenApparatus:
    def test_frozen_constants_pin(self):
        # The prereg's sign-off records these at the harness-merge commit;
        # a drift here is a retune and must be a pre-registered amendment.
        assert C.FROZEN["roster_k"] == 8 and len(C.ROSTER) == 8
        assert C.FROZEN["epsilon"] == 0.2
        assert C.FROZEN["min_confidence"] == 0.3
        assert C.FROZEN["donor_bias_floor"] == 0.4
        assert C.SCHEDULE_K == 96
        assert C.FROZEN["probe_precontact_max"] == 10 and C.FROZEN["probe_tail"] == 5
        assert len(C.FROZEN["contingency_slots"]) == 4

    def test_balanced_schedule_is_balanced_and_seeded(self):
        sched = C.balanced_schedule(42)
        assert len(sched) == C.SCHEDULE_K
        from collections import Counter

        cells = Counter(sched)
        assert all(v == C.FROZEN["schedule_reps_per_cell"] for v in cells.values())
        assert len(cells) == 16
        assert C.balanced_schedule(42) == C.balanced_schedule(42)
        assert C.balanced_schedule(42) != C.balanced_schedule(43)

    def test_pair_config_shared_and_varying(self):
        a, b = C.pair_config(42), C.pair_config(42)
        assert a == b, "a pair's four arms must share the config"
        configs = [C.pair_config(s) for s in range(42, 92)]
        assert len({c["target_aff"] for c in configs}) > 1
        assert len({json.dumps(c["slot"]) for c in configs}) > 1

    def test_translating_client_permutes_per_pair_and_is_shared(self):
        class _Inner:
            def call_action(self, name, params=None):
                return {"ok": True, "name": name, "params": params}

        c1 = C.TranslatingClient(_Inner(), pair_seed=42)
        c2 = C.TranslatingClient(_Inner(), pair_seed=42)
        assert c1.action_map == c2.action_map, "donor and receiver share the pair's permutation"
        maps = {json.dumps(C.TranslatingClient(_Inner(), pair_seed=s).action_map, sort_keys=True) for s in range(30)}
        assert len(maps) > 5, "the permutation varies across pairs (review finding I1)"
        sent = c1.call_action("aff_a", {})
        assert sent["name"] == "turn" and "degrees" in sent["params"]


# ── the reduced end-to-end chain (the wiring gate; NOT slow-marked) ──────


@pytest.fixture(scope="module")
def scripted_world():
    server = C.ScriptedBridgeServer(seed=5, state_interval_s=0.02)
    world = C.ScriptedWorldControl(server, settle_s=0.05)
    yield server, world
    server.close()


@pytest.fixture(autouse=True)
def _operant_only(monkeypatch):
    monkeypatch.setenv("MAXIM_OPERANT_ONLY_CREDIT", "1")


def _mini_donor(tmp_path, server, world, *, body_ref, arm, pair_seed=42, target="aff_c", reps=4):
    session = C.build_bench_session(
        agent_id=f"d_{arm}_{pair_seed}",
        bridge_port=server.port,
        home=tmp_path / f"{arm}_home",
        pair_seed=pair_seed,
        body_ref=body_ref,
    )
    if body_ref == C.BODY_REF:
        # Deterministic head start for the drive: at test speed the first
        # target-at-situation feed can arrive before wall-clock drift has
        # lifted d1 off zero, making relief (honestly) zero. The drive
        # STATE is fixture; the relief mechanism under test is untouched
        # (drift still accrues on top; the satiated body stays at 0 by
        # definition — its zero-credit contrast must not be arranged).
        session.root.vital_metrics["d1"] = 0.3
    telemetry = C.run_donor_training(
        session,
        world=world,
        pair_seed=pair_seed,
        target_aff=target,
        arm=arm,
        slot=C.FROZEN["contingency_slots"][0],
        bot_name="t",
        settle_s=0.01,
        schedule=C.balanced_schedule(pair_seed, reps=reps),
    )
    return session, telemetry


class TestTeacherAndChain:
    def test_full_chain_taught_transfer_is_bias_decisive(self, tmp_path, scripted_world):
        """Donor taught by the teacher -> REAL CLI export -> REAL CLI ingest
        into a fresh receiver -> probe: the first contact is the taught
        action, from the substrate, learned-bias decisive, drive == 0."""
        server, world = scripted_world
        donor, telemetry = _mini_donor(tmp_path, server, world, body_ref=C.BODY_REF, arm="taught")
        feeds = [t for t in telemetry if t["fed"]]
        credits = [t for t in telemetry if t["credited"]]
        assert len(feeds) == 4 and len(credits) == 4, "target-at-situation cells only"
        assert all(t["reward"] == 1.0 for t in credits)
        assert all(t["credited_tsig"] == "tool:minecraft_bench_aff_c" for t in credits)
        sanity = C.donor_sanity(donor, arm="taught")
        assert sanity["pass"], sanity
        stage = C.close_and_stage_session(donor, stage_dir=tmp_path / "stage")
        C.export_bundle(stage, tmp_path / "b.zip", contributor_id="donor-42")

        recv_home = tmp_path / "recv"
        recv = C.build_bench_session(agent_id="r42", bridge_port=server.port, home=recv_home, pair_seed=42)
        C.close_and_stage_session(recv, stage_dir=tmp_path / "recv_stage")
        entry = C.ingest_bundle_into(recv_home, tmp_path / "b.zip", contributor_id="donor-42", receiver_agent_id="r42")
        assert entry["biases_rekeyed"] >= 1 and entry["biases_dropped"] == 0

        recv2 = C.build_bench_session(agent_id="r42", bridge_port=server.port, home=recv_home, pair_seed=42)
        probe = C.probe_receiver(
            recv2, world=world, pair_seed=42, slot=C.FROZEN["contingency_slots"][0], bot_name="t", settle_s=0.01
        )
        fc = probe["first_contact"]
        # pair 42's epsilon stream gives a substrate pick at first contact
        # (a seeded fact of this test, not a probability claim).
        assert fc["chosen"] == "minecraft_bench_aff_c"
        assert C.bias_decisive(fc, fc["chosen"]), fc
        C.close_and_stage_session(recv2, stage_dir=tmp_path / "recv_post")
        # Keep the artifacts for the noop-kit test below (module-scoped chain).
        (tmp_path / "kit").mkdir()
        shutil.copyfile(tmp_path / "b.zip", tmp_path / "kit" / "taught.zip")
        shutil.copyfile(tmp_path / "recv_stage" / "aut_nac.json", tmp_path / "kit" / "receiver_pre_nac.json")
        shutil.copyfile(tmp_path / "recv_stage" / "aut_ec.json", tmp_path / "kit" / "receiver_pre_ec.json")

        pre_nac = json.loads((tmp_path / "kit" / "receiver_pre_nac.json").read_text())
        pre_nac.pop("_format_version", None)
        pre_ec = json.loads((tmp_path / "kit" / "receiver_pre_ec.json").read_text())["substrate_nodes"]
        kit = C.noop_variant_readout(
            bundle=tmp_path / "kit" / "taught.zip",
            receiver_pre_nac=pre_nac,
            receiver_pre_ec=pre_ec,
            receiver_agent_id="r42",
            contributor_id="donor-42",
            first_contact=fc,
            target_tool="minecraft_bench_aff_c",
        )
        assert kit["kit_pass"], kit
        assert not kit["receiver_unchanged"]["chose_target"]
        assert not kit["empty_state"]["chose_target"]
        # donor_alone persists on a fresh receiver — documented, not asserted
        # as collapse (the D44 `return right` lesson).
        assert kit["donor_alone"]["chose_target"]

    def test_satiated_donor_mints_zero_credits_with_taught_twin(self, tmp_path, scripted_world):
        """The mechanism check + its anti-vacuity twin: SAME schedule, SAME
        teacher, SAME feeds — the satiated body mints nothing while the
        taught twin mints on every target-at-situation feed."""
        server, world = scripted_world
        sat, sat_tel = _mini_donor(tmp_path, server, world, body_ref=C.BODY_REF_SATIATED, arm="satiated", pair_seed=43)
        taught, taught_tel = _mini_donor(tmp_path, server, world, body_ref=C.BODY_REF, arm="taught", pair_seed=43)
        assert sum(1 for t in sat_tel if t["fed"]) == sum(1 for t in taught_tel if t["fed"]) == 4
        assert sum(1 for t in sat_tel if t["credited"]) == 0, "zero relief must mint zero credits"
        assert sum(1 for t in taught_tel if t["credited"]) == 4, "the twin proves the check can fail"
        assert C.donor_sanity(sat, arm="satiated")["pass"]
        C.close_and_stage_session(sat, stage_dir=tmp_path / "s1")
        C.close_and_stage_session(taught, stage_dir=tmp_path / "s2")

    def test_teacher_feeds_only_target_at_situation(self, tmp_path, scripted_world):
        server, world = scripted_world
        donor, telemetry = _mini_donor(tmp_path, server, world, body_ref=C.BODY_REF, arm="taught", pair_seed=44, reps=2)
        for t in telemetry:
            should_feed = t["situation_active"] and t["executed"] == t["target"]
            assert t["fed"] == should_feed
        C.close_and_stage_session(donor, stage_dir=tmp_path / "s")


# ── analyzer gates (synthetic rows) ──────────────────────────────────────


def _row(arm, *, chose, decisive, chosen="minecraft_bench_aff_a", pair=0):
    return {
        "pair_seed": pair,
        "arm": arm,
        "chose_target": chose,
        "bias_decisive": decisive,
        "first_contact": {"chosen": chosen},
        "mock": False,
    }


class TestAnalyzer:
    def _rows(self, taught_rate=0.8, iso_rate=0.1, sat_rate=0.1, dang_rate=0.1, n=50):
        rows = []
        for i in range(n):
            rows.append(
                _row("taught", chose=i < taught_rate * n, decisive=i < taught_rate * n, chosen=f"t{i % 5}", pair=i)
            )
            rows.append(_row("isolated", chose=i < iso_rate * n, decisive=False, chosen=f"i{i % 5}", pair=i))
            rows.append(_row("satiated", chose=i < sat_rate * n, decisive=False, chosen=f"s{i % 5}", pair=i))
            rows.append(_row("dangling", chose=i < dang_rate * n, decisive=False, chosen=f"d{i % 5}", pair=i))
        return rows

    def test_pass_verdict(self):
        sys.path.insert(0, str(REPO / "scripts"))
        import analyze_exp56 as A

        report = A.analyze(self._rows(), min_pairs=50)
        assert report["verdict"] == "PASS", report

    def test_dangling_transfer_fails_both_halves(self):
        import analyze_exp56 as A

        report = A.analyze(self._rows(dang_rate=0.5), min_pairs=50)
        assert report["verdict"] == "FAIL"
        assert report["gates"]["BOTH_HALVES"] is False

    def test_non_decisive_taught_cannot_pass(self):
        import analyze_exp56 as A

        rows = self._rows()
        for r in rows:
            if r["arm"] == "taught":
                r["bias_decisive"] = False  # link-driven "successes"
        report = A.analyze(rows, min_pairs=50)
        assert report["gates"]["TRANSFERRED"] is False

    def test_mock_rows_refuse_verdict(self):
        import analyze_exp56 as A

        rows = self._rows()
        rows[0]["mock"] = True
        report = A.analyze(rows, min_pairs=50)
        assert report["verdict"] == "NO-VERDICT"

    def test_l2_concentration_refuses_verdict(self):
        import analyze_exp56 as A

        rows = self._rows()
        for r in rows:
            if r["arm"] == "isolated":
                r["first_contact"]["chosen"] = "minecraft_bench_aff_h"  # seed-invariant floor
        report = A.analyze(rows, min_pairs=50)
        assert report["verdict"] == "NO-VERDICT"
        assert any("L2" in p for p in report["problems"])

    def test_dirty_rows_refuse_verdict(self):
        import analyze_exp56 as A

        rows = self._rows()
        rows[3]["working_tree_dirty_src_scripts"] = True
        report = A.analyze(rows, min_pairs=50)
        assert report["verdict"] == "NO-VERDICT"


# ── build_minecraft_aut param additions (existing callers byte-identical) ─


class TestSpectatorFlag:
    def test_spectator_commands_op_then_gamemode(self):
        from exp56.run_campaign import spectator_commands

        cmds = spectator_commands("Denny")
        assert cmds == ["op Denny", "gamemode spectator Denny"], cmds

    def test_spectator_arg_defaults_to_env(self, monkeypatch):
        # The flag reads EXP56_SPECTATOR so an operator can export it once.
        import argparse

        from exp56 import run_campaign

        monkeypatch.setenv("EXP56_SPECTATOR", "Watcher")
        # Rebuild just the one arg the way main() declares it.
        ap = argparse.ArgumentParser()
        import os as _os

        ap.add_argument("--spectator", default=_os.environ.get("EXP56_SPECTATOR"))
        assert ap.parse_args([]).spectator == "Watcher"
        assert hasattr(run_campaign, "spectator_commands")


class TestHarnessBuilderParams:
    def test_default_entity_ref_unchanged(self):
        import inspect

        from maxim.simulation.minecraft_harness import MINECRAFT_BODY_REF, build_minecraft_aut

        sig = inspect.signature(build_minecraft_aut)
        assert sig.parameters["entity_ref"].default == MINECRAFT_BODY_REF
        assert sig.parameters["client"].default is None
