#!/usr/bin/env python3
"""Exp 58 offline instrument gates 1–3 (prereg §Instrument gates) — run BEFORE any live trial.

G1  READ-PATH THROUGH THE PRODUCTION CALLER (wiring W-2): drive the real
    ``propose_via_substrate`` per tick against a scripted bridge — training ticks stage
    dark+damage states and the TICK ITSELF does everything (encode → note_active_clusters →
    evaluate_failures → pain → Wire-4 fear write → threat read); no hand-composed credit,
    no ``record_outcome`` calls, no executor.execute. Gate: after K training ticks, the
    healthy-dark probe tick proposes ``flee`` (and the lit probe tick does NOT).

G2  SAME-CLUSTER ASSERTION (bio DNB-2): the pain-time snapshot (dark + hurt +
    hostile-adjacent) and the healthy-dark probe snapshot must encode to the SAME world
    cluster — else fear books where the healthy prober never looks (unreadable write,
    false null). G1 depends on G2; both are reported.

G3  BOUNDARY-ACTIVATION SWEEP (confounding D1 / env SF-5): encode a lit→cave-mouth→interior
    gradient and report which states activate the trained dark cluster — this LICENSES (or
    not) the secondary entry-avoidance DV and sets the dark-zone measurement line.

Gated evidence (the prereg freezes citing this record): every completed run writes
``docs/experiments/data/exp58_offline_gates.json`` through the evidence path. Exit codes:
0 all gates pass, 2 usage, 3 provenance refusal, 4 gate FAIL or instrument error.

    python scripts/survival_world/exp58_offline_gates.py --write-experiment-results
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))

from _provenance import (  # noqa: E402
    DirtyTreeError,
    ProvenanceError,
    evidence_out_paths_or_exit,
    in_process_code_provenance,
)
from exp56 import common as C  # noqa: E402
from survival_world.common import make_fresh_encoder, settle_until  # noqa: E402
from survival_world.dark_danger_probe import DARK_THREAT, LIT_SAFE, ScriptedSurvivalBridge  # noqa: E402

AGENT_ID = "exp58_gates"
TRAIN_TICKS = 6  # dark+damage training ticks; fear saturates at cap (alpha 0.5) in 2

# The dark+hurt+hostile PAIN-TIME state (G2's left side) and the healthy-dark
# PROBE state (G2's right side; also G1's probe tick).
PAIN_TIME = {**DARK_THREAT, "health": 8.0}
HEALTHY_DARK = dict(DARK_THREAT)  # health 20, hostiles still sensed — the classroom probe

# G3 gradient: surface daylight → cave mouth → interior (the features the real cave
# co-varies: light falls, altitude falls, sky exposure goes).
BOUNDARY_SWEEP = [
    ("lit_surface", LIT_SAFE),
    ("mouth_outer", {**LIT_SAFE, "light_level": 8, "y_altitude": 60}),
    ("mouth_inner", {**LIT_SAFE, "light_level": 3, "y_altitude": 50, "nearest_hostile_dist": 24}),
    ("interior", HEALTHY_DARK),
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="docs/experiments/data/exp58_offline_gates.json")
    ap.add_argument("--write-experiment-results", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args(argv)

    out_arg = Path(args.out)
    out_abs = out_arg if out_arg.is_absolute() else (C.REPO_ROOT / out_arg)
    out_path = evidence_out_paths_or_exit(
        C.REPO_ROOT,
        [str(out_abs)],
        write_experiment_results=args.write_experiment_results,
        allow_dirty=args.allow_dirty,
    )[0]

    import maxim

    try:
        provenance = in_process_code_provenance(
            C.REPO_ROOT, maxim.__file__, out_path=out_path, allow_dirty=args.allow_dirty
        )
    except (DirtyTreeError, ProvenanceError) as exc:
        print(f"[FAIL] provenance: {exc}")
        return 3

    from maxim.runtime.agent_loop import _encode_current_clusters, propose_via_substrate
    from maxim.simulation.minecraft_harness import MinecraftSyncPump, build_minecraft_aut

    report: dict = {"ts": time.time(), "train_ticks": TRAIN_TICKS, "provenance": provenance, "instrument_error": None}

    def _finish(code: int) -> int:
        report["all_pass"] = bool(
            report.get("G1_read_path", {}).get("pass") and report.get("G2_same_cluster", {}).get("pass")
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        print(json.dumps({k: v for k, v in report.items() if k.startswith("G") or k == "instrument_error"}, indent=2))
        print(f"exp58 offline gates: {'PASS' if report['all_pass'] else 'FAIL'} -> {out_path}")
        return code

    bridge = ScriptedSurvivalBridge(LIT_SAFE)
    persistence_dir = tempfile.mkdtemp(prefix="exp58_gates_")
    aut = build_minecraft_aut(
        agent_id=AGENT_ID,
        bridge_port=bridge.port,
        bridge_host="127.0.0.1",
        persistence_dir=persistence_dir,
        entity_ref="bodies/minecraft_player",
    )
    encoder = make_fresh_encoder(aut)
    pump = MinecraftSyncPump(aut, interval_s=0.05)
    pump.start()

    # These ticks bypass the loop's live pass, which is what advances the experience clock
    # (memory-strength Phase 2S, #848): drive it once per tick here.
    from maxim.runtime.experience_time import ExperienceClockDriver

    clock_driver = ExperienceClockDriver(aut.bio.hippocampus.experience_clock, percept_source=None)
    # Memory 2S-d: resolved ONCE, so a hub with no cue (no ATL) fails here, not mid-episode.
    situation_cue = aut.bio.memory_hub.situation_cue

    def _tick() -> object:
        """ONE production tick: propose_via_substrate does encode → note →
        evaluate_failures (pain) → threat read → recommend. The whole Wire-4
        composition, through its real caller (W-2's requirement)."""
        proposal = propose_via_substrate(
            nac=aut.bio.nac,
            agent_id=AGENT_ID,
            executor=aut.executor,
            situation_cue=situation_cue,
            sensor_encoder=encoder,
        )
        clock_driver.on_live_pass()
        return proposal

    def _stage(state: dict, key: str, value: float) -> bool:
        bridge.set_state(state)
        return settle_until(aut, lambda vm: vm.get(key) == value, timeout_s=5.0) is not None

    try:
        if not _stage(LIT_SAFE, "light_level", 15):
            raise RuntimeError("bridge never delivered the lit state")
        _tick()  # baseline tick (notes lit clusters; no pain)

        # ── G2 first (G1 depends on it): encode both sides through production ──
        if not _stage(PAIN_TIME, "health", 8.0):
            raise RuntimeError("pain-time state never settled")
        pain_clusters = _encode_current_clusters(encoder, AGENT_ID, aut.executor)
        if not _stage(HEALTHY_DARK, "health", 20):
            raise RuntimeError("healthy-dark state never settled")
        probe_clusters = _encode_current_clusters(encoder, AGENT_ID, aut.executor)
        report["G2_same_cluster"] = {
            "pain_time_world": pain_clusters.get("world"),
            "healthy_dark_world": probe_clusters.get("world"),
            "pass": bool(pain_clusters.get("world")) and pain_clusters.get("world") == probe_clusters.get("world"),
        }

        # ── G1 training: dark + damage ticks through the production caller ──
        pain_before = aut.bio.pain_bus.get_stats().get("total_published", 0)
        for _ in range(TRAIN_TICKS):
            if not _stage(PAIN_TIME, "health", 8.0):
                raise RuntimeError("training damage state never settled")
            _tick()  # pain fires IN the tick, keyed to THIS tick's noted clusters
            if not _stage(HEALTHY_DARK, "health", 20):
                raise RuntimeError("training heal state never settled")
            _tick()  # healthy tick clears the publish latch (observed recovery)
        pain_after = aut.bio.pain_bus.get_stats().get("total_published", 0)
        dark_world = pain_clusters.get("world")
        fear = aut.bio.nac.cluster_fear(AGENT_ID, dark_world)
        lit_clusters_now = None

        # Probe A: healthy-dark — flee must be proposed (production min_confidence).
        if not _stage(HEALTHY_DARK, "health", 20):
            raise RuntimeError("probe dark state never settled")
        dark_proposal = _tick()
        # Probe B: lit — flee must NOT be proposed.
        if not _stage(LIT_SAFE, "light_level", 15):
            raise RuntimeError("probe lit state never settled")
        lit_proposal = _tick()
        lit_clusters_now = _encode_current_clusters(encoder, AGENT_ID, aut.executor)

        def _tool(p: object) -> str | None:
            return None if p is None else p.action.get("tool_name")

        report["G1_read_path"] = {
            "pain_published_during_training": pain_after - pain_before,
            "dark_world_fear": round(fear, 4),
            "lit_world_fear": round(aut.bio.nac.cluster_fear(AGENT_ID, lit_clusters_now.get("world")), 4),
            "dark_probe_tool": _tool(dark_proposal),
            "lit_probe_tool": _tool(lit_proposal),
            "pass": fear < 0
            and _tool(dark_proposal) is not None
            and str(_tool(dark_proposal)).endswith("_flee")
            and (lit_proposal is None or not str(_tool(lit_proposal)).endswith("_flee")),
        }

        # ── G3 boundary sweep (informational: licenses the entry-avoidance DV) ──
        sweep = []
        for label, state in BOUNDARY_SWEEP:
            if not _stage(state, "light_level", state["light_level"]):
                raise RuntimeError(f"sweep state {label} never settled")
            cl = _encode_current_clusters(encoder, AGENT_ID, aut.executor)
            sweep.append(
                {"state": label, "world_cluster": cl.get("world"), "is_dark_cluster": cl.get("world") == dark_world}
            )
        report["G3_boundary_sweep"] = {
            "sweep": sweep,
            "entry_avoidance_licensed": any(s["is_dark_cluster"] and s["state"].startswith("mouth") for s in sweep),
        }
    except Exception as exc:
        report["instrument_error"] = repr(exc)
        import traceback

        traceback.print_exc()
        return _finish(4)
    finally:
        try:
            pump.stop()
        except Exception as exc:
            print(f"WARNING: pump stop raised: {exc!r}")
        try:
            aut.bio.on_session_end()
        except Exception as exc:
            print(f"WARNING: bio teardown raised: {exc!r}")
        try:
            aut.client.close()
        except (OSError, ConnectionError) as exc:
            print(f"WARNING: client close raised: {exc!r}")
        bridge.close()
        shutil.rmtree(persistence_dir, ignore_errors=True)

    code = _finish(0 if (report["G1_read_path"]["pass"] and report["G2_same_cluster"]["pass"]) else 4)
    if code != 0:
        if not report["G2_same_cluster"]["pass"]:
            print(
                "  G2: pain-time and healthy-dark states encode to DIFFERENT world clusters —\n"
                "  fear books where the probe never looks. Redesign before build (bio DNB-2):\n"
                "  slimmer classroom body, or stage episodes so pain lands in the probe cluster."
            )
        if not report["G1_read_path"]["pass"]:
            print(
                "  G1: the production caller did not turn trained fear into a flee proposal —\n"
                "  inspect dark_world_fear (write side) vs dark_probe_tool (read side); the\n"
                "  Wire-4 seam that failed is the one the live run would silently null on."
            )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
