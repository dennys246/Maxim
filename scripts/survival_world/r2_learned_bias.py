#!/usr/bin/env python3
"""R2 learned-bias measurement (rung 1) — the frozen prereg is docs/experiments/r2_learned_bias_prereg.md.

Does break-2's measured DRIVE-RELIEF credit, booked through the REAL record_outcome path over
train-to-plateau hungry->eat->relief episodes, TEACH a substrate-primary agent a learned CLUSTER
bias that changes behaviour beyond its innate prior + the generic tool:eat causal link?

PRIMARY METRIC — MARGINAL cluster probe (prereg pre-data amendment): the raw None->eat flip is
confounded by the cluster-INDEPENDENT tool:eat causal link (nac.observe) that the ablation does NOT
suppress and that alone flips the probe in every arm. So per seed, at each probe food, on the SAME
trained NAc: recommend_action(current_clusters=encoded) picks eat AND NOT
recommend_action(current_clusters=None) picks eat. Prior + causal link are in BOTH and cancel; the
difference is purely the learned cluster bias. NO-CREDIT/SATIATED have cluster-bias 0 -> marginal 0.

DRIVE-RELIEF, not the floor: the generic tool-success floor also books to the interoception cluster,
so a floor-sourced eat would contaminate the same bias. We ENFORCE relief-only — any LEARNING seed
with a floor episode (a successful eat that measured no relief) is REFUSED — so the measured bias is
provably drive-relief-sourced (at food<=4 every eat measures +relief, so this should never fire; we
enforce rather than hope).

Three arms (interleaved within each seed; controls cycle-matched to LEARNING's plateau length):
  LEARNING / NO-CREDIT (harness-level ablation, suppresses relief+floor before record_outcome) /
  SATIATED (kept satiated, NO eat episodes, matched decision-cycle count).

Refusals ENFORCED -> exit 3 / REFUSED-UNVERIFIED (never a false null): credit_did_not_book,
ablation_leaked, did_not_plateau, drain_stalled, instrument_drift, floor_contaminated, error, and
config-fingerprint drift. A pre-run PREFLIGHT verifies the loop composes before spending 60 arm-runs.

Runs the REAL consumers (build_minecraft_aut, SensorEncoder, _encode_current_clusters,
record_outcome, read_learning_side_effects) — no hand-composed credit (D43). Substrate-primary (no
LLM in the action path) -> safe on the bridge box beside qwen32b. Bridge binds 127.0.0.1 -> run ON it.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))

from _provenance import evidence_out_paths, executed_code_provenance  # noqa: E402
from exp56.common import RconControl  # noqa: E402

FROZEN: dict[str, Any] = {
    "body_ref": "bodies/minecraft_player",
    "probe_food": (11, 12, 13),  # None-band (cold prior returns None; flips to eat at food<=10)
    "probe_health": 20.0,
    "N": 20,  # seeds/arm
    "M": 0.20,  # min meaningful marginal-flip-fraction
    "train_target_food": 4.0,  # deep-deficit training band
    "satiated_floor": 16.0,  # satiated arm keeps food at/above satisfaction
    "plateau_window": 5,
    "plateau_frac": 0.01,  # <1% running-bias change over the window = plateau
    "plateau_floor": 1e-6,  # bias must have MOVED off zero to count as plateaued (no flat-zero stop)
    "cap_episodes": 60,
    "min_confidence": 0.0,
    "arms": ("learning", "no_credit", "satiated"),
}

# Frozen config surface (absolute — asserted per seed; a stable-but-wrong ambient ~/.maxim must be
# CAUGHT, not merely cross-arm consistent). Values are the shipped dataclass defaults; the EC
# pattern_complete_threshold (0.44) is the food-4<->food-11 clustering lever the result hinges on.
FROZEN_CONFIG: dict[str, float] = {
    "substrate_explore_bonus_weight": 0.0,
    "max_cluster_reward_bias": 1.0,
    "max_reward_bias": 0.20,
    "reward_bias_decay_tau": 50.0,
    "cluster_reward_bias_decay_tau": 300.0,
    "cluster_bias_wall_decay_half_life_s": 86400.0,
    "ec_pattern_complete_threshold": 0.44,
    "encoder_pattern_threshold": 0.85,
}


# --------------------------------------------------------------------------- helpers


def _food(aut: Any) -> float | None:
    aut.backend.sync_world_sensors()
    vm = getattr(aut.executor.embodiment.root, "vital_metrics", {}) or {}
    if "food" not in vm:
        return None
    try:
        return float(vm["food"])
    except (TypeError, ValueError):
        return None


def _drain_until(aut: Any, rcon: Any, user: str, *, target: float, timeout_s: float) -> float | None:
    t0 = time.time()
    f = _food(aut)
    while time.time() - t0 < timeout_s:
        rcon.command(f"effect give {user} minecraft:hunger 1000 20")
        time.sleep(1.5)
        f = _food(aut)
        if f is not None and f <= target:
            break
    rcon.command(f"effect clear {user} minecraft:hunger")
    time.sleep(0.5)
    return _food(aut)


def _feed_to(aut: Any, rcon: Any, user: str, *, floor: float, timeout_s: float = 20.0) -> float | None:
    rcon.command(f"effect clear {user} minecraft:hunger")
    rcon.command(f"effect give {user} minecraft:saturation 3 20")  # game-native regen to full
    t0 = time.time()
    f = _food(aut)
    while time.time() - t0 < timeout_s:
        time.sleep(1.0)
        f = _food(aut)
        if f is not None and f >= floor:
            break
    return f


def _make_encoder(aut: Any) -> Any:
    from maxim.similarity.encoder import SensorEncoder, SensorEncoderConfig

    return SensorEncoder(ec=aut.bio.ec, config=SensorEncoderConfig())


def _eat_bias(aut: Any, agent_id: str, clusters: dict[str, str], eat_tool: str) -> float:
    from maxim.embodiment.sensory_streams import INTEROCEPTION_TAG

    cid = clusters.get(INTEROCEPTION_TAG)
    if not cid:
        return 0.0
    try:
        return float(aut.bio.nac.cluster_reward_bias(agent_id, cid, f"tool:{eat_tool}"))
    except Exception:
        return 0.0


def _picks_eat(
    aut: Any,
    agent_id: str,
    available: list[str],
    drives: dict[str, float],
    clusters: dict[str, str] | None,
    eat_tool: str,
) -> bool:
    rec = aut.bio.nac.recommend_action(
        agent_id=agent_id,
        available_tools=available,
        current_drives=drives,
        current_clusters=clusters,
        min_confidence=FROZEN["min_confidence"],
    )
    return (rec is not None) and (rec.get("tool_name") == eat_tool)


def _set_probe_state(aut: Any, food: float) -> dict[str, float]:
    from maxim.runtime.agent_loop import _read_drive_states

    aut.executor.embodiment.root.vital_metrics.update({"food": float(food), "health": FROZEN["probe_health"]})
    return _read_drive_states(aut.executor)


def _cold_prior_eats(aut: Any, agent_id: str, available: list[str], eat_tool: str, food: float) -> bool:
    """Cold-prior check WITHOUT encoding clusters — so the pre-probe does not seed the EC cluster
    space at the probe band before training (a methodological note the re-review raised). This is the
    `without-clusters` half of the marginal probe; the disclosure says it must be non-eat at food>=11."""
    return _picks_eat(aut, agent_id, available, _set_probe_state(aut, food), None, eat_tool)


def _marginal_probe(
    aut: Any, encoder: Any, agent_id: str, available: list[str], eat_tool: str, food: float
) -> dict[str, bool]:
    """Post-training probe: set the drive state in-memory (no world sync — the interoception channel
    reads vital_metrics directly), encode THAT state's clusters, read the frozen NAc twice: WITH
    encoded clusters (prior+causal+cluster-bias) and WITHOUT (prior+causal). marginal = with AND NOT
    without = the learned cluster bias's isolated behavioural effect. recommend_action is read-only."""
    from maxim.runtime.agent_loop import _encode_current_clusters

    drives = _set_probe_state(aut, food)
    clusters = _encode_current_clusters(encoder, agent_id, aut.executor)
    with_c = _picks_eat(aut, agent_id, available, drives, clusters or None, eat_tool)
    without_c = _picks_eat(aut, agent_id, available, drives, None, eat_tool)
    return {"with": with_c, "without": without_c, "marginal": (with_c and not without_c)}


def _config_fingerprint(aut: Any) -> dict[str, Any]:
    """The config surface that governs substrate selection + clustering (real attribute names —
    a prior draft read two that don't exist and silently recorded 0.0, a vacuous guard). -1.0
    sentinel makes a missing field obvious rather than a plausible value."""
    from maxim.similarity.encoder import SensorEncoderConfig

    nc = aut.bio.nac.config
    ec_cfg = getattr(getattr(aut.bio, "ec", None), "config", None)
    return {
        "substrate_explore_bonus_weight": float(getattr(nc, "substrate_explore_bonus_weight", -1.0)),
        "max_cluster_reward_bias": float(getattr(nc, "max_cluster_reward_bias", -1.0)),
        "max_reward_bias": float(getattr(nc, "max_reward_bias", -1.0)),
        "reward_bias_decay_tau": float(getattr(nc, "reward_bias_decay_tau", -1.0)),
        "cluster_reward_bias_decay_tau": float(getattr(nc, "cluster_reward_bias_decay_tau", -1.0)),
        "cluster_bias_wall_decay_half_life_s": float(getattr(nc, "cluster_bias_wall_decay_half_life_s", -1.0)),
        "ec_pattern_complete_threshold": float(getattr(ec_cfg, "pattern_complete_threshold", -1.0)) if ec_cfg else -1.0,
        "encoder_pattern_threshold": float(SensorEncoderConfig().pattern_threshold),
    }


def _assert_frozen_apparatus(fp: dict[str, Any]) -> None:
    """Refuse on a diverged apparatus BEFORE it can silently change the result (the n-ctx-drift
    lesson). Absolute checks against FROZEN_CONFIG + the credit-routing env var the prereg says
    must be unset (MAXIM_OPERANT_ONLY_CREDIT reroutes credit off the interoception cluster)."""
    from maxim.prompts.cluster_bias_annotation import annotation_disabled_via_env

    if annotation_disabled_via_env(os.environ.get("MAXIM_OPERANT_ONLY_CREDIT")):
        raise RuntimeError(
            "MAXIM_OPERANT_ONLY_CREDIT is set — it reroutes credit off the interoception cluster and "
            "kills the floor; unset it and re-run (this is the game-native rung-1 apparatus)."
        )
    for k, want in FROZEN_CONFIG.items():
        got = fp.get(k)
        if got is None or abs(float(got) - want) > 1e-9:
            raise RuntimeError(
                f"frozen-apparatus divergence: {k}={got} (frozen {want}) — ambient ~/.maxim config or "
                "env changed the substrate/clustering surface; restore defaults and re-run."
            )


# --------------------------------------------------------------------------- one arm


def _build_aut(agent_id: str, home: str, bridge_host: str, bridge_port: int) -> Any:
    """Pre-build the client with confirm+retry (the one-client bridge rejects 'bridge busy' under
    churn — exp56's build_bench_session pattern; a bare connect() crashed a campaign ~88% in)."""
    from maxim.simulation.minecraft import MinecraftClient
    from maxim.simulation.minecraft_harness import build_minecraft_aut

    client = MinecraftClient(bridge_host, bridge_port)
    client.connect(confirm_timeout_s=4.0, retries=8, backoff_s=0.5)
    return build_minecraft_aut(
        agent_id=agent_id,
        bridge_port=bridge_port,
        bridge_host=bridge_host,
        persistence_dir=home,
        entity_ref=FROZEN["body_ref"],
        client=client,
    )


def _run_arm(
    *,
    arm: str,
    seed: int,
    bridge_host: str,
    bridge_port: int,
    rcon: Any,
    username: str,
    drain_timeout_s: float,
    target_episodes: int | None,
) -> dict[str, Any]:
    """Fresh substrate -> cold-prior pre-check -> train (to plateau if target_episodes is None, else
    exactly target_episodes) -> post marginal-probe. Returns the seed record incl. validity flags."""
    from maxim.agents.context_pool import ContextPool
    from maxim.runtime.agent_loop import _encode_current_clusters, _read_drive_states
    from maxim.runtime.tool_dispatch import read_learning_side_effects, record_outcome
    from maxim.tools.introspection import INTROSPECTION_TOOL_NAMES

    agent_id = f"r2lb_{arm}_{seed}"
    home = tempfile.mkdtemp(prefix=f"r2lb_{arm}_{seed}_")
    r: dict[str, Any] = {"arm": arm, "seed": seed, "agent_id": agent_id}
    aut = None
    try:
        aut = _build_aut(agent_id, home, bridge_host, bridge_port)
        fp = _config_fingerprint(aut)
        _assert_frozen_apparatus(fp)
        r["config_fingerprint"] = fp
        aut.bio.memory_hub.on_session_start()  # open the hub or nothing persists (D41/D42)
        encoder = _make_encoder(aut)
        available = [t for t in aut.executor.registry.list() if t not in INTROSPECTION_TOOL_NAMES]
        eat_tool = next((t for t in available if t.endswith("_eat")), None)
        if eat_tool is None:
            raise RuntimeError(f"no *_eat tool in roster {available!r}")
        pool: Any = ContextPool()
        recent: list[dict[str, Any]] = []

        # Cold-prior pre-check (no cluster encoding — does not seed the probe-band clusters). The
        # disclosure says the cold prior is non-eat at every probe_food; if not, instrument drift.
        pre = {str(f): _cold_prior_eats(aut, agent_id, available, eat_tool, f) for f in FROZEN["probe_food"]}
        r["pre_cold_eats"] = pre
        r["instrument_drift"] = any(pre.values())

        # TRAIN.
        bias_trace: list[float] = []
        relief_eps = floor_eps = not_selected = drain_stalls = 0
        plateaued = False
        ep = 0
        while True:
            if arm == "satiated":
                _feed_to(aut, rcon, username, floor=FROZEN["satiated_floor"])
                clusters = _encode_current_clusters(encoder, agent_id, aut.executor)
                drives = _read_drive_states(aut.executor)
                _picks_eat(aut, agent_id, available, drives, clusters or None, eat_tool)  # matched cycle
                bias_trace.append(_eat_bias(aut, agent_id, clusters, eat_tool))
            else:
                food_now = _drain_until(
                    aut, rcon, username, target=FROZEN["train_target_food"], timeout_s=drain_timeout_s
                )
                if food_now is None or food_now > FROZEN["train_target_food"]:
                    drain_stalls += 1
                clusters = _encode_current_clusters(encoder, agent_id, aut.executor)
                drives = _read_drive_states(aut.executor)
                if not _picks_eat(aut, agent_id, available, drives, clusters or None, eat_tool):
                    not_selected += 1  # substrate did not choose eat at deep deficit (disclosure: it should)
                out = aut.executor.execute({"tool_name": eat_tool, "params": {}})
                side = read_learning_side_effects(out)
                dpd, dch, dwh = side.drive_potential_diff, side.drive_relief_channel, side.drive_credit_withheld
                if dpd is not None and abs(dpd) > 1e-9:
                    relief_eps += 1
                elif bool(getattr(out, "success", False)):
                    floor_eps += 1  # successful eat, no measured relief -> would book via the tool-success floor
                if arm == "no_credit":
                    dpd, dch, dwh = None, None, True  # HARNESS-LEVEL ABLATION (suppresses relief + floor)
                record_outcome(
                    agent_id=agent_id,
                    tool_name=eat_tool,
                    success=bool(getattr(out, "success", False)),
                    result_summary=str(getattr(out, "output", ""))[:80] or None,
                    error=getattr(out, "error", None),
                    reasoning="r2 learned-bias training",
                    recent_outcomes=recent,
                    max_recent=20,
                    llm_worker=None,
                    context_pool=pool,
                    nac=aut.bio.nac,
                    tool_params={},
                    cluster_id=clusters.get("interoception"),
                    clusters=clusters,
                    embodiment_failed=side.embodiment_failed,
                    drive_potential_diff=dpd,
                    drive_credit_withheld=dwh,
                    drive_relief_channel=dch,
                    outcome_valence=side.outcome_valence,
                )
                bias_trace.append(_eat_bias(aut, agent_id, clusters, eat_tool))

            ep += 1
            if target_episodes is not None:
                if ep >= target_episodes:
                    break
            else:
                w = FROZEN["plateau_window"]
                if len(bias_trace) >= w + 1:
                    win = bias_trace[-w:]
                    base = abs(statistics.mean(win))
                    spread = max(win) - min(win)
                    if base > FROZEN["plateau_floor"] and spread / (base or 1.0) < FROZEN["plateau_frac"]:
                        plateaued = True
                        break
                if ep >= FROZEN["cap_episodes"]:
                    break

        r.update(
            episodes=ep,
            final_eat_bias=(bias_trace[-1] if bias_trace else 0.0),
            relief_episodes=relief_eps,
            floor_episodes=floor_eps,
            not_selected=not_selected,
            drain_stalls=drain_stalls,
            plateaued=(plateaued if target_episodes is None else None),
        )
        # Validity flags (ENFORCED in main for the gated write).
        r["did_not_plateau"] = (arm == "learning") and (not plateaued)
        r["credit_did_not_book"] = (arm == "learning") and abs(r["final_eat_bias"]) < 1e-9
        r["ablation_leaked"] = (arm == "no_credit") and abs(r["final_eat_bias"]) >= 1e-9
        r["drain_stalled"] = (arm != "satiated") and drain_stalls > 0
        r["floor_contaminated"] = (arm == "learning") and floor_eps > 0  # bias must be pure drive-relief

        # POST marginal-probe (frozen learned NAc).
        post = {str(f): _marginal_probe(aut, encoder, agent_id, available, eat_tool, f) for f in FROZEN["probe_food"]}
        r["post_probe"] = post
        r["marginal_flip"] = {str(f): post[str(f)]["marginal"] for f in FROZEN["probe_food"]}
        r["raw_flip"] = {str(f): (post[str(f)]["with"] and not pre[str(f)]) for f in FROZEN["probe_food"]}
    except Exception as exc:  # per-seed isolation: record + continue (don't lose an expensive run)
        r["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if aut is not None:
            try:
                aut.bio.memory_hub.on_session_end()
            except Exception:
                pass
            try:
                aut.client.close()
            except Exception:
                pass
        shutil.rmtree(home, ignore_errors=True)
    return r


# --------------------------------------------------------------------------- stats


def _perm_test(a: list[int], b: list[int], *, iters: int = 20000, seed: int = 12345) -> float:
    """One-sided permutation test: P(mean(a) - mean(b) as-or-more extreme | labels exchangeable)."""
    import random as _r

    if not a or not b:
        return 1.0
    obs = (sum(a) / len(a)) - (sum(b) / len(b))
    pooled = a + b
    na = len(a)
    rng = _r.Random(seed)
    ge = 0
    for _ in range(iters):
        rng.shuffle(pooled)
        diff = (sum(pooled[:na]) / na) - (sum(pooled[na:]) / (len(pooled) - na))
        if diff >= obs:
            ge += 1
    return (ge + 1) / (iters + 1)


_REFUSAL_FLAGS = (
    "error",
    "did_not_plateau",
    "credit_did_not_book",
    "ablation_leaked",
    "drain_stalled",
    "instrument_drift",
    "floor_contaminated",
)


def _seed_refusals(r: dict[str, Any]) -> list[str]:
    return [f"{r['arm']} seed {r['seed']}: {flag}" for flag in _REFUSAL_FLAGS if r.get(flag)]


# --------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="docs/experiments/data/r2_learned_bias.jsonl")
    ap.add_argument("--bridge-host", default="127.0.0.1")
    ap.add_argument("--bridge-port", type=int, default=25567)
    ap.add_argument("--rcon-host", default="127.0.0.1")
    ap.add_argument("--rcon-port", type=int, default=25575)
    ap.add_argument("--rcon-password", required=True)
    ap.add_argument("--username", default="maxim")
    ap.add_argument("--drain-timeout-s", type=float, default=120.0)
    ap.add_argument("--n", type=int, default=FROZEN["N"])
    ap.add_argument("--skip-preflight", action="store_true", help="(debug only) skip the pre-run instrument check")
    ap.add_argument("--write-experiment-results", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args(argv)

    repo_root = SCRIPTS_DIR.parent
    out_path = evidence_out_paths(
        repo_root, [args.out], write_experiment_results=args.write_experiment_results, allow_dirty=args.allow_dirty
    )[0]
    prov = executed_code_provenance(repo_root, "maxim", out_path=out_path, allow_dirty=args.allow_dirty)

    def _arm(arm: str, seed: int, target: int | None) -> dict[str, Any]:
        return _run_arm(
            arm=arm,
            seed=seed,
            bridge_host=args.bridge_host,
            bridge_port=args.bridge_port,
            rcon=rcon,
            username=args.username,
            drain_timeout_s=args.drain_timeout_s,
            target_episodes=target,
        )

    rcon = RconControl(args.rcon_host, args.rcon_port, args.rcon_password)
    rows: list[dict[str, Any]] = []
    fp0: dict[str, Any] | None = None
    out_fh = open(out_path, "w")  # incremental JSONL — a crash keeps completed seeds
    try:
        # PREFLIGHT — verify the loop composes before spending 60 arm-runs (prereg verify-the-instrument).
        if not args.skip_preflight:
            pf = _arm("learning", -1, 3)  # short real-path learning run
            pf_bad = [
                f
                for f in ("error", "credit_did_not_book", "drain_stalled", "instrument_drift", "floor_contaminated")
                if pf.get(f)
            ]
            out_fh.write(json.dumps({"preflight": pf}) + "\n")
            out_fh.flush()
            if pf_bad:
                print(f"PREFLIGHT FAILED: {pf_bad} — {pf.get('error', '')}\nInstrument not ready; refusing.")
                return 4

        # Interleave arms WITHIN each seed (drift can't alias onto arm); LEARNING first, controls
        # cycle-matched to its plateau length.
        for seed in range(args.n):
            learn = _arm("learning", seed, None)
            match_ep = int(learn.get("episodes") or FROZEN["cap_episodes"])
            nc = _arm("no_credit", seed, match_ep)
            sat = _arm("satiated", seed, match_ep)
            for r in (learn, nc, sat):
                rows.append(r)
                out_fh.write(json.dumps({"seed_record": r}) + "\n")
                out_fh.flush()
                if fp0 is None and "config_fingerprint" in r:
                    fp0 = r["config_fingerprint"]
                print(
                    f"[{r['arm']} seed {seed}] mflip={r.get('marginal_flip')} bias={r.get('final_eat_bias')} "
                    f"ep={r.get('episodes')} relief={r.get('relief_episodes')} floor={r.get('floor_episodes')} "
                    + " ".join(f.upper() for f in _REFUSAL_FLAGS if r.get(f))
                )
    finally:
        rcon.close()

    # ENFORCED refusals (the prereg's anti-vacuity, enforced not just recorded).
    refusals: list[str] = [msg for r in rows for msg in _seed_refusals(r)]
    fp_drift = sum(1 for r in rows if r.get("config_fingerprint") and r["config_fingerprint"] != fp0)
    if fp_drift:
        refusals.append(f"config_fingerprint drift across {fp_drift} arm-runs")

    def mflips(arm: str, f: int) -> list[int]:
        return [1 if r["marginal_flip"][str(f)] else 0 for r in rows if r["arm"] == arm and "marginal_flip" in r]

    verdict: dict[str, Any] = {"per_state": {}}
    held_any = False
    for f in FROZEN["probe_food"]:
        L, NC, S = mflips("learning", f), mflips("no_credit", f), mflips("satiated", f)
        fl_L = statistics.mean(L) if L else 0.0
        p_nc, p_sat = _perm_test(L, NC), _perm_test(L, S)
        held = (fl_L >= FROZEN["M"]) and (p_nc < 0.05) and (p_sat < 0.05)
        held_any = held_any or held
        verdict["per_state"][str(f)] = {
            "mflipfrac_learning": fl_L,
            "mflipfrac_no_credit": statistics.mean(NC) if NC else 0.0,
            "mflipfrac_satiated": statistics.mean(S) if S else 0.0,
            "p_vs_no_credit": p_nc,
            "p_vs_satiated": p_sat,
            "held": held,
        }
    verdict["premise_held"] = held_any and not refusals
    verdict["status"] = "REFUSED-UNVERIFIED" if refusals else ("PREMISE-HELD" if held_any else "PREMISE-NULL")
    verdict["refusals"] = refusals

    record = {
        "ts": time.time(),
        "rung": "R2-learned-bias",
        "frozen": FROZEN,
        "frozen_config": FROZEN_CONFIG,
        "provenance": prov,
        "verdict": verdict,
        "config_fingerprint": fp0,
    }
    out_fh.write(json.dumps({"summary": record}) + "\n")
    out_fh.close()

    print(f"\n=== {verdict['status']} ===")
    for f, s in verdict["per_state"].items():
        print(
            f"  food {f}: L={s['mflipfrac_learning']:.2f} NC={s['mflipfrac_no_credit']:.2f} "
            f"S={s['mflipfrac_satiated']:.2f} p_nc={s['p_vs_no_credit']:.3f} p_sat={s['p_vs_satiated']:.3f} "
            f"{'HELD' if s['held'] else '-'}"
        )
    if refusals:
        print(f"REFUSED ({len(refusals)}): " + "; ".join(refusals[:8]) + (" …" if len(refusals) > 8 else ""))
    print(f"written: {out_path}")
    return 3 if refusals else 0


if __name__ == "__main__":
    raise SystemExit(main())
