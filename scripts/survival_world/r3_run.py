"""R3 — the lethal-window survival benchmark (1.3 Phase 3): `cal`, `bench`, `report`.

Pre-registration: docs/experiments/r3_survival_benchmark_prereg.md (v3.1). R3 is an INSTRUMENT plus
a FROZEN BASELINE, never a graduated claim. The unit is ONE unrescued submersion per fresh agent
(`WaterTrial.lethal_event`); the arms are declared by learning channel; the primary DVs are the COST of
the event — time to air and per-drive pain-seconds — with survival a declared ceiling and
`escaped_before_damage` a route-order flag.

    cal     the ONE verification cell: the innate-only floor arm, n = 12, at the Exp 60 depth → the
            gauntlet file (`r3_gauntlet.json`): the floor's distributions, the reservoir band, the
            loop's tick-period band, the world roster, the code hash — what `bench` refuses drift on.
    bench   the five arms, interleaved by seed, one fresh agent per row, at ONE code hash.
    report  pure over the rows: per-arm distributions with intervals, contrasts named by arm with
            the predicted mechanism beside each, INCOMPLETE causes named, nothing graduated.

Arms (prereg §Arms): A innate-only (fresh, fear subscriber DETACHED), B in-situ learner (fresh,
ATTACHED), C self-learned (Exp 60 FEAR training, no post probe), D shared (Exp 61 lifecycle: a
C-protocol donor exported, a fresh receiver ingests, reboots, loop-OFF gate), E exposed-ablated
(Exp 60 ABLATED training, DETACHED). Donor and receiver seams are Exp 61's (`exp61_run`), verbatim.

Apparatus: the water classroom (`setup_world water_classroom`), the DATED apparatus re-check
(`exp60_water_apparatus_2026-09-17.json`) and the re-stamped anchor. Rig: big-mac-mini, bridge at
`--state_interval_ms=100` started with the flee anchor at the shore, no second player, no rain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _provenance import (  # noqa: E402
    DirtyTreeError,
    ProvenanceError,
    evidence_out_paths_or_exit,
    in_process_code_provenance,
)
from exp56 import common as C  # noqa: E402
from survival_world.common import InstrumentError  # noqa: E402
from survival_world.exp60_run import ANCHOR_FILE, GATE_RECORD  # noqa: E402
from survival_world.exp60_run import FROZEN as FROZEN60  # noqa: E402
from survival_world.exp61_run import (  # noqa: E402
    BODY_REF,
    Exp61Campaign,
    build_aut,
    close_and_stage,
    decision_decisive,
    ingest_gate,
    load_json,
    rss_mb,
    wilson_interval,
    world_ids,
)
from survival_world.water_trial import (  # noqa: E402
    R3_GAMERULES,
    Refusal,
    WaterTrial,
    _detach_fear_subscriber,
    _median,
    min_pain_edge_s,
)

APPARATUS_RECORD = "docs/experiments/data/exp60_water_apparatus_2026-09-17.json"  # the DATED re-check
GAUNTLET_DEFAULT = "docs/experiments/data/r3_gauntlet.json"
ARMS: tuple[str, ...] = ("A_innate_only", "B_in_situ", "C_self_learned", "D_shared", "E_exposed_ablated")
DETACHED = {"A_innate_only", "E_exposed_ablated"}
TRAINED = {"C_self_learned": "fear", "E_exposed_ablated": "ablated"}
ESCAPE_SUFFIX = "_escape_water"

FROZEN: dict[str, Any] = {
    "version": "v3.1",
    "arms": {arm: 12 for arm in ARMS},
    "seeds": {arm: list(range(300 + 20 * i, 300 + 20 * i + 12)) for i, arm in enumerate(ARMS)},
    "cal": {"arm": "A_innate_only", "n": 12, "seeds": list(range(400, 412))},
    "depth": 5,
    "lethal_cap_s": 45.0,
    "hold_hz": 4.0,
    "state_age_max_s": 0.15,
    "gamerules": R3_GAMERULES,
    "settle_guard": {"is_raining": 0.0, "nearest_player_dist": 64.0, "hostile_count": 0.0},
    "reservoir_tolerance": 1.0,  # foodSaturationLevel / foodLevel band half-width around the cal band
    "tick_period_iqr_mult": 2.0,  # bench refuses a row whose tick-period median leaves cal's median ± mult·IQR
    "foreign_fear_discount": 0.75,
    "read_floor": 0.5,
    "exp60_frozen_sha256": hashlib.sha256(json.dumps(FROZEN60, sort_keys=True, default=str).encode()).hexdigest(),
}


# ── pure numerics ──


def bootstrap_median_ci(xs: list[float], *, n_boot: int = 2000, seed: int = 0) -> tuple[float, float] | None:
    if len(xs) < 2:
        return None
    rng = random.Random(seed)
    meds = sorted(_median([rng.choice(xs) for _ in xs]) for _ in range(n_boot))
    return (round(meds[int(0.025 * n_boot)], 3), round(meds[int(0.975 * n_boot) - 1], 3))


def mann_whitney_p(a: list[float], b: list[float]) -> float | None:
    """Two-sided Mann–Whitney U (scipy); None when either side is empty."""
    if not a or not b:
        return None
    from scipy.stats import mannwhitneyu

    return float(mannwhitneyu(a, b, alternative="two-sided").pvalue)


def idle_tick_period_median_s(event: dict[str, Any]) -> float | None:
    """The loop's IDLE cadence: the median period between consecutive ticks where NEITHER proposes,
    over the event's whole telemetry (pure over `event.ticks`). The harness's in-window
    `tick_period_median_s` degenerates to 1–2 periods on a ≈ 3 s event — one of them always the
    `flee` tie-break dispatch (≈ 0.77 s in every arm) — so it is not a cadence there (bio-faithful
    lens on the amendments, 2026-09-18); this one is."""
    ticks = [t for t in (event.get("ticks") or []) if "t" in t]
    periods = [
        b["t"] - a["t"]
        for a, b in zip(ticks, ticks[1:])
        if not a.get("proposal") and not b.get("proposal") and b["t"] > a["t"]
    ]
    return round(_median(periods), 3) if periods else None


def _dist(xs: list[float]) -> dict[str, Any]:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return {"n": 0}
    return {
        "n": len(xs),
        "min": round(xs[0], 3),
        "median": round(_median(xs), 3),
        "max": round(xs[-1], 3),
        "median_ci95": bootstrap_median_ci(xs),
    }


def report(
    rows: list[dict[str, Any]],
    *,
    campaign_id: str | None = None,
    gauntlet: dict[str, Any] | None = None,
    hash_rule_satisfied_by_ancestry: bool = False,
) -> dict[str, Any]:
    """Pure over the rows. Reported with intervals; nothing gated except completeness (and, with the
    gauntlet given, D5: bench rows at a hash other than the gauntlet's read INCOMPLETE — unless the
    caller has established Amendment 1's ancestry + unchanged-harness precondition and says so)."""
    if campaign_id:
        rows = [r for r in rows if r.get("campaign_id") == campaign_id]
    events = [r for r in rows if r.get("kind") == "event"]
    refused = [r for r in events if r.get("refusal")]
    clean: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    for r in events:
        if not r.get("refusal") and r.get("arm") in clean:
            clean[r["arm"]].append(r)
    hashes = sorted({str((r.get("provenance") or {}).get("executed_git_hash")) for r in rows})
    arms: dict[str, Any] = {}
    for arm, rs in clean.items():
        ev = [r["event"] for r in rs]
        k = sum(1 for e in ev if e.get("survived"))
        arms[arm] = {
            "n": len(rs),
            "survived": k,
            "survived_wilson95": list(wilson_interval(k, len(rs))) if rs else None,
            "escaped_before_damage": sum(1 for e in ev if e.get("escaped_before_damage")),
            "t_surface": _dist([e.get("t_surface") for e in ev if e.get("t_surface") is not None]),
            "censored": sum(1 for e in ev if e.get("t_surface") is None),
            "oxygen_pain_s": _dist([(e.get("pain_seconds") or {}).get("oxygen") for e in ev]),
            "health_pain_s": _dist([(e.get("pain_seconds") or {}).get("health") for e in ev]),
            "health_lost": _dist([e.get("health_lost") for e in ev]),
            "drive_decisive": sum(1 for r in rs if r.get("decisive")),
            # in-window median: 1–2 periods on a short event, dominated by the flee tie-break — NOT a cadence
            "tick_period_median_s_in_window": _dist([e.get("tick_period_median_s") for e in ev]),
            # the loop's idle cadence, the covariate that IS comparable across arms
            "idle_tick_period_median_s": _dist([idle_tick_period_median_s(e) for e in ev]),
        }

    def ts(arm: str) -> list[float]:
        return [r["event"]["t_surface"] for r in clean[arm] if r["event"].get("t_surface") is not None]

    def pain(arm: str) -> list[float]:
        xs = [(r["event"].get("pain_seconds") or {}).get("oxygen") for r in clean[arm]]
        return [x for x in xs if x is not None]  # a missing drive spec is not zero pain

    contrasts = {}
    for name, hi, lo, mech in (
        ("C_minus_A", "C_self_learned", "A_innate_only", "carried fear vs the innate health reflex"),
        ("B_minus_A", "B_in_situ", "A_innate_only", "in-situ Wire-4 acquisition vs the innate reflex"),
        (
            "C_minus_B",
            "C_self_learned",
            "B_in_situ",
            "carrying the fear vs learning it there (the pain-free descent to the oxygen-12 publish)",
        ),
        ("C_minus_E", "C_self_learned", "E_exposed_ablated", "the drive vs the exposure without it"),
        (
            "D_beside_C",
            "D_shared",
            "C_self_learned",
            "the discounted vicarious fear beside the self-learned one — reported, never a contrast",
        ),
    ):
        a, b = ts(hi), ts(lo)
        contrasts[name] = {
            "mechanism": mech,
            "t_surface_median_diff_s": round(_median(a) - _median(b), 3) if a and b else None,
            "t_surface_mann_whitney_p": mann_whitney_p(a, b),
            "oxygen_pain_median_diff_s": round(_median(pain(hi)) - _median(pain(lo)), 3)
            if pain(hi) and pain(lo)
            else None,
            "censored": {hi: arms[hi]["censored"], lo: arms[lo]["censored"]},
        }
    incomplete: list[str] = []
    for arm, n_need in FROZEN["arms"].items():
        if arms[arm]["n"] < n_need:
            incomplete.append(f"{arm}: {arms[arm]['n']} clean rows < {n_need}")
    if len(hashes) > 1:
        incomplete.append(f"rows span {len(hashes)} code hashes: {hashes}")
    if not any(r.get("kind") == "apparatus" and not r.get("refusal") for r in rows):
        incomplete.append("no clean apparatus row")
    if gauntlet is not None:
        off = [h for h in hashes if h != str(gauntlet.get("cal_code_hash"))]
        if off and not hash_rule_satisfied_by_ancestry:
            incomplete.append(f"rows at hash(es) {off} other than the gauntlet's {gauntlet.get('cal_code_hash')}")
        dirty = [r for r in events if not r.get("refusal") and r.get("dirty")]
        if dirty:
            incomplete.append(f"{len(dirty)} row(s) dirty (a settle-guard breach after the DV) — reported, not counted")
    # anti-vacuity: the floor arm's executed escape is drive-decisive with NO fear; the in-situ arm's WITH fear
    for arm in ("A_innate_only", "B_in_situ"):
        if clean[arm] and not any(r.get("decisive") for r in clean[arm]):
            incomplete.append(f"{arm}: no drive-decisive executed escape read (anti-vacuity)")
    return {
        "_format_version": "1.0",
        "kind": "r3_report",
        "campaign_id": campaign_id,
        "frozen_version": FROZEN["version"],
        "refused": [{"arm": r.get("arm"), "seed": r.get("seed"), "refusal": r.get("refusal")} for r in refused],
        "code_hashes": hashes,
        "arms": arms,
        "contrasts": contrasts,
        "status": "INCOMPLETE" if incomplete else "COMPLETE",
        "incomplete_cause": "; ".join(incomplete) or None,
        "what_this_is": "an instrument and a frozen baseline; nothing here is graduated",
    }


# ── the gauntlet file ──


def write_gauntlet(
    cal_rows: list[dict[str, Any]], apparatus_row: dict[str, Any], *, campaign_id: str
) -> dict[str, Any]:
    clean = [
        r
        for r in cal_rows
        if r.get("kind") == "event" and not r.get("refusal") and r.get("arm") == FROZEN["cal"]["arm"]
    ]
    if not clean:
        raise SystemExit("[FAIL] no clean calibration row (the floor arm) — no gauntlet")
    ev = [r["event"] for r in clean]
    sat = [e["food_at_teleport"]["foodSaturationLevel"] for e in ev]
    food = [e["food_at_teleport"]["foodLevel"] for e in ev]
    tp = [e["tick_period_median_s"] for e in ev if e.get("tick_period_median_s") is not None]
    iqr = [e["tick_period_iqr_s"] for e in ev if e.get("tick_period_iqr_s") is not None]
    hashes = sorted({str(r["provenance"]["executed_git_hash"]) for r in clean})
    if len(hashes) != 1:
        raise SystemExit(f"[FAIL] calibration rows span hashes {hashes}")
    return {
        "_format_version": "1.0",
        "kind": "r3_gauntlet",
        "campaign_id": campaign_id,
        "frozen": {k: v for k, v in FROZEN.items() if k != "seeds"},
        "depth": FROZEN["depth"],
        "cal_code_hash": hashes[0],
        "cal_rows_sha256": hashlib.sha256(json.dumps(clean, sort_keys=True, default=str).encode()).hexdigest(),
        "n_cal": len(clean),
        "apparatus_record": APPARATUS_RECORD,
        "apparatus_record_ts": clean[0].get("apparatus_record_ts"),
        "anchor_measured": clean[0].get("anchor_measured"),
        "gamerules": [list(x) for x in R3_GAMERULES],
        "flee_preflight": apparatus_row.get("flee_preflight"),
        "reservoir_band": {
            "foodSaturationLevel": [min(sat) - FROZEN["reservoir_tolerance"], max(sat) + FROZEN["reservoir_tolerance"]],
            "foodLevel": [min(food) - FROZEN["reservoir_tolerance"], max(food) + FROZEN["reservoir_tolerance"]],
        },
        "tick_period_band_s": [
            round(_median(tp) - FROZEN["tick_period_iqr_mult"] * _median(iqr), 3),
            round(_median(tp) + FROZEN["tick_period_iqr_mult"] * _median(iqr), 3),
        ]
        if tp and iqr
        else None,  # never a zero-width band; an absent band is DRIFT for every bench row
        "floor_arm": {
            "survived": sum(1 for e in ev if e.get("survived")),
            "t_surface": _dist([e.get("t_surface") for e in ev if e.get("t_surface") is not None]),
            "health_lost": _dist([e.get("health_lost") for e in ev]),
            "oxygen_pain_s": _dist([(e.get("pain_seconds") or {}).get("oxygen") for e in ev]),
        },
        "written_at": time.time(),
    }


def gauntlet_drift(g: dict[str, Any], row: dict[str, Any]) -> list[str]:
    """Why a bench row does not belong to this gauntlet (empty = it does)."""
    why: list[str] = []
    ev = row.get("event") or {}
    for key in ("reservoir_band", "tick_period_band_s", "apparatus_record_ts", "anchor_measured", "cal_code_hash"):
        if not g.get(key):
            why.append(f"the gauntlet carries no {key} — not a frozen gauntlet")
    if row.get("apparatus_record_ts") != g.get("apparatus_record_ts"):
        why.append("apparatus record differs from the gauntlet's")
    if row.get("anchor_measured") != g.get("anchor_measured"):
        why.append("anchor measured edges differ from the gauntlet's")
    if (row.get("frozen") or {}).get("exp60_frozen_sha256") != (g.get("frozen") or {}).get("exp60_frozen_sha256"):
        why.append("Exp 60 FROZEN constants differ from the gauntlet's")
    food = ev.get("food_at_teleport") or {}
    for key, band in (g.get("reservoir_band") or {}).items():
        v = food.get(key)
        if v is None or not (band[0] <= float(v) <= band[1]):
            why.append(f"reservoir {key}={v} outside the gauntlet band {band}")
    band = g.get("tick_period_band_s")
    tp = ev.get("tick_period_median_s")
    if band and (tp is None or not (band[0] <= float(tp) <= band[1])):
        why.append(f"tick period median {tp} outside the gauntlet band {band}")
    if ev.get("max_state_age_s") is not None and float(ev["max_state_age_s"]) > FROZEN["state_age_max_s"]:
        why.append(f"a sample was stale ({ev['max_state_age_s']} s > {FROZEN['state_age_max_s']})")
    return why


def validate_gauntlet(g: dict[str, Any]) -> list[str]:
    """The gauntlet file must be THIS harness's: kind, frozen constants (R3's own, not only Exp 60's), bands."""
    why: list[str] = []
    if g.get("kind") != "r3_gauntlet":
        why.append(f"kind {g.get('kind')!r} is not r3_gauntlet")
    gf = g.get("frozen") or {}
    for key in (
        "version",
        "depth",
        "lethal_cap_s",
        "hold_hz",
        "state_age_max_s",
        "settle_guard",
        "exp60_frozen_sha256",
        "foreign_fear_discount",
        "read_floor",
    ):
        if gf.get(key) != FROZEN[key]:
            why.append(f"frozen {key}: gauntlet {gf.get(key)!r} != running {FROZEN[key]!r}")
    if [list(x) for x in R3_GAMERULES] != [list(x) for x in (g.get("gamerules") or [])]:
        why.append("gamerule roster differs from the running R3_GAMERULES")
    for key in ("reservoir_band", "tick_period_band_s", "apparatus_record_ts", "anchor_measured", "cal_code_hash"):
        if not g.get(key):
            why.append(f"no {key}")
    return why


# ── Amendments (2026-09-18, post-data, INSTRUMENT-ONLY; prereg §Amendments) ──

TICK_BAND_REFUSAL = "gauntlet drift: tick period median"
# behaviour-bearing TRACKED inputs: harness, runtime, tests, the dependency manifest, data/ (util +
# robots.yaml), scenarios/. Not governed by any git rule — the bridge, the Minecraft server, the venv;
# only the bridge cadence check and the fingerprint guard those.
HARNESS_PATHS = ("scripts/", "src/", "tests/", "pyproject.toml", "data/", "scenarios/")


def harness_unchanged_between(cal_hash: str, bench_hash: str) -> tuple[bool, list[str]]:
    """Amendment 1's real precondition: the calibration hash is an ANCESTOR of the bench hash AND no
    behaviour-bearing tracked file changed between them (the frozen wording "rows at a hash other
    than the gauntlet's" was structurally unsatisfiable — the calibration PR itself advances main).
    Fail-closed: any git failure or a non-ancestor is False."""
    try:
        anc = subprocess.run(
            ["git", "merge-base", "--is-ancestor", cal_hash, bench_hash],
            cwd=C.REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if anc.returncode != 0:
            return False, [f"{cal_hash} is not an ancestor of {bench_hash}"]
        out = subprocess.run(
            ["git", "diff", "--name-only", cal_hash, bench_hash],
            cwd=C.REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, [f"git failed: {exc}"]
    if out.returncode != 0:
        return False, [f"git diff failed: {out.stderr.strip()[:120]}"]
    touched = [p for p in out.stdout.splitlines() if p.strip() and p.startswith(HARNESS_PATHS)]
    return not touched, touched


def _band_hi(core: str) -> float | None:
    """The upper edge of the band quoted in a tick-band refusal ('... outside the gauntlet band [lo, hi]')."""
    try:
        return float(core.split("[", 1)[1].split("]", 1)[0].split(",")[1])
    except (IndexError, ValueError):
        return None


def reclassify_under_amendments(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Amendment 2: the loop tick-period band was frozen on 46-period floor-arm windows; a ≈3 s
    escape window holds ONE or TWO periods, one of them always the `flee` tie-break dispatch
    (0.70–0.79 s in every arm), so the in-window "median" degenerates to that interval and the band
    refuses rows by tick PHASE, not cadence (idle cadence is arm-invariant, ±15 ms). A row whose ONLY
    refusal is that band, whose median is numeric and ABOVE the band (a below-band or missing median
    is the regression signature the band was frozen for), whose event is COMPLETE (a surface or a
    death with its detector read), and for which no LATER clean row exists for the same (arm, seed)
    (the frozen supersede rule wins) is counted, with the reason recorded on the row; the covariate
    the report carries is the idle cadence. Every other refusal (a stale sample, a guard breach, a
    non-decisive escape, drift on any other field) stands. Pure; returns (rows, the rows re-counted)."""
    out: list[dict[str, Any]] = []
    recounted: list[dict[str, Any]] = []
    later_clean: dict[tuple[str, int], float] = {}
    for r in rows:
        if r.get("kind") == "event" and not r.get("refusal") and r.get("seed") is not None:
            key = (str(r.get("arm")), int(r["seed"]))
            later_clean[key] = max(later_clean.get(key, 0.0), float(r.get("ts") or 0.0))
    for r in rows:
        ref = str(r.get("refusal") or "")
        core = ref.removeprefix("Refusal: ")
        ev = r.get("event") or {}
        tp = ev.get("tick_period_median_s")
        hi = _band_hi(core) if core.startswith(TICK_BAND_REFUSAL) else None
        key = (str(r.get("arm")), int(r["seed"])) if r.get("seed") is not None else None
        superseded = key in later_clean and later_clean[key] > float(r.get("ts") or 0.0)
        if (
            r.get("kind") == "event"
            and core.startswith(TICK_BAND_REFUSAL)
            and ";" not in core  # ONLY the band (gauntlet_drift joins reasons with '; ')
            and ev.get("end") in ("surface", "death")
            and isinstance(tp, (int, float))
            and hi is not None
            and float(tp) > hi
            and not superseded
        ):
            r2 = dict(r)
            r2["refusal"] = None
            r2["amended"] = {
                "amendment": 2,
                "original_refusal": ref,
                "tick_period_median_s": ev.get("tick_period_median_s"),
            }
            out.append(r2)
            recounted.append(r2)
        else:
            out.append(r)
    return out, recounted


def _is_ancestor_of_main(sha: str) -> bool:
    try:
        subprocess.run(["git", "fetch", "-q", "origin", "main"], cwd=C.REPO_ROOT, check=False, timeout=60)
        rc = subprocess.run(["git", "merge-base", "--is-ancestor", sha, "origin/main"], cwd=C.REPO_ROOT).returncode
        return rc == 0
    except (OSError, subprocess.SubprocessError):
        return False


# ── the live campaign ──


class _R3:
    def __init__(
        self, args: argparse.Namespace, *, provenance: dict[str, Any], out_path: Path, campaign_id: str
    ) -> None:
        self.args = args
        self.provenance = provenance
        self.out_path = out_path
        self.campaign_id = campaign_id
        apparatus = load_json(C.REPO_ROOT / APPARATUS_RECORD)
        if not apparatus.get("all_pass"):
            raise SystemExit(f"[FAIL] {APPARATUS_RECORD} does not carry all_pass")
        self.geom = load_json(ANCHOR_FILE)
        if not (self.geom.get("measured") or {}).get("t_damage_onset_min_s"):
            raise SystemExit("[FAIL] the anchor record carries no measured edges — run exp60_water_check first")
        if int(self.geom.get("depth", -1)) != FROZEN["depth"]:
            raise SystemExit(f"[FAIL] classroom depth {self.geom.get('depth')} != frozen {FROZEN['depth']}")
        pain_edge_min = min_pain_edge_s(apparatus)
        if pain_edge_min is None:
            raise SystemExit("[FAIL] apparatus record carries no measured t_pain_edge")
        self.pain_edge_min = pain_edge_min
        self.probe_cap_s = pain_edge_min - FROZEN60["probe_cap_margin_s"]
        self.train_cap_s = float(self.geom["measured"]["t_damage_onset_min_s"]) - FROZEN60["train_cap_margin_s"]
        self.apparatus_ts = apparatus.get("ts")
        self.rcon = C.RconControl(args.rcon_host, args.rcon_port, args.rcon_password)
        self.workdir = Path(args.workdir).expanduser().resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)
        self._exp61: Any = None
        self.gauntlet: dict[str, Any] | None = None

    @property
    def exp61(self) -> Any:
        """Exp 61's campaign object, for arm D's DONOR (its `donor` writes a `donor` row into our file)."""
        if self._exp61 is None:
            self._exp61 = Exp61Campaign(
                self.args, provenance=self.provenance, out_path=self.out_path, campaign_id=self.campaign_id
            )
        return self._exp61

    def trial(self, aut: Any, encoder: Any, *, agent_id: str, home: Path) -> WaterTrial:
        return WaterTrial(
            aut=aut,
            rcon=self.rcon,
            username=self.args.username,
            geom=self.geom,
            frozen=FROZEN60,
            probe_cap_s=self.probe_cap_s,
            train_cap_s=self.train_cap_s,
            persistence_dir=home,
            agent_id=agent_id,
            encoder=encoder,
            settle_guard=FROZEN["settle_guard"],
        )

    def base_row(self, kind: str, arm: str, seed: int | None) -> dict[str, Any]:
        return {
            "ts": time.time(),
            "kind": kind,
            "campaign_id": self.campaign_id,
            "arm": arm,
            "seed": seed,
            "frozen": FROZEN,
            "probe_cap_s": self.probe_cap_s,
            "train_cap_s": self.train_cap_s,
            "pain_edge_min_s": self.pain_edge_min,
            "provenance": self.provenance,
            "apparatus_record": APPARATUS_RECORD,
            "apparatus_record_ts": self.apparatus_ts,
            "anchor_measured": self.geom.get("measured"),
            "rss_mb": rss_mb(),
            "refusal": None,
        }

    def write(self, row: dict[str, Any]) -> None:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.out_path, "a") as fh:
            fh.write(json.dumps(row, default=str) + "\n")

    # ── rows ──

    def apparatus_row(self) -> dict[str, Any]:
        """Once per campaign start, on a throwaway agent: the Exp 60/61 preflights plus R3's own."""
        row = self.base_row("apparatus", "-", None)
        home = self.workdir / "apparatus"
        shutil.rmtree(home, ignore_errors=True)
        aut, encoder, pump = build_aut(self.args, agent_id="r3_apparatus", home=home)
        trial = self.trial(aut, encoder, agent_id="r3_apparatus", home=home)
        trial.attach_instruments()
        try:
            row["bridge_state_interval_s"] = trial.check_bridge()
            row["loop_liveness_ticks"] = trial.check_liveness()
            trial.check_gamerules(R3_GAMERULES)
            trial.check_surface_cell_air()
            trial.resolve_tools()
            row["flee_preflight"] = trial.check_flee_anchor()
            row["clusters_distinct"] = list(trial.check_clusters_distinct())
            trial.rescue("apparatus")
            trial.submerge("actuation")
            row["actuation"] = trial.check_escape_actuation()
            trial.preflight_deaths_objective()
            row["food_state"] = trial.read_food_state()
        except (Refusal, InstrumentError) as exc:
            row["refusal"] = str(exc)
            row.update(getattr(exc, "partial", None) or {})
        finally:
            trial.detach_instruments()
            try:
                close_and_stage(aut, pump, None)
            except Exception as exc:  # the throwaway's close is not evidence
                print(f"WARNING: apparatus close: {exc!r}")
        self.write(row)
        print(f"apparatus: refusal={row['refusal']} flee={row.get('flee_preflight')} actuation={row.get('actuation')}")
        return row

    def _boundary(self, trial: WaterTrial, aut: Any) -> None:
        """No positive escape link and no reward bias at the gauntlet boundary (every arm)."""
        if trial.positive_escape_links():
            raise Refusal("a positive escape link at the gauntlet boundary")
        nac = aut.bio.nac.dump()
        if nac.get("reward_bias"):
            raise Refusal(f"reward_bias not empty at the boundary ({len(nac['reward_bias'])} keys)")
        if nac.get("links"):
            raise Refusal(f"links not empty at the boundary ({len(nac['links'])})")

    def _finish_event(self, row: dict[str, Any], ev: dict[str, Any], trial: WaterTrial, aut: Any, arm: str) -> None:
        row["event"] = ev
        dec, why = decision_decisive(ev.get("executed_escape_event"), ESCAPE_SUFFIX)
        row["decisive"], row["decisive_why"] = dec, why
        # what "decisive" READS: the aggregate `drive` score component (> 0) with causal 0 and learned 0 on
        # the executed escape's NAc_RECOMMEND event — the event carries no per-need breakdown (no named
        # `threat` term), so the route is named by the arm's declared channel + this read, not by a term
        row["decisive_read"] = "aggregate drive component > 0, causal 0, learned 0 (no per-need term on the event)"
        cut = (
            ev.get("t_surface")
            if ev.get("t_surface") is not None
            else (ev.get("t_death") if ev.get("t_death") is not None else ev.get("t_end"))
        )
        breach = ev.get("guard_breach")
        if breach is not None:
            if breach["t"] < (cut or 0.0):
                raise Refusal(f"settle guard breached before the DV: {breach}")
            row["dirty"] = breach  # after the DV: reported, never counted
        if ev.get("max_state_age_s") is not None and float(ev["max_state_age_s"]) > FROZEN["state_age_max_s"]:
            raise Refusal(
                f"a sample was stale ({ev['max_state_age_s']} s > {FROZEN['state_age_max_s']}) — cal and bench alike"
            )
        row["fear_after"] = trial.fear_dump()
        row["positive_escape_links_after"] = trial.positive_escape_links()
        escaped = any(str(c["tool"]).endswith(ESCAPE_SUFFIX) and not c.get("post_event") for c in ev["calls"])
        if escaped and not dec:
            raise Refusal(f"the executed escape was not drive-decisive: {why}")
        if ev["end"] == "surface" and not escaped:
            raise Refusal("surfaced with no executed escape inside the window (a teleport artefact?)")
        if arm in DETACHED and row["fear_after"]:
            raise Refusal(f"fear booked on a DETACHED arm: {row['fear_after']}")
        if arm == "B_in_situ" and ev["end"] == "surface" and not row["fear_after"]:
            raise Refusal("the in-situ learner surfaced without booking fear (which route fired?)")
        if self.gauntlet is not None:
            drift = gauntlet_drift(self.gauntlet, row)
            if drift:
                raise Refusal("gauntlet drift: " + "; ".join(drift))

    def event_row(self, arm: str, seed: int) -> dict[str, Any]:
        """Arms A, B, C, E: one fresh agent, its protocol, then the lethal event, in one object."""
        row = self.base_row("event", arm, seed)
        agent_id = f"r3_{arm}_{seed}"
        home = self.workdir / arm / f"agent_{seed}"
        stage = self.workdir / arm / f"stage_{seed}"
        shutil.rmtree(home, ignore_errors=True)
        shutil.rmtree(stage, ignore_errors=True)
        aut, encoder, pump = build_aut(self.args, agent_id=agent_id, home=home)
        row["subscriber_detached"] = arm in DETACHED
        row["detached_count"] = _detach_fear_subscriber(aut) if arm in DETACHED else 0
        trial = self.trial(aut, encoder, agent_id=agent_id, home=home)
        trial.attach_instruments()
        try:
            row["fingerprint_live"] = trial.check_fingerprint(FROZEN60["usable_oxygen_max"])
            row["bridge_state_interval_s"] = trial.check_bridge()
            row["loop_liveness_ticks"] = trial.check_liveness()
            if trial.calls:
                raise Refusal(f"executor call(s) during the shore liveness window: {[c['tool'] for c in trial.calls]}")
            trial.check_gamerules(R3_GAMERULES)
            trial.check_surface_cell_air()
            trial.resolve_tools()
            row["flee_preflight"] = trial.check_flee_anchor()  # per row (15 ms live)
            _shore_pre, water_pre = trial.check_clusters_distinct()
            trial.rescue("ready")
            if arm in TRAINED:
                trial.deaths0 = trial.deaths()
                row["training"], episode_clusters = trial.train()
                row.update(trial.live_g2(TRAINED[arm], episode_clusters, water_pre))
            self._boundary(trial, aut)
            row["fear_before"] = trial.fear_dump()
            ev = trial.lethal_event(f"{arm}-{seed}", cap_s=FROZEN["lethal_cap_s"], hold_hz=FROZEN["hold_hz"])
            self._finish_event(row, ev, trial, aut, arm)
        except (Refusal, InstrumentError, OSError, ValueError, ConnectionError) as exc:
            row["refusal"] = f"{type(exc).__name__}: {exc}"
            row.update(getattr(exc, "partial", None) or {})
        finally:
            trial.detach_instruments()
            try:
                close_and_stage(aut, pump, stage)
            except Exception as exc:
                print(f"WARNING: {arm} {seed} close: {exc!r}")
        self.write(row)
        ev = row.get("event") or {}
        print(
            f"{arm} {seed}: end={ev.get('end')} t_surface={ev.get('t_surface')} health_lost={ev.get('health_lost')} "
            f"pain={ev.get('pain_seconds')} decisive={row.get('decisive')} refusal={row['refusal']}"
        )
        return row

    def shared_row(self, seed: int) -> dict[str, Any]:
        """Arm D: a C-protocol donor (Exp 61's `donor`, verbatim) → a fresh receiver: pre-stage, ingest
        through the real CLI, reboot, loop-OFF representation gate (Exp 61 steps 1–4, verbatim) → the
        lethal event in place of Exp 61's first-contact placement."""
        arm = "D_shared"
        pair_dir = self.workdir / arm / f"pair_{seed}"
        pair_dir.mkdir(parents=True, exist_ok=True)
        donor = self.exp61.donor(seed, pair_dir, "fear")
        row = self.base_row("event", arm, seed)
        row["donor_row_refusal"] = donor.get("refusal")
        if donor.get("refusal"):
            row["refusal"] = f"no clean donor: {donor['refusal']}"
            self.write(row)
            return row
        recv_id = f"r3_D_recv_{seed}"
        home = pair_dir / "recv"
        pre_stage = pair_dir / "recv_pre"
        stage_out = pair_dir / "recv_stage"
        for p in (home, pre_stage, stage_out):
            shutil.rmtree(p, ignore_errors=True)
        stage = pair_dir / "donor_fear_stage"
        aut = pump = trial = None
        try:
            donor_meta = load_json(stage / "donor_meta.json")
            aut, encoder, pump = build_aut(self.args, agent_id=recv_id, home=home)
            close_and_stage(aut, pump, pre_stage)
            aut = pump = (
                None  # a refusal before the reboot must NOT re-close the pre-ingest object over the ingested files
            )
            if world_ids(pre_stage / "aut_ec.json") or load_json(pre_stage / "aut_nac.json").get("cluster_fear"):
                raise Refusal("fresh receiver holds world nodes or fear before ingest")
            bundle = pair_dir / "fear.zip"
            row["bundle_sha256"] = hashlib.sha256(bundle.read_bytes()).hexdigest()
            try:
                entry = C.ingest_bundle_into(
                    home,
                    bundle,
                    contributor_id=donor_meta["contributor_id"],
                    receiver_agent_id=recv_id,
                    receiver_body=BODY_REF,
                )
            except (RuntimeError, OSError, ValueError) as exc:
                raise Refusal(f"substrate ingest refused: {exc}") from exc
            row["ingest"] = {
                k: entry.get(k)
                for k in ("fear_rekeyed", "fear_dropped", "fear_below_floor", "fear_discount", "donor_nodes")
            }
            bad = ingest_gate("transferred", entry, shipped=int(donor_meta["fear_shipped"]))
            if bad:
                raise Refusal(f"ingest gate: {bad}")
            aut, encoder, pump = build_aut(self.args, agent_id=recv_id, home=home)  # reboot
            trial = self.trial(aut, encoder, agent_id=recv_id, home=home)
            trial.attach_instruments()
            row["fingerprint_live"] = trial.check_fingerprint(FROZEN60["usable_oxygen_max"])
            row["bridge_state_interval_s"] = trial.check_bridge()
            trial.resolve_tools()
            row["flee_preflight"] = trial.check_flee_anchor()
            self._boundary(trial, aut)
            row["loop_liveness_ticks"] = trial.check_liveness()
            if trial.calls:
                raise Refusal("executor call(s) during the shore liveness window")
            trial.check_gamerules(R3_GAMERULES)
            trial.check_surface_cell_air()
            trial.rescue("shore-tag")
            shore_node = trial.encode_world_cluster()
            n_sig = len(trial.signals)
            trial.submerge("representation-gate")
            water_node = trial.encode_world_cluster()
            trial.rescue("representation-gate")
            if [s for s in trial.signals[n_sig:] if s["failure_mode"] in ("drive:oxygen", "drive:health")]:
                raise Refusal("the representation gate published pain — not US-free")
            need = aut.bio.nac.anticipatory_threat_need(recv_id, {"world": water_node})
            gate = {
                "water_node": water_node,
                "shore_node": shore_node,
                "transferred_node": donor_meta["water_node"],
                "need": need,
                "water_fear": aut.bio.nac.cluster_fear(recv_id, water_node),
                "shore_fear": aut.bio.nac.cluster_fear(recv_id, shore_node),
            }
            row["representation_gate"] = gate
            if water_node == shore_node or water_node != donor_meta["water_node"]:
                raise Refusal("receiver's submerged reading is not the transferred node")
            if not need > FROZEN["read_floor"] or gate["shore_fear"] != 0.0:
                raise Refusal(f"transferred fear not readable as designed: need {need}, shore {gate['shore_fear']}")
            row["fear_before"] = trial.fear_dump()
            ev = trial.lethal_event(f"{arm}-{seed}", cap_s=FROZEN["lethal_cap_s"], hold_hz=FROZEN["hold_hz"])
            self._finish_event(row, ev, trial, aut, arm)
        except (Refusal, InstrumentError, OSError, ValueError) as exc:
            row["refusal"] = str(exc)
            row.update(getattr(exc, "partial", None) or {})
        finally:
            if trial is not None:
                trial.detach_instruments()
            if aut is not None:
                try:
                    close_and_stage(aut, pump, stage_out)
                except Exception as exc:
                    print(f"WARNING: D {seed} close: {exc!r}")
        self.write(row)
        ev = row.get("event") or {}
        print(
            f"{arm} {seed}: end={ev.get('end')} t_surface={ev.get('t_surface')} decisive={row.get('decisive')} refusal={row['refusal']}"
        )
        return row


def _existing_clean(out_path: Path, campaign_id: str) -> set[tuple[str, int]]:
    done: set[tuple[str, int]] = set()
    if not out_path.exists():
        return done
    for ln in out_path.read_text().splitlines():
        if not ln.strip():
            continue
        r = json.loads(ln)
        if r.get("campaign_id") == campaign_id and r.get("kind") == "event" and not r.get("refusal"):
            done.add((str(r.get("arm")), int(r.get("seed"))))
    return done


def _setup(args: argparse.Namespace) -> tuple[_R3, Path, str] | int:
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
    campaign_id = args.campaign_id or uuid.uuid4().hex[:12]
    if args.resume and not args.campaign_id:
        print("[FAIL] --resume requires --campaign-id")
        return 2
    return _R3(args, provenance=provenance, out_path=out_path, campaign_id=campaign_id), out_path, campaign_id


def _cal(args: argparse.Namespace) -> int:
    got = _setup(args)
    if isinstance(got, int):
        return got
    camp, out_path, campaign_id = got
    print(
        f"R3-cal {campaign_id}: {FROZEN['cal']['arm']} × {FROZEN['cal']['n']} at depth {FROZEN['depth']} -> {out_path}"
    )
    ap = camp.apparatus_row()
    if ap["refusal"]:
        print(f"[FAIL] apparatus refused: {ap['refusal']}")
        return 1
    done = _existing_clean(out_path, campaign_id) if args.resume else set()
    for seed in FROZEN["cal"]["seeds"][: FROZEN["cal"]["n"]]:
        if (FROZEN["cal"]["arm"], seed) in done:
            continue
        camp.event_row(FROZEN["cal"]["arm"], seed)
    all_rows = [json.loads(ln) for ln in out_path.read_text().splitlines() if ln.strip()]
    g = write_gauntlet([r for r in all_rows if r.get("campaign_id") == campaign_id], ap, campaign_id=campaign_id)
    bad = validate_gauntlet(g)
    if bad:
        print("[FAIL] the written gauntlet does not validate: " + "; ".join(bad))
        return 1
    gpath = Path(args.gauntlet)
    gpath = gpath if gpath.is_absolute() else C.REPO_ROOT / gpath
    gpath.write_text(json.dumps(g, indent=2, default=str) + "\n")
    print(
        f"gauntlet -> {gpath}: floor {g['floor_arm']}; reservoir {g['reservoir_band']}; ticks {g['tick_period_band_s']}"
    )
    return 0


def _bench(args: argparse.Namespace) -> int:
    got = _setup(args)
    if isinstance(got, int):
        return got
    camp, out_path, campaign_id = got
    gpath = Path(args.gauntlet)
    gpath = gpath if gpath.is_absolute() else C.REPO_ROOT / gpath
    g = load_json(gpath)
    bad = validate_gauntlet(g)
    if bad:
        print("[FAIL] gauntlet: " + "; ".join(bad))
        return 3
    if not _is_ancestor_of_main(str(g.get("cal_code_hash"))):
        print(
            f"[FAIL] the gauntlet's calibration hash {g.get('cal_code_hash')} is not on origin/main (merge the cal PR with a MERGE COMMIT)"
        )
        return 3
    camp.gauntlet = g
    if not args.only or "D_shared" in args.only.split(","):
        _ = camp.exp61  # construct Exp 61's campaign NOW (its own record checks SystemExit here, not at bench row 4)
    print(f"R3-bench {campaign_id} on gauntlet {g['cal_code_hash'][:8]} -> {out_path}")
    ap = camp.apparatus_row()
    if ap["refusal"]:
        print(f"[FAIL] apparatus refused: {ap['refusal']}")
        return 1
    done = _existing_clean(out_path, campaign_id) if args.resume else set()
    n_max = max(FROZEN["arms"].values())
    for i in range(n_max):
        for arm in ARMS:
            if i >= FROZEN["arms"][arm]:
                continue
            seed = FROZEN["seeds"][arm][i]
            if (arm, seed) in done:
                continue
            if args.only and arm not in args.only.split(","):
                continue
            print(f"\n=== {arm} seed {seed} ===")
            if arm == "D_shared":
                camp.shared_row(seed)
            else:
                camp.event_row(arm, seed)
    print(f"bench {campaign_id} -> {out_path}")
    return 0


def _report(args: argparse.Namespace) -> int:
    rows = [json.loads(ln) for ln in Path(args.data).read_text().splitlines() if ln.strip()]
    g = None
    if args.gauntlet:
        gp = Path(args.gauntlet)
        g = load_json(gp if gp.is_absolute() else C.REPO_ROOT / gp)
    amended: dict[str, Any] | None = None
    if args.amended:
        # Amendment 1: the hash rule is ANCESTRY + no harness change between the hashes (checked here, recorded)
        bench_hashes = sorted(
            {
                str((r.get("provenance") or {}).get("executed_git_hash"))
                for r in rows
                if r.get("campaign_id") == args.campaign_id
            }
        )
        cal_hash = str((g or {}).get("cal_code_hash"))
        checks = {h: harness_unchanged_between(cal_hash, h) for h in bench_hashes}
        rows, recounted = reclassify_under_amendments(rows)
        amended = {
            "amendment_1": {
                "cal_code_hash": cal_hash,
                "bench_hashes": bench_hashes,
                "harness_unchanged": {h: ok for h, (ok, _) in checks.items()},
                "touched": {h: t for h, (_, t) in checks.items()},
            },
            "amendment_2": {
                "recounted": [
                    {"arm": r["arm"], "seed": r["seed"], "tick_period_median_s": r["amended"]["tick_period_median_s"]}
                    for r in recounted
                ]
            },
        }
    rep = report(
        rows,
        campaign_id=args.campaign_id,
        gauntlet=g,
        hash_rule_satisfied_by_ancestry=amended is not None and all(ok for ok, _ in checks.values()),
    )
    if amended is not None:
        rep["amended"] = amended
        if not all(amended["amendment_1"]["harness_unchanged"].values()):
            rep["status"] = "INCOMPLETE"
            rep["incomplete_cause"] = (
                (rep.get("incomplete_cause") or "")
                + "; Amendment 1 unmet: the calibration hash is not an ancestor with an unchanged harness"
            ).strip("; ")
    print(json.dumps(rep, indent=2, default=str))
    print(f"STATUS: {rep['status']}" + (f" ({rep['incomplete_cause']})" if rep["incomplete_cause"] else ""))
    if args.json:
        Path(args.json).write_text(json.dumps(rep, indent=2, default=str) + "\n")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("cal", _cal), ("bench", _bench)):
        r = sub.add_parser(name)
        r.add_argument("--rcon-host", default="127.0.0.1")
        r.add_argument("--rcon-port", type=int, default=25575)
        r.add_argument("--rcon-password", required=True)
        r.add_argument("--username", default="maxim")
        r.add_argument("--bridge-host", default="127.0.0.1")
        r.add_argument("--bridge-port", type=int, default=25567)
        r.add_argument("--workdir", required=True)
        r.add_argument("--campaign-id", default=None)
        r.add_argument("--resume", action="store_true")
        r.add_argument("--gate-record", default=GATE_RECORD)  # Exp 61's donor flow reads it (arm D)
        r.add_argument("--gauntlet", default=GAUNTLET_DEFAULT)
        r.add_argument(
            "--out",
            default="docs/experiments/data/r3_cal.jsonl" if name == "cal" else "docs/experiments/data/r3_bench.jsonl",
        )
        r.add_argument("--only", default="", help="bench: comma list of arms")
        r.add_argument("--write-experiment-results", action="store_true")
        r.add_argument("--allow-dirty", action="store_true")
        r.set_defaults(func=fn)
    v = sub.add_parser("report")
    v.add_argument("--data", required=True)
    v.add_argument("--json", default=None)
    v.add_argument("--campaign-id", default=None)
    v.add_argument("--gauntlet", default=None, help="compare every row's hash to the gauntlet's cal hash (D5)")
    v.add_argument(
        "--amended", action="store_true", help="apply the 2026-09-18 instrument-only amendments (prereg §Amendments)"
    )
    v.set_defaults(func=_report)
    args = p.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
