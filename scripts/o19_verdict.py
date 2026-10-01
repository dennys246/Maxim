#!/usr/bin/env python3
"""The O19 re-run verdict: Exp 10 (T1-1) and Exp 09 (T3-9), computed from committed bytes only.

Pre-registrations (read them first; this module implements them and nothing else):
  docs/experiments/protocols/exp10_rerun_2026-09-30_preregistration.md
  docs/experiments/protocols/exp09_rerun_2026-09-30_preregistration.md

This module also holds the PROTOCOL (goals, turn caps, flags, model) that ``o19_rerun.py`` runs, so the
verdict's ``verdict_source_sha256`` covers what was run as well as how it was judged.

What the verdict reads, and how each input could lie:
  * the rows file (``rows.jsonl``), read ONCE: its bytes are hashed into the verdict (``stamp_verdict``);
  * the copied session files the rows name, each re-hashed against the row's SHA-256 (a mismatch refuses);
  * the start markers (annotated tags ``refs/tags/o19/<exp>/attempt-<k>-<run_id>``) on ``origin``, the tag ruleset
    that keeps them, and its history (``gh api``, read-only) — an attempt discarded before it was committed is still
    visible as a marker, and removing a marker needs a ruleset change, which its history shows;
  * ``origin/main``'s first-parent history of the rows file — each attempt landed before the next one started.
  A failure of any of these REFUSES (exit 2, no verdict written): no status change, the row stays STALE. Rig-clock
  times (row ``ts``, tagger dates) are trusted as the prereg says: the gate catches forgetting, not evasion.

Exits: 0 PASS; 1 FAIL; 4 PARTIAL (Exp 09) or ABORT; 2 refused; 3 provenance (the imported maxim is not this repo's).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
REPO_ROOT = SCRIPTS_DIR.parent

# ── the protocol (frozen by the preregs; a test asserts each value against the prereg text) ─────────────────

MODEL_PROFILE = "mistral-7b"  # what the operator configures (`maxim config set llm.profile mistral-7b`)
# What the report stamps: the router's profile is the NORMALIZED name (owner decision 2026-09-30, prereg amended
# before any data; a test pins it to ``normalize_llm_profile(MODEL_PROFILE)``).
MODEL_PROFILE_STAMPED = "mistral-7b-instruct-v0.2"
MODEL_GGUF = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
N_CTX = 8192
MAX_ATTEMPTS = 3
MARKER_NAMESPACE = "refs/tags/o19"
RULESET_INCLUDE = "refs/tags/o19/**"

EXP10_GOAL_DUNGEON = "escape a dungeon with a sleeping guard"
EXP10_GOAL_GARDEN = "you are in a peaceful garden, enjoy the flowers"
EXP09_GOAL = (
    "You are an adventurer in a dark cave. A dragon attacks you repeatedly with claws and fire breath. The dragon "
    "roars deafeningly. It slams you against the wall. A freezing wind blows through the cave."
)

# One entry per phase: (phase name, goal, turn cap, resumes phase 1?, extra argv, extra env).
PROTOCOL: dict[str, dict] = {
    "10": {
        "scope": "rerun_exp10_o19",
        "kind": "exp10_verdict",
        "prereg": "docs/experiments/protocols/exp10_rerun_2026-09-30_preregistration.md",
        "phases": [
            ("baseline", EXP10_GOAL_DUNGEON, 8, False, [], {}),
            ("gate", EXP10_GOAL_DUNGEON, 8, True, [], {}),
            ("negative_transfer", EXP10_GOAL_GARDEN, 5, True, [], {}),
        ],
    },
    "09": {
        "scope": "rerun_exp09_o19",
        "kind": "exp09_verdict",
        "prereg": "docs/experiments/protocols/exp09_rerun_2026-09-30_preregistration.md",
        "phases": [
            (
                "session",
                EXP09_GOAL,
                8,
                False,
                ["--embodiment", "bodies/base_humanoid"],
                {"MAXIM_SUBSTRATE_PATH": "1", "MAXIM_BACKEND_TRACE": "1"},
            ),
        ],
    },
}

RUN_LOG = "run_log.jsonl"  # the copied MAXIM_LOG_FILE (stored gzipped)
# The stores a resume restores (``simulation/report.py::RESUME_STORES``; a test pins the two together).
RESUME_STORES = ("hippocampus", "nac", "ec", "atl")
SIM_PORT = 8100  # the sim's local server (MAXIM_AUTO_SPAWN_PORT is not passed to the sims)
# The MAXIM_* environment a sim receives, beyond its protocol's own keys: the harness builds it from this list
# (everything else MAXIM_* is dropped), and C4 requires the recorded set to be exactly this plus the protocol's.
HARNESS_ENV = {"MAXIM_LOG_FILE_MAX_BYTES": "0"}


def required_files(exp: str) -> tuple[str, ...]:
    """The copied files a phase must carry: its report and run log, and Exp 10's hippocampus store (its C2)."""
    return ("report.json", RUN_LOG, "aut_hippocampus.json") if exp == "10" else ("report.json", RUN_LOG)


def expected_env(exp: str, index: int) -> dict[str, str]:
    return {**HARNESS_ENV, **PROTOCOL[exp]["phases"][index][5]}


def phase_argv(exp: str, index: int, resume_session: str | None) -> list[str]:
    """The sim argv after ``python -m maxim`` for one phase: the original protocol's command, unchanged."""
    _name, goal, cap, resumes, extra, _env = PROTOCOL[exp]["phases"][index]
    argv = ["--sim", goal, "--interactive", "false", "--sim-max-turns", str(cap), *extra]
    if resumes:
        if not resume_session:
            raise ValueError(f"phase {index} resumes phase 1 and needs its session id")
        argv += ["--resume-sim", resume_session]
    return argv


def data_dir(exp: str) -> str:
    return f"docs/experiments/data/{PROTOCOL[exp]['scope']}"


def rows_path(exp: str) -> str:
    return f"{data_dir(exp)}/rows.jsonl"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ── reading committed bytes ───────────────────────────────────────────────────────────────────────────────


class Refusal(Exception):
    """The verdict cannot be established from these bytes: no verdict is written (exit 2)."""


def read_copied(session_dir: Path, name: str) -> bytes:
    """A copied session file's UNCOMPRESSED bytes: ``<name>`` as stored, or ``<name>.gz``."""
    plain, packed = session_dir / name, session_dir / f"{name}.gz"
    if plain.is_file():
        return plain.read_bytes()
    if packed.is_file():
        return gzip.decompress(packed.read_bytes())
    raise FileNotFoundError(f"{session_dir}/{name}")


def log_lines(data: bytes, *, strict: bool = False) -> list[dict]:
    """The run log's JSON lines (compact form: event key ``e``, time ``t``). The LAST line may be torn (the sim was
    still writing); with ``strict``, any other line that is not a JSON object refuses: a log the reader cannot
    fully parse cannot be counted over."""
    out: list[dict] = []
    raw_lines = data.decode("utf-8", errors="replace").splitlines()
    for i, raw in enumerate(raw_lines):
        try:
            rec = json.loads(raw)
        except ValueError:
            rec = None
        if isinstance(rec, dict):
            out.append(rec)
        elif strict and raw.strip() and i != len(raw_lines) - 1:
            raise Refusal(f"run log line {i + 1} is not a JSON object")
    return out


_TURN_ENTER = re.compile(r"Bridge\.send_and_wait ENTER turn=(\d+)\b")


def turn_windows(lines: list[dict]) -> dict[int, tuple[float, float]]:
    """``turn -> [start, end)`` from the ``sim_exec`` ``Bridge.send_and_wait ENTER turn=N`` markers; the last turn
    runs to the end of the log. A turn entered twice keeps its first entry (a retry is the same turn)."""
    starts: dict[int, float] = {}
    for rec in lines:
        if rec.get("e") != "sim_exec":
            continue
        m = _TURN_ENTER.search(str(rec.get("message", "")))
        if m and int(m.group(1)) not in starts:
            starts[int(m.group(1))] = float(rec.get("t", 0.0))
    ordered = sorted(starts.items(), key=lambda kv: kv[1])
    windows: dict[int, tuple[float, float]] = {}
    for i, (turn, start) in enumerate(ordered):
        end = ordered[i + 1][1] if i + 1 < len(ordered) else float("inf")
        windows[turn] = (start, end)
    return windows


def hippocampus_ids(store: dict) -> list[str]:
    return [str(m.get("id")) for m in store.get("memories", []) if isinstance(m, dict) and m.get("id")]


# ── the complete-attempt condition (C1–C4), per phase ────────────────────────────────────────────────────


def complete_problems(
    exp: str, index: int, row: dict, report: dict, phase1_session: str | None, phase1_stores: set[str]
) -> list[str]:
    """Every way this phase's run falls short of the prereg's complete-attempt condition ([] = complete)."""
    name, goal, cap, resumes, extra, env = PROTOCOL[exp]["phases"][index]
    prov = report.get("provenance") or {}
    p: list[str] = []
    files = row.get("files") or {}
    for needed in required_files(exp):
        if needed not in files:
            p.append(
                f"C2 {name}: {needed} was not copied"
                if needed.startswith("aut_")
                else f"C1 {name}: {needed} was not copied"
            )
    # C1: ran to the cap.
    if report.get("finish_reason") != "max_turns":
        p.append(f"C1 {name}: finish_reason {report.get('finish_reason')!r}, not 'max_turns'")
    if not isinstance(report.get("turns"), int) or report["turns"] < cap:
        p.append(f"C1 {name}: turns {report.get('turns')!r} < {cap}")
    # C2: the resume chain (Exp 10's phases 2 and 3; nothing else resumes).
    resume = prov.get("resume")
    if resumes:
        stores = (resume or {}).get("stores") or {}
        if not (resume or {}).get("resume_loaded"):
            p.append(f"C2 {name}: resume_loaded is not true ({resume!r})")
        if (resume or {}).get("resumed_from_session") != phase1_session:
            p.append(
                f"C2 {name}: resumed {(resume or {}).get('resumed_from_session')!r}, not phase 1's {phase1_session!r}"
            )
        if stores.get("hippocampus") != "loaded":
            p.append(f"C2 {name}: stores.hippocampus is {stores.get('hippocampus')!r}, not 'loaded'")
        for store in sorted(phase1_stores):
            if stores.get(store) != "loaded":
                p.append(f"C2 {name}: phase 1 saved aut_{store}.json but it reads {stores.get(store)!r}")
    elif resume is not None:
        p.append(f"C2 {name}: phase resumes nothing, but its report carries a resume stamp")
    # C3: one known, unchanged code tree, the harness's own.
    tree = prov.get("code_tree_sha256")
    harness_tree = (row.get("provenance") or {}).get("code_tree_sha256")
    if prov.get("working_tree_dirty_src_scripts") is not False:
        p.append(f"C3 {name}: working_tree_dirty_src_scripts is {prov.get('working_tree_dirty_src_scripts')!r}")
    if prov.get("code_changed_during_run") is not False:
        p.append(f"C3 {name}: code_changed_during_run is {prov.get('code_changed_during_run')!r}")
    for label, value in (
        ("code_tree_sha256", tree),
        ("end_code_tree_sha256", prov.get("end_code_tree_sha256")),
        ("harness code_tree_sha256", harness_tree),
    ):
        if not isinstance(value, str) or not value or value.startswith("unknown"):
            p.append(f"C3 {name}: {label} is {value!r}")
    if not (tree == prov.get("end_code_tree_sha256") == harness_tree):
        p.append(
            f"C3 {name}: start {tree!r}, end {prov.get('end_code_tree_sha256')!r} and harness {harness_tree!r} differ"
        )
    # C4: the pre-registered apparatus.
    for role in ("language", "aut"):
        if prov.get(f"{role}_profile") != MODEL_PROFILE_STAMPED:
            p.append(f"C4 {name}: {role}_profile {prov.get(f'{role}_profile')!r}, not {MODEL_PROFILE_STAMPED!r}")
        if prov.get(f"{role}_router_n_ctx") != N_CTX:
            p.append(f"C4 {name}: {role}_router_n_ctx {prov.get(f'{role}_router_n_ctx')!r}, not {N_CTX}")
    if prov.get("configured_n_ctx") != N_CTX:
        p.append(f"C4 {name}: configured_n_ctx {prov.get('configured_n_ctx')!r}, not {N_CTX}")
    if prov.get("configured_n_ctx_source") != "config":
        p.append(f"C4 {name}: configured_n_ctx_source {prov.get('configured_n_ctx_source')!r}, not 'config'")
    if report.get("goal") != goal:
        p.append(f"C4 {name}: goal {report.get('goal')!r} is not the prereg's")
    expected_argv = phase_argv(exp, index, phase1_session if resumes else None)
    if row.get("sim_argv") != expected_argv:
        p.append(f"C4 {name}: recorded argv {row.get('sim_argv')!r} is not {expected_argv!r}")
    if (row.get("sim_env") or None) != expected_env(exp, index):
        p.append(f"C4 {name}: the sim's MAXIM_* env {row.get('sim_env')!r} is not {expected_env(exp, index)!r}")
    served = row.get("served_model") or {}
    reads = [r for r in served.get("reads") or [] if r.get("served") is not None]
    if not reads:
        p.append(f"C4 {name}: the served model was never read during the run")
    elif any(r.get("match") is not True for r in reads):
        p.append(f"C4 {name}: a served-model read did not match {MODEL_GGUF} ({reads!r})")
    endpoint = str(report.get("language_endpoint") or "").rstrip("/")
    if not endpoint or str(served.get("url") or "").rstrip("/") != endpoint:
        p.append(f"C4 {name}: served model read at {served.get('url')!r}, but the sim used {endpoint!r}")
    return p


# ── the gates ─────────────────────────────────────────────────────────────────────────────────────────────


def exp10_gates(phases: list[dict]) -> dict:
    """P0–P2 and R1–R2 over one complete attempt's three phases (``phases[i]`` = report, store, lines)."""
    base, gate, garden = phases
    ids1 = hippocampus_ids(base["store"])
    n1 = len(ids1)
    out: dict = {"N1": n1}

    def first_trace(lines: list[dict]) -> dict | None:
        return next((r for r in lines if r.get("e") == "enrichment_trace"), None)

    out["P0"] = {"pass": n1 >= 3, "N1": n1}
    p1 = {}
    for label, phase in (("gate", gate), ("negative_transfer", garden)):
        tr = first_trace(phase["lines"])
        size = None if tr is None else tr.get("hippocampus_size")
        p1[label] = {"first_trace_hippocampus_size": size, "pass": isinstance(size, int) and size >= n1}
    out["P1"] = {**p1, "pass": all(v["pass"] for v in p1.values())}
    p2 = {}
    for label, phase in (("gate", gate), ("negative_transfer", garden)):
        missing = sorted(set(ids1) - set(hippocampus_ids(phase["store"])))
        p2[label] = {"missing": len(missing), "pass": not missing}
    out["P2"] = {**p2, "pass": all(v["pass"] for v in p2.values())}

    windows = turn_windows(gate["lines"])
    traces = [r for r in gate["lines"] if r.get("e") == "enrichment_trace" and str(r.get("goal") or "").strip()]
    per_turn = {}
    for turn in range(1, 9):
        window = windows.get(turn)
        in_turn = [r for r in traces if window and window[0] <= float(r.get("t", 0.0)) < window[1]]
        first = in_turn[0] if in_turn else None
        per_turn[turn] = {
            "entered": window is not None,
            "traces": len(in_turn),
            "first_memories": None if first is None else first.get("memories"),
            "first_hippocampus_size": None if first is None else first.get("hippocampus_size"),
        }
    out["R1"] = {"pass": all(v["traces"] >= 1 for v in per_turn.values()), "turns": per_turn}
    out["R2"] = {
        "pass": all(isinstance(v["first_memories"], int) and v["first_memories"] >= 3 for v in per_turn.values())
    }
    out["turns_at_N1"] = [t for t, v in per_turn.items() if v["first_hippocampus_size"] == n1]
    k = None if per_turn[1]["first_hippocampus_size"] is None else per_turn[1]["first_hippocampus_size"] - n1
    out["carried_recall_turn1"] = (
        "not shown (no first-turn trace)"
        if k is None
        else "not shown (the first trace holds fewer memories than phase 1 saved)"
        if k < 0
        else "carried recall shown on the first resume turn"
        if k == 0
        else f"at least {3 - k} of the 3 were carried"
        if k < 3
        else "carried recall is not shown"
    )
    # Negative transfer: NOT MEASURED. Reported, never gating.
    new = [m for m in garden["store"].get("memories", []) if isinstance(m, dict) and m.get("id") not in set(ids1)]
    out["negative_transfer"] = {
        "status": "NOT MEASURED",
        "agent_actions": garden["report"].get("total_actions"),
        "new_memories": len(new),
        "new_memories_mentioning_dungeon_or_guard": sum(
            1 for m in new if re.search(r"dungeon|guard", json.dumps(m).lower())
        ),
    }
    passed = all(out[g]["pass"] for g in ("P0", "P1", "P2", "R1", "R2"))
    out["verdict"] = "PASS" if passed else "FAIL"
    return out


_DAMAGE = re.compile(r"component damage: (\w+)\.integrity\b.*\(source=(\w+)")


def h5_lower_bound(intensity: float, raw: float, n: int) -> Fraction:
    """The reconstructed sensitization factor's lower bound: ``(intensity − 0.0005)(1 + 0.3·n) / raw``, exactly,
    as fractions of the logged decimals (``intensity`` is logged to 3 dp, so the full rounding error is taken off
    before it is amplified)."""
    i, r = Fraction(str(intensity)), Fraction(str(raw))
    return (i - Fraction(5, 10000)) * (1 + Fraction(3, 10) * n) / r


def exp09_gates(phase: dict, log_bytes: bytes) -> dict:
    """H1–H7 over the one session's run log."""
    lines = phase["lines"]
    reflexes = [r for r in lines if r.get("e") == "sim_reflex"]
    out: dict = {}
    flinch = [r for r in reflexes if r.get("reflex") == "attack_flinch"]
    out["H1"] = {"status": "PASS" if flinch else "NOT MET", "attack_flinch_records": len(flinch)}

    components = set()
    for r in lines:
        if r.get("e") != "sim_sem_damage" or r.get("agent_id") != "sim_aut":
            continue
        m = _DAMAGE.search(str(r.get("message", "")))
        if m and m.group(2).startswith("reflex_"):
            components.add(m.group(1))
    out["H2"] = {
        "status": "PASS" if len(components) >= 2 and "legs" in components else "NOT MET",
        "components": sorted(components),
    }
    out["H3"] = {
        "status": "NOT MEASURED",
        "why": "no committed byte carries a published pain signal's reflex source (#1026)",
    }
    seq = [r.get("intensity") for r in flinch]
    out["H4"] = {
        "status": "PASS"
        if len(seq) >= 3 and all(isinstance(x, (int, float)) for x in seq) and seq[-1] < seq[0]
        else "NOT MET",
        "attack_flinch_intensities": seq,
    }
    seen: dict[str, int] = {}
    best = None
    for r in reflexes:
        name = str(r.get("reflex"))
        n = seen.get(name, 0)
        seen[name] = n + 1
        intensity, raw = r.get("intensity"), r.get("raw_intensity")
        if not isinstance(intensity, (int, float)) or not isinstance(raw, (int, float)) or raw <= 0:
            continue
        bound = h5_lower_bound(intensity, raw, n)
        if best is None or bound > best[0]:
            best = (bound, name, n)
    out["H5"] = {
        "status": "PASS" if best is not None and best[0] > 1 else "NOT MET",
        "max_lower_bound": None if best is None else f"{float(best[0]):.6f}",
        "at": None if best is None else {"reflex": best[1], "n": best[2]},
    }
    hits = log_bytes.count(b"auto_attack") + log_bytes.count(b"auto_damage")
    out["H6"] = {"status": "PASS" if hits == 0 else "NOT MET", "occurrences": hits}
    reflex_enrichment = [r for r in lines if r.get("e") == "sim_enrichment" and r.get("system") == "reflex"]
    out["H7"] = {"status": "PASS" if reflex_enrichment else "NOT MET", "records": len(reflex_enrichment)}
    row_metric = ("H1", "H4", "H5", "H6")
    if any(out[h]["status"] != "PASS" for h in row_metric):
        out["verdict"] = "FAIL"
    elif all(out[h]["status"] == "PASS" for h in out):
        out["verdict"] = "PASS"
    else:
        out["verdict"] = "PARTIAL"
    out["not_passed"] = [h for h in ("H1", "H2", "H3", "H4", "H5", "H6", "H7") if out[h]["status"] != "PASS"]
    return out


# ── attempts, markers, the ruleset, ordering ─────────────────────────────────────────────────────────────


def attempts_from_rows(rows: list[dict]) -> dict[str, list[dict]]:
    """Harness rows grouped by attempt (the harness run id), each in phase order."""
    out: dict[str, list[dict]] = {}
    for row in rows:
        if row.get("record_kind") != "harness_row":
            continue
        run_id = (row.get("provenance") or {}).get("harness_run_id")
        if not run_id:
            raise Refusal("a harness row carries no provenance.harness_run_id")
        out.setdefault(run_id, []).append(row)
    for run_rows in out.values():
        run_rows.sort(key=lambda r: r.get("phase_index", 0))
    return out


_MARKER = re.compile(r"attempt-(\d+)-([0-9a-f]{32})$")


def parse_markers(exp: str, ls_remote: str) -> dict[str, dict]:
    """``git ls-remote --tags origin`` lines for this exp's namespace -> ``run_id -> {k, ref, object, peeled}``.
    A tag with no ``^{}`` line is lightweight (no tagger date): refused."""
    prefix = f"{MARKER_NAMESPACE}/{exp}/"
    objects: dict[str, str] = {}
    peeled: dict[str, str] = {}
    for line in ls_remote.splitlines():
        if not line.strip():
            continue
        sha, ref = line.split()
        if not ref.startswith(prefix):
            continue
        if ref.endswith("^{}"):
            peeled[ref[: -len("^{}")]] = sha
        else:
            objects[ref] = sha
    markers: dict[str, dict] = {}
    for ref, sha in sorted(objects.items()):
        m = _MARKER.search(ref)
        if not m or ref != f"{prefix}{m.group(0)}":
            raise Refusal(f"marker {ref} is not attempt-<k>-<32-hex run id>")
        if ref not in peeled:
            raise Refusal(f"marker {ref} is a lightweight tag (no tagger date)")
        run_id = m.group(2)
        if run_id in markers:
            raise Refusal(f"two markers name run id {run_id}")
        markers[run_id] = {"k": int(m.group(1)), "ref": ref, "object": sha, "peeled": peeled[ref]}
    ks = sorted(m["k"] for m in markers.values())
    if ks != list(range(1, len(ks) + 1)):
        raise Refusal(f"marker k values {ks} are not unique and 1..n with no gaps")
    if len(ks) > MAX_ATTEMPTS:
        raise Refusal(f"{len(ks)} markers: more than {MAX_ATTEMPTS} attempts")
    return markers


def ruleset_problems(
    rulesets: list[dict],
    details: dict[int, dict],
    history: dict[int, list[dict]],
    first_marker_ts: float,
    parse_ts: Callable[[str], float],
) -> list[str]:
    """Why the tag ruleset does not keep the markers, or [] (prereg (a)). ``rulesets`` is the repo's ruleset list,
    ``details``/``history`` the full ruleset and its ``/history`` by id; ``first_marker_ts`` the earliest tagger date."""
    matches = []
    for rs in rulesets:
        d = details.get(rs.get("id"))
        if not d or d.get("target") != "tag":
            continue
        cond = (d.get("conditions") or {}).get("ref_name") or {}
        if cond.get("include") == [RULESET_INCLUDE] and not cond.get("exclude"):
            matches.append(d)
    if len(matches) != 1:
        return [f"{len(matches)} tag rulesets include exactly {RULESET_INCLUDE!r}; need exactly one"]
    d = matches[0]
    p: list[str] = []
    if d.get("enforcement") != "active":
        p.append(f"ruleset enforcement is {d.get('enforcement')!r}, not 'active'")
    types = {r.get("type") for r in d.get("rules") or []}
    for needed in ("deletion", "update"):
        if needed not in types:
            p.append(f"ruleset does not carry the {needed!r} rule")
    if d.get("bypass_actors") != []:
        p.append(f"ruleset bypass_actors is {d.get('bypass_actors')!r}, not []")
    if d.get("current_user_can_bypass") != "never":
        p.append(f"current_user_can_bypass is {d.get('current_user_can_bypass')!r}, not 'never'")
    stamps = [("created_at", d.get("created_at")), ("updated_at", d.get("updated_at"))]
    stamps += [(f"history {h.get('version_id')} updated_at", h.get("updated_at")) for h in history.get(d.get("id"), [])]
    for label, value in stamps:
        try:
            when = parse_ts(str(value))
        except ValueError:
            p.append(f"ruleset {label} {value!r} is unreadable")
            continue
        if when >= first_marker_ts:
            p.append(f"ruleset {label} {value} is not before the first marker")
    return p


def ordering_problems(order: list[dict]) -> list[str]:
    """``order`` = the attempts in k order, each ``{run_id, start, landed, has_rows, first_ts}``: ``start`` is its
    marker's tagger date (the declared start), ``landed`` when its rows first reached ``main`` (None = never). An
    attempt with rows must be on main before the next attempt's marker, and its rows start no earlier than its own
    marker. A marker with no rows (the harness died after the push) is an aborted attempt with nothing to order."""
    p = []
    for a, nxt in zip(order, order[1:]):
        if a["has_rows"] and (a["landed"] is None or a["landed"] >= nxt["start"]):
            p.append(
                f"attempt {a['run_id']}'s rows reached main at {a['landed']}, not before attempt {nxt['run_id']}'s "
                f"marker ({nxt['start']})"
            )
    for a in order:
        if a["has_rows"] and (a["first_ts"] is None or a["first_ts"] < a["start"]):
            p.append(f"attempt {a['run_id']}'s first row ({a['first_ts']}) precedes its own marker ({a['start']})")
    return p


def history_problems(versions: list[bytes], judged: bytes) -> list[str]:
    """The rows file is append-only on ``main``: each first-parent version (oldest first) is a byte prefix of the
    next, and the newest is exactly the bytes judged. A dropped or edited attempt fails here."""
    if not versions:
        return ["the rows file is not on origin/main"]
    p = [f"version {i + 1} of the rows file on main is not a prefix of version {i + 2}" for i, (a, b) in
         enumerate(zip(versions, versions[1:])) if not b.startswith(a)]  # fmt: skip
    if versions[-1] != judged:
        p.append("the rows file judged is not origin/main's")
    return p


def landing_times(history: list[tuple[str, float, bytes]]) -> dict[str, float]:
    """``run_id -> `` the commit time of the first first-parent version of the rows file that holds its rows."""
    landed: dict[str, float] = {}
    for _sha, when, data in history:
        for line in data.decode("utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            rid = (json.loads(line).get("provenance") or {}).get("harness_run_id")
            if rid and rid not in landed:
                landed[rid] = when
    return landed


# ── main ──────────────────────────────────────────────────────────────────────────────────────────────────


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout


def _git_bytes(*args: str) -> bytes | None:
    out = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True)
    return out.stdout if out.returncode == 0 else None


def _is_ancestor(commit: str, of: str) -> bool:
    return subprocess.run(["git", "merge-base", "--is-ancestor", commit, of], cwd=REPO_ROOT).returncode == 0


def _gh_json(path: str):
    out = subprocess.run(["gh", "api", path], cwd=REPO_ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        raise Refusal(f"gh api {path} failed (an unreadable ruleset refuses, never skips): {out.stderr.strip()[:300]}")
    return json.loads(out.stdout)


def _gh_list(path: str) -> list:
    """One page of up to 100; a full page refuses (a list the verdict cannot see whole is not read)."""
    items = _gh_json(f"{path}{'&' if '?' in path else '?'}per_page=100")
    if not isinstance(items, list) or len(items) >= 100:
        raise Refusal(f"gh api {path}: not a list, or 100+ entries (unpaginated)")
    return items


def _iso(value: str) -> float:
    from datetime import datetime

    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def rows_history(rel_rows: str) -> list[tuple[str, float, bytes]]:
    """The rows file at each first-parent commit of ``origin/main`` that touched it, oldest first (a deletion
    reads as empty bytes)."""
    out = []
    for line in reversed(_git("log", "--first-parent", "--format=%H %cI", "origin/main", "--", rel_rows).splitlines()):
        sha, when = line.split()
        out.append((sha, _iso(when), _git_bytes("show", f"{sha}:{rel_rows}") or b""))
    return out


def check_apparatus(exp: str, rel_rows: str, judged: bytes, attempts: dict[str, list[dict]]) -> tuple[list[dict], dict]:
    """The remote half: markers, the ruleset, the rows file's history on ``origin/main`` and each attempt's code on
    main. Returns the attempts in k order (``[{run_id, k, rows}]``, rows [] for a marker with none) and the
    apparatus record the verdict stamps. Raises :class:`Refusal`."""
    _git("fetch", "--quiet", "origin", "main", f"+{MARKER_NAMESPACE}/{exp}/*:{MARKER_NAMESPACE}/{exp}/*")
    markers = parse_markers(exp, _git("ls-remote", "--tags", "origin", f"{MARKER_NAMESPACE}/{exp}/*"))
    if not markers:
        raise Refusal("no start markers on origin")
    if set(attempts) - set(markers):
        raise Refusal(f"rows name attempts with no start marker: {sorted(set(attempts) - set(markers))}")
    for run_id, m in markers.items():
        fields = _git("for-each-ref", "--format=%(objecttype) %(objectname) %(taggerdate:unix)", m["ref"]).split()
        if len(fields) != 3 or fields[0] != "tag" or fields[1] != m["object"] or not fields[2].isdigit():
            raise Refusal(f"marker {m['ref']} is not the annotated tag origin lists ({fields})")
        if _git("rev-parse", f"{m['ref']}^{{commit}}").strip() != m["peeled"]:
            raise Refusal(f"marker {m['ref']} peels to another commit locally")
        m["tagger_date"] = float(fields[2])
        executed = {(r.get("provenance") or {}).get("executed_git_hash") for r in attempts.get(run_id, [])}
        if executed and executed != {m["peeled"]}:
            raise Refusal(f"attempt {run_id} ran on {sorted(map(str, executed))}, not its marker's commit")
        if not _is_ancestor(m["peeled"], "origin/main"):
            raise Refusal(f"attempt {run_id} ran on {m['peeled']}, which is not on origin/main")
    first_marker = min(m["tagger_date"] for m in markers.values())
    nwo = _gh_json("repos/{owner}/{repo}")["full_name"]
    rulesets = _gh_list(f"repos/{nwo}/rulesets?includes_parents=false")
    details = {rs["id"]: _gh_json(f"repos/{nwo}/rulesets/{rs['id']}") for rs in rulesets}
    history = {rid: _gh_list(f"repos/{nwo}/rulesets/{rid}/history") for rid in details}
    problems = ruleset_problems(rulesets, details, history, first_marker, _iso)
    rows_hist = rows_history(rel_rows)
    problems += history_problems([data for _s, _w, data in rows_hist], judged)
    landed = landing_times(rows_hist)
    ordered = sorted(markers.items(), key=lambda kv: kv[1]["k"])
    order = [
        {
            "run_id": rid,
            "start": m["tagger_date"],
            "landed": landed.get(rid),
            "has_rows": bool(attempts.get(rid)),
            "first_ts": min((r["ts"] for r in attempts.get(rid, [])), default=None),
        }
        for rid, m in ordered
    ]
    problems += ordering_problems(order)
    if problems:
        raise Refusal("; ".join(problems))
    kept = [
        d
        for d in details.values()
        if (d.get("conditions") or {}).get("ref_name", {}).get("include") == [RULESET_INCLUDE]
    ]
    record = {
        "markers": [{**m, "run_id": rid} for rid, m in ordered],
        "ruleset": {
            "id": kept[0].get("id"),
            "created_at": kept[0].get("created_at"),
            "updated_at": kept[0].get("updated_at"),
            "history_versions": [h.get("version_id") for h in history.get(kept[0].get("id"), [])],
        },
        "rows_history": [{"commit": sha, "committed_at": when} for sha, when, _d in rows_hist],
        "landed": landed,
    }
    return [{"run_id": rid, "k": m["k"], "rows": attempts.get(rid, [])} for rid, m in ordered], record


def judge(exp: str, attempts_in_order: list[dict], data_root: Path) -> dict:
    """Re-hash every copied file, decide each attempt's completeness from its committed bytes, and gate the
    first complete one. Pure over the bytes (no git, no network). Raises :class:`Refusal` on an integrity break."""
    listing = []
    deciding = None
    for attempt in attempts_in_order:
        rows = attempt["rows"]
        if deciding is not None:
            # The harness refuses a new attempt once the file holds a complete one; an attempt after it would let
            # the result be chosen.
            raise Refusal(f"attempt {attempt['run_id']} follows the complete attempt {deciding[0]['run_id']}")
        if [r.get("phase_index") for r in rows] != list(range(len(rows))):
            raise Refusal(f"attempt {attempt['run_id']}'s phase rows are not 0..n-1 exactly")
        entry = {"run_id": attempt["run_id"], "k": attempt.get("k"), "complete": False, "problems": []}
        listing.append(entry)
        if not rows:
            entry["problems"].append("a start marker with no rows: an aborted attempt")
            continue
        phases = []
        phase1_session = None
        phase1_stores: set[str] = set()
        for row in rows:
            index = row["phase_index"]
            session = row.get("session_id")
            if row.get("status") != "ok" or not session:
                entry["problems"].append(f"phase {index} row ended {row.get('status')!r}: {row.get('reason')}")
                break
            sdir = data_root / session
            files = row.get("files") or {}
            for name, digest in files.items():
                try:
                    data = read_copied(sdir, name)
                except FileNotFoundError as exc:
                    raise Refusal(f"{sdir}/{name}: named by its row but not committed") from exc
                if sha256_bytes(data) != digest:
                    raise Refusal(f"{sdir}/{name}: SHA-256 differs from its row")
            missing = [n for n in required_files(exp) if n not in files]
            if missing:
                entry["problems"].append(f"phase {index}: {missing} not copied")
                break
            report = json.loads(read_copied(sdir, "report.json"))
            if index == 0:
                phase1_session = session
                phase1_stores = {s for s in RESUME_STORES if f"aut_{s}.json" in files}
            entry["problems"] += complete_problems(exp, index, row, report, phase1_session, phase1_stores)
            log_bytes = read_copied(sdir, RUN_LOG)
            store = json.loads(read_copied(sdir, "aut_hippocampus.json")) if "aut_hippocampus.json" in files else {}
            phases.append(
                {"report": report, "store": store, "lines": log_lines(log_bytes, strict=True), "log_bytes": log_bytes}
            )
        if len(phases) != len(PROTOCOL[exp]["phases"]) and not entry["problems"]:
            entry["problems"].append(f"{len(phases)} of {len(PROTOCOL[exp]['phases'])} phases recorded")
        entry["complete"] = not entry["problems"]
        if entry["complete"]:
            deciding = (entry, phases)  # the first complete attempt decides
    out: dict = {"attempts": listing, "experiment": exp}
    if deciding is None:
        out["verdict"] = "ABORT"
        out["deciding_attempt"] = None
        return out
    entry, phases = deciding
    out["deciding_attempt"] = entry["run_id"]
    gates = exp10_gates(phases) if exp == "10" else exp09_gates(phases[0], phases[0]["log_bytes"])
    out["gates"] = gates
    out["verdict"] = gates["verdict"]
    return out


# The files that define what an attempt ran and how it is judged: identical at every executed commit, on main and
# at the verdict's own commit, or the verdict refuses (a post-data change to the gate is a new experiment).
BOUND_FILES = ("scripts/o19_verdict.py", "scripts/o19_rerun.py")

EXIT = {"PASS": 0, "FAIL": 1, "PARTIAL": 4, "ABORT": 4}


def bound_blobs(exp: str, commits: list[str]) -> dict[str, dict[str, str | None]]:
    """``path -> {commit -> blob sha}`` for the prereg and the two O19 scripts."""
    out: dict[str, dict[str, str | None]] = {}
    for path in (PROTOCOL[exp]["prereg"], *BOUND_FILES):
        out[path] = {}
        for commit in commits:
            blob = _git_bytes("rev-parse", f"{commit}:{path}")
            out[path][commit] = blob.decode().strip() if blob else None
    return out


def check_bound(exp: str, rows: list[dict]) -> dict:
    """The verdict runs on ``main``, and the prereg and both O19 scripts are byte-identical at every executed commit,
    on ``origin/main`` and at the verdict's own commit. Returns what the verdict stamps; raises :class:`Refusal`."""
    head = _git("rev-parse", "HEAD").strip()
    if not _is_ancestor(head, "origin/main"):
        raise Refusal(f"the verdict runs at {head}, which is not on origin/main")
    executed = sorted({(r.get("provenance") or {}).get("executed_git_hash") for r in rows} - {None})
    blobs = bound_blobs(exp, [*executed, "origin/main", head])
    for path, by_commit in blobs.items():
        if None in by_commit.values() or len(set(by_commit.values())) != 1:
            raise Refusal(f"{path} differs between the executed commits, origin/main and the verdict: {by_commit}")
    return {"bound_files": {path: next(iter(b.values())) for path, b in blobs.items()}, "verdict_commit": head}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", required=True, choices=sorted(PROTOCOL))
    ap.add_argument("--data", required=True, help="the rows file (docs/experiments/data/rerun_exp<NN>_o19/rows.jsonl)")
    ap.add_argument("--json", required=True, help="where the verdict is written")
    ap.add_argument("--write-experiment-results", action="store_true")
    ap.add_argument(
        "--offline",
        action="store_true",
        help="skip the markers/ruleset/history/blob checks; mock rows only, never written as evidence",
    )
    args = ap.parse_args(argv)
    if args.offline and args.write_experiment_results:
        print("REFUSED: an --offline verdict is a smoke and never replaces committed evidence", file=sys.stderr)
        return 2

    from _provenance import (  # noqa: PLC0415
        DirtyTreeError,
        ProvenanceError,
        any_not_stamped_real,
        evidence_out_path,
        in_process_code_provenance,
        stamp_verdict,
    )

    import maxim  # noqa: PLC0415

    exp = args.exp
    data = Path(args.data).resolve()
    try:
        out_path = evidence_out_path(REPO_ROOT, args.json, write_experiment_results=args.write_experiment_results)
        provenance = in_process_code_provenance(REPO_ROOT, maxim.__file__, out_path=out_path)
    except (ProvenanceError, DirtyTreeError) as exc:
        print(f"PROVENANCE: {exc}", file=sys.stderr)
        return 3
    extra: dict = {"apparatus_checked": not args.offline}
    try:
        data_bytes = data.read_bytes()  # read once: the verdict and its hash see the same bytes
        rows = [json.loads(ln) for ln in data_bytes.decode().splitlines() if ln.strip()]
        mock = any_not_stamped_real(rows)
        attempts = attempts_from_rows(rows)
        if args.offline:
            if not mock:
                raise Refusal("--offline skips the apparatus checks and is for mock rows only")
            ordered = [
                {"run_id": rid, "k": i + 1, "rows": rs}
                for i, (rid, rs) in enumerate(sorted(attempts.items(), key=lambda kv: min(r["ts"] for r in kv[1])))
            ]
        else:
            try:
                rel_rows = data.relative_to(REPO_ROOT).as_posix()
            except ValueError as exc:
                raise Refusal(f"rows file {data} is outside the repo") from exc
            if rel_rows != rows_path(exp):
                raise Refusal(f"rows file {rel_rows} is not the prereg's {rows_path(exp)}")
            ordered, extra["apparatus"] = check_apparatus(exp, rel_rows, data_bytes, attempts)
            extra.update(check_bound(exp, rows))
        report = judge(exp, ordered, data.parent)
    except Refusal as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    except (subprocess.CalledProcessError, OSError, ValueError, KeyError, TypeError) as exc:
        # Anything the verdict could not read is a refusal (no status change), never a FAIL (exit 1).
        print(f"REFUSED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    report.update(extra)
    report["verdict_source_sha256"] = sha256_bytes(Path(__file__).read_bytes())
    stamp_verdict(
        report,
        repo_root=REPO_ROOT,
        kind=PROTOCOL[exp]["kind"],
        data=data,
        data_bytes=data_bytes,
        scope={"all_rows": True},
        mock=mock,
    )
    report["provenance"] = provenance
    from maxim.utils.atomic_io import atomic_write_json  # noqa: PLC0415
    from maxim.utils.format_version import with_format_version  # noqa: PLC0415

    out_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_path, with_format_version(report), indent=2)
    print(json.dumps({k: report[k] for k in ("verdict", "deciding_attempt", "mock")}, indent=2))
    return EXIT[report["verdict"]]


if __name__ == "__main__":
    raise SystemExit(main())
