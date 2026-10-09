#!/usr/bin/env python3
"""The O19 re-run harness: ONE attempt of an Exp 10 (T1-1), Exp 09 (T3-9) or Exp 63 (T1-16) campaign per invocation.

Pre-registrations (this harness runs them and nothing else; the protocol constants and the campaign table live in
``o19_verdict.py`` so the verdict binds what was run as well as how it was judged). ``--exp`` names a CAMPAIGN:
  docs/experiments/protocols/exp10_rerun_2026-10-02_preregistration.md   (--exp 10c2; campaign "10" is closed)
  docs/experiments/protocols/exp09_rerun_2026-09-30_preregistration.md   (--exp 09)
  docs/experiments/exp63_carried_recall_prereg.md                        (--exp 63)

    python scripts/o19_rerun.py preflight --exp 10c2    # every refusal below + a dry-run marker push; spawns nothing
    python scripts/o19_rerun.py --exp 10c2 --write-experiment-results     # one attempt (`run` is the default)
    python scripts/o19_rerun.py run --exp 10 --mock     # offline end to end: fabricated sims, a temp rows file

An attempt, in order:
  1. Refusals, none of which is an attempt: the interpreter's ``maxim`` is not this repo's (3); the tree is dirty
     (3); HEAD is not on ``origin/main``'s history (2); the rows file is not ``origin/main``'s, or main's history of
     it is not append-only (2); it would mix code trees — keep the rig at the first attempt's commit (2); it already
     holds a complete attempt, or 3 attempts are declared (2); the campaign is closed, or a successor's predecessor
     has no pinned ABORT closure verdict on main before now (2); a campaign key outside ``PRE_1079_KEYS`` whose
     earlier attempt leaked a FAILED gate into its committed phases, or left a start marker with no rows (2, #1079:
     ``check_within_campaign``); the model config does not resolve to the prereg's
     (2); something already listens on the sim's port (2); another harness holds the lock (2); a git step fails (2);
     Exp 63 only: ``memory.strategy`` does not resolve to ``access_based`` in the attempt's fresh data home (2; the
     value read is stamped as ``memory_strategy`` in every row, C4').
  2. The start marker: an annotated tag ``refs/tags/o19/<campaign>/attempt-<k>-<run_id>`` on HEAD, pushed to ``origin``.
     A failed push is a refusal, not an attempt. From here on the attempt counts, whatever happens: every phase
     that starts writes a row, an interrupted one too.
  3. Each phase from ONE fresh ``MAXIM_DATA_HOME`` (models symlinked, no ``active_llm_model`` state, so the
     configured profile decides). The sim's environment is the operator's minus every ``MAXIM_*`` key, plus the
     harness's and the protocol's own (recorded exactly). While the sim runs, the harness reads the served model
     from the sim's port (``/v1/models``, with the key the sim itself uses) every 30 s. The sim's own report is
     found by run id (``spawn_evidence``); the session directory and its run log are copied into the data dir,
     the files hashed (uncompressed) into the row; the row says ``failed`` when the complete-attempt condition
     (C1–C4, ``o19_verdict.complete_problems``) does not hold, and later phases do not run.
  4. Phase 1 is copied the moment it ends and re-hashed before phase 3 resumes it.
The operator then commits the rows and copies to ``main`` (a merge-committed data PR) before any next attempt, and
leaves the rig at this attempt's commit.

Exits: 0 the attempt is complete; 1 the attempt aborted (recorded); 2 refused; 3 provenance/dirty tree.
"""

from __future__ import annotations

import argparse
import fcntl
import gzip
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
REPO_ROOT = SCRIPTS_DIR.parent

import _provenance  # noqa: E402
import o19_verdict as v  # noqa: E402

HARNESS = "o19_rerun"
POLL_S = 5.0
SERVED_EVERY_S = 30.0
TERM_GRACE_S = 120.0


class Refused(Exception):
    """A refusal before the start marker: not an attempt (exit 2)."""


# ── the sim's environment ────────────────────────────────────────────────────────────────────────────────


def base_env() -> tuple[dict[str, str], list[str]]:
    """The operator's environment minus every ``MAXIM_*`` key, and the keys dropped (recorded in the row)."""
    dropped = sorted(k for k in os.environ if k.startswith("MAXIM_"))
    return {k: val for k, val in os.environ.items() if not k.startswith("MAXIM_")}, dropped


# Per-run values the harness itself sets: present on every sim, recorded by presence (their values are run-local).
RUN_LOCAL_KEYS = ("MAXIM_DATA_HOME", "MAXIM_HARNESS_RUN_ID", "MAXIM_LOG_FILE")


def recorded_env(env: dict[str, str]) -> dict[str, str]:
    """The ``MAXIM_*`` environment actually handed to the sim, as C4 reads it: every key but the run-local three,
    which must all be present (a missing one is recorded as such, so C4 sees it)."""
    out = {k: val for k, val in env.items() if k.startswith("MAXIM_") and k not in RUN_LOCAL_KEYS}
    missing = [k for k in RUN_LOCAL_KEYS if not env.get(k)]
    if missing:
        out["_missing_run_local"] = ",".join(missing)
    return out


def sim_environment(exp: str, index: int, *, home: Path, run_id: str, run_log: Path) -> dict[str, str]:
    env, _dropped = base_env()
    env["MAXIM_DATA_HOME"] = str(home)
    env["MAXIM_HARNESS_RUN_ID"] = run_id
    env["MAXIM_LOG_FILE"] = str(run_log)
    env.update(v.expected_env(exp, index))
    return env


# ── refusals before the marker ───────────────────────────────────────────────────────────────────────────


def read_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def attempt_complete(exp: str, rows: list[dict]) -> bool:
    """The harness's own reading: every phase row ``ok`` (the verdict re-decides from the committed bytes)."""
    oks = [r for r in rows if r.get("record_kind") == "harness_row" and r.get("status") == "ok"]
    return len(oks) == len(v.PROTOCOL[exp]["phases"])


def check_on_main(rows_file: Path) -> None:
    """HEAD is on ``origin/main``'s history, and the rows file on disk is ``origin/main``'s, reached append-only."""
    v._git("fetch", "--quiet", "origin", "main")
    head = v._git("rev-parse", "HEAD").strip()
    if not v._is_ancestor(head, "origin/main"):
        raise Refused(f"HEAD {head} is not on origin/main: run from merged code (the rig stays on a main commit)")
    rel = rows_file.resolve().relative_to(REPO_ROOT).as_posix()
    history = v.rows_history(rel)
    on_disk = rows_file.read_bytes() if rows_file.is_file() else None
    if on_disk is None and not history:
        return
    problems = v.history_problems([data for _s, _w, data in history], on_disk or b"")
    if problems:
        raise Refused(
            f"{rel}: " + "; ".join(problems) + " — commit the last attempt to main (a data PR) first; if it is on "
            f"main but not here (the rig stays at its commit), restore it: git show origin/main:{rel} > {rel}"
        )


def run_probe(probe: str, env: dict[str, str], label: str) -> dict:
    """Run a preflight probe under ``env``; its last stdout line is a JSON object. Every way it can fail (a non-zero
    exit, a timeout, no output, not JSON, not an object) is a refusal, never a crash (E2)."""
    try:
        out = subprocess.run([sys.executable, "-c", probe], env=env, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired as exc:
        raise Refused(f"{label} probe timed out after {exc.timeout}s") from exc
    if out.returncode != 0:
        raise Refused(f"{label} probe failed: {out.stderr.strip()[-400:]}")
    lines = (out.stdout or "").strip().splitlines()
    try:
        got = json.loads(lines[-1]) if lines else None
    except ValueError:
        got = None
    if not isinstance(got, dict):
        raise Refused(f"{label} probe printed no JSON object: {(out.stdout or '')[-200:]!r}")
    return got


def check_model_config(home: Path, exp: str) -> dict:
    """The profile and context the sims will resolve, read by THIS repo's maxim under the sims' environment."""
    # The sim's own startup order (cli.main): role detection, then the C7a cloud auto-detect, which may switch a
    # solo run with a cloud key to a cloud profile through the environment — only then the setting resolution.
    probe = (
        "import json, logging\n"
        "from maxim.runtime.role import detect_and_apply_role\n"
        "detect_and_apply_role(json.loads(%r))\n"
        "from maxim.cli_utils import configure_cloud_solo_auto_detect\n"
        "configure_cloud_solo_auto_detect(logging.getLogger('o19'))\n"
        "from maxim.runtime.config_loader import resolve_setting\n"
        "from maxim.models.language.config import load_llm_config\n"
        "p, ps = resolve_setting('llm.profile')\n"
        "n, ns = resolve_setting('llm.n_ctx')\n"
        "cfg = load_llm_config(profile_override=p) if p else None\n"
        "print(json.dumps({'profile': p, 'profile_source': ps, 'n_ctx': n, 'n_ctx_source': ns,"
        " 'stamped_profile': getattr(cfg, 'profile', None), 'model_path': getattr(cfg, 'model_path', None)}))\n"
    ) % json.dumps(v.phase_argv(exp, 0, None))
    env, _dropped = base_env()
    env["MAXIM_DATA_HOME"] = str(home)
    got = run_probe(probe, env, "model config")
    problems = []
    if got.get("profile") != v.MODEL_PROFILE or got.get("profile_source") != "config":
        problems.append(f"llm.profile resolves to {got.get('profile')!r} from {got.get('profile_source')!r}")
    if got.get("stamped_profile") != v.MODEL_PROFILE_STAMPED:
        problems.append(f"the profile normalizes to {got.get('stamped_profile')!r}")
    if got.get("n_ctx") != v.N_CTX or got.get("n_ctx_source") != "config":
        problems.append(f"llm.n_ctx resolves to {got.get('n_ctx')!r} from {got.get('n_ctx_source')!r}")
    path = got.get("model_path") or ""
    if Path(path).name != v.MODEL_GGUF or not Path(path).is_file():
        problems.append(f"the profile's GGUF is {path!r}")
    if problems:
        raise Refused(
            "the model is not the prereg's (`maxim config set llm.profile mistral-7b`, "
            "`maxim config set llm.n_ctx 8192`): " + "; ".join(problems)
        )
    return got


def check_memory_strategy(home: Path) -> str:
    """Exp 63's C4': the retention model the sims will run, read by THIS repo's maxim through its own resolver
    (``config_loader.resolve_memory_strategy``) in the attempt's fresh data home, under the sims' environment (every
    operator ``MAXIM_*`` key dropped). Run before the marker: a failed import or read is a refusal, not an attempt."""
    probe = (
        "import json\n"
        "from maxim.runtime.config_loader import resolve_memory_strategy\n"
        "print(json.dumps({'memory_strategy': resolve_memory_strategy()}))\n"
    )
    env, _dropped = base_env()
    env["MAXIM_DATA_HOME"] = str(home)
    strategy = run_probe(probe, env, "memory strategy").get("memory_strategy")
    if strategy != v.EXP63_MEMORY_STRATEGY:
        raise Refused(f"memory.strategy resolves to {strategy!r}, not {v.EXP63_MEMORY_STRATEGY!r} (C4')")
    return strategy


def check_port_free(port: int) -> None:
    """The sim must start its own server at the configured n_ctx: one already answering would be reused, with a
    context the run cannot know (a quiet box)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        if s.connect_ex(("127.0.0.1", port)) == 0:
            raise Refused(f"something already listens on 127.0.0.1:{port}: stop it (the box must be quiet)")


def remote_markers(exp: str) -> dict[str, dict]:
    out = v._git("ls-remote", "--tags", "origin", f"{v.MARKER_NAMESPACE}/{exp}/*")
    try:
        return v.parse_markers(exp, out)
    except v.Refusal as exc:
        raise Refused(f"the start markers on origin are not well formed: {exc}") from exc


def push_marker(exp: str, k: int, run_id: str, *, dry_run: bool) -> str:
    """The start marker: an annotated (unsigned) tag on HEAD, pushed. ``dry_run`` pushes a scratch name with
    ``--dry-run`` (credentials and remote reachable; a dry run does not exercise the ruleset)."""
    if dry_run:
        ref = f"{v.MARKER_NAMESPACE}/preflight/{run_id}"
        out = subprocess.run(
            ["git", "push", "--dry-run", "origin", f"HEAD:{ref}"], cwd=REPO_ROOT, capture_output=True, text=True
        )
        if out.returncode != 0:
            raise Refused(f"dry-run marker push failed: {out.stderr.strip()[-400:]}")
        return ref
    ref = f"{v.MARKER_NAMESPACE}/{exp}/attempt-{k}-{run_id}"
    name = ref[len("refs/tags/") :]
    tag = subprocess.run(
        ["git", "-c", "tag.gpgSign=false", "tag", "-a", name, "-m", f"O19 Exp {exp} attempt {k} ({run_id})", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if tag.returncode != 0:
        raise Refused(f"could not create the marker tag (not an attempt): {tag.stderr.strip()[-400:]}")
    out = subprocess.run(["git", "push", "origin", ref], cwd=REPO_ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        subprocess.run(["git", "tag", "-d", name], cwd=REPO_ROOT, capture_output=True)
        raise Refused(f"marker push failed (not an attempt): {out.stderr.strip()[-400:]}")
    return ref


def take_lock(exp: str):
    """One harness per experiment on this box: two would race for the same k."""
    handle = (Path(tempfile.gettempdir()) / f"o19-exp{exp}.lock").open("w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise Refused(f"another o19 harness for Exp {exp} is running") from exc
    return handle


# ── one phase ────────────────────────────────────────────────────────────────────────────────────────────


# llama_cpp.server answers /v1/models under the same model lock as a completion, so a read made mid-generation
# waits for it. Campaigns 1-2 read with the backend's 4 s discovery budget and 26 of 36 reads came back null; their
# completions took up to 24 s, and the narrator and the AUT share the server, so a read may wait for two.
SERVED_READ_TIMEOUT = {"connect_s": 2.0, "read_s": 60.0, "total_s": 60.0}


def check_argv_parses(exp: str) -> None:
    """Every phase's argv parses on this build, BEFORE the marker: a flag the build lacks (e.g. a protocol merged
    ahead of the code it needs) would otherwise end every phase at argparse and cost an attempt."""
    from maxim.cli_parser import _build_parser  # noqa: PLC0415

    for index, phase in enumerate(v.PROTOCOL[exp]["phases"]):
        argv = v.phase_argv(exp, index, "preflight" if phase[3] else None)
        try:
            _build_parser().parse_args(argv)
        except SystemExit as exc:
            raise Refused(
                f"phase {index}'s argv does not parse on this build (argparse's error above): {argv}"
            ) from exc


def served_reader():
    """The served-model reader, imported BEFORE the marker (a failed import must not cost an attempt). It reads
    ``GET /v1/models`` the way the peer backend's discovery does (the first ``data[].id``), with a budget that
    outlasts the completions it queues behind, and records why a read failed."""
    from maxim.runtime.llm_server import _served_model_matches  # noqa: PLC0415
    from maxim.tunnel.keys import read_key  # noqa: PLC0415
    from maxim.utils import http as _http  # noqa: PLC0415

    def read(port: int, gguf_path: str) -> dict:
        key = read_key()
        served, error = None, None
        try:
            resp = _http.fetch_url(
                f"http://127.0.0.1:{port}/v1/models",
                headers={"Authorization": f"Bearer {key}"} if key else None,
                timeout=_http.TimeoutPolicy(**SERVED_READ_TIMEOUT),
            )
            data = json.loads(resp.content or b"{}").get("data")
            first = data[0] if isinstance(data, list) and data else None
            served = (first.get("id") or None) if isinstance(first, dict) and isinstance(first.get("id"), str) else None
            error = None if served else f"no data[].id in the response (HTTP {resp.status})"
        except Exception as exc:  # noqa: BLE001 -- a failed read is recorded, and C4 judges the reads
            error = f"{type(exc).__name__}: {str(exc)[:200]}"
        row = {"served": served, "match": _served_model_matches(served, gguf_path, v.MODEL_PROFILE), "at": time.time()}
        if error:
            row["error"] = error
        return row

    return read


def copy_session(session_dir: Path, run_log: Path | None, console: Path | None, dest: Path) -> dict[str, str]:
    """Copy the session directory (``*.jsonl`` and the run log gzipped) into ``dest``; returns each file's name
    (uncompressed) -> SHA-256 of its uncompressed bytes."""
    dest.mkdir(parents=True, exist_ok=False)
    digests: dict[str, str] = {}
    sources = sorted(p for p in session_dir.iterdir() if p.is_file())
    extra = [(run_log, v.RUN_LOG), (console, "console.out")]
    for src, name in [(p, p.name) for p in sources] + [(s, n) for s, n in extra if s is not None and s.is_file()]:
        data = src.read_bytes()
        digests[name] = v.sha256_bytes(data)
        if name.endswith(".jsonl") or name == "console.out":
            (dest / f"{name}.gz").write_bytes(gzip.compress(data, mtime=0))
        else:
            (dest / name).write_bytes(data)
    return digests


def stop_sim(proc) -> None:
    """SIGTERM first (the sim's handler stops the server it spawned), then SIGKILL after a grace period."""
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=TERM_GRACE_S)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


class HarnessSignal(BaseException):
    """SIGTERM/SIGHUP to the harness: raised so the running phase still writes its row and stops its sim."""


def _raise_signal(signum, _frame):
    raise HarnessSignal(f"signal {signum}")


def tree_digests(session_dir: Path) -> dict[str, str]:
    return {p.name: v.sha256_bytes(p.read_bytes()) for p in sorted(session_dir.iterdir()) if p.is_file()}


def spawn_phase(
    exp: str,
    index: int,
    *,
    home: Path,
    run_id: str,
    resume: str | None,
    gguf: str,
    timeout_s: float,
    read_served,
    **_kw,
):
    """Run one phase's sim. Returns ``(session_dir | None, report | None, row_fields, run_log, console, error)``."""
    argv = v.phase_argv(exp, index, resume)
    logs = Path(tempfile.mkdtemp(prefix="o19-log-"))
    run_log, console = logs / v.RUN_LOG, logs / "console.out"
    env = sim_environment(exp, index, home=home, run_id=run_id, run_log=run_log)
    cmd = [sys.executable, "-m", "maxim", *argv]
    before = _provenance.list_sessions(home)
    fields: dict = {
        "sim_argv": argv,
        "sim_env": recorded_env(env),
        "dropped_operator_env": base_env()[1],
        "hostname": socket.gethostname(),
        "ts": time.time(),
    }
    url = f"http://127.0.0.1:{v.SIM_PORT}/v1"
    reads: list[dict] = []
    fields["served_model"] = {"url": url, "reads": reads}
    timed_out = False
    last_read = 0.0
    with console.open("wb") as out:
        proc = subprocess.Popen(cmd, env=env, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        deadline = time.monotonic() + timeout_s
        try:
            while proc.poll() is None:
                if time.monotonic() > deadline:
                    timed_out = True
                    break
                if time.monotonic() - last_read >= SERVED_EVERY_S:
                    reads.append(read_served(v.SIM_PORT, gguf))
                    # after the read: a read that waited out a completion keeps the spacing
                    last_read = time.monotonic()
                time.sleep(POLL_S)
        finally:
            stop_sim(proc)  # a timeout or an interrupted harness must not leave the sim (and its server) running
    fields["end_ts"] = time.time()
    fields["returncode"] = proc.returncode
    fields["depends_on"] = _provenance.depends_on(home, before)
    if timed_out:
        return None, None, fields, run_log, console, f"timed out after {timeout_s:.0f}s"
    if Path(f"{run_log}.1").exists():
        return None, None, fields, run_log, console, "the run log rotated (MAXIM_LOG_FILE_MAX_BYTES ignored)"
    try:
        session_dir, report = _provenance.spawn_evidence(home, run_id, before, returncode=proc.returncode)
    except _provenance.SimRunFailed as exc:
        fields.update({k: val for k, val in _provenance.failed_row(exc).items() if k != "record_kind"})
        sims = exc.sims or []
        found = home / "sim_reports" / sims[0]["session_id"] if len(sims) == 1 else None
        return found if found and found.is_dir() else None, None, fields, run_log, console, fields["reason"]
    fields["sims"] = [_provenance.sim_evidence(session_dir, report)]
    return session_dir, report, fields, run_log, console, None


# ── mock (offline end to end) ────────────────────────────────────────────────────────────────────────────


def mock_phase(exp: str, index: int, *, home: Path, run_id: str, resume: str | None, provenance: dict, **_kw):
    """A fabricated phase that satisfies the protocol, written where a sim would write it, so the copy, hash,
    complete-attempt and verdict paths run end to end without an LLM. Rows from it are stamped mock."""
    name, goal, cap, resumes, _extra, _env = v.PROTOCOL[exp]["phases"][index]
    session = f"mock_{run_id[:8]}_{index}"
    sdir = home / "sim_reports" / session
    sdir.mkdir(parents=True)
    tree = provenance["code_tree_sha256"]
    prov = {
        "harness_run_id": run_id,
        "working_tree_dirty_src_scripts": False,
        "code_changed_during_run": False,
        "code_tree_sha256": tree,
        "end_code_tree_sha256": tree,
        "configured_n_ctx": v.N_CTX,
        "configured_n_ctx_source": "config",
        "language_profile": v.MODEL_PROFILE_STAMPED,
        "aut_profile": v.MODEL_PROFILE_STAMPED,
        "language_router_n_ctx": v.N_CTX,
        "aut_router_n_ctx": v.N_CTX,
    }
    if resumes:
        prov["resume"] = {
            "resume_loaded": True,
            "resumed_from_session": resume,
            "stores": {s: "loaded" for s in v.RESUME_STORES},
        }
    endpoint = f"http://127.0.0.1:{v.SIM_PORT}/v1"
    report = {
        "finish_reason": "max_turns",
        "turns": cap,
        "goal": goal,
        "language_endpoint": endpoint,
        "total_actions": cap,
        "provenance": prov,
    }
    (sdir / "report.json").write_text(json.dumps(report))
    if v.experiment_of(exp) == "63":
        lines = mock63_session(home, session, index, goal, cap, resume)
    else:
        lines = _mock_stores_and_lines(exp, sdir, home, session, name, goal, cap, resumes, resume)
    log = Path(tempfile.mkdtemp(prefix="o19-mocklog-")) / v.RUN_LOG
    log.write_text("".join(json.dumps(x) + "\n" for x in lines))
    now = time.time()
    fields = {
        "sim_argv": v.phase_argv(exp, index, resume),
        "sim_env": recorded_env(
            sim_environment(exp, index, home=home, run_id=run_id, run_log=Path(tempfile.gettempdir()) / "mock.log")
        ),
        "dropped_operator_env": [],
        "hostname": "mock",
        "ts": now,
        "end_ts": now,
        "returncode": 0,
        "served_model": {"url": endpoint, "reads": [{"served": v.MODEL_GGUF, "match": True, "at": now}]},
        "depends_on": [],
        "sims": [_provenance.sim_evidence(sdir, report)],
    }
    return sdir, report, fields, log, None, None


def _mock_stores_and_lines(exp, sdir, home, session, name, goal, cap, resumes, resume) -> list[dict]:
    """Exp 10's and Exp 09's mock stores (written) and run-log lines (returned)."""
    prior = []
    if resumes:
        prior = json.loads((home / "sim_reports" / resume / "aut_hippocampus.json").read_text())["memories"]
    memories = prior + [{"id": f"{session}-m{i}", "text": f"{name} memory {i}"} for i in range(4)]
    for store in v.RESUME_STORES:
        payload = {"memories": memories} if store == "hippocampus" else {}
        (sdir / f"aut_{store}.json").write_text(json.dumps(payload))
    lines = []
    t = 1000.0
    for turn in range(1, cap + 1):
        lines.append({"t": t, "e": "sim_exec", "message": f"Bridge.send_and_wait ENTER turn={turn} text_len=9"})
        lines.append(
            {"t": t + 0.1, "e": "enrichment_trace", "goal": goal, "memories": 3, "hippocampus_size": len(prior)}
        )
        if v.experiment_of(exp) == "09":
            for reflex, inten in (("attack_flinch", 0.15 - 0.01 * turn), ("impact_brace", 0.2)):
                lines.append(
                    {
                        "t": t + 0.2,
                        "e": "sim_reflex",
                        "reflex": reflex,
                        "intensity": round(inten, 3),
                        "raw_intensity": 0.1,
                    }
                )
            for comp in ("torso", "legs"):
                lines.append(
                    {
                        "t": t + 0.3,
                        "e": "sim_sem_damage",
                        "agent_id": "sim_aut",
                        "message": f"component damage: {comp}.integrity → 0.8 (source=reflex_attack, amount=0.1)",
                    }
                )
            lines.append({"t": t + 0.4, "e": "sim_enrichment", "system": "reflex", "agent_id": "sim_aut"})
        t += 10.0
    return lines


# Which Exp 63 mock the harness fabricates (tests set it): "pass" -- carried memories outrank the new ones, every
# turn decisive; "not_shown" -- three new, better-ranked memories land per turn, so after turn 1 no turn is decisive;
# "unrendered" -- a C5(c) instrument fault (one rendered id's enrichment activation was never counted).
MOCK63_VARIANT = "pass"


def _mock63_memory(mid: str, seq: int, ts: float, goal: str, tool: str, objects: list[str], text: str) -> dict:
    """One record as the real store saves it (``EpisodicMemory.to_dict``)."""
    from maxim.memory.types import Action, Context, EpisodicMemory, Perception  # noqa: PLC0415

    memory = EpisodicMemory(
        id=mid,
        timestamp=ts,
        run_id="mock",
        created_at=ts,
        accessed_at=ts,
        perception=Perception(observations={"text": text}, detected_objects=objects),
        context=Context(active_goal=goal),
        action=Action(tool_name=tool),
        capture_seq=seq,
    )
    return memory.to_dict()


def mock63_session(home: Path, session: str, index: int, goal: str, cap: int, resume: str | None) -> list[dict]:
    """An Exp 63 phase as the instrumented sim writes it: ``aut_*.json`` stores (the Hippocampus with
    ``capture_seq``s, shared timestamps and enrichment activations that match the renders) and a run log of turn
    markers, the AUT's deliberation and traces (ids chosen by the judge's frozen ranker over what each trace could
    see), plus a narrator trace the gates must ignore."""
    sdir = home / "sim_reports" / session
    records: dict[str, dict] = {}
    if resume is not None:
        prior = json.loads((home / "sim_reports" / resume / "aut_hippocampus.json").read_text())["memories"]
        records = {m["id"]: m for m in prior}
    loaded = set(records)
    seq = max((m["capture_seq"] for m in records.values()), default=-1) + 1
    t0 = 1_790_000_000.0 + 1000.0 * index
    variant = MOCK63_VARIANT if index == 1 else "pass"
    per_turn = 3 if variant == "not_shown" else 1
    lines: list[dict] = []
    rendered: list[str] = []
    for turn in range(1, cap + 1):
        t = t0 + 10.0 * turn
        lines.append({"t": t, "e": "sim_exec", "message": f"Bridge.send_and_wait ENTER turn={turn} text_len=9",
                      "agent_id": "sim_orchestrator"})  # fmt: skip
        lines.append({"t": t, "e": "sim_deliberation", "message": "deliberation cycle 1/3", "agent_id": "sim_aut"})
        horizon = max((m["capture_seq"] for m in records.values()), default=-1)
        ranked = v.rank_by_relevance(list(records.values()), goal, v.EXP63_GOAL_LIMIT)[: v.EXP63_TOP]
        ids = [m["id"] for m in ranked]
        rendered += ids
        lines.append(
            {"t": t, "e": "enrichment_trace", "agent_id": "sim_aut", "query_text": "where is the guard",
             "goal": goal, "memories": len(ids), "memory_ids": ids, "memory_paths": ["goal"] * len(ids),
             "goal_path_horizon": horizon, "goal_path_holes": [], "hippocampus_size": len(records)}
        )  # fmt: skip
        lines.append({"t": t, "e": "enrichment_trace", "agent_id": "sim_orchestrator", "goal": "",
                      "memories": 1, "memory_ids": ["narrator-1"], "memory_paths": ["substring"]})  # fmt: skip
        # Captures land after the trace: phase 1's share a timestamp per pair; carried ones rank on the goal.
        for k in range(per_turn):
            outranks = index == 0 or variant == "not_shown"
            mid = f"{session}-m{seq}"
            records[mid] = _mock63_memory(
                mid,
                seq,
                t0 + 10.0 * (turn - turn % 2) + 1.0 + (k if index else 0),
                goal if outranks else "look around the cell",
                "examine" if outranks else "look",
                ["door"] if turn % 3 == 0 else [],
                "the guard sleeps by the door" if outranks else "a damp stone cell",
            )
            seq += 1
    for mid in rendered:  # _activate_rendered: one enrichment activation per id rendered in memory_ids[:3]
        rec = records[mid]
        rec["activation_sources"]["enrichment"] = rec["activation_sources"].get("enrichment", 0) + 1
        rec["activation_count"] += 1
        rec["access_count"] += 1  # recall touches what it returns (mutable bookkeeping P3 ignores)
        rec["accessed_at"] = t0 + 99.0
    if variant == "unrendered" and rendered:
        rec = records[rendered[0]]
        rec["activation_sources"]["enrichment"] -= 1
    store = {"_format_version": "1.0", "saved_at": t0 + 100.0, "memories": list(records.values())}
    (sdir / "aut_hippocampus.json").write_text(json.dumps(store))
    for name in v.RESUME_STORES:
        if name != "hippocampus":
            (sdir / f"aut_{name}.json").write_text(json.dumps({}))
    assert loaded <= set(records)
    return lines


# ── the attempt ──────────────────────────────────────────────────────────────────────────────────────────


def exp63_attempt_problems(exp: str, data_root: Path, sessions: list) -> list[str]:
    """Exp 63's C5 and R3a observability, read from the COPIED phases once the last one is in: the harness's own
    reading of completeness then matches the verdict's (an attempt C5 makes incomplete is an aborted attempt, and
    the next one may start)."""
    try:
        phases = []
        for session in sessions:
            sdir = data_root / str(session)
            phases.append(
                {
                    "report": json.loads(v.read_copied(sdir, "report.json")),
                    "store": json.loads(v.read_copied(sdir, "aut_hippocampus.json")),
                    "lines": v.log_lines(v.read_copied(sdir, v.RUN_LOG), strict=True),
                }
            )
        return v.exp63_c5_problems(phases, goal=v.PROTOCOL[exp]["phases"][-1][1])
    except (OSError, ValueError, v.Refusal) as exc:
        return [f"C5 the copied phases could not be read: {type(exc).__name__}: {exc}"]


def append_row(rows_file: Path, row: dict) -> None:
    rows_file.parent.mkdir(parents=True, exist_ok=True)
    with rows_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def check_campaign(exp: str) -> None:
    """The campaign may start an attempt: the table is sound, the campaign is not closed, and a successor's
    predecessor closed with a pinned ABORT verdict that, with this campaign's prereg, is already on ``origin/main``
    (owner decisions 2026-10-02), no predecessor leaked a FAILED gate and the subject is unchanged (#1059). Real
    attempts only: a mock attempt writes nothing."""
    if problems := v.protocol_problems():
        raise Refused(f"the campaign table is unsound: {problems}")
    if exp in (closed := v.closed_keys()):
        raise Refused(f"campaign {exp} is closed (superseded by {closed[exp]}): run --exp {closed[exp]}")
    sup = v.PROTOCOL[exp].get("supersedes")
    if sup is not None:
        # #1059 S7: the preflight half of the strict successor gate (the evidence gate stays the authority): no FAILED
        # gate leaked into a predecessor's committed phases, and the subject this attempt runs on is byte-identical
        # to every predecessor's executed commits.
        with tempfile.TemporaryDirectory() as tmp:
            problems = v.successor_problems(
                exp,
                v._git_bytes("show", f"origin/main:{sup['verdict']}"),
                v.landed_on_main(sup["verdict"], sup["verdict_sha256"]),
                v.landed_on_main(v.PROTOCOL[exp]["prereg"]),
                time.time(),
                data_root=v.materialize("origin/main", v.predecessors(exp), Path(tmp)),
            )
        problems += v.subject_problems(exp, v._git("rev-parse", "HEAD").strip())
        if problems:
            raise Refused("; ".join(problems))


def check_within_campaign(exp: str, markers: dict[str, dict]) -> None:
    """#1079 D2 (owner decision 2026-10-08): the preflight half of the within-campaign bar (the evidence gate stays
    the authority). For a campaign key outside ``PRE_1079_KEYS``, no earlier attempt of THIS campaign leaked a FAILED
    gate into its committed phases, and no earlier start marker is rowless (D1): once one has, no later attempt of the
    campaign can supply support, so it must not burn rig time. ``markers`` are origin's (``ls-remote``, N4), so an
    attempt whose rows never landed is still seen; rows and session files are read as ``origin/main`` holds them."""
    if exp in v.PRE_1079_KEYS:
        return
    with tempfile.TemporaryDirectory() as tmp:
        data_root = v.materialize("origin/main", [exp], Path(tmp)) / v.PROTOCOL[exp]["scope"]
        try:
            rows = read_rows(data_root / "rows.jsonl")
        except ValueError as exc:  # a malformed line on main: the bar cannot be ruled out, so refuse (exit 2)
            raise Refused(f"campaign {exp}'s rows on origin/main cannot be read ({exc}) (#1079)") from exc
        attempts = v.attempts_from_rows(rows)
        ordered = [
            {"run_id": rid, "k": m["k"], "rows": attempts.get(rid, [])}
            for rid, m in sorted(markers.items(), key=lambda kv: kv[1]["k"])
        ]
        _structured, notes = v.within_campaign_leaks(exp, ordered, None, data_root)
    if notes:
        raise Refused(
            f"an earlier attempt of campaign {exp} leaked a FAILED gate or cannot be judged (#1079): no later attempt "
            f"of this campaign can supply support; close it with its verdict: {notes[:3]}"
        )


def prepare(args: argparse.Namespace, exp: str, mock: bool):
    """Everything before the marker. Returns ``(rows_file, provenance, run_id, k, home, gguf, read_served,
    lock, stamps)`` (``stamps``: what every row of the attempt carries beyond its phase, Exp 63's
    ``memory_strategy``); raises :class:`Refused`, or ``_provenance.ProvenanceError`` (exit 3)."""
    if mock:
        rows_file = _provenance.evidence_out_path(REPO_ROOT, v.rows_path(exp), write_experiment_results=False)
    else:
        rows_file = (REPO_ROOT / v.rows_path(exp)).resolve()
    _provenance.assert_repo_interpreter(REPO_ROOT, sys.executable, exempt=mock)
    provenance = _provenance.executed_code_provenance(REPO_ROOT, sys.executable, out_path=rows_file)
    run_id = _provenance.harness_run_id()
    lock = None if mock else take_lock(exp)
    refusal = _provenance.append_refusal(rows_file, provenance)
    if refusal:
        raise Refused(refusal + " (keep the rig at the first attempt's commit)")
    try:
        check_argv_parses(exp)
        if not mock:
            check_on_main(rows_file)
            check_campaign(exp)
        attempts = v.attempts_from_rows(read_rows(rows_file))
        if any(attempt_complete(exp, rs) for rs in attempts.values()):
            raise Refused("the rows file already holds a complete attempt: the first complete attempt decides")
        if mock:
            markers = {rid: {"k": i + 1} for i, rid in enumerate(attempts)}
        else:
            markers = remote_markers(exp)
            if set(attempts) - set(markers):
                raise Refused(f"rows name attempts with no start marker: {sorted(set(attempts) - set(markers))}")
            check_within_campaign(exp, markers)
    except (subprocess.CalledProcessError, v.Refusal) as exc:
        raise Refused(f"{type(exc).__name__}: {exc}") from exc
    k = len(markers) + 1
    if k > v.MAX_ATTEMPTS:
        raise Refused(f"{len(markers)} attempts are already declared: at most {v.MAX_ATTEMPTS}")
    home = Path(tempfile.mkdtemp(prefix="o19-home-"))
    models = Path(os.environ.get("MAXIM_DATA_HOME") or Path.home() / ".maxim") / "models"
    if models.is_dir():
        (home / "models").symlink_to(models)
    gguf, read_served = "", None
    stamps: dict = {}
    try:
        if v.experiment_of(exp) == "63":  # mock or not: the mock runs the same read in its own fresh home
            stamps["memory_strategy"] = check_memory_strategy(home)
        if not mock:
            gguf = check_model_config(home, exp)["model_path"]
            check_port_free(v.SIM_PORT)
            read_served = served_reader()
    except BaseException:  # any refusal or failure before the marker: the fresh home goes too (E2)
        shutil.rmtree(home, ignore_errors=True)
        raise
    return rows_file, provenance, run_id, k, home, gguf, read_served, lock, stamps


def run(args: argparse.Namespace) -> int:
    exp = args.exp
    mock = bool(args.mock)
    if mock and args.write_experiment_results:
        print("REFUSED: a mock attempt never writes committed evidence", file=sys.stderr)
        return 2
    if args.command == "run" and not mock and not args.write_experiment_results:
        print("REFUSED: a real attempt writes committed evidence: pass --write-experiment-results", file=sys.stderr)
        return 2
    try:
        rows_file, provenance, run_id, k, home, gguf, read_served, lock, stamps = prepare(args, exp, mock)
    except _provenance.ProvenanceError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 3
    except (Refused, OSError, ImportError, ValueError, subprocess.SubprocessError) as exc:
        # Nothing is pushed yet: any failure here is a refusal (2), never "attempt aborted" (1).
        print(f"REFUSED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    try:
        if args.command == "preflight":
            if not mock:
                push_marker(exp, k, run_id, dry_run=True)
            print(f"preflight ok: attempt {k} of Exp {exp} may start (nothing spawned, no marker pushed)")
            return 0
        marker = None if mock else push_marker(exp, k, run_id, dry_run=False)
    except Refused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        shutil.rmtree(home, ignore_errors=True)
        return 2
    finally:
        if args.command == "preflight":
            shutil.rmtree(home, ignore_errors=True)
    try:
        return run_attempt(args, exp, mock, rows_file, provenance, run_id, k, home, gguf, read_served, marker, stamps)
    finally:
        shutil.rmtree(home, ignore_errors=True)
        if lock is not None:
            lock.close()


def run_attempt(args, exp, mock, rows_file, provenance, run_id, k, home, gguf, read_served, marker, stamps=None) -> int:
    import signal  # noqa: PLC0415

    saved = {sig: signal.signal(sig, _raise_signal) for sig in (signal.SIGTERM, signal.SIGHUP)}
    try:
        return _run_phases(
            args, exp, mock, rows_file, provenance, run_id, k, home, gguf, read_served, marker, stamps or {}
        )
    finally:
        for sig, handler in saved.items():
            signal.signal(sig, handler)  # only while this attempt runs (a test process must get its own back)


def _run_phases(args, exp, mock, rows_file, provenance, run_id, k, home, gguf, read_served, marker, stamps) -> int:
    data_root = rows_file.parent
    print(f"attempt {k} of Exp {exp}: run {run_id}, marker {marker}, rows {rows_file}")
    phase1_session: str | None = None
    phase1_source: dict[str, str] = {}
    phase1_stores: set[str] = set()
    for index in range(len(v.PROTOCOL[exp]["phases"])):
        resume = phase1_session if v.PROTOCOL[exp]["phases"][index][3] else None
        base = {"harness": HARNESS, "exp": exp, "attempt_k": k, "phase_index": index, "marker": marker}
        row: dict = {**base, **stamps, "ts": time.time(), "provenance": provenance}
        problems: list[str] = []
        try:
            if index == 2 and phase1_session and tree_digests(home / "sim_reports" / phase1_session) != phase1_source:
                problems.append("phase 1's session changed before phase 3")
            else:
                phase = mock_phase if mock else spawn_phase
                sdir, report, fields, run_log, console, error = phase(
                    exp, index, home=home, run_id=run_id, resume=resume, provenance=provenance,
                    gguf=gguf, timeout_s=args.timeout_s, read_served=read_served,
                )  # fmt: skip
                row.update(fields)
                if sdir is not None:
                    row["session_id"] = sdir.name
                    row["files"] = copy_session(sdir, run_log, console, data_root / sdir.name)
                    if index == 0:
                        phase1_session = sdir.name
                        phase1_source = tree_digests(sdir)
                        phase1_stores = {s for s in v.RESUME_STORES if f"aut_{s}.json" in row["files"]}
                if error:
                    problems.append(error)
                if report is not None:
                    problems += v.complete_problems(exp, index, row, report, phase1_session, phase1_stores)
                elif not error:
                    problems.append("no report")
                last = index == len(v.PROTOCOL[exp]["phases"]) - 1
                if not problems and last and v.experiment_of(exp) == "63" and sdir is not None:
                    problems += exp63_attempt_problems(exp, data_root, [phase1_session, sdir.name])
        except BaseException as exc:
            # The attempt counts once its marker is pushed: an interrupted or crashed phase still writes its row.
            reason = f"harness interrupted: {type(exc).__name__}: {exc}"[:500]
            problems.append(reason)
            row.update({"status": "failed", "reason": reason, "complete_problems": problems})
            append_row(rows_file, _provenance.stamp_harness_row(row, mock=mock))
            raise
        if problems:
            row["status"] = "failed"
            row["reason"] = row.get("reason") or problems[0]
            row["complete_problems"] = problems
        _provenance.stamp_harness_row(row, mock=mock)
        assert row["record_kind"] == "harness_row"
        append_row(rows_file, row)
        name = v.PROTOCOL[exp]["phases"][index][0]
        print(f"  phase {index} ({name}): {row['status']}" + (f" — {problems}" if problems else ""))
        if problems:
            return 1
    if not mock:
        print(
            "Next: commit the rows file and the copied sessions to main (a merge-committed data PR); keep the rig here."
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", default="run", choices=["run", "preflight"])
    ap.add_argument("--exp", required=True, choices=sorted(v.PROTOCOL))
    ap.add_argument("--write-experiment-results", action="store_true")
    ap.add_argument("--mock", action="store_true", help="offline end to end: fabricated sims, a temp rows file")
    ap.add_argument("--timeout-s", type=float, default=3 * 3600.0, help="per phase")
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
