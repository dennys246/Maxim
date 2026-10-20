"""The serving, scenario and session half of a sim report's provenance (baseline floors B0).

``report.py::capture_start_provenance`` calls :func:`capture_serving_start` at sim entry and stamps only the
returned integer token (the stamp stays plain JSON); the capture itself is held here. ``report.py::run_provenance``
folds :func:`serving_provenance` into the report at the end. Design and the reason for every rule:
docs/plans/baseline_floors_b0.md.

What it establishes: which models and serving answered the sim's LLM calls (per router serial, over the sim's
window, with a completeness check), the scenario files its loaders read, the global seed, and the session
index. What it does NOT establish: which model served the agent under test (``aut_model_attribution`` is
always ``"not_established"``; B0b).

Overlap: sims register per thread. Sims on one thread run one after another, so a new sim drops its thread's
earlier registration (one that died before its report); registrations whose thread has ended are dropped too.
A sim whose window overlapped a live sim on another thread is stamped ``concurrent_sims`` and refused. This
covers one process only: two sim PROCESSES sharing a server each stamp ``False``. A sub-sim run synchronously
on its parent's thread replaces the parent's registration, so neither is marked; both windows then hold the
sub-sim's calls, which were made inside both. A sim that dies on a thread that lives on (a console's main
thread) stays registered until that thread starts another sim, so sims on other threads meanwhile read
concurrent (fail-closed).

The owned server's model file is hashed from its spawn (``runtime/llm_server.py::set_active_spawner``) once
this process has started a sim; a server spawned before the process's first sim is hashed from that sim's
start, so a file swapped before then is not seen. The digest is collected here with a bounded wait; an unfinished or failed hash, or a file that changed, stamps
``local_file_unhashed``, which the evidence gate refuses.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import threading
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from maxim.models.language import call_accounting
from maxim.simulation.report import redact_urls, report_url
from maxim.utils import scenario_inputs
from maxim.utils.file_digest import BackgroundDigest, file_stat

logger = logging.getLogger(__name__)

# How long report time waits for a model-file hash, and for a role router's in-flight call to finish.
HASH_JOIN_TIMEOUT_S = 300.0
ROUTER_IDLE_TIMEOUT_S = 30.0

_LOCK = threading.Lock()
_TOKENS = itertools.count(1)
_STARTS: dict[int, dict[str, Any]] = {}  # token -> the private start capture
_ACTIVE: dict[int, dict[str, Any]] = {}  # token -> {"thread": Thread, "overlapped": bool}


def _register_sim() -> int:
    me = threading.current_thread()
    with _LOCK:
        for token, reg in list(_ACTIVE.items()):
            if reg["thread"] is me or not reg["thread"].is_alive():
                del _ACTIVE[token]
                _STARTS.pop(token, None)
        token = next(_TOKENS)
        overlapped = bool(_ACTIVE)
        for reg in _ACTIVE.values():
            reg["overlapped"] = True
        _ACTIVE[token] = {"thread": me, "overlapped": overlapped}
        return token


def _release_sim(token: int) -> tuple[dict[str, Any] | None, bool]:
    """The start capture and whether the sim overlapped another (an unknown token reads as overlapped)."""
    with _LOCK:
        reg = _ACTIVE.pop(token, None)
        return _STARTS.pop(token, None), (reg is None or reg["overlapped"])


def capture_serving_start(*, scenario_objects: Iterable[Any] = ()) -> int:
    """Take the start capture and return its token. ``scenario_objects`` are what the sim was handed already
    loaded: an object with ``source_inputs`` (a DM campaign) has those files re-recorded into this sim's
    window; plain data (``pre_campaign_turns``, a list or dict) has its content hashed into the sim's
    ``scenario_sha256``, since its file was read before the sim or never recorded."""
    from maxim.runtime import llm_server

    token = _register_sim()
    mark = scenario_inputs.mark()
    content = []
    for obj in scenario_objects:
        if isinstance(obj, (list, dict)):
            content.append(obj)
        elif obj is not None:
            scenario_inputs.renote(getattr(obj, "source_inputs", None) or ())
    llm_server.request_spawn_digests()  # this process runs sims: hash owned models from their spawn
    capture = {
        "calls": call_accounting.snapshot(),
        "scenario_mark": mark,
        "scenario_content_sha256": _content_sha256(content) if content else None,
        "spawner": llm_server.active_spawner(),
    }
    with _LOCK:
        _STARTS[token] = capture
    return token


_LOOPBACK = ("localhost", "[::1]", "0.0.0.0")


def _content_sha256(content: list[Any]) -> str:
    """The SHA-256 of scenario data handed over as turns. Key order is normalised when the keys allow it; data
    whose keys cannot be sorted (mixed types) is hashed in its given order rather than failing the sim's start."""
    try:
        text = json.dumps(content, sort_keys=True, default=str)
    except TypeError:
        text = json.dumps(content, default=str)
    return hashlib.sha256(text.encode()).hexdigest()


def _root(url: str) -> str:
    """A base URL without its trailing ``/v1``, with loopback host names written one way (no DNS), so a
    provider configured as ``localhost`` matches the spawner's ``127.0.0.1``."""
    base = (url or "").rstrip("/")
    base = base[: -len("/v1")] if base.endswith("/v1") else base
    for name in _LOOPBACK:
        for sep in (":", "/"):
            base = base.replace(f"://{name}{sep}", f"://127.0.0.1{sep}")
        if base.endswith(f"://{name}"):
            base = base[: -len(name)] + "127.0.0.1"
    return base


def _owned_identity(start_spawner: Any | None) -> tuple[dict[str, Any] | None, bool]:
    """The owned server's identity fields, and whether the model changed during the run. A server spawned
    inside the window (none at the start) is a fresh spawn, not a change."""
    from maxim.runtime import llm_server

    now = llm_server.active_spawner()
    changed = start_spawner is not None and now is not start_spawner
    if now is None:
        return None, changed
    digest = llm_server.active_spawner_digest()
    sha = digest.result(HASH_JOIN_TIMEOUT_S) if digest is not None else None
    if digest is not None and file_stat(digest.path) != digest.stat:
        changed = True
    if sha is None:
        logger.warning(
            "serving provenance: owned model %s not hashed (%s)",
            getattr(now, "model_path", None),
            digest and digest.error,
        )
    return {
        "root": _root(now.base_url),
        "identity": sha,
        "identity_source": "local_file" if sha else "local_file_unhashed",
        "kv_quant": now.kv_cache_type,
    }, changed


def _join_identity(entry: dict[str, Any], owned: dict[str, Any] | None, llama_cache: dict[str, Any]) -> None:
    kind = entry["backend_kind"]
    if kind == "peer" and owned is not None and _root(entry["endpoint"]) == owned["root"]:
        entry.update({k: owned[k] for k in ("identity", "identity_source", "kv_quant")})
    elif kind == "llama":
        if entry["endpoint"] not in llama_cache:
            sha = BackgroundDigest(entry["endpoint"]).result(HASH_JOIN_TIMEOUT_S) if entry["endpoint"] else None
            llama_cache[entry["endpoint"]] = {
                "identity": sha,
                "identity_source": "local_file" if sha else "local_file_unhashed",
                "kv_quant": None,
            }
        entry.update(llama_cache[entry["endpoint"]])
    elif kind == "peer":
        entry.update({"identity": entry["served_id"] or None, "identity_source": "server_reported", "kv_quant": None})
    else:
        entry.update({"identity": entry["model"] or None, "identity_source": "provider_name", "kv_quant": None})


def _session_index(resume: dict[str, Any] | None, *, adopted: bool) -> int | None:
    """0 for a fresh run on a fresh agent (the sim AUT's stores live in a new temp dir); the prior report's
    index + 1 when a resume loaded; ``None`` whenever prior state could exist and its index is unknown."""
    if adopted:
        return None
    if resume is None:
        return 0
    if not resume.get("resume_loaded") or not resume.get("state_dir"):
        return None
    try:
        prior = json.loads((Path(resume["state_dir"]) / "report.json").read_text())
    except (OSError, ValueError) as e:
        logger.warning("serving provenance: prior report unreadable (%s)", e)
        return None
    index = (prior.get("provenance") or {}).get("session_index") if isinstance(prior, dict) else None
    return index + 1 if type(index) is int else None


def serving_provenance(
    token: Any,
    *,
    llm_worker: Any | None,
    aut_worker: Any | None,
    resume: dict[str, Any] | None,
    adopted: bool,
    goal: str,
) -> dict[str, Any]:
    """The B0 fields for the sim whose start capture is ``token``. Releases the sim's registration."""
    start, overlapped = _release_sim(token) if type(token) is int else (None, True)
    if start is None:
        logger.warning("serving provenance: no start capture for token %r", token)
        return {"serving_captured": False, "concurrent_sims": overlapped, "aut_model_attribution": "not_established"}
    routers = [getattr(w, "router", None) for w in (llm_worker, aut_worker)]
    for router in routers:
        if router is not None and hasattr(router, "wait_idle") and not router.wait_idle(ROUTER_IDLE_TIMEOUT_S):
            logger.warning("serving provenance: router %s still busy at report time", router.call_serial)
    end = call_accounting.snapshot()
    window, problems = call_accounting.delta(start["calls"], end)
    problems += call_accounting.window_problems(start["calls"], end, window)
    owned, changed = _owned_identity(start["spawner"])
    role_serials = {getattr(r, "call_serial", None) for r in routers if r is not None}
    llama_cache: dict[str, Any] = {}
    served_ids: dict[str, set[str]] = {}
    llm_calls: dict[str, Any] = {}
    for serial, r in sorted(window.items()):
        if serial not in role_serials and not any(r["counts"].values()):
            continue
        entries = []
        for key, n in r["entries"].items():
            entry = call_accounting.entry_dict(key, n)
            _join_identity(entry, owned, llama_cache)
            if entry["backend_kind"] != "llama":  # a llama endpoint is its model file path, not a URL
                entry["endpoint"] = report_url(entry["endpoint"])  # after the join, which matches the full URL (#997)
            entries.append(entry)
            if entry["backend_kind"] == "peer" and entry["served_id"]:
                served_ids.setdefault(entry["provider_key"], set()).add(entry["served_id"])
        llm_calls[str(serial)] = {"entries": entries, "counts": r["counts"]}
    changed = changed or any(len(ids) > 1 for ids in served_ids.values())
    heads: dict[str, str | None] = {}
    for router in {id(r): r for r in routers if r is not None and hasattr(r, "serving_probes")}.values():
        for probe in router.serving_probes():
            cached, fresh = probe["cached_served_id"], probe["fresh_served_id"]
            changed = changed or bool(cached and fresh and cached != fresh)
            heads[report_url(_root(probe["endpoint"]))] = probe["leader_disk_head"]
    inputs = scenario_inputs.since(start["scenario_mark"])
    content_sha = start["scenario_content_sha256"]
    scenario_sha = hashlib.sha256(
        json.dumps({"goal": goal, "inputs": inputs, "content": content_sha}, sort_keys=True).encode()
    ).hexdigest()
    from maxim.utils.seeding import get_global_seed

    problems = [redact_urls(p) for p in problems]  # an entry key in a problem string carries its endpoint (#997)
    if problems:
        logger.warning("serving provenance: incomplete call accounting: %s", "; ".join(problems))
    return {
        "serving_captured": True,
        "llm_calls": llm_calls,
        **{
            f"{role}_router_serial": getattr(getattr(w, "router", None), "call_serial", None)
            for role, w in (("language", llm_worker), ("aut", aut_worker))
        },
        "sampling_complete": not problems,
        "sampling_problems": problems,
        "model_changed_during_run": changed,
        "concurrent_sims": overlapped,
        "leader_disk_heads": heads,
        "scenario_inputs": inputs,
        "scenario_content_sha256": content_sha,
        "scenario_sha256": scenario_sha,
        "global_seed": get_global_seed(),
        "session_index": _session_index(resume, adopted=adopted),
        "aut_model_attribution": "not_established",
    }
