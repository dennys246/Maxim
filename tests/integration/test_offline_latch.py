"""#1315 item 10 strict red gates: the offline latch (V8, step 0b), its call-site refusals and its config key.

WHAT THIS FILE IS. Tests only, written before the code (``docs/plans/atonia_gate_approach_notes.md``, DRAFT v6.4:
§0, the V1–V10 decisions, §1A (which wins over §2–§6) and §6; ``docs/plans/advisor_mode.md`` §"Item 10"). Each gate
is ``xfail(strict=True, raises=AssertionError)``: it fails TODAY with an ``AssertionError`` for the reason its
docstring names, and the day the behaviour is right it XPASSes, which strict turns into a failure until the decorator
is removed. A gate that does not flip is data; do not re-point it.

THE GATES, AND WHAT FLIPS EACH
  * ``test_engaged_latch_refuses_*`` / ``test_engaged_latch_allows_inbound_listen`` /
    ``test_offline_refusal_is_a_runtime_error_not_an_oserror`` -- §6 layer 1, "While engaged, it refuses". One child
    engages and drives one call of each channel; each must raise ``OfflineRefusal`` and land in the ledger, and an
    inbound ``bind``/``listen`` on 127.0.0.1 must pass. Flip with step 0b (``maxim/_offline.py``).
  * ``test_self_engage_from_env`` / ``test_no_hook_without_the_variable`` -- §6 "Self-engage in children".
  * ``test_one_truthy_parser`` (parametrised) / ``test_annotation_parser_delegates_to_offline_truthy`` -- §1A.1.
  * ``test_offline_module_is_stdlib_only`` -- §1A.1 (the name §1A.1 gives the guard; it lives here, not in unit/).
  * ``test_offline_git_out_spawns_nothing`` / ``test_offline_code_stamp_says_offline`` /
    ``test_offline_version_info_spawns_nothing`` -- §1A.7.
  * ``test_no_effect_subprocess`` -- §6 "The red gate" / §1A.9 invariant 3: the census gate over ``CENSUS_SITES``,
    run with the conftest-forced ``HF_HUB_OFFLINE``/``TRANSFORMERS_OFFLINE`` deleted (advisor_mode.md §Item 10).
    Flips with item 10's call-site refusals on top of 0b.
  * ``test_offline_implies_internet_lowered`` -- §1A.1, the read-side coupling. ``test_evidence_records_refuse_an_
    offline_code_stamp`` is a passing guard (it holds today). ``test_offline_sim_report_says_offline`` -- §1A.7's
    composed gate at the report writer (its docstring states the scope).
  * ``test_a_client_pooled_before_engage_cannot_send_after`` / ``test_engage_sets_already_imported_library_flags``
    -- notes v6.4 §1A.1, ``engage()``'s steps 3 and 2. ``test_no_caller_of_reset_for_tests_outside_tests`` is a
    passing lint-style guard.
  * ``test_router_refuses_cloud_and_remote_lanes_at_build`` / ``test_router_refuses_a_pre_engage_lane_at_dispatch``
    -- §6 layer 3, the router predicate.
  * ``test_llm_offline_is_a_config_key`` / ``test_config_version_owns_llm_offline`` /
    ``test_downgrade_warns_when_it_drops_llm_offline`` -- §1A.6 / §6 "Config".
  * ``test_conftest_pops_maxim_offline_before_importing_maxim`` -- §1A.1 "Tests".

SEAMS THESE GATES PIN (the notes leave them open; chosen here, change them only with the implementing PR):
  * ``maxim._offline``: ``engage(reason: str) -> None``; ``is_offline() -> bool``; ``OfflineRefusal(RuntimeError)``;
    ``truthy(value: str | None) -> bool``; ``ledger() -> Sequence`` (one entry per refusal, appended BEFORE the
    raise, a fresh sequence per call); ``hook_installed() -> bool`` (whether THIS process ever added the audit hook);
    ``_reset_for_tests()`` (in ``src/`` per v6.4; it disengages and restores the five env keys, its callers are
    linted to ``tests/``, and the conftest autouse fixture calls it); ``on_engage(fn)`` and ``engage_callbacks()``
    (the read seam chosen here, a sequence of the registered callbacks).
  * "No hook without the variable" is checked by the installed flag AND behaviourally (a synthetic
    ``sys.audit("subprocess.Popen", ...)`` and ``sys.audit("socket.getaddrinfo", ...)`` pass), because Python
    offers no API that lists audit hooks.
  * The census sites a library backs (``ddgs``, ultralytics, faster-whisper, ``huggingface_hub``, twilio, the
    Anthropic SDK, ``sentence_transformers``) are NOT APPLICABLE where that library is not installed: reported in a
    warning and in the failure message, never counted as a pass, and the gate refuses to run with every site
    not-applicable. ``HF_BACKED_SITES`` (§6 layer 2: ``local_files_only`` + ``HF_HUB_OFFLINE``) also pass on
    ``OfflineModeIsEnabled`` or a degraded record: ``maxim._offline.degraded()`` (a sequence that grows) or a WARNING
    naming "degraded" during the call.
  * The census counts a site as refused when an ``OfflineRefusal`` is raised (anywhere in the exception chain) or the
    ledger grew during its call, AND the blocking spy (added before ``engage``) saw no connect, send,
    resolve or spawn while it ran. Heavy third-party libraries (``httpx``, ``huggingface_hub``,
    ``sentence_transformers`` when installed) are imported BEFORE ``engage`` so an import-time side effect is not
    charged to a site; that also exercises §6's "already imported" rule.
  * The router's build refusal is ``LLMRouter(cfg)`` raising ``OfflineRefusal``; its dispatch refusal is
    ``generate_json`` raising it (or the ledger growing) with the pre-built backend never dispatched to.
  * The downgrade warning is a WARNING naming ``llm.offline`` and saying the run goes ``online``.

THE SPY BLOCKS, AND IS NEVER THE PASS. Every child that could egress carries the test's own audit hook ``_spy``,
which records AND then raises ``_SpyBlocked(RuntimeError)`` on any connect (``AF_INET``/``AF_INET6``, loopback
included, and ``AF_UNIX``), any send to an INET destination, any forward or reverse resolve and any spawn. So neither a
broken latch nor an unfixed call site can make a CI child do a real connect, DNS lookup, ``docker pull`` or HF
download. Inbound ``bind``/``listen`` is not blocked. Where the gate needs to see the LATCH's refusal (the channel,
self-engage and router gates) the spy is installed AFTER ``engage()``: audit hooks run in installation order and the
first raise stops the rest, so the spy sees only what the latch let through. Where the gate is "refused before any
socket event" (the census and provenance gates) it is installed FIRST. Either way, a call the spy stopped is a FAILURE
(``egress reached the spy: the latch did not refuse``), never a pass; egress outside any driven call is
``ScenarioBroken``.

PRECONDITIONS ARE NOT THE GATE. A broken harness (a child that cannot report, a ``maxim`` imported from another
checkout, a scenario that did not happen) raises ``ScenarioBroken``, never an ``AssertionError``, so it FAILS the run
instead of hiding as an expected failure. A child that finds no ``maxim._offline`` reports it, and the parent turns
that into the existence ``AssertionError``.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import importlib.util
import json
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
OFFLINE_MODULE = SRC / "maxim" / "_offline.py"
CONFTEST = REPO / "tests" / "conftest.py"
SCHEMA_FIXTURE = REPO / "tests" / "fixtures" / "config_schema_by_version.json"
MARKER = "@@OFFLINE_GATE@@"
CHILD_TIMEOUT_S = 300

# §6's census, as the sites this gate drives. A new fetch site joins HERE and gets a driver in ``_CENSUS_DRIVERS``
# below; the child refuses to run a name it has no driver for (ScenarioBroken), so the two cannot drift apart.
CENSUS_SITES: tuple[str, ...] = (
    "utils.http.fetch_url",
    "utils.http.download_to_file",
    "utils.http.get[registered endpoint]",
    "models.download.ensure_available[missing profile]",
    "models.download.ensure_tts_voice[auto=True]",
    "similarity.encoder[missing model]",
    "simulation.container_runner.ensure_image",
    "hivemind.substrate_client.list_releases",
    "tools.internet_search[ddgs package]",
    "tools.internet_search._search_duckduckgo_lite[urllib3 bypass]",
    "models.vision.ultralytics_engine[attempt_download_asset]",
    "inference.transcribe_audio[faster-whisper WhisperModel]",
    "huggingface_hub.hf_hub_download[direct]",
    "comms.channels.twilio_channel.send",
    "models.vision.rtm_engine[download_vision]",
    "router[cloud lane dispatch]",
)

# HF-backed sites (§6 layer 2): ``local_files_only`` plus ``HF_HUB_OFFLINE`` is their handling, so they also pass on
# ``huggingface_hub``'s ``OfflineModeIsEnabled`` or a ``degraded`` record (``maxim._offline.degraded()`` growing, or
# a WARNING naming "degraded"), always with zero socket events.
HF_BACKED_SITES: frozenset[str] = frozenset(
    {
        "similarity.encoder[missing model]",
        "inference.transcribe_audio[faster-whisper WhisperModel]",
        "huggingface_hub.hf_hub_download[direct]",
    }
)

# The child's own spy: these audit events are a connect, a send, a resolve or a spawn.
_SPY_EVENT_PREFIXES = (
    "socket.connect",
    "socket.sendto",
    "socket.sendmsg",
    "socket.getaddrinfo",
    "socket.gethostbyname",
    "socket.gethostbyaddr",
    "socket.getnameinfo",
    "subprocess.Popen",
    "os.system",
    "os.exec",
    "os.posix_spawn",
    "os.spawn",
    "os.fork",
)

_HF_FORCED = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")

_OFFLINE_MISSING = "maxim._offline does not exist (step 0b not built)"
SPY_BLOCKED_MSG = "SPY_BLOCKED: egress reached the test spy"
_LATCH_DID_NOT_REFUSE = "egress reached the spy: the latch did not refuse"


def _assert_spy_saw_nothing(name: str, res: dict[str, Any]) -> None:
    """The spy is the safety net, never the pass: if it is what stopped the call, the gate fails."""
    assert not (res.get("spy_blocked") or res.get("spy_events")), (
        f"{name}: {_LATCH_DID_NOT_REFUSE} ({res.get('spy_events') or res.get('msg')})"
    )


class ScenarioBroken(Exception):
    """The scenario did not happen as designed: NOT the defect a gate pins (module docstring)."""


def require(cond: Any, msg: str) -> None:
    if not cond:
        raise ScenarioBroken(msg)


# ── the child harness ─────────────────────────────────────────────────────

_PRELUDE = textwrap.dedent(
    r"""
    import sys, json
    CFG = json.load(open(sys.argv[1]))
    sys.path.insert(0, CFG["src"])
    sys.dont_write_bytecode = True
    import os, importlib, importlib.util

    MARKER = CFG["marker"]
    OUT = {"spy": [], "error": None, "start_env": sorted(k for k in os.environ if k.startswith(("HF_", "TRANSFORMERS_", "MAXIM_")))}
    STATE = {"window": None}
    SPY = tuple(CFG["spy_prefixes"])

    import socket as _socket

    class _SpyBlocked(RuntimeError):
        pass  # raised by the test's own spy: egress got past Maxim's latch (or happened before it)

    SPY_BLOCKED = CFG["spy_blocked_msg"]
    OUT["spy_outside"] = []
    _INET = (_socket.AF_INET, _socket.AF_INET6)

    def _is_egress(event, args):
        if event == "socket.connect":  # connect and connect_ex; AF_UNIX too (docker.sock, forkserver)
            return getattr(args[0], "family", None) in (*_INET, _socket.AF_UNIX)
        if event in ("socket.sendto", "socket.sendmsg"):  # a None destination is a connected socket
            return args[1] is not None and getattr(args[0], "family", None) in _INET
        return event.startswith(SPY)  # resolve, reverse resolve, every spawn

    def _spy(event, args):
        # BLOCKS as well as records, so a broken latch or an unfixed call site can never make real egress.
        if not event.startswith(SPY) or not _is_egress(event, args):
            return
        rec = {"window": STATE["window"], "event": event, "args": repr(args)[:160]}
        if STATE["window"] == "__preimport__":
            OUT.setdefault("spy_preimport", []).append(rec)  # a library's import-time probe, blocked; not a site
        else:
            (OUT["spy"] if STATE["window"] is not None else OUT["spy_outside"]).append(rec)
        raise _SpyBlocked(f"{SPY_BLOCKED}: {event}")

    def install_spy():
        sys.addaudithook(_spy)

    if CFG.get("spy") == "first":  # before engage(): it sees (and stops) egress before the latch does
        install_spy()

    def emit():
        print(MARKER + json.dumps(OUT, default=repr), flush=True)

    def require_offline_module():
        try:
            spec = importlib.util.find_spec("maxim._offline")
        except ModuleNotFoundError:
            spec = None
        import maxim
        OUT["maxim_file"] = maxim.__file__
        if spec is None:
            OUT["missing"] = "maxim._offline"
            emit()
            sys.exit(0)
        return importlib.import_module("maxim._offline")

    def chain_has(exc, cls):
        seen = set()
        while exc is not None and id(exc) not in seen:
            seen.add(id(exc))
            if isinstance(exc, cls):
                return True
            exc = exc.__cause__ or exc.__context__
        return False

    import logging as _logging

    class _Warnings(_logging.Handler):
        def emit(self, record):
            if record.levelno >= _logging.WARNING and STATE["window"] is not None:
                OUT.setdefault("warnings", []).append({"window": STATE["window"], "msg": record.getMessage()[:200]})

    _logging.getLogger().addHandler(_Warnings())

    def chain_names(exc):
        names, seen = [], set()
        while exc is not None and id(exc) not in seen:
            seen.add(id(exc))
            names.append(type(exc).__name__)
            exc = exc.__cause__ or exc.__context__
        return names

    def _degraded_len(off):
        fn = getattr(off, "degraded", None)
        return len(fn()) if callable(fn) else 0

    def attempt(off, name, fn):
        before = len(off.ledger())
        degraded_before = _degraded_len(off)
        STATE["window"] = name
        try:
            fn()
            res = {"raised": None, "refusal": False, "spy_blocked": False}
        except BaseException as e:  # noqa: BLE001 -- the gate reports whatever the call did
            res = {
                "raised": type(e).__name__,
                "refusal": chain_has(e, off.OfflineRefusal),
                "spy_blocked": chain_has(e, _SpyBlocked),
                "hf_offline": "OfflineModeIsEnabled" in chain_names(e),
                "msg": str(e)[:200],
            }
        finally:
            STATE["window"] = None
        res["ledger_delta"] = len(off.ledger()) - before
        res["degraded"] = _degraded_len(off) > degraded_before or any(
            w["window"] == name and "degraded" in w["msg"].lower() for w in OUT.get("warnings", [])
        )
        res["spy_events"] = [e for e in OUT["spy"] if e["window"] == name]
        if res["spy_events"]:
            res["refusal"] = False  # the spy stopped it, whatever the chain says: never a pass
        return res
    """
)


def _run_child(
    tmp_path: Path,
    body: str,
    *,
    env_extra: dict[str, str] | None = None,
    spy: str | None = "first",
    cwd: Path | None = None,
    cfg_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run ``_PRELUDE + body`` in ``sys.executable -I`` with a fresh env (no HF_* / TRANSFORMERS_* / MAXIM_* from
    the pytest process: the conftest-forced HF vars are deleted by construction). Returns the child's report."""
    harness = tmp_path / "harness"
    harness.mkdir(exist_ok=True)
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    work = cwd if cwd is not None else tmp_path / "cwd"
    work.mkdir(exist_ok=True)
    script = harness / "child.py"
    script.write_text(_PRELUDE + "\n" + textwrap.dedent(body))
    cfg = {
        "src": str(SRC),
        "marker": MARKER,
        "spy": spy,
        "spy_prefixes": list(_SPY_EVENT_PREFIXES),
        "spy_blocked_msg": SPY_BLOCKED_MSG,
    }
    cfg.update(cfg_extra or {})
    cfg_path = harness / "cfg.json"
    cfg_path.write_text(json.dumps(cfg))
    env = {
        "HOME": str(home),
        "TMPDIR": str(tmp_path),
        "XDG_CONFIG_HOME": str(tmp_path / "xdg-config"),
        "XDG_CACHE_HOME": str(tmp_path / "xdg-cache"),
        "MPLCONFIGDIR": str(tmp_path / "mpl"),
        "PYTHONPATH": str(SRC),  # ignored under -I; the child inserts SRC itself from its config
        "PATH": "/usr/bin:/bin",
        "MAXIM_SKIP_REMOTE_PROBE": "1",  # no build-time probe of a lane URL (the router children's 127.0.0.1:9)
    }
    env.update(env_extra or {})
    for var in _HF_FORCED:
        require(var not in env, f"{var} leaked into the child env: the census would pass for no reason")
    proc = subprocess.run(  # noqa: S603 -- our own interpreter, our own script
        [sys.executable, "-I", "-B", str(script), str(cfg_path)],
        cwd=str(work),
        env=env,
        capture_output=True,
        text=True,
        timeout=CHILD_TIMEOUT_S,
    )
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith(MARKER)), None)
    if line is None:
        raise ScenarioBroken(f"child did not report (rc={proc.returncode}): {proc.stderr[-3000:]}")
    out = json.loads(line[len(MARKER) :])
    require(
        str(out.get("maxim_file", "")).startswith(str(SRC)),
        f"the child imported a maxim outside this checkout: {out.get('maxim_file')!r}",
    )
    require(not out.get("spy_outside"), f"egress outside any driven call reached the spy: {out.get('spy_outside')}")
    out["_stdout"] = proc.stdout
    out["_stderr"] = proc.stderr
    return out


def _assert_offline_built(out: dict[str, Any]) -> None:
    assert out.get("missing") != "maxim._offline", _OFFLINE_MISSING


def _require_offline_in_process() -> Any:
    try:
        spec = importlib.util.find_spec("maxim._offline")
    except ModuleNotFoundError:
        spec = None
    assert spec is not None, _OFFLINE_MISSING
    return importlib.import_module("maxim._offline")


def _reason(item: str, flips: str) -> str:
    return f"#1315 item 10 {item}: {flips}"


# ── 1. engaging the latch refuses every outbound channel ──────────────────

_ENGAGE_BODY = r"""
import socket, subprocess, tempfile
off = require_offline_module()
OUT["refusal_mro"] = [c.__name__ for c in off.OfflineRefusal.__mro__]
OUT["is_runtime_error"] = issubclass(off.OfflineRefusal, RuntimeError)
OUT["is_oserror"] = issubclass(off.OfflineRefusal, OSError)
pair = socket.socketpair()  # made BEFORE engage: its fd is adopted after it (family=-1)
off.engage("test")
install_spy()  # AFTER the latch: hooks run in installation order, so the latch refuses first
OUT["engaged"] = off.is_offline()

def _connect4():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.settimeout(1); s.connect(("127.0.0.1", 9))
def _connect4_remote():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.settimeout(1); s.connect(("192.0.2.1", 9))
def _connect6():
    s = socket.socket(socket.AF_INET6, socket.SOCK_STREAM); s.settimeout(1); s.connect(("::1", 9, 0, 0))
def _sendto():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.sendto(b"x", ("192.0.2.1", 53))
def _getaddrinfo():
    socket.getaddrinfo("example.com", 80)
def _gethostbyaddr():
    socket.gethostbyaddr("8.8.8.8")
UNIX_DIR = tempfile.mkdtemp(dir="/tmp")  # short: macOS caps sun_path at 103 bytes, checked BEFORE the audit event
def _unix():
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); s.connect(os.path.join(UNIX_DIR, "s.sock"))
FAMILY = next((f for f in ("AF_PACKET", "AF_NETLINK", "AF_SYSTEM", "AF_ROUTE", "AF_VSOCK", "AF_BLUETOOTH")
               if hasattr(socket, f)), None)
OUT["exotic_family"] = FAMILY
def _exotic():
    socket.socket(getattr(socket, FAMILY), socket.SOCK_RAW)
def _adopt():
    socket.socket(fileno=os.dup(pair[0].fileno()))
def _popen():
    subprocess.Popen(["/bin/true"]).wait()
def _system():
    os.system("true")
def _posix_spawn():
    pid = os.posix_spawn("/bin/true", ["/bin/true"], dict(os.environ)); os.waitpid(pid, 0)
def _inbound():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.bind(("127.0.0.1", 0)); s.listen(1); s.close()

OUT["attempts"] = {name: attempt(off, name, fn) for name, fn in [
    ("ipv4_connect_loopback", _connect4), ("ipv4_connect", _connect4_remote), ("ipv6_connect", _connect6),
    ("udp_sendto_literal", _sendto), ("getaddrinfo", _getaddrinfo), ("gethostbyaddr", _gethostbyaddr),
    ("af_unix_connect", _unix), ("socket_non_inet_family", _exotic), ("fd_adoption_family_minus_1", _adopt),
    ("subprocess_popen", _popen), ("os_system", _system), ("os_posix_spawn", _posix_spawn),
    ("inbound_bind_listen", _inbound),
]}
emit()
"""

REFUSED_CHANNELS = (
    "ipv4_connect_loopback",
    "ipv4_connect",
    "ipv6_connect",
    "udp_sendto_literal",
    "getaddrinfo",
    "gethostbyaddr",
    "af_unix_connect",
    "socket_non_inet_family",
    "fd_adoption_family_minus_1",
    "subprocess_popen",
    "os_system",
    "os_posix_spawn",
)


@pytest.fixture(scope="module")
def engaged_child(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return _run_child(tmp_path_factory.mktemp("engage"), _ENGAGE_BODY, spy="body")


@pytest.mark.parametrize("channel", REFUSED_CHANNELS)
def test_engaged_latch_refuses_each_channel(engaged_child: dict[str, Any], channel: str) -> None:
    """After ``engage("test")`` one call of this channel raises ``OfflineRefusal`` and the ledger records it (§6:
    "While engaged, it refuses"). Fails today: ``maxim._offline`` does not exist."""
    _assert_offline_built(engaged_child)
    require(engaged_child.get("engaged") is True, "engage() did not latch: is_offline() is not True")
    if channel == "socket_non_inet_family":
        require(engaged_child.get("exotic_family"), "no non-INET socket family is available on this platform")
    res = engaged_child["attempts"][channel]
    _assert_spy_saw_nothing(channel, res)
    assert res["refusal"], f"{channel}: not refused with OfflineRefusal (raised {res['raised']}: {res.get('msg')})"
    assert res["ledger_delta"] >= 1, f"{channel}: refused but the refusal ledger did not record it"


def test_engaged_latch_allows_inbound_listen(engaged_child: dict[str, Any]) -> None:
    """Inbound ``bind``/``listen`` on 127.0.0.1 is NOT refused (§6: "Inbound is not refused"). Fails today on the
    existence check (``maxim._offline`` does not exist)."""
    _assert_offline_built(engaged_child)
    res = engaged_child["attempts"]["inbound_bind_listen"]
    _assert_spy_saw_nothing("inbound_bind_listen", res)
    assert res["raised"] is None, f"inbound bind/listen raised {res['raised']}: {res.get('msg')}"
    assert res["ledger_delta"] == 0, "an inbound bind/listen was booked as a refusal"


def test_offline_refusal_is_a_runtime_error_not_an_oserror(engaged_child: dict[str, Any]) -> None:
    """``OfflineRefusal`` subclasses ``RuntimeError`` and never ``OSError``, so the narrow ``except OSError`` sites
    (``git_out``, ``_find_orphans_ps``, ``git_root_refusal``) cannot absorb it. Fails today: no module."""
    _assert_offline_built(engaged_child)
    assert engaged_child["is_runtime_error"], f"OfflineRefusal MRO {engaged_child['refusal_mro']} has no RuntimeError"
    assert not engaged_child["is_oserror"], f"OfflineRefusal MRO {engaged_child['refusal_mro']} includes OSError"


# ── 2. self-engage in children ────────────────────────────────────────────

_SELF_ENGAGE_BODY = r"""
import socket
import maxim
OUT["maxim_file"] = maxim.__file__
off = sys.modules.get("maxim._offline")
OUT["loaded_by_import_maxim"] = off is not None
if off is None:
    try:
        spec = importlib.util.find_spec("maxim._offline")
    except ModuleNotFoundError:
        spec = None
    if spec is None:
        OUT["missing"] = "maxim._offline"
        emit(); sys.exit(0)
    off = importlib.import_module("maxim._offline")
OUT["is_offline"] = off.is_offline()
install_spy()  # after the self-engage, so the latch refuses first
r = attempt(off, "getaddrinfo", lambda: socket.getaddrinfo("example.com", 80))
OUT["getaddrinfo"] = r
emit()
"""


def test_self_engage_from_env(tmp_path: Path) -> None:
    """A child started with ``MAXIM_OFFLINE=yes`` (a canonical truthy value that is not "1") is latched after
    ``import maxim``, and the latch is real (a resolve refuses). Fails today: no ``maxim._offline``."""
    out = _run_child(tmp_path, _SELF_ENGAGE_BODY, env_extra={"MAXIM_OFFLINE": "yes"}, spy="body")
    _assert_offline_built(out)
    assert out["loaded_by_import_maxim"], "import maxim did not even load maxim._offline under MAXIM_OFFLINE=yes"
    assert out["is_offline"] is True, "MAXIM_OFFLINE=yes did not latch the child at import maxim"
    _assert_spy_saw_nothing("getaddrinfo", out["getaddrinfo"])
    assert out["getaddrinfo"]["refusal"], f"self-engaged, yet getaddrinfo was not refused: {out['getaddrinfo']}"


_NO_HOOK_BODY = r"""
import maxim
OUT["maxim_file"] = maxim.__file__
off = require_offline_module()
OUT["hook_installed"] = off.hook_installed()
OUT["is_offline"] = off.is_offline()
synthetic = {}
for event, args in (("subprocess.Popen", ("/bin/true", ["/bin/true"], None, None)),
                    ("socket.getaddrinfo", ("example.com", 80, 0, 0, 0))):
    try:
        sys.audit(event, *args)
        synthetic[event] = None
    except BaseException as e:  # noqa: BLE001
        synthetic[event] = type(e).__name__
OUT["synthetic_audit_raised"] = synthetic
emit()
"""


def test_no_hook_without_the_variable(tmp_path: Path) -> None:
    """Without ``MAXIM_OFFLINE`` the child carries NO audit hook after ``import maxim``: ``hook_installed()`` is
    False and a synthetic spawn / resolve audit event passes (module docstring: there is no API that lists hooks).
    Fails today: no ``maxim._offline`` (the existence check)."""
    out = _run_child(tmp_path, _NO_HOOK_BODY, spy=None)  # no spy: the gate is that NO hook exists
    _assert_offline_built(out)
    assert out["hook_installed"] is False, "an audit hook was installed in a process that never engaged"
    assert out["is_offline"] is False, "the child is offline without MAXIM_OFFLINE"
    assert all(v is None for v in out["synthetic_audit_raised"].values()), out["synthetic_audit_raised"]


# ── 3. one parser ─────────────────────────────────────────────────────────

TRUTHY_TABLE: tuple[tuple[str | None, bool], ...] = (
    ("1", True),
    ("true", True),
    ("TRUE", True),
    ("yes", True),
    ("Yes", True),
    ("on", True),
    (" on ", True),
    ("0", False),
    ("false", False),
    ("no", False),
    ("off", False),
    ("", False),
    (None, False),
    ("2", False),
    ("y", False),
    ("enabled", False),
)


@pytest.mark.parametrize(("raw", "expected"), TRUTHY_TABLE)
def test_one_truthy_parser(raw: str | None, expected: bool) -> None:
    """``maxim._offline.truthy`` and ``cluster_bias_annotation.annotation_disabled_via_env`` agree on every row.
    Fails today: no ``maxim._offline``."""
    off = _require_offline_in_process()
    from maxim.prompts.cluster_bias_annotation import annotation_disabled_via_env

    assert off.truthy(raw) is expected, f"truthy({raw!r}) != {expected}"
    assert annotation_disabled_via_env(raw) is off.truthy(raw), f"the two parsers disagree on {raw!r}"


def test_annotation_parser_delegates_to_offline_truthy(monkeypatch: pytest.MonkeyPatch) -> None:
    """``annotation_disabled_via_env`` calls ``maxim._offline.truthy`` (one parser, not two that agree today): a spy
    swapped in for ``truthy`` (on the module, and on ``cluster_bias_annotation`` when it binds the name) is called.
    Fails today: no ``maxim._offline``."""
    off = _require_offline_in_process()
    from maxim.prompts import cluster_bias_annotation as cba

    calls: list[Any] = []
    real = off.truthy

    def spy(value: Any) -> bool:
        calls.append(value)
        return real(value)

    monkeypatch.setattr(off, "truthy", spy)
    if getattr(cba, "truthy", None) is real:
        monkeypatch.setattr(cba, "truthy", spy)
    cba.annotation_disabled_via_env("yes")
    assert calls == ["yes"], "annotation_disabled_via_env did not delegate to maxim._offline.truthy"


# ── 4. stdlib only ────────────────────────────────────────────────────────


def test_offline_module_is_stdlib_only() -> None:
    """``maxim/_offline.py`` imports only the standard library (the ``test_clean_flag_998`` AST pattern), so importing
    it never runs ``maxim.utils``. Fails today: the file does not exist."""
    assert OFFLINE_MODULE.is_file(), _OFFLINE_MISSING
    imported: set[str] = set()
    for node in ast.walk(ast.parse(OFFLINE_MODULE.read_text())):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "maxim/_offline.py uses a relative import"
            if node.module:
                imported.add(node.module.split(".")[0])
    assert imported <= set(sys.stdlib_module_names) | {"__future__"}, imported - set(sys.stdlib_module_names)


# ── 5. offline provenance (§1A.7) ─────────────────────────────────────────

_PROVENANCE_BODY = r"""
from pathlib import Path
off = require_offline_module()
from maxim.utils import code_tree
from maxim.simulation import report as sim_report
import maxim
off.engage("test")
which = CFG["which"]
res = {}
STATE["window"] = which
try:
    if which == "git_out":
        res["value"] = code_tree.git_out(Path(CFG["repo"]), "rev-parse", "HEAD")
    elif which == "code_stamp":
        res["value"] = sim_report._code_stamp(Path(maxim.__file__).resolve())
    elif which == "version_info":
        res["value"] = maxim.get_version_info()
    res["raised"] = None
except BaseException as e:  # noqa: BLE001
    res["raised"] = type(e).__name__
    res["spy_blocked"] = chain_has(e, _SpyBlocked)
    res["msg"] = str(e)[:200]
finally:
    STATE["window"] = None
OUT["result"] = res
emit()
"""


def _provenance_child(tmp_path: Path, which: str) -> dict[str, Any]:
    out = _run_child(tmp_path, _PROVENANCE_BODY, cwd=REPO, cfg_extra={"which": which, "repo": str(REPO)})
    _assert_offline_built(out)
    require((REPO / ".git").exists(), "the gate needs a git checkout, and this tree has no .git")
    return out


def test_offline_git_out_spawns_nothing(tmp_path: Path) -> None:
    """Latched, ``code_tree.git_out([...])`` returns ``None`` and spawns nothing (its stdlib probe of
    ``sys.modules["maxim._offline"]``). Fails today: no ``maxim._offline``."""
    out = _provenance_child(tmp_path, "git_out")
    res = out["result"]
    _assert_spy_saw_nothing("git_out", {"spy_events": out["spy"], **res})
    assert res["raised"] is None, f"git_out raised {res['raised']} under offline: {res.get('msg')}"
    assert res["value"] is None, f"git_out returned {res['value']!r} under offline"
    assert not out["spy"], f"git_out spawned under offline: {out['spy']}"


def test_offline_code_stamp_says_offline(tmp_path: Path) -> None:
    """Latched, in a git checkout, ``simulation/report.py::_code_stamp`` yields ``executed_git_hash ==
    "unknown: offline"`` and spawns nothing. Fails today: no ``maxim._offline``."""
    out = _provenance_child(tmp_path, "code_stamp")
    res = out["result"]
    _assert_spy_saw_nothing("_code_stamp", {"spy_events": out["spy"], **res})
    assert res["raised"] is None, f"_code_stamp raised {res['raised']} under offline: {res.get('msg')}"
    assert res["value"]["executed_git_hash"] == "unknown: offline", res["value"]
    assert not out["spy"], f"_code_stamp spawned under offline: {out['spy']}"


def test_offline_version_info_spawns_nothing(tmp_path: Path) -> None:
    """Latched, ``maxim.get_version_info()`` spawns nothing (an ``is_offline()`` early return, not a swallowed
    refusal). Fails today: no ``maxim._offline``."""
    out = _provenance_child(tmp_path, "version_info")
    res = out["result"]
    _assert_spy_saw_nothing("get_version_info", {"spy_events": out["spy"], **res})
    assert res["raised"] is None, f"get_version_info raised {res['raised']} under offline: {res.get('msg')}"
    assert not out["spy"], f"get_version_info spawned (or tried to) under offline: {out['spy']}"


# ── 6. the item 10 census gate ────────────────────────────────────────────

_CENSUS_BODY = r"""
import dataclasses, tempfile
from pathlib import Path
from unittest.mock import MagicMock
off = require_offline_module()
TMP = Path(tempfile.mkdtemp())
# Pre-import the heavy libraries BEFORE engage (module docstring): an import-time side effect is not a site's. A
# library's import-time probe (ultralytics' is_online) is blocked by the spy and reported as "spy_preimport".
OPTIONAL = ("httpx", "huggingface_hub", "sentence_transformers", "ddgs", "ultralytics", "faster_whisper", "twilio",
            "anthropic")
AVAILABLE = {}
STATE["window"] = "__preimport__"
for mod in OPTIONAL:
    AVAILABLE[mod] = importlib.util.find_spec(mod) is not None  # installed: its site is driven, whatever the import does
    if not AVAILABLE[mod]:
        continue
    try:
        importlib.import_module(mod)
    except Exception as e:  # noqa: BLE001 -- recorded, never a silent miss; the site's driver imports it again
        OUT.setdefault("preimport_failed", {})[mod] = type(e).__name__ + ": " + str(e)[:120]
STATE["window"] = None
OUT["available"] = AVAILABLE
# ddgs dials through a native client (primp) that raises no audit event, so neither the latch nor the spy sees it.
# Its constructor is the spy for that site: reaching it is egress (recorded and blocked), never a pass.
if "ddgs" in sys.modules:
    class _DDGSBlocked:
        def __init__(self, *a, **k):
            OUT["spy"].append({"window": STATE["window"], "event": "ddgs.DDGS (native client, latch-blind)", "args": ""})
            raise _SpyBlocked(SPY_BLOCKED + ": ddgs.DDGS")
    sys.modules["ddgs"].DDGS = _DDGSBlocked
from maxim.utils import http
from maxim.models import download
from maxim.similarity import encoder
from maxim.simulation import container_runner
from maxim.hivemind import substrate_client
from maxim.tools import internet_search

def _get_registered():
    http.register_endpoint(http.HTTPEndpoint(name="offline-gate-probe", base_url="http://127.0.0.1:9"))
    http.get("offline-gate-probe", "/")

def _encoder():
    # The model load every LinguisticEncoder goes through (``_ensure_model``); ec/atl are not on the fetch path.
    enc = encoder.LinguisticEncoder(MagicMock(), MagicMock(), encoder.EncoderConfig(model_name="maxim-offline-gate/missing-model"))
    enc._ensure_model()

def _ultralytics():
    from maxim.models.vision import ultralytics_engine
    ultralytics_engine._ensure_weight_file(str(TMP / "yolo" / "yolov8n-seg.pt"), "yolov8n-seg.pt")

def _whisper():
    from maxim.inference.transcribe_audio import WhisperTranscriber
    WhisperTranscriber(model_size_or_path="tiny")

def _hf_direct():
    import huggingface_hub
    huggingface_hub.hf_hub_download("gpt2", "config.json", cache_dir=str(TMP / "hf"))

def _twilio():
    from maxim.comms.channels.twilio_channel import TwilioChannel
    ch = TwilioChannel({"account_sid": "AC" + "0" * 32, "auth_token": "x", "from_number": "+15550000000"},
                       conv_manager=MagicMock(), gateway=MagicMock())
    ch.send("+15550001111", "hello")

def _rtm():
    from maxim.models.vision import rtm_engine
    rtm_engine._resolve_model_path("rtmdet-m.onnx", str(TMP / "rtm"))

def _cloud_lane():
    from maxim.models.language.config import LLMConfig
    from maxim.models.language.router import LLMRouter
    cfg = dataclasses.replace(
        LLMConfig(), enabled=True, cloud_enabled=True,
        routing={"max_cost_per_request": 1.0, "max_cost_per_hour": 1.0, "max_cost_per_day": 1.0,
                 "max_cost_per_month": 1.0},
        providers={"claude": {"type": "anthropic", "model": "claude-sonnet-4-6", "api_key_env": "ANTHROPIC_API_KEY",
                              "pricing_required": False, "redaction_policy": "standard"}})
    LLMRouter(cfg).generate_json("hello")

DRIVERS = {
    "utils.http.fetch_url": lambda: http.fetch_url("https://example.com/"),
    "utils.http.download_to_file": lambda: http.download_to_file("https://example.com/f.bin", TMP / "f.bin"),
    "utils.http.get[registered endpoint]": _get_registered,
    "models.download.ensure_available[missing profile]":
        lambda: download.ensure_available("smollm-135m-instruct", auto=True, interactive=False),
    "models.download.ensure_tts_voice[auto=True]":
        lambda: download.ensure_tts_voice(auto=True, interactive=False, models_dir=TMP / "tts"),
    "similarity.encoder[missing model]": _encoder,
    "simulation.container_runner.ensure_image":
        lambda: container_runner.LocalDockerRunner().ensure_image("alpine:3.19"),
    "hivemind.substrate_client.list_releases": lambda: substrate_client.list_releases("https://oasis.invalid"),
    "tools.internet_search[ddgs package]": lambda: internet_search._search_with_ddg_package("maxim", 1),
    "tools.internet_search._search_duckduckgo_lite[urllib3 bypass]":
        lambda: internet_search._search_duckduckgo_lite("maxim", 1),
    "models.vision.ultralytics_engine[attempt_download_asset]": _ultralytics,
    "inference.transcribe_audio[faster-whisper WhisperModel]": _whisper,
    "huggingface_hub.hf_hub_download[direct]": _hf_direct,
    "comms.channels.twilio_channel.send": _twilio,
    "models.vision.rtm_engine[download_vision]": _rtm,
    "router[cloud lane dispatch]": _cloud_lane,
}
NEEDS = {
    "similarity.encoder[missing model]": "sentence_transformers",
    "tools.internet_search[ddgs package]": "ddgs",
    "models.vision.ultralytics_engine[attempt_download_asset]": "ultralytics",
    "inference.transcribe_audio[faster-whisper WhisperModel]": "faster_whisper",
    "huggingface_hub.hf_hub_download[direct]": "huggingface_hub",
    "comms.channels.twilio_channel.send": "twilio",
    "router[cloud lane dispatch]": "anthropic",
}
unknown = [s for s in CFG["sites"] if s not in DRIVERS]
if unknown:
    OUT["error"] = "no driver for census site(s): " + ", ".join(unknown)
    emit(); sys.exit(0)
off.engage("test")
OUT["sites"] = {}
for site in CFG["sites"]:
    lib = NEEDS.get(site)
    if lib is not None and not AVAILABLE.get(lib):
        OUT["sites"][site] = {"not_applicable": f"{lib} is not installed"}
        continue
    OUT["sites"][site] = attempt(off, site, DRIVERS[site])
emit()
"""


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=_reason("census gate", "each CENSUS_SITES call site refuses before any socket/spawn event"),
)
def test_no_effect_subprocess(tmp_path: Path) -> None:
    """With the conftest-forced ``HF_HUB_OFFLINE``/``TRANSFORMERS_OFFLINE`` deleted from the child's env, every
    site in ``CENSUS_SITES`` raises or records ``OfflineRefusal`` before any socket event, and the spy hook saw no
    connect, send, resolve or spawn while it ran. Fails today: no ``maxim._offline``."""
    out = _run_child(
        tmp_path,
        _CENSUS_BODY,
        cfg_extra={"sites": list(CENSUS_SITES)},
        env_extra={"ANTHROPIC_API_KEY": "sk-ant-red-gate-fake"},  # the cloud lane would really dial without a refusal
    )
    _assert_offline_built(out)
    require(out.get("error") is None, str(out.get("error")))
    require(not any(k in out["start_env"] for k in _HF_FORCED), f"HF vars present in the child: {out['start_env']}")
    require(sorted(out["sites"]) == sorted(CENSUS_SITES), f"the child drove {sorted(out['sites'])}")
    failures: list[str] = []
    not_applicable = {site: res["not_applicable"] for site, res in out["sites"].items() if "not_applicable" in res}
    require(len(not_applicable) < len(CENSUS_SITES), "every census site is not-applicable: nothing was driven")
    for site, res in out["sites"].items():
        if site in not_applicable:
            continue
        events = [e for e in out["spy"] if e["window"] == site]
        if events or res.get("spy_blocked"):
            failures.append(f"{site}: {_LATCH_DID_NOT_REFUSE} (nor the call site): {events[:3] or res.get('msg')}")
            continue
        refused = res["refusal"] or res["ledger_delta"] > 0
        if site in HF_BACKED_SITES:
            refused = refused or res.get("hf_offline") or res.get("degraded")
        if not refused:
            failures.append(f"{site}: no OfflineRefusal raised or recorded (raised {res['raised']}: {res.get('msg')})")
    if not_applicable:  # reported, never counted as a pass
        warnings.warn(f"census sites not applicable here (library not installed): {not_applicable}", stacklevel=1)
    na = f"\n(not applicable, not passed: {sorted(not_applicable)})" if not_applicable else ""
    assert not failures, "\n".join(failures) + na


# ── 7. the router predicate ───────────────────────────────────────────────

_ROUTER_BUILD_BODY = r"""
import dataclasses
off = require_offline_module()
from maxim.models.language.config import LLMConfig
from maxim.models.language.router import LLMRouter
LANES = {
    "cloud": {"claude": {"type": "anthropic", "model": "claude-sonnet-4-6", "api_key_env": "ANTHROPIC_API_KEY",
                         "pricing_required": False}},
    "remote": {"lane": {"type": "maxim_peer", "base_url": "http://127.0.0.1:9/v1", "api_key_env": "FAKE_KEY",
                        "model": "m", "allow_local_endpoints": True, "pricing_required": False}},
}
off.engage("test")
install_spy()
OUT["build"] = {}
for kind, providers in LANES.items():
    cfg = dataclasses.replace(LLMConfig(), enabled=True, cloud_enabled=(kind == "cloud"), providers=providers)
    OUT["build"][kind] = attempt(off, kind, lambda: LLMRouter(cfg))
emit()
"""


@pytest.mark.xfail(
    strict=True, raises=AssertionError, reason=_reason("§6 layer 3", "the router predicate refuses at build")
)
def test_router_refuses_cloud_and_remote_lanes_at_build(tmp_path: Path) -> None:
    """Latched, building a router with a cloud lane or a remote (endpoint) lane raises ``OfflineRefusal`` at build
    (V8: only in-process ``llama_cpp``/``transformers`` lanes run offline). Fails today: no ``maxim._offline``."""
    out = _run_child(tmp_path, _ROUTER_BUILD_BODY, spy="body")
    _assert_offline_built(out)
    for kind, res in out["build"].items():
        _assert_spy_saw_nothing(f"{kind} lane build", res)
        assert res["refusal"], f"{kind} lane: LLMRouter built under offline (raised {res['raised']})"


_ROUTER_DISPATCH_BODY = r"""
import dataclasses
from unittest.mock import MagicMock
off = require_offline_module()
from maxim.models.language.config import LLMConfig
from maxim.models.language.router import LLMRouter
cfg = dataclasses.replace(LLMConfig(), enabled=True, providers={"lane": {
    "type": "maxim_peer", "base_url": "http://127.0.0.1:9/v1", "api_key_env": "FAKE_KEY", "model": "m",
    "allow_local_endpoints": True, "pricing_required": False}})
router = LLMRouter(cfg)
fake = MagicMock()
fake.complete.return_value = '{"ok": 1}'
fake.complete_with_usage.return_value = ('{"ok": 1}', None)
router._backends["lane"] = fake
router._backend = fake

def dispatched():
    return any(getattr(fake, m).called for m in ("complete", "complete_with_usage", "generate", "generate_json"))

router.generate_json("hello")
OUT["pre_engage_dispatched"] = dispatched()
fake.reset_mock()
off.engage("test")
install_spy()
OUT["post"] = attempt(off, "dispatch", lambda: router.generate_json("hello"))
OUT["post_engage_dispatched"] = dispatched()
emit()
"""


@pytest.mark.xfail(
    strict=True, raises=AssertionError, reason=_reason("§6 layer 3", "the router predicate is checked at dispatch")
)
def test_router_refuses_a_pre_engage_lane_at_dispatch(tmp_path: Path) -> None:
    """A lane built BEFORE ``engage()`` refuses at dispatch: ``generate_json`` raises (or records) ``OfflineRefusal``
    and the live backend is never dispatched to. Fails today: no ``maxim._offline``."""
    out = _run_child(tmp_path, _ROUTER_DISPATCH_BODY, spy="body")
    _assert_offline_built(out)
    require(out["pre_engage_dispatched"], "before engage the router never reached the fake backend")
    assert not out["post_engage_dispatched"], "a pre-engage lane was dispatched to after engage()"
    res = out["post"]
    _assert_spy_saw_nothing("dispatch", res)
    assert res["refusal"] or res["ledger_delta"] > 0, f"dispatch was not refused as offline: {res}"


# ── 7b. the read-side couplings and the composed report (arch review S14) ─

_LOWERED_BODY = r"""
off = require_offline_module()
off.engage("test")
install_spy()  # after the latch
from maxim.utils.internet_access import session_internet_lowered
STATE["window"] = "lowered"
OUT["lowered"] = session_internet_lowered()
STATE["window"] = None
OUT["env_lowered"] = os.environ.get("MAXIM_INTERNET_LOWERED")
emit()
"""


def test_offline_implies_internet_lowered(tmp_path: Path) -> None:
    """Latched, ``utils/internet_access.py::session_internet_lowered()`` returns True on the READ side, with
    ``MAXIM_INTERNET_LOWERED`` never written (``engage()`` calls nothing in ``maxim.utils``). Fails today: no
    ``maxim._offline``."""
    out = _run_child(tmp_path, _LOWERED_BODY, spy="body")
    _assert_offline_built(out)
    assert out["lowered"] is True, "session_internet_lowered() is False with the offline latch engaged"
    assert out["env_lowered"] is None, f"engage() wrote MAXIM_INTERNET_LOWERED={out['env_lowered']!r}"


def test_evidence_records_refuse_an_offline_code_stamp() -> None:
    """``scripts/_evidence_records.py::unknown`` treats ``"unknown: offline"`` as unknown, so an offline run's
    ``executed_git_hash`` cannot gate a claim (§1A.7, M1). A guard: it holds today (the ``"unknown:"`` prefix)."""
    scripts = str(REPO / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    records = importlib.import_module("_evidence_records")
    assert records.unknown("unknown: offline") is True, "an offline code stamp reads as a known commit"
    assert records.unknown("0" * 40) is False, "the check refuses a real-looking hash too: it discriminates nothing"


_REPORT_BODY = r"""
import json as _json, tempfile
from pathlib import Path
off = require_offline_module()
from maxim.simulation import report as sim_report
TMP = Path(tempfile.mkdtemp())
off.engage("test")
STATE["window"] = "report"
try:
    start = sim_report.capture_start_provenance()
    prov = sim_report.run_provenance(start, llm_worker=None, aut_worker=None, resume=None, adopted=False, goal="g")
    rep = sim_report.SimulationReport(session_id="offline-gate", provenance=prov)
    path = sim_report.save_report(rep, base_dir=str(TMP))
    data = _json.loads(Path(path).read_text())
    OUT["executed_git_hash"] = (data.get("provenance") or {}).get("executed_git_hash")
    OUT["raised"] = None
except BaseException as e:  # noqa: BLE001
    OUT["raised"] = type(e).__name__ + ": " + str(e)[:200]
    OUT["spy_blocked"] = chain_has(e, _SpyBlocked)
finally:
    STATE["window"] = None
emit()
"""


def test_offline_sim_report_says_offline(tmp_path: Path) -> None:
    """§1A.7's composed gate, at the report writer: latched, in a git checkout, the shipped provenance-to-report path
    (``capture_start_provenance`` -> ``run_provenance`` -> ``SimulationReport`` -> ``save_report``) writes a
    ``report.json`` whose ``executed_git_hash`` is ``"unknown: offline"``, and spawns nothing (the spy is first).

    Scope, stated: this is the report writer the sim loop calls, not a whole offline sim. A full sim with its LLM
    mocked is not cheap here (the orchestrator's AUT, narrator and in-process-model startup); the item 10 PR owes
    that end-to-end run. Fails today: no ``maxim._offline``."""
    out = _run_child(tmp_path, _REPORT_BODY, cwd=REPO)
    _assert_offline_built(out)
    require((REPO / ".git").exists(), "the gate needs a git checkout, and this tree has no .git")
    _assert_spy_saw_nothing("the report writer", {"spy_events": out["spy"], "spy_blocked": out.get("spy_blocked")})
    assert out["raised"] is None, f"the offline report writer raised: {out['raised']}"
    assert out["executed_git_hash"] == "unknown: offline", f"report.json says {out['executed_git_hash']!r}"


# ── 7c. engage()'s other two steps: the pooled client and the library flags (notes v6.4 §1A.1) ──

_POOLED_BODY = r"""
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
off = require_offline_module()
SEEN = []  # the client port of every request the listener received

class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # keep-alive, so the client pools the connection
    def do_GET(self):
        SEEN.append(self.client_address[1])
        body = b"ok"
        self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers()
        self.wfile.write(body)
    def log_message(self, *a):
        pass

server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)  # inbound bind/listen on loopback: allowed
threading.Thread(target=server.serve_forever, daemon=True).start()
from maxim.utils import http
OUT["has_on_engage"] = callable(getattr(off, "on_engage", None))
callbacks = getattr(off, "engage_callbacks", None)
OUT["close_all_registered"] = callable(callbacks) and any(cb is http.close_all for cb in callbacks())
http.register_endpoint(http.HTTPEndpoint(name="offline-pool-probe", base_url=f"http://127.0.0.1:{server.server_address[1]}"))
http.get("offline-pool-probe", "/")
http.get("offline-pool-probe", "/")
OUT["pre_ports"] = list(SEEN)
off.engage("test")
install_spy()  # after the latch
OUT["post"] = attempt(off, "pooled_request", lambda: http.get("offline-pool-probe", "/"))
OUT["seen"] = list(SEEN)
server.shutdown()
emit()
"""


def test_a_client_pooled_before_engage_cannot_send_after(tmp_path: Path) -> None:
    """A ``utils/http.py`` client that pooled a keep-alive connection to a local listener BEFORE ``engage()`` cannot
    send after it: the next request raises or records ``OfflineRefusal`` and the listener receives nothing more.
    ``send``/``sendall`` raise no audit event, so only ``engage()``'s ``on_engage`` callback (``close_all``) can stop
    that socket. Scope, stated: this covers ``utils.http``'s registry pools only. The pools outside it
    (``openai_backend``'s ``address_checked_client``, and the Anthropic and OpenAI SDKs' own clients) are not closed
    by ``close_all``; the notes cover them by another layer (the router predicate refuses at dispatch, and cloud SDKs
    are never built after the latch), gated by ``test_router_refuses_a_pre_engage_lane_at_dispatch``. Also pins the
    seam: ``maxim._offline.on_engage`` exists, and ``maxim._offline.engage_callbacks()``
    (the read seam chosen here) holds ``utils.http.close_all``. Fails today: no ``maxim._offline``."""
    out = _run_child(tmp_path, _POOLED_BODY, spy="body")
    _assert_offline_built(out)
    require(
        len(out["pre_ports"]) == 2 and out["pre_ports"][0] == out["pre_ports"][1],
        f"the client did not pool a keep-alive connection before engage: {out['pre_ports']}",
    )
    assert out["has_on_engage"], "maxim._offline has no on_engage"
    assert out["close_all_registered"], "utils.http did not register close_all through on_engage"
    _assert_spy_saw_nothing("pooled_request", out["post"])
    assert len(out["seen"]) == 2, f"the listener received a request after engage(): ports {out['seen']}"
    res = out["post"]
    assert res["refusal"] or res["ledger_delta"] > 0, f"the pooled request was not refused as offline: {res}"


_FLAGS_BODY = r"""
import tempfile
STATE["window"] = "__preimport__"
import huggingface_hub, huggingface_hub.constants
ULTRA = importlib.util.find_spec("ultralytics") is not None
if ULTRA:
    # Its import-time is_online() resolves a host: import it with YOLO_OFFLINE for this one import (no egress), then
    # set ONLINE back to True, standing in for a process that imported it online before the latch went up.
    os.environ["YOLO_OFFLINE"] = "true"
    try:
        import ultralytics.utils
        ultralytics.utils.ONLINE = True
    except Exception as e:  # noqa: BLE001
        OUT["ultra_import"] = type(e).__name__ + ": " + str(e)[:120]
    finally:
        os.environ.pop("YOLO_OFFLINE", None)
STATE["window"] = None
off = require_offline_module()
OUT["hf_before"] = huggingface_hub.constants.HF_HUB_OFFLINE
off.engage("test")
OUT["hf_after"] = huggingface_hub.constants.HF_HUB_OFFLINE
OUT["ultra_online_after"] = getattr(sys.modules.get("ultralytics.utils"), "ONLINE", "absent") if ULTRA else "not installed"
OUT["direct"] = attempt(off, "hf_hub_download", lambda: huggingface_hub.hf_hub_download(
    "gpt2", "config.json", cache_dir=tempfile.mkdtemp()))
emit()
"""


def test_engage_sets_already_imported_library_flags(tmp_path: Path) -> None:
    """With ``huggingface_hub`` imported FIRST (and ultralytics, where installed), ``engage()`` sets
    ``huggingface_hub.constants.HF_HUB_OFFLINE = True`` and ``ultralytics.utils.ONLINE = False`` through
    ``sys.modules``, and a direct ``hf_hub_download`` then raises ``OfflineModeIsEnabled`` with no socket event (the
    spy is installed first). Fails today: no ``maxim._offline``."""
    out = _run_child(tmp_path, _FLAGS_BODY)
    _assert_offline_built(out)
    require(out["hf_before"] is False, f"HF_HUB_OFFLINE was already {out['hf_before']!r} before engage()")
    assert out["hf_after"] is True, "engage() did not set huggingface_hub.constants.HF_HUB_OFFLINE"
    require(
        not out.get("ultra_import"), f"ultralytics could not be imported for the flag check: {out.get('ultra_import')}"
    )
    if out["ultra_online_after"] != "not installed":
        assert out["ultra_online_after"] is False, f"ultralytics.utils.ONLINE is {out['ultra_online_after']!r}"
    _assert_spy_saw_nothing("hf_hub_download", out["direct"])
    assert out["direct"].get("hf_offline"), f"hf_hub_download did not raise OfflineModeIsEnabled: {out['direct']}"


_RESET = "_reset_for_tests"


def _reset_references(roots: tuple[Path, ...], base: Path) -> list[str]:
    """Every reference to ``_reset_for_tests`` under ``roots``: an attribute or a name, an import alias (``from ...
    import _reset_for_tests as r``), a star import from ``maxim._offline``, and the bare string (``getattr(m,
    "_reset_for_tests")``). The ``def`` itself is not a reference."""
    hits: list[str] = []
    for root in roots:
        for path in sorted(root.rglob("*.py")):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError as e:
                raise ScenarioBroken(f"{path} does not parse: {e}") from e
            rel = str(path.relative_to(base))
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute):
                    hit = node.attr == _RESET
                elif isinstance(node, ast.Name):
                    hit = node.id == _RESET
                elif isinstance(node, ast.Constant):
                    hit = isinstance(node.value, str) and _RESET in node.value
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    star = isinstance(node, ast.ImportFrom) and (node.module or "").endswith("_offline")
                    hit = any(
                        _RESET in (alias.name, alias.asname)
                        or alias.name.endswith(f".{_RESET}")
                        or (star and alias.name == "*")
                        for alias in node.names
                    )
                else:
                    continue
                if hit:
                    hits.append(f"{rel}:{node.lineno}")
    return hits


def test_no_caller_of_reset_for_tests_outside_tests(tmp_path: Path) -> None:
    """``maxim._offline._reset_for_tests`` disengages the latch, so nothing outside ``tests/`` may reference it
    (notes v6.4: a guard forbids any caller outside ``tests/``). An AST scan of ``src/`` and ``scripts/`` that also
    sees aliased imports, star imports and the getattr string. It holds today, when nothing references it; the
    deletion-style probe below proves the scan is not blind to an aliased caller."""
    require(any(True for _ in SRC.rglob("*.py")), "the scan found no src/ files")
    hits = _reset_references((SRC, REPO / "scripts"), REPO)
    assert not hits, f"_reset_for_tests is referenced outside tests/: {hits}"

    probe = tmp_path / "probe_tree"
    probe.mkdir()
    (probe / "aliased.py").write_text("from maxim._offline import _reset_for_tests as r\n\nr()\n")
    (probe / "by_string.py").write_text('import maxim._offline as m\n\ngetattr(m, "_reset_for_tests")()\n')
    (probe / "star.py").write_text("from maxim._offline import *\n")
    (probe / "clean.py").write_text("import maxim._offline as m\n\nm.is_offline()\n")
    caught = {hit.split(":")[0] for hit in _reset_references((probe,), probe)}
    assert caught == {"aliased.py", "by_string.py", "star.py"}, f"the scan is blind to a caller form: {sorted(caught)}"


# ── 8. config (§1A.6) ─────────────────────────────────────────────────────
# §1A.6's number ("1.5") depends on whether #1431 (console.talk_tools, 1.4) lands first; if it slips past the gate,
# the gate takes 1.4. So these read CONFIG_FORMAT_VERSION from config_loader and never hard-code it.


def _vkey(v: str) -> tuple[int, int]:
    major, minor = (int(p) for p in v.split(".")[:2])
    return major, minor


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=_reason("§1A.6", "LLMConfigSection gains offline"))
def test_llm_offline_is_a_config_key() -> None:
    """``llm.offline`` is a field of the SHAPE-FROZEN ``LLMConfigSection`` (no env alias). Fails today: no field."""
    from maxim.runtime.config_loader import LLMConfigSection

    assert "offline" in {f.name for f in dataclasses.fields(LLMConfigSection)}, "llm.offline is not a config key"


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=_reason("§1A.6", "the format version that owns it"))
def test_config_version_owns_llm_offline() -> None:
    """``CONFIG_FORMAT_VERSION`` was bumped to a version that OWNS ``llm.offline``: its first appearance in the
    append-only ``config_schema_by_version.json`` is a version after 1.3 and at most the current one. Fails today:
    no pinned version lists it."""
    from maxim.runtime.config_loader import CONFIG_FORMAT_VERSION

    pinned = json.loads(SCHEMA_FIXTURE.read_text())
    versions = sorted((k for k in pinned if not k.startswith("_")), key=_vkey)
    owners = [v for v in versions if "llm.offline" in pinned[v]]
    assert owners, f"no pinned config version lists llm.offline (current {CONFIG_FORMAT_VERSION})"
    owner = owners[0]
    assert _vkey("1.3") < _vkey(owner) <= _vkey(CONFIG_FORMAT_VERSION), (owner, CONFIG_FORMAT_VERSION)


# The §1A.6 downgrade-warning gate lives in tests/unit/test_config_downgrade_974.py: CI's IM2 grep allows
# callers of the config downgrade writer only in the config test files.


# ── 9. conftest ───────────────────────────────────────────────────────────


def _imports_maxim(node: ast.stmt) -> bool:
    if isinstance(node, ast.Import):
        return any(a.name == "maxim" or a.name.startswith("maxim.") for a in node.names)
    if isinstance(node, ast.ImportFrom):
        return node.level == 0 and bool(node.module) and (node.module == "maxim" or node.module.startswith("maxim."))
    return False


def _pops_maxim_offline(node: ast.stmt) -> bool:
    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
        return False
    call = node.value
    func = call.func
    is_environ_pop = (
        isinstance(func, ast.Attribute)
        and func.attr == "pop"
        and isinstance(func.value, ast.Attribute)
        and func.value.attr == "environ"
    )
    first = call.args[0] if call.args else None
    return is_environ_pop and isinstance(first, ast.Constant) and first.value == "MAXIM_OFFLINE"


def test_conftest_pops_maxim_offline_before_importing_maxim() -> None:
    """``tests/conftest.py`` runs ``os.environ.pop("MAXIM_OFFLINE", None)`` as a top-level statement BEFORE its first
    ``import maxim`` (a developer shell must not latch the pytest process). Fails today: there is no such pop."""
    body = ast.parse(CONFTEST.read_text()).body
    first_import = next((i for i, n in enumerate(body) if _imports_maxim(n)), None)
    require(first_import is not None, "tests/conftest.py no longer imports maxim at top level")
    pops = [i for i, n in enumerate(body) if _pops_maxim_offline(n)]
    assert pops, "tests/conftest.py never pops MAXIM_OFFLINE at module level"
    assert pops[0] < first_import, (
        f"MAXIM_OFFLINE is popped at statement {pops[0]}, after import maxim at {first_import}"
    )
