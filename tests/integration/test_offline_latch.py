"""#1315 item 10 strict red gates: the offline latch (V8, step 0b), its call-site refusals and its config key.

WHAT THIS FILE IS. Tests only, written before the code (``docs/plans/atonia_gate_approach_notes.md``, DRAFT v6.2:
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
  * ``test_router_refuses_cloud_and_remote_lanes_at_build`` / ``test_router_refuses_a_pre_engage_lane_at_dispatch``
    -- §6 layer 3, the router predicate.
  * ``test_llm_offline_is_a_config_key`` / ``test_config_version_owns_llm_offline`` /
    ``test_downgrade_warns_when_it_drops_llm_offline`` -- §1A.6 / §6 "Config".
  * ``test_conftest_pops_maxim_offline_before_importing_maxim`` -- §1A.1 "Tests".

SEAMS THESE GATES PIN (the notes leave them open; chosen here, change them only with the implementing PR):
  * ``maxim._offline``: ``engage(reason: str) -> None``; ``is_offline() -> bool``; ``OfflineRefusal(RuntimeError)``;
    ``truthy(value: str | None) -> bool``; ``ledger() -> Sequence`` (one entry per refusal, appended BEFORE the
    raise, a fresh sequence per call); ``hook_installed() -> bool`` (whether THIS process ever added the audit hook);
    ``_reset_for_tests()`` (named by ``tests/integration/test_advisor_gate.py``).
  * "No hook without the variable" is checked by the installed flag AND behaviourally (a synthetic
    ``sys.audit("subprocess.Popen", ...)`` and ``sys.audit("socket.getaddrinfo", ...)`` pass), because Python
    offers no API that lists audit hooks.
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
import logging
import subprocess
import sys
import textwrap
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

    def attempt(off, name, fn):
        before = len(off.ledger())
        STATE["window"] = name
        try:
            fn()
            res = {"raised": None, "refusal": False, "spy_blocked": False}
        except BaseException as e:  # noqa: BLE001 -- the gate reports whatever the call did
            res = {
                "raised": type(e).__name__,
                "refusal": chain_has(e, off.OfflineRefusal),
                "spy_blocked": chain_has(e, _SpyBlocked),
                "msg": str(e)[:200],
            }
        finally:
            STATE["window"] = None
        res["ledger_delta"] = len(off.ledger()) - before
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
def _unix():
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); s.connect(os.path.join(tempfile.gettempdir(), "no.sock"))
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
@pytest.mark.xfail(
    strict=True, raises=AssertionError, reason=_reason("V8 / step 0b", "maxim._offline's audit hook refuses it")
)
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


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=_reason("V8 / step 0b", "inbound passes the latch"))
def test_engaged_latch_allows_inbound_listen(engaged_child: dict[str, Any]) -> None:
    """Inbound ``bind``/``listen`` on 127.0.0.1 is NOT refused (§6: "Inbound is not refused"). Fails today on the
    existence check (``maxim._offline`` does not exist)."""
    _assert_offline_built(engaged_child)
    res = engaged_child["attempts"]["inbound_bind_listen"]
    _assert_spy_saw_nothing("inbound_bind_listen", res)
    assert res["raised"] is None, f"inbound bind/listen raised {res['raised']}: {res.get('msg')}"
    assert res["ledger_delta"] == 0, "an inbound bind/listen was booked as a refusal"


@pytest.mark.xfail(
    strict=True, raises=AssertionError, reason=_reason("§6 'Engaging the latch'", "OfflineRefusal(RuntimeError)")
)
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


@pytest.mark.xfail(
    strict=True, raises=AssertionError, reason=_reason("§6 self-engage", "maxim/__init__ engages on MAXIM_OFFLINE")
)
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


@pytest.mark.xfail(
    strict=True, raises=AssertionError, reason=_reason("§6 self-engage", "the hook is added only by engage()")
)
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
@pytest.mark.xfail(strict=True, raises=AssertionError, reason=_reason("§1A.1", "maxim._offline.truthy exists"))
def test_one_truthy_parser(raw: str | None, expected: bool) -> None:
    """``maxim._offline.truthy`` and ``cluster_bias_annotation.annotation_disabled_via_env`` agree on every row.
    Fails today: no ``maxim._offline``."""
    off = _require_offline_in_process()
    from maxim.prompts.cluster_bias_annotation import annotation_disabled_via_env

    assert off.truthy(raw) is expected, f"truthy({raw!r}) != {expected}"
    assert annotation_disabled_via_env(raw) is off.truthy(raw), f"the two parsers disagree on {raw!r}"


@pytest.mark.xfail(
    strict=True, raises=AssertionError, reason=_reason("§1A.1", "annotation_disabled_via_env delegates to truthy")
)
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


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=_reason("§1A.1", "maxim/_offline.py exists, stdlib-only"))
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


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=_reason("§1A.7", "git_out returns None when offline"))
def test_offline_git_out_spawns_nothing(tmp_path: Path) -> None:
    """Latched, ``code_tree.git_out([...])`` returns ``None`` and spawns nothing (its stdlib probe of
    ``sys.modules["maxim._offline"]``). Fails today: no ``maxim._offline``."""
    out = _provenance_child(tmp_path, "git_out")
    res = out["result"]
    _assert_spy_saw_nothing("git_out", {"spy_events": out["spy"], **res})
    assert res["raised"] is None, f"git_out raised {res['raised']} under offline: {res.get('msg')}"
    assert res["value"] is None, f"git_out returned {res['value']!r} under offline"
    assert not out["spy"], f"git_out spawned under offline: {out['spy']}"


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=_reason("§1A.7", "_code_stamp labels 'unknown: offline'"))
def test_offline_code_stamp_says_offline(tmp_path: Path) -> None:
    """Latched, in a git checkout, ``simulation/report.py::_code_stamp`` yields ``executed_git_hash ==
    "unknown: offline"`` and spawns nothing. Fails today: no ``maxim._offline``."""
    out = _provenance_child(tmp_path, "code_stamp")
    res = out["result"]
    _assert_spy_saw_nothing("_code_stamp", {"spy_events": out["spy"], **res})
    assert res["raised"] is None, f"_code_stamp raised {res['raised']} under offline: {res.get('msg')}"
    assert res["value"]["executed_git_hash"] == "unknown: offline", res["value"]
    assert not out["spy"], f"_code_stamp spawned under offline: {out['spy']}"


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=_reason("§1A.7", "get_version_info early-returns"))
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
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
off = require_offline_module()
TMP = Path(tempfile.mkdtemp())
# Pre-import the heavy libraries BEFORE engage (module docstring): an import-time side effect is not a site's.
for mod in ("httpx", "huggingface_hub", "sentence_transformers"):
    try:
        importlib.import_module(mod)
    except ImportError:
        pass
ST_AVAILABLE = "sentence_transformers" in sys.modules
from maxim.utils import http
from maxim.models import download
from maxim.similarity import encoder
from maxim.simulation import container_runner
from maxim.hivemind import substrate_client

def _get_registered():
    http.register_endpoint(http.HTTPEndpoint(name="offline-gate-probe", base_url="http://127.0.0.1:9"))
    http.get("offline-gate-probe", "/")

def _encoder():
    # The model load every LinguisticEncoder goes through (``_ensure_model``); ec/atl are not on the fetch path.
    enc = encoder.LinguisticEncoder(MagicMock(), MagicMock(), encoder.EncoderConfig(model_name="maxim-offline-gate/missing-model"))
    enc._ensure_model()

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
}
unknown = [s for s in CFG["sites"] if s not in DRIVERS]
if unknown:
    OUT["error"] = "no driver for census site(s): " + ", ".join(unknown)
    emit(); sys.exit(0)
off.engage("test")
OUT["sites"] = {}
for site in CFG["sites"]:
    if site == "similarity.encoder[missing model]" and not ST_AVAILABLE:
        OUT["sites"][site] = {"not_applicable": "sentence_transformers is not installed: the encoder has no fetch path"}
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
    out = _run_child(tmp_path, _CENSUS_BODY, cfg_extra={"sites": list(CENSUS_SITES)})
    _assert_offline_built(out)
    require(out.get("error") is None, str(out.get("error")))
    require(not any(k in out["start_env"] for k in _HF_FORCED), f"HF vars present in the child: {out['start_env']}")
    require(sorted(out["sites"]) == sorted(CENSUS_SITES), f"the child drove {sorted(out['sites'])}")
    failures: list[str] = []
    for site, res in out["sites"].items():
        if "not_applicable" in res:
            continue
        events = [e for e in out["spy"] if e["window"] == site]
        if events or res.get("spy_blocked"):
            failures.append(f"{site}: {_LATCH_DID_NOT_REFUSE} (nor the call site): {events[:3] or res.get('msg')}")
            continue
        if not (res["refusal"] or res["ledger_delta"] > 0):
            failures.append(f"{site}: no OfflineRefusal raised or recorded (raised {res['raised']}: {res.get('msg')})")
    assert not failures, "\n".join(failures)


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


@pytest.mark.xfail(
    strict=True, raises=AssertionError, reason=_reason("§1A.6", "config downgrade warns on dropping llm.offline")
)
def test_downgrade_warns_when_it_drops_llm_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """``maxim config downgrade`` of a newer file carrying ``llm.offline: true`` warns that the run now goes ONLINE,
    naming the key. Today the generic "set aside" WARNING names it but never says the run goes online.

    The build that drops the key is one that does not know it (§1A.6's stated fail-open). This build will know it
    once the gate lands, so ``unknown_config_entries`` is wrapped to always report ``llm.offline`` as dropped (a
    no-op today, when it genuinely is unknown): the gate pins the warning, not which build emits it."""
    from maxim.runtime import config_loader
    from maxim.runtime.config_writer import downgrade_config

    real = config_loader.unknown_config_entries

    def dropping(data: dict[str, Any]) -> dict[str, Any]:
        return {**real(data), "llm.offline": True}

    monkeypatch.setattr(config_loader, "unknown_config_entries", dropping)
    major, minor = _vkey(config_loader.CONFIG_FORMAT_VERSION)
    target = tmp_path / "config.json"
    target.write_text(json.dumps({"_format_version": f"{major}.{minor + 1}", "llm": {"offline": True}}))
    with caplog.at_level(logging.WARNING):
        result = downgrade_config(target)
    require("llm.offline" in result.preserved, f"the downgrade did not drop llm.offline: {result.preserved}")
    warned = [
        r.getMessage()
        for r in caplog.records
        if r.levelno >= logging.WARNING and "llm.offline" in r.getMessage() and " online" in r.getMessage().lower()
    ]
    assert warned, "no WARNING says dropping llm.offline: true makes the run go online"


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


@pytest.mark.xfail(
    strict=True, raises=AssertionError, reason=_reason("§1A.1 'Tests'", "conftest pops MAXIM_OFFLINE first")
)
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
