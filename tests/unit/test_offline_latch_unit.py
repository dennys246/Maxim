"""In-process unit tests for the offline latch (#1315 step 0b; notes §1A.1, §1A.7, §6).

The child-process gates in ``tests/integration/test_offline_latch.py`` prove the latch end to end, but coverage
cannot see them: CPython runs audit hooks untraced, and the offline branches of the read sites run only in child
processes there. These call the hook's decision functions directly and drive every offline branch in-process, with
the conftest autouse ``_reset_offline_latch`` fixture disengaging the latch after each test.
"""

from __future__ import annotations

import os
import socket
import sys
import types
from pathlib import Path

import pytest

from maxim import _offline


class _Sock:
    def __init__(self, family: int) -> None:
        self.family = family


# ── the hook's decision, one branch at a time ────────────────────────────


@pytest.mark.parametrize(
    ("event", "args"),
    [
        ("socket.__new__", (None, getattr(socket, "AF_ROUTE", 17), socket.SOCK_RAW, 0)),
        ("socket.__new__", (None, -1, socket.SOCK_STREAM, 0)),  # fd adoption without a family
        ("socket.connect", (_Sock(socket.AF_INET), ("127.0.0.1", 9))),
        ("socket.connect", (_Sock(socket.AF_UNIX), "/tmp/x.sock")),
        ("socket.sendto", (_Sock(socket.AF_INET), ("192.0.2.1", 53))),
        ("socket.sendto", (_Sock(socket.AF_UNIX), "/tmp/dgram.sock")),  # Executor S3: any family
        ("socket.sendmsg", (_Sock(socket.AF_INET6), ("::1", 53, 0, 0))),
        ("socket.getaddrinfo", ("example.com", 80, 0, 0, 0)),
        ("socket.gethostbyname", ("example.com",)),
        ("socket.gethostbyaddr", ("8.8.8.8",)),
        ("socket.getnameinfo", (("127.0.0.1", 80), 0)),
        ("os.system", (b"true",)),
        ("subprocess.Popen", ("/bin/true", ["/bin/true"], None, None)),
        ("os.exec", ("/bin/true", ["/bin/true"], {})),
        ("os.posix_spawn", ("/bin/true", ["/bin/true"], {})),
        ("os.spawn", (0, "/bin/true", ["/bin/true"], None)),
        ("os.fork", ()),  # owner decision (a) 2026-10-26
        ("os.forkpty", ()),
    ],
)
def test_refused_events(event: str, args: tuple) -> None:
    assert _offline._refused_detail(event, args), f"{event} passes the latch"


@pytest.mark.parametrize(
    ("event", "args"),
    [
        ("socket.__new__", (None, socket.AF_INET, socket.SOCK_STREAM, 0)),
        ("socket.__new__", (None, socket.AF_INET6, socket.SOCK_DGRAM, 0)),
        ("socket.__new__", (None, socket.AF_UNIX, socket.SOCK_STREAM, 0)),
        ("socket.sendto", (_Sock(socket.AF_INET), None)),  # connected socket: its connect was refused
        ("socket.sendmsg", (_Sock(socket.AF_UNIX), None)),
        ("socket.bind", (_Sock(socket.AF_INET), ("127.0.0.1", 0))),  # inbound passes
        ("socket.listen", (_Sock(socket.AF_INET), 1)),
        ("os.putenv", (b"K", b"V")),
        ("open", ("/tmp/x", "r", 0)),
    ],
)
def test_allowed_events(event: str, args: tuple) -> None:
    assert _offline._refused_detail(event, args) is None


def test_hook_is_inert_while_disengaged() -> None:
    assert not _offline.is_offline()
    _offline._hook("socket.getaddrinfo", ("example.com", 80, 0, 0, 0))  # no raise
    assert _offline.ledger() == ()


def test_hook_books_then_raises_while_engaged() -> None:
    _offline.engage("unit: hook")
    _offline._hook("open", ("/tmp/x", "r", 0))  # not a socket/os/subprocess event
    _offline._hook("socket.bind", (_Sock(socket.AF_INET), ("127.0.0.1", 0)))  # allowed
    assert _offline.ledger() == ()
    with pytest.raises(_offline.OfflineRefusal, match="os.fork"):
        _offline._hook("os.fork", ())
    (entry,) = _offline.ledger()
    assert (entry.event, entry.reason) == ("os.fork", "unit: hook")


def test_refusal_is_a_runtime_error_and_never_an_oserror() -> None:
    assert issubclass(_offline.OfflineRefusal, RuntimeError)
    assert not issubclass(_offline.OfflineRefusal, OSError)


def test_refuse_books_a_call_site_refusal() -> None:
    _offline.engage("unit: refuse")
    with pytest.raises(_offline.OfflineRefusal, match="site-x refused: detail"):
        _offline.refuse("site-x", "detail")
    assert [e.event for e in _offline.ledger()] == ["site-x"]


# ── engage / is_offline / callbacks / reset ───────────────────────────────


def test_is_offline_engages_on_a_truthy_variable_at_any_call(monkeypatch) -> None:
    monkeypatch.setenv("MAXIM_OFFLINE", "y")  # not canonical
    assert _offline.is_offline() is False
    monkeypatch.setenv("MAXIM_OFFLINE", " On ")
    assert _offline.is_offline() is True
    assert _offline.hook_installed() is True


def test_engage_is_one_way_and_idempotent() -> None:
    _offline.engage("first")
    _offline.engage("second")
    with pytest.raises(_offline.OfflineRefusal, match=r"offline \(first\)"):
        _offline.refuse("site")


def test_on_engage_runs_now_when_already_engaged_and_once_per_callable() -> None:
    calls: list[str] = []

    def cb() -> None:
        calls.append("cb")

    _offline.engage("unit: callbacks")
    try:
        _offline.on_engage(cb)
        _offline.on_engage(cb)
        assert calls == ["cb", "cb"]  # run at once on each registration, listed once
        assert _offline.engage_callbacks().count(cb) == 1
    finally:
        _offline._callbacks.remove(cb)


def test_a_failing_callback_does_not_stop_the_others(monkeypatch) -> None:
    ran: list[str] = []

    def boom() -> None:
        raise ValueError("callback failed")

    def after() -> None:
        ran.append("after")

    monkeypatch.setattr(_offline, "_callbacks", [boom, after])
    with pytest.raises(ValueError, match="callback failed"):
        _offline.engage("unit: failing callback")
    assert ran == ["after"]
    assert _offline.is_offline() is True  # the latch is up anyway


def test_reset_restores_the_env_keys_and_library_flags(monkeypatch) -> None:
    """Executor N1 / Wire F5: without the restore every later test would see the engaged values."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.delenv("YOLO_OFFLINE", raising=False)
    hf = types.SimpleNamespace(HF_HUB_OFFLINE=False)
    ultra = types.SimpleNamespace(ONLINE=True)
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", hf)
    monkeypatch.setitem(sys.modules, "ultralytics.utils", ultra)
    _offline.engage("unit: reset")
    assert os.environ["HF_HUB_OFFLINE"] == "1" and os.environ["YOLO_OFFLINE"] == "true"
    assert hf.HF_HUB_OFFLINE is True and ultra.ONLINE is False
    _offline._reset_for_tests()
    assert os.environ["HF_HUB_OFFLINE"] == "0"
    assert "YOLO_OFFLINE" not in os.environ
    assert hf.HF_HUB_OFFLINE is False and ultra.ONLINE is True
    assert _offline.is_offline() is False and _offline.ledger() == ()


# ── the read sites' offline branches ──────────────────────────────────────


@pytest.mark.parametrize(("raw", "engaged"), [(None, False), ("0", False), ("y", False), ("yes", True)])
def test_import_maxim_self_engage(monkeypatch, raw: str | None, engaged: bool) -> None:
    import maxim

    if raw is None:
        monkeypatch.delenv("MAXIM_OFFLINE", raising=False)
    else:
        monkeypatch.setenv("MAXIM_OFFLINE", raw)
    maxim._self_engage_offline()
    assert _offline._engaged is engaged


def test_get_version_info_offline_spawns_nothing() -> None:
    import maxim

    _offline.engage("unit: version")
    assert maxim.get_version_info() == {"version": maxim.__version__}
    assert _offline.ledger() == ()  # an early return, not a refused (and swallowed) git spawn


def test_git_out_offline_returns_none_without_spawning() -> None:
    from maxim.utils import code_tree

    _offline.engage("unit: git_out")
    assert code_tree.git_out(Path(__file__).resolve().parents[2], "rev-parse", "HEAD") is None
    assert _offline.ledger() == ()


def test_code_stamp_offline_says_offline() -> None:
    import maxim
    from maxim.simulation import report

    _offline.engage("unit: code stamp")
    stamp = report._code_stamp(Path(maxim.__file__).resolve())
    assert stamp["executed_git_hash"] == "unknown: offline"
    assert stamp["code_tree_sha256"] == "unknown: offline"
    assert stamp["working_tree_dirty_src_scripts"] is True
    assert _offline.ledger() == ()


def test_internet_is_lowered_and_named_offline(tmp_path, monkeypatch, caplog) -> None:
    """Wire F3: when the latch is the cause, the source and every message say offline."""
    from maxim.utils import internet_access as ia

    monkeypatch.delenv(ia.SESSION_LOWERED_ENV, raising=False)
    _offline.engage("unit: internet")
    assert ia.session_internet_lowered() is True
    ia.lower_internet_for_session()
    assert ia.SESSION_LOWERED_ENV not in os.environ  # read side only: nothing written
    policy = ia.load_internet_policy(tmp_path / "policy.json", access_path=tmp_path / "access.json")
    assert (policy.enabled, policy.source) == (False, "offline")
    assert "offline" in policy.disabled_reason() and ia.SESSION_LOWERED_ENV not in policy.disabled_reason()
    with caplog.at_level("WARNING", logger=ia.logger.name):
        ia.live_internet_policy_getter(True, policy_path=tmp_path / "policy.json", access_path=tmp_path / "a.json")
    assert any("offline" in r.getMessage() for r in caplog.records)
    assert not any(ia.SESSION_LOWERED_ENV in r.getMessage() for r in caplog.records)


def test_the_toggle_tool_names_offline_as_the_remedy() -> None:
    from maxim.tools import internet_search
    from maxim.utils import internet_access as ia

    _offline.engage("unit: toggle")
    tool_cls = next(
        c
        for c in vars(internet_search).values()
        if isinstance(c, type) and getattr(c, "name", None) == "internet_access_toggle"
    )
    result = tool_cls().execute(enabled=True, reason="unit")
    assert not result.success
    assert "MAXIM_OFFLINE" in result.error and ia.SESSION_LOWERED_ENV not in result.error


# ── the former replica parsers agree with the one canonical parser (Architecture S1, Wire F1/F2) ──


@pytest.mark.parametrize("raw", ["1", "true", " YES ", "on", "t", "y", "0", "garbage"])
def test_former_replica_parsers_agree_with_truthy(monkeypatch, raw: str) -> None:
    from maxim.decisions.nac import _read_nac_reward_bias_disabled_env
    from maxim.integration.memory_hub import body_state_prompt_enabled
    from maxim.models.download import _auto_download_enabled
    from maxim.utils.plotting import _is_truthy

    expected = _offline.truthy(raw)
    monkeypatch.setenv("MAXIM_ENABLE_BODY_STATE_PROMPT", raw)
    monkeypatch.setenv("MAXIM_NAC_REWARD_BIAS_DISABLED", raw)
    monkeypatch.setenv("MAXIM_AUTO_DOWNLOAD_MODELS", raw)
    assert body_state_prompt_enabled() is expected
    assert _read_nac_reward_bias_disabled_env() is expected
    assert _auto_download_enabled() is expected
    assert _is_truthy(raw) is expected
