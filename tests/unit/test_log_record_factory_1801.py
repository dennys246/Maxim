"""#1801 family 1: the chained, URL-redacting ``logging`` record factory ``configure_logging`` installs
(docs/plans/log_url_redaction_1801_approach.md v28, §Shape and §Guards).

A credential URL reaches a handler through ``msg``, ``args``, ``exc_info``, ``stack_info`` and ``extra=data``; each
test plants one and asserts only the origin survives. ``SECRET`` is the planted credential.
"""

from __future__ import annotations

import copy
import io
import json
import logging
import logging.handlers
import os
import pickle
import subprocess
import sys
import types
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import pytest

from maxim.utils import logging as mlog
from maxim.utils.logging import _RedactingLogRecord, configure_logging, ensure_redacting_record_factory

URL = "https://user:SECRET@host.example/private/path?key=SECRET"
# A token-as-username URL: a cut before reduction leaves ``https://tok_SECRE`` as a "host" (lowercased by
# ``url_origin``), so the checks below look for ``tok_s`` case-insensitively.
TOKEN_URL = "https://tok_SECRETTOKENABCDEFGHIJKLMNOPQRSTUVWXYZ@github.com/o/r?key=SECRET"


def _token_clean(text: str) -> bool:
    return "tok_s" not in text.lower() and "secret" not in text.lower()


@pytest.fixture
def installed():
    """The factory installed as ``configure_logging`` installs it (the autouse conftest fixture restores it)."""
    logging.setLogRecordFactory(logging.LogRecord)
    mlog._chained_factory = logging.LogRecord
    mlog._factory_installed_once = False
    assert ensure_redacting_record_factory() is True
    return logging.getLogRecordFactory()


def _render(fmt: str, *args: object) -> str:
    rec = logging.getLogRecordFactory()("t", logging.WARNING, __file__, 1, fmt, args, None)
    return rec.getMessage()


def _stream_logger(name: str) -> tuple[logging.Logger, io.StringIO]:
    buf = io.StringIO()
    lg = logging.getLogger(name)
    lg.handlers[:] = []
    h = logging.StreamHandler(buf)
    h.setFormatter(logging.Formatter("%(message)s"))
    lg.addHandler(h)
    lg.propagate = False
    lg.setLevel(logging.DEBUG)
    return lg, buf


# ─── install and the handlers configure_logging builds ────────────────────


def test_configure_logging_installs_the_factory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    jsonl = tmp_path / "run.jsonl"
    plain = tmp_path / "plain.log"
    monkeypatch.setenv("MAXIM_LOG_FILE", str(jsonl))
    logging.setLogRecordFactory(logging.LogRecord)
    configure_logging(1, force=True, log_file=str(plain))
    # a basicConfig(force=True) rebuild keeps the factory (it is not a handler)
    logging.basicConfig(force=True, handlers=list(logging.getLogger().handlers))
    root = logging.getLogger()
    buf = io.StringIO()
    stream = logging.StreamHandler(buf)
    root.addHandler(stream)
    try:
        log = logging.getLogger("maxim.test1801")
        try:
            raise RuntimeError(f"GET {URL} failed")
        except RuntimeError as e:
            log.warning(
                "failed %s (%.40s)", e, URL, exc_info=True, extra={"event": "e1801", "data": {"u": URL, "err": e}}
            )
        log.warning(f"msg itself {URL}")
        for h in root.handlers:
            h.flush()
    finally:
        root.removeHandler(stream)
        for h in list(root.handlers):
            if isinstance(h, logging.FileHandler):
                root.removeHandler(h)
                h.close()
    out = buf.getvalue() + plain.read_text() + jsonl.read_text()
    assert "SECRET" not in out
    assert "https://host.example" in buf.getvalue()
    assert any(json.loads(line).get("e") == "e1801" for line in jsonl.read_text().splitlines()), jsonl.read_text()


def test_record_args_untouched(installed) -> None:
    """R3: ``msg``/``args`` stay raw (``test_followup_parse_log_1751`` reads ``record.args``); only rendering
    reduces."""
    rec = installed("t", logging.INFO, __file__, 1, "x %s", (URL,), None)
    assert type(rec) is _RedactingLogRecord
    assert rec.args == (URL,)
    assert "SECRET" not in rec.getMessage()


def test_log_buffer_and_display_handler_reduce(installed) -> None:
    from maxim.runtime import leader_proxy
    from maxim.simulation.sim_logger import _DisplayLoggingHandler

    leader_proxy._log_buffer = None
    buf = leader_proxy._ensure_log_buffer()
    shown: list[str] = []

    class _Display:
        def log(self, kind: str, msg: str) -> None:
            shown.append(msg)

    disp = _DisplayLoggingHandler(_Display())
    lg = logging.getLogger("maxim.test1801.buffer")
    lg.addHandler(disp)
    try:
        lg.error("boom %s", URL)
        entries = buf.get_since(since_seq=-1)
    finally:
        lg.removeHandler(disp)
        logging.getLogger("maxim").removeHandler(buf)
        leader_proxy._log_buffer = None
    assert entries and all("SECRET" not in e["message"] for e in entries)
    assert shown and all("SECRET" not in m for m in shown)


def test_log_buffer_emit_reduces_without_the_factory() -> None:
    """R4 belt-and-braces: ``_LogBuffer.emit`` reduces its formatted string even when a handler re-renders text."""
    from maxim.runtime.leader_proxy import _LogBuffer

    logging.setLogRecordFactory(logging.LogRecord)
    b = _LogBuffer()
    b.setFormatter(logging.Formatter("%(message)s"))
    b.emit(logging.LogRecord("t", logging.ERROR, __file__, 1, "x %s", (URL,), None))
    assert "SECRET" not in b.get_since(since_seq=-1)[0]["message"]


# ─── P1 / Q1 / T1: arguments, cut lengths and types ──────────────────────


class _Custom:
    def __str__(self) -> str:
        return TOKEN_URL

    __repr__ = __str__


class _NInt(int):
    def __str__(self) -> str:
        return TOKEN_URL


class _S1(str):
    def __str__(self) -> str:
        return TOKEN_URL


def _cut_values() -> list[object]:
    import httpx

    err = RuntimeError(TOKEN_URL)
    return [
        TOKEN_URL,
        err,
        [TOKEN_URL],
        err.args,
        TOKEN_URL.encode(),
        httpx.URL(TOKEN_URL),
        _Custom(),
        _NInt(3),
        _S1("x"),
        Path("/a") / TOKEN_URL,
    ]


@pytest.mark.parametrize("cut", [1, 10, 15, 17, 30, 50, 80, 200])
def test_every_argument_kind_at_every_cut(installed, cut: int) -> None:
    for v in _cut_values():
        for conv in ("s", "r", "a"):
            out = _render(f"x %.{cut}{conv}", v)
            assert _token_clean(out), (v, conv, out)


@pytest.mark.parametrize("cut", [1, 10, 17, 30, 80])
def test_mapping_value_cut(installed, cut: int) -> None:
    out = _render(f"x %(k).{cut}s %(n)d", {"k": TOKEN_URL, "n": 4})
    assert _token_clean(out)
    assert out.endswith(" 4")


def test_precision_cut_through_every_call_form(installed) -> None:
    """P1: whatever the logger's name, a message held in a variable, ``logger.log(level, ...)``."""
    lg, buf = _stream_logger("anything.not.maxim")
    err = RuntimeError(TOKEN_URL)
    fmt = "failed: %.50s"
    lg.warning("failed: %.50s", err)
    lg.log(30, "%.50r", err)
    lg.warning(fmt, err)

    class _Svc:
        log = lg

    _Svc().log.warning("svc: %.20s", err)
    lg.warning("d=%d f=%5.2f x=%x c=%c none=%s", 3, 2.5, 255, "z", None)
    text = buf.getvalue()
    assert _token_clean(text)
    assert "d=3 f= 2.50 x=ff c=z none=None" in text


def test_numbers_and_one_char_strings_pass_through(installed) -> None:
    assert _render("%d %5.2f %x %c %c %s", 3, 2.5, 255, "z", 65, None) == "3  2.50 ff z A None"
    assert _render("%s %s %s", Decimal("1.5"), Fraction(1, 3), complex(1, 2)) == "1.5 1/3 (1+2j)"


def test_proxied_str_under_number_directive_raises_type_error(installed) -> None:
    """U4 / T3: ``%d`` / ``%f`` of a str raise ``TypeError`` as unproxied."""
    with pytest.raises(TypeError):
        _render("%d", "abc")
    with pytest.raises(TypeError):
        _render("%f", "abc")


def test_index_only_object_renders_d_and_f(installed) -> None:
    """V5 / W3: an object with only ``__index__`` renders ``%d`` and ``%f`` (CPython's own fallbacks)."""

    class _Idx:
        def __index__(self) -> int:
            return 7

    assert _render("%d %f %x", _Idx(), _Idx(), _Idx()) == "7 7.000000 7"


def test_format_spec_on_proxy() -> None:
    p = mlog._ReducedArg(URL)
    assert "SECRET" not in format(p, ".50")
    assert format(mlog._ReducedArg(2.5), ".1f") == "2.5"


# ─── numpy (U1, T3, V4, X5, Y5, Z5, AA5, AB4) ────────────────────────────


def test_numpy_text_and_arrays_reduced_numbers_unchanged(installed) -> None:
    np = pytest.importorskip("numpy")
    mlog._numpy_numeric_cache = None
    for v in (np.str_(TOKEN_URL), np.bytes_(TOKEN_URL.encode()), np.array([TOKEN_URL])):
        for cut in (5, 15, 17, 40, 100):
            assert _token_clean(_render(f"%.{cut}s", v))
    assert _render("%d %5.2f %s", np.int64(3), np.float64(2.5), np.float32(1.5)) == "3  2.50 1.5"
    zero_d = np.array(2.5)
    assert _render("%f %d", zero_d, zero_d) == "2.500000 2"
    assert _render("%x", np.array(255)) == "ff"


def test_numpy_subclass_and_fake_module_are_proxied(installed) -> None:
    np = pytest.importorskip("numpy")
    mlog._numpy_numeric_cache = None

    class _F64(np.float64):
        def __str__(self) -> str:
            return TOKEN_URL

    fake = type("float64", (), {"__module__": "numpy", "__str__": lambda self: TOKEN_URL})
    assert _token_clean(_render("%.30s", _F64(1.0)))
    assert _token_clean(_render("%.30s", fake()))


def test_get_message_never_imports_numpy(tmp_path: Path) -> None:
    """X5: the factory builds the numpy set from ``sys.modules`` only."""
    code = (
        "import sys, logging\n"
        "from maxim.utils.logging import ensure_redacting_record_factory\n"
        "ensure_redacting_record_factory()\n"
        "before = set(sys.modules)\n"
        "r = logging.getLogRecordFactory()('t', 30, 'f', 1, '%s %s', (object(), 'https://u:S@h/p'), None)\n"
        "print(r.getMessage())\n"
        "print('numpy' in sys.modules, sorted(set(sys.modules) - before))\n"
    )
    env = {**os.environ, "HOME": str(tmp_path), "MAXIM_OFFLINE": "1"}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, env=env, check=True)
    lines = out.stdout.strip().splitlines()
    assert "S@" not in lines[-2]
    assert lines[-1] == "False []", out.stdout + out.stderr


def _fake_numpy(monkeypatch: pytest.MonkeyPatch, *, initializing: bool, raises: type[Exception] | None) -> None:
    fake = types.ModuleType("numpy")
    fake.__spec__ = types.SimpleNamespace(_initializing=initializing)  # type: ignore[assignment]
    if raises is not None:

        class _Dict:
            def values(self):  # noqa: ANN202
                raise raises("numpy build failure")

        fake.sctypeDict = _Dict()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "numpy", fake)
    monkeypatch.setattr(mlog, "_numpy_numeric_cache", None)
    monkeypatch.setattr(mlog, "_numpy_failure_logged", False)


def test_part_way_imported_numpy_neither_loses_nor_caches(installed, monkeypatch: pytest.MonkeyPatch) -> None:
    """Y5 / AB4: a part-way imported numpy renders the record, caches nothing and does not spend the one-shot log."""
    _fake_numpy(monkeypatch, initializing=True, raises=None)
    assert "SECRET" not in _render("%s %d %s", URL, 3, object())  # the object() consults the numpy set
    assert mlog._numpy_numeric_cache is None
    assert mlog._numpy_failure_logged is False


@pytest.mark.parametrize("exc", [AttributeError, TypeError])
def test_failing_numpy_build_logs_once_and_caches_empty(
    installed, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, exc: type[Exception]
) -> None:
    """Z5 / AA5: a failing build neither loses the record nor rebuilds per record; it logs ONCE (a stream of records
    makes no nested build, no ``RecursionError``)."""
    from maxim.utils.logging import _reset_swallow_seen_for_test

    _reset_swallow_seen_for_test()
    _fake_numpy(monkeypatch, initializing=False, raises=exc)
    lg = logging.getLogger("maxim.test1801.np")
    with caplog.at_level(logging.DEBUG, logger="maxim"):
        for _ in range(50):
            lg.warning("v=%s n=%d o=%s", URL, 2, object())
    assert mlog._numpy_numeric_cache == frozenset()
    assert mlog._numpy_failure_logged is True
    swallows = [r for r in caplog.records if getattr(r, "event", None) == "swallowed_exception"]
    assert len(swallows) == 1
    assert all("SECRET" not in r.getMessage() for r in caplog.records)


# ─── exc_text / stack_info (H2, J1, G3) and serialization (S3, F1) ───────


def _exc_record(factory) -> logging.LogRecord:  # noqa: ANN001
    try:
        raise ConnectionError(f"connect to {URL} refused")
    except ConnectionError:
        return factory("t", logging.ERROR, __file__, 1, "x %s", (URL,), sys.exc_info())


class _PlainChain:
    """A chaining third-party factory that returns a plain ``LogRecord``."""

    def __init__(self, inner) -> None:  # noqa: ANN001
        self.inner = inner

    def __call__(self, *a, **k) -> logging.LogRecord:  # noqa: ANN002, ANN003
        rec = self.inner(*a, **k)
        rec.third_party = True
        return rec


@pytest.mark.parametrize("behind_third_party", [False, True])
@pytest.mark.parametrize("with_exc", [False, True])
def test_records_serialize_clean(installed, behind_third_party: bool, with_exc: bool) -> None:
    factory = _PlainChain(installed) if behind_third_party else installed
    if behind_third_party:
        logging.setLogRecordFactory(factory)
        assert ensure_redacting_record_factory() is False  # the chain still reduces: no double wrap
    rec = _exc_record(factory) if with_exc else factory("t", logging.ERROR, __file__, 1, "x %s", (URL,), None)
    # pickle round trip renders reduced (a traceback never pickles: a receiver formats then drops exc_info)
    to_pickle = copy.copy(rec)
    if with_exc:
        logging.Formatter("%(message)s").format(to_pickle)
        to_pickle.exc_info = None
    blob_raw = pickle.dumps(to_pickle)
    back = pickle.loads(blob_raw)
    assert "SECRET" not in back.getMessage()
    if with_exc:
        assert "SECRET" not in (back.__dict__.get("exc_text") or "")
    # SocketHandler.makePickle's blob holds no credential (it formats exc_text first)
    sh = logging.handlers.SocketHandler("localhost", 0)
    sh.setFormatter(logging.Formatter("%(message)s"))
    blob = sh.makePickle(rec)
    assert b"SECRET" not in blob
    # QueueHandler.prepare renders reduced
    qh = logging.handlers.QueueHandler(None)  # type: ignore[arg-type]
    prepared = qh.prepare(copy.copy(rec))
    assert "SECRET" not in prepared.getMessage() and "SECRET" not in str(prepared.msg)
    if with_exc:
        formatted = logging.Formatter("%(message)s").format(rec)
        assert "ConnectionError" in formatted and "SECRET" not in formatted
        assert "SECRET" not in (rec.__dict__.get("exc_text") or "")


def _log_with_stack(lg: logging.Logger) -> None:
    lg.warning("stack", stack_info=True)  # https://user:SECRET@host.example/in-source-line


def test_stack_info_is_reduced_everywhere(installed) -> None:
    """J1: a ``stack_info=True`` record whose stack text carries a credential is clean through ``makePickle``,
    ``vars()``, ``pickle`` and ``QueueHandler.prepare``."""
    lg = logging.getLogger("maxim.test1801.stack")
    captured: list[logging.LogRecord] = []

    class _Grab(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record)

    h = _Grab()
    lg.addHandler(h)
    try:
        _log_with_stack(lg)
    finally:
        lg.removeHandler(h)
    rec = captured[0]
    assert "https://host.example" in rec.stack_info and "SECRET" not in rec.stack_info  # the source line, reduced
    assert "SECRET" not in str(vars(rec).get("stack_info"))
    assert b"SECRET" not in pickle.dumps(rec)
    assert b"SECRET" not in logging.handlers.SocketHandler("localhost", 0).makePickle(rec)
    prepared = logging.handlers.QueueHandler(None).prepare(rec)  # type: ignore[arg-type]
    assert "SECRET" not in str(prepared.msg)


def test_make_log_record_exc_text_formats_reduced(installed) -> None:
    """G3: ``makeLogRecord``'s dict update bypasses ``__set__``; the read is reduced."""
    rec = logging.makeLogRecord({"msg": "m %s", "args": (URL,), "exc_text": f"Traceback\nError: {URL}"})
    out = logging.Formatter("%(message)s").format(rec)
    assert "SECRET" not in out and "Error: https://host.example" in out


# ─── subclass records (H3, G4, L3, K5) ───────────────────────────────────


class _SlotRecord(logging.LogRecord):
    __slots__ = ("extra_slot",)

    def getMessage(self) -> str:  # noqa: N802
        return "OWN:" + super().getMessage()


def _subclass_factory(*a, **k) -> logging.LogRecord:  # noqa: ANN002, ANN003
    return _SlotRecord(*a, **k)


def test_subclass_record_rendered_eagerly(installed) -> None:
    mlog._chained_factory = _subclass_factory
    rec = logging.getLogRecordFactory()("t", logging.ERROR, __file__, 1, "x %.20s", (URL,), None)
    assert type(rec) is _SlotRecord
    assert rec.args == () and "SECRET" not in rec.msg and rec.msg.startswith("OWN:")


def test_subclass_bad_format_does_not_raise_and_keeps_own_get_message(installed) -> None:
    """H3 / G4: a bad format through a subclass record neither raises out of the ``logger`` call nor loses the
    subclass's ``getMessage``; the traceback fields are still reduced (K5)."""
    mlog._chained_factory = _subclass_factory
    lg = logging.getLogger("maxim.test1801.sub")
    try:
        raise RuntimeError(URL)
    except RuntimeError:
        rec = logging.getLogRecordFactory()("t", logging.ERROR, __file__, 1, "x %d", ("notint",), sys.exc_info())
    assert type(rec) is _SlotRecord and rec.args == ("notint",)
    assert rec.exc_text and "SECRET" not in rec.exc_text
    with pytest.raises(TypeError):
        rec.getMessage()  # at emit, handleError reports it (the stated terminal-only gap)
    assert lg.makeRecord("t", logging.WARNING, __file__, 1, "x %d", ("notint",), None) is not None  # the logger call


def test_subclass_record_without_exception(installed) -> None:
    """L3: a subclass record logged with no exception does not raise."""
    mlog._chained_factory = _subclass_factory
    rec = logging.getLogRecordFactory()("t", logging.INFO, __file__, 1, "plain", (), None)
    assert rec.msg == "OWN:plain"


def test_subclass_arg_whose_str_raises_does_not_escape_the_logger_call(installed) -> None:
    """Code review SF2: ANY exception from rendering a subclass record (here a ``RuntimeError`` from an argument's
    ``__str__``) stays inside the factory, reported in the Stage-1 form, and the record is left untouched."""

    class _Boom:
        def __str__(self) -> str:
            raise RuntimeError("str failed")

    mlog._chained_factory = _subclass_factory
    boom = _Boom()
    rec = logging.getLogRecordFactory()("t", logging.ERROR, __file__, 1, "x %s", (boom,), None)
    assert type(rec) is _SlotRecord and rec.args == (boom,)


# ─── the behavioural chain check (R4, N2) ────────────────────────────────


def test_non_chaining_factory_is_caught_with_one_warning(installed, caplog: pytest.LogCaptureFixture) -> None:
    logging.setLogRecordFactory(lambda *a, **k: logging.LogRecord(*a, **k))
    with caplog.at_level(logging.WARNING, logger="maxim"):
        assert ensure_redacting_record_factory() is True
        assert ensure_redacting_record_factory() is False
    warnings = [r for r in caplog.records if "does not chain" in r.getMessage()]
    assert len(warnings) == 1
    assert "SECRET" not in _render("%s", URL)


def test_racing_installs_keep_the_host_factory_and_warn_nothing(caplog: pytest.LogCaptureFixture) -> None:
    """Code review SF1: two installs racing on different threads. Without the install lock both see a chain that
    does not reduce, the second chains to the first's factory (dropping the host factory) and logs a false WARNING.
    The barrier holds the first probe until the second thread arrives, or times out when the lock serializes them."""
    import threading

    def _host(*a, **k) -> logging.LogRecord:  # noqa: ANN002, ANN003
        rec = logging.LogRecord(*a, **k)
        rec.tag = "host"
        return rec

    logging.setLogRecordFactory(_host)
    mlog._chained_factory = logging.LogRecord
    mlog._factory_installed_once = False
    real_probe = mlog._current_factory_reduces
    barrier = threading.Barrier(2)

    def _slow_probe() -> bool:
        result = real_probe()
        try:
            barrier.wait(timeout=0.5)
        except threading.BrokenBarrierError:
            pass  # serialized by the lock: the other thread is waiting to enter
        return result

    mlog._current_factory_reduces = _slow_probe  # type: ignore[assignment]
    try:
        with caplog.at_level(logging.WARNING, logger="maxim"):
            threads = [threading.Thread(target=ensure_redacting_record_factory) for _ in range(2)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)
    finally:
        mlog._current_factory_reduces = real_probe  # type: ignore[assignment]
    rec = logging.getLogRecordFactory()("t", logging.INFO, __file__, 1, "m %s", (URL,), None)
    assert getattr(rec, "tag", None) == "host"
    assert "SECRET" not in rec.getMessage()
    assert not [r for r in caplog.records if "does not chain" in r.getMessage()]


def test_configure_logging_rechecks_the_chain(installed) -> None:
    logging.setLogRecordFactory(logging.LogRecord)
    configure_logging(0)
    assert "SECRET" not in _render("%s", URL)


def test_reentrant_factory_cannot_recurse(installed) -> None:
    """A factory over ours that calls back into ours and returns a fresh plain record fails the check; the
    re-install chains to it and the re-entry guard stops the recursion."""
    ours = logging.getLogRecordFactory()

    def _replacing(*a, **k) -> logging.LogRecord:  # noqa: ANN002, ANN003
        ours(*a, **k)
        return logging.LogRecord(*a, **k)

    logging.setLogRecordFactory(_replacing)
    assert ensure_redacting_record_factory() is True
    assert "SECRET" not in _render("%s", URL)


# ─── two processes (D4) ──────────────────────────────────────────────────


def test_cli_child_logs_reduced(tmp_path: Path) -> None:
    """The red-team's split: a child entering through ``cli.main`` logs a credential URL; its stderr and its
    ``MAXIM_LOG_FILE`` JSONL hold only the origin."""
    jsonl = tmp_path / "child.jsonl"
    code = (
        "import logging\n"
        "from maxim.cli import main\n"
        "main(['config', 'path'])\n"
        "log = logging.getLogger('maxim.child1801')\n"
        f"log.warning('child %s', {URL!r}, extra={{'event': 'child1801', 'data': {{'u': {URL!r}}}}})\n"
        "try:\n"
        f"    raise OSError({URL!r})\n"
        "except OSError:\n"
        "    log.exception('child exc')\n"
    )
    env = {**os.environ, "HOME": str(tmp_path), "MAXIM_OFFLINE": "1", "MAXIM_LOG_FILE": str(jsonl)}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, env=env)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "child https://host.example" in out.stderr
    assert "SECRET" not in out.stderr + out.stdout
    assert "SECRET" not in jsonl.read_text()


# ─── R6: the autouse conftest fixture restores the factory between tests ─


def test_zz_factory_left_installed_by_one_test() -> None:
    logging.setLogRecordFactory(lambda *a, **k: logging.LogRecord(*a, **k))


def test_zz_factory_restored_for_the_next_test() -> None:
    assert logging.getLogRecordFactory() is logging.LogRecord
