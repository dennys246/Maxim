"""#997: a sim record keeps an endpoint's scheme, host and port, never its credentials, path or query, and no
record producer stamps the process's raw argv.

``report.json`` is committed with experiment evidence. ``language_endpoint`` / ``aut_endpoint`` and the B0 serving
provenance (``llm_calls[*].entries[*].endpoint``, the ``leader_disk_heads`` keys) used to hold the router's base URL
verbatim, so a tunnel URL with ``user:pass@`` or a token in its path or query would be committed with the record.
The port is kept: ``scripts/o19_verdict.py`` C4 tells two local servers apart by it, and it carries no secret.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
SECRET_URL = "https://alice:s3cr3t@Leader.Example.com:8443/tok-abc123/v1?key=hunter2#frag"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (SECRET_URL, "https://leader.example.com:8443"),
        ("http://127.0.0.1:8100/v1", "http://127.0.0.1:8100"),
        ("https://api.anthropic.com/v1", "https://api.anthropic.com"),
        ("http://[::1]:8100/v1", "http://[::1]:8100"),
        ("https://u:p@ss@host.example/v1", "https://host.example"),  # '@' inside the password
        ("  https://host.example/v1  ", "https://host.example"),
        ("http://host.example:99999/v1", ""),  # an invalid port: urlparse's .port raises
        ("http://host.example:abc/v1", ""),
        ("http://tok%3Asecret%40host.example/v1", ""),  # percent-encoded userinfo left in the host
        ("HTTP://Host.Example", "http://host.example"),
        ("//host.example:1/x", ""),
        ("http://[::1", ""),  # an unbalanced bracket: urlsplit itself raises
        ("unix:///tmp/llama.sock", ""),
        ("host.example:8100/v1", ""),  # no scheme
        ("", ""),
        (None, ""),
    ],
)
def test_url_origin_keeps_only_scheme_host_and_port(raw, expected) -> None:
    from maxim.utils.http import url_origin

    assert url_origin(raw) == expected


def test_build_report_reduces_both_endpoints(tmp_path) -> None:
    from maxim.simulation.bridge import SimulationBridge
    from maxim.simulation.report import build_report, save_report

    report = build_report(
        goal="g",
        mode="m",
        bridge=SimulationBridge(),
        duration_s=1.0,
        finish_reason="completed",
        language_endpoint=SECRET_URL,
        aut_endpoint="http://bob:pw@10.0.0.2:8101/v1?t=x",
        started_at=1790000000.0,
        provenance={},
    )
    text = Path(save_report(report, base_dir=str(tmp_path))).read_text()
    saved = json.loads(text)
    assert saved["language_endpoint"] == "https://leader.example.com:8443"
    assert saved["aut_endpoint"] == "http://10.0.0.2:8101"
    for secret in ("s3cr3t", "alice", "tok-abc123", "hunter2", "bob", "pw@"):
        assert secret not in text


def test_serving_provenance_reduces_entry_endpoints_and_head_keys(monkeypatch) -> None:
    from maxim.models.language import call_accounting as CA
    from maxim.simulation import serving_provenance as SP

    monkeypatch.setattr(SP, "_ACTIVE", {})
    monkeypatch.setattr(SP, "_STARTS", {})
    serial = CA.register_router()
    token = SP.capture_serving_start()
    for kind, endpoint in (("other", SECRET_URL), ("peer", "http://u:p@127.0.0.1:8100/v1"), ("llama", "/m/x.gguf")):
        CA.record_start(serial)
        served = CA.ServedCall(
            provider_key=kind,
            model="m",
            backend_kind=kind,
            endpoint=endpoint,
            served_id="",
            quantization=None,
            seed=None,
            seed_honoured=False,
            temp=0.3,
        )
        CA.record_outcome(serial, "success", served)
    router = SimpleNamespace(
        call_serial=serial,
        wait_idle=lambda _t: True,
        serving_probes=lambda: [
            {"endpoint": SECRET_URL, "cached_served_id": "", "fresh_served_id": "", "leader_disk_head": "abc"}
        ],
    )
    monkeypatch.setattr(SP, "BackgroundDigest", lambda _p: SimpleNamespace(result=lambda _t: None))
    prov = SP.serving_provenance(
        token, llm_worker=SimpleNamespace(router=router), aut_worker=None, resume=None, adopted=False, goal="g"
    )
    endpoints = {e["backend_kind"]: e["endpoint"] for e in prov["llm_calls"][str(serial)]["entries"]}
    assert endpoints == {
        "other": "https://leader.example.com:8443",
        "peer": "http://127.0.0.1:8100",
        "llama": "/m/x.gguf",  # a model FILE path, not a URL (path hygiene is #997's item 3, filed separately)
    }
    assert prov["leader_disk_heads"] == {"https://leader.example.com:8443": "abc"}
    text = json.dumps(prov)
    for secret in ("s3cr3t", "alice", "tok-abc123", "hunter2", "u:p@"):
        assert secret not in text


# ── item 2: no record producer stamps the process's raw argv ─────────────────────────────────────────


def _record_producers() -> list[str]:
    """Every module under ``simulation/`` plus every script that stamps a record's provenance (it calls the
    provenance stampers, or is one)."""
    found = sorted(str(p.relative_to(REPO)) for p in (REPO / "src/maxim/simulation").rglob("*.py"))
    stampers = ("executed_code_provenance", "stamp_harness_row")
    for path in sorted((REPO / "scripts").rglob("*.py")):
        if path.name == "_provenance.py" or any(name in path.read_text() for name in stampers):
            found.append(str(path.relative_to(REPO)))
    found += ["src/maxim/utils/code_tree.py"]
    return found


def _argv_reads(tree: ast.AST) -> list[int]:
    """Lines that read the process's command line: ``sys.argv`` / ``sys.orig_argv`` under any alias,
    ``from sys import argv``, ``getattr(sys, "argv")``, a ``cmdline()`` call (psutil) or ``/proc/self/cmdline``."""
    sys_names = {"sys"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            sys_names |= {a.asname or a.name for a in node.names if a.name == "sys"}
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "sys":
            if any(a.name in {"argv", "orig_argv", "*"} for a in node.names):
                hits.append(node.lineno)
        elif isinstance(node, ast.Attribute) and node.attr in {"argv", "orig_argv"}:
            if isinstance(node.value, ast.Name) and node.value.id in sys_names:
                hits.append(node.lineno)
        elif isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id == "getattr" and len(node.args) >= 2:
                if isinstance(node.args[1], ast.Constant) and node.args[1].value in {"argv", "orig_argv"}:
                    hits.append(node.lineno)
            elif isinstance(fn, ast.Attribute) and fn.attr == "cmdline":
                hits.append(node.lineno)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and "/proc/self/cmdline" in node.value:
            hits.append(node.lineno)
    return hits


def test_the_producer_scan_covers_the_known_writers() -> None:
    producers = set(_record_producers())
    for rel in (
        "src/maxim/simulation/report.py",
        "src/maxim/simulation/serving_provenance.py",
        "src/maxim/simulation/orchestrator.py",
        "scripts/_provenance.py",
        "scripts/o19_rerun.py",
    ):
        assert rel in producers, rel


@pytest.mark.parametrize(
    "source",
    [
        "import sys\nx = sys.argv",
        "import sys as s\nx = s.orig_argv",
        "from sys import argv",
        "import sys\nx = getattr(sys, 'argv')",
        "import psutil\nx = psutil.Process().cmdline()",
        "x = open('/proc/self/cmdline').read()",
    ],
)
def test_the_argv_scan_sees_each_spelling(source: str) -> None:
    assert _argv_reads(ast.parse(source))


def test_no_record_producer_reads_the_raw_argv() -> None:
    """A record that needs the command stamps named parsed fields, never the process's argv: other CLIs carry
    secrets in flags (``bench/cli.py --api-key``, ``hive_cli.py --api-key``), so raw argv would be safe only by
    accident."""
    offenders = {}
    for rel in _record_producers():
        hits = _argv_reads(ast.parse((REPO / rel).read_text(), filename=rel))
        if hits:
            offenders[rel] = hits
    assert offenders == {}


# ── design-pass folds ────────────────────────────────────────────────────────────────────────────────


def test_every_writer_reduces_the_record_however_the_report_was_built(tmp_path, capsys) -> None:
    """A ``SimulationReport`` built directly (not through ``build_report``) is reduced at serialization too, and
    an httpx-style error string keeps no credentials."""
    from maxim.simulation.report import SimulationReport, emit_report_json, save_report

    err = f"Client error '401 Unauthorized' for url '{SECRET_URL}'"
    report = SimulationReport(
        session_id="s997",
        language_endpoint=SECRET_URL,
        aut_endpoint=SECRET_URL,
        provider_failures={"p": {"consecutive_errors": 1, "last_error": err, "backoff_s": 0.0}},
    )
    saved = Path(save_report(report, base_dir=str(tmp_path))).read_text()
    emit_report_json(report, "-")
    emitted = capsys.readouterr().out
    for text in (saved, emitted):
        record = json.loads(text)
        assert record["language_endpoint"] == record["aut_endpoint"] == "https://leader.example.com:8443"
        assert "https://leader.example.com:8443" in record["provider_failures"]["p"]["last_error"]
        for secret in ("s3cr3t", "alice", "tok-abc123", "hunter2"):
            assert secret not in text


def test_redact_urls_reduces_each_url_in_free_text() -> None:
    from maxim.utils.http import redact_urls

    text = f"entry ('p', 'm', 'peer', '{SECRET_URL}', '') vanished; see unix:///s.sock and http://h:1/x?t=y"
    out = redact_urls(text)
    assert out == ("entry ('p', 'm', 'peer', 'https://leader.example.com:8443', '') vanished; see <url> and http://h:1")
    assert redact_urls("no url here") == "no url here"


def test_the_sampling_problems_keep_no_endpoint_credentials(monkeypatch) -> None:
    from maxim.models.language import call_accounting as CA
    from maxim.simulation import serving_provenance as SP

    monkeypatch.setattr(SP, "_ACTIVE", {})
    monkeypatch.setattr(SP, "_STARTS", {})
    token = SP.capture_serving_start()
    problem = f"router 1: entry ('p', 'm', 'peer', '{SECRET_URL}') vanished"
    monkeypatch.setattr(CA, "delta", lambda _a, _b: ({}, [problem]))
    prov = SP.serving_provenance(token, llm_worker=None, aut_worker=None, resume=None, adopted=False, goal="g")
    assert prov["sampling_complete"] is False
    assert "s3cr3t" not in json.dumps(prov) and "https://leader.example.com:8443" in prov["sampling_problems"][0]


def test_o19_verdict_reduces_both_sides_of_the_c4_endpoint_check() -> None:
    """The standalone verdict script keeps its own copy of the rule; it must agree with ``url_origin``."""
    import importlib.util

    from maxim.utils.http import url_origin

    spec = importlib.util.spec_from_file_location("o19_verdict_997", REPO / "scripts/o19_verdict.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for raw in (
        SECRET_URL,
        "http://127.0.0.1:8100/v1",
        "http://127.0.0.1:8100",
        "http://[::1]:8100/",
        "",
        "x",
        "http://a%40b/",
        "http://h:abc/",
        "http://[::1",
    ):
        assert mod._endpoint_origin(raw) == url_origin(raw), raw


# ── review-round folds ───────────────────────────────────────────────────────────────────────────────


def test_redact_urls_never_raises_and_keeps_no_password_tail() -> None:
    from maxim.utils.http import redact_urls

    assert redact_urls("bad url http://[oops and http://a]b/") == "bad url <url> and <url>"
    assert redact_urls("http://u:p'w@host.example/v1 failed") == "http://host.example failed"
    assert redact_urls("(see http://h:1/x), next.") == "(see http://h:1), next."


def test_provider_failures_scrub_before_the_cut(monkeypatch) -> None:
    """A URL straddling the 200-character cut: the scrub runs first, so neither its credentials nor a half URL
    that would break the second pass reach the record."""
    from maxim.simulation.report import SimulationReport, _provider_failures, _record_dict

    err = "x" * 150 + " " + SECRET_URL + " tail"  # the raw URL crosses 200; the reduced one fits
    state = SimpleNamespace(consecutive_errors=2, last_error=err, backoff_until=0.0)
    healthy = SimpleNamespace(consecutive_errors=0, last_error="", backoff_until=0.0)
    failures = _provider_failures(SimpleNamespace(_provider_states={"p": state, "ok": healthy}))
    assert set(failures) == {"p"}
    assert "s3cr3t" not in failures["p"]["last_error"] and "alice" not in failures["p"]["last_error"]
    assert failures["p"]["last_error"].endswith("https://leader.example.com:8443 tail")
    unbalanced = SimpleNamespace(
        consecutive_errors=1, last_error="x" * 170 + " http://[::1]:8100/v1/chat/completions", backoff_until=0.0
    )
    assert _provider_failures(SimpleNamespace(_provider_states={"p": unbalanced}))["p"]["last_error"].endswith(
        "http://[::1]:8100"
    )
    record = _record_dict(SimulationReport(provider_failures={"p": {"last_error": "x" * 190 + " http://[::1"}}))
    assert record["provider_failures"]["p"]["last_error"].endswith("<url>")


def test_redact_urls_is_linear_on_long_hostile_text() -> None:
    """The scrub runs on untruncated error text (it precedes the 200-character cut), which can embed a whole
    response body: a long letter run before ``://`` must not make the scheme scan quadratic."""
    import time

    from maxim.utils.http import redact_urls

    for text in ("a" * 100_000 + "://", "a1" * 50_000):
        start = time.perf_counter()
        redact_urls(text)
        assert time.perf_counter() - start < 1.0, text[:8]
