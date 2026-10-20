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


@pytest.mark.xfail(strict=True, reason="#997 red gate: the record keeps the URL verbatim")
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
        ("unix:///tmp/llama.sock", ""),
        ("host.example:8100/v1", ""),  # no scheme
        ("", ""),
        (None, ""),
    ],
)
def test_report_url_keeps_only_scheme_host_and_port(raw, expected) -> None:
    from maxim.simulation.report import report_url

    assert report_url(raw) == expected


@pytest.mark.xfail(strict=True, reason="#997 red gate: the record keeps the URL verbatim")
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


@pytest.mark.xfail(strict=True, reason="#997 red gate: the record keeps the URL verbatim")
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

RECORD_PRODUCERS = (
    "src/maxim/simulation/report.py",
    "src/maxim/simulation/serving_provenance.py",
    "src/maxim/simulation/orchestrator.py",
    "src/maxim/utils/code_tree.py",
    "scripts/_provenance.py",
    "scripts/o19_rerun.py",
)


@pytest.mark.parametrize("rel", RECORD_PRODUCERS)
def test_no_record_producer_reads_the_raw_argv(rel: str) -> None:
    """A record that needs the command stamps named parsed fields, never ``sys.argv``: other CLIs carry secrets in
    flags (``bench/cli.py --api-key``, ``hive_cli.py --api-key``), so raw argv would be safe only by accident."""
    tree = ast.parse((REPO / rel).read_text(), filename=rel)
    hits = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr in {"argv", "orig_argv"}
        and isinstance(node.value, ast.Name)
        and node.value.id == "sys"
    ]
    assert hits == [], f"{rel} reads sys.argv at lines {hits}"


@pytest.mark.xfail(strict=True, reason="#997 red gate: the record keeps the URL verbatim")
def test_o19_verdict_reduces_both_sides_of_the_c4_endpoint_check() -> None:
    """The standalone verdict script keeps its own copy of the rule; it must agree with ``report_url``."""
    import importlib.util

    from maxim.simulation.report import report_url

    spec = importlib.util.spec_from_file_location("o19_verdict_997", REPO / "scripts/o19_verdict.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for raw in (SECRET_URL, "http://127.0.0.1:8100/v1", "http://127.0.0.1:8100", "http://[::1]:8100/", "", "x"):
        assert mod._endpoint_origin(raw) == report_url(raw), raw
