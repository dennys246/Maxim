"""Console run history + health — read-only, auth-gated, resolver-bounded (contract 0.6.0).

The Console becomes the place to explore what an agent did and why: past runs,
their ordered actions with per-action credit, and the process's health
(heartbeat snapshots, stalls, swallowed exceptions grouped by site). Design +
adversarial pass: maxim-pulse docs/plans/log_explorer.md §4.

The resolver is a SECURITY BOUNDARY. `utils/paths.py::resolve_run_dir` must
never see request data: a path-like argument returns any directory on disk, it
searches the CWD first, and it follows symlinks out of sim_reports. Every
refusal test below is paired with an accepting counterpart so a guard that
refuses everything cannot pass.

The last section holds STRICT red gates for the producers this contract reads
but does not ship (stall hook, per-action credit, per-run event trail): each
flips when its producer lands and must then be un-xfailed, never deleted.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="requires the `console` extra (fastapi/uvicorn)")

from fastapi.testclient import TestClient  # noqa: E402

from maxim.console.server import build_app  # noqa: E402
from maxim.utils.paths import _reset_caches  # noqa: E402
from tests.policy_helpers import TEST_POLICY  # noqa: E402

_TOKEN = "mxc_" + "h" * 43
_AUTH = {"Authorization": f"Bearer {_TOKEN}"}
_SRC = Path(__file__).resolve().parents[2] / "src" / "maxim"

NEWEST = "20260903_005659"
OLDER = "20260901_120000"
RESEARCH = "research_20260801_000000"
ABORTED = "20260902_101010"
LINKED = "20260801_000001"
LEAK_SENTINEL = "the-operators-private-file-do-not-serve"

posix_only = pytest.mark.skipif(not hasattr(os, "O_DIRECTORY"), reason="run history needs anchored opens (POSIX)")


def _write_jsonl(path: Path, rows: list[dict | str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write((row if isinstance(row, str) else json.dumps(row)) + "\n")


def _report(session_id: str, **over) -> dict:
    base = {
        "_format_version": "1.0",
        "session_id": session_id,
        "timestamp": "2026-09-03 01:06:46",  # build/END time, local
        "goal": "dm:the_darkened_cavern",
        "mode": "generative",
        "duration_s": 407.0,
        "turns": 1,
        "finish_reason": "llm_wedged",
        "total_actions": 3,
        "blocked_actions": 0,
        "cost_usd": 0.0,
        "tool_usage": {"sense_tools": 2, "respond": 1},
        "tool_success_rates": {"sense_tools": 0.5, "respond": 1.0},
        "response_texts": ["hello"],
        "latency_p50_ms": 120.0,
    }
    base.update(over)
    return base


@pytest.fixture()
def runs(monkeypatch, tmp_path):
    """A data home with the shapes this machine actually holds (see the plan §1)."""
    home = tmp_path / "data"
    monkeypatch.setenv("MAXIM_DATA_HOME", str(home))
    monkeypatch.delenv("MAXIM_CONSOLE_SANDBOX", raising=False)
    monkeypatch.delenv("MAXIM_CONSOLE_ALLOWED_ORIGINS", raising=False)
    _reset_caches()
    root = home / "sim_reports"
    root.mkdir(parents=True)

    # The newest run: no start time in report.json (true of every local
    # report), an actions file written by a credit-recording (1.2) writer.
    newest = root / NEWEST
    newest.mkdir()
    (newest / "report.json").write_text(json.dumps(_report(NEWEST)))
    _write_jsonl(
        newest / "actions.jsonl",
        [
            {"_format_version": "1.2", "_record_kind": "header", "session_id": NEWEST},
            {"timestamp": 1788419020.0, "tool": "sense_tools", "params": {}, "success": True, "blocked": False},
            {
                "timestamp": 1788419021.0,
                "tool": "sense_tools",
                "params": {"q": "door"},
                "success": False,
                "error": "no such door",
                "blocked": False,
                "rpe": -0.42,
                "pain": 0.3,
                "nac_failure_booked": "tool:sense_tools:failure",
            },
            {"timestamp": 1788419022.0, "tool": "respond", "params": {}, "success": True, "blocked": False},
        ],
    )

    older = root / OLDER
    older.mkdir()
    # `ts` (start epoch) is newer than every local report but must be honoured;
    # an old report carries its mode as `persona`.
    (older / "report.json").write_text(json.dumps(_report(OLDER, ts=1788264000.0, mode=None, persona="adventure")))
    _write_jsonl(older / "actions.jsonl", [{"_format_version": "1.1", "_record_kind": "header"}])

    # A pre-credit (1.1) actions file whose line happens to carry a
    # credit-named key: it must NOT be read as recorded credit.
    research = root / RESEARCH
    research.mkdir()
    (research / "report.json").write_text(json.dumps(_report(RESEARCH)))
    _write_jsonl(
        research / "actions.jsonl",
        [
            {"_format_version": "1.1", "_record_kind": "header"},
            {"timestamp": 1.0, "tool": "respond", "success": True, "rpe": 0.9},
            {"timestamp": 2.0, "tool": "respond", "success": True},
        ],
    )

    aborted = root / ABORTED
    aborted.mkdir()
    (aborted / "generated_campaign.yaml").write_text("name: x\n")

    (root / "campaign_analysis_20260801.json").write_text("{}")

    # Hostile shapes: a run DIR symlinked outside the root, and a run whose
    # report.json / actions.jsonl are symlinks to a secret.
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "report.json").write_text(json.dumps(_report("evil", goal=LEAK_SENTINEL)))
    (root / "evil_link").symlink_to(outside, target_is_directory=True)

    secret = tmp_path / "secret.txt"
    secret.write_text(LEAK_SENTINEL)
    linked = root / LINKED
    linked.mkdir()
    (linked / "report.json").symlink_to(secret)
    (linked / "actions.jsonl").symlink_to(secret)

    yield root
    _reset_caches()


@pytest.fixture()
def client(runs):
    with TestClient(build_app(None, auth_token=_TOKEN)) as c:
        yield c


def _get(client, path, **params):
    return client.get(path, headers=_AUTH, params=params)


def _sandbox_env(monkeypatch):
    monkeypatch.setenv("MAXIM_CONSOLE_SANDBOX", "1")
    monkeypatch.setenv("MAXIM_CONSOLE_ALLOWED_ORIGINS", "https://sandbox.example")


def _new_run(root: Path, name: str, report: dict | str | None = None, actions: list | None = None) -> Path:
    d = root / name
    d.mkdir()
    if report is not None:
        (d / "report.json").write_text(report if isinstance(report, str) else json.dumps(report))
    if actions is not None:
        _write_jsonl(d / "actions.jsonl", actions)
    return d


# ── the resolver: a security boundary ────────────────────────────────────────


@posix_only
class TestResolver:
    def test_a_real_run_opens(self, runs):
        from maxim.console.run_store import open_console_run

        fd = open_console_run(NEWEST)
        try:
            assert os.fstat(fd).st_ino == os.stat(runs / NEWEST).st_ino
        finally:
            os.close(fd)

    @pytest.mark.parametrize(
        "bad",
        ["", ".", "..", "../sim_reports", "a/b", "/etc", "~", "~root", "%2e%2e", "a\x00b", "é", "x" * 65,
         " " + NEWEST, NEWEST + "\n"],
    )  # fmt: skip
    def test_ids_that_are_not_plain_names_are_not_found(self, runs, bad):
        from maxim.console.run_store import RunNotFound, open_console_run

        with pytest.raises(RunNotFound):
            open_console_run(bad)

    def test_a_symlinked_run_dir_is_not_found(self, runs):
        from maxim.console.run_store import RunNotFound, open_console_run

        with pytest.raises(RunNotFound):
            open_console_run("evil_link")

    def test_a_symlinked_run_dir_pointing_back_inside_is_still_refused(self, runs):
        from maxim.console.run_store import RunNotFound, open_console_run

        (runs / "inner_link").symlink_to(runs / NEWEST, target_is_directory=True)
        with pytest.raises(RunNotFound):
            open_console_run("inner_link")

    def test_the_cwd_is_never_searched(self, runs, tmp_path, monkeypatch):
        # resolve_run_dir searches cwd FIRST; the console resolver must not.
        from maxim.console.run_store import RunNotFound, open_console_run

        decoy = tmp_path / "cwd"
        (decoy / "20991231_235959").mkdir(parents=True)
        (decoy / "20991231_235959" / "report.json").write_text(json.dumps(_report("decoy")))
        monkeypatch.chdir(decoy)
        with pytest.raises(RunNotFound):
            open_console_run("20991231_235959")

    def test_a_non_directory_entry_is_not_a_run(self, runs):
        from maxim.console.run_store import RunNotFound, open_console_run

        with pytest.raises(RunNotFound):
            open_console_run("campaign_analysis_20260801.json")

    def test_without_anchored_opens_history_is_unavailable_not_weaker(self, runs, monkeypatch):
        from maxim.console import run_store

        monkeypatch.setattr(run_store, "HISTORY_SUPPORTED", False)
        with pytest.raises(run_store.RunNotFound):
            run_store.open_console_run(NEWEST)
        assert run_store.list_runs(cursor=None, limit=10, signer=run_store.CursorSigner())["runs"] == []
        with TestClient(build_app(None, auth_token=_TOKEN)) as c:
            seams = {s["name"]: s for s in _get(c, "/api/identity").json()["seams"]}
        assert seams["runs"]["live"] is False


@posix_only
class TestMembers:
    def test_a_hardlinked_member_is_refused(self, runs, tmp_path):
        # Passes lstat AND O_NOFOLLOW; only the link count gives it away.
        secret = tmp_path / "secret_actions.jsonl"
        secret.write_text(json.dumps({"tool": LEAK_SENTINEL}) + "\n")
        d = _new_run(runs, "20260905_000000", report=_report("h"))
        os.link(secret, d / "actions.jsonl")
        with TestClient(build_app(None, auth_token=_TOKEN)) as c:
            r = _get(c, "/api/runs/20260905_000000/actions")
        assert r.status_code == 404 and LEAK_SENTINEL not in r.text

    def test_a_fifo_member_is_refused_without_hanging(self, runs):
        # Driven on a daemon thread with a bounded join, not through a
        # TestClient: a regression to a blocking open must FAIL here, not hang
        # the suite on client shutdown.
        from maxim.console import run_store

        d = _new_run(runs, "20260905_000001", report=_report("f"))
        os.mkfifo(d / "actions.jsonl")
        outcome: list[str] = []

        def call() -> None:
            try:
                run_store.page_actions("20260905_000001", cursor=None, limit=10, signer=run_store.CursorSigner())
                outcome.append("served")
            except run_store.RunNotFound:
                outcome.append("refused")

        t = threading.Thread(target=call, daemon=True)
        t.start()
        t.join(timeout=5)
        assert not t.is_alive(), "a FIFO hung the request (opened blocking)"
        assert outcome == ["refused"]

    def test_a_run_dir_swapped_for_a_symlink_after_listing_is_not_followed(self, runs, tmp_path):
        # The anchored open is what makes this safe: the dir is opened relative
        # to the root with O_NOFOLLOW at READ time, not checked then re-walked.
        from maxim.console import run_store

        outside = tmp_path / "swap_target"
        outside.mkdir()
        (outside / "report.json").write_text(json.dumps(_report("x", goal=LEAK_SENTINEL)))
        d = _new_run(runs, "20260905_000002", report=_report("ok"))
        for child in d.iterdir():
            child.unlink()
        d.rmdir()
        d.symlink_to(outside, target_is_directory=True)
        with pytest.raises(run_store.RunNotFound):
            run_store.read_report("20260905_000002")


# ── auth + sandbox posture ───────────────────────────────────────────────────

_NEW_ROUTES = [
    "/api/runs",
    f"/api/runs/{NEWEST}",
    f"/api/runs/{NEWEST}/actions",
    "/api/health",
    "/api/health/degraded",
]


class TestPosture:
    @pytest.mark.parametrize("path", _NEW_ROUTES)
    def test_401_without_the_token(self, client, path):
        r = client.get(path)
        assert r.status_code == 401
        assert r.headers.get("www-authenticate") == "Bearer"

    @pytest.mark.parametrize("path", _NEW_ROUTES)
    def test_serves_with_the_token(self, client, path):
        assert client.get(path, headers=_AUTH).status_code == 200

    @pytest.mark.parametrize("path", _NEW_ROUTES)
    def test_refused_in_sandbox_mode(self, runs, monkeypatch, path):
        # A public sandbox visitor must never read run history or host health.
        _sandbox_env(monkeypatch)
        with TestClient(build_app(None, auth_token=_TOKEN)) as c:
            assert c.get(path).status_code == 403

    @posix_only
    def test_identity_declares_runs_live(self, client):
        seams = {s["name"]: s for s in _get(client, "/api/identity").json()["seams"]}
        assert seams["runs"]["live"] is True

    def test_the_health_seam_is_live_only_while_sampling(self, client, runs, monkeypatch):
        # Without the monitor (an embedder, a test) nothing samples: say so.
        seams = {s["name"]: s for s in _get(client, "/api/identity").json()["seams"]}
        assert seams["health"]["live"] is False
        _fake_monitor(monkeypatch)
        with TestClient(build_app(None, auth_token=_TOKEN, health_monitor=True)) as c:
            seams = {s["name"]: s for s in c.get("/api/identity", headers=_AUTH).json()["seams"]}
        assert seams["health"]["live"] is True

    def test_degraded_is_live_whenever_the_server_serves(self, client):
        # The degraded view works without sampling (the swallow seam is always
        # on), so a device that does not sample must not hide it.
        seams = {s["name"]: s for s in _get(client, "/api/identity").json()["seams"]}
        assert seams["health"]["live"] is False
        assert seams["degraded"]["live"] is True

    def test_identity_marks_all_three_closed_in_sandbox(self, runs, monkeypatch):
        _sandbox_env(monkeypatch)
        with TestClient(build_app(None, auth_token=_TOKEN)) as c:
            seams = {s["name"]: s for s in c.get("/api/identity").json()["seams"]}
        assert seams["runs"]["live"] is False
        assert seams["health"]["live"] is False
        assert seams["degraded"]["live"] is False

    def test_the_routes_are_read_only(self, client):
        for path in _NEW_ROUTES:
            assert client.post(path, headers=_AUTH, json={}).status_code == 405

    def test_the_routes_are_tagged_for_the_viewer_operator_split(self, client):
        spec = client.app.openapi()
        for path in ("/api/runs", "/api/runs/{session_id}", "/api/runs/{session_id}/actions", "/api/health"):
            assert "history" in spec["paths"][path]["get"]["tags"]


# ── the run list ─────────────────────────────────────────────────────────────


@posix_only
class TestRunList:
    def test_lists_runs_newest_first_and_only_real_run_dirs(self, client):
        ids = [r["session_id"] for r in _get(client, "/api/runs").json()["runs"]]
        assert ids[0] == NEWEST
        assert set(ids) == {NEWEST, ABORTED, OLDER, LINKED, RESEARCH}
        assert "evil_link" not in ids  # a symlinked dir is skipped, not followed
        assert "campaign_analysis_20260801.json" not in ids

    def test_summary_fields_and_derived_start(self, client):
        runs = {r["session_id"]: r for r in _get(client, "/api/runs").json()["runs"]}
        newest = runs[NEWEST]
        # no `ts` in the report → start parsed from the %Y%m%d_%H%M%S id
        assert newest["started_at"] == time.mktime(time.strptime(NEWEST, "%Y%m%d_%H%M%S"))
        assert newest["mode"] == "generative"
        assert newest["finish_reason"] == "llm_wedged"
        assert newest["cost_usd"] == 0.0
        assert newest["turns"] == 1
        assert newest["has_report"] is True
        assert runs[OLDER]["started_at"] == 1788264000.0  # `ts` wins when present
        assert runs[OLDER]["mode"] == "adventure"  # the `persona` fallback for old reports

    def test_an_aborted_run_is_listed_without_a_report(self, client):
        runs = {r["session_id"]: r for r in _get(client, "/api/runs").json()["runs"]}
        assert runs[ABORTED]["has_report"] is False
        assert runs[ABORTED]["mode"] is None

    def test_a_symlinked_report_never_leaks(self, client):
        assert LEAK_SENTINEL not in _get(client, "/api/runs").text

    def test_one_hostile_report_never_500s_the_listing(self, runs):
        _new_run(runs, "20260904_000000", report="[" * 200_000)  # RecursionError in json.loads
        with TestClient(build_app(None, auth_token=_TOKEN)) as c:
            r = _get(c, "/api/runs")
            assert r.status_code == 200
            rows = {x["session_id"]: x for x in r.json()["runs"]}
            assert rows["20260904_000000"]["has_report"] is False
            assert _get(c, "/api/runs/20260904_000000").status_code == 404

    def test_an_unexpected_error_in_one_report_never_500s_the_listing(self, client, monkeypatch):
        from maxim.console import run_store

        real = run_store._load_report

        def flaky(dir_fd):
            raise RuntimeError("a bug in one report's path")

        run_store._summary_cache.clear()
        monkeypatch.setattr(run_store, "_load_report", flaky)
        assert _get(client, "/api/runs").status_code == 200
        monkeypatch.setattr(run_store, "_load_report", real)
        run_store._summary_cache.clear()

    def test_paging_covers_every_run_exactly_once(self, client):
        seen: list[str] = []
        cursor = None
        for _ in range(10):
            params = {"limit": 2} | ({"cursor": cursor} if cursor else {})
            page = _get(client, "/api/runs", **params).json()
            seen += [r["session_id"] for r in page["runs"]]
            cursor = page["next_cursor"]
            if cursor is None:
                break
        assert len(seen) == len(set(seen)) == 5

    @pytest.mark.parametrize("limit", [0, -1, 101, 10**9])
    def test_limit_is_bounded(self, client, limit):
        assert _get(client, "/api/runs", limit=limit).status_code == 422

    @pytest.mark.parametrize("cursor", ["garbage", "e30.AAAA", "e30." + "A" * 22])
    def test_a_forged_cursor_is_refused(self, client, cursor):
        assert _get(client, "/api/runs", cursor=cursor).status_code == 400

    def test_an_oversized_cursor_is_refused_before_it_is_parsed(self, client):
        assert _get(client, "/api/runs", cursor="x" * 5000).status_code == 422


# ── one run ──────────────────────────────────────────────────────────────────


@posix_only
class TestRunReport:
    def test_the_report(self, client):
        r = _get(client, f"/api/runs/{NEWEST}").json()
        assert r["session_id"] == NEWEST
        assert r["goal"] == "dm:the_darkened_cavern"
        assert r["tool_usage"] == {"sense_tools": 2, "respond": 1}
        # both ends are epochs, so a client can compare them
        assert r["ended_at"] == time.mktime(time.strptime("2026-09-03 01:06:46", "%Y-%m-%d %H:%M:%S"))

    def test_wrong_typed_report_fields_are_nulled_not_500(self, runs):
        _new_run(
            runs,
            "20260904_000001",
            report=_report("w", tool_usage={"t": 1e20, "u": "3", "v": 2}, response_texts=[None, "ok", 7], turns="x"),
        )
        with TestClient(build_app(None, auth_token=_TOKEN)) as c:
            r = _get(c, "/api/runs/20260904_000001")
        assert r.status_code == 200
        body = r.json()
        assert body["tool_usage"] == {"v": 2}
        assert body["response_texts"] == ["ok"]
        assert body["turns"] is None

    @pytest.mark.parametrize(
        "path",
        ["/api/runs/..", "/api/runs/%2e%2e", "/api/runs/..%2f..%2fetc", "/api/runs/evil_link", "/api/runs/nope",
         f"/api/runs/{ABORTED}"],
    )  # fmt: skip
    def test_refusals_are_all_404(self, client, path):
        r = client.get(path, headers=_AUTH)
        assert r.status_code == 404
        assert LEAK_SENTINEL not in r.text

    def test_a_symlinked_report_file_is_404_not_the_secret(self, client):
        r = _get(client, f"/api/runs/{LINKED}")
        assert r.status_code == 404
        assert LEAK_SENTINEL not in r.text

    def test_every_report_field_the_console_reads_exists_on_the_writer(self):
        # ~25 field names are read by string; a rename in SimulationReport
        # would otherwise serve null ("not recorded") forever, silently.
        import re as _re

        from maxim.simulation.report import SimulationReport

        src = (_SRC / "console" / "run_store.py").read_text()
        read = set(_re.findall(r'report\.get\("([a-z_]+)"\)', src))
        written = {f.name for f in dataclasses.fields(SimulationReport)} | {"_format_version", "persona"}
        assert len(read) > 15, "the pin found too few report reads — the regex no longer matches run_store"
        assert read <= written, f"run_store reads fields SimulationReport never writes: {sorted(read - written)}"


@posix_only
class TestActions:
    def test_actions_in_order_header_skipped(self, client):
        page = _get(client, f"/api/runs/{NEWEST}/actions").json()
        assert page["available"] is True
        assert page["format_version"] == "1.2"
        assert [a["tool"] for a in page["actions"]] == ["sense_tools", "sense_tools", "respond"]
        assert page["next_cursor"] is None

    def test_per_action_credit_from_a_credit_recording_writer(self, client):
        actions = _get(client, f"/api/runs/{NEWEST}/actions").json()["actions"]
        booked = actions[1]
        assert booked["success"] is False
        assert booked["rpe"] == -0.42
        assert booked["pain"] == 0.3
        assert booked["nac_failure_booked"] == "tool:sense_tools:failure"
        # recorded-and-absent is still RECORDED: a 1.2 writer may omit null keys
        assert actions[0]["credit_recorded"] is True
        assert actions[0]["rpe"] is None

    def test_a_pre_credit_file_never_reads_as_recorded(self, client):
        page = _get(client, f"/api/runs/{RESEARCH}/actions").json()
        assert page["format_version"] == "1.1"
        assert page["actions"][0]["credit_recorded"] is False
        assert page["actions"][0]["rpe"] is None  # the stray key is not trusted

    def test_a_run_without_an_actions_file_says_so(self, client):
        page = _get(client, f"/api/runs/{ABORTED}/actions").json()
        assert page["available"] is False
        assert page["actions"] == []
        assert page["reason"]

    def test_paging_carries_the_header_across_pages(self, client):
        first = _get(client, f"/api/runs/{NEWEST}/actions", limit=2).json()
        assert [a["tool"] for a in first["actions"]] == ["sense_tools", "sense_tools"]
        rest = _get(client, f"/api/runs/{NEWEST}/actions", limit=2, cursor=first["next_cursor"]).json()
        assert [a["tool"] for a in rest["actions"]] == ["respond"]
        assert rest["format_version"] == "1.2"  # the header was on page 1 only
        assert rest["actions"][0]["credit_recorded"] is True
        assert rest["next_cursor"] is None

    def test_a_cursor_bound_to_another_run_is_refused_even_at_a_valid_boundary(self, client):
        # Offset 0 is always a line boundary in any file: only the binding refuses it.
        forged = client.app.state.cursor_signer.issue(
            {"run": NEWEST, "file": "actions.jsonl", "offset": 0, "state": {}}
        )
        assert _get(client, f"/api/runs/{RESEARCH}/actions", cursor=forged).status_code == 400
        own = client.app.state.cursor_signer.issue({"run": RESEARCH, "file": "actions.jsonl", "offset": 0, "state": {}})
        assert _get(client, f"/api/runs/{RESEARCH}/actions", cursor=own).status_code == 200

    def test_a_cursor_from_another_process_is_refused(self, runs, client):
        cur = _get(client, f"/api/runs/{NEWEST}/actions", limit=1).json()["next_cursor"]
        with TestClient(build_app(None, auth_token=_TOKEN)) as other:
            assert _get(other, f"/api/runs/{NEWEST}/actions", cursor=cur).status_code == 400

    def test_a_signed_mid_line_cursor_is_refused(self, client):
        signer = client.app.state.cursor_signer
        forged = signer.issue({"run": NEWEST, "file": "actions.jsonl", "offset": 3, "state": {}})
        assert _get(client, f"/api/runs/{NEWEST}/actions", cursor=forged).status_code == 400

    @pytest.mark.parametrize("offset", [10**12, -1, True, "12", 1.5])
    def test_a_cursor_offset_that_is_not_a_position_in_the_file_is_refused(self, client, offset):
        signer = client.app.state.cursor_signer
        forged = signer.issue({"run": NEWEST, "file": "actions.jsonl", "offset": offset, "state": {}})
        assert _get(client, f"/api/runs/{NEWEST}/actions", cursor=forged).status_code == 400

    def test_a_symlinked_actions_file_never_leaks(self, client):
        r = _get(client, f"/api/runs/{LINKED}/actions")
        assert r.status_code == 404
        assert LEAK_SENTINEL not in r.text

    def test_hostile_lines_are_skipped_and_counted(self, runs, client):
        path = runs / NEWEST / "actions.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write("{this is not json\n")
            f.write("[1, 2]\n")  # valid JSON, not an object
            f.write("[" * 200_000 + "\n")  # RecursionError, not ValueError
            f.write(json.dumps({"tool": "huge", "output": "y" * (1024 * 1024 + 10)}) + "\n")
            f.write(json.dumps({"tool": "after"}) + "\n")
        page = _get(client, f"/api/runs/{NEWEST}/actions", limit=500).json()
        tools = [a["tool"] for a in page["actions"]]
        assert "huge" not in tools and "after" in tools
        assert page["skipped"] == 4

    def test_an_unterminated_complete_last_line_is_served(self, runs, client):
        with open(runs / NEWEST / "actions.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"tool": "last"}))  # no trailing newline
        tools = [a["tool"] for a in _get(client, f"/api/runs/{NEWEST}/actions").json()["actions"]]
        assert tools[-1] == "last"

    def test_a_fresh_partial_last_line_is_left_for_the_next_page(self, runs, client):
        with open(runs / NEWEST / "actions.jsonl", "a", encoding="utf-8") as f:
            f.write('{"tool": "half')  # a writer mid-line
        page = _get(client, f"/api/runs/{NEWEST}/actions").json()
        assert [a["tool"] for a in page["actions"]][-1] == "respond"
        assert page["skipped"] == 0
        assert page["next_cursor"] is not None

    def test_a_stale_torn_tail_ends_paging_instead_of_looping(self, runs, client):
        path = runs / NEWEST / "actions.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write('{"tool": "crashed mid-wri')  # a crashed writer's torn tail
        old = time.time() - 3600
        os.utime(path, (old, old))
        page = _get(client, f"/api/runs/{NEWEST}/actions").json()
        assert page["next_cursor"] is None
        assert page["skipped"] == 1


# ── health ───────────────────────────────────────────────────────────────────


class _FakeMonitor:
    """Stands in for the process singleton: never shells out, records wiring."""

    def __init__(self) -> None:
        self.listeners: list = []
        self.started = 0
        self.stopped = 0
        self.has_loop_state_hook = False
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    def register_listener(self, fn) -> None:
        self.listeners.append(fn)

    def unregister_listener(self, fn) -> None:
        self.listeners.remove(fn)

    def start(self) -> None:
        if not self._running:
            self.started += 1
        self._running = True

    def stop(self) -> None:
        self.stopped += 1
        self._running = False


def _fake_monitor(monkeypatch) -> _FakeMonitor:
    mon = _FakeMonitor()
    monkeypatch.setattr("maxim.runtime.heartbeat.get_heartbeat_monitor", lambda: mon)
    return mon


def _log(tmp_path, **kw):
    from maxim.console.health import HealthLog

    log = HealthLog(tmp_path / "health.jsonl", **kw)
    log.start()
    return log


def _lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


class TestHealthLog:
    def test_snapshots_are_allow_listed(self, tmp_path):
        log = _log(tmp_path)
        log.on_heartbeat(
            "heartbeat",
            {
                "timestamp": 100.0,
                "gpu": {"utilization_pct": 40.0, "vram_used_gb": 10.0},
                "cpu": {"usage_pct": 12.0},
                "memory": {"usage_pct": 50.0},
                "disk": {"path": "/Users/someone", "usage_pct": 70.0},
                "network": {"en0": {"ipv4": "192.168.1.9"}},
                "wifi": {"ssid": "HomeNet"},
                "platform": {"hostname": "box"},
                "lanes": {"large": {"p50_latency_ms": 900.0}},
                "loop": {"idle_s": 2.0, "state": "act"},
            },
        )
        log.stop()
        latest = log.view(since=0.0)["latest"]
        assert set(latest) <= {"timestamp", "gpu", "cpu", "memory", "disk", "lanes", "loop"}
        assert "path" not in latest["disk"]
        blob = (tmp_path / "health.jsonl").read_text() + json.dumps(latest)
        for leak in ("192.168.1.9", "HomeNet", "box", "/Users/someone"):
            assert leak not in blob

    def test_non_finite_numbers_become_null(self, tmp_path):
        # /ws would otherwise send a bare NaN token, which JSON.parse rejects.
        log = _log(tmp_path)
        log.on_heartbeat("heartbeat", {"cpu": {"usage_pct": float("nan"), "per": [float("inf"), 1.0]}})
        latest = log.view(since=0.0)["latest"]
        assert latest["cpu"] == {"usage_pct": None, "per": [None, 1.0]}
        json.dumps(latest, allow_nan=False)
        log.stop()

    def test_every_persisted_record_carries_a_format_version(self, tmp_path):
        log = _log(tmp_path, persist_interval_s=0.0)
        log.on_heartbeat("heartbeat", {"cpu": {"usage_pct": 1.0}})
        log.on_heartbeat("heartbeat_stall", {"idle_s": 40.0})
        log.on_swallowed("s", "E", "m")
        log.stop()
        recs = _lines(tmp_path / "health.jsonl")
        assert {r["kind"] for r in recs} == {"health", "health_stall", "swallowed_exception"}
        assert all(r["_format_version"] == "1.0" for r in recs)

    def test_stalls_are_kept_and_published(self, tmp_path):
        published: list[tuple[str, dict]] = []
        log = _log(tmp_path, publish=lambda kind, data: published.append((kind, data)))
        log.on_heartbeat("heartbeat_stall", {"idle_s": 45.0, "state": "deliberate", "threshold_s": 30.0})
        view = log.view(since=0.0)
        assert len(view["stalls"]) == 1 and view["stalls"][0]["data"]["idle_s"] == 45.0
        assert published and published[0][0] == "health_stall"
        log.stop()

    def test_stall_detection_is_reported_honestly(self, tmp_path):
        log = _log(tmp_path)
        view = log.view(since=0.0)
        assert view["stall_detection"] is False  # empty `stalls` here means unknown, not healthy
        assert view["recorded_since"] is None
        log.stop()

    def test_degraded_groups_swallowed_exceptions_by_site(self, tmp_path):
        log = _log(tmp_path)
        for _ in range(3):
            log.on_swallowed("executor.py:_fail_ran", "TypeError", "bad arg")
        log.on_swallowed("nac.py:_record_outcome_impl:collector", "KeyError", "'x'")
        out = log.degraded(since=0.0)
        sites = {s["site"]: s for s in out["sites"]}
        assert sites["executor.py:_fail_ran"]["count"] == 3
        assert sites["executor.py:_fail_ran"]["exc_type"] == "TypeError"
        assert sites["nac.py:_record_outcome_impl:collector"]["count"] == 1
        assert out["counted_since"] is not None
        log.stop()

    def test_degraded_counts_survive_a_restart(self, tmp_path):
        log = _log(tmp_path)
        for _ in range(4):
            log.on_swallowed("site.py:f", "KeyError", "k")
        log.stop()  # flushes the unwritten delta
        again = _log(tmp_path)
        assert {s["site"]: s["count"] for s in again.degraded(since=0.0)["sites"]} == {"site.py:f": 4}
        again.stop()

    def test_exception_samples_are_truncated_and_scrubbed(self, tmp_path):
        log = _log(tmp_path)
        home = os.path.expanduser("~")
        log.on_swallowed(
            "s",
            "ValueError",
            f"open {home}/x failed; GET https://u:pw@api.example/v1?key=abc; token=mxc_{'q' * 40} " + "z" * 5000,
        )
        sample = log.degraded(since=0.0)["sites"][0]["sample"]
        assert len(sample) <= 300
        for leak in (home + "/x", "u:pw@", "key=abc", "mxc_" + "q" * 20):
            assert leak not in sample
        log.stop()

    def test_snapshots_are_downsampled_on_disk_but_stalls_are_not(self, tmp_path):
        log = _log(tmp_path, persist_interval_s=3600.0)
        for i in range(12):
            log.on_heartbeat("heartbeat", {"timestamp": float(i), "cpu": {"usage_pct": 1.0}})
        log.on_heartbeat("heartbeat_stall", {"idle_s": 40.0})
        log.stop()
        kinds = [r["kind"] for r in _lines(tmp_path / "health.jsonl")]
        assert kinds.count("health") == 1
        assert kinds.count("health_stall") == 1

    def test_history_survives_a_restart_and_old_records_age_out(self, tmp_path):
        path = tmp_path / "health.jsonl"
        _write_jsonl(
            path,
            [
                {
                    "_format_version": "1.0",
                    "kind": "health_stall",
                    "ts": time.time() - 8 * 86400,
                    "data": {"idle_s": 99.0},
                },
                {"_format_version": "1.0", "kind": "health_stall", "ts": time.time() - 60, "data": {"idle_s": 31.0}},
            ],
        )
        log = _log(tmp_path)
        assert [s["data"]["idle_s"] for s in log.view(since=0.0)["stalls"]] == [31.0]
        log.stop()

    def test_a_reloaded_snapshot_is_re_allow_listed(self, tmp_path):
        # Never trust the file's shape: an older or edited file keeps no secrets.
        _write_jsonl(
            tmp_path / "health.jsonl",
            [{"_format_version": "1.0", "kind": "health", "ts": time.time(),
              "data": {"cpu": {"usage_pct": 1.0}, "wifi": {"ssid": "HomeNet"}, "disk": {"path": "/x", "usage_pct": 2}}}],
        )  # fmt: skip
        log = _log(tmp_path)
        point = log.view(since=0.0)["series"][0]["data"]
        assert "wifi" not in point and "path" not in point["disk"]
        log.stop()

    def test_age_out_rewrites_the_file_not_just_the_view(self, tmp_path):
        path = tmp_path / "health.jsonl"
        _write_jsonl(
            path,
            [
                {
                    "_format_version": "1.0",
                    "kind": "health_stall",
                    "ts": time.time() - 8 * 86400,
                    "data": {"idle_s": 99.0},
                }
            ],
        )
        log = _log(tmp_path)
        log.on_heartbeat("heartbeat_stall", {"idle_s": 31.0})  # first write triggers the hourly age-out
        log.stop()
        assert [r["data"]["idle_s"] for r in _lines(path)] == [31.0]

    def test_a_record_from_a_newer_writer_is_skipped_not_misread(self, tmp_path):
        _write_jsonl(
            tmp_path / "health.jsonl",
            [{"_format_version": "9.0", "kind": "health_stall", "ts": time.time(), "data": {"idle_s": 1.0}}],
        )
        log = _log(tmp_path)
        assert log.view(since=0.0)["stalls"] == []
        log.stop()

    def test_a_malformed_file_never_stops_the_console_starting(self, tmp_path):
        now = time.time()
        _write_jsonl(
            tmp_path / "health.jsonl",
            [
                {"_format_version": "1.0", "kind": "swallowed_exception", "ts": now, "data": {"site": "a", "count_delta": 1, "last_seen": None}},
                {"_format_version": "1.0", "kind": "swallowed_exception", "ts": now, "data": {"site": "a", "count_delta": 1, "last_seen": "x", "first_seen": "y"}},
                {"_format_version": "1.0", "kind": "swallowed_exception", "ts": now, "data": {"site": "b", "count_delta": True}},
                {"_format_version": "1.0", "kind": "health_stall", "ts": "now", "data": {}},
                "[" * 200_000,
                "not json",
                {"_format_version": "1.0", "kind": "health_stall", "ts": now, "data": {"idle_s": 35.0}},
            ],
        )  # fmt: skip
        log = _log(tmp_path)
        assert [s["data"]["idle_s"] for s in log.view(since=0.0)["stalls"]] == [35.0]
        sites = {s["site"]: s for s in log.degraded(since=0.0)["sites"]}
        assert sites["a"]["count"] == 2 and "b" not in sites
        from maxim.console.schemas import DegradedResponse

        DegradedResponse(**log.degraded(since=0.0))  # the route's model accepts what loaded
        log.stop()

    def test_a_symlinked_log_is_neither_read_nor_written(self, tmp_path):
        target = tmp_path / "elsewhere.txt"
        target.write_text("")
        (tmp_path / "health.jsonl").symlink_to(target)
        log = _log(tmp_path, persist_interval_s=0.0)
        log.on_heartbeat("heartbeat_stall", {"idle_s": 40.0})
        log.stop()
        assert target.read_text() == ""

    def test_the_file_is_capped(self, tmp_path):
        log = _log(tmp_path, max_bytes=20_000, persist_interval_s=0.0)
        for i in range(500):
            log.on_heartbeat("heartbeat", {"timestamp": float(i), "cpu": {"usage_pct": float(i)}})
        log.stop()
        assert (tmp_path / "health.jsonl").stat().st_size <= 20_000

    def test_the_writer_survives_a_record_it_cannot_age_out(self, tmp_path):
        # Delta review BLOCKER: a `ts` the loader tolerated killed the writer on
        # the first age-out; the queue then filled and shutdown hung.
        _write_jsonl(
            tmp_path / "health.jsonl",
            [{"_format_version": "2.0", "kind": "health", "ts": "2026-10-13T00:00:00Z", "data": {}},
             {"_format_version": "1.0", "kind": "health", "ts": None, "data": {}}, "[" * 200_000],
        )  # fmt: skip
        log = _log(tmp_path, persist_interval_s=0.0)
        log.on_heartbeat("heartbeat_stall", {"idle_s": 40.0})  # first write runs the age-out
        time.sleep(0.5)
        assert log._writer is not None and log._writer.is_alive()
        log.on_heartbeat("heartbeat_stall", {"idle_s": 41.0})
        t0 = time.monotonic()
        log.stop()
        assert time.monotonic() - t0 < 3
        assert [r["data"]["idle_s"] for r in _lines(tmp_path / "health.jsonl")] == [40.0, 41.0]

    def test_a_full_queue_drops_and_shutdown_stays_bounded(self, tmp_path, monkeypatch):
        from maxim.console import health

        monkeypatch.setattr(health, "_QUEUE_MAX", 2)
        monkeypatch.setattr(health, "_STOP_JOIN_S", 0.5)
        log = _log(tmp_path, persist_interval_s=0.0)
        gate = threading.Event()
        monkeypatch.setattr(log, "_append", lambda batch: gate.wait(5))  # a writer stuck on disk
        for i in range(10):
            log.on_heartbeat("heartbeat_stall", {"idle_s": float(i)})
        assert log.dropped >= 6  # producers never blocked
        t0 = time.monotonic()
        log.stop()
        assert time.monotonic() - t0 < 2
        gate.set()

    def test_an_unreadable_file_never_stops_the_console_starting(self, tmp_path):
        if os.geteuid() == 0:
            pytest.skip("root reads anything")
        path = tmp_path / "health.jsonl"
        path.write_text("{}\n")
        path.chmod(0)
        try:
            log = _log(tmp_path)  # must not raise
            assert log.view(since=0.0)["stalls"] == []
            log.stop()
        finally:
            path.chmod(0o600)

    def test_restarting_a_log_does_not_count_its_history_twice(self, tmp_path):
        log = _log(tmp_path)
        log.on_swallowed("s.py:f", "E", "m")
        log.stop()
        log.start()
        assert log.degraded(since=0.0)["sites"][0]["count"] == 1
        log.stop()

    def test_producers_never_touch_the_disk(self, tmp_path, monkeypatch):
        # The swallow seam runs inside a failing thread's `except`: memory +
        # enqueue only. Disk belongs to the writer thread.
        import builtins

        log = _log(tmp_path, persist_interval_s=0.0)
        caller = threading.current_thread()
        real_open = builtins.open
        opened_on_caller: list[str] = []

        def spy(*a, **k):
            if threading.current_thread() is caller:
                opened_on_caller.append(str(a[0]))
            return real_open(*a, **k)

        monkeypatch.setattr(builtins, "open", spy)
        log.on_swallowed("s", "E", "m")
        log.on_heartbeat("heartbeat", {"cpu": {"usage_pct": 1.0}})
        log.on_heartbeat("heartbeat_stall", {"idle_s": 40.0})
        monkeypatch.setattr(builtins, "open", real_open)
        log.stop()
        assert opened_on_caller == []


class TestHealthRoutes:
    def test_monitoring_off_by_default_in_build_app(self, client):
        body = _get(client, "/api/health").json()
        assert body["monitoring"] is False
        assert body["stall_detection"] is False
        assert body["latest"] is None

    def test_maxim_serve_style_app_wires_the_monitor(self, runs, monkeypatch):
        mon = _fake_monitor(monkeypatch)
        with TestClient(build_app(None, auth_token=_TOKEN, health_monitor=True)) as c:
            assert mon.started == 1 and len(mon.listeners) == 1
            mon.listeners[0]("heartbeat", {"cpu": {"usage_pct": 7.0}})
            body = _get(c, "/api/health").json()
            assert body["monitoring"] is True
            assert body["latest"]["cpu"]["usage_pct"] == 7.0
        assert mon.listeners == [] and mon.stopped == 1  # started it → stops it

    def test_a_monitor_someone_else_started_is_left_running(self, runs, monkeypatch):
        mon = _fake_monitor(monkeypatch)
        mon._running = True  # the router started the singleton first
        with TestClient(build_app(None, auth_token=_TOKEN, health_monitor=True)):
            pass
        assert mon.stopped == 0 and mon.listeners == []

    def test_one_app_shutting_down_does_not_stop_the_sampler_under_another(self, runs, monkeypatch):
        mon = _fake_monitor(monkeypatch)
        a = TestClient(build_app(None, auth_token=_TOKEN, health_monitor=True))
        b = TestClient(build_app(None, auth_token=_TOKEN, health_monitor=True))
        a.__enter__()
        b.__enter__()
        a.__exit__(None, None, None)
        assert mon.is_running and mon.stopped == 0
        b.__exit__(None, None, None)
        assert not mon.is_running and mon.stopped == 1

    def test_stall_detection_is_read_live_not_at_startup(self, runs, monkeypatch):
        # The agent loop will register its hook when a run starts — after the
        # server started. The flag must follow it.
        mon = _fake_monitor(monkeypatch)
        with TestClient(build_app(None, auth_token=_TOKEN, health_monitor=True)) as c:
            assert _get(c, "/api/health").json()["stall_detection"] is False
            mon.has_loop_state_hook = True
            assert _get(c, "/api/health").json()["stall_detection"] is True

    def test_two_apps_on_one_data_home_count_a_swallow_once(self, runs):
        from maxim.utils.logging import log_swallowed_exception

        a = TestClient(build_app(None, auth_token=_TOKEN))
        b = TestClient(build_app(None, auth_token=_TOKEN))
        a.__enter__()
        b.__enter__()
        try:
            assert a.app.state.health_log is b.app.state.health_log
            try:
                raise KeyError("k")
            except KeyError:
                log_swallowed_exception(site="test_console_run_history.py:shared")
            sites = {s["site"]: s for s in _get(a, "/api/health/degraded").json()["sites"]}
            assert sites["test_console_run_history.py:shared"]["count"] == 1
        finally:
            b.__exit__(None, None, None)
            a.__exit__(None, None, None)

    def test_a_monitor_that_fails_to_start_leaves_nothing_behind(self, runs, monkeypatch):
        from maxim.console import health
        from maxim.utils import logging as mlog

        mon = _fake_monitor(monkeypatch)

        def boom() -> None:
            raise RuntimeError("no sampler")

        monkeypatch.setattr(mon, "start", boom)
        before = len(mlog._swallow_listeners)
        with pytest.raises(RuntimeError):
            with TestClient(build_app(None, auth_token=_TOKEN, health_monitor=True)):
                pass
        assert len(mlog._swallow_listeners) == before
        assert health._monitor_users == 0

    def test_two_streaming_apps_each_hold_a_publisher_entry(self, runs, monkeypatch):
        """Each app's acquire adds an entry and its release removes one (F3: both are the hub's one publisher, so the
        entry count is what tells them apart; test_two_streaming_apps_publish_each_health_event_once checks delivery)."""
        _fake_monitor(monkeypatch)
        a = TestClient(build_app(None, auth_token=_TOKEN, health_monitor=True))
        b = TestClient(build_app(None, auth_token=_TOKEN, health_monitor=True))
        a.__enter__()
        b.__enter__()
        log = a.app.state.health_log
        try:
            assert log._publishers == [a.app.state.health_publish, b.app.state.health_publish]
            a.__exit__(None, None, None)
            assert log._publishers == [b.app.state.health_publish]  # B's entry remains after A leaves
        finally:
            b.__exit__(None, None, None)

    def test_two_streaming_apps_publish_each_health_event_once(self, runs, monkeypatch):
        """F3 (#1268 follow-up): the /ws hub is process-global, so two apps' publishers fanned every health event out
        to the same clients twice. One heartbeat reaches the hub once while both serve, and still once after one
        leaves."""
        from maxim.console import server

        _fake_monitor(monkeypatch)
        sent: list[str] = []
        monkeypatch.setattr(server._event_hub, "publish", lambda kind, *a, **k: sent.append(kind))
        a = TestClient(build_app(None, auth_token=_TOKEN, health_monitor=True))
        b = TestClient(build_app(None, auth_token=_TOKEN, health_monitor=True))
        a.__enter__()
        b.__enter__()
        log = a.app.state.health_log
        a_open = True
        try:
            log.on_heartbeat("heartbeat_stall", {"outstanding_s": 40.0})
            assert sent == ["health_stall"], sent
            a.__exit__(None, None, None)
            a_open = False
            sent.clear()
            log.on_heartbeat("heartbeat_stall", {"outstanding_s": 41.0})
            assert sent == ["health_stall"], "B still streams after A leaves"
        finally:
            if a_open:
                a.__exit__(None, None, None)
            b.__exit__(None, None, None)
        sent.clear()
        log.on_heartbeat("heartbeat_stall", {"outstanding_s": 42.0})
        assert sent == [], "no app serving: nothing streams"

    def test_each_app_on_a_shared_log_brings_its_own_publisher(self, runs, monkeypatch):
        # Second delta SF-3: a sandbox app registering first must not silence a
        # later non-sandbox app's health stream (or vice versa).
        _sandbox_env(monkeypatch)
        sandboxed = TestClient(build_app(None, auth_token=_TOKEN))
        monkeypatch.delenv("MAXIM_CONSOLE_SANDBOX")
        _fake_monitor(monkeypatch)
        normal = TestClient(build_app(None, auth_token=_TOKEN, health_monitor=True))
        sandboxed.__enter__()
        normal.__enter__()
        try:
            log = normal.app.state.health_log
            assert log is sandboxed.app.state.health_log
            assert log._publishers == [normal.app.state.health_publish]
        finally:
            normal.__exit__(None, None, None)
            sandboxed.__exit__(None, None, None)
        assert log._publishers == []

    def test_an_app_arriving_as_the_last_one_leaves_keeps_a_writer(self, runs):
        from maxim.console.health import shared_health_log
        from maxim.utils.paths import console_health_log

        log = shared_health_log(console_health_log())
        log.acquire(sample=False)
        real_stop = log.stop
        stopping = threading.Event()

        def slow_stop() -> None:  # hold the race window open, deterministically
            stopping.set()
            time.sleep(0.3)
            real_stop()

        log.stop = slow_stop  # type: ignore[method-assign]
        t = threading.Thread(target=log.release, kwargs={"sample": False})
        t.start()
        stopping.wait(5)  # the last user is mid-stop...
        log.stop = real_stop  # type: ignore[method-assign]
        log.acquire(sample=False)  # ...when a new app arrives
        t.join(timeout=10)
        try:
            assert log._users == 1
            assert log._writer is not None and log._writer.is_alive()
        finally:
            log.release(sample=False)

    def test_a_user_arriving_mid_stop_restarts_the_sampler(self, runs, monkeypatch):
        # Second delta SF-2: a stop outside the lock let a new user adopt a
        # sampler that was shutting down, and never see a sample.
        from maxim.console import health

        mon = _fake_monitor(monkeypatch)
        real_stop = mon.stop

        def slow_stop() -> None:
            time.sleep(0.3)
            real_stop()

        monkeypatch.setattr(mon, "stop", slow_stop)
        first = health._acquire_monitor()
        t = threading.Thread(target=health._release_monitor, args=(first,))
        t.start()
        time.sleep(0.05)  # the release is mid-stop
        second = health._acquire_monitor()
        t.join(timeout=5)
        try:
            assert mon.is_running, "the arriving user was left with a stopped sampler"
        finally:
            health._release_monitor(second)

    def test_dropped_records_are_reported_not_hidden(self, runs, monkeypatch):
        with TestClient(build_app(None, auth_token=_TOKEN)) as c:
            c.app.state.health_log.dropped = 3
            assert _get(c, "/api/health").json()["dropped_records"] == 3

    def test_shutdown_stops_the_writer(self, runs):
        with TestClient(build_app(None, auth_token=_TOKEN)) as c:
            log = c.app.state.health_log
            assert log._writer is not None
        assert log._writer is None

    def test_no_monitor_and_no_publishing_under_sandbox(self, runs, monkeypatch):
        # The REST refusal is no refusal if the same data rides /ws: the gate
        # is at the producer.
        _sandbox_env(monkeypatch)
        mon = _fake_monitor(monkeypatch)
        with TestClient(build_app(None, auth_token=_TOKEN, health_monitor=True)) as c:
            assert mon.started == 0 and mon.listeners == []
            assert c.app.state.health_publish is None
            assert c.app.state.health_log._publishers == []

    def test_run_serve_turns_the_monitor_on(self, runs, monkeypatch):
        from maxim.console import server

        seen: dict = {}

        def fake_build_app(*a, **k):
            seen.update(k)
            raise SystemExit(0)  # stop before uvicorn

        monkeypatch.setattr(server, "build_app", fake_build_app)
        monkeypatch.setattr("maxim.tunnel.keys.ensure_console_token", lambda: _TOKEN)
        with pytest.raises(SystemExit):
            server.run_serve(["--port", "8799"])
        assert seen.get("health_monitor") is True

    def test_degraded_route_sees_a_real_swallowed_exception(self, client):
        from maxim.utils.logging import log_swallowed_exception

        try:
            raise TypeError("probe failure")
        except TypeError:
            log_swallowed_exception(site="test_console_run_history.py:probe")
            log_swallowed_exception(site="test_console_run_history.py:probe")  # 2nd fire is DEBUG
        sites = {s["site"]: s for s in _get(client, "/api/health/degraded").json()["sites"]}
        # Exact even though the 2nd fire logs at DEBUG: the console listens to
        # the seam, not to a log level.
        assert sites["test_console_run_history.py:probe"]["count"] == 2

    def test_the_swallow_listener_is_removed_on_shutdown(self, runs):
        from maxim.utils import logging as mlog

        before = len(mlog._swallow_listeners)
        with TestClient(build_app(None, auth_token=_TOKEN)):
            assert len(mlog._swallow_listeners) == before + 1
        assert len(mlog._swallow_listeners) == before

    @pytest.mark.parametrize("since", ["nan", "-inf", "inf"])
    def test_since_must_be_finite(self, client, since):
        assert _get(client, "/api/health", since=since).status_code in (400, 422)


class TestHealthOnTheWire:
    def test_health_kinds_are_opt_in(self):
        from maxim.console.schemas import SubscribeFrame
        from maxim.console.server import _WsConn

        unfiltered = _WsConn()
        assert unfiltered.matches("health", "clean", "") is False  # a 0.5.0 client never sees one
        assert unfiltered.matches("health_stall", "clean", "") is False
        assert unfiltered.matches("nac", "bio", "NAc") is True  # everything else unchanged
        assert unfiltered.matches("heartbeat", "clean", "") is True
        opted = _WsConn()
        opted.apply_frame(SubscribeFrame(kinds=["health", "health_stall"]))
        assert opted.matches("health", "clean", "") is True
        assert opted.matches("health_stall", "clean", "") is True

    def test_opting_in_adds_health_and_starves_nothing(self):
        # Delta review B-1: `kinds` was both the opt-in switch and a filter, so
        # asking for health silenced chat, activity and thinking.
        from maxim.console.schemas import SubscribeFrame
        from maxim.console.server import _WsConn

        everything = _WsConn()
        everything.apply_frame(SubscribeFrame(kinds=["health"]))
        assert everything.matches("nac", "bio", "NAc") is True
        assert everything.matches("response", "clean", "RESPONSE") is True
        assert everything.matches("health", "clean", "") is True
        assert everything.matches("health_stall", "clean", "") is False  # only what was named

        robot = _WsConn()  # the Reachy shape: clean tier plus health
        robot.apply_frame(SubscribeFrame(tier="clean", kinds=["health", "health_stall"]))
        assert robot.matches("response", "clean", "RESPONSE") is True
        assert robot.matches("nac", "bio", "NAc") is False  # the tier still filters
        assert robot.matches("health", "clean", "") is True

        narrowed = _WsConn()  # ordinary kinds still narrow, opt-ins still add
        narrowed.apply_frame(SubscribeFrame(kinds=["nac", "health"]))
        assert narrowed.matches("nac", "bio", "NAc") is True
        assert narrowed.matches("hippocampus", "bio", "HIPPOCAMPUS") is False
        assert narrowed.matches("health", "clean", "") is True

    def test_an_explicitly_empty_kinds_list_keeps_its_0_5_0_meaning(self):
        from maxim.console.schemas import SubscribeFrame
        from maxim.console.server import _WsConn

        conn = _WsConn()
        conn.apply_frame(SubscribeFrame(kinds=[]))
        assert conn.matches("nac", "bio", "NAc") is False  # no ordinary events, as before
        assert conn.matches("heartbeat", "clean", "") is True  # meta-kinds still pass

    def test_no_sim_log_subsystem_collides_with_an_opt_in_kind(self):
        # `health` is a reserved wire name: a future HEALTH subsystem would be
        # swallowed by the opt-in rule. Make the reservation mechanical.
        from maxim.console.server import _OPT_IN_KINDS
        from maxim.simulation import sim_logger

        subsystems = {str(k).lower() for k in getattr(sim_logger, "_SUBSYSTEM_TIERS", {})}
        assert subsystems, "the guard found no subsystem table to check"
        assert not (subsystems & _OPT_IN_KINDS)

    def test_health_is_not_attributed_to_the_active_run(self):
        from maxim.console import server

        captured: list[dict] = []

        class _Loop:
            def is_closed(self) -> bool:
                return False

            def call_soon_threadsafe(self, fn, subsystem, evt) -> None:
                captured.append(evt)

        hub = server._EventHub()
        hub._loop = _Loop()  # type: ignore[assignment]
        hub.set_run("20260903_005659")
        hub.publish("health", "", {"cpu": {}}, run_scoped=False)
        hub.publish("run", "started", {})
        assert captured[0]["run_id"] is None
        assert captured[1]["run_id"] == "20260903_005659"


# ── the producer seams the health log listens on ─────────────────────────────


class TestProducerSeams:
    def test_the_heartbeat_monitor_notifies_listeners(self, monkeypatch):
        from maxim.runtime import heartbeat as hb

        monkeypatch.setattr(
            "maxim.runtime.system_metrics.collect_all", lambda: {"timestamp": 1.0, "cpu": {"usage_pct": 5.0}}
        )
        large = {"jobs_completed": 3, "jobs_failed": 0, "failure_rate": 0.0, "p50_latency_ms": 800.0}
        monkeypatch.setattr(
            "maxim.models.language.lane_metrics.get_metrics_registry",
            lambda: type("R", (), {"snapshot": lambda self: {"large": large}})(),
        )
        mon = hb.HeartbeatMonitor(interval_s=60)
        assert mon.has_loop_state_hook is False and mon.is_running is False
        mon.set_loop_state_hook(lambda: {"outstanding_s": 45.0, "state": "planning", "loops": 1})  # past the floor
        assert mon.has_loop_state_hook is True
        seen: list[tuple[str, dict]] = []
        listener = lambda kind, data: seen.append((kind, data))  # noqa: E731
        mon.register_listener(listener)
        data = mon._collect()
        mon._emit(data)
        mon._check_stall(data)
        assert [k for k, _ in seen] == ["heartbeat", "heartbeat_stall"]
        assert seen[1][1]["outstanding_s"] == 45.0 and seen[1][1]["p50_latency_ms"] == 800.0
        mon.unregister_listener(listener)
        mon._emit(data)
        assert len(seen) == 2  # unregistered: no more notifications

    def test_a_raising_heartbeat_listener_does_not_break_the_monitor(self):
        from maxim.runtime import heartbeat as hb

        mon = hb.HeartbeatMonitor(interval_s=60)

        def boom(kind, data):
            raise RuntimeError("listener bug")

        mon.register_listener(boom)
        mon._notify("heartbeat", {})  # must not raise

    def test_swallowed_exception_listener_sees_every_fire(self):
        from maxim.utils import logging as mlog

        seen: list[tuple[str, str]] = []

        def listener(site, exc_type, exc):
            seen.append((site, exc_type))

        mlog.register_swallow_listener(listener)
        try:
            for _ in range(3):
                try:
                    raise KeyError("k")
                except KeyError:
                    mlog.log_swallowed_exception(site="seam-test:site")
        finally:
            mlog.unregister_swallow_listener(listener)
        assert seen == [("seam-test:site", "KeyError")] * 3

    def test_a_raising_swallow_listener_is_contained_and_warned_once(self, caplog):
        from maxim.utils import logging as mlog

        def boom(site, exc_type, exc):
            raise RuntimeError("listener bug")

        mlog.register_swallow_listener(boom)
        try:
            with caplog.at_level(logging.WARNING, logger="maxim"):
                for _ in range(2):
                    try:
                        raise ValueError("v")
                    except ValueError:
                        mlog.log_swallowed_exception(site="seam-test:boom")  # must not raise
        finally:
            mlog.unregister_swallow_listener(boom)
        assert sum("swallow listener" in r.getMessage() for r in caplog.records) == 1


# ── strict red gates: the producers this contract reads but does not ship ────


def _real_monitor(monkeypatch):
    """The real HeartbeatMonitor (so its stall rule runs), with the system sampler stubbed and a long interval: the
    test drives each tick itself."""
    from maxim.runtime import heartbeat

    monkeypatch.setattr("maxim.runtime.system_metrics.collect_all", lambda: {})
    mon = heartbeat.HeartbeatMonitor(interval_s=3600.0)
    monkeypatch.setattr(heartbeat, "get_heartbeat_monitor", lambda: mon)
    return mon


def _stalled_run(mon, *, outstanding: float | None, in_flight: bool):
    """Drive one heartbeat tick inside a decorated run whose gate saw a worker with a planning turn ``outstanding``
    seconds old (None: no turn outstanding)."""
    from maxim.runtime import llm_call_registry, loop_state

    class _Worker:
        def outstanding_attempt_s(self):
            return outstanding

    @loop_state.notes_loop_worker
    def tick(*, llm_worker):
        return None

    @loop_state.loop_liveness_scope
    def run():
        tick(llm_worker=_Worker())
        call = llm_call_registry.register_call_start(tier="large", allowed_s=600.0) if in_flight else None
        try:
            assert mon.has_loop_state_hook
            mon._check_stall(mon._collect())
        finally:
            if call is not None:
                llm_call_registry.register_call_end(call)

    run()


def test_the_ws_stall_line_reads_the_stall_payload():
    """The /ws line for a ``health_stall`` reads the heartbeat's stall keys (outstanding_s, wedged_s): a renamed key
    would print "?" on every stall the console pushes."""
    from maxim.console.server import _health_message

    assert _health_message("health_stall", {"outstanding_s": 62.4, "wedged_s": None}) == (
        "stall: a planning turn outstanding 62s (no call in flight)"
    )
    assert _health_message("health_stall", {"outstanding_s": 90.0, "wedged_s": 75.2}).endswith("(its call wedged 75s)")


@pytest.mark.parametrize(
    ("outstanding", "in_flight", "stalls"),
    [(10_000.0, False, 1), (10_000.0, True, 0), (None, False, 0), (5.0, False, 0)],
)
def test_a_stuck_planning_turn_is_one_stall_on_api_health(runs, monkeypatch, outstanding, in_flight, stalls):
    """End to end (#1268 dormant producer 1, owner decisions 2026-10-14): a stall is a STUCK PLANNING TURN. A run
    registers its worker through the decorators; the real monitor's tick declares a stall only for a turn outstanding
    past compute_stall_threshold with no live call (stall_suppression); /api/health shows it. A live call, no turn
    (waiting for a human) or a young turn is no stall."""
    mon = _real_monitor(monkeypatch)
    with TestClient(build_app(None, auth_token=_TOKEN, health_monitor=True)) as c:
        _stalled_run(mon, outstanding=outstanding, in_flight=in_flight)
        body = _get(c, "/api/health").json()
        assert len(body["stalls"]) == stalls, body["stalls"]
        assert body["stall_detection"] is False  # the run ended: no run is registered, so no stale stall source


def test_gate_actions_jsonl_records_per_action_credit(tmp_path):
    from types import SimpleNamespace

    from maxim.simulation.report import save_action_log

    action = SimpleNamespace(
        timestamp=1.0, tool_name="t", tool_args={}, result_success=False, result_output=None, result_error="e",
        blocked=False, block_reason=None, agent_id=None, session_id="s", entity_class=None,
        rpe=-0.4, pain=0.2, nac_failure_booked="tool:t:failure",
    )  # fmt: skip
    bridge = SimpleNamespace(get_all_actions=lambda: [action])
    path = save_action_log(bridge, str(tmp_path), "s")
    lines = [json.loads(line) for line in Path(path).read_text().splitlines()]
    assert tuple(int(x) for x in lines[0]["_format_version"].split(".")) >= (1, 2)
    assert lines[1]["rpe"] == -0.4 and lines[1]["nac_failure_booked"] == "tool:t:failure"


def test_a_bridge_booked_failure_reaches_the_console_with_its_credit(runs):
    """End to end: a failure the tool-pain bridge books, run through the sim's instrumented executor and sink, saved by
    save_action_log and read back by the console, carries its credit. The writer-only gate above feeds save_action_log
    a stand-in that already has the fields; this one proves the real records carry them."""
    from types import SimpleNamespace

    from maxim.console import run_store
    from maxim.decisions.nac import NAc, NACConfig
    from maxim.runtime.bootstrap import build_executor
    from maxim.simulation.instrumented_executor import InstrumentedExecutor
    from maxim.simulation.report import save_action_log
    from maxim.simulation.sinks import RecordingSink
    from maxim.tools.base import Tool, ToolOutput
    from maxim.tools.registry import ToolRegistry

    class _Fails(Tool):
        name = "flaky_credit"
        description = "stub"
        input_schema: dict = {}

        def execute(self, **kwargs):
            return ToolOutput(success=False, error="boom")

    registry = ToolRegistry()
    registry.register(_Fails())
    sink = RecordingSink()
    executor = InstrumentedExecutor(
        build_executor(registry, pain_bus=None, policy=TEST_POLICY, nac=NAc(NACConfig())), sink
    )
    out = executor.execute({"tool_name": "flaky_credit", "params": {}})
    assert out.nac_failure_booked == "tool:flaky_credit"

    session_id = "20261013_000001"
    save_action_log(SimpleNamespace(get_all_actions=lambda: sink.actions), str(runs), session_id)
    _reset_caches()
    page = run_store.page_actions(session_id, cursor=None, limit=10, signer=run_store.CursorSigner())
    [action] = page["actions"]
    assert action["credit_recorded"] is True
    assert action["nac_failure_booked"] == "tool:flaky_credit"
    assert action["rpe"] == out.rpe and action["rpe"] is not None
    assert action["pain"] == out.pain  # None here (no pain source watched): pinned so a dropped copy shows


@pytest.mark.xfail(strict=True, reason="no writer produces a per-run events.jsonl trail yet")
def test_gate_a_run_writes_its_event_trail():
    writers = [p for p in (_SRC / "simulation").rglob("*.py") if "events.jsonl" in p.read_text(errors="replace")]
    assert writers, "the run event trail has no writer: the /api/runs/{id}/events route waits on it"


# ── refusal and fallback paths (each asserts the refusal, not just reaches it) ─


class TestCursorRefusals:
    def _forge(self, signer, body: bytes) -> str:
        from maxim.console.run_store import _b64e

        return f"{_b64e(body)}.{_b64e(signer._mac(body))}"

    def test_undecodable_base64_is_refused(self):
        # One base64 character is never a valid length: the decoder RAISES
        # (non-alphabet characters, by contrast, are silently discarded).
        from maxim.console.run_store import BadCursor, CursorSigner

        with pytest.raises(BadCursor, match="malformed"):
            CursorSigner().verify("A.A")

    def test_a_correctly_signed_body_that_is_not_json_is_refused(self):
        from maxim.console.run_store import BadCursor, CursorSigner

        signer = CursorSigner()
        with pytest.raises(BadCursor, match="malformed"):
            signer.verify(self._forge(signer, b"not json at all"))

    def test_a_correctly_signed_body_that_is_not_an_object_is_refused(self):
        from maxim.console.run_store import BadCursor, CursorSigner

        signer = CursorSigner()
        with pytest.raises(BadCursor, match="malformed"):
            signer.verify(self._forge(signer, b"[1, 2]"))

    @posix_only
    def test_a_signed_runs_cursor_with_wrong_key_types_is_refused(self, client):
        forged = client.app.state.cursor_signer.issue({"listing": "runs", "k": "late", "n": 7})
        assert _get(client, "/api/runs", cursor=forged).status_code == 400

    @posix_only
    def test_a_cursor_on_a_run_without_actions_is_refused(self, client):
        forged = client.app.state.cursor_signer.issue(
            {"run": ABORTED, "file": "actions.jsonl", "offset": 0, "state": {}}
        )
        assert _get(client, f"/api/runs/{ABORTED}/actions", cursor=forged).status_code == 400


@posix_only
class TestRunStoreFallbacks:
    def test_an_unopenable_root_is_not_found(self, runs, monkeypatch, tmp_path):
        from maxim.console import run_store

        monkeypatch.setattr(run_store, "_root", lambda: tmp_path / "no-such-root")
        with pytest.raises(run_store.RunNotFound):
            run_store.open_console_run(NEWEST)

    def test_a_member_outside_the_allow_list_is_refused(self, runs):
        from maxim.console import run_store

        fd = run_store.open_console_run(NEWEST)
        try:
            with pytest.raises(run_store.RunNotFound):
                run_store._open_member(fd, "aut_hippocampus.json")
        finally:
            os.close(fd)

    def test_an_oversized_report_is_not_served(self, runs, monkeypatch):
        from maxim.console import run_store

        monkeypatch.setattr(run_store, "MAX_REPORT_BYTES", 64)
        with pytest.raises(run_store.RunNotFound):
            run_store.read_report(NEWEST)

    @pytest.mark.parametrize(
        ("run_id", "report", "started"),
        [
            ("bisect-abc123", None, None),  # no timestamp in the id, no ts
            ("20261399_999999", None, None),  # timestamp-shaped, not a date
            (NEWEST, {"ts": 100.0}, 100.0),
        ],
    )
    def test_start_time_derivation(self, run_id, report, started):
        from maxim.console.run_store import _started_at

        assert _started_at(run_id, report) == started

    @pytest.mark.parametrize(
        ("report", "ended"),
        [
            ({"ts": 100.0, "duration_s": 5.0}, 105.0),  # timezone-safe when both exist
            ({"timestamp": 12345}, None),  # not a string
            ({"timestamp": "yesterday-ish"}, None),  # not the written format
        ],
    )
    def test_end_time_derivation(self, report, ended):
        from maxim.console.run_store import _ended_at

        assert _ended_at(report) == ended

    @pytest.mark.parametrize(("raw", "parsed"), [(None, None), (12, None), ("1.x", None), ("1.2", (1, 2))])
    def test_version_parsing(self, raw, parsed):
        from maxim.console.run_store import _version_tuple

        assert _version_tuple(raw) == parsed

    def test_an_unparseable_header_version_never_reads_as_recorded_credit(self, runs, client):
        _new_run(
            runs,
            "20260905_000003",
            report=_report("v"),
            actions=[{"_format_version": "1.x", "_record_kind": "header"}, {"tool": "t", "rpe": 0.5}],
        )
        page = _get(client, "/api/runs/20260905_000003/actions").json()
        assert page["actions"][0]["credit_recorded"] is False

    def test_an_entry_that_cannot_be_stat_ed_is_skipped_not_fatal(self, runs, monkeypatch):
        from maxim.console import run_store

        real_scandir = os.scandir

        class _Broken:
            name = "20260906_000000"

            def is_dir(self, follow_symlinks=True):
                raise OSError("vanished")

        class _Wrap:
            def __init__(self, path):
                self._it = real_scandir(path)

            def __enter__(self):
                return iter([*self._it, _Broken()])

            def __exit__(self, *exc):
                self._it.close()

        monkeypatch.setattr(run_store.os, "scandir", _Wrap)
        ids = [
            r["session_id"] for r in run_store.list_runs(cursor=None, limit=50, signer=run_store.CursorSigner())["runs"]
        ]
        assert "20260906_000000" not in ids and NEWEST in ids

    def test_a_file_that_shrank_after_fstat_ends_the_page_cleanly(self, runs, monkeypatch):
        # fstat reports more bytes than the file holds (truncated mid-read):
        # readline returns b"" and the page ends, with the oversized tail drained.
        from maxim.console import run_store

        path = runs / NEWEST / "actions.jsonl"
        with open(path, "ab") as f:
            f.write(b'{"tool": "' + b"y" * (run_store.MAX_LINE_BYTES + 10))  # oversized, unterminated
        real_fstat = os.fstat

        def inflated(fd):
            st = real_fstat(fd)
            if stat_is_reg(st):
                return os.stat_result((*st[:6], st.st_size + 4096, *st[7:]))
            return st

        import stat as _stat

        def stat_is_reg(st):
            return _stat.S_ISREG(st.st_mode)

        monkeypatch.setattr(run_store.os, "fstat", inflated)
        page = run_store.page_actions(NEWEST, cursor=None, limit=500, signer=run_store.CursorSigner())
        assert [a["tool"] for a in page["actions"]] == ["sense_tools", "sense_tools", "respond"]
        assert page["skipped"] == 1


class TestHealthLogFallbacks:
    def test_deeply_nested_values_are_capped_not_recursed_forever(self):
        from maxim.console.health import _finite

        nested: dict = {}
        cur = nested
        for _ in range(12):
            cur["x"] = {}
            cur = cur["x"]
        out = _finite(nested)
        depth = 0
        while isinstance(out, dict):
            out, depth = out.get("x"), depth + 1
        assert out is None and depth <= 10

    def test_start_is_idempotent(self, tmp_path):
        log = _log(tmp_path)
        writer = log._writer
        log.start()
        assert log._writer is writer
        log.stop()

    def test_a_long_series_is_downsampled_for_the_wire(self, tmp_path, monkeypatch):
        from maxim.console import health

        monkeypatch.setattr(health, "_MAX_SERIES_POINTS", 10)
        log = _log(tmp_path, persist_interval_s=0.0)
        for i in range(60):
            log.on_heartbeat("heartbeat", {"cpu": {"usage_pct": float(i)}})
        assert len(log.view(since=0.0)["series"]) <= 10
        log.stop()

    def test_aged_out_records_are_pruned_from_memory_as_new_ones_arrive(self, tmp_path):
        log = _log(tmp_path, keep_s=3600)
        old = time.time() - 7200
        log._history.append({"kind": "health", "ts": old, "data": {}})
        log._stalls.append({"kind": "health_stall", "ts": old, "data": {}})
        log._sites["old.py:f"] = {"site": "old.py:f", "count": 1, "first_seen": old, "last_seen": old,
                                  "exc_type": None, "sample": None, "unflushed": 0, "flushed_at": 0.0}  # fmt: skip
        log._recorded_since = old
        log.on_heartbeat("heartbeat_stall", {"idle_s": 31.0})  # an insert triggers the prune
        assert [r["data"] for r in log._history] == []
        assert [r["data"].get("idle_s") for r in log._stalls] == [31.0]
        assert "old.py:f" not in log._sites
        assert log._recorded_since >= time.time() - 3600 - 5
        log.stop()

    def test_a_raising_publisher_never_stops_health_capture(self, tmp_path):
        def boom(kind, data):
            raise RuntimeError("ws is down")

        log = _log(tmp_path, publish=boom)
        log.on_heartbeat("heartbeat_stall", {"idle_s": 33.0})  # must not raise
        assert log.view(since=0.0)["stalls"][0]["data"]["idle_s"] == 33.0
        log.stop()

    def test_a_log_that_is_not_started_writes_nothing(self, tmp_path):
        from maxim.console.health import HealthLog

        log = HealthLog(tmp_path / "health.jsonl")  # never started: no writer
        log.on_heartbeat("heartbeat_stall", {"idle_s": 40.0})
        assert not (tmp_path / "health.jsonl").exists()
        assert log.view(since=0.0)["stalls"][0]["data"]["idle_s"] == 40.0  # memory still serves

    def test_a_pathless_log_serves_from_memory_only(self):
        from maxim.console.health import HealthLog

        log = HealthLog(None)
        log.start()
        assert log._writer is None
        log.on_swallowed("s", "E", "m")
        assert log.degraded(since=0.0)["sites"][0]["count"] == 1
        log.stop()

    def test_a_busy_lock_drops_the_batch_and_says_so(self, tmp_path, monkeypatch):
        from filelock import FileLock

        from maxim.console import health

        monkeypatch.setattr(health, "_LOCK_TIMEOUT_S", 0.05)
        log = _log(tmp_path, persist_interval_s=0.0)
        with FileLock(str(tmp_path / "health.jsonl") + ".lock"):  # another process holds it
            log.on_heartbeat("heartbeat_stall", {"idle_s": 40.0})
            deadline = time.time() + 5
            while log.dropped == 0 and time.time() < deadline:
                time.sleep(0.02)
        assert log.dropped >= 1
        assert log.view(since=0.0)["dropped_records"] >= 1
        log.stop()

    def test_a_file_far_past_the_cap_is_not_parsed_at_start(self, tmp_path, monkeypatch):
        from maxim.console import health

        _write_jsonl(
            tmp_path / "health.jsonl",
            [{"_format_version": "1.0", "kind": "health_stall", "ts": time.time(), "data": {"idle_s": 1.0}}],
        )
        monkeypatch.setattr(health, "_MAX_LOAD_BYTES", 10)
        log = _log(tmp_path)
        assert log.view(since=0.0)["stalls"] == []
        log.stop()

    def test_lines_that_are_not_records_or_of_unknown_kind_are_skipped(self, tmp_path):
        _write_jsonl(
            tmp_path / "health.jsonl",
            [
                "[1, 2]",
                {"_format_version": "1.0", "kind": "mystery", "ts": time.time(), "data": {"x": 1}},
                {"_format_version": "1.0", "kind": "health_stall", "ts": time.time(), "data": {"idle_s": 36.0}},
            ],
        )
        log = _log(tmp_path)
        view = log.view(since=0.0)
        assert [s["data"]["idle_s"] for s in view["stalls"]] == [36.0]
        log.stop()
