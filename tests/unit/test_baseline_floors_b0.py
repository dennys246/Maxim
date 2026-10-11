"""Baseline floors B0: per-router call accounting, the report's serving/scenario/session fields, and the
evidence gate's checks on them (docs/plans/baseline_floors_b0.md)."""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from maxim.models.language import call_accounting as CA
from maxim.models.language.config import LLMConfig
from maxim.models.language.router import LLMRouter
from maxim.simulation import serving_provenance as SP
from maxim.utils import scenario_inputs

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import _evidence_records as R  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_sim_registry(monkeypatch):
    # Isolate this file's registrations; the leak case itself is tested below with real threads.
    monkeypatch.setattr(SP, "_ACTIVE", {})
    monkeypatch.setattr(SP, "_STARTS", {})


def _served(temp: float = 0.3, model: str = "m") -> CA.ServedCall:
    return CA.ServedCall(
        provider_key="p",
        model=model,
        backend_kind="other",
        endpoint="",
        served_id="",
        quantization=None,
        seed=None,
        seed_honoured=False,
        temp=temp,
    )


# ── call_accounting ──────────────────────────────────────────────────────────


def test_a_new_router_is_registered_with_zero_counts() -> None:
    serial = CA.register_router()
    snap = CA.snapshot()[serial]
    assert snap["entries"] == {}
    assert snap["counts"] == {"started": 0, "success": 0, "exhausted": 0, "budget_reject": 0, "raised": 0}


def test_a_success_records_its_served_call_and_balances() -> None:
    serial = CA.register_router()
    CA.record_start(serial)
    CA.record_outcome(serial, "success", _served())
    snap = {serial: CA.snapshot()[serial]}
    assert CA.check_balanced(snap) == []
    assert list(snap[serial]["entries"].values()) == [1]


def test_a_success_without_its_served_call_leaves_the_router_unbalanced() -> None:
    serial = CA.register_router()
    CA.record_start(serial)
    CA.record_outcome(serial, "success", None)
    assert CA.check_balanced({serial: CA.snapshot()[serial]})


def test_no_outcome_leaves_the_router_unbalanced() -> None:
    serial = CA.register_router()
    CA.record_start(serial)
    CA.record_outcome(serial, None, None)
    assert CA.check_balanced({serial: CA.snapshot()[serial]})


def test_record_outcome_never_raises_and_leaves_an_unrecordable_call_incomplete() -> None:
    serial = CA.register_router()
    CA.record_start(serial)
    CA.record_outcome(serial, "success", _served(temp=float("nan")))
    CA.record_start(serial)
    CA.record_outcome(serial, "bogus", None)
    assert CA.check_balanced({serial: CA.snapshot()[serial]})


def test_the_delta_keeps_a_router_built_after_the_start_snapshot() -> None:
    start = CA.snapshot()
    serial = CA.register_router()
    CA.record_start(serial)
    CA.record_outcome(serial, "success", _served())
    window, problems = CA.delta(start, CA.snapshot())
    assert problems == []
    assert window[serial]["counts"]["success"] == 1
    assert list(window[serial]["entries"].values()) == [1]


def test_the_delta_drops_calls_made_before_the_start() -> None:
    serial = CA.register_router()
    CA.record_start(serial)
    CA.record_outcome(serial, "success", _served(model="before"))
    start = CA.snapshot()
    window, problems = CA.delta(start, CA.snapshot())
    assert problems == [] and window[serial]["entries"] == {} and window[serial]["counts"]["success"] == 0


def test_the_delta_refuses_a_vanished_router_or_a_falling_count() -> None:
    start = {1: {"entries": {("k",): 2}, "counts": {"started": 2, "success": 2}}}
    assert CA.delta(start, {})[1]
    end = {1: {"entries": {("k",): 1}, "counts": {"started": 1, "success": 1}}}
    assert CA.delta(start, end)[1]


# ── the router records one outcome per dispatch ──────────────────────────────


def _router() -> LLMRouter:
    cfg = dataclasses.replace(
        LLMConfig(),
        enabled=True,
        providers={
            "p": {"type": "openai", "base_url": "http://127.0.0.1:9/v1", "model": "m-1", "allow_local_endpoints": True}
        },
    )
    return LLMRouter(cfg)


def _dispatch(router: LLMRouter, status: str | None, *, providers=("p",), raises: bool = False):
    def fake_try_provider(*, provider_key, temperature, **_kwargs):
        # Stands in for the real attempt, which sets the served call from its own values on success.
        if raises:
            raise RuntimeError("boom")
        if status == "success":
            router._dispatch_served = router._served_call_for(provider_key, "m-1", object(), temperature)
        return ("text" if status == "success" else ""), None, status

    with patch.object(router, "_try_provider", side_effect=fake_try_provider):
        with patch.object(router, "_candidate_providers", return_value=(list(providers), "normal", {})):
            with router._inference_lock:
                return router._complete_text_locked("", "hi", temperature=0.7, max_tokens=1)


@pytest.mark.parametrize(
    ("status", "providers", "outcome"),
    [("success", ("p",), "success"), ("budget_reject", ("p",), "budget_reject"), ("failed", ("p",), "exhausted"),
     (None, (), "exhausted")],
)  # fmt: skip
def test_each_dispatch_records_exactly_one_terminal_outcome(status, providers, outcome) -> None:
    router = _router()
    _dispatch(router, status, providers=providers)
    snap = CA.snapshot()[router.call_serial]
    assert snap["counts"]["started"] == 1 and snap["counts"][outcome] == 1
    assert CA.check_balanced({router.call_serial: snap}) == []


def test_a_raising_dispatch_records_raised() -> None:
    router = _router()
    with pytest.raises(RuntimeError):
        _dispatch(router, None, raises=True)
    assert CA.snapshot()[router.call_serial]["counts"]["raised"] == 1


def test_a_failing_served_call_never_breaks_the_dispatch_and_reads_as_incomplete() -> None:
    router = _router()
    with patch.object(router, "_served_call_for", return_value=None):
        text, _ = _dispatch_real_success(router)
    assert text == "text"
    assert CA.check_balanced({router.call_serial: CA.snapshot()[router.call_serial]})


def _dispatch_real_success(router: LLMRouter):
    # The real _try_provider path down to _invoke_backend, so the attempt sets the served call itself.
    with patch.object(router, "_invoke_backend", return_value=("text", None)):
        with patch.object(router, "_get_backend_for_provider", return_value=object()):
            with patch.object(router, "_candidate_providers", return_value=(["p"], "normal", {})):
                with router._inference_lock:
                    return router._complete_text_locked("", "hi", temperature=0.2, max_tokens=1)


def test_the_real_attempt_records_its_own_model_and_temperature() -> None:
    router = _router()
    text, _ = _dispatch_real_success(router)
    assert text == "text"
    (entry,) = [CA.entry_dict(k, n) for k, n in CA.snapshot()[router.call_serial]["entries"].items()]
    assert entry["model"] == "m-1" and entry["temp"] == 0.2 and entry["backend_kind"] == "other"


def test_a_success_records_the_served_model_and_the_call_temperature() -> None:
    router = _router()
    _dispatch(router, "success")
    (entry,) = [CA.entry_dict(k, n) for k, n in CA.snapshot()[router.call_serial]["entries"].items()]
    assert entry["provider_key"] == "p" and entry["model"] == "m-1" and entry["temp"] == 0.7
    assert entry["backend_kind"] == "other" and entry["seed_honoured"] is False and entry["n"] == 1


# ── scenario inputs ──────────────────────────────────────────────────────────


def test_read_recorded_hashes_the_bytes_it_returns(tmp_path) -> None:
    import hashlib

    before = tmp_path / "before.yaml"
    before.write_text("a: 1\n")
    scenario_inputs.read_recorded(before)
    mark = scenario_inputs.mark()
    f = tmp_path / "s.yaml"
    f.write_text("b: 2\n")
    assert scenario_inputs.read_recorded(f) == b"b: 2\n"
    (item,) = scenario_inputs.since(mark)
    assert item == {"path": os.path.realpath(f), "sha256": hashlib.sha256(b"b: 2\n").hexdigest()}


def _loader_calls():
    from maxim.embodiment.reflex import load_reflex_specs
    from maxim.embodiment.spec import load_spec
    from maxim.simulation.arcs import load_arc_yaml
    from maxim.simulation.dm_schema import load_campaign
    from maxim.simulation.scenario_source import load_scenario

    return [load_spec, load_reflex_specs, load_arc_yaml, load_campaign, load_scenario]


@pytest.mark.parametrize("loader", _loader_calls(), ids=lambda f: f.__name__)
def test_each_scenario_loader_records_its_file(tmp_path, loader) -> None:
    f = tmp_path / "x.yaml"
    f.write_text("name: x\n")
    mark = scenario_inputs.mark()
    try:
        loader(f)
    except Exception:  # noqa: BLE001 — only the recording matters, not whether the toy file validates
        pass
    assert any(i["path"] == os.path.realpath(f) for i in scenario_inputs.since(mark))


def test_the_component_registry_records_its_files_on_every_get_including_cache_hits() -> None:
    from maxim.embodiment.component_registry import ComponentRegistry

    reg = ComponentRegistry()
    ref = sorted(reg._index)[0]
    mark = scenario_inputs.mark()
    reg.get(ref)
    first = scenario_inputs.since(mark)
    assert first
    mark = scenario_inputs.mark()
    reg.get(ref)  # served from the cache
    assert scenario_inputs.since(mark) == first


def test_a_campaign_loaded_before_the_sim_enters_its_scenario_inputs() -> None:
    from maxim.simulation.dm_schema import load_campaign

    path = REPO / "scenarios/campaigns/arena_v1.yaml"
    campaign = load_campaign(path)  # loaded BEFORE the start capture, as the CLI does
    assert any(i["path"] == os.path.realpath(path) for i in campaign.source_inputs)
    token = SP.capture_serving_start(scenario_objects=(campaign,))
    prov = SP.serving_provenance(token, llm_worker=None, aut_worker=None, resume=None, adopted=False, goal="g")
    assert any(i["path"] == os.path.realpath(path) for i in prov["scenario_inputs"])


# ── the report's B0 fields ───────────────────────────────────────────────────


def test_serving_provenance_reports_the_window_and_never_the_aut_model(tmp_path) -> None:
    router = _router()
    worker = SimpleNamespace(router=router)
    token = SP.capture_serving_start()
    _dispatch(router, "success")
    prov = SP.serving_provenance(token, llm_worker=worker, aut_worker=worker, resume=None, adopted=False, goal="g")
    assert prov["aut_model_attribution"] == "not_established"
    assert prov["language_router_serial"] == router.call_serial
    assert prov["sampling_complete"] is True and prov["concurrent_sims"] is False
    (entry,) = prov["llm_calls"][str(router.call_serial)]["entries"]
    assert entry["identity_source"] == "provider_name" and entry["identity"] == "m-1"
    assert prov["session_index"] == 0
    json.dumps(prov)  # report.json must serialize it


def test_session_index_is_unknown_for_an_adopted_agent_or_an_unloaded_resume(tmp_path) -> None:
    assert SP._session_index(None, adopted=True) is None
    assert SP._session_index({"resume_loaded": False}, adopted=False) is None
    (tmp_path / "report.json").write_text(json.dumps({"provenance": {"session_index": 2}}))
    assert SP._session_index({"resume_loaded": True, "state_dir": str(tmp_path)}, adopted=False) == 3
    (tmp_path / "report.json").write_text(json.dumps({"provenance": {}}))
    assert SP._session_index({"resume_loaded": True, "state_dir": str(tmp_path)}, adopted=False) is None


def test_an_incomplete_window_is_stamped_not_hidden() -> None:
    router = _router()
    token = SP.capture_serving_start()
    CA.record_start(router.call_serial)  # a dispatch that never ends
    with patch.object(SP, "ROUTER_IDLE_TIMEOUT_S", 0.01):
        prov = SP.serving_provenance(
            token, llm_worker=SimpleNamespace(router=router), aut_worker=None, resume=None, adopted=False, goal=""
        )
    assert prov["sampling_complete"] is False and prov["sampling_problems"]


def test_the_report_carries_the_schema_and_the_b0_fields() -> None:
    from maxim.simulation.report import PROVENANCE_SCHEMA, capture_start_provenance, run_provenance

    prov = run_provenance(capture_start_provenance(), llm_worker=None, aut_worker=None, resume=None,
                          adopted=False, goal="")  # fmt: skip
    assert prov["provenance_schema"] == PROVENANCE_SCHEMA == 2
    assert prov["aut_model_attribution"] == "not_established"
    assert "serving_token" not in prov
    json.dumps(prov)


def test_the_start_stamp_stays_plain_json_after_calls_were_recorded() -> None:
    from maxim.simulation.report import capture_start_provenance

    _dispatch(_router(), "success")
    json.dumps(capture_start_provenance())


# ── sim registration: overlap, and no leak from a sim that died early ────────


def test_a_new_sim_on_the_same_thread_drops_an_earlier_sim_that_never_reported() -> None:
    SP.capture_serving_start()  # this sim dies before its report
    token = SP.capture_serving_start()
    prov = SP.serving_provenance(token, llm_worker=None, aut_worker=None, resume=None, adopted=False, goal="")
    assert prov["concurrent_sims"] is False


def test_a_sim_on_a_thread_that_ended_is_dropped() -> None:
    import threading

    t = threading.Thread(target=SP.capture_serving_start)
    t.start()
    t.join()
    token = SP.capture_serving_start()
    assert SP.serving_provenance(token, llm_worker=None, aut_worker=None, resume=None, adopted=False,
                                 goal="")["concurrent_sims"] is False  # fmt: skip


def test_a_live_sim_on_another_thread_marks_both() -> None:
    import threading

    started, release = threading.Event(), threading.Event()
    other: dict = {}

    def run() -> None:
        other["token"] = SP.capture_serving_start()
        started.set()
        release.wait(5)
        other["prov"] = SP.serving_provenance(other["token"], llm_worker=None, aut_worker=None, resume=None,
                                              adopted=False, goal="")  # fmt: skip

    t = threading.Thread(target=run)
    t.start()
    started.wait(5)
    token = SP.capture_serving_start()
    mine = SP.serving_provenance(token, llm_worker=None, aut_worker=None, resume=None, adopted=False, goal="")
    release.set()
    t.join()
    assert mine["concurrent_sims"] is True and other["prov"]["concurrent_sims"] is True


# ── the owned server: hashed from its spawn ──────────────────────────────────


class _FakeSpawner:
    def __init__(self, path: Path) -> None:
        self.model_path = str(path)
        self.is_running = True
        self.base_url = "http://127.0.0.1:8100/v1"
        self.kv_cache_type = "q8_0"


def test_a_server_spawned_inside_the_window_is_a_fresh_spawn_hashed_from_its_file(tmp_path) -> None:
    import hashlib

    from maxim.runtime import llm_server

    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"weights")
    token = SP.capture_serving_start()  # no owned server yet
    try:
        llm_server.set_active_spawner(_FakeSpawner(gguf))
        owned, changed = SP._owned_identity(SP._STARTS[token]["spawner"])
    finally:
        llm_server.set_active_spawner(None)
    assert changed is False
    assert owned["identity"] == hashlib.sha256(b"weights").hexdigest() and owned["identity_source"] == "local_file"
    assert owned["kv_quant"] == "q8_0"


def test_a_replaced_server_is_a_model_change(tmp_path) -> None:
    from maxim.runtime import llm_server

    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"w")
    try:
        llm_server.set_active_spawner(_FakeSpawner(gguf))
        token = SP.capture_serving_start()
        llm_server.set_active_spawner(_FakeSpawner(gguf))
        _, changed = SP._owned_identity(SP._STARTS[token]["spawner"])
    finally:
        llm_server.set_active_spawner(None)
    assert changed is True


# ── the gate ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("source", "want"),
    [(None, None), (b"X = 1\n", None), (b"PROVENANCE_SCHEMA = 2\n", 2), (b"PROVENANCE_SCHEMA = 2\nPROVENANCE_SCHEMA = 3\n", 3),
     (b"PROVENANCE_SCHEMA: int = 2\n", R.UNPARSEABLE), (b"PROVENANCE_SCHEMA = X = 2\n", R.UNPARSEABLE),
     (b"PROVENANCE_SCHEMA, Y = 2, 3\n", R.UNPARSEABLE), (b"if True:\n    PROVENANCE_SCHEMA = 2\n", R.UNPARSEABLE),
     (b"PROVENANCE_SCHEMA = 2\nPROVENANCE_SCHEMA += 1\n", R.UNPARSEABLE), (b"PROVENANCE_SCHEMA = '2'\n", R.UNPARSEABLE)],
)  # fmt: skip
def test_the_schema_is_read_only_from_a_plain_int_assignment(source, want) -> None:
    assert R.provenance_schema_in(source) == want


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgSign=false", *args],
        cwd=root, capture_output=True, text=True, check=True,
    ).stdout.strip()  # fmt: skip


def _commit_report(root: Path, body: str) -> str:
    path = root / R.PROVENANCE_SCHEMA_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "c")
    return _git(root, "rev-parse", "HEAD")


@pytest.fixture
def history(tmp_path):
    _git(tmp_path, "init", "-q", "-b", "main")
    pre = _commit_report(tmp_path, "X = 1\n")
    b0 = _commit_report(tmp_path, "PROVENANCE_SCHEMA = 2\n")
    down = _commit_report(tmp_path, "PROVENANCE_SCHEMA = 1\n")
    return SimpleNamespace(root=tmp_path, pre=pre, b0=b0, down=down)


def _good_sim(serial: int = 1) -> dict:
    return {
        "provenance_schema": 2,
        "concurrent_sims": False,
        "sampling_complete": True,
        "model_changed_during_run": False,
        "aut_model_attribution": "not_established",
        "scenario_sha256": "a" * 64,
        "session_index": 0,
        "language_profile": "x",
        "language_router_serial": serial,
        "llm_calls": {
            str(serial): {
                "counts": {"started": 2, "success": 1, "exhausted": 1, "budget_reject": 0, "raised": 0},
                "entries": [{"identity_source": "server_reported", "identity": "qwen", "seed_honoured": False, "n": 1}],
            }
        },
    }


def _judge(history, sim: dict, executed: str) -> list[str]:
    j = R.Judgement(path="t")
    ctx = SimpleNamespace(repo=R.Repo(history.root), base=history.down)
    R.judge_b0_fields(sim, j, ctx, "sim", executed)
    return j.reasons


def test_a_well_formed_b0_record_passes(history) -> None:
    assert _judge(history, _good_sim(), history.b0) == []


def test_a_pre_b0_commit_is_not_judged(history) -> None:
    assert _judge(history, {}, history.pre) == []


def test_a_downgraded_schema_after_the_anchor_fails(history) -> None:
    assert _judge(history, {**_good_sim(), "provenance_schema": 1}, history.down)


def test_a_record_that_drops_its_schema_still_gets_judged(history) -> None:
    sim = _good_sim()
    del sim["provenance_schema"]
    assert _judge(history, sim, history.b0)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s.update(concurrent_sims=True),
        lambda s: s.update(sampling_complete=False),
        lambda s: s.update(model_changed_during_run=True),
        lambda s: s.update(aut_model_attribution="aut"),
        lambda s: s.update(scenario_sha256=None),
        lambda s: s.update(language_router_serial=9),
        lambda s: s.pop("llm_calls"),
        lambda s: s["llm_calls"]["1"]["counts"].update(started=3),
        lambda s: s["llm_calls"]["1"]["entries"][0].update(n=2),
        lambda s: s["llm_calls"]["1"]["entries"][0].update(identity_source="local_file_unhashed"),
        lambda s: s["llm_calls"]["1"]["entries"][0].update(identity_source="local_file", identity="abc"),
        lambda s: s["llm_calls"]["1"]["entries"][0].update(identity=None),
        lambda s: s["llm_calls"]["1"]["entries"][0].update(seed_honoured=None),
    ],
)
def test_every_b0_field_is_checked(history, mutate) -> None:
    sim = _good_sim()
    mutate(sim)
    assert _judge(history, sim, history.b0)


def test_a_roleless_router_is_judged_too(history) -> None:
    sim = _good_sim()
    sim["llm_calls"]["7"] = {"counts": {"started": 1, "success": 0, "exhausted": 0, "budget_reject": 0, "raised": 0},
                             "entries": []}  # fmt: skip
    assert _judge(history, sim, history.b0)


def test_the_sim_evidence_projection_carries_the_b0_fields() -> None:
    import _provenance as P

    report = {"provenance": {**_good_sim(), "language_router_serial": 1}}
    row = P.sim_evidence("/tmp/s", report)
    for key in ("provenance_schema", "llm_calls", "concurrent_sims", "language_router_serial", "scenario_sha256"):
        assert row[key] == report["provenance"][key]


def test_an_in_flight_start_and_an_in_flight_finish_cannot_cancel_out() -> None:
    serial = CA.register_router()
    CA.record_start(serial)  # in flight at the start (A)
    start = CA.snapshot()
    CA.record_outcome(serial, "success", _served())  # A finishes in the window
    CA.record_start(serial)  # B starts and is still in flight at the end
    end = CA.snapshot()
    window, problems = CA.delta(start, end)
    assert problems == [] and CA.check_balanced({serial: window[serial]}) == []  # the window alone balances
    assert CA.window_problems(start, end, window)  # but the snapshots do not


def test_an_idle_router_with_an_older_imbalance_is_not_this_sims() -> None:
    serial = CA.register_router()
    CA.record_start(serial)
    start = CA.snapshot()
    end = CA.snapshot()
    window, _ = CA.delta(start, end)
    assert CA.window_problems(start, end, window) == []


def _full_sim(history, executed: str) -> dict:
    """A sim that passes every pre-B0 check, so only the B0 fields decide."""
    tree = "b" * 64
    return {
        **_good_sim(),
        "finish_reason": "completed",
        "code_changed_during_run": False,
        "working_tree_dirty_src_scripts": False,
        "code_tree_sha256": tree,
        "end_code_tree_sha256": tree,
        "configured_n_ctx": 4096,
        "language_router_n_ctx": 4096,
        "ts": float(_git(history.root, "show", "-s", "--format=%ct", executed)) + 10,
        "executed_git_hash": executed,
    }


def test_judge_sim_fields_runs_the_b0_checks(history) -> None:
    ctx = SimpleNamespace(repo=R.Repo(history.root), base=history.down)
    good = R.Judgement(path="t")
    R.judge_sim_fields(_full_sim(history, history.b0), good, ctx, "sim", None)
    assert good.reasons == []
    bad = R.Judgement(path="t")
    R.judge_sim_fields({**_full_sim(history, history.b0), "concurrent_sims": True}, bad, ctx, "sim", None)
    assert any("concurrent_sims" in r for r in bad.reasons)


def test_a_component_resolved_every_turn_does_not_change_the_scenario_hash() -> None:
    from maxim.embodiment.component_registry import ComponentRegistry

    reg = ComponentRegistry()
    ref = sorted(reg._index)[0]

    def run(gets: int) -> dict:
        token = SP.capture_serving_start()
        for _ in range(gets):
            reg.get(ref)
        return SP.serving_provenance(token, llm_worker=None, aut_worker=None, resume=None, adopted=False, goal="g")

    one, many = run(1), run(25)
    assert one["scenario_sha256"] == many["scenario_sha256"]
    assert len(many["scenario_inputs"]) == len(one["scenario_inputs"])


def test_scenario_content_handed_over_as_turns_enters_the_hash() -> None:
    def run(turns) -> str:
        token = SP.capture_serving_start(scenario_objects=(None, turns))
        return SP.serving_provenance(token, llm_worker=None, aut_worker=None, resume=None, adopted=False,
                                     goal="same")["scenario_sha256"]  # fmt: skip

    assert run([{"text": "a"}]) != run([{"text": "b"}])
    assert run([{"text": "a"}]) == run([{"text": "a"}])


def test_loopback_host_names_match_the_spawners_url() -> None:
    assert SP._root("http://localhost:8100/v1") == SP._root("http://127.0.0.1:8100/v1") == "http://127.0.0.1:8100"


def test_a_process_that_never_started_a_sim_does_not_hash_at_spawn(tmp_path, monkeypatch) -> None:
    from maxim.runtime import llm_server

    monkeypatch.setattr(llm_server, "_digests_requested", False)
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"w")
    try:
        llm_server.set_active_spawner(_FakeSpawner(gguf))
        assert llm_server._spawn_digest is None
    finally:
        llm_server.set_active_spawner(None)


def test_a_server_spawned_before_the_first_sim_is_hashed_from_the_sims_start(tmp_path, monkeypatch) -> None:
    import hashlib

    from maxim.runtime import llm_server

    monkeypatch.setattr(llm_server, "_digests_requested", False)
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"early")
    try:
        llm_server.set_active_spawner(_FakeSpawner(gguf))  # e.g. a CLI warmup, before any sim
        token = SP.capture_serving_start()
        owned, changed = SP._owned_identity(SP._STARTS[token]["spawner"])
    finally:
        llm_server.set_active_spawner(None)
    assert changed is False
    assert owned["identity_source"] == "local_file" and owned["identity"] == hashlib.sha256(b"early").hexdigest()


def test_turn_metadata_with_unsortable_keys_does_not_fail_the_sims_start() -> None:
    token = SP.capture_serving_start(scenario_objects=([{"metadata": {1: "a", "b": 2}}],))
    prov = SP.serving_provenance(token, llm_worker=None, aut_worker=None, resume=None, adopted=False, goal="")
    assert len(prov["scenario_content_sha256"]) == 64


def test_loopback_without_a_port_matches_too() -> None:
    assert SP._root("http://localhost/v1") == "http://127.0.0.1"
