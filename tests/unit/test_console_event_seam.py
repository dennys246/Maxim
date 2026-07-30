"""EVENT seam regression guards (reachy_app_maxim_seams.md § EVENT).

Pins the five spec guards:
  1. envelope round-trip — sim_log emission → /ws client with correct kind
     (lowercased), tier (incl. unknown-subsystem→bio), run_id, epoch ts,
     monotonic seq;
  2. backpressure — full queue drops OLDEST, keeps newest, surfaces a
     "dropped" count; the publishing thread never blocks;
  3. filter — SubscribeFrame channels=["memory"] passes HIPPOCAMPUS/NAc/SCN/ATL
     kinds and suppresses others;
  4. a raising sink never propagates into sim_log's caller;
  5. kind="deliberation" events carry the reasoning text that previously only
     reached display.set_thinking.

Skips cleanly when the `console` extra is absent (groups 4–5 are pure
sim_logger and run regardless).
"""

from __future__ import annotations

import asyncio
import time

import pytest

from maxim.simulation.sim_logger import (
    disable_sim_logging,
    enable_sim_logging,
    expand_channels,
    get_sim_records,
    register_sim_sink,
    sim_deliberation_end,
    sim_deliberation_update,
    sim_log,
    subsystem_wire_tier,
    unregister_sim_sink,
)


@pytest.fixture()
def sim_logging():
    """Enable sim logging for the test, always disable after."""
    enable_sim_logging(use_color=False)
    try:
        yield
    finally:
        disable_sim_logging()


# ── sim_logger-side guards (no console extra needed) ─────────────────────────


class TestSimSinks:
    def test_sink_receives_record_shape(self, sim_logging):
        seen: list[dict] = []
        register_sim_sink(seen.append)
        try:
            sim_log("HIPPOCAMPUS", "stored episode", {"episode_id": "e1"}, agent_id="a1")
        finally:
            unregister_sim_sink(seen.append)
        # last record is ours (enable_sim_logging emits a PIPELINE line first)
        rec = seen[-1]
        assert rec["subsystem"] == "HIPPOCAMPUS"
        assert rec["message"] == "stored episode"
        assert rec["data"] == {"episode_id": "e1"}
        assert rec["agent_id"] == "a1"
        assert isinstance(rec["t"], float)  # sim-elapsed, NOT epoch

    def test_sink_fires_regardless_of_display_tier(self, sim_logging):
        # Same rule as JSONL persistence: sinks sit BEFORE the tier gate.
        # Default display tier is CLEAN; a DEBUG-tier subsystem must still
        # reach the sink.
        seen: list[dict] = []
        sink = seen.append
        register_sim_sink(sink)
        try:
            sim_log("PIPELINE", "debug-tier event")
        finally:
            unregister_sim_sink(sink)
        assert any(r["subsystem"] == "PIPELINE" and r["message"] == "debug-tier event" for r in seen)

    def test_raising_sink_never_propagates(self, sim_logging, caplog):
        # Guard 4: a broken sink must not take down the agent loop — and must
        # not fail silently either (warn once per sink).
        def bad_sink(record):
            raise RuntimeError("boom")

        good: list[dict] = []
        register_sim_sink(bad_sink)
        register_sim_sink(good.append)
        try:
            sim_log("NAc", "first")
            sim_log("NAc", "second")
        finally:
            unregister_sim_sink(bad_sink)
            unregister_sim_sink(good.append)
        # No exception reached us, later sinks still ran, warned exactly once.
        assert sum(1 for r in good if r["subsystem"] == "NAc") == 2
        warnings = [r for r in caplog.records if "sim sink" in r.getMessage()]
        assert len(warnings) == 1

    def test_sinks_survive_disable_sim_logging(self):
        # CROSS-CONFIRMED BLOCKER: start_simulation_mode calls
        # disable_sim_logging() at the end of EVERY campaign. When sink
        # dispatch was gated on _sim_active, one finished adventure
        # permanently silenced the console's /ws stream — and talk's reply
        # travels ONLY on the wire, so it was unrecoverable and silent.
        seen: list[dict] = []
        register_sim_sink(seen.append)
        try:
            enable_sim_logging(use_color=False)
            sim_log("USER", "before")
            disable_sim_logging()  # what every campaign end does
            sim_log("RESPONSE", "after")
        finally:
            unregister_sim_sink(seen.append)
            disable_sim_logging()
        assert [r["subsystem"] for r in seen if r["subsystem"] in ("USER", "RESPONSE")] == ["USER", "RESPONSE"]

    def test_sink_fires_with_no_sim_ever_enabled(self):
        # The console registers a sink and never calls enable_sim_logging.
        seen: list[dict] = []
        register_sim_sink(seen.append)
        try:
            sim_log("LEARN", "no sim active")
        finally:
            unregister_sim_sink(seen.append)
        assert [r["message"] for r in seen] == ["no sim active"]
        # …and sim-elapsed stays sane rather than being an epoch timestamp.
        assert 0.0 <= seen[0]["t"] < 10_000_000.0

    def test_unregister_is_idempotent(self, sim_logging):
        sink = lambda r: None  # noqa: E731
        register_sim_sink(sink)
        unregister_sim_sink(sink)
        unregister_sim_sink(sink)  # no-op
        sim_log("NAc", "after unregister")  # must not call the removed sink

    def test_unregister_by_equal_bound_method_clears_warned_entry(self, sim_logging):
        # Cross-confirmed review fold: unregistering via a FRESH bound-method
        # object (equal, not identical — the console's unregister_sim_sink(
        # hub.sink) shape) must discard the STORED object's warned-set entry;
        # discarding id(argument) was a no-op → id-reuse could suppress a
        # future sink's only warning.
        from maxim.simulation import sim_logger as sl

        class Hub:
            def sink(self, record):
                raise RuntimeError("boom")

        hub = Hub()
        register_sim_sink(hub.sink)  # bound-method object A
        stored = sl._sim_sinks[-1]
        sim_log("NAc", "trigger warn")  # warns once; id(stored) enters the set
        assert id(stored) in sl._warned_sinks
        unregister_sim_sink(hub.sink)  # DIFFERENT object B, == A
        assert stored not in sl._sim_sinks
        assert id(stored) not in sl._warned_sinks


class TestDeliberationRecords:
    def test_deliberation_update_emits_record_with_text(self, sim_logging):
        # Guard 5: previously set_thinking-only — no record, JSONL + /ws blind.
        sim_deliberation_update("I should examine the sword first.", cycle=1, max_cycles=3, salience=0.7)
        recs = [r for r in get_sim_records() if r["subsystem"] == "DELIBERATION"]
        assert recs, "sim_deliberation_update emitted no DELIBERATION record"
        assert recs[-1]["data"]["text"] == "I should examine the sword first."
        assert recs[-1]["data"]["cycle"] == 1
        assert recs[-1]["data"]["completed"] is False

    def test_deliberation_end_emits_completed_record(self, sim_logging):
        sim_deliberation_end(cycle=2, max_cycles=3, summary="Chose to examine.")
        recs = [r for r in get_sim_records() if r["subsystem"] == "DELIBERATION"]
        assert recs[-1]["data"]["completed"] is True
        assert recs[-1]["data"]["text"] == "Chose to examine."


class TestWireHelpers:
    def test_subsystem_wire_tier_known_and_unknown(self):
        assert subsystem_wire_tier("SCENE") == "clean"
        assert subsystem_wire_tier("HIPPOCAMPUS") == "bio"
        assert subsystem_wire_tier("PIPELINE") == "debug"
        # Unknown → bio: the opt-out-not-opt-in default, kept on the wire.
        assert subsystem_wire_tier("SOME_FUTURE_SUBSYSTEM") == "bio"

    def test_expand_channels_matches_terminal_map(self):
        assert expand_channels(["memory"]) == {"HIPPOCAMPUS", "NAc", "SCN", "ATL"}
        # Unknown names pass through as raw subsystem names, uppercased.
        assert expand_channels(["custom_thing"]) == {"CUSTOM_THING"}


class TestDisplaySuggestionEvents:
    def test_escalate_and_revert_emit_display_records(self, sim_logging):
        from maxim.simulation.sim_logger import (
            DisplayTier,
            agent_escalate_display,
            revert_display_to_floor,
            set_display_tier,
        )

        # Pin the floor: earlier tests in the suite may leave tier/floor
        # non-default (module-level globals), which flips escalate/revert
        # outcomes. reset_sim_display_state() in teardown restores defaults.
        set_display_tier(DisplayTier.CLEAN)
        try:
            assert agent_escalate_display(DisplayTier.BIO, reason="surfacing learning") is True
            revert_display_to_floor()
            recs = [r for r in get_sim_records() if r["subsystem"] == "DISPLAY"]
            assert [r["data"]["action"] for r in recs] == ["escalate", "revert"]
            assert recs[0]["data"]["tier"] == "bio"
            assert recs[0]["data"]["reason"] == "surfacing learning"
        finally:
            from maxim.simulation.sim_logger import reset_sim_display_state

            reset_sim_display_state()


# ── console-side guards (need the console extra) ─────────────────────────────

fastapi = pytest.importorskip("fastapi", reason="requires the `console` extra (fastapi/uvicorn)")

from maxim.console.schemas import ConsoleEvent, SubscribeFrame  # noqa: E402
from maxim.console.server import _EventHub, _WsConn, build_app  # noqa: E402


def _first_stream_event(ws):
    """Drain the identity hello frame and return the first real event.

    /ws now opens with kind="identity" so a client knows which backend it is
    attached to before interpreting anything else.
    """
    evt = ws.receive_json()
    if evt.get("kind") == "identity":
        evt = ws.receive_json()
    return evt


def _drain(conn: _WsConn) -> list[dict]:
    out = []
    while True:
        try:
            out.append(conn.queue.get_nowait())
        except asyncio.QueueEmpty:
            break
    return out


class TestEnvelope:
    def test_sink_builds_v2_envelope(self):
        # Guard 1 (unit level): record → envelope field mapping.
        hub = _EventHub()
        conn = _WsConn()
        loop = asyncio.new_event_loop()
        try:
            hub.attach(loop)
            hub.add_conn(conn)
            hub.set_run("run_42")
            before = time.time()
            hub.sink(
                {
                    "t": 12.5,
                    "subsystem": "HIPPOCAMPUS",
                    "message": "stored",
                    "data": {"episode_id": "e1"},
                    "agent_id": "a1",
                    "agent": "Roy",
                }
            )
            hub.sink({"t": 13.0, "subsystem": "SOME_FUTURE_SUBSYSTEM", "message": "?", "data": {}})
            loop.run_until_complete(asyncio.sleep(0))  # run the call_soon_threadsafe callbacks
            events = _drain(conn)
        finally:
            loop.close()
        assert len(events) == 2
        first, second = events
        assert first["kind"] == "hippocampus"  # lowercased subsystem
        assert first["tier"] == "bio"
        assert first["run_id"] == "run_42"
        assert first["elapsed_s"] == 12.5  # sim-elapsed travels separately…
        assert first["ts"] >= before  # …ts is epoch, stamped at bridge time
        assert first["agent_id"] == "a1" and first["agent"] == "Roy"
        assert first["message"] == "stored"
        assert first["data"] == {"episode_id": "e1"}
        assert second["tier"] == "bio"  # unknown subsystem → bio on the wire
        # Monotonic per-connection seq.
        assert [e["seq"] for e in events] == [0, 1]
        # Every event validates against the wire model.
        for e in events:
            ConsoleEvent.model_validate(e)

    def test_sink_without_loop_or_conns_is_noop(self):
        hub = _EventHub()
        hub.sink({"t": 0.0, "subsystem": "NAc", "message": "x", "data": {}})  # no loop: no-op, no raise
        loop = asyncio.new_event_loop()
        try:
            hub.attach(loop)
            hub.sink({"t": 0.0, "subsystem": "NAc", "message": "x", "data": {}})  # no conns: no-op
        finally:
            loop.close()


class TestBackpressure:
    def test_drop_oldest_keeps_newest_and_counts(self):
        # Guard 2: full queue drops the OLDEST; seq gap marks where.
        conn = _WsConn()
        maxsize = conn.queue.maxsize
        for i in range(maxsize + 5):
            conn.enqueue({"kind": "nac", "i": i})
        assert conn.dropped == 5
        events = _drain(conn)
        assert len(events) == maxsize
        assert events[0]["i"] == 5  # oldest five gone
        assert events[-1]["i"] == maxsize + 4  # newest kept
        # seq was assigned pre-drop: the surviving head starts at 5 → gap 0-4.
        assert events[0]["seq"] == 5

    def test_publishing_thread_never_blocks(self):
        # enqueue is put_nowait/get_nowait only — bound the wall-clock to prove
        # no hidden blocking put.
        conn = _WsConn()
        start = time.monotonic()
        for i in range(conn.queue.maxsize * 3):
            conn.enqueue({"kind": "nac", "i": i})
        assert time.monotonic() - start < 2.0


class TestSubscribeFilter:
    def _conn_with_frame(self, **kwargs) -> _WsConn:
        conn = _WsConn()
        conn.apply_frame(SubscribeFrame(**kwargs))
        return conn

    def test_channels_memory_passes_memory_suppresses_others(self):
        # Guard 3: the terminal _CHANNEL_MAP lifted to the socket.
        conn = self._conn_with_frame(channels=["memory"])
        for subsystem in ("HIPPOCAMPUS", "NAc", "SCN", "ATL"):
            assert conn.matches(subsystem.lower(), "bio", subsystem) is True
        assert conn.matches("motor", "bio", "MOTOR") is False
        assert conn.matches("scene", "clean", "SCENE") is False

    def test_tier_clean_suppresses_bio_and_debug(self):
        conn = self._conn_with_frame(tier="clean")
        assert conn.matches("scene", "clean", "SCENE") is True
        assert conn.matches("nac", "bio", "NAc") is False
        assert conn.matches("pipeline", "debug", "PIPELINE") is False

    def test_kinds_and_channels_union_within_axis(self):
        conn = self._conn_with_frame(channels=["memory"], kinds=["motor"])
        assert conn.matches("motor", "bio", "MOTOR") is True  # via kinds
        assert conn.matches("nac", "bio", "NAc") is True  # via channels
        assert conn.matches("scene", "clean", "SCENE") is False

    def test_mixed_case_subsystem_reachable_via_raw_channel_name(self):
        # Cross-confirmed review fold: "NAc" is the one mixed-case canonical
        # subsystem; pre-fold, channels=["nac"]→"NAC" never matched it —
        # silent empty stream on that axis. Matching is now case-insensitive.
        for spelling in ("nac", "NAc", "NAC"):
            conn = self._conn_with_frame(channels=[spelling])
            assert conn.matches("nac", "bio", "NAc") is True, spelling

    def test_meta_kinds_bypass_filters(self):
        conn = self._conn_with_frame(channels=["memory"], tier="clean")
        for kind in ("heartbeat", "run", "dropped", "display"):
            assert conn.matches(kind, "clean", "") is True

    def test_new_frame_replaces_filter_and_all_none_resets(self):
        conn = self._conn_with_frame(channels=["memory"])
        assert conn.matches("motor", "bio", "MOTOR") is False
        conn.apply_frame(SubscribeFrame())  # all-None → everything
        assert conn.matches("motor", "bio", "MOTOR") is True


class TestWsEndToEnd:
    def test_sim_log_reaches_ws_client(self, sim_logging):
        # Guard 1 (end-to-end): sim_log on a worker thread → /ws frame.
        import threading

        from fastapi.testclient import TestClient

        app = build_app(None)
        with TestClient(app) as client:  # context manager runs the lifespan (sink registration)
            with client.websocket_connect("/ws") as ws:
                # Emit from a different thread — the real topology (sim thread
                # publishes, event loop fans out).
                t = threading.Thread(
                    target=sim_log, args=("LEARN", "reward_bias updated"), kwargs={"data": {"delta": 0.2}}
                )
                t.start()
                t.join()
                evt = _first_stream_event(ws)
                assert evt["kind"] == "learn"
                assert evt["tier"] == "clean"
                assert evt["message"] == "reward_bias updated"
                assert evt["data"] == {"delta": 0.2}
                ConsoleEvent.model_validate(evt)

    def test_subscribe_frame_filters_stream(self, sim_logging):
        # Deterministic (review fold — the original 0.2s sleep raced the recv
        # task applying the frame): emit MOTOR+HIPPOCAMPUS rounds until a
        # round's hippocampus arrives WITHOUT its paired motor — that round
        # proves the filter is live. Bounded, fails loudly if never applied.
        import threading

        from fastapi.testclient import TestClient

        app = build_app(None)
        with TestClient(app) as client:
            with client.websocket_connect("/ws") as ws:
                assert ws.receive_json()["kind"] == "identity"  # hello frame
                ws.send_json({"channels": ["memory"]})
                motor_seen: set[int] = set()
                for i in range(50):
                    t = threading.Thread(
                        target=lambda i=i: (
                            sim_log("MOTOR", f"m{i}", {"i": i}),
                            sim_log("HIPPOCAMPUS", f"h{i}", {"i": i}),
                        )
                    )
                    t.start()
                    t.join()
                    # Drain until this round's hippocampus event arrives.
                    while True:
                        evt = ws.receive_json()
                        if evt["kind"] == "motor":
                            motor_seen.add(evt["data"]["i"])
                        elif evt["kind"] == "hippocampus" and evt["data"]["i"] == i:
                            break
                    if i not in motor_seen:
                        return  # filter provably applied: hippo passed, motor suppressed
                pytest.fail("SubscribeFrame never took effect after 50 rounds")

    def test_malformed_frame_keeps_connection_alive(self, sim_logging):
        # Cross-confirmed review fold: non-JSON text (and binary frames)
        # previously killed the recv task and escaped the endpoint as an
        # unhandled ASGI exception. Now: ignored, filter unchanged.
        import threading

        from fastapi.testclient import TestClient

        app = build_app(None)
        with TestClient(app) as client:
            with client.websocket_connect("/ws") as ws:
                ws.send_text("not json {{{")
                time.sleep(0.1)  # let the recv task process (and survive) it
                t = threading.Thread(target=sim_log, args=("LEARN", "still alive"))
                t.start()
                t.join()
                evt = _first_stream_event(ws)
                assert evt["kind"] == "learn"

    def test_openapi_carries_subscribe_frame(self):
        schema = build_app(None).openapi()
        assert "SubscribeFrame" in schema["components"]["schemas"]
        assert "/api/events/subscribe-frame" in schema["paths"]
        # Review fold: tier/seq/message are REQUIRED so the generated TS type
        # is non-optional (the server always populates them).
        required = set(schema["components"]["schemas"]["ConsoleEvent"]["required"])
        assert {"kind", "tier", "seq", "ts", "message"} <= required


class TestRunLifecycleEvents:
    def test_run_ended_payload_derives_report_path(self):
        # Review fold (BLOCKING): SimulationResult has session_id/session_dir,
        # NOT report_path — the pre-fold getattr(result, "report_path") was
        # structurally always None on the wire.
        from maxim.console.server import _event_hub, _run_campaign_thread
        from maxim.simulation.sim_types import SimulationResult

        result = SimulationResult(
            goal="g",
            persona="p",
            turns=1,
            total_actions=1,
            blocked_actions=0,
            duration_s=0.1,
            finish_reason="completed",
            session_id="sim_abc",
            session_dir="/tmp/sessions/sim_abc",
        )

        class FakeHandle:
            def play_campaign(self, path):
                return result

        conn = _WsConn()
        loop = asyncio.new_event_loop()
        try:
            _event_hub.attach(loop)
            _event_hub.add_conn(conn)
            _run_campaign_thread(FakeHandle(), "camp.yaml", "run_1")
            loop.run_until_complete(asyncio.sleep(0))
            events = _drain(conn)
        finally:
            _event_hub.remove_conn(conn)
            _event_hub.detach()
            loop.close()
        assert [e["data"]["status"] for e in events] == ["started", "ended"]
        ended = events[-1]["data"]
        assert ended["sim_session_id"] == "sim_abc"
        assert ended["report_path"].endswith("sim_abc/report.json")
        # Started is stamped with the console run_id; ended cleared it after.
        assert events[0]["run_id"] == "run_1"

    def test_run_ended_empty_result_fields_coerce_to_none(self):
        from maxim.console.server import _event_hub, _run_campaign_thread
        from maxim.simulation.sim_types import SimulationResult

        result = SimulationResult(goal="g", persona="p", turns=0, total_actions=0, blocked_actions=0, duration_s=0.0)

        class FakeHandle:
            def play_campaign(self, path):
                return result

        conn = _WsConn()
        loop = asyncio.new_event_loop()
        try:
            _event_hub.attach(loop)
            _event_hub.add_conn(conn)
            _run_campaign_thread(FakeHandle(), "camp.yaml", "run_2")
            loop.run_until_complete(asyncio.sleep(0))
            events = _drain(conn)
        finally:
            _event_hub.remove_conn(conn)
            _event_hub.detach()
            loop.close()
        ended = events[-1]["data"]
        # Dataclass defaults are "" — the wire must carry None, not "".
        assert ended["sim_session_id"] is None
        assert ended["report_path"] is None


class TestNarrativeReachesTheStream:
    """Campaign PROSE — the thing an Adventure viewer is there to read.

    display_scene/turn/summary were terminal-only: they called _emit directly
    and produced no record, so a web client saw the narrative only as the
    200-char BIO-tier PERCEPT summary. They now also emit CLEAN-tier records.
    """

    def test_scene_emits_a_clean_tier_record_with_full_text(self, sim_logging):
        from maxim.simulation.sim_logger import display_scene, subsystem_wire_tier

        prose = "The cavern narrows. " * 40  # well past the 200-char percept cap
        display_scene(prose)
        recs = [r for r in get_sim_records() if r["subsystem"] == "SCENE"]
        assert len(recs) == 1
        assert recs[0]["data"]["text"] == prose, "prose must not be truncated on the record"
        assert subsystem_wire_tier("SCENE") == "clean"

    def test_turn_and_summary_emit_records(self, sim_logging):
        from maxim.simulation.sim_logger import display_summary, display_turn, subsystem_wire_tier

        display_turn(7)
        display_summary(["Score: 3", "Done."])
        turns = [r for r in get_sim_records() if r["subsystem"] == "TURN"]
        summaries = [r for r in get_sim_records() if r["subsystem"] == "SUMMARY"]
        assert turns[-1]["data"]["turn"] == 7
        assert summaries[-1]["data"]["lines"] == ["Score: 3", "Done."]
        assert subsystem_wire_tier("TURN") == "clean"
        assert subsystem_wire_tier("SUMMARY") == "clean"

    def test_narrative_reaches_a_registered_sink(self, sim_logging):
        # The /ws bridge is a sink — this is the actual delivery path.
        from maxim.simulation.sim_logger import display_scene

        seen: list[dict] = []
        register_sim_sink(seen.append)
        try:
            display_scene("A door creaks open.")
        finally:
            unregister_sim_sink(seen.append)
        assert [r["data"]["text"] for r in seen if r["subsystem"] == "SCENE"] == ["A door creaks open."]

    def test_terminal_is_not_double_printed(self, sim_logging, capsys):
        # display_* IS the terminal renderer; sim_log must not print too.
        # This is the same duplication class as the percept double-log.
        from maxim.simulation.sim_logger import display_scene

        capsys.readouterr()
        display_scene("Only once please.")
        assert capsys.readouterr().out.count("Only once please.") == 1


class TestGateRejectionTellsTheTruth:
    """`score=0.00 < 0.00` was a FABRICATED log line.

    agent_loop computed a real GateDecision (score, threshold_used, reason)
    then discarded it and called sim_pre_deliberation with hardcoded 0.0/0.0.
    Every rejection therefore rendered as a threshold comparison — including
    refractory, energy-exhausted and empty-working-memory, which are not
    threshold comparisons at all. A live console session read that number and
    concluded the gate was scoring zero; it had never been measured.
    """

    def test_reason_is_shown_instead_of_a_fake_comparison(self, sim_logging):
        from maxim.simulation.sim_logger import sim_pre_deliberation

        sim_pre_deliberation(
            gate_passed=False, score=0.0, threshold=0.0, enrichment_sections=0, reason="empty working memory"
        )
        rec = [r for r in get_sim_records() if r["subsystem"] == "THOUGHT"][-1]
        assert "empty working memory" in rec["message"]
        assert "0.00 < 0.00" not in rec["message"]
        assert rec["data"]["reason"] == "empty working memory"

    def test_falls_back_to_the_comparison_when_no_reason(self, sim_logging):
        from maxim.simulation.sim_logger import sim_pre_deliberation

        sim_pre_deliberation(gate_passed=False, score=0.3, threshold=0.4, enrichment_sections=0)
        rec = [r for r in get_sim_records() if r["subsystem"] == "THOUGHT"][-1]
        assert "0.30 < 0.40" in rec["message"]

    def test_gate_rejection_reports_measured_values_not_zeros(self, sim_logging):
        # BEHAVIOURAL, not source-text: the previous version asserted a string
        # was absent from the module source, which passes on any reformat
        # (score=0.0,\n threshold=0.0) and cannot tell a wrong-but-present
        # number from a right one.
        from maxim.simulation.sim_logger import sim_pre_deliberation

        sim_pre_deliberation(gate_passed=False, score=0.42, threshold=0.55, enrichment_sections=0)
        rec = [r for r in get_sim_records() if r["subsystem"] == "THOUGHT"][-1]
        assert rec["data"]["score"] == 0.42 and rec["data"]["threshold"] == 0.55
        assert "0.42 < 0.55" in rec["message"]

    def test_pass_branch_does_not_render_as_rejected(self, sim_logging):
        # The enrichment-empty branch reused the PASS reason, printing
        # "gate rejected (deliberation approved)".
        from maxim.simulation.sim_logger import sim_pre_deliberation

        sim_pre_deliberation(
            gate_passed=False,
            score=0.9,
            threshold=0.4,
            enrichment_sections=0,
            reason="enrichment produced no sections",
        )
        msg = [r for r in get_sim_records() if r["subsystem"] == "THOUGHT"][-1]["message"]
        assert "deliberation approved" not in msg
        assert "enrichment produced no sections" in msg


class TestDmNarrationIsRecordedInAutomatedMode:
    def test_display_scene_emits_a_record_for_automated_narration(self, sim_logging):
        # BEHAVIOURAL: the old version counted a substring in the source, which
        # would pass under `if False:` and SKIPPED (rather than failed) if the
        # class were renamed. What actually matters is that narration reaches
        # the record stream — display_scene is the shared path both DM branches
        # now use.
        from maxim.simulation.sim_logger import display_scene

        display_scene("The cavern mouth yawns open.")
        recs = [r for r in get_sim_records() if r["subsystem"] == "SCENE"]
        assert recs[-1]["data"]["text"] == "The cavern mouth yawns open."
