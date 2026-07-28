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

    def test_unregister_is_idempotent(self, sim_logging):
        sink = lambda r: None  # noqa: E731
        register_sim_sink(sink)
        unregister_sim_sink(sink)
        unregister_sim_sink(sink)  # no-op
        sim_log("NAc", "after unregister")  # must not call the removed sink


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
                evt = ws.receive_json()
                assert evt["kind"] == "learn"
                assert evt["tier"] == "clean"
                assert evt["message"] == "reward_bias updated"
                assert evt["data"] == {"delta": 0.2}
                ConsoleEvent.model_validate(evt)

    def test_subscribe_frame_filters_stream(self, sim_logging):
        import threading

        from fastapi.testclient import TestClient

        app = build_app(None)
        with TestClient(app) as client:
            with client.websocket_connect("/ws") as ws:
                ws.send_json({"channels": ["memory"]})
                # Give the receive task a beat to apply the frame, then emit a
                # suppressed kind followed by a passing kind.
                time.sleep(0.2)

                def _emit():
                    sim_log("MOTOR", "should be filtered")
                    sim_log("HIPPOCAMPUS", "should pass")

                t = threading.Thread(target=_emit)
                t.start()
                t.join()
                evt = ws.receive_json()
                assert evt["kind"] == "hippocampus", f"MOTOR leaked through the filter: {evt}"

    def test_openapi_carries_subscribe_frame(self):
        schema = build_app(None).openapi()
        assert "SubscribeFrame" in schema["components"]["schemas"]
        assert "/api/events/subscribe-frame" in schema["paths"]
