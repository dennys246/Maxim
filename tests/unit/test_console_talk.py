"""HANDLE talk-mode guards (the live-loop conversational surface).

Talk is the console's PRIMARY interaction — the difference between a console
you observe and one you use. These pin the contract without needing an LLM:

  * the reply reaches the ``/ws`` stream as CLEAN-tier records (``USER`` then
    ``RESPONSE``) — the web chat renders from the stream, not the HTTP body;
  * talk and an adventure never drive the same bio-stack concurrently;
  * ``RespondTool``/``SpeakTool`` actually exist on the handle's registry
    (without a ResponseOutput the agent has NO way to reply and talk would
    silently always come back empty);
  * a turn that produces no reply still says so on the wire.
"""

from __future__ import annotations

import threading

import pytest

from maxim.simulation.sim_logger import (
    disable_sim_logging,
    enable_sim_logging,
    get_sim_records,
    reset_sim_display_state,
)


@pytest.fixture()
def sim_logging():
    enable_sim_logging(use_color=False)
    try:
        yield
    finally:
        disable_sim_logging()
        reset_sim_display_state()


def _bare_handle(bridge=None):
    """A MaximHandle with no agent construction (no LLM, no bio-stack)."""
    from maxim.console.handle import MaximHandle

    h = MaximHandle.__new__(MaximHandle)
    h.agent_id = "test_agent"
    h._stopped = False
    h._campaign_lock = threading.Lock()
    h._talk_lock = threading.Lock()
    h._talk_bridge = bridge
    h._talk_thread = None
    h._talk_stop = None
    h._talk_worker = None
    return h


class _FakeBridge:
    def __init__(self, response="Hello back."):
        self._response = response
        self.sent: list[str] = []
        self.finished = False

    def send_and_wait(self, text, **kw):
        self.sent.append(text)
        return {"turn": 1, "response": self._response, "actions": [], "timed_out": False, "duration_ms": 1.0}

    def finish(self):
        self.finished = True


class TestTalkStreamContract:
    def test_utterance_and_reply_reach_the_stream(self, sim_logging, monkeypatch):
        bridge = _FakeBridge("I remember the cavern.")
        h = _bare_handle(bridge)
        monkeypatch.setattr(type(h), "_ensure_talk_loop", lambda self: bridge)

        result = h.talk("what do you remember?")

        assert result["response"] == "I remember the cavern."
        assert bridge.sent == ["what do you remember?"]
        kinds = [
            (r["subsystem"], r["data"].get("text")) for r in get_sim_records() if r["subsystem"] in ("USER", "RESPONSE")
        ]
        assert kinds == [("USER", "what do you remember?"), ("RESPONSE", "I remember the cavern.")]

    def test_user_and_response_are_clean_tier(self):
        # The web chat renders CLEAN-tier events as conversation lines; if
        # either dropped to bio/debug it would vanish from a filtered client.
        from maxim.simulation.sim_logger import subsystem_wire_tier

        assert subsystem_wire_tier("USER") == "clean"
        assert subsystem_wire_tier("RESPONSE") == "clean"

    def test_silent_turn_still_reports_on_the_wire(self, sim_logging, monkeypatch):
        bridge = _FakeBridge(response=None)
        h = _bare_handle(bridge)
        monkeypatch.setattr(type(h), "_ensure_talk_loop", lambda self: bridge)
        h.talk("hello?")
        responses = [r for r in get_sim_records() if r["subsystem"] == "RESPONSE"]
        assert len(responses) == 1
        assert responses[0]["data"]["text"] is None  # explicit "no reply", not silence

    def test_empty_utterance_rejected(self):
        h = _bare_handle()
        with pytest.raises(ValueError, match="non-empty"):
            h.talk("   ")

    def test_talk_refused_while_a_campaign_holds_the_lock(self):
        h = _bare_handle()
        h._campaign_lock.acquire()
        try:
            with pytest.raises(RuntimeError, match="adventure is running"):
                h.talk("hi")
        finally:
            h._campaign_lock.release()

    def test_talk_refused_after_stop(self):
        h = _bare_handle()
        h._stopped = True
        with pytest.raises(RuntimeError, match="stopped"):
            h.talk("hi")


class TestReplyExtraction:
    """The production RespondTool returns a delivery RECEIPT, not the words."""

    def _action(self, tool_name, args, output=None):
        from maxim.simulation.sinks import ActionRecord

        return ActionRecord(timestamp=0.0, tool_name=tool_name, tool_args=args, result_output=output)

    def test_prefers_tool_args_message_over_receipt_output(self):
        from maxim.console.handle import _extract_reply

        turn = {
            # What the live loop actually produces: output is the receipt.
            "actions": [self._action("respond", {"message": "Hello there."}, {"delivered": True, "mode": "cli"})],
            "response": "{'delivered': True, 'mode': 'cli'}",
        }
        assert _extract_reply(turn) == "Hello there."

    def test_receipt_dict_never_leaks_as_the_reply(self):
        from maxim.console.handle import _extract_reply

        # No tool_args to recover from → the receipt must NOT be shown to the
        # user as if it were speech.
        turn = {"actions": [self._action("respond", {}, {"delivered": True})], "response": {"delivered": True}}
        assert _extract_reply(turn) is None

    def test_falls_back_to_bridge_response_for_sim_respond_tool(self):
        from maxim.console.handle import _extract_reply

        # SimRespondTool's output IS the text — that path must keep working.
        assert _extract_reply({"actions": [], "response": "spoken text"}) == "spoken text"

    def test_joins_multiple_utterances_in_one_turn(self):
        from maxim.console.handle import _extract_reply

        turn = {
            "actions": [
                self._action("respond", {"message": "First."}),
                self._action("bash", {"command": "ls"}),  # non-speech ignored
                self._action("speak", {"message": "Second."}),
            ]
        }
        assert _extract_reply(turn) == "First.\nSecond."


class TestTalkPermissions:
    """CROSS-CONFIRMED BLOCKER: SupervisionPolicy.allowed_tools is NEVER
    consulted at AUTONOMOUS ("only safety constraints apply"), so the original
    allow-list was a silent no-op and bash/write_file — scoped to the SERVER'S
    CWD — were reachable from small talk. The restriction must live in
    SafetyConstraints, which IS enforced at every level."""

    def _controller(self, registry_tools):
        from maxim.agents.autonomy import (
            AutonomyController,
            AutonomyLevel,
            SafetyConstraints,
            SupervisionPolicy,
        )

        from maxim.console.handle import TALK_CONVERSATIONAL_TOOLS

        conversational = set(TALK_CONVERSATIONAL_TOOLS)
        mutating = {t for t in registry_tools if t not in conversational}
        return AutonomyController(
            initial_level=AutonomyLevel.AUTONOMOUS,
            supervision_policy=SupervisionPolicy(allowed_tools=set(conversational)),
            safety_constraints=SafetyConstraints(
                forbidden_tools=frozenset(mutating | set(SafetyConstraints().forbidden_tools))
            ),
        )

    @pytest.mark.parametrize(
        "tool",
        [
            "bash",
            "write_file",
            "edit_file",
            "git_commit",
            # In AutonomyController.ALWAYS_ALLOWED_TOOLS: the shortcut used to
            # run BEFORE SafetyConstraints, so these executed from Talk despite
            # being derived-forbidden (P4d).
            "search_code",
            "git_diff",
        ],
    )
    def test_mutating_tools_are_refused_at_autonomous(self, tool):
        ctrl = self._controller({"respond", "bash", "write_file", "edit_file", "git_commit", "search_code", "git_diff"})
        allowed, reason = ctrl.can_execute_action({"tool_name": tool})
        assert allowed is False, f"{tool} must not be reachable from a talk turn"
        assert reason

    @pytest.mark.parametrize("tool", ["glob", "read_file", "list_directory"])
    def test_always_allowed_never_beats_an_explicit_forbid(self, tool):
        # Talk keeps these conversational, so its derived set does not forbid
        # them — but a controller that DOES forbid one must win over the
        # always-allowed shortcut. Pins the general rule the P4d fix restored.
        from maxim.agents.autonomy import AutonomyController, AutonomyLevel, SafetyConstraints

        ctrl = AutonomyController(
            initial_level=AutonomyLevel.AUTONOMOUS,
            safety_constraints=SafetyConstraints(forbidden_tools=frozenset({tool})),
        )
        assert tool in AutonomyController.ALWAYS_ALLOWED_TOOLS
        allowed, reason = ctrl.can_execute_action({"tool_name": tool})
        assert allowed is False
        assert "forbidden" in (reason or "")

    @pytest.mark.parametrize("tool", ["respond", "speak", "read_file"])
    def test_conversational_tools_still_allowed(self, tool):
        # The fix must not break talk's own reply path.
        ctrl = self._controller({"respond", "speak", "read_file", "bash"})
        allowed, _ = ctrl.can_execute_action({"tool_name": tool})
        assert allowed is True

    def test_restriction_is_derived_not_enumerated(self):
        # A newly-registered tool is denied by construction, not by someone
        # remembering to add it to a list.
        ctrl = self._controller({"respond", "some_brand_new_tool"})
        assert ctrl.can_execute_action({"tool_name": "some_brand_new_tool"})[0] is False


class TestTalkLifecycle:
    def test_campaign_stops_the_talk_loop_first(self, monkeypatch):
        # Two agent loops on one bio-stack is the failure to design out.
        h = _bare_handle(_FakeBridge())
        stopped: list[bool] = []
        monkeypatch.setattr(type(h), "_stop_talk_loop", lambda self, **kw: stopped.append(True))
        monkeypatch.setattr(
            "maxim.simulation.orchestrator.start_simulation_mode",
            lambda **kw: "result",
        )

        class _Lease:
            @staticmethod
            def snapshot(reg):
                return _Lease()

            def restore(self, reg):
                return [], []

        monkeypatch.setattr("maxim.simulation.orchestrator._CampaignToolLease", _Lease)
        h.instance = type("I", (), {"tool_registry": object()})()
        assert h._run_sim(goal="g") == "result"
        assert stopped == [True]

    def test_stop_talk_loop_is_idempotent_and_finishes_bridge(self):
        bridge = _FakeBridge()
        h = _bare_handle(bridge)
        h._talk_stop = threading.Event()
        h._stop_talk_loop()
        assert bridge.finished is True
        assert h._talk_bridge is None
        h._stop_talk_loop()  # second call is a no-op, not a crash


class TestRespondToolWiring:
    def test_registry_gains_respond_when_response_output_present(self, tmp_path):
        # Without a ResponseOutput there is no respond/speak tool at all —
        # talk() would always return None and the chat would never fill.
        from maxim.runtime.bootstrap import build_tool_registry
        from maxim.utils.response_output import ResponseOutput

        bare = build_tool_registry(operational_mode="active")
        assert "respond" not in set(bare.list_all())

        wired = build_tool_registry(
            operational_mode="active", response_output=ResponseOutput(sandbox_path=str(tmp_path))
        )
        assert {"respond", "speak"} <= set(wired.list_all())


fastapi = pytest.importorskip("fastapi", reason="requires the `console` extra (fastapi/uvicorn)")

from fastapi.testclient import TestClient  # noqa: E402

from maxim.console.server import build_app  # noqa: E402

_TOKEN = "mxc_" + "t" * 43
_AUTH = {"Authorization": f"Bearer {_TOKEN}"}


class TestTalkEndpoint:
    def test_requires_input(self):
        with TestClient(build_app(None, auth_token=_TOKEN), headers=_AUTH) as c:
            assert c.post("/api/run", json={"mode": "talk", "input": "  "}).status_code == 422

    def test_only_sim_remains_501(self):
        # rest is LIVE now; sim is deliberately a pointer at the CLI rather
        # than a console surface (see TestSimModeIsAPointerNotAStub).
        with TestClient(build_app(None, auth_token=_TOKEN), headers=_AUTH) as c:
            assert c.post("/api/run", json={"mode": "sim"}).status_code == 501

    def test_turn_is_run_id_scoped_and_reply_goes_to_the_stream(self, monkeypatch):
        # run_id scoping is what keeps a background adventure's narration from
        # interleaving into a conversation on the client.
        from maxim.console import server as srv

        seen_run_ids: list = []

        class FakeHandle:
            agent_id = "console_agent"

            def talk(self, text, **kw):
                seen_run_ids.append(srv._event_hub._run_id)
                return {"turn": 1, "response": "hi", "actions": [1], "timed_out": False}

        monkeypatch.setattr(srv, "_get_handle", lambda: FakeHandle())
        with TestClient(build_app(None, auth_token=_TOKEN), headers=_AUTH) as c:
            r = c.post("/api/run", json={"mode": "talk", "input": "hello"})
        assert r.status_code == 200
        body = r.json()
        assert body["mode"] == "talk"
        assert body["session_id"].startswith("talk_")  # per-turn id, not a constant
        assert body["status"] == "completed"  # the call blocked; work is done
        assert body["reply"] == "hi"  # also on /ws, but robust to a late client
        assert "/ws" in body["detail"]  # points the caller at the stream
        assert seen_run_ids == [srv._TALK_RUN_ID]  # stamped during the turn
        assert srv._event_hub._run_id is None  # and cleared after

    def test_talk_rejected_while_an_adventure_runs(self, monkeypatch):
        from maxim.console import server as srv

        alive = threading.Event()
        t = threading.Thread(target=alive.wait, daemon=True)
        t.start()
        monkeypatch.setitem(srv._active_run, "thread", t)
        try:
            with TestClient(build_app(None, auth_token=_TOKEN), headers=_AUTH) as c:
                r = c.post("/api/run", json={"mode": "talk", "input": "hello"})
            assert r.status_code == 409
            assert "adventure" in r.json()["detail"].lower()
        finally:
            alive.set()
            t.join(timeout=5)

    def test_handle_failure_surfaces_as_500_and_clears_run_id(self, monkeypatch):
        from maxim.console import server as srv

        class BoomHandle:
            agent_id = "console_agent"

            def talk(self, text, **kw):
                raise RuntimeError("no backend")

        monkeypatch.setattr(srv, "_get_handle", lambda: BoomHandle())
        with TestClient(build_app(None, auth_token=_TOKEN), headers=_AUTH) as c:
            r = c.post("/api/run", json={"mode": "talk", "input": "hello"})
        assert r.status_code == 500
        assert "no backend" in r.json()["detail"]
        assert srv._event_hub._run_id is None  # not stranded on the failure path
        assert not srv._talk_lock.locked()  # lock released for the next turn


class TestRestMode:
    """rest is a MODE, not an alias for stop: it consolidates and the agent
    stays usable, so a later talk turn sees the consolidated substrate."""

    def _handle_with_hippo(self, hippo):
        h = _bare_handle()
        h.instance = type("I", (), {"hippocampus": hippo})()
        return h

    def test_uses_clustering_when_scn_is_wired(self, sim_logging):
        calls: list[str] = []

        class Hippo:
            scn = object()

            def sleep_with_clustering(self):
                calls.append("clustered")
                return {"promoted": 2, "removed": 1}

            def sleep(self):  # pragma: no cover
                calls.append("plain")
                return {}

        h = self._handle_with_hippo(Hippo())
        assert h.rest() == {"promoted": 2, "removed": 1}
        assert calls == ["clustered"]

    def test_falls_back_to_plain_sleep_without_scn(self, sim_logging):
        calls: list[str] = []

        class Hippo:
            scn = None

            def sleep(self):
                calls.append("plain")
                return {"promoted": 1}

        h = self._handle_with_hippo(Hippo())
        h.rest()
        assert calls == ["plain"]

    def test_does_not_stop_the_agent(self, sim_logging):
        # The distinction from stop(): the handle remains usable afterwards.
        class Hippo:
            scn = None

            def sleep(self):
                return {}

        h = self._handle_with_hippo(Hippo())
        h.rest()
        assert h._stopped is False

    def test_stops_a_live_talk_loop_first(self, sim_logging, monkeypatch):
        # Consolidating under a live loop would race its reads of the same
        # substrate; required=True means refuse rather than run both.
        stopped: list[dict] = []

        class Hippo:
            scn = None

            def sleep(self):
                return {}

        h = self._handle_with_hippo(Hippo())
        monkeypatch.setattr(type(h), "_stop_talk_loop", lambda self, **kw: stopped.append(kw))
        h.rest()
        assert stopped == [{"required": True}]

    def test_refused_during_a_campaign_and_after_stop(self, sim_logging):
        h = self._handle_with_hippo(None)
        h._campaign_lock.acquire()
        try:
            with pytest.raises(RuntimeError, match="adventure is running"):
                h.rest()
        finally:
            h._campaign_lock.release()
        h._stopped = True
        with pytest.raises(RuntimeError, match="stopped"):
            h.rest()

    def test_no_hippocampus_returns_empty_not_error(self, sim_logging):
        h = self._handle_with_hippo(None)
        assert h.rest() == {}


class TestSimModeIsAPointerNotAStub:
    def test_sim_501_names_the_cli_alternative(self):
        with TestClient(build_app(None, auth_token=_TOKEN), headers=_AUTH) as c:
            r = c.post("/api/run", json={"mode": "sim"})
        assert r.status_code == 501
        detail = r.json()["detail"]
        assert "maxim --sim" in detail, "the 501 must point somewhere, not read as an unfinished stub"
        assert "talk" in detail and "adventure" in detail and "rest" in detail


class TestRestEndpoint:
    def test_rest_returns_completed_with_counts(self, monkeypatch):
        from maxim.console import server as srv

        class FakeHandle:
            agent_id = "console_agent"

            def rest(self):
                return {"promoted": 3, "removed": 1}

        monkeypatch.setattr(srv, "_get_handle", lambda: FakeHandle())
        with TestClient(build_app(None, auth_token=_TOKEN), headers=_AUTH) as c:
            r = c.post("/api/run", json={"mode": "rest"})
        body = r.json()
        assert r.status_code == 200
        assert body["mode"] == "rest" and body["status"] == "completed"
        assert "promoted=3" in body["detail"]
        assert srv._event_hub._run_id is None  # scope restored

    def test_rest_refused_while_an_adventure_runs(self, monkeypatch):
        from maxim.console import server as srv

        alive = threading.Event()
        t = threading.Thread(target=alive.wait, daemon=True)
        t.start()
        monkeypatch.setitem(srv._active_run, "thread", t)
        try:
            with TestClient(build_app(None, auth_token=_TOKEN), headers=_AUTH) as c:
                assert c.post("/api/run", json={"mode": "rest"}).status_code == 409
        finally:
            alive.set()
            t.join(timeout=5)


class TestSilentTurnsExplainThemselves:
    """A turn must never end silently.

    Reported from live console use: a question ran internet_search, the search
    failed, and the turn produced no words — so the chat showed "(no reply)"
    and the only explanation was an ❌ FAIL record buried in the bio panel.
    """

    def _act(self, name, ok=True, blocked=False):
        from maxim.simulation.sinks import ActionRecord

        return ActionRecord(timestamp=0.0, tool_name=name, result_success=ok, blocked=blocked)

    def test_failed_tool_is_named(self):
        from maxim.console.handle import _describe_silent_turn

        msg = _describe_silent_turn({"actions": [self._act("internet_search", ok=False)]})
        assert "internet_search" in msg and "failed" in msg

    def test_blocked_tool_counts_as_failed(self):
        from maxim.console.handle import _describe_silent_turn

        assert "bash" in _describe_silent_turn({"actions": [self._act("bash", blocked=True)]})

    def test_succeeded_tools_are_still_named(self):
        from maxim.console.handle import _describe_silent_turn

        msg = _describe_silent_turn({"actions": [self._act("read_file")]})
        assert "read_file" in msg

    def test_timeout_is_distinguished(self):
        from maxim.console.handle import _describe_silent_turn

        assert "timed out" in _describe_silent_turn({"actions": [], "timed_out": True})

    def test_placeholder_carries_structured_reason_on_the_wire(self, sim_logging, monkeypatch):
        # The web chat renders from the stream, so the reason must be in the
        # RECORD, not only in the human string.
        from maxim.console.handle import MaximHandle

        class Bridge:
            def send_and_wait(self, text, **kw):
                return {
                    "turn": 1,
                    "response": None,
                    "actions": [
                        __import__("maxim.simulation.sinks", fromlist=["ActionRecord"]).ActionRecord(
                            timestamp=0.0, tool_name="internet_search", result_success=False
                        )
                    ],
                    "timed_out": False,
                }

        h = _bare_handle(Bridge())
        monkeypatch.setattr(MaximHandle, "_ensure_talk_loop", lambda self: Bridge())
        h.talk("what's the weather?")
        rec = [r for r in get_sim_records() if r["subsystem"] == "RESPONSE"][-1]
        assert rec["data"]["failed_actions"] == ["internet_search"]
        assert "internet_search" in rec["data"]["no_reply_reason"]
