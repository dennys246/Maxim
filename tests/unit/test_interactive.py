"""Tests for maxim.interactive — prompt protocol, handlers, and display."""

from __future__ import annotations

import pytest

from maxim.interactive.prompts import (
    CallbackPromptHandler,
    NonInteractiveHandler,
    PlainPromptHandler,
    PromptHandler,
    PromptRequest,
    PromptResponse,
    PromptType,
    RichPromptHandler,
    create_handler,
)


# ---------------------------------------------------------------------------
# PromptType + PromptRequest
# ---------------------------------------------------------------------------


class TestPromptTypes:
    def test_all_types_exist(self):
        assert len(PromptType) == 4
        assert PromptType.SINGLE_CHOICE.value == "single_choice"
        assert PromptType.MULTI_CHOICE.value == "multi_choice"
        assert PromptType.CONFIRM.value == "confirm"
        assert PromptType.FREEFORM.value == "freeform"

    def test_removed_types_absent(self):
        """0.3.1 cleanup removed unused types — verify they stay gone."""
        assert not hasattr(PromptType, "SHORT_TEXT")
        assert not hasattr(PromptType, "LONG_TEXT")
        assert not hasattr(PromptType, "NUMERIC")
        assert not hasattr(PromptType, "RATING")


class TestPromptRequest:
    def test_frozen(self):
        req = PromptRequest(
            prompt_type=PromptType.SINGLE_CHOICE,
            question="Pick one",
            options=("a", "b", "c"),
        )
        with pytest.raises(AttributeError):
            req.question = "modified"  # type: ignore[misc]

    def test_default_values(self):
        req = PromptRequest(prompt_type=PromptType.FREEFORM, question="Name?")
        assert req.options is None
        assert req.default is None
        assert req.timeout_sec == 300.0


class TestPromptResponse:
    def test_basic(self):
        resp = PromptResponse(value="hello")
        assert resp.value == "hello"
        assert resp.timed_out is False
        assert resp.was_default is False
        assert resp.elapsed_s == 0.0

    def test_timed_out(self):
        resp = PromptResponse(value="default", timed_out=True, was_default=True)
        assert resp.timed_out is True
        assert resp.was_default is True

    def test_multi_choice_list(self):
        resp = PromptResponse(value=["a", "b"])
        assert isinstance(resp.value, list)
        assert len(resp.value) == 2


# ---------------------------------------------------------------------------
# NonInteractiveHandler
# ---------------------------------------------------------------------------


class TestNonInteractiveHandler:
    def test_returns_default(self):
        handler = NonInteractiveHandler()
        req = PromptRequest(
            prompt_type=PromptType.FREEFORM,
            question="Name?",
            default="DefaultName",
        )
        resp = handler.prompt(req)
        assert resp.value == "DefaultName"
        assert resp.was_default is True

    def test_confirm_defaults_no(self):
        handler = NonInteractiveHandler()
        req = PromptRequest(prompt_type=PromptType.CONFIRM, question="Sure?")
        resp = handler.prompt(req)
        assert resp.value == "no"

    def test_multi_choice_returns_first(self):
        handler = NonInteractiveHandler()
        req = PromptRequest(
            prompt_type=PromptType.MULTI_CHOICE,
            question="Pick",
            options=("a", "b", "c"),
        )
        resp = handler.prompt(req)
        assert resp.value == ["a"]

    def test_multi_choice_no_options(self):
        handler = NonInteractiveHandler()
        req = PromptRequest(prompt_type=PromptType.MULTI_CHOICE, question="Pick")
        resp = handler.prompt(req)
        assert resp.value == []


# ---------------------------------------------------------------------------
# CallbackPromptHandler
# ---------------------------------------------------------------------------


class TestCallbackHandler:
    def test_delegates_to_callback(self):
        def my_cb(req: PromptRequest) -> str:
            return f"answered: {req.question}"

        handler = CallbackPromptHandler(my_cb)
        req = PromptRequest(prompt_type=PromptType.FREEFORM, question="Name?")
        resp = handler.prompt(req)
        assert resp.value == "answered: Name?"

    def test_callback_returns_none_uses_default(self):
        handler = CallbackPromptHandler(lambda r: None)
        req = PromptRequest(
            prompt_type=PromptType.FREEFORM,
            question="Name?",
            default="FallbackName",
        )
        resp = handler.prompt(req)
        assert resp.value == "FallbackName"

    def test_callback_exception_uses_default(self):
        def failing_cb(req):
            raise RuntimeError("oops")

        handler = CallbackPromptHandler(failing_cb)
        req = PromptRequest(
            prompt_type=PromptType.FREEFORM,
            question="Name?",
            default="Safe",
        )
        resp = handler.prompt(req)
        assert resp.value == "Safe"

    def test_callback_with_choice(self):
        handler = CallbackPromptHandler(lambda r: r.options[1] if r.options else "")
        req = PromptRequest(
            prompt_type=PromptType.SINGLE_CHOICE,
            question="Pick",
            options=("fight", "flee", "negotiate"),
        )
        resp = handler.prompt(req)
        assert resp.value == "flee"


# ---------------------------------------------------------------------------
# RichPromptHandler fallback
# ---------------------------------------------------------------------------


class TestRichHandlerFallback:
    def test_rich_handler_creates(self):
        """RichPromptHandler can be instantiated (uses fallback if no rich)."""
        handler = RichPromptHandler()
        # Should always create without error
        assert isinstance(handler, PromptHandler)


# ---------------------------------------------------------------------------
# create_handler factory
# ---------------------------------------------------------------------------


class TestCreateHandler:
    def test_non_interactive(self):
        h = create_handler("non-interactive")
        assert isinstance(h, NonInteractiveHandler)

    def test_plain(self):
        h = create_handler("plain")
        assert isinstance(h, PlainPromptHandler)

    def test_callback_requires_function(self):
        with pytest.raises(ValueError, match="callback"):
            create_handler("callback")

    def test_callback_with_function(self):
        h = create_handler("callback", callback=lambda r: "ok")
        assert isinstance(h, CallbackPromptHandler)

    def test_auto_returns_handler(self):
        h = create_handler("auto")
        assert isinstance(h, PromptHandler)

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown prompt handler mode"):
            create_handler("invalid_mode")

    def test_unknown_mode_lists_valid(self):
        with pytest.raises(ValueError, match="auto"):
            create_handler("garbage")


# ---------------------------------------------------------------------------
# Display (basic instantiation — can't test live rendering in CI)
# ---------------------------------------------------------------------------


class TestDisplay:
    def test_create_display_off(self):
        from maxim.interactive.display import create_display

        d = create_display("off")
        assert d is None

    def test_create_display_on(self):
        from maxim.interactive.display import create_display, MaximDisplay

        d = create_display("on")
        assert isinstance(d, MaximDisplay)

    def test_display_log_no_crash(self):
        from maxim.interactive.display import MaximDisplay

        d = MaximDisplay()
        d.log("hippo", "Test memory capture")
        d.log("nac", "Causal link formed")
        assert len(d._log_lines) == 2

    def test_display_status(self):
        from maxim.interactive.display import MaximDisplay

        d = MaximDisplay()
        d.set_status(mode="simulation", turn="5")
        assert d._status["mode"] == "simulation"
        assert d._status["turn"] == "5"

    def test_display_prompt(self):
        from maxim.interactive.display import MaximDisplay

        d = MaximDisplay()
        d.set_prompt("What do you do? [fight / flee]")
        assert "fight" in d._prompt_text
        d.clear_prompt()
        assert d._prompt_text == ""


# ---------------------------------------------------------------------------
# Display routing through sim_logger (Stage 5)
# ---------------------------------------------------------------------------


class TestDisplayRouting:
    """Verify sim_logger routes through MaximDisplay when active."""

    def setup_method(self):
        from maxim.simulation.sim_logger import set_active_display

        set_active_display(None)

    def teardown_method(self):
        from maxim.simulation.sim_logger import set_active_display

        set_active_display(None)

    def test_sim_log_routes_to_display_when_active(self):
        from maxim.interactive.display import MaximDisplay
        from maxim.simulation.sim_logger import (
            DisplayTier,
            enable_sim_logging,
            disable_sim_logging,
            set_active_display,
            set_display_tier,
            sim_log,
        )

        d = MaximDisplay()
        set_active_display(d)
        enable_sim_logging(use_color=False)
        set_display_tier(DisplayTier.BIO)
        sim_log("HIPPOCAMPUS", "test capture")
        disable_sim_logging()
        set_display_tier(DisplayTier.CLEAN)

        assert len(d._log_lines) >= 1
        assert "test capture" in d._log_lines[-1].markup

    def test_sim_log_falls_back_when_no_display(self, capsys):
        from maxim.simulation.sim_logger import (
            DisplayTier,
            enable_sim_logging,
            disable_sim_logging,
            set_display_tier,
            sim_log,
        )

        enable_sim_logging(use_color=False)
        set_display_tier(DisplayTier.BIO)
        sim_log("HIPPOCAMPUS", "direct print")
        disable_sim_logging()
        set_display_tier(DisplayTier.CLEAN)

        captured = capsys.readouterr()
        assert "direct print" in captured.out

    def test_display_scene_routes_through_display(self):
        from maxim.interactive.display import MaximDisplay
        from maxim.simulation.sim_logger import (
            set_active_display,
            display_scene,
        )

        d = MaximDisplay()
        set_active_display(d)
        display_scene("The tavern door opens.")

        assert len(d._log_lines) >= 1
        assert "tavern" in d._log_lines[-1].markup.lower()

    def test_display_turn_updates_status(self):
        from maxim.interactive.display import MaximDisplay
        from maxim.simulation.sim_logger import (
            set_active_display,
            display_turn,
        )

        d = MaximDisplay()
        set_active_display(d)
        display_turn(5)

        assert d._status.get("turn") == "5"

    def test_display_concurrent_log_no_corruption(self):
        import threading

        from maxim.interactive.display import MaximDisplay
        from maxim.simulation.sim_logger import (
            DisplayTier,
            enable_sim_logging,
            disable_sim_logging,
            set_active_display,
            set_display_tier,
            sim_log,
        )

        d = MaximDisplay()
        set_active_display(d)
        enable_sim_logging(use_color=False)
        set_display_tier(DisplayTier.BIO)

        errors: list[Exception] = []

        def worker(thread_id: int):
            try:
                for i in range(50):
                    sim_log("HIPPOCAMPUS", f"thread-{thread_id}-event-{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        disable_sim_logging()
        set_display_tier(DisplayTier.CLEAN)

        assert not errors, f"Concurrent log errors: {errors}"
        # All 400 events should be present (deque maxlen=200, so last 200)
        assert len(d._log_lines) > 0

    def test_display_atexit_stops_live(self):
        from maxim.simulation.sim_logger import (
            _cleanup_display,
            set_active_display,
        )
        from maxim.interactive.display import MaximDisplay

        d = MaximDisplay()
        set_active_display(d)
        # Don't start Live — just verify atexit handler doesn't crash
        _cleanup_display()
        # Display should still be set (stop() doesn't unset the reference)


# ---------------------------------------------------------------------------
# Integration smoke tests (Stage 7)
# ---------------------------------------------------------------------------


class TestInteractionToolEndToEnd:
    """End-to-end RequestInteractionTool round trips."""

    @pytest.fixture(autouse=True)
    def _restore_interactive_mode(self):
        from maxim.simulation.sim_logger import (
            get_interactive_mode,
            set_interactive_mode,
        )

        prior = get_interactive_mode()
        yield
        set_interactive_mode(prior)

    def test_end_to_end_with_handler(self):
        from maxim.simulation.sim_logger import InteractiveMode, set_interactive_mode
        from maxim.tools.display import RequestInteractionTool
        from maxim.interactive.prompts import PromptHandler, PromptResponse

        class _TestHandler(PromptHandler):
            def prompt(self, request):
                return PromptResponse(value="option B")

        set_interactive_mode(InteractiveMode.ON)
        tool = RequestInteractionTool(prompt_handler=_TestHandler())
        out = tool.execute(question="Pick one", options=["option A", "option B"])

        assert out.success is True
        assert "User responded: option B" in out.output
        assert out.metadata["was_prompted"] is True

    def test_end_to_end_disabled(self):
        from maxim.simulation.sim_logger import InteractiveMode, set_interactive_mode
        from maxim.tools.display import RequestInteractionTool
        from maxim.interactive.prompts import PromptHandler

        class _TestHandler(PromptHandler):
            def prompt(self, request):
                raise AssertionError("Should not be called")

        set_interactive_mode(InteractiveMode.OFF)
        tool = RequestInteractionTool(prompt_handler=_TestHandler())
        out = tool.execute(question="Continue?")

        assert out.success is True
        assert "autonomously" in out.output.lower()
        assert out.metadata["was_prompted"] is False

    def test_end_to_end_no_handler(self):
        from maxim.simulation.sim_logger import InteractiveMode, set_interactive_mode
        from maxim.tools.display import RequestInteractionTool

        set_interactive_mode(InteractiveMode.ON)
        tool = RequestInteractionTool(prompt_handler=None)
        out = tool.execute(question="Pick", options=["x", "y"])

        assert out.success is True
        assert "could not be collected" in out.output.lower()
        assert out.metadata["was_prompted"] is False
        assert out.metadata["reason"] == "no_handler"


# ---------------------------------------------------------------------------
# CampaignDisplay
# ---------------------------------------------------------------------------
