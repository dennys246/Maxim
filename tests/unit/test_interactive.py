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
    freeze_context,
)


# ---------------------------------------------------------------------------
# PromptType + PromptRequest
# ---------------------------------------------------------------------------


class TestPromptTypes:
    def test_all_types_exist(self):
        assert len(PromptType) == 8
        assert PromptType.SINGLE_CHOICE.value == "single_choice"
        assert PromptType.MULTI_CHOICE.value == "multi_choice"
        assert PromptType.CONFIRM.value == "confirm"
        assert PromptType.SHORT_TEXT.value == "short_text"
        assert PromptType.LONG_TEXT.value == "long_text"
        assert PromptType.FREEFORM.value == "freeform"
        assert PromptType.NUMERIC.value == "numeric"
        assert PromptType.RATING.value == "rating"


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
        req = PromptRequest(prompt_type=PromptType.SHORT_TEXT, question="Name?")
        assert req.options is None
        assert req.default is None
        assert req.timeout_sec == 300.0
        assert req.context == ()

    def test_freeze_context_helper(self):
        ctx = freeze_context(category="creation", step=1)
        assert isinstance(ctx, tuple)
        assert ("category", "creation") in ctx
        assert ("step", 1) in ctx

    def test_with_context(self):
        req = PromptRequest(
            prompt_type=PromptType.SHORT_TEXT,
            question="Name?",
            context=freeze_context(phase="character_creation"),
        )
        assert dict(req.context)["phase"] == "character_creation"


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
            prompt_type=PromptType.SHORT_TEXT,
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
        req = PromptRequest(prompt_type=PromptType.SHORT_TEXT, question="Name?")
        resp = handler.prompt(req)
        assert resp.value == "answered: Name?"

    def test_callback_returns_none_uses_default(self):
        handler = CallbackPromptHandler(lambda r: None)
        req = PromptRequest(
            prompt_type=PromptType.SHORT_TEXT,
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
            prompt_type=PromptType.SHORT_TEXT,
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
# CampaignDisplay
# ---------------------------------------------------------------------------


class TestCampaignDisplay:
    def test_creates_without_state(self):
        from maxim.interactive.dm_display import CampaignDisplay
        ext = CampaignDisplay()
        assert ext.panel_name() == "Campaign — Encounter"

    def test_set_panel(self):
        from maxim.interactive.dm_display import CampaignDisplay
        ext = CampaignDisplay()
        ext._set_panel("character")
        assert ext._active_panel == "character"
        assert "Character" in ext.panel_name()

    def test_add_note(self):
        from maxim.interactive.dm_display import CampaignDisplay
        ext = CampaignDisplay()
        ext.add_note("Remember to check the guard's pocket")
        assert len(ext._notes) == 1

    def test_key_bindings(self):
        from maxim.interactive.dm_display import CampaignDisplay
        ext = CampaignDisplay()
        bindings = ext.key_bindings()
        assert "c" in bindings  # character
        assert "i" in bindings  # inventory
        assert "e" in bindings  # encounter

    def test_render_encounter_with_state(self):
        from maxim.interactive.dm_display import CampaignDisplay
        from maxim.simulation.dm_runtime import CampaignState

        state = CampaignState(
            current_encounter="tavern",
            turn_count=3,
            flags={"cooperated", "has_sword"},
            choices_made=[{"encounter": "gate", "choice": "enter"}],
        )
        ext = CampaignDisplay(campaign_state=state)
        ext._set_panel("encounter")
        rendered = ext.render()
        # Should produce a renderable without error
        assert rendered is not None

    def test_render_history_with_choices(self):
        from maxim.interactive.dm_display import CampaignDisplay
        from maxim.simulation.dm_runtime import CampaignState

        state = CampaignState(
            choices_made=[
                {"encounter": "gate", "choice": "enter"},
                {"encounter": "tavern", "choice": "talk"},
            ],
        )
        ext = CampaignDisplay(campaign_state=state)
        ext._set_panel("history")
        rendered = ext.render()
        assert rendered is not None
