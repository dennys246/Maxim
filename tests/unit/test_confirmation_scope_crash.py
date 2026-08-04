"""Guard for the 2026-08-04 live crash: typed input while a confirmation
was pending killed the agentic loop with UnboundLocalError.

``handle_confirmation`` imported ``display_status`` INSIDE the approve
branch's try-block — a Python scoping trap: any import anywhere in a
function makes the name local to the WHOLE function, so the cancel and
modification branches (which run when the user types "no" or free text)
raised ``UnboundLocalError: cannot access local variable 'display_status'``
before the import ever executed. Live session 2026-08-04 16:57-17:02: an
LLM-hallucinated ``adjust_yaw`` parked an approval prompt; the user typed
"Maxim can you focus on sounds"; the modification branch crashed the loop.

These tests drive the cancel and modification branches directly on a
minimally-constructed LoopController — they fail with UnboundLocalError
on the pre-fix code (verified)."""

from __future__ import annotations

from unittest.mock import MagicMock


from maxim.runtime.loop_controller import LoopController
from maxim.runtime.loop_types import PendingConfirmation


def _controller():
    state = MagicMock()
    state.data = {}
    ctrl = LoopController(
        agent=MagicMock(),
        environment=MagicMock(),
        state=state,
        memory=MagicMock(),
        decision_engine=MagicMock(),
        executor=MagicMock(),
        autonomy_controller=MagicMock(),
    )
    # record_outcome sinks wired later in production bootstrap:
    ctrl.context_pool = MagicMock()
    ctrl.recent_outcomes = []
    ctrl.set_pending_confirmation(
        PendingConfirmation(
            action={"tool_name": "adjust_yaw", "params": {}},
            reasoning="test",
            confidence=0.9,
            tool_name="adjust_yaw",
        )
    )
    return ctrl


class TestConfirmationBranchScope:
    def test_modification_branch_does_not_crash(self):
        """Free text while a confirmation is pending = modification request.
        Pre-fix: UnboundLocalError on display_status → loop death."""
        ctrl = _controller()
        assert ctrl.handle_confirmation("Maxim can you focus on sounds") is True
        assert ctrl.get_pending_confirmation() is None
        assert "pending_modification" in ctrl.state.data

    def test_cancel_branch_does_not_crash(self):
        ctrl = _controller()
        assert ctrl.handle_confirmation("no") is True
        assert ctrl.get_pending_confirmation() is None
