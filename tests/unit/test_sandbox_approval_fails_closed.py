"""#796 — SUPERVISED sandbox approval FAILS CLOSED.

`ExecuteSandboxScriptTool` documented SUPERVISED as an approval gate but installed a callback that
returned True unconditionally, and `SandboxExecutor.execute` itself ran a script whose approval was
required whenever no callback was wired. Each refusal below is checked by the script's SIDE EFFECT
(a marker file), not only by the returned status — a gate is proven by what did not happen.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from maxim.agents.autonomy import AutonomyController, AutonomyLevel
from maxim.tools.sandbox import ExecuteSandboxScriptTool
from maxim.utils.sandbox_executor import ExecutionStatus, SandboxExecutor


@pytest.fixture
def sandbox(tmp_path: Path):
    # A SHELL script: the approval gate sits before the per-type dispatch, so the type is
    # immaterial to it — and the .py path has an independent defect of its own (#800).
    executor = SandboxExecutor(sandbox_dir=str(tmp_path / "sb"))
    marker = Path(executor.sandbox_dir) / "workspace" / "ran.txt"  # the default working dir
    script = Path(executor.sandbox_dir) / "scripts" / "touch.sh"
    script.write_text("printf x >> ran.txt\n")
    return executor, str(script), marker


def _tool(executor: SandboxExecutor, level: AutonomyLevel | None) -> ExecuteSandboxScriptTool:
    ctl = None if level is None else AutonomyController(initial_level=level)
    return ExecuteSandboxScriptTool(executor, ctl)


# ── the red gates: each executed the script before the fix ──


def test_supervised_without_an_approver_refuses(sandbox) -> None:
    executor, script, marker = sandbox
    res = _tool(executor, AutonomyLevel.SUPERVISED).execute(script_path=script)
    assert not res.success
    assert "no approval_callback" in (res.error or "")
    assert not marker.exists()
    assert executor.approval_callback is None, "the tool must not install an approver of its own"


def test_no_autonomy_controller_refuses_too(sandbox) -> None:
    executor, script, marker = sandbox
    res = _tool(executor, None).execute(script_path=script)
    assert not res.success
    assert "no approval_callback" in (res.error or "")
    assert not marker.exists()


def test_executor_required_approval_without_callback_is_blocked(sandbox) -> None:
    executor, script, marker = sandbox
    result = executor.execute(script_path=script, require_approval=True)
    assert result.status == ExecutionStatus.APPROVAL_UNAVAILABLE, "a missing approver is not a denial"
    assert not marker.exists()


# ── the gate still opens for a real answer, and only for it ──


def test_supervised_denial_blocks(sandbox) -> None:
    executor, script, marker = sandbox
    executor.approval_callback = lambda path, content, h: False
    res = _tool(executor, AutonomyLevel.SUPERVISED).execute(script_path=script)
    assert not res.success
    assert res.error == "Script execution not approved"
    assert not marker.exists()


@pytest.mark.parametrize("answer", ["no", {"approved": False}, object()], ids=["str", "dict", "object"])
def test_a_truthy_non_true_answer_is_not_approval(sandbox, answer) -> None:
    """Only a literal True approves — a response object must not read as a yes."""
    executor, script, marker = sandbox
    executor.approval_callback = lambda path, content, h: answer
    res = _tool(executor, AutonomyLevel.SUPERVISED).execute(script_path=script)
    assert res.error == "Script execution not approved"
    assert not marker.exists()


def test_an_async_approver_that_denies_does_not_run(sandbox) -> None:
    """A sync call to an async approver returns a (truthy) coroutine, never its answer."""
    executor, script, marker = sandbox

    async def deny(path: str, content: str, h: str) -> bool:
        return False

    def call(path: str, content: str, h: str):
        coro = deny(path, content, h)
        coro.close()  # never awaited by the executor; close to avoid a RuntimeWarning
        return coro

    executor.approval_callback = call
    res = _tool(executor, AutonomyLevel.SUPERVISED).execute(script_path=script)
    assert res.error == "Script execution not approved"
    assert not marker.exists()


def test_supervised_approval_runs_once_asked_then_not_until_content_changes(sandbox) -> None:
    executor, script, marker = sandbox
    asked: list[str] = []
    executor.approval_callback = lambda path, content, h: asked.append(h) is None
    tool = _tool(executor, AutonomyLevel.SUPERVISED)
    assert tool.execute(script_path=script).success
    assert tool.execute(script_path=script).success
    assert marker.read_text() == "xx"
    assert len(asked) == 1, "an approved hash re-runs without asking again"
    Path(script).write_text(Path(script).read_text() + "# changed\n")
    assert tool.execute(script_path=script).success
    assert len(asked) == 2, "changed content must be re-approved"


def test_autonomous_needs_no_approver(sandbox) -> None:
    executor, script, marker = sandbox
    assert _tool(executor, AutonomyLevel.AUTONOMOUS).execute(script_path=script).success
    assert marker.exists()
