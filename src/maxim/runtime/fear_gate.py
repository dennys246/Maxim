"""FearGatedExecutor — safety review layer wrapping any Executor.

FearAgent reviews every tool call before execution. This operates
independently of DefaultNetwork, ensuring safety gating works in
headless, simulation, and robot modes alike.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from maxim.tools.base import ToolOutput

if TYPE_CHECKING:
    from maxim.agents.fear_agent import FearAgent
    from maxim.bridges.pain_bridge import PainCircuitBridge

logger = logging.getLogger(__name__)


class FearGatedExecutor:
    """Wraps an Executor with FearAgent safety review.

    Every tool call passes through FearAgent.review_action() before
    reaching the underlying executor. Blocked actions return a
    ToolOutput with success=False and the rejection reason.

    This is independent of DefaultNetwork — it works in all modes:
    robot, headless, and simulation.

    Example:
        fear_agent = FearAgent(llm=llm_router)
        gated = FearGatedExecutor(executor, fear_agent)
        result = gated.execute({"tool_name": "bash", "params": {"command": "rm -rf /"}})
        # result.success == False, result.error contains rejection reason
    """

    def __init__(
        self,
        executor: Any,
        fear_agent: FearAgent,
        pain_bridge: PainCircuitBridge | None = None,
    ) -> None:
        self._executor = executor
        self._fear_agent = fear_agent
        self._pain_bridge = pain_bridge
        self._blocks: int = 0
        self._reviews: int = 0

    def execute(self, action: dict[str, Any]) -> ToolOutput:
        """Review action with FearAgent, then execute if allowed."""
        tool_name = action.get("tool_name", "unknown")
        params = action.get("params", {}) if isinstance(action.get("params"), dict) else {}

        self._reviews += 1

        # Determine action type for FearAgent
        action_type = _classify_action(tool_name)

        # Build action params for review
        review_params: dict[str, Any] = {
            "tool_name": tool_name,
            **params,
        }

        # Add command for shell execution review
        if action_type == "shell_exec" and "command" in params:
            review_params["command"] = params["command"]

        # Add path for file write review
        if action_type == "file_write" and "path" in params:
            review_params["path"] = params["path"]

        # Review the action
        review = self._fear_agent.review_action(
            action_type=action_type,
            action_params=review_params,
            agent_id="executor",
            pain_bridge=self._pain_bridge,
        )

        if not review.allow:
            self._blocks += 1
            reason = review.summary or "Blocked by FearAgent"
            logger.warning(
                "FearAgent BLOCKED %s: %s (risk=%s, findings=%d)",
                tool_name,
                reason,
                review.risk.value,
                len(review.findings),
            )

            # Simulation verbosity
            try:
                from maxim.simulation.sim_logger import sim_fear

                sim_fear(tool_name, allowed=False, reason=reason)
            except ImportError:
                pass
            except Exception as e:
                logger.debug("sim_fear log failed: %s", e)

            return ToolOutput(
                success=False,
                error=f"BLOCKED by FearAgent: {reason}",
                metadata={"fear_agent_blocked": True, "risk": review.risk.value},
            )

        # Also review code content if the tool writes/executes code
        code_content = _extract_code_content(tool_name, params)
        if code_content:
            code_review = self._fear_agent.review_code(
                code_content,
                source=f"tool:{tool_name}",
                context=f"Code submitted via {tool_name}",
            )
            if not code_review.allow:
                self._blocks += 1
                reason = code_review.summary or "Code blocked by FearAgent"
                logger.warning(
                    "FearAgent blocked code in %s: %s",
                    tool_name,
                    reason,
                )
                return ToolOutput(
                    success=False,
                    error=f"BLOCKED by FearAgent (code review): {reason}",
                    metadata={"fear_agent_blocked": True, "risk": code_review.risk.value},
                )

        # Simulation verbosity — log approval
        try:
            from maxim.simulation.sim_logger import sim_fear

            sim_fear(tool_name, allowed=True)
        except ImportError:
            pass
        except Exception as e:
            logger.debug("sim_fear log failed: %s", e)

        logger.info("FearAgent approved %s — executing", tool_name)

        # FearAgent approved — forward to real executor
        result = self._executor.execute(action)

        # Log execution outcome (always, not just sim)
        if not result.success:
            logger.info(
                "Tool %s failed: %s",
                tool_name,
                (result.error or "unknown error")[:200],
            )

        # Simulation verbosity — log execution result
        try:
            from maxim.simulation.sim_logger import sim_action

            summary = str(result.output)[:80] if result.output else (result.error or "")[:80]
            sim_action(tool_name, result.success, summary)
        except ImportError:
            pass
        except Exception as e:
            logger.debug("sim_action log failed: %s", e)

        return result

    def get_stats(self) -> dict[str, int]:
        """Return FearGate statistics."""
        return {
            "reviews": self._reviews,
            "blocks": self._blocks,
            "allowed": self._reviews - self._blocks,
        }

    # Forward all other attributes to the wrapped executor
    def __getattr__(self, name: str) -> Any:
        return getattr(self._executor, name)


def _classify_action(tool_name: str) -> str:
    """Map tool name to FearAgent action type."""
    shell_tools = {"bash", "execute_file", "execute_sandbox_script"}
    file_write_tools = {"write_file", "edit_file", "create_sandbox_script", "write_sandbox_file"}
    network_tools = {"internet_search", "http_fetch", "internet_access"}

    if tool_name in shell_tools:
        return "shell_exec"
    if tool_name in file_write_tools:
        return "file_write"
    if tool_name in network_tools:
        return "network_request"
    return "tool_call"


def _extract_code_content(tool_name: str, params: dict[str, Any]) -> str | None:
    """Extract code content from tool params for review."""
    # Bash commands
    if tool_name == "bash" and "command" in params:
        return params["command"]
    # Script execution
    if tool_name in ("execute_file", "execute_sandbox_script") and "content" in params:
        return params["content"]
    # File writes that might contain code
    if tool_name in ("write_file", "create_sandbox_script", "write_sandbox_file"):
        content = params.get("content", "")
        if content and len(content) > 10:
            return content
    # Edit file new text
    if tool_name == "edit_file":
        new_text = params.get("new_text", "")
        if new_text and len(new_text) > 10:
            return new_text
    return None
