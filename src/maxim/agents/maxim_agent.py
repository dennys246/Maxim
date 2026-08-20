"""MaximAgent: Composite agent embodying the full agentic architecture.

Integrates:
- PerceptionAgent: Observes and classifies inputs
- MemoryAgent: Maintains salient context (CORE EXPLORATION FOCUS)
- ExecAgent: Proposes goals via LLM
- AgenticGoalAgent: Holds and executes goals with sub-goal chaining

Root goal: "Understand reality and help people."
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any

from maxim.agents.base import Agent
from maxim.agents.bus import AgentBus, ToolResult
from maxim.agents.perception_agent import PerceptionAgent
from maxim.agents.memory_agent import MemoryAgent
from maxim.agents.exec_agent import ExecAgent
from maxim.agents.agentic_goal_agent import AgenticGoalAgent

if TYPE_CHECKING:
    from maxim.runtime.capture import CaptureManager


class MaximAgent(Agent):
    """
    Composite agent embodying the full Maxim agentic architecture.

    Root goal: "Understand reality and help people."

    Components:
    - PerceptionAgent: Observes and classifies inputs
    - MemoryAgent: Maintains salient context (CORE EXPLORATION FOCUS)
    - ExecAgent: Proposes goals via LLM
    - AgenticGoalAgent: Holds and executes goals with sub-goal chaining

    Data Flow:
    1. ReachyEnv.observe() → raw observation
    2. PerceptionAgent → Percept (normalized, classified, salience/novelty)
    3. MemoryAgent → updates salient memory, builds StructuredContext
    4. ExecAgent → proposes ProposedGoal based on context + LLM
    5. AgenticGoalAgent → accepts/rejects goal, executes via tools
    6. ToolResult → fed back to MemoryAgent for outcome tracking
    """

    agent_name = "maxim"
    state_name = "MaximAgent"

    ROOT_GOAL = "Understand reality and help people."

    def __init__(
        self,
        *,
        name: str | None = None,
        enabled: bool = True,
        interests: list[int] | None = None,
        llm_profile: str = "mistral-7b-instruct-v0.2",
        quantization: str = "Q4_K_M",
        memory_persistence_path: str | None = None,
        data_folder: str | None = None,
        enable_embeddings: bool = False,
        reset_on_startup: bool = False,
        capture_manager: CaptureManager | None = None,
        agent_id: str = "maxim",
    ) -> None:
        super().__init__(name=name, enabled=enabled)
        self._agent_id = agent_id

        # Create the message bus
        self._bus = AgentBus()
        self._data_folder = data_folder

        # Default persistence path if not specified
        if memory_persistence_path is None and data_folder:
            memory_persistence_path = os.path.join(data_folder, "memory", "memories.json")

        # Store capture manager reference (Phase 3)
        self._capture_manager = capture_manager

        # Create agents in dependency order. F0.5: agent_id flows through
        # to percept producers so every Percept produced by this runtime
        # carries the producer's identity in PerceptContext.agent_id.
        self.perception = PerceptionAgent(
            self._bus,
            interests=set(interests or []),
            capture_manager=capture_manager,
            agent_id=agent_id,
        )

        from maxim.utils.filesystem_policy import get_workspace_path

        self.memory = MemoryAgent(
            self._bus,
            max_short_term=100,
            max_long_term=500,
            salience_threshold=0.2,
            persistence_path=memory_persistence_path,
            enable_embeddings=enable_embeddings,
            reset_on_startup=reset_on_startup,
            workspace_path=get_workspace_path(),
            agent_id=agent_id,
        )

        self.exec_agent = ExecAgent(
            self._bus,
            self.memory,
            llm_profile=llm_profile,
            quantization=quantization,
            agent_id=agent_id,
        )

        # Wire WorkingMemorySet: Exec owns it, MemoryAgent writes to it (0.8).
        self.memory._working_memory = self.exec_agent.working_memory

        # PlanManager (created early, wired to agents; services filled in wire_memory_hub)
        from maxim.planning.plan_manager import PlanManager, PlanServices
        from maxim.planning.plan_document import LongHorizonConfig

        plan_config = LongHorizonConfig()
        plans_dir = os.path.join(data_folder or "data", plan_config.plan_persistence_path)
        self._plan_manager = PlanManager(
            plans_dir=plans_dir,
            bus=self._bus,
            config=plan_config,
            services=PlanServices(),
        )

        # Plan dashboard + logger (bus-driven, write to workspace)
        from maxim.planning.plan_dashboard import PlanDashboard
        from maxim.planning.plan_logger import PlanLogger

        workspace_path = get_workspace_path()
        self._plan_dashboard = PlanDashboard(workspace_path, self._bus)
        self._plan_logger = PlanLogger(workspace_path, self._bus)

        self.goal = AgenticGoalAgent(
            self._bus,
            memory_agent=self.memory,
            plan_manager=self._plan_manager,
        )

        # Give memory agent plan awareness
        self.memory._plan_manager = self._plan_manager

        # Math cognition for StatisticianAgent (lazy imports to avoid circular dependency)
        from maxim.agents.statistician_agent import StatisticianAgent
        from maxim.math.ips import IPS
        from maxim.math.angular_gyrus import AngularGyrus

        ips = IPS()
        angular_gyrus = AngularGyrus()

        statistician_path = None
        if data_folder:
            statistician_path = os.path.join(data_folder, "util", "statistician_state.json")

        self.statistician = StatisticianAgent(
            bus=self._bus,
            ips=ips,
            angular_gyrus=angular_gyrus,
            persistence_path=statistician_path,
        )

        # Angular Gyrus reference (also used by MemoryHub integration)
        self._angular_gyrus = angular_gyrus

        # Throttle control
        self._last_intent_time = 0.0
        self._min_intent_interval = 0.1  # 10 Hz max
        self._last_log_time = 0.0
        self._log_interval = 1.0  # Log at most once per second

    def wire_memory_hub(self, memory_hub: Any) -> None:
        """Wire MemoryHub for multi-layer knowledge queries and promotion.

        Call after MemoryHub is created but before on_start().
        Registers StatisticianAgent as a promotion source and gives
        MemoryAgent access to knowledge queries. Also wires PlanManager
        services for rich replan context.

        Phase 0: Also injects Hippocampus reference into MemoryAgent so
        it can capture memories through the single store.
        """
        # Give MemoryAgent access for knowledge context building
        self.memory._memory_hub = memory_hub

        # Phase 0: Wire Hippocampus reference for unified memory storage
        hippocampus = getattr(memory_hub, "hippocampus", None)
        if hippocampus is not None:
            self.memory.connect_hippocampus(hippocampus)
            # Register cleanup callback so MemoryAgent tracks deletions
            hippocampus.register_deletion_callback(self.memory._on_memory_deleted)

        # Register StatisticianAgent as a promotion source
        if hasattr(memory_hub, "register_promotion_source"):
            memory_hub.register_promotion_source(self.statistician)

        # Wire PlanManager services for rich replan context
        if hasattr(self, "_plan_manager") and self._plan_manager:
            svc = self._plan_manager._services
            svc.hippocampus = getattr(memory_hub, "hippocampus", None)
            svc.nac = getattr(memory_hub, "nac", None)
            svc.statistician = self.statistician

        # Wire PatternCompleter for concept graph-chaining predictions
        pattern_completer = getattr(memory_hub, "_pattern_completer", None)
        if pattern_completer is not None:
            self.memory.set_pattern_completion_fn(pattern_completer.complete)

        # Wire NAc to ExecAgent for contemplation outcome learning
        nac = getattr(memory_hub, "nac", None)
        if nac is not None:
            self.exec_agent.wire_nac(nac)

        # Phase 3e: Wire LSH similarity indices
        from maxim.memory.context_index import SimilarityIndex

        self._context_index = SimilarityIndex(num_hashes=64, num_bands=8)
        self._percept_index = SimilarityIndex(num_hashes=64, num_bands=8)
        self.memory._association_index.set_context_index(self._context_index)
        # Wire indices to ExecAgent for recall_deep (Phase 3f)
        self.exec_agent._context_index_ref = self._context_index
        self.exec_agent._percept_index_ref = self._percept_index
        if hippocampus is not None:
            hippocampus._percept_index = self._percept_index

        # Phase 3b: Wire acute staging
        if hippocampus is not None:
            data_folder = self._data_folder or "data"
            staging_dir = os.path.join(data_folder, "short_term_memory")
            scn = getattr(memory_hub, "scn", None)
            weights_path = os.path.join(data_folder, "util", "significance_weights.json")
            self.exec_agent.wire_staging(
                staging_dir=staging_dir,
                hippocampus=hippocampus,
                scn=scn,
                weights_path=weights_path,
            )

    def wire_provenance(self, collector: Any) -> None:
        """Wire ProvenanceCollector for decision tracing.

        Follows wire_memory_hub() pattern. Reaches into MemoryHub to
        wire Hippocampus, ConceptExtractor, ConceptGrounder, NAc for
        Tier 2 (background activity) recording.
        """
        self._collector = collector
        self.memory._collector = collector

        # Wire to MemoryHub internals for Tier 2 (activity log)
        hub = getattr(self.memory, "_memory_hub", None)
        if hub is not None:
            hub._collector = collector
            hub.hippocampus._collector = collector
            if hub._concept_extractor:
                hub._concept_extractor._collector = collector
            if hub._concept_grounder:
                hub._concept_grounder._collector = collector
            if hub.nac:
                hub.nac._collector = collector

    def on_start(self, **kwargs: Any) -> None:
        """Start all component agents."""
        self.perception.on_start(**kwargs)
        self.memory.on_start(**kwargs)
        self.exec_agent.on_start(**kwargs)
        self.goal.on_start(**kwargs)
        try:
            self.statistician.on_start(**kwargs)
        except Exception:
            pass  # Non-critical — don't block startup

    def on_stop(self, **kwargs: Any) -> None:
        """Stop all component agents."""
        try:
            self.statistician.on_stop(**kwargs)
        except Exception:
            pass  # Non-critical — don't block shutdown
        self.goal.on_stop(**kwargs)
        self.exec_agent.on_stop(**kwargs)
        self.memory.on_stop(**kwargs)
        self.perception.on_stop(**kwargs)
        self._bus.clear()

    def propose_intent(self, state: Any, memory: Any, **kwargs: Any) -> dict[str, Any] | None:
        """
        Run the full perception-memory-goal pipeline.

        1. Perception processes observation
        2. Memory updates and builds context
        3. ExecAgent proposes goals (async)
        4. GoalAgent returns current goal as intent
        """
        now = time.time()

        # Throttle to avoid spinning too fast
        elapsed = now - self._last_intent_time
        if elapsed < self._min_intent_interval:
            return None  # Let agent loop handle idle sleep

        self._last_intent_time = now

        # Step 0: Statistical analysis (lightweight, rate-limited internally)
        # Isolated so failures never block the perception-decision-action pipeline
        try:
            self.statistician.on_step(**kwargs)
        except Exception:
            pass

        # Step 1: Perception (publishes Percept to bus)
        self.perception.propose_intent(state, memory, **kwargs)

        # Step 2: ExecAgent checks for proposals needed
        exec_result = self.exec_agent.propose_intent(state, memory, **kwargs)

        # Step 3: GoalAgent returns pending tool call
        goal_result = self.goal.propose_intent(state, memory, **kwargs)
        if goal_result:
            self._log_status("goal_active", goal_result)
            return goal_result

        # Fall back to exec result (startup, file change, default focus)
        if exec_result:
            self._log_status("exec_result", exec_result)
            return exec_result

        # Log idle status periodically
        self._log_status("idle", None)
        return None

    def _log_status(self, status: str, result: Any) -> None:
        """Log status at most once per second."""
        now = time.time()
        if now - self._last_log_time >= self._log_interval:
            self._last_log_time = now
            if status == "idle":
                self.log.debug("Idle - no goal pending")
            elif status == "goal_active":
                goal = result.get("goal") if result else None
                if goal is None:
                    goal = {}
                tool = goal.get("tool_name") if isinstance(goal, dict) else str(goal)
                self.log.info("Goal: %s", tool)
            elif status == "exec_result":
                source = result.get("source", "unknown") if result else "unknown"
                self.log.info("Action from %s", source)

    def wire_communication(
        self,
        gateway: Any,
        nac: Any | None = None,
        memory_hub: Any | None = None,
    ) -> None:
        """Wire CommunicationGateway and CommunicationBridge.

        Call after MemoryHub is created. Sets up:
        - CommunicationBridge for NAc reward attribution
        - Registers bridge as PromotionSource with MemoryHub
        """
        self._gateway = gateway
        if nac is not None:
            from maxim.bridges.communication_bridge import CommunicationBridge

            self._comm_bridge = CommunicationBridge(self._bus, nac)
            if memory_hub and hasattr(memory_hub, "register_promotion_source"):
                memory_hub.register_promotion_source(self._comm_bridge)

    def wire_preemption(
        self,
        preemption_circuit: Any,
        execution_tracker: Any,
        robot: Any = None,
    ) -> None:
        """Wire PreemptionCircuit for interrupt handling.

        Call before on_start(). Gives the goal agent preemption awareness.
        """
        self._preemption_circuit = preemption_circuit
        self._execution_tracker = execution_tracker
        self.goal.execution_tracker = execution_tracker
        self.goal.preemption_circuit = preemption_circuit
        self.wire_robot(robot)

    def wire_robot(self, robot: Any) -> None:
        """Expose the active controller to goal/preemption bookkeeping."""
        self.goal.robot = robot

    def publish_tool_result(
        self,
        tool_call_id: str,
        tool_name: str,
        success: bool,
        result: Any = None,
        error: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> None:
        """Publish tool result for feedback."""
        tr = ToolResult(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            success=success,
            result=result,
            error=error,
            params=params or {},
        )
        self._bus.publish(tr)

    def report_tool_result(
        self,
        tool_call_id: str,
        tool_name: str,
        success: bool,
        result: Any = None,
        error: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> None:
        """Alias for publish_tool_result for compatibility."""
        self.publish_tool_result(tool_call_id, tool_name, success, result, error, params)

    def update_interests(self, add: list[int] | None = None, remove: list[int] | None = None) -> None:
        """Update object class interests for salience boosting.

        Interest classes get a 2x salience multiplier. All COCO classes are
        always detected; this controls prioritization, not visibility.
        """
        self.perception.update_interests(add=add, remove=remove)

    def record_cli_input(self, command: str) -> None:
        """Record a CLI input for context (flows to abstraction stream)."""
        self.memory.add_cli_input(command)

    def get_current_goal(self) -> str:
        """Get description of current goal."""
        goal = self.goal.get_current_goal()
        if goal:
            return goal.description
        return self.ROOT_GOAL

    def get_pending_sub_goals(self) -> list[str]:
        """Get pending sub-goals for current goal."""
        goal = self.goal.get_current_goal()
        if goal and goal.sub_goals:
            return [sg.description for sg in goal.sub_goals if sg.status.name not in ("COMPLETED", "SKIPPED")]
        return []

    def get_current_goal_id(self) -> str | None:
        """Get ID of current goal."""
        goal = self.goal.get_current_goal()
        return goal.id if goal else None

    def cancel_goal(self, reason: str = "Cancelled") -> bool:
        """Cancel the current goal."""
        return self.goal.cancel_current_goal(reason)

    def get_mode(self) -> str:
        """Get the current operating mode."""
        ctx = self.memory.build_context()
        return ctx.mode

    def get_context_summary(self) -> dict[str, Any]:
        """Get a summary of current context for debugging."""
        ctx = self.memory.build_context()
        return {
            "mode": ctx.mode,
            "active_goal": ctx.active_goal,
            "sub_goals_pending": len(ctx.active_goal_sub_goals),
            "recent_percepts": len(ctx.recent_percepts),
            "relevant_memories": len(ctx.relevant_memories),
            "detected_objects": len(ctx.detected_objects),
            "detected_people": len(ctx.detected_people),
            "detected_speech": len(ctx.detected_speech),
            "cli_inputs": len(ctx.cli_inputs),
        }


# Backwards-compatibility alias
AgenticMaximAgent = MaximAgent
