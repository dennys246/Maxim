from __future__ import annotations

import dataclasses
import logging
import threading
import time
import uuid
from typing import TYPE_CHECKING, Any, Callable

from maxim.tools.base import Tool, ToolOutput
from maxim.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from maxim.agents.permissions import AgentPermissions
    from maxim.bridges.tool_pain_bridge import ToolPainBridge
    from maxim.embodiment.body import Embodiment
    from maxim.proprioception.pain import PainDetector


# ── Tool alias map ────────────────────────────────────────────────────────
# LLMs (especially small ones) hallucinate tool names from their training
# data rather than using the registered tool list.  This map silently
# redirects common hallucinations to the correct registered tool.
#
# How to expand: add entries mapping the hallucinated name (lowercase) to
# the registered tool name.  The executor normalises the incoming name to
# lowercase before lookup, so casing variations are handled automatically.
#
# See also: docs/troubleshooting/tool_aliases.md
TOOL_ALIASES: dict[str, str] = {
    # Memory / recall → memory_recall
    "remember": "memory_recall",
    "recall": "memory_recall",
    "recall_memory": "memory_recall",
    "search_memory": "memory_recall",
    # Speech / dialogue → say
    "speech_recognition": "say",
    "speechrecognition": "say",
    "speech": "say",
    "dialogue": "say",
    "talk": "say",
    # NLP / analysis → think
    "natural_language_processing": "think",
    "nlp": "think",
    "nlp_extractor": "think",
    "nlp_understanding": "think",
    "reflection": "think",
    "analyze_text": "think",
    "research": "think",
    # Dialogue parsing → think
    "dialogue_parser": "think",
    "dialogueparser": "think",
    "parse_dialogue": "think",
    # Internet search → memory_recall (in sim, there's no internet)
    "internet_search": "memory_recall",
    "web_search": "memory_recall",
    # Choice / decision → choose (DM campaigns)
    "pick": "choose",
    "select": "choose",
    "decide": "choose",
    "choose_option": "choose",
    "make_choice": "choose",
    "reflect": "think",
    # Inspection / observation → examine
    "inspect": "examine",
    "look": "examine",
    "observe": "examine",
    "look_at": "examine",
    "investigate": "examine",
}

# Guard mutations of TOOL_ALIASES for thread safety. Single-key reads
# (dict.get) are atomic under CPython's GIL, but multi-key mutations
# (update, pop in a loop) need serialization against concurrent readers.
_TOOL_ALIASES_LOCK = threading.RLock()


_log = logging.getLogger(__name__)


# The "Only use tools from the list" reply names these when the agent has them (#1042).
_UNREGISTERED_HINTS = (("memory_recall", "remember"), ("say", "speak aloud"), ("think", "reason"))


class Executor:
    def __init__(
        self,
        tool_registry: ToolRegistry,
        pain_detector: "PainDetector | None" = None,
        tool_pain_bridge: "ToolPainBridge | None" = None,
        permissions: "AgentPermissions | None" = None,
        embodiment: "Embodiment | None" = None,
        *,
        cerebellum: Any | None = None,
        entity_map: Any | None = None,
    ) -> None:
        self.registry = tool_registry
        self._pain_detector = pain_detector
        self._tool_pain_bridge = tool_pain_bridge
        self._permissions = permissions
        # Optional SEM Embodiment reference. Set by build_executor when
        # entity_ref is provided so callers can fetch the body without
        # re-instantiating it. Read pre-wrap (FearGatedExecutor and
        # other wrappers do not proxy this attribute).
        self.embodiment: "Embodiment | None" = embodiment
        # D79 (fix (b), the counting rule's answer): the executor's
        # GENERATION-RELEVANT collaborators are declared constructor
        # fields, and every tool (re)generation this object performs goes
        # through generate_entity_tools() — one helper holding ALL of
        # them, so a new collaborator cannot be forgotten per-site. The
        # pre-fix comment here claimed `_entity_map` was "Set by
        # build_executor"; NOTHING ever assigned it, so Mechanism-B
        # acquisition was a silent no-op through the canonical builder
        # (the third takes-but-does-not-stash miss at this seam, after
        # D77 embodiment= and D79's cerebellum=).
        self._cerebellum: Any | None = cerebellum
        self._entity_map: Any | None = entity_map
        # The LIVE run mode, read at every dispatch (#826): a mode's tool list used to shape only the
        # prompt, so a tool outside it still ran when the model named it. Set by the agent loop, which
        # owns the run mode (``set_mode_source``). Read ONLY through ``effective_operational_mode``.
        self._mode_source: Callable[[], str | None] | None = None
        # The operator's launch grant (`--operational-mode`, #829): when set it IS the operational mode,
        # whatever the loop's run mode says. Read ONLY through ``effective_operational_mode`` (#963).
        self._operational_override: str | None = None
        self._lock = threading.Lock()
        # (tool_name, start_time, invocation_id) or None
        self._running: tuple[str, float, str] | None = None
        # Track alias redirects for experiment analysis
        self.alias_redirects: list[tuple[str, str]] = []
        # Tool usage tracking (Phase 5c)
        self._tools_attempted: list[str] = []
        self._tools_succeeded: list[str] = []
        self._tools_hallucinated: list[str] = []
        self._consecutive_failures: int = 0

    def register_aliases(self, aliases: dict[str, str]) -> None:
        """Register additional tool aliases at runtime.

        Used by DM runtime to map encounter choice names to the choose tool.
        E.g., {"accept_job": "choose", "decline": "choose", "fight": "choose"}
        """
        with _TOOL_ALIASES_LOCK:
            TOOL_ALIASES.update(aliases)

    def remove_aliases(self, names: list[str]) -> None:
        """Remove previously registered runtime aliases."""
        with _TOOL_ALIASES_LOCK:
            for name in names:
                TOOL_ALIASES.pop(name.lower(), None)

    def set_mode_source(self, source: Callable[[], str | None] | None) -> None:
        """Read the live run mode from ``source`` at every dispatch (#826). None clears it."""
        self._mode_source = source

    def set_operational_override(self, mode: str | None) -> None:
        """The operator's launch grant (#829): ``mode`` becomes the operational mode, taking precedence
        over the loop's run mode. None clears it. An unknown name is refused here, at launch, rather than
        failing closed at every dispatch. The name is stored RAW (``"PASSIVE"``, ``"observe"``), as the
        operator gave it (#963 Q2): ``get_mode`` resolves it wherever it is enforced, and the prompt shows
        it as given."""
        from maxim.modes.definitions import get_mode  # noqa: PLC0415 -- runtime layer, read lazily

        if mode is not None and get_mode(mode) is None:
            raise ValueError(f"unknown operational mode {mode!r}")
        self._operational_override = mode

    def effective_operational_mode(self) -> str | None:
        """THE operational mode (#963): in the agent loop and at dispatch, the ONE precedence that dispatch and
        everything the model is shown read (the loop reads it through ``loop_state.operational_mode``; the wrappers
        forward it). Two copies remain outside it: ``bootstrap.build_tool_registry``'s ``get_mode`` callback (#1193)
        and the CLI's ``_current_operational_mode`` / ``_registry_operational_mode`` /
        ``_runtime_mode_switch_allowed`` (#922).

        1. The operator's launch grant, when set (``set_operational_override``).
        2. Else, when a mode source is set (the loop sets one, ``set_mode_source``): its name; ``observe``
           (``DEFAULT_RUN_MODE``) when it gives ``""`` or None; ``passive`` (fail closed) when it gives
           something that is not a string.
        3. Else None: no source and no grant, the one unrestricted case.

        The name is the raw one (``observe``, ``live``, ...), never canonicalised (#963 Q2).
        """
        from maxim.modes.definitions import DEFAULT_RUN_MODE  # noqa: PLC0415 -- runtime layer, read lazily

        if self._operational_override is not None:
            return self._operational_override
        if self._mode_source is None:
            return None
        mode = self._mode_source()
        if isinstance(mode, str):
            return mode or DEFAULT_RUN_MODE
        return DEFAULT_RUN_MODE if mode is None else "passive"

    def _mode_denial(self, tool_name: str) -> str | None:
        """Why the LIVE mode refuses to run *tool_name* (canonical name), or None (#826).

        By capability (``ModeDefinition.dispatch_refusal``): the mode's forbidden tools and the tools
        its capabilities exclude -- for passive, the host-acting ones. The mode's allow-list shapes the
        prompt only. The mode is ``effective_operational_mode()`` (the grant, else the run mode, #963). A
        mode NAME that resolves to no definition fails closed (#829): it is enforced as passive, never as
        unrestricted. Only an executor with no mode source and no grant restricts nothing.
        """
        from maxim.modes.definitions import get_mode  # noqa: PLC0415 -- runtime layer, read lazily

        mode_name = self.effective_operational_mode()
        if mode_name is None:
            return None
        mode_def = get_mode(mode_name)
        if mode_def is None:
            passive = get_mode("passive")
            if passive is None:  # pragma: no cover - the mode table always defines passive
                raise RuntimeError("the passive mode definition is missing")
            refusal = passive.dispatch_refusal(tool_name)
            return f"{refusal} (the mode {mode_name!r} is unknown, so it is enforced as passive)" if refusal else None
        return mode_def.dispatch_refusal(tool_name)

    def _permission_denial(self, tool_name: str, *, deny_only: bool = False) -> str | None:
        """Return the denial reason for *tool_name*, or ``None`` when allowed.

        ``deny_only=True`` applies just the deny half — the pre-alias check,
        where an allow-list must NOT be judged yet (``recall`` → ``memory_recall``).

        The ``kind:<kind>`` selectors in ``AgentPermissions`` need the
        tool's declared ``Tool.kind``; the registry lookup lives here so
        ``agents/permissions.py`` stays registry-free. An unregistered
        name has no kind and is checked by name alone (it fails later at
        ``registry.get`` anyway).
        """
        if not deny_only:
            # The mode is an allow-list judged on the CANONICAL name, like the allow half below.
            mode_denial = self._mode_denial(tool_name)
            if mode_denial is not None:
                return mode_denial
        if self._permissions is None:
            return None
        tool = self.registry._tools.get(tool_name)
        kind = getattr(tool, "kind", None) if tool is not None else None
        if deny_only:
            return self._permissions.denial_reason(tool_name, kind=kind)
        allowed, reason = self._permissions.can_invoke_tool(tool_name, kind=kind)
        if allowed:
            return None
        return reason or "Permission denied."

    def _runnable_tools(self) -> list[str]:
        """Advertised tools the gate would let run -- what an error may suggest (#826, D82; a decoy is never
        suggested, #1042)."""
        return sorted(t for t in self.registry.advertised() if self.permits(t))

    def permits(self, tool_name: str) -> bool:
        """True when the permission gate -- the live mode (#826) and ``AgentPermissions`` -- would
        let *tool_name* run.

        The prompt roster asks this before ADVERTISING a tool: a tool the
        executor refuses at dispatch must not be offered to the model, or the
        model spends turns choosing tools that only ever return a denial
        (bugs ledger D82). No mode source and no permissions → everything permits.
        """
        return self._permission_denial(tool_name) is None

    def execute(self, action: dict[str, Any]) -> ToolOutput:
        """Execute a tool action, returning raw ToolOutput.

        The caller (agent loop) is responsible for converting this to a
        bus ToolResult (agents.bus.ToolResult) with tool_call_id/tool_name/params
        before publishing on the bus.

        If the requested tool name is not registered but matches an entry
        in TOOL_ALIASES, the request is silently redirected to the correct
        tool.  This is logged and tracked in ``self.alias_redirects`` for
        experiment analysis.
        """
        tool_name = action.get("tool_name")
        raw_params = action.get("params")
        params: dict[str, Any] = raw_params if isinstance(raw_params, dict) else {}
        if not isinstance(tool_name, str) or not tool_name:
            return ToolOutput(success=False, error=f"Invalid action: {action!r}")

        self._tools_attempted.append(tool_name)

        # ── Enforced permissions check (O(1) frozenset lookup) ───────
        # Two passes. BEFORE alias resolution: the DENY half only, on the
        # raw name, so a deny that targets the alias source (deny `shell`)
        # still applies. AFTER alias resolution (below, for every call, not
        # only aliased ones): the full check on the canonical name — an
        # allow-list judged on the raw name refused `recall` even when
        # `memory_recall` was allowed (review finding, sandbox-launch).
        denial = self._permission_denial(tool_name, deny_only=True)
        if denial is not None:
            self._tools_hallucinated.append(tool_name)
            self._consecutive_failures += 1
            return ToolOutput(success=False, error=denial)

        # ── Alias resolution ─────────────────────────────────────────
        original_name = tool_name
        if tool_name not in self.registry._tools:
            alias_target = TOOL_ALIASES.get(tool_name.lower())
            if alias_target and alias_target in self.registry._tools:
                import logging

                logging.getLogger(__name__).info(
                    "Tool alias: %s → %s",
                    tool_name,
                    alias_target,
                )
                self.alias_redirects.append((tool_name, alias_target))
                # For choose aliases: inject the original tool name as the option param
                if alias_target == "choose" and "option" not in params:
                    params = {**params, "option": original_name}
                tool_name = alias_target
                # Update the action so downstream (bus, hippocampus) sees
                # the real tool name
                action = {**action, "tool_name": tool_name, "params": params}

        # Full check on the CANONICAL name (deny + allow), aliased or not.
        denial = self._permission_denial(tool_name)
        if denial is not None:
            self._tools_hallucinated.append(tool_name)
            self._consecutive_failures += 1
            return ToolOutput(success=False, error=denial)

        invocation_id = str(uuid.uuid4())

        with self._lock:
            self._running = (tool_name, time.time(), invocation_id)

        # Suppress NAc causal learning during interactive mode — human-directed
        # tool calls would corrupt the causal model with patterns that depend
        # on human presence rather than environmental facts. Rationale and the AUTO caveat live
        # on ``proprioception/pain_bus.py::_human_is_driving``; the longer-term single-home fix is
        # #864. (The old pointer to plans/README.md "Interactive NAc attribution" never resolved.)
        # Unguarded on purpose (#864, review round -- both lenses). This was a bare
        # `except Exception: pass` that left `_suppress_nac = False` on any failure, so a fault
        # here booked human-directed tool calls into NAc's direct-attribution map. It guards the
        # PRIMARY path: `build_pain_bus`'s docstring records that tool-invoked pain reaches NAc
        # through `ToolPainBridge` REGARDLESS of the bus subscriptions, so the three gates in
        # `proprioception/pain_bus.py` cover only out-of-band pain and this one covers the rest.
        # Nothing here can raise (`maxim.simulation` is in-tree; `get_interactive_mode` returns a
        # module global), which is exactly why catching was never protection -- only concealment.
        #
        # Deliberately a second copy of `pain_bus.py::_human_is_driving`'s one-line predicate
        # rather than an import of it: `runtime` reaching into `proprioception` for a `simulation`
        # fact would buy one source of truth with a worse edge. Giving the predicate a single home
        # is #864's open layering half.
        from maxim.simulation.sim_logger import InteractiveMode, get_interactive_mode

        _suppress_nac = get_interactive_mode() == InteractiveMode.ON

        if self._tool_pain_bridge is not None and not _suppress_nac:
            self._tool_pain_bridge.record_tool_start(tool_name, invocation_id, context={"params": params})

        try:
            return self._run_started(tool_name, original_name, params, invocation_id, suppress_nac=_suppress_nac)
        finally:
            # Retire the invocation on EVERY exit, raises included (#851). The completion, embodiment-failure
            # and failure-pain paths each retire it when they run, but a failure whose pain never reached the
            # bridge ran none of them, and its pending entry switched off world-driven body-pain attribution
            # for the rest of the session. Pain dispatch is synchronous, so every one of those paths has run.
            if self._tool_pain_bridge is not None:
                self._tool_pain_bridge.finish_invocation(tool_name, invocation_id)

    def _run_started(
        self,
        tool_name: str,
        original_name: str,
        params: dict[str, Any],
        invocation_id: str,
        *,
        suppress_nac: bool,
    ) -> ToolOutput:
        """Run an invocation ``execute`` has started; ``execute`` retires it however this exits (#851)."""
        # Gate on active status — deactivated scene tools must not execute
        # even if the LLM hallucinates a remembered name from a prior scene.
        # Only check for tools that ARE registered but inactive (scene tools).
        # Non-existent tools fall through to the KeyError path below.
        scene = self.registry.get_tool_scene(tool_name)
        if scene is not None and not self.registry.is_tool_active(tool_name):
            with self._lock:
                self._running = None
            self._consecutive_failures += 1
            error_msg = f"Tool {tool_name!r} is not active (belongs to scene {scene!r})."
            error_msg += f" Available tools: {', '.join(self._runnable_tools())}."
            result = ToolOutput(success=False, error=error_msg)
            self._report_failure(tool_name, invocation_id, result, params)
            return self._stamp_invocation(result, invocation_id, None)

        try:
            tool = self.registry.get(tool_name)
        except KeyError:
            with self._lock:
                self._running = None
            self._tools_hallucinated.append(original_name)
            self._consecutive_failures += 1
            error_msg = f"Tool not registered: {tool_name!r}."
            runnable = self._runnable_tools()
            # find_similar searches deactivated scene tools too (on purpose); a decoy is never suggested (#1042).
            suggestions = [
                t
                for t in self.registry.find_similar(original_name, limit=5)
                if self.permits(t) and getattr(self.registry._tools.get(t), "advertised", True)
            ][:3]
            if suggestions:
                error_msg += f" Did you mean: {', '.join(suggestions)}?"
            # Phase 5d: proactive tool list after repeated failures
            if self._consecutive_failures >= 2:
                error_msg += f" Available tools: {', '.join(runnable)}."
            else:
                error_msg += " Only use tools from the Available Tools list."
                # Name only the agent's own tools (#1042).
                hints = [f"'{t}' to {use}" for t, use in _UNREGISTERED_HINTS if t in runnable]
                if hints:
                    error_msg += f" Use {', '.join(hints)}."
            result = ToolOutput(success=False, error=error_msg)
            self._report_failure(tool_name, invocation_id, result, params)
            return self._stamp_invocation(result, invocation_id, None)

        pressure_before = self._drive_pressure_snapshot()
        try:
            result = tool.run(**params)
        except Exception as e:
            with self._lock:
                self._running = None
            result = ToolOutput(success=False, error=f"Tool {tool_name!r} execution failed: {e}")
            self._report_failure(tool_name, invocation_id, result, params)
            return self._stamp_invocation(result, invocation_id, pressure_before)

        with self._lock:
            self._running = None

        if result.success:
            self._tools_succeeded.append(tool_name)
            self._consecutive_failures = 0
            if self._tool_pain_bridge is not None and not suppress_nac:
                # Embodiment-failure side channel: the tool ran, but
                # the body produced SEM failures (e.g., rusty_sword
                # shattered on slash). Route to direct-attribution
                # path instead of record_tool_complete so NAc learns
                # tool→negative by event_id, not by the broken
                # context-similarity path in _on_embodiment_pain.
                # See ToolPainBridge.record_tool_embodiment_failure
                # and tools/base.py::ToolOutput.side_effects.
                #
                # Wrapped in try/except per CLAUDE.md invariant that
                # bridge callbacks must not crash the agent loop. A
                # bug in NAc.record_outcome, _create_causal_edges, or
                # the reflection path would otherwise propagate up
                # through execute() into the loop controller. Bridge
                # failures degrade learning, not availability.
                try:
                    embodiment_failures: list[dict[str, Any]] | None = None
                    if result.side_effects:
                        raw = result.side_effects.get("embodiment_failures")
                        if isinstance(raw, list) and raw:
                            embodiment_failures = raw
                    if embodiment_failures is not None:
                        self._tool_pain_bridge.record_tool_embodiment_failure(
                            tool_name,
                            invocation_id,
                            embodiment_failures,
                        )
                    else:
                        # D53: a tool that RAN but accomplished nothing
                        # attributable (a motion clamped at a joint limit, a
                        # turn that could not be verified to reach its target)
                        # books NEUTRAL, not POSITIVE. It is neither a success
                        # nor harm, so it must land in neither
                        # get_positive_outcomes nor get_negative_outcomes.
                        # This call site hardcoded success=True, which meant
                        # every completion booked a full POSITIVE causal link.
                        # Read through the SHARED registry parser rather than
                        # hand-rolling a second read of the same key — see
                        # docs/user/tool_side_effects.md.
                        from maxim.decisions.causal_link import Valence as _Val
                        from maxim.runtime.tool_dispatch import read_learning_side_effects

                        _reported = read_learning_side_effects(result).outcome_valence
                        self._tool_pain_bridge.record_tool_complete(
                            tool_name,
                            invocation_id,
                            success=True,
                            outcome_valence=_reported if _reported is not None else _Val.POSITIVE,
                        )

                    # -- Entity acquisition/release (Mechanism B) --
                    if result.side_effects:
                        self._handle_entity_acquisition(result.side_effects)

                except Exception as bridge_err:
                    import logging as _logging

                    _logging.getLogger(__name__).warning(
                        "tool_pain_bridge post-execute attribution failed for %s: %s",
                        tool_name,
                        bridge_err,
                    )
        else:
            self._report_failure(tool_name, invocation_id, result, params)

        return self._stamp_invocation(result, invocation_id, pressure_before)

    def _report_failure(
        self,
        tool_name: str,
        invocation_id: str,
        result: ToolOutput,
        params: dict[str, Any],
    ) -> None:
        """Report a tool failure to pain detector and bridge."""
        if self._pain_detector is not None:
            from maxim.agents.bus import ToolErrorKind

            self._pain_detector.record_tool_error(
                tool_name=tool_name,
                error=result.error or "unknown",
                error_kind=result.error_kind or ToolErrorKind.EXTERNAL_FAILURE,
                context={
                    "params": params,
                    "invocation_id": invocation_id,
                    "metadata": result.metadata,
                },
            )

    def generate_entity_tools(self, entity: Any) -> list[Tool]:
        """Generate + register an entity's affordance tools with EVERY collaborator.

        THE single (re)generation seam for this executor (D79 fix (b)):
        ``build_executor``'s initial generation and Mechanism-B acquisition
        regeneration both call this, so the collaborator list lives in one
        place — forgetting to thread a new one becomes a one-line change
        here instead of a per-site silent no-op (the D77/D79 class:
        ``embodiment=`` and ``cerebellum=`` were each dropped at exactly
        one of the two sites).
        """
        from maxim.embodiment.tool_bridge import generate_tools_for_entity

        return generate_tools_for_entity(
            entity,
            self.registry,
            embodiment=self.embodiment,
            cerebellum=self._cerebellum,
            entity_map=self._entity_map,
        )

    def _handle_entity_acquisition(self, side_effects: dict[str, Any]) -> None:
        """Handle entity_acquired / entity_released side_effects (Mechanism B).

        When an agent picks up an acquirable entity, the entity is
        reparented to the agent's body and its tools are registered.
        When dropped, the entity is reparented back to scene and tools
        are deregistered.
        """
        import logging as _logging

        _log = _logging.getLogger(__name__)

        entity_acquired = side_effects.get("entity_acquired")
        if entity_acquired and self._entity_map is not None and self.embodiment is not None:
            entity = self._entity_map.resolve(entity_acquired)
            if entity is not None and not self._entity_map.is_self(entity):
                # Reparent to agent body root
                entity.reparent(self.embodiment.root)
                self._entity_map.transfer_to_self(entity)
                # Register the acquired entity's tools through the ONE
                # generation seam (D79 fix (b)) — regeneration as a
                # separate weaker call is the defect class this closes
                # (D77 dropped embodiment= here; D79 found cerebellum=
                # undroppable because it was never stashed at all).
                try:
                    tools = self.generate_entity_tools(entity)
                    _log.info("Entity acquired: %s (%d tools registered)", entity_acquired, len(tools))
                except Exception as exc:
                    _log.warning("Failed to register tools for acquired entity %s: %s", entity_acquired, exc)
            elif entity is None:
                _log.debug("entity_acquired: %s not found in entity_map", entity_acquired)

        entity_released = side_effects.get("entity_released")
        if entity_released and self._entity_map is not None and self.embodiment is not None:
            entity = self._entity_map.resolve(entity_released)
            if entity is not None and self._entity_map.is_self(entity):
                # Deregister the entity's tools
                try:
                    for tool_name in list(self.registry.list_all()):
                        if tool_name.startswith(f"{entity.name}_") or tool_name == f"sense_{entity.name}":
                            self.registry.deregister(tool_name)
                except Exception as exc:
                    _log.warning("Failed to deregister tools for released entity %s: %s", entity_released, exc)
                # Reparent back to scene (detach from agent body)
                # Use the embodiment root's parent or create orphan
                entity.reparent(self.embodiment.root.parent or self.embodiment.root)
                self._entity_map.transfer_to_scene(entity)
                _log.info("Entity released: %s", entity_released)

    def _drive_pressure_snapshot(self) -> tuple[tuple[str, float], ...] | None:
        """How hard each of the body's drives is pushing RIGHT NOW, before the action runs.

        The memory-strength encoding record (Phase 2b-ii) needs the pressure the action was taken
        UNDER: read afterwards, an ``eat`` would show its own hunger already relieved. ``None`` when
        no body is attached, and a drive whose value or range cannot be read is simply absent.
        """
        embodiment = self.embodiment
        if embodiment is None or getattr(embodiment, "root", None) is None:
            return None
        from maxim.embodiment.sem import drive_pressure
        from maxim.embodiment.sem import _read_sensor_value
        from maxim.runtime.substrate_proposal import _read_drive_ranges

        # A body-read glitch must never take down the action: this runs OUTSIDE tool.run's guard,
        # and the executor's contract is that a bad invocation is a failed ToolOutput, not a raise.
        try:
            ranges = _read_drive_ranges(self)
            pressures: dict[str, float] = {}
            for entity in embodiment.root.walk():
                for name, spec in (getattr(entity, "drive_specs", {}) or {}).items():
                    # The embodiment's one resolution rule (#1125): a qualified drive
                    # (``arms.thermal``) is read from its modulator, not the root.
                    # ``None`` = missing or non-numeric: skip the drive, never the action.
                    reading = _read_sensor_value(entity, name)
                    if reading is None:
                        continue
                    lo, hi = ranges.get(name, (float("nan"), float("nan")))
                    measured = drive_pressure(spec, reading, lo, hi)
                    if measured is not None:
                        pressures[name] = measured
        except Exception as e:  # noqa: BLE001 - a record must not cost the action
            _log.debug("drive-pressure snapshot failed: %s", e)
            return None
        return tuple(sorted(pressures.items())) if pressures else None

    def _drive_relief(self, result: ToolOutput) -> tuple[tuple[str, float], ...] | None:
        """Per-drive relief this invocation produced, as a fraction of what each drive could give.

        Reads the record-only ``drive_progress_by_drive`` side effect (raw signed units, emitted by
        both relief producers) and normalises it here, where the body's declared ranges are in
        reach. Deliberately NOT read through ``read_learning_side_effects``: that parser is the
        credit path, and this is a record. A drive the action moved AWAY from comfort records 0.0 —
        it touched that drive, and harm is the pain channel's to carry.
        """
        side_effects = result.side_effects or {}
        progress = side_effects.get("drive_progress_by_drive")
        if not isinstance(progress, dict) or not progress:
            return None
        embodiment = self.embodiment
        root = getattr(embodiment, "root", None) if embodiment is not None else None
        if root is None:
            return None
        # Whose body produced it? A tool can act on ANOTHER entity (simulation/tools.py's actor
        # invocation copies that tool's side effects verbatim), and drive names collide across
        # bodies, so an unlabelled or foreign record is not this agent's to normalise.
        producer = side_effects.get("drive_progress_body")
        if producer is not None and producer != getattr(root, "full_path", None):
            _log.debug("drive progress from %r is not this body's; not recorded", producer)
            return None
        from maxim.embodiment.sem import relief_fraction_from_progress
        from maxim.runtime.substrate_proposal import _read_drive_ranges

        try:
            specs: dict[str, Any] = {}
            for entity in root.walk():
                specs.update(getattr(entity, "drive_specs", {}) or {})
            ranges = _read_drive_ranges(self)
            relief: dict[str, float] = {}
            for name, raw in progress.items():
                spec = specs.get(name)
                if spec is None:
                    continue
                lo, hi = ranges.get(name, (float("nan"), float("nan")))
                try:
                    fraction = relief_fraction_from_progress(spec, float(raw), lo, hi)
                except (TypeError, ValueError):
                    continue
                if fraction is not None:
                    relief[name] = fraction
        except Exception as e:  # noqa: BLE001 - a record must not cost the action
            _log.debug("drive-relief normalisation failed: %s", e)
            return None
        return tuple(sorted(relief.items())) if relief else None

    def _stamp_invocation(
        self,
        result: ToolOutput,
        invocation_id: str,
        pressure_before: tuple[tuple[str, float], ...] | None,
    ) -> ToolOutput:
        """Attach what THIS invocation carried: its surprise and the body around it.

        The Rescorla-Wagner error NAc computed for this invocation's outcome travels on the
        ToolOutput (#847), so a capture reads the surprise of the action it captures rather than an
        earlier tool's; the drive pressure it acted under and the relief it produced ride along the
        same way (memory-strength Phase 2b-ii), and so does its pain (Phase 2S-c). The executor is
        the only writer of all four.
        """
        if not isinstance(result, ToolOutput):
            return result
        bridge = self._tool_pain_bridge
        rpe = bridge.pop_invocation_rpe(invocation_id) if bridge is not None else None
        # Popped on EVERY path, like the surprise, so no invocation's pain outlives it (2S-c).
        pain = bridge.pop_invocation_pain(invocation_id) if bridge is not None else None
        relief = self._drive_relief(result)
        if (result.rpe, result.drive_pressure_before, result.drive_relief, result.pain) == (
            rpe,
            pressure_before,
            relief,
            pain,
        ):
            return result
        return dataclasses.replace(
            result, rpe=rpe, drive_pressure_before=pressure_before, drive_relief=relief, pain=pain
        )

    def tool_usage_stats(self) -> dict[str, Any]:
        """Get tool usage statistics for experiment analysis."""
        return {
            "tools_attempted": list(self._tools_attempted),
            "tools_succeeded": list(self._tools_succeeded),
            "tools_hallucinated": list(self._tools_hallucinated),
            "alias_redirects": [(orig, target) for orig, target in self.alias_redirects],
            "total_attempts": len(self._tools_attempted),
            "total_successes": len(self._tools_succeeded),
            "total_hallucinated": len(self._tools_hallucinated),
            "hallucination_rate": (
                len(self._tools_hallucinated) / len(self._tools_attempted) if self._tools_attempted else 0.0
            ),
        }

    def get_running_tool(self) -> tuple[str, float, str] | None:
        """Get the currently running tool info.

        Returns:
            Tuple of (tool_name, start_time, invocation_id) or None.
        """
        with self._lock:
            return self._running
