"""Pain-signal executor wrappers — two-layer predictive aversion.

Two composable wrappers form the pain layering:

- **AnticipatoryPainExecutor** (Layer 1): fires BEFORE execute, based
  on NAc prediction + innate priors. The signal is a felt anticipation
  that flows into the AUT's context via hippocampus → LLM prompt.
- **PainInterceptorExecutor** (Layer 2): fires AFTER execute, based on
  actual paths touched. This is the ground-truth signal that trains
  NAc to produce better anticipatory predictions.

Both share the same sensitive-path prior table and path-extraction
helpers, so they agree on what counts as painful — Layer 1 predicts
it, Layer 2 confirms it. The two classes live in the same file so
the ordering contract is obvious.

Canonical executor stack (outermost first):

    executor = Executor(registry)
    executor._tool_pain_bridge = bridge                       # NAc learning
    executor = PainInterceptorExecutor(executor, ...)         # Layer 2 (after)
    executor = AnticipatoryPainExecutor(executor, assessor)   # Layer 1 (before)
    executor = FearGatedExecutor(executor, fear_agent)        # safety gate
"""

from __future__ import annotations

import logging
import time
from typing import Any

from maxim.decisions.causal_link import Valence
from maxim.proprioception.pain import PainType
from maxim.proprioception.perceived_pain import (
    DEFAULT_SENSITIVE_PRIORS,
    SensitivePathPrior,
    _path_matches_prior,
    extract_paths_from_params,
)

logger = logging.getLogger(__name__)


class PainInterceptorExecutor:
    """Executor wrapper that fires consequence-pain on sensitive-path access.

    Delegates to an inner executor via ``execute()``. Before and after
    each call, scans the action's params for paths that match the
    configured sensitive-path priors. On a match, publishes a
    ``PainType.EXTERNAL_SIGNAL`` signal through the PainBus.
    """

    def __init__(
        self,
        inner: Any,
        *,
        pain_bus: Any = None,
        priors: tuple[SensitivePathPrior, ...] = DEFAULT_SENSITIVE_PRIORS,
    ) -> None:
        self._inner = inner
        self._pain_bus = pain_bus
        self._priors = priors
        self._events: list[dict[str, Any]] = []

    # ── Delegation: make wrapper transparent to callers ──────────────────

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the inner executor.

        Keeps ``_tool_pain_bridge``, ``get_last_rpe``, etc. reachable
        when this wrapper sits in the middle of an executor stack.
        """
        return getattr(self._inner, name)

    # ── Core ─────────────────────────────────────────────────────────────

    def execute(self, action: dict[str, Any]) -> Any:
        """Run the inner executor; fire consequence-pain on success match."""
        result = self._inner.execute(action)

        tool_name = str(action.get("tool_name") or "")
        if not tool_name:
            return result

        params = action.get("params") if isinstance(action.get("params"), dict) else {}
        accesses = extract_paths_from_params(tool_name, params)
        if not accesses:
            return result

        # Only fire pain on successful execution — failed calls are
        # their own kind of feedback and are handled elsewhere.
        success = getattr(result, "success", True)
        if not success:
            return result

        # Match accesses against priors; fire pain for the strongest match.
        best_prior: SensitivePathPrior | None = None
        best_intensity = 0.0
        best_access: Any = None
        for access in accesses:
            for prior in self._priors:
                if _path_matches_prior(access.path, prior, access.operation):
                    if prior.intensity > best_intensity:
                        best_intensity = prior.intensity
                        best_prior = prior
                        best_access = access

        if best_prior is None:
            return result

        path_list = [a.path for a in accesses]
        from maxim.reactions.types import Reaction, ReactionContext, TraceSnapshot

        now = time.time()
        reaction = Reaction(
            kind="pain",
            intensity=round(best_intensity, 3),
            valence=Valence.NEGATIVE,
            timestamp=now,
            source=f"pain_interceptor:{PainType.EXTERNAL_SIGNAL.value}",
            context=ReactionContext(
                bindings={
                    "paths": TraceSnapshot(percept_id=",".join(path_list[:5])),
                }
                if path_list
                else {},
            ),
        )
        self._events.append(
            {
                "timestamp": now,
                "tool_name": tool_name,
                "intensity": reaction.intensity,
                "paths": path_list,
            }
        )

        if self._pain_bus is not None:
            try:
                self._pain_bus.reaction_bus.publish(reaction)
            except Exception as e:
                logger.debug("ReactionBus publish failed: %s", e)

        # Sim-visibility trace.
        try:
            from maxim.simulation.sim_logger import sim_pain

            sim_pain(
                f"consequence ({best_access.operation if best_access else '?'} {tool_name})",
                best_intensity,
                paths=path_list[:2],
            )
        except Exception:
            pass

        logger.info(
            "Consequence pain %.2f fired: tool=%s op=%s paths=%s",
            best_intensity,
            tool_name,
            best_access.operation if best_access else "?",
            path_list,
        )
        return result

    @property
    def events(self) -> list[dict[str, Any]]:
        """Read back consequence-pain events (for reports / introspection)."""
        return list(self._events)


class AnticipatoryPainExecutor:
    """Executor wrapper that fires anticipated pain before execute (Layer 1).

    Holds a ``PerceivedPainAssessor`` and calls ``assess(action)``
    before delegating to the inner executor. The assessor publishes
    to the PainBus internally; this wrapper just threads the call.
    """

    def __init__(self, inner: Any, assessor: Any) -> None:
        self._inner = inner
        self._assessor = assessor

    def __getattr__(self, name: str) -> Any:
        """Transparent delegation for non-execute methods."""
        return getattr(self._inner, name)

    def execute(self, action: dict[str, Any]) -> Any:
        """Assess anticipated pain, then delegate to inner executor."""
        if self._assessor is not None:
            try:
                self._assessor.assess(action)
            except Exception as e:
                logger.debug("Perceived-pain assessment failed: %s", e)
        return self._inner.execute(action)

    @property
    def events(self) -> list[dict[str, Any]]:
        """Read back anticipation events (for reports / introspection)."""
        return list(self._assessor.events) if self._assessor else []
