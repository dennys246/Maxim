"""Exp 44 counterfactual harness — Pass 1: live paired-prompt capture.

Wraps ``LLMWorker._prompt_builder.build_prompt`` so that at every LLM
submission we compose the prompt TWICE from the *same live request* —

  * arm A (full)     — the prompt that actually runs, substrate ON
  * arm B (ablated)  — the substrate carriers nulled, everything else identical

— and log both strings + a world-state digest to JSONL. Only arm A executes;
arm B is logged for the offline temp-0 re-query (``rerun_ablated_offline.py``).

**Why compose both live instead of reconstructing offline:** the live
``LLMRequest`` is the faithful object — no serialization round-trip, no risk of
a dropped field. The only thing deferred to "offline" is the LLM *re-query*.

**The ablation is scoped to the carriers that transmit the LEARNED ORIENT** to
the LLM prompt — the three that the sanctioned env flags null:

  | env flag it mirrors                    | carrier nulled                       |
  |----------------------------------------|--------------------------------------|
  | MAXIM_DISABLE_CLUSTER_BIAS_ANNOTATION  | ctx.cluster_bias_annotations = None  |
  | MAXIM_ENABLE_BODY_STATE_PROMPT (→off)  | ctx.body_state = None                |
  | MAXIM_DISABLE_COACH_BODY_LAYERS        | acting_coach.body_state_layers=False |

The Wire-1 variance annotation (``_risk_profile`` → tool-description text) is a
SEPARATE risk-sensitivity mechanism, NOT orient, so it is deliberately held
CONSTANT across arms (see the design doc). Coach Layers 1+3 (the non-substrate
default exploration policy) also stay in both arms — only Layers 2+4 (pain
anticipation + drive modulation) are substrate-derived and removed.

Zero core edits: this is a harness monkeypatch installed around the worker.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import threading
from pathlib import Path
from typing import Any

# Marker for the cluster-bias (Wire-A) section — the exact header the composer
# emits (src/maxim/prompts/cluster_bias_annotation.py). If this string survives
# ablation, cluster biases leaked into arm B.
CLUSTER_BIAS_MARKER = "Substrate associations from prior experience"


def _ablate(request: Any) -> Any:
    """Return a copy of ``request`` with the orient-substrate carriers removed.

    Does NOT mutate the original request/context (arm A must be untouched).
    """
    ctx = copy.copy(request.context)  # shallow copy; setattr on the copy only
    ctx.cluster_bias_annotations = None
    ctx.body_state = None

    coach = getattr(request, "acting_coach", None)
    if coach is not None:
        # ActingCoachConfig is frozen — replace to disable Layers 2+4.
        coach = dataclasses.replace(coach, body_state_layers=False)

    return dataclasses.replace(request, context=ctx, acting_coach=coach)


def assert_fully_ablated(prompt_ablated: str, *, arm_a_body_state: str | None) -> None:
    """Fail LOUD if any orient-substrate content leaked into the ablated prompt.

    This is what makes field-nulling a *verifiable* ablation rather than a
    band-aid: if a carrier is missed, arm B silently keeps substrate content and
    the whole experiment is invalid. Better to crash the run.
    """
    leaks: list[str] = []
    if CLUSTER_BIAS_MARKER in prompt_ablated:
        leaks.append("cluster_bias_annotations")
    # body_state is free-form text; the exact arm-A string must not survive.
    if arm_a_body_state and arm_a_body_state.strip() and arm_a_body_state in prompt_ablated:
        leaks.append("body_state")
    if leaks:
        raise AssertionError(
            f"ablation leaked substrate carriers into arm B: {leaks}. "
            "Update _ablate() (a carrier was added upstream) before trusting this run."
        )


def _digest(request: Any, prompt_full: str) -> dict[str, Any]:
    """A compact world-state fingerprint for joining/slicing offline."""
    ctx = request.context
    return {
        "triggering_input": getattr(request, "triggering_input", None),
        "available_tools": sorted(getattr(request, "available_tools", []) or []),
        "body_state": getattr(ctx, "body_state", None),
        "has_cluster_bias": getattr(ctx, "cluster_bias_annotations", None) is not None,
        "prompt_full_sha1": hashlib.sha1(prompt_full.encode("utf-8")).hexdigest()[:12],
    }


class PairedPromptCapture:
    """Installs the build_prompt wrapper and appends paired rows to a JSONL."""

    def __init__(self, log_path: str | Path) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._n = 0
        self._lock = threading.Lock()
        self._orig: Any = None
        self._builder: Any = None

    def install(self, worker: Any) -> "PairedPromptCapture":
        """Wrap ``worker._prompt_builder.build_prompt``. Returns self."""
        builder = worker._prompt_builder
        self._builder = builder
        self._orig = builder.build_prompt

        def _wrapped(request: Any) -> str:
            prompt_full = self._orig(request)
            try:
                ablated_req = _ablate(request)
                prompt_ablated = self._orig(ablated_req)
                arm_a_body_state = getattr(request.context, "body_state", None)
                assert_fully_ablated(prompt_ablated, arm_a_body_state=arm_a_body_state)
                self._append(request, prompt_full, prompt_ablated)
            except AssertionError:
                raise  # ablation leak — fail the run, do not swallow
            except Exception as e:  # capture must never break the live run
                self._append_error(e)
            return prompt_full  # arm A is what executes

        builder.build_prompt = _wrapped
        return self

    def uninstall(self) -> None:
        if self._builder is not None and self._orig is not None:
            self._builder.build_prompt = self._orig

    def __enter__(self) -> "PairedPromptCapture":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.uninstall()

    def _append(self, request: Any, prompt_full: str, prompt_ablated: str) -> None:
        with self._lock:
            row = {
                "decision_id": self._n,
                "world_state": _digest(request, prompt_full),
                "prompt_full": prompt_full,
                "prompt_ablated": prompt_ablated,
            }
            self._n += 1
            with self.log_path.open("a") as f:
                f.write(json.dumps(row) + "\n")

    def _append_error(self, e: Exception) -> None:
        with self._lock:
            with self.log_path.open("a") as f:
                f.write(json.dumps({"decision_id": self._n, "capture_error": repr(e)}) + "\n")
            self._n += 1


def install_capture(worker: Any, log_path: str | Path) -> PairedPromptCapture:
    """Convenience: install and return the capture handle.

    Usage from an Exp 44 runner (after the AUT's LLMWorker is constructed,
    before the loop starts)::

        from scripts.exp44.capture_paired_prompts import install_capture
        cap = install_capture(aut_worker, "data/exp44/paired_prompts.jsonl")
        ...  # run the status-quo (flags-ON) episode
        cap.uninstall()
    """
    return PairedPromptCapture(log_path).install(worker)
