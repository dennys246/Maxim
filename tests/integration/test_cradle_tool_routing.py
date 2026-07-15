"""Cradle Arm-A tool-routing smoke (slow, opt-in).

PR D regression smoke for docs/plans/archive/cradle_activation_fixes.md (Finding B).
The pre-fix cradle scenario burned ~60% of turns on silent
``Tool not registered: 'respond'`` errors because:

  1. prompt_builder told the LLM to call ``respond`` (conversational mode), and
  2. generative_runner deregistered ``respond`` / ``say`` for embodied arcs.

PR D propagates ``LLMRequest.is_embodied`` from the producer (LLMWorker) and
gates the conversational prompt sections instead of mutating the registry,
which lets the prompt steer the LLM toward the body-prefixed affordances.

This smoke fires the cradle scenario end-to-end via the CLI subprocess and
parses the resulting ``actions.jsonl`` for ``Tool not registered: 'respond'``
entries. Asserts the count is zero (or strictly below a threshold) over a
short turn budget.

**Run manually** before re-firing Exp 37::

    MAXIM_CRADLE_SMOKE_LLM=qwen2.5-14b-instruct pytest \
        tests/integration/test_cradle_tool_routing.py -v -m slow

The test is skipped by default — it requires (a) a working LLM router
configuration via env / config (peer.yml routing to a leader serving the
named profile, or a local model the named profile can resolve) and (b)
the slow marker to be selected.

**Run from a peer machine, NOT from the leader** — the harness lesson
in CLAUDE.md (cradle smoke 2026-06-04) explains why running cradle sims
on the same machine as the LLM server causes role-detection and port
collision cascades. This test sets ``--home-dir`` to a tmp tree so it
doesn't collide with any sim state in ``~/.maxim/``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _model_env() -> str | None:
    """Return the configured cradle-smoke LLM model name, or None to skip."""
    return os.environ.get("MAXIM_CRADLE_SMOKE_LLM")


def _count_unregistered_respond_errors(actions_jsonl: Path) -> int:
    """Count ``Tool not registered: 'respond'`` entries in an actions log.

    Tolerant of partial / malformed lines (returns count of strictly
    matching entries; everything else is ignored).
    """
    if not actions_jsonl.exists():
        return 0
    count = 0
    with actions_jsonl.open() as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue
            # The error surfaces in multiple shapes depending on dispatch
            # path; check the most common ones.
            err = rec.get("error") or rec.get("blocked_reason") or ""
            if isinstance(err, str) and "Tool not registered: 'respond'" in err:
                count += 1
    return count


@pytest.mark.slow
@pytest.mark.skipif(
    _model_env() is None,
    reason="Set MAXIM_CRADLE_SMOKE_LLM=<profile> to enable the cradle Arm-A smoke",
)
def test_cradle_arm_a_no_respond_loop(tmp_path: Path) -> None:
    """Fire the cradle scenario through the CLI and assert the action log
    contains zero 'Tool not registered: respond' entries.

    Uses a short turn budget (8) to keep wall time bounded; a real Arm-A
    Exp 37 run uses more. Eight turns is enough to surface the pre-fix
    silent loop pattern (the smoke trace in cradle_activation_fixes.md
    Finding B saw 8 such errors in 12 turns).
    """
    model = _model_env()
    assert model is not None  # mypy

    # ``--home-dir`` redirects ALL Maxim data (sessions, memory, logs) to
    # the given directory. The sim writes per-session artifacts to
    # ``<home_dir>/sim_reports/<session_id>/`` so we can scope the search
    # to one tmp tree. (Pre-fix this test passed ``--session-dir`` which
    # doesn't exist as a maxim CLI flag — the subprocess would argparse-
    # fail before producing any data, silently passing the assertion.)
    home_dir = tmp_path / "maxim_home"
    home_dir.mkdir()

    cmd = [
        sys.executable,
        "-m",
        "maxim",
        "--sim",
        "cradle",
        "--embodiment",
        "bodies/infant_humanoid",
        "--language-model",
        model,
        "--sim-max-turns",
        "8",
        "--interactive",
        "false",
        "--home-dir",
        str(home_dir),
    ]

    env = {**os.environ, "MAXIM_LOG_FILE": str(home_dir / "maxim.jsonl")}

    # 5-minute wall budget — generous for slow local models, well within
    # the slow-test contract.
    try:
        result = subprocess.run(cmd, env=env, timeout=300, check=False)
    except subprocess.TimeoutExpired:
        pytest.fail(f"Cradle smoke timed out (>300s); home_dir at {home_dir}")

    if result.returncode != 0:
        pytest.fail(
            f"maxim --sim cradle exited with code {result.returncode}. "
            f"home_dir at {home_dir}; stderr/stdout merged at "
            f"{home_dir / 'maxim.jsonl'}. The non-zero exit is a hard failure "
            f"(distinct from the 'Tool not registered' regression this test "
            f"specifically guards) — investigate before re-enabling the smoke."
        )

    # Find the actions.jsonl the sim wrote. Default layout under
    # ``--home-dir`` is ``<home_dir>/sim_reports/<session_id>/actions.jsonl``;
    # rglob covers any alternate placement.
    candidates = list(home_dir.rglob("actions.jsonl"))
    if not candidates:
        pytest.fail(
            f"No actions.jsonl produced under {home_dir}; the sim may not "
            f"have reached the per-session writer. Inspect "
            f"{home_dir / 'maxim.jsonl'} for the failure mode."
        )
    actions = candidates[0]

    bad = _count_unregistered_respond_errors(actions)
    assert bad == 0, (
        f"Found {bad} 'Tool not registered: respond' entries in {actions}. "
        "PR D's prompt-gating regressed — the LLM is being told to call "
        "respond by the prompt builder. See "
        "docs/plans/archive/cradle_activation_fixes.md Finding B."
    )
