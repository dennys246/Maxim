"""Substrate persistence harness — true subprocess round-trip testing.

S3 of simulator_upgrades_plan. Tests that bio-stack state survives a
real serialize → kill → fresh-interpreter → reload cycle.

The harness:
  1. Saves bio-stack state (Hippocampus, NAc, ATL, PerceptTraceBuffer) to temp files
  2. Runs a probe function against the live state, capturing the result
  3. Spawns a fresh Python subprocess that loads state and re-runs the probe
  4. Compares pre-shutdown and post-reload probe results within tolerance
  5. Fails explicitly if the child crashes or results diverge

Probes are specified as "module.path:function_name" strings — NOT closures.
A closure over module state would round-trip fine in-process and explode
in a fresh interpreter, which is the exact bug this harness catches.

Usage in test files:
    def test_hippocampus_round_trip(persistence_round_trip):
        result = persistence_round_trip(
            state={"hippocampus": hippo_instance},
            probe="tests.substrate.probes:count_episodes",
        )
        assert result.success
"""

from __future__ import annotations

import importlib
import json
import logging
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RoundTripResult:
    """Result of a persistence round-trip test."""

    success: bool
    pre_shutdown: Any = None
    post_reload: Any = None
    child_returncode: int = 0
    child_stderr: str = ""
    error: str = ""
    state_files: dict[str, str] = field(default_factory=dict)


def _resolve_probe(probe_ref: str) -> Any:
    """Resolve a 'module.path:function_name' string to a callable."""
    if ":" not in probe_ref:
        raise ValueError(f"Probe reference must be 'module.path:function_name', got: {probe_ref}")
    module_path, func_name = probe_ref.rsplit(":", 1)
    mod = importlib.import_module(module_path)
    func = getattr(mod, func_name, None)
    if func is None:
        raise AttributeError(f"Probe function '{func_name}' not found in {module_path}")
    if not callable(func):
        raise TypeError(f"Probe '{probe_ref}' resolved to non-callable: {type(func)}")
    return func


def _save_component(name: str, component: Any, tmp_dir: Path) -> str | None:
    """Save a bio-system component to a temp file. Returns the path or None."""
    path = tmp_dir / f"{name}.json"

    if name == "percept_trace_buffer":
        # PerceptTraceBuffer has no save() — manually serialize snapshot
        try:
            entries = component.snapshot()
            data = {
                "entries": [
                    {
                        "agent_id": e.agent_id,
                        "percept_id": e.percept_id,
                        "tick": e.tick,
                        "activation_strength": e.activation_strength,
                        "registered_at": e.registered_at,
                    }
                    for e in entries
                ],
                "tick_counter": component.current_tick,
            }
            from maxim.utils.atomic_io import atomic_write_json

            atomic_write_json(str(path), data)
            return str(path)
        except Exception as e:
            logger.warning("Failed to save %s: %s", name, e)
            return None

    # Hippocampus, NAc, ATL all have save(path)
    try:
        component.save(str(path))
        return str(path)
    except Exception as e:
        logger.warning("Failed to save %s: %s", name, e)
        return None


def run_round_trip(
    *,
    state: dict[str, Any],
    probe: str,
    tolerance: float = 0.0,
    timeout_s: float = 30.0,
) -> RoundTripResult:
    """Run a full serialize → subprocess → reload → probe round-trip.

    Args:
        state: Dict mapping component names to live instances.
            Supported keys: "hippocampus", "nac", "atl", "percept_trace_buffer"
        probe: Module reference string "module.path:function_name".
            The function receives a dict of {name: loaded_component} and
            returns a JSON-serializable result.
        tolerance: Numeric tolerance for comparing pre/post results.
            0.0 means exact match. For floats, uses abs(a - b) <= tolerance.
        timeout_s: Max seconds to wait for child subprocess.

    Returns:
        RoundTripResult with success/failure, pre/post values, and diagnostics.
    """
    # Validate probe is resolvable in THIS process first
    try:
        probe_fn = _resolve_probe(probe)
    except Exception as e:
        return RoundTripResult(success=False, error=f"Probe resolution failed: {e}")

    with tempfile.TemporaryDirectory(prefix="maxim_s3_") as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Step 1: Save all components
        state_files: dict[str, str] = {}
        for name, component in state.items():
            if component is None:
                continue
            saved = _save_component(name, component, tmp_path)
            if saved:
                state_files[name] = saved

        if not state_files:
            return RoundTripResult(
                success=False,
                error="No components were saved — nothing to round-trip",
                state_files=state_files,
            )

        # Step 2: Run probe against live state
        try:
            pre_shutdown = probe_fn(state)
        except Exception as e:
            return RoundTripResult(
                success=False,
                error=f"Pre-shutdown probe failed: {e}",
                state_files=state_files,
            )

        # Step 3: Write probe result and config for child
        pre_result_path = tmp_path / "pre_result.json"
        post_result_path = tmp_path / "post_result.json"
        config_path = tmp_path / "config.json"

        config = {
            "state_files": state_files,
            "probe": probe,
            "result_path": str(post_result_path),
        }

        from maxim.utils.atomic_io import atomic_write_json

        atomic_write_json(str(pre_result_path), pre_shutdown)
        atomic_write_json(str(config_path), config)

        # Step 4: Spawn child subprocess
        child_cmd = [
            sys.executable,
            "-m",
            "tests.substrate.persistence_child",
            "--config",
            str(config_path),
        ]

        try:
            result = subprocess.run(
                child_cmd,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=str(Path(__file__).resolve().parents[2]),  # repo root
            )
        except subprocess.TimeoutExpired:
            return RoundTripResult(
                success=False,
                pre_shutdown=pre_shutdown,
                error=f"Child subprocess timed out after {timeout_s}s",
                state_files=state_files,
            )

        if result.returncode != 0:
            return RoundTripResult(
                success=False,
                pre_shutdown=pre_shutdown,
                child_returncode=result.returncode,
                child_stderr=result.stderr[-2000:] if result.stderr else "",
                error=f"Child subprocess exited with code {result.returncode}",
                state_files=state_files,
            )

        # Step 5: Read child's probe result and compare
        if not post_result_path.exists():
            return RoundTripResult(
                success=False,
                pre_shutdown=pre_shutdown,
                child_returncode=result.returncode,
                child_stderr=result.stderr[-2000:] if result.stderr else "",
                error="Child did not write result file",
                state_files=state_files,
            )

        try:
            with open(post_result_path) as f:
                post_reload = json.load(f)
        except Exception as e:
            return RoundTripResult(
                success=False,
                pre_shutdown=pre_shutdown,
                error=f"Failed to read child result: {e}",
                state_files=state_files,
            )

        # Step 6: Compare results
        match = _compare_results(pre_shutdown, post_reload, tolerance)

        return RoundTripResult(
            success=match,
            pre_shutdown=pre_shutdown,
            post_reload=post_reload,
            child_returncode=result.returncode,
            child_stderr=result.stderr[-2000:] if result.stderr else "",
            error="" if match else f"Results diverged: pre={pre_shutdown!r} post={post_reload!r}",
            state_files=state_files,
        )


def _compare_results(pre: Any, post: Any, tolerance: float) -> bool:
    """Compare pre-shutdown and post-reload probe results."""
    if tolerance == 0.0:
        return pre == post

    # Recursive numeric tolerance comparison
    return _compare_recursive(pre, post, tolerance)


def _compare_recursive(a: Any, b: Any, tol: float) -> bool:
    """Recursively compare with numeric tolerance."""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= tol

    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(_compare_recursive(a[k], b[k], tol) for k in a)

    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        return all(_compare_recursive(x, y, tol) for x, y in zip(a, b))

    return a == b
