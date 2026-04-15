"""Substrate persistence harness — true subprocess round-trip testing.

S3 of simulator_upgrades_plan. Tests that bio-stack state survives a
real serialize → kill → fresh-interpreter → reload cycle.

Two modes:

- **per_component** (legacy S3 path): each bio-system is dumped to its
  own JSON file via its native ``save(path)`` method; the child loads
  each via the matching ``load(path)``. Used by tests that exercise a
  single bio-system in isolation.
- **session_snapshot** (P3.5 Stage 2): all six bio-systems are captured
  into a single ``SessionSnapshot`` envelope via ``capture(strict=True)``
  and written to one JSON file. The child loads the envelope via
  ``SessionSnapshot.from_file`` and calls ``restore_into(strict=True)``
  against freshly-constructed instances. This is the path P4's
  cross-modal mug test will use.

Both modes use the same ``persistence_child.py`` entry point; the config
JSON carries a ``mode`` field that the child branches on.

Probes are specified as "module.path:function_name" strings — NOT closures.
A closure over module state would round-trip fine in-process and explode
in a fresh interpreter, which is the exact bug this harness catches.
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
        # Stage 2 review M4 fold — use the Stage 1 dump() contract so
        # the legacy per-component PTB path round-trips activation
        # strength faithfully (the pre-fix path serialized via
        # snapshot() and lost τ-decay state, which the child then
        # re-recorded at activation=1.0).
        try:
            from maxim.utils.atomic_io import atomic_write_json

            atomic_write_json(str(path), component.dump())
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
            "mode": "per_component",
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


def run_session_round_trip(
    *,
    systems: dict[str, Any],
    probe: str,
    tolerance: float = 0.0,
    timeout_s: float = 60.0,
) -> RoundTripResult:
    """P3.5 Stage 2 — round-trip a full SessionSnapshot through a subprocess.

    Args:
        systems: Mapping of bio-system kind → live instance. Keys must
            be a subset of ``maxim.memory.snapshot.SNAPSHOT_KINDS``. ALL
            provided systems are captured into a single envelope via
            ``SessionSnapshot.capture(strict=True)``; the child constructs
            fresh instances of each kind and calls ``restore_into``.
        probe: ``module.path:function_name`` string. Same shape as
            ``run_round_trip``. Receives ``state`` dict keyed by bio-system
            name, returns a JSON-serializable result.
        tolerance: Numeric tolerance for pre/post comparison.
        timeout_s: Subprocess wall-clock budget.

    Returns:
        ``RoundTripResult`` — the ``state_files`` field contains a single
        entry ``{"session": <path>}``.
    """
    from maxim.memory.snapshot import SNAPSHOT_KINDS, SessionSnapshot

    unknown = set(systems.keys()) - set(SNAPSHOT_KINDS)
    if unknown:
        return RoundTripResult(
            success=False,
            error=f"unknown bio-system kinds in session round-trip: {sorted(unknown)}",
        )

    # Duck-type check at the harness boundary so a wrong-type instance
    # (e.g., systems={"atl": object()}) gets a clean error instead of
    # a confusing AttributeError deep inside SessionSnapshot.capture.
    bad_types = [name for name, obj in systems.items() if not (hasattr(obj, "dump") and hasattr(obj, "load_state"))]
    if bad_types:
        return RoundTripResult(
            success=False,
            error=(
                f"systems values must implement BioSystemSnapshot (dump + load_state); "
                f"missing methods on: {sorted(bad_types)}"
            ),
        )

    try:
        probe_fn = _resolve_probe(probe)
    except Exception as e:
        return RoundTripResult(success=False, error=f"Probe resolution failed: {e}")

    with tempfile.TemporaryDirectory(prefix="maxim_p3_5_s2_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        snapshot_path = tmp_path / "session_snapshot.json"

        # Step 1: capture full session and write single envelope file
        try:
            snapshot = SessionSnapshot.capture(strict=True, **systems)
            snapshot.write(snapshot_path)
        except Exception as e:
            return RoundTripResult(
                success=False,
                error=f"SessionSnapshot.capture/write failed: {e}",
            )

        state_files = {"session": str(snapshot_path)}

        # Step 2: probe live state in-process (state dict is what the
        # probe will see in the child — so use the same shape here)
        try:
            pre_shutdown = probe_fn(systems)
        except Exception as e:
            return RoundTripResult(
                success=False,
                error=f"Pre-shutdown probe failed: {e}",
                state_files=state_files,
            )

        pre_result_path = tmp_path / "pre_result.json"
        post_result_path = tmp_path / "post_result.json"
        config_path = tmp_path / "config.json"

        config = {
            "mode": "session_snapshot",
            "session_path": str(snapshot_path),
            "kinds": sorted(systems.keys()),
            "probe": probe,
            "result_path": str(post_result_path),
        }

        from maxim.utils.atomic_io import atomic_write_json

        atomic_write_json(str(pre_result_path), pre_shutdown)
        atomic_write_json(str(config_path), config)

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
                cwd=str(Path(__file__).resolve().parents[2]),
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
