"""Subprocess entry point for persistence round-trip testing.

S3 of simulator_upgrades_plan. This module is invoked as:
    python -m tests.substrate.persistence_child --config /path/to/config.json

It runs in a FRESH Python interpreter — no shared state with the parent.
If any bio-system component relies on closures, module-level singletons,
or transient in-memory state, deserialization will fail here (which is
the exact bug this harness is designed to catch).

The config JSON contains:
    state_files: dict mapping component names to saved JSON file paths
    probe: "module.path:function_name" string
    result_path: where to write the probe output
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from typing import Any


def _load_component(name: str, path: str) -> Any:
    """Load a bio-system component from a saved JSON file."""
    if name == "hippocampus":
        from maxim.memory.hippocampus import Hippocampus, HippocampusConfig

        h = Hippocampus(config=HippocampusConfig())
        h.load(path)
        return h

    if name == "nac":
        from maxim.decisions.nac import NAc

        n = NAc()
        n.load(path)
        return n

    if name == "atl":
        from maxim.memory.atl import ATL

        a = ATL()
        a.load(path)
        return a

    if name == "percept_trace_buffer":
        # Stage 2 review M4 fold — pre-fix this path used buf.record()
        # to replay entries, which lost the τ-decay activation values
        # (every entry came back at activation=1.0). With Stage 1's
        # PerceptTraceBuffer.dump/load_state in place, the legacy
        # per-component path now uses the same load_state contract as
        # the session_snapshot path. Backwards-compatible: it accepts
        # both the new Stage 1 dump shape AND the pre-Stage-1 minimal
        # shape that ``persistence_harness._save_component`` writes
        # for PTB (entries + tick_counter only).
        from maxim.memory.percept_trace_buffer import PerceptTraceBuffer

        ptb = PerceptTraceBuffer()
        with open(path) as f:
            data = json.load(f)
        ptb.load_state(data)
        return ptb

    raise ValueError(f"Unknown component type: {name}")


def _resolve_probe(probe_ref: str) -> Any:
    """Resolve a 'module.path:function_name' string to a callable."""
    module_path, func_name = probe_ref.rsplit(":", 1)
    mod = importlib.import_module(module_path)
    func = getattr(mod, func_name)
    return func


def _build_fresh_instance(kind: str) -> Any:
    """Construct a default-wired instance for one of the SNAPSHOT_KINDS.

    P3.5 Stage 2 — used by the session_snapshot child path. Each instance
    is constructed with its zero-arg / default-config form so the
    SessionSnapshot.restore_into() call can mutate state without
    overwriting wires (the load_state contract).
    """
    if kind == "atl":
        from maxim.memory.atl import ATL

        return ATL()
    if kind == "hippocampus":
        from maxim.memory.hippocampus import Hippocampus, HippocampusConfig

        return Hippocampus(config=HippocampusConfig())
    if kind == "nac":
        from maxim.decisions.nac import NAc

        return NAc()
    if kind == "scn":
        from maxim.time.scn import SCN

        return SCN()
    if kind == "percept_trace_buffer":
        from maxim.memory.percept_trace_buffer import PerceptTraceBuffer

        return PerceptTraceBuffer()
    if kind == "cross_layer_graph":
        from maxim.memory.cross_layer import CrossLayerGraph

        return CrossLayerGraph()
    raise ValueError(f"Unknown bio-system kind: {kind}")


def _load_session_snapshot(session_path: str, kinds: list[str]) -> dict[str, Any]:
    """Construct fresh instances and restore the SessionSnapshot into them."""
    from maxim.memory.snapshot import SessionSnapshot

    snapshot = SessionSnapshot.from_file(session_path)
    instances: dict[str, Any] = {kind: _build_fresh_instance(kind) for kind in kinds}
    snapshot.restore_into(strict=True, **instances)
    return instances


def main() -> None:
    parser = argparse.ArgumentParser(description="Persistence round-trip child process")
    parser.add_argument("--config", required=True, help="Path to config JSON")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = json.load(f)

    mode: str = config.get("mode", "per_component")
    probe_ref: str = config["probe"]
    result_path: str = config["result_path"]

    # Load components — branches on mode
    state: dict[str, Any] = {}
    if mode == "session_snapshot":
        try:
            state = _load_session_snapshot(config["session_path"], config["kinds"])
        except Exception as e:
            error_result = {"__error__": f"Failed to load session snapshot: {e}"}
            with open(result_path, "w") as f:
                json.dump(error_result, f)
            print(f"CHILD ERROR: Failed to load session snapshot: {e}", file=sys.stderr)
            sys.exit(1)
    elif mode == "per_component":
        state_files: dict[str, str] = config["state_files"]
        for name, path in state_files.items():
            try:
                state[name] = _load_component(name, path)
            except Exception as e:
                error_result = {"__error__": f"Failed to load {name}: {e}"}
                with open(result_path, "w") as f:
                    json.dump(error_result, f)
                print(f"CHILD ERROR: Failed to load {name}: {e}", file=sys.stderr)
                sys.exit(1)
    else:
        error_result = {"__error__": f"Unknown mode: {mode}"}
        with open(result_path, "w") as f:
            json.dump(error_result, f)
        print(f"CHILD ERROR: Unknown mode: {mode}", file=sys.stderr)
        sys.exit(1)

    # Resolve and run probe
    try:
        probe_fn = _resolve_probe(probe_ref)
        result = probe_fn(state)
    except Exception as e:
        error_result = {"__error__": f"Probe failed: {e}"}
        with open(result_path, "w") as f:
            json.dump(error_result, f)
        print(f"CHILD ERROR: Probe failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Write result
    with open(result_path, "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
