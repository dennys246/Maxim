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
        from maxim.memory.percept_trace_buffer import PerceptTraceBuffer

        ptb = PerceptTraceBuffer()
        with open(path) as f:
            data = json.load(f)
        for entry_data in data.get("entries", []):
            ptb.record(
                agent_id=entry_data["agent_id"],
                percept_id=entry_data["percept_id"],
                activation=entry_data["activation_strength"],
            )
        # Advance tick counter to match saved state
        for _ in range(data.get("tick_counter", 0)):
            ptb._tick_counter += 1
        return ptb

    raise ValueError(f"Unknown component type: {name}")


def _resolve_probe(probe_ref: str) -> Any:
    """Resolve a 'module.path:function_name' string to a callable."""
    module_path, func_name = probe_ref.rsplit(":", 1)
    mod = importlib.import_module(module_path)
    func = getattr(mod, func_name)
    return func


def main() -> None:
    parser = argparse.ArgumentParser(description="Persistence round-trip child process")
    parser.add_argument("--config", required=True, help="Path to config JSON")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = json.load(f)

    state_files: dict[str, str] = config["state_files"]
    probe_ref: str = config["probe"]
    result_path: str = config["result_path"]

    # Load all components from saved state
    state: dict[str, Any] = {}
    for name, path in state_files.items():
        try:
            state[name] = _load_component(name, path)
        except Exception as e:
            # Write error as result so parent gets a clear message
            error_result = {"__error__": f"Failed to load {name}: {e}"}
            with open(result_path, "w") as f:
                json.dump(error_result, f)
            print(f"CHILD ERROR: Failed to load {name}: {e}", file=sys.stderr)
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
