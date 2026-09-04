#!/usr/bin/env python3
"""Two-AUT-one-world smoke benchmark — the 1.1.4 ship-gate runner (PR 4).

Thin CLI over `maxim.simulation.minecraft_harness` (the glue is importable
and CI-tested; `tests/unit/test_minecraft_harness.py::TestReducedEndToEndSmoke`
runs this exact assembly against the FakeBridgeServer every suite run).

Two modes:

  --fake-bridge          spin up the deterministic in-process world (no
                         Minecraft needed) — the reproducible smoke
  --bridge-ports A,B     dial two REAL Mineflayer bridges (one bot each,
                         scripts/minecraft_bridge/README.md) — the live run,
                         where L11's re-measure obligation attaches

Live-run coherence (PR 3 obligation): do NOT spawn `items/minecraft_bread`
into live runs — it is a SEM-only finite-resource item, a phantom against
real game inventory.

The verdict is NON-VACUOUS by construction (D64): world-modality EC nodes
must exist live AND in each AUT's persisted ec.json after the FULL close,
or the exit code is 1. Writing the verdict under docs/experiments/data/
requires a clean tree (gated-record preflight).

Usage
-----
    python scripts/minecraft_two_aut.py --fake-bridge --ticks 60
    python scripts/minecraft_two_aut.py --bridge-ports 25567,25568 --ticks 240 \
        --json /tmp/minecraft_smoke.json
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from maxim.simulation.minecraft_harness import (  # noqa: E402
    FakeBridgeServer,
    MinecraftSyncPump,
    build_minecraft_aut,
    run_minecraft_aut,
    smoke_verdict,
)


def main(argv: "list[str] | None" = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fake-bridge", action="store_true", help="in-process deterministic world")
    p.add_argument("--bridge-ports", default="", help="two real bridge ports, comma-separated")
    p.add_argument("--ticks", type=int, default=60)
    p.add_argument("--target-hz", type=float, default=4.0)
    p.add_argument("--home", default="", help="agent-home root (default: a fresh tempdir)")
    p.add_argument("--json", default="", help="write the verdict here")
    p.add_argument("--allow-dirty", action="store_true")
    args = p.parse_args(argv)

    provenance = None
    if args.json:
        sys.path.insert(0, str(_REPO_ROOT / "scripts"))
        import maxim
        from _provenance import DirtyTreeError, ProvenanceError, in_process_code_provenance

        try:
            provenance = in_process_code_provenance(
                _REPO_ROOT, maxim.__file__, out_path=args.json, allow_dirty=args.allow_dirty
            )
        except (ProvenanceError, DirtyTreeError) as exc:
            print(f"[FAIL] gated-record preflight: {exc}", file=sys.stderr)
            return 3

    server = None
    if args.fake_bridge:
        server = FakeBridgeServer(seed=42, state_interval_s=0.1)
        ports = [server.port, server.port]
    elif args.bridge_ports:
        ports = [int(x) for x in args.bridge_ports.split(",")]
        if len(ports) != 2:
            print("[FAIL] --bridge-ports wants exactly two ports", file=sys.stderr)
            return 2
    else:
        print("[FAIL] pass --fake-bridge or --bridge-ports", file=sys.stderr)
        return 2

    home = Path(args.home) if args.home else Path(tempfile.mkdtemp(prefix="mc_two_aut_"))
    auts, pumps, threads = [], [], []
    try:
        for name, port in zip(("aut_a", "aut_b"), ports):
            aut = build_minecraft_aut(agent_id=name, bridge_port=port, persistence_dir=str(home / name))
            auts.append(aut)
            pump = MinecraftSyncPump(aut, interval_s=0.5)
            pump.start()
            pumps.append(pump)
        threads = [
            threading.Thread(
                target=run_minecraft_aut,
                kwargs={"aut": a, "max_steps": args.ticks, "target_hz": args.target_hz},
            )
            for a in auts
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        for pump in pumps:
            pump.stop()
        for aut in auts:
            aut.client.close()
        if server is not None:
            server.close()

    verdicts = [smoke_verdict(a) for a in auts]
    green = all(v["world_nodes_live"] > 0 and v["world_nodes_persisted"] > 0 for v in verdicts)
    for v in verdicts:
        print(f"  {v}")
    print(f"SMOKE: {'GREEN' if green else 'RED'} (home: {home})")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "harness": "scripts/minecraft_two_aut.py",
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "provenance": provenance,
                    "mode": "fake-bridge" if args.fake_bridge else "live-bridge",
                    "ticks": args.ticks,
                    "verdicts": verdicts,
                    "green": green,
                },
                indent=2,
            )
        )
        print(f"written: {out}")
    return 0 if green else 1


if __name__ == "__main__":
    raise SystemExit(main())
