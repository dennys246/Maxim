# Robot Setup

## Overview

Maxim is **robot-agnostic** at every layer above `src/maxim/hardware/`. The agent loop, default network, planning, memory, and SEM (Sensor-Entity-Modulator) systems all consume an abstract `RobotController` interface — they have no SDK assumptions. Reachy Mini is the reference implementation that ships in-tree, but adding Atlas, Spot, a custom drone, or a fully simulated robot is a 3-step plugin process (see [Adding a New Robot](#adding-a-new-robot) below).

This page covers Reachy Mini setup in detail. The same principles (controller class + tool registration + optional SEM template) apply to any robot.

## Reachy Mini Setup

Maxim connects to Reachy Mini over your local network using Zenoh peer discovery. The robot runs a daemon that Maxim communicates with.

## Requirements
- Reachy Mini on the same LAN or Wi-Fi network as your computer
- SSH access to the robot (default user: pollen)
- Pollen Robotics SDK installed (link: https://github.com/pollen-robotics/reachy_mini/blob/develop/docs/SDK/installation.md)

## Starting the Robot Daemon

SSH into your Reachy:
```bash
ssh pollen@<REACHY_IP>
```

Stop the default daemon and start with wireless support:
```bash
sudo systemctl stop reachy-mini-daemon
source /venvs/mini_daemon/bin/activate
python -m reachy_mini.daemon.app.main --wireless-version --no-localhost-only
```

The daemon runs on:
- Port 7447 (Zenoh) — robot communication
- Port 8443 (WebRTC) — video streaming

## Connecting Maxim

```bash
# Default (auto-discovers "reachy_mini" on network)
maxim --mode agentic

# Custom robot name
maxim --robot-name my_reachy

# Increase connection timeout
maxim --timeout 60
```

## Headless Mode (No Robot)
Maxim works without a robot connected. The agent runtime, LLM, planning, and coding tools all work headless. Robot-specific tools (MoveTool, TrackTargetTool) return stub responses.

```bash
maxim --mode agentic --language-model mistral-7b
```

## Running Diagnostics
```bash
# Built-in diagnostic tool
maxim-diagnostics --host <REACHY_IP>

# Or the script directly
python scripts/check_reachy_connection.py --host <REACHY_IP>
```

Diagnostics check:
- Network connectivity
- Zenoh peer discovery
- WebRTC video stream
- Joint state reading
- Motor control

## Network Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Can't find robot | Wrong network | Ensure same LAN/Wi-Fi |
| Port 8443 refused | Daemon not running | Restart daemon (see above) |
| Port 7447 refused | System daemon blocking | `systemctl stop reachy-mini-daemon` |
| Connection timeout | Firewall | Check firewall allows ports 7447, 8443 |
| Intermittent drops | Wi-Fi instability | Use wired ethernet if possible |

## RTSP Bridge (Optional)
For external video streaming:
```bash
python scripts/rtsp_bridge.py
```
See `docs/mediaMTX.md` for MediaMTX relay setup.

---

## Adding a New Robot

Reachy Mini was the first robot Maxim supported, but the architecture was always designed to host more. The hardware abstraction lives in [`src/maxim/hardware/`](../../src/maxim/hardware/) and the integration points are intentionally narrow:

### What's robot-agnostic (no changes needed for new robots)

- **`embodiment/`** — the SEM protocol (Sensor / Entity / Modulator) is pure abstraction. Sensors return readings, modulators expose affordances, entities compose into trees. No SDK assumptions.
- **`runtime/`, `agents/`, `memory/`, `default_network/`, `planning/`** — these consume the abstract `RobotController` interface and the SEM graph. They never reach into hardware specifics.
- **`hardware/controller.py::RobotController`** — abstract base class. Implement it for your SDK.
- **`hardware/registry.py::RobotRegistry`** — singleton that auto-discovers plugins via the `maxim.robots` entry-point group.
- **`hardware/capabilities.py`** — `RobotCapabilities` exposes generic flags (HEAD_PAN, ARMS, GRIPPER, WHEELS, LEGS, VIDEO, AUDIO_INPUT, AUDIO_OUTPUT) plus a free-form `custom: frozenset[str]` for robot-specific advertisements.
- **`hardware/controller.py::MotionTarget`** — generic head/body fields plus an `extras: dict[str, float]` for robot-specific joints (Reachy antennas, Atlas arm yaw, Spot foot pressure).

### The 3-step pattern

1. **Implement `RobotController`** in your own package (e.g. `pymaxim-atlas`):

   ```python
   # src/maxim_atlas/controller.py
   from maxim.hardware import RobotController, MotionTarget, PixelTarget, RobotCapabilities, MotionCapability, StreamCapability

   class AtlasController(RobotController):
       @property
       def robot_type(self) -> str:
           return "atlas"

       def connect(self, timeout=30.0): ...
       def disconnect(self): ...
       def goto_target(self, target: MotionTarget) -> bool:
           # Read generic head fields
           yaw = target.head_yaw or 0.0
           # Read robot-specific joints from extras
           left_arm = target.extras.get("left_arm_yaw", 0.0)
           gripper = target.extras.get("gripper_left", 0.0)
           return self._sdk.move(yaw=yaw, left_arm=left_arm, gripper=gripper, ...)
       def look_at_pixel(self, target: PixelTarget) -> bool: ...
       def get_current_pose(self) -> dict[str, float]: ...
       def wake_up(self) -> bool: ...
       def goto_sleep(self) -> bool: ...
       # ... see hardware/reachy/controller.py for the full list
   ```

2. **Register via the `maxim.robots` entry-point group** in your package's `pyproject.toml`:

   ```toml
   [project.entry-points."maxim.robots"]
   atlas = "maxim_atlas.controller:AtlasController"
   ```

   `RobotRegistry` auto-discovers it on next startup. No core code changes needed.

3. **(Optional) Ship a SEM template** so the agent has a body model:

   ```yaml
   # maxim_atlas/components/atlas.yaml
   component:
     name: atlas
     tags: [robot, body, embodied, humanoid]
     category: bodies

   entity:
     name: atlas
     entity_type: robot
     sensors:
       battery: {unit: ratio, range: [0, 1], initial: 1.0}
       left_arm_yaw: {unit: radians, range: [-3.14, 3.14], initial: 0.0}
       gripper_left: {unit: ratio, range: [0, 1], initial: 0.0}
       # ...
     modulators:
       motion:
         affordances:
           walk_to: {params: {x: float, y: float}, description: "Walk to a 2D coordinate"}
           grip: {params: {object: str, force: float}, description: "Close the left gripper"}
   ```

   Drop it in `~/.maxim/components/bodies/` (or ship it as part of your plugin) and the registry picks it up.

4. **(Optional) Ship robot-specific tools** in `maxim_atlas/tools.py`. Register them via `maxim.register_tool()` or your package's entry point.

That's it. The agent loop, default network, planning, and memory systems will use your robot through the `RobotController` interface and the SEM graph — no Maxim-side changes required.

### The Reachy Mini SEM template

As a reference, Maxim ships a SEM template for Reachy Mini at `bodies/reachy_mini` — an ode to the original embodiment that started this project. It models the robot's joints (head pose, body yaw, antennas), camera/microphone health, battery, motor temperature, and pose confidence as SEM sensors, with `look_at`, `nod`, `antenna_alert`, and lifecycle modulators. Use it as a reference when building a SEM template for your own robot:

```python
from maxim.embodiment.component_registry import ComponentRegistry

registry = ComponentRegistry()
reachy = registry.instantiate("bodies/reachy_mini", name="my_reachy")
print(reachy.sensors.keys())  # ['head_yaw', 'head_pitch', ..., 'antenna_left', ...]
```

### MotionTarget extras: how robot-specific joints work

`MotionTarget` keeps generic head/body fields on the dataclass, but pushes everything else into `extras: dict[str, float]`:

```python
from maxim.hardware import MotionTarget

# Reachy: antennas
target = MotionTarget(
    head_pitch=0.2,
    extras={"antenna_left": 0.5, "antenna_right": -0.5},
)

# Atlas: arms + grippers
target = MotionTarget(
    extras={
        "left_arm_yaw": 1.0,
        "left_arm_pitch": -0.3,
        "gripper_left": 0.7,
    },
)

# Spot: legs (each foot's pressure target)
target = MotionTarget(
    extras={
        "front_left_foot": 1.0,
        "front_right_foot": 1.0,
        "rear_left_foot": 0.8,
        "rear_right_foot": 0.8,
    },
)
```

A controller reads its own keys out of `extras` and ignores keys it doesn't recognize. This keeps the canonical motion command SDK-agnostic — the agent loop produces a single `MotionTarget`, every controller speaks its dialect through `extras`.

### What's still planned

The `selfy.py` mixin stack in `embodied_runtime/` is currently hardcoded for the Reachy SDK. Decomposing it into a `RobotController`-driven generic loop is tracked as the **"Embodiment Hardware Adapter + selfy.py decomposition"** item in [docs/plans/future_plans.md](../plans/future_plans.md). When that lands, you'll be able to run the live media-capture loop against any registered robot, not just Reachy.

Until then, the recommended path for non-Reachy robots is:
- Use Maxim **headless** with your robot (the agent loop, planning, memory, and tools all work without `selfy.py`'s media loop)
- OR write a thin per-robot variant of the live loop that uses your SDK's capture API
