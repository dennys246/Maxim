# Robot Setup

## Overview
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
