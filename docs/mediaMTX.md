# MediaMTX

MediaMTX is the RTSP relay that sits between Maxim (publisher) and consumers like ShredderSegmenter. It's a single binary with zero configuration needed for basic use.

## Auto-Start (Default Behavior)

**Maxim auto-starts MediaMTX when a protocol needs RTSP streaming.** If the RTSP port (default 8554) is not in use, the RTSPBridge starts MediaMTX automatically and stops it on deactivation. You only need `mediamtx` in your PATH.

```bash
# Just put mediamtx on PATH — Maxim handles the rest
sudo cp mediamtx /usr/local/bin/

# Or set a custom path in protocol config
ShredderSegmenterProtocol(mediamtx_path="/opt/mediamtx/mediamtx")
```

To disable auto-start (e.g., when MediaMTX runs as a system service or on a remote host), set `auto_start_mediamtx=False` in `RTSPStreamingConfig`.

## Manual Setup

For deployments where MediaMTX runs separately (cloud relay, shared server, systemd service), manage it yourself:

## Network Topology

```
┌─────────────────────────────┐              ┌─────────────────────┐
│  Reachy Mini + Maxim        │    RTSP      │  ShredderSegmenter  │
│  + ffmpeg + MediaMTX        │◀─────────────│  (consumer)         │
│                             │    pull       │                     │
│  Network A                  │              │  Network B          │
└─────────────────────────────┘              └─────────────────────┘
```

**Typical setup: two networks.** Reachy, Maxim, ffmpeg, and MediaMTX all run on the same host (Network A). ShredderSegmenter connects from its own network (Network B) by pulling the RTSP stream from MediaMTX. The only requirement is that ShredderSegmenter can reach the MediaMTX port (default 8554) on the Reachy host.

For advanced deployments, MediaMTX can also run on a separate relay server:

| Location | When to use |
|----------|------------|
| On the Reachy host (default) | Simplest setup — Maxim auto-starts MediaMTX locally |
| On a cloud/VPS server | When ShredderSegmenter can't reach the Reachy host directly |
| On the ShredderSegmenter host | If Reachy can reach that machine |

The `rtsp_url` config tells ffmpeg where to publish. Set it to wherever MediaMTX is running.

## Installation

### Download

```bash
# Pick your architecture
ARCH=linux_amd64        # x86 Linux
# ARCH=linux_arm64v8    # ARM64 (Raspberry Pi, Jetson)
# ARCH=darwin_arm64     # macOS Apple Silicon
# ARCH=darwin_amd64     # macOS Intel

VERSION=1.12.2
wget "https://github.com/bluenviron/mediamtx/releases/download/v${VERSION}/mediamtx_v${VERSION}_${ARCH}.tar.gz"
tar xzf mediamtx_*.tar.gz
```

### Run

```bash
# Default: RTSP on :8554, HTTP API on :9997
./mediamtx
```

MediaMTX auto-creates stream paths when a publisher first connects. No configuration file needed.

## Deployment Scenarios

### Scenario 1: Same Machine as Reachy (default, recommended)

Maxim auto-starts MediaMTX on the Reachy host. No manual setup needed — just put `mediamtx` on PATH.

```bash
# Maxim handles MediaMTX automatically
maxim --mode agentic
# Say: "run shredder segmenter protocol"
```

ShredderSegmenter connects from its network to `rtsp://<reachy-ip>:8554/reachy`. Port 8554 must be reachable from ShredderSegmenter's network.

### Scenario 2: Cloud Relay (when ShredderSegmenter can't reach Reachy directly)

If ShredderSegmenter can't reach the Reachy host (e.g., different NATs with no port forwarding), run MediaMTX on a server both sides can reach:

```bash
# On the cloud server
./mediamtx
```

Configure Maxim to publish there:

```bash
export SHREDDER_API_URL="http://shredder.example.com:8000"
export SHREDDER_LICENSE_ID="your-license"
```

Then instantiate the protocol with the remote URL:

```python
ShredderSegmenterProtocol(
    rtsp_url="rtsp://cloud-server.example.com:8554/reachy",
    shredder_api_url="http://shredder.example.com:8000",
    # ...
)
```

Or set the RTSP URL via environment/config before starting Maxim.

### Scenario 3: On ShredderSegmenter's Host

If the Reachy can reach the ShredderSegmenter machine:

```bash
# On the ShredderSegmenter host
./mediamtx &

# Configure Maxim to publish to that host
# rtsp_url="rtsp://<shredder-host>:8554/reachy"
```

## Firewall / Port Requirements

| Port | Protocol | Direction | Purpose |
|------|----------|-----------|---------|
| 8554 | TCP | Inbound to MediaMTX | RTSP publish (ffmpeg) and consume (ShredderSegmenter) |
| 8322 | UDP | Inbound to MediaMTX | RTP (if using UDP transport instead of TCP) |
| 9997 | TCP | Inbound to MediaMTX | HTTP API (optional, for status/metrics) |

If running on a cloud server, open port 8554 at minimum.

## Persistent Deployment (systemd)

For always-on deployments (e.g., ski season recording):

```ini
# /etc/systemd/system/mediamtx.service
[Unit]
Description=MediaMTX RTSP Server
After=network-online.target

[Service]
ExecStart=/opt/mediamtx/mediamtx
Restart=always
RestartSec=5
User=mediamtx

[Install]
WantedBy=multi-user.target
```

```bash
sudo cp mediamtx /opt/mediamtx/
sudo useradd -r -s /bin/false mediamtx
sudo systemctl enable --now mediamtx
```

## Verifying the Stream

### Check MediaMTX is accepting connections

```bash
# HTTP API (if port 9997 is open)
curl http://localhost:9997/v3/paths/list
```

### Watch the stream with ffplay or VLC

```bash
# ffplay (comes with ffmpeg)
ffplay rtsp://localhost:8554/reachy

# VLC
vlc rtsp://localhost:8554/reachy
```

### Check from a remote machine

```bash
ffplay rtsp://<mediamtx-host>:8554/reachy
```

If this works, ShredderSegmenter can reach it too.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| "Connection refused" from ffmpeg | MediaMTX not running | Start `./mediamtx` first |
| ffmpeg starts then exits immediately | MediaMTX unreachable | Check host/port, firewall |
| Stream plays but ShredderSegmenter can't connect | Firewall blocking 8554 | Open port on MediaMTX host |
| High latency (>2s) | Network distance or buffering | Use TCP transport (`-rtsp_transport tcp`), reduce `gop_size` |
| Frame drops | Bandwidth too low for bitrate | Lower `bitrate` (e.g., "1M") or `fps` |
| "Bridge stopped unexpectedly" in Maxim | MediaMTX crashed or was stopped | Restart MediaMTX, protocol will need re-activation |

## Configuration Reference

MediaMTX works with zero config, but for advanced setups create `mediamtx.yml`:

```yaml
# Optional: restrict who can publish/read
paths:
  reachy:
    # Allow publishing only from Reachy's IP
    publishIPs: [192.168.1.0/24]
    # Allow reading from anywhere (or restrict to ShredderSegmenter IPs)
    readIPs: []
```

See [MediaMTX documentation](https://github.com/bluenviron/mediamtx) for full configuration options.
