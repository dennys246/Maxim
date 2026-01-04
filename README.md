# ------- | Maxim | -------

A Reachy Mini repo for orchestrating data streaming to and from a PC and Reachy Mini to orchestrate agents and models.

## - Getting Started with Maxim

Run the Reachy Mini daemon on the robot, then run `maxim` from any computer on the same LAN/Wi‑Fi (Zenoh peer discovery).

```bash
ssh pollen@<INSERT YOUR REACHY IP>
```

Then enter the default password 'root' if first logging on or the unique password you reset it too.

Stop the process if something is using it.

```bash
sudo systemctl stop reachy-mini-daemon
```

Check to see if you can start a new daemon process

```bash
source /venvs/mini_daemon/bin/activate
python -m reachy_mini.daemon.app.main --wireless-version --no-localhost-only
```

On your controller computer clone this repo into a folder of your choosing

```bash
git clone https://github.com/dennys246/Maxim.git
```

Prepare a computing environment for running Maxim by creating a new python virtual environment. Avoid installing requirements into a virtual environment you typically use for machine learning as it may mess up your tensorflow or pytorch dependencies and how your GPU is handled.

```bash
cd Maxim
python -m venv maxim-env
source maxim-env/bin/activate
pip install -e .
```

If you previously installed an older version, re-run `pip install -e .` to refresh the `maxim` command.

After `pip install -e .`, run the `maxim` command (from anywhere in that environment) to initiate basic observation using Ultralytics incredibly efficient YOLO8 model. This dynamically find objects of interest and center the Reach Mini vision on them. Audio is recorded is transcribed when enabled.

```bash
maxim
```

NOTE: You can also set `MAXIM_ROBOT_NAME=reachy_mini` and run `maxim`.

Legacy entrypoint (still supported when running from a cloned checkout):

```bash
python scripts/main.py
```

You can also run Maxim straight from a python shell or your own script by importing it (package name: `maxim`)

```python
from maxim.conscience.selfy import Maxim

maxim = Maxim()

# Starts the live loop (capture → inference/control → record artifacts)
maxim.live()

# General movement wrapper to the Reachy SDK
maxim.move(y = 10, yaw = 3)

```

Of course extensions of the Maxim class using the datastreams set up are more than welcomed!

## Outputs (Default)

Each run writes a timestamped set of artifacts under `data/`:
- `videos/reachy_video_<YYYY-MM-DD_HHMMSS>.mp4`
- `audio/reachy_audio_<YYYY-MM-DD_HHMMSS>.wav`
- `transcript/reachy_transcript_<YYYY-MM-DD_HHMMSS>.jsonl` (when `--audio true` and Whisper is available)
- `logs/reachy_log_<YYYY-MM-DD_HHMMSS>.log`

Shared model artifacts and weights live under `data/models/` (e.g., `MotorCortex/`, `YOLO/`).

## CLI Flags

- `--mode`: `passive-interaction` (default), `live`, `train`, `sleep` (audio-only; no `wake_up()`)
- `--verbosity`: `0`, `1`, `2`
- `--audio`: `True/False` (enables audio recording + transcription)
- `--audio_len`: seconds per transcription chunk (default `5.0`)

## Keyboard Shortcuts

While `maxim` is running in a terminal, it listens for single-key presses configured in `data/util/key_responses.json` (or `$MAXIM_KEY_RESPONSES`).

Default:
- `c`: center vision (pauses training briefly in `--mode train`)

Default movement presets are defined in `data/motion/default_actions.json`.
Default head poses (including `centered`) are defined in `data/motion/default_poses.json`.

## Smoke Tests

Quick local checks live under `src/tests/`:
- `bash src/tests/basic_vision.sh`
- `bash src/tests/basic_audio.sh` (set `MAXIM_TEST_REAL_WHISPER=1` to attempt real transcription)
- `bash src/tests/basic_learn.sh` (skips if `tensorflow/keras` not installed)
- `bash src/tests/basic_move.sh --require-robot` (requires a Reachy daemon on the network)

For easy future use consider editing your Reachy's .bashrc...

```bash
nano ~/.bashrc
```

and adding aliases so you can run simple commands to start processes

```bash
alias mini-env='source /venvs/mini_daemon/bin/activate'

alias list-daemon='ss -lntp | grep 8000'
alias clear-daemon='sudo systemctl stop reachy-mini-daemon'
alias start-daemon='python -m reachy_mini.daemon.app.main --wireless-version --no-localhost-only'

alias list-zenoh='ss -lntp | grep 7447'

MAXIM_ROBOT_NAME=reachy_mini
REACHY_IP=<INSERT YOUR REACHY IP>
```
Then you can simply type commands like list-daemon, clear-daemon or start-daemon.

## Networking

Make sure you are on the same network as your Reachy Mini with no VPN. With a VPN you may be able to do simple things like start the daemon but the python SDK will struggle to connect to the Reachy.

## Troubleshooting

1. Reachy Mini immendiately closing down on running or not running at all.

Check if the reachy mini port 8000 is occupied by ssh into you Reachy Mini then
checking if the port is occupied...

```bash
ss -lntp | grep 8000
```

Stop the process if something is using it.

```bash
sudo systemctl stop reachy-mini-daemon
```

Check to see if you can start a new process

```bash
python -m reachy_mini.daemon.app.main --wireless-version --no-localhost-only
```
