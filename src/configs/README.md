# Configs (Maxim)

This folder contains **version-controlled config templates** and quick reference notes.

## Runtime Locations

Most runtime-editable files live under `data/`:
- Run artifacts (recordings + logs): `data/audio/`, `data/videos/`, `data/images/`, `data/transcript/`, `data/logs/`
- Training sample logs: `data/training/` (e.g., `motor_training_set.jsonl`)
- Motion presets: `data/motion/default_actions.json`
- Default head poses: `data/motion/default_poses.json` (e.g., `centered`)
- Key bindings: `data/util/key_responses.json` (override via `$MAXIM_KEY_RESPONSES`)
- Model artifacts (weights/checkpoints): `data/models/`
- Zenoh/networking config (optional): `data/networking/`
- Agent/credential files (optional): `data/agents/`, `data/credentials/`

## Templates

Copy templates from `src/configs/templates/` into `data/` (or point env vars to them) if you want a known-good starting point:
- `src/configs/templates/key_responses.json`
- `src/configs/templates/default_actions.json`
- `src/configs/templates/default_poses.json`
- `src/configs/templates/motor_cortex.json`
- `src/configs/templates/zenoh.json5`
- `src/configs/templates/env.example`
