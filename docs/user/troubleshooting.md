# Troubleshooting

## Quick Diagnostics
```bash
# Environment check (network, GPU, models, disk, etc.)
maxim doctor

# Check robot connection
maxim --timeout 60 --verbosity 2

# Verbose logging
maxim --verbosity 2

# Extra verbose agentic logging
maxim --mode agentic --agentic-verbosity 3
```

## Common Issues

### Connection Problems
| Issue | Solution |
|-------|----------|
| "No robot found" | Check robot is on same network, daemon is running |
| Port 8443 refused | Restart daemon: `sudo systemctl stop reachy-mini-daemon` then start manually |
| Port 7447 refused | Same as above — system daemon may be holding the port |
| Connection timeout | Increase with `--timeout 60`, check firewall |

### LLM Issues
| Issue | Solution |
|-------|----------|
| "Model not found" | `python -m maxim.models.download --llm <model-name>` or `maxim.download_model("model-name")` from Python |
| `--llm` flag ignored | Fixed in v0.2.0 — `detect_tiers()` now respects `MAXIM_LLM_PROFILE`. Update to latest. |
| Wrong model loads on restart | Your `--llm` choice now persists automatically. Clear with: delete `~/.maxim/util/active_llm_model.txt` |
| VRAM not released on exit | Fixed in v0.2.0 — shutdown now kills the full process tree. If stuck: `maxim --delete-model` won't help, but restarting will auto-kill stale servers. |
| Out of memory | Use `smollm-1.7b` or lower quantization (`Q3_K_M`). Delete unused models: `maxim --delete-model <name>` |
| Slow inference | Use smaller model (`smollm-1.7b`) or lower quantization |
| Gibberish output | Check `prompt_style` matches model in `~/.maxim/config/llm.json` |
| No LLM response | Ensure `MAXIM_LLM_ENABLED=1` or `enabled: true` in `~/.maxim/config/llm.json` |
| Disk space low from models | `maxim --list-models` shows download status. Delete with `maxim --delete-model <name>` or `maxim.delete_model("name")` |

### Vision Issues
| Issue | Solution |
|-------|----------|
| No detections | Check camera connection, run diagnostics |
| OpenCV Qt warnings | `MAXIM_DISABLE_IMSHOW=1 maxim` |
| Vision models missing | `python -m maxim.models.download --vision` |
| Slow frame rate | Use RTM engine (default), not YOLO |

### Audio Issues
| Issue | Solution |
|-------|----------|
| Whisper segfaults | `MAXIM_WHISPER_COMPUTE_TYPE=float32 maxim` |
| Speech not detected | Lower `vad_threshold` in `~/.maxim/util/whisper.json` (try 0.15) |
| Too much noise | Raise `vad_threshold` (try 0.4) |
| No microphone | Check audio device, ensure `--audio True` |

### System Issues
| Issue | Solution |
|-------|----------|
| Matplotlib crash | `rm -rf ~/.cache/matplotlib && fc-cache -f` (matplotlib is optional — `pip install matplotlib` if needed) |
| Import errors | Ensure you're in the virtual environment |
| Permission denied | Check filesystem policy, try `--autonomy planning` |
| High memory usage | Use smaller LLM model (`smollm-1.7b`) or lower quantization |
| Bytecode issues | `maxim --clear-cache` |

## Resetting State
If the system behaves unexpectedly after learning:
```bash
# Reset specific learning
maxim --clear-memory focus           # Motor gains
maxim --clear-memory bounds          # Workspace limits
maxim --clear-memory fear,escalation # Safety thresholds
maxim --clear-memory hippo           # Episodic memories
maxim --clear-memory nac             # Causal learning
maxim --clear-memory scn             # Temporal patterns

# Nuclear option
maxim --clear-memory all
```

## Getting Help
- Issues: https://github.com/dennys246/Maxim/issues
- Architecture: see `ARCHITECTURE.md`
- Internal docs: see `docs/` directory

## Debug Logging
```bash
# Standard verbose
maxim --verbosity 2

# Agentic loop detail (0=quiet, 1=info, 2=debug, 3=trace)
maxim --mode agentic --agentic-verbosity 3

# Provenance tracing (see why decisions were made)
MAXIM_PROVENANCE_VERBOSITY=2 maxim --mode agentic
```
