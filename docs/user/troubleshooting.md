# Troubleshooting

## Quick Diagnostics
```bash
# Check robot connection
maxim-diagnostics --host <REACHY_IP>

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
| "Model not found" | Run `./scripts/download_models.sh --llm --enable` |
| Out of memory | Use `smollm-1.7b` or lower quantization (`Q3_K_M`) |
| Slow inference | Use `--prompt-profile minimal` |
| Gibberish output | Check `prompt_style` matches model in `data/util/llm.json` |
| No LLM response | Ensure `MAXIM_LLM_ENABLED=1` or `enabled: true` in llm.json |

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
| Speech not detected | Lower `vad_threshold` in `data/util/whisper.json` (try 0.15) |
| Too much noise | Raise `vad_threshold` (try 0.4) |
| No microphone | Check audio device, ensure `--audio True` |

### System Issues
| Issue | Solution |
|-------|----------|
| Matplotlib crash | `rm -rf ~/.cache/matplotlib && fc-cache -f` |
| Import errors | Ensure you're in the virtual environment |
| Permission denied | Check filesystem policy, try `--autonomy planning` |
| High memory usage | Use `--prompt-profile minimal`, smaller LLM model |
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
