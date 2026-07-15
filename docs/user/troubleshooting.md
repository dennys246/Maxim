# Troubleshooting

## Quick Diagnostics
```bash
# Environment check (network, GPU, models, disk, etc.)
maxim doctor

# Check robot connection
maxim --timeout 60 --log-level 2

# Verbose logging
maxim --log-level 2

# Extra verbose agentic logging
maxim --mode agentic --log-level 2 --display debug
```

## Common Issues

### Connection Problems
| Issue | Solution |
|-------|----------|
| "No robot found" | Check robot is on same network, daemon is running |
| Port 8443 refused | Restart daemon: `sudo systemctl stop reachy-mini-daemon` then start manually |
| Port 8000 refused | Daemon down/not ready — `curl http://<robot>:8000/api/daemon/status`; see [docs/embodiment/reachy_mini/troubleshooting.md](../embodiment/reachy_mini/troubleshooting.md) |
| Connection timeout | Increase with `--timeout 60`, check firewall |

### LLM Issues
| Issue | Solution |
|-------|----------|
| "Model not found" | `python -m maxim.models.download --llm <model-name>` or `maxim.download_model("model-name")` from Python |
| `--llm` flag ignored | Fixed in v1.0.0 — `detect_tiers()` now respects `MAXIM_LLM_PROFILE`. Update to latest. |
| Wrong model loads on restart | Your `--llm` choice now persists automatically. Clear with: delete `~/.maxim/util/active_llm_model.<role>.txt` (e.g. `active_llm_model.leader.txt`) |
| VRAM not released on exit | Fixed in v1.0.0 — shutdown now kills the full process tree. If stuck: `maxim --delete-model` won't help, but restarting will auto-kill stale servers. |
| Out of memory | Use `smollm-1.7b` or lower quantization (`Q3_K_M`). Delete unused models: `maxim --delete-model <name>` |
| Slow inference | Use smaller model (`smollm-1.7b`) or lower quantization |
| Gibberish output | Check `prompt_style` matches model: `maxim config get llm.profile` |
| No LLM response | Ensure `MAXIM_LLM_ENABLED=1` or run `maxim config set llm.enabled true` |
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

## Bio-System Issues

### Memory not persisting across sessions

Episodic memory and learned causal links are written to `~/.maxim/memory/` at session end. Check that the directory exists and is non-empty:

```bash
ls -lh ~/.maxim/memory/
# Expect: hippocampus.json, atl.json (if [semantic] is installed), etc.
```

If the directory is missing or empty:

1. **Check permissions** — Maxim must be able to write to `~/.maxim/`. If `MAXIM_DATA_HOME` is set, that path is used instead.
2. **Check for a clean session end** — An abrupt kill (`kill -9`, power loss) can skip the save step. Use `Ctrl+C` to stop Maxim gracefully; it saves on shutdown.
3. **Check `maxim doctor`** — The doctor check surfaces disk-space and permission issues that prevent writes.

For named agents (created via `AgentFactory`), persistence files live under `~/.maxim/agents/{agent_id}/` (e.g., `~/.maxim/agents/scout/hippocampus.json`).

Simulation AUT bio-system snapshots (from `--sim` runs) are written to the sim reports directory: `~/.maxim/sim_reports/{session_id}/aut_hippocampus.json` and `aut_nac.json`.

### Substrate encoding / concept learning not occurring

The dual-write path (LinguisticEncoder → EntorhinalCortex → ATL) is gated behind an env var and requires the `[semantic]` extra:

```bash
# Enable the substrate encoding path
export MAXIM_SUBSTRATE_PATH=1

# Verify semantic dependencies are installed
python -c "from sentence_transformers import SentenceTransformer; print('ok')"
# If this fails: pip install 'pymaxim[semantic]'
```

Without `MAXIM_SUBSTRATE_PATH=1`, percepts are encoded into episodic memory but not into the semantic substrate (EC/ATL) used for concept-level pattern completion and reward-bias modulation.

Without `[semantic]`, the LinguisticEncoder falls back to a bag-of-words hash embedding. Causal links are still formed, but pattern completion and paraphrase-level collapse will not work — substrate-dependent features (affordance concept transfer, Wire-A annotation) will produce lower-quality results.

### Causal learning not occurring (NAc)

The NAc (causal inference system) records links between actions and outcomes. To inspect the current state after a session:

```bash
# After a sim run, the AUT's NAc snapshot is in the sim reports dir:
ls ~/.maxim/sim_reports/
# Look for aut_nac.json in the most recent session directory

# For the persistent agent, NAc state lives alongside hippocampus.json:
ls ~/.maxim/memory/nac.json          # if using default memory path
ls ~/.maxim/agents/{agent_id}/nac.json  # if using named agents
```

If the file is missing or has zero causal links:

- **Pain bus not wired** — If running headless via the Python API with `pain_bus=None`, tool-outcome learning is disabled by design. See [Python API docs](python-api.md).
- **No tool calls occurred** — NAc learns from tool invocations with measurable outcomes. Passive observation (no tools) produces no causal links.
- **`MAXIM_NAC_REWARD_BIAS_DISABLED=1`** — This env var zeroes out reward-bias surfaces (used in ablation experiments). Unset it for normal operation.

To reset causal learning:

```bash
maxim --clear-memory nac
```

### Poor memory / substrate quality

If memory quality seems low (concepts don't associate, paraphrase recall misses, substrate-aware prompts look generic):

1. **Install `[semantic]`** — The `[all]` extra does NOT include `[semantic]`. Sentence-transformers (used by the LinguisticEncoder, ComponentIndex, and EC pattern completion) is a separate install:

   ```bash
   pip install 'pymaxim[semantic]'
   # Or for a combined install:
   pip install 'pymaxim[all,semantic]'
   ```

   Verify the encoder is using real embeddings (not the hash fallback):
   ```bash
   python -c "
   from maxim.similarity.encoder import LinguisticEncoder
   enc = LinguisticEncoder()
   print('using_fallback:', enc.using_fallback)
   "
   # Should print: using_fallback: False
   ```

2. **Enable concept decomposition** — With spaCy installed (`[semantic]`), enable noun-phrase extraction before EC encoding:

   ```bash
   export MAXIM_CONCEPT_DECOMPOSITION=1
   ```

3. **Enable the substrate path** — See above (`MAXIM_SUBSTRATE_PATH=1`).

4. **Check EC threshold** — The default pattern-completion threshold is 0.44. Lower values cause centroid drift (concepts collapse together). Do not lower it below 0.44 without reading the EC centroid-drift architecture notes.

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
maxim --log-level 2

# Agentic loop detail (0=warnings/errors, 1=info, 2=debug)
maxim --mode agentic --log-level 2 --display debug

# Provenance tracing (see why decisions were made)
MAXIM_PROVENANCE_VERBOSITY=2 maxim --mode agentic
```
