# FIT — does the Maxim substrate fit the Reachy Pi?

**FIT** is the gate for the on-device / mesh story (the `maxim-pulse` app line; see
`docs/plans/reachy_app_maxim_seams.md` once that lands). Question: with the large LLM
tier **remote** (mesh), does the substrate that stays **local** on the robot — the
encoder + EC / NAc / Hippocampus / ATL / SCN — fit in the Raspberry Pi's RAM alongside
the `reachy-mini` daemon + GStreamer?

Harness: [`scripts/fit_substrate_footprint.py`](../../../scripts/fit_substrate_footprint.py).

## The finding that reframes FIT (dev-box first pass, 2026-07-23)

Measured on a dev box (macOS/arm64, py3.12), LLM not loaded:

| stage | RSS | Δ |
|---|---|---|
| baseline | 20 MB | — |
| `import maxim` | 69 MB | +49 (numpy/scipy) |
| `build_bio_stack` (substrate wired) | 71 MB | **+1.5** |
| encoder warmup (model loaded) | 501 MB | **+430** |
| +100 encodes (steady state) | 508 MB | +7 |
| **substrate footprint** | | **≈ 488 MB** |

**The bio-systems are essentially free (~1.5 MB). The entire footprint is the encoder
model (torch + `all-mpnet-base-v2` ≈ 430 MB).** So FIT collapses to one decision:

- **Real `sentence-transformers` (torch): ≈ 488 MB** — full semantic quality.
- **Bag-of-words fallback (no torch): ≈ 70 MB** — the encoder degrades to a
  deterministic hash (see `similarity/encoder.py::_get_encoder`), ~7× lighter.

The encoder backend is the lever, not the bio-stack. (Aarch64 torch RSS typically runs
*higher* than x86/arm-mac — budget ~500–700 MB for the torch path on the Pi.)

### Middle path priced (2026-07-23): it's **torch**, not the model

Ran the harness against smaller 384-dim models to find a footprint middle ground — there isn't one *within torch*:

| encoder | weights (disk) | substrate RSS |
|---|---|---|
| all-mpnet-base-v2 (768) | ~420 MB | ~488 MB |
| all-MiniLM-L6-v2 (384) | ~80 MB | ~477 MB |
| paraphrase-MiniLM-L3-v2 (384) | ~61 MB | ~539 MB |
| bag-of-words fallback | 0 | ~70 MB |

A 7× smaller model saved ~nothing (L3 was even heavier). **RSS is torch-framework-dominated (~400 MB floor), not weight-dominated** — the model on top is noise. So the choice is binary: **torch (any model, ~480–540 MB) vs no-torch (~70 MB)**. The genuine middle path is dropping torch, not shrinking the model.

## Run it on the actual Pi (the real answer)

The dev-box number is a ballpark; the binding measurement is on-hardware, under load.

1. **Install pymaxim on the Pi, both ways, to bracket the decision:**
   - Torch path: `pip install 'pymaxim[reachy,semantic]'` (pulls sentence-transformers → torch; confirm an aarch64 wheel exists and doesn't blow the SD card — this is also a `PKG`-seam check).
   - Lean path: `pip install 'pymaxim[reachy]'` (no `semantic` → bag-of-words fallback).
2. **Measure idle free RAM first:** `free -m` with the `reachy-mini` daemon running and a camera/audio stream active (GStreamer loaded). That's the RAM the substrate must fit inside.
3. **Run the harness under that load:**
   ```bash
   PYTHONPATH=src python scripts/fit_substrate_footprint.py --exercise 200 --json
   ```
   Do it once per install (torch vs lean). The printed `encoder_backend` confirms which path ran.
4. **Verdict:** substrate footprint < (free RAM after daemon+GStreamer, with headroom). On a 4 GB Pi the torch path is tight; on 8 GB it's comfortable. The lean path fits either.

## The decision this feeds

Which encoder ships on the Reachy shell:

- **Torch path** if it fits with headroom — best substrate quality (real embeddings feed EC pattern-separation; the bag-of-words fallback measurably degrades clustering).
- **Lean/fallback path** if torch doesn't fit (or the aarch64 wheel / SD-card cost is prohibitive) — the app still works; substrate quality is lower. This is a conscious, documented trade, not a silent degrade (`warn_optional_fallback` already logs it once).
- **The real middle path — a torch-free embedding runtime.** Smaller torch models don't help (above); dropping torch does. `onnxruntime` / `fastembed` running a MiniLM exported to ONNX gives real MiniLM-quality embeddings at ~100 MB (onnxruntime is ~5× lighter than torch). This is the option that actually buys quality-without-the-430MB, and it's the next thing to price (needs `onnxruntime`/`fastembed` — a light new dep). A GGUF embedding model via llama.cpp is a second torch-free route.

Record on-Pi numbers (both torch + fallback, under daemon+GStreamer load) and any ONNX-path number back in the tables above when they exist.
