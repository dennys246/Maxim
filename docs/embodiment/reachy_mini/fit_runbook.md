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

| encoder | runtime | substrate RSS |
|---|---|---|
| all-mpnet-base-v2 (768) | torch | ~488 MB |
| all-MiniLM-L6-v2 (384) | torch | ~477 MB |
| paraphrase-MiniLM-L3-v2 (384) | torch | ~539 MB |
| **bge-small-en-v1.5 (384)** | **onnxruntime/fastembed** | **~441 MB** |
| bag-of-words fallback | none | ~70 MB |

Two negative results, both important:
1. **A 7× smaller torch model saves ~nothing** (MiniLM-L3 was even heavier) — RSS is not weight-dominated.
2. **Dropping torch for onnxruntime saves only ~10%** (~441 vs ~488) — *and* onnxruntime grows its arena during inference (+59 MB over 100 encodes). **The neural-inference RUNTIME is the ~350–430 MB floor, torch or onnx alike** — not the model, not the framework choice.

⚠️ Correction: an earlier draft of this doc claimed onnxruntime would land ~100 MB. **Measured, it does not** (~441 MB). There is **no cheap middle path with real embeddings on-device** — the choice is genuinely **neural (~440–490 MB) vs bag-of-words (~70 MB)**.

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
- **Place the encoder on the leader (the architecturally clean answer).** If ~450 MB doesn't fit the Pi, don't shrink the encoder — *move* it. Per [`perception_pipeline_placement.md`](../../plans/perception_pipeline_placement.md), the encoder is a **placeable stage**: the Pi ships raw text percepts, the leader (e.g. a Mac Mini) encodes them at full neural quality, and embeddings return to the Pi's substrate. Keeps quality, moves the runtime floor off the constrained device. FIT's outcome feeds this placement decision directly.
- **A tuned onnxruntime is an open measurement, not a promise** — int8-quantized model + `enable_cpu_mem_arena=False` might shave more off the ~441 MB, but the runtime floor is real; measure before relying on it. (The earlier ~100 MB claim was falsified above.)

Record on-Pi numbers (torch + fallback, under daemon+GStreamer load) back in the tables above when they exist; add any tuned-onnx or encoder-on-leader numbers as they're measured.
