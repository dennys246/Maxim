# Installing Maxim on a Raspberry Pi (aarch64)

The `PKG` seam's user-facing half. Target: **Raspberry Pi 5, 64-bit Raspberry Pi
OS (bookworm) — aarch64, glibc 2.36, CPython 3.11.**

## The one-liner

```bash
sudo apt install -y python3-dev gcc pkg-config libgirepository1.0-dev libcairo2-dev
pip install 'pymaxim[pi]'
```

The apt line is **not optional** — see [Why apt first](#why-apt-first).

## What `[pi]` is, and what it deliberately leaves out

`pi` = `reachy` + `console` + `llm-anthropic` + `tts`.

The Pi is a **sensor/actuator peer**: LLM inference is REMOTE (mesh → the
owner's leader box) through `_MaximPeerBackend`, which uses only core deps
(`httpx`). So **no LLM extra is needed for the mesh branch at all**;
`llm-anthropic` is included only because the setup wizard also offers a cloud
branch, and it is a small pure-Python client.

| Excluded | Why |
|---|---|
| `semantic` | Declares `torch>=2.1`. Measured on aarch64 it resolves **torch + triton + nvidia-\* CUDA shards** — and torch alone is a ~450 MB runtime floor ([FIT](fit_runbook.md)). The encoder belongs on the **leader** ([perception_pipeline_placement.md](../../plans/perception_pipeline_placement.md)). |
| `llm-llama` / `llm-server` | No local inference on the peer. |
| `training` | TensorFlow. |
| `yolo` | AGPL, and not needed for the MVP. |
| `database`, `comms`, `search`, `temporal` | Leader/server-side concerns. |

> **Note:** earlier docs described the Pi combo as
> `pymaxim[reachy,llm-anthropic,semantic]` and claimed it "pulls no torch".
> That was self-contradictory — `semantic` declares torch directly. `[pi]` is
> the corrected combo.

### What you give up without `semantic`

The substrate encoder degrades to a **documented bag-of-words fallback**
(SHA-256 over sorted unique words, 384-dim). It warns once and nothing raises.
Concretely: EC pattern-separation quality drops — recognising "the same thing
said differently" across sessions is the capability that suffers.

**What does NOT degrade:** `RECALL` (the console's "what Maxim remembers about
you" read) is entirely embedding-free, and talk/adventure/PROBE/SETUP/EVENT are
unaffected. Placing the encoder on the leader gets full neural quality *and* the
lean Pi footprint — that is the recommended topology, not a compromise.

## Why apt first

`reachy-mini` depends on **PyGObject** (and transitively **pycairo**) on Linux.
Neither publishes a wheel **for any platform** — pip always compiles them from
source, which needs the GObject-introspection and Cairo headers plus a C
compiler. Without the apt line you get a build error from deep inside a
dependency, which reads like a Maxim bug and is not one.

This is also why CI has a *real install* job and not only a resolution check: a
dry resolve can never detect a missing system package.

On Linux, `reachy-mini` skips the `gstreamer-bundle` wheel entirely
(`sys_platform != "linux"`) and uses system GStreamer through PyGObject — which
is why the apt prerequisite exists on the Pi but not on macOS/Windows.

## Verifying an install

```bash
python -c "import maxim; print(maxim.__version__)"
python -c "import importlib.util as u; \
  assert all(u.find_spec(m) is None for m in ('torch','llama_cpp','tensorflow')), \
  'heavy backend leaked'; print('lean install OK')"
```

To check resolution *without* a Pi (needs [uv](https://docs.astral.sh/uv/)):

```bash
python scripts/check_aarch64_install.py --print-resolved
```

That cross-resolves `pymaxim[pi]` for `aarch64-manylinux_2_36` / cp311 and fails
if any heavy backend appears. **It uses uv rather than `pip --platform`
deliberately:** pip evaluates environment markers against the *host*, so it
reports zero `nvidia-*` for a target where CUDA would install
([pypa/pip#6117](https://github.com/pypa/pip/issues/6117)) — unsound for exactly
this assertion.

## Known-good baseline

At the time of writing, `pymaxim[pi]` resolves to **81 packages** on
`aarch64-manylinux_2_36` / cp311 with **zero** torch / nvidia-\* /
llama-cpp-python / triton / tensorflow.
