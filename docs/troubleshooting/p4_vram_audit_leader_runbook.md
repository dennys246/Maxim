# P4 Stage 2 VRAM audit — leader runbook (debug edition)

**Audience:** a Claude agent running on the RTX 5080 leader desktop tasked with getting `scripts/p4_vram_audit.py` to produce a valid report, then committing the result back to main.

**Context:** P4 Stage 2 Phase 2E needs an authoritative VRAM co-residency measurement on the RTX 5080 with Qwen-14B loaded. The audit script is already in the repo (`scripts/p4_vram_audit.py`). The Mac peer reference run succeeded; the leader's first attempt failed with a torchvision error that looks like `RuntimeError: Dataset not found or corrupted. You can use download=True to download it`. This runbook walks you through the debug + run + report-back cycle.

## TL;DR — the fast path

If you're in a hurry and the leader has a working internet connection, just do this:

```bash
cd ~/Scripts/Maxim  # or wherever your leader checkout lives
git pull origin main
rm -rf ~/.cache/maxim/p4_flowers    # wipe any partial download
PYTHONPATH=src python scripts/p4_vram_audit.py
```

If that produces `docs/experiments/p4_vram_audit.md` with a **VERDICT: PASS** or **VERDICT: WARN** line, skip to the "Commit results" section at the bottom.

If it errors again, read the full runbook below.

## Why Phase 2B's successful download didn't help

The Phase 2B calibration sweep (commit `73b91e2`) was run on the **Mac peer**, not the leader. Its cache lives at the Mac's `~/.cache/maxim/p4_flowers/`, which has nothing to do with the leader's filesystem. The leader's first encounter with Oxford Flowers-102 is this VRAM audit, and torchvision's download path is exercising fresh here.

## The failure mode

`torchvision.datasets.Flowers102` downloads three files from the Oxford VGG site:

- `102flowers.tgz` — the image archive (~330 MB)
- `imagelabels.mat` — per-image class labels
- `setid.mat` — train/val/test split indices

Each file has an MD5 checksum baked into torchvision's source. If ANY file is:

- partially downloaded (network interrupted mid-transfer)
- replaced by an HTTP error page (some mirrors return HTML instead of binary)
- corrupted in transit
- blocked by a firewall/proxy that returns 403

...then torchvision's `_check_integrity()` fails on the next run and raises `RuntimeError("Dataset not found or corrupted. You can use download=True to download it")`. Notably, the error message does NOT tell you WHICH file failed — you have to inspect the cache directory manually.

There is also a subtler failure: some Oxford VGG mirrors have SSL cert issues that cause torchvision's download to complete with a zero-byte or tiny response body. `102flowers.tgz` is expected to be ~330 MB — anything smaller is a red flag.

## Debugging decision tree

### Step 1 — Inspect what's actually in the cache

```bash
ls -la ~/.cache/maxim/p4_flowers/flowers-102/ 2>/dev/null || echo "cache missing"
```

Expected contents after a successful download:

```
102flowers.tgz       (~330 MB)
imagelabels.mat      (~500 bytes)
setid.mat            (~14 KB)
jpg/                 (extracted image directory, ~8200 files)
```

**Decision:**

- **No directory at all** → proceed to Step 3 (fresh download)
- **Directory exists but missing `jpg/`** → the tarball downloaded but extraction failed. Proceed to Step 2.
- **Directory exists with `jpg/` populated** → torchvision's integrity check is failing on a file checksum even though files look present. Proceed to Step 2.
- **`102flowers.tgz` present but smaller than ~300 MB** → partial download. Go to Step 3 (wipe + retry).
- **`102flowers.tgz` present but opens as HTML when you `head -c 100` it** → mirror served an error page. Try Step 4 (manual download).

### Step 2 — Verify file integrity

```bash
cd ~/.cache/maxim/p4_flowers/flowers-102

# Size check on the image archive
ls -l 102flowers.tgz
# Expected: roughly 330000000 bytes (328 MB). If it's 1-2 MB or 0, it's not a real download.

# Check if it's actually a tarball, not HTML
file 102flowers.tgz
# Expected: "gzip compressed data" or similar. If it says "HTML document", it's a mirror error page.

# Known torchvision MD5 hashes — verify each file
python3 -c "
import hashlib
import pathlib
expected = {
    '102flowers.tgz': '52808999861908f626f3c1f4e79d11fa',
    'imagelabels.mat': 'e0620be6f572b9609742df49c70aed4d',
    'setid.mat': 'a5357ecc9cb78c4bef273ce3793fc85c',
}
for name, want in expected.items():
    p = pathlib.Path(name)
    if not p.exists():
        print(f'{name}: MISSING')
        continue
    h = hashlib.md5(p.read_bytes()).hexdigest()
    ok = 'OK' if h == want else f'MISMATCH (want {want})'
    print(f'{name}: {h} {ok}')
"
```

**Decision:**

- **All three match expected hashes** → the download is intact, the problem is elsewhere (Step 6).
- **Any hash mismatches or file MISSING** → proceed to Step 3 (wipe + retry).

### Step 3 — Clean slate re-download

```bash
rm -rf ~/.cache/maxim/p4_flowers

# Try the direct torchvision download in isolation, outside the audit
# script, so the error message is unambiguous
PYTHONPATH=src python3 -c "
from torchvision.datasets import Flowers102
import pathlib, os
os.makedirs(pathlib.Path.home() / '.cache/maxim/p4_flowers', exist_ok=True)
ds = Flowers102(root=str(pathlib.Path.home() / '.cache/maxim/p4_flowers'), split='test', download=True)
print(f'OK: {len(ds)} images loaded')
"
```

**Possible outcomes:**

- **Prints `OK: 6149 images loaded`** → the re-download worked. Go to Step 6 (run the audit).
- **SSL error** (`CERTIFICATE_VERIFY_FAILED` or `SSLEOFError`) → Oxford VGG's cert chain is flaky. Try Step 4.
- **HTTP 403/404** → torchvision's URL is broken or a firewall is blocking. Try Step 4.
- **Connection timeout / refused** → the leader's network can't reach `www.robots.ox.ac.uk`. Try Step 5.
- **`urllib.error.URLError` in download** → same — try Step 4 or Step 5.

### Step 4 — Manual download bypass

If torchvision's bundled download is broken, you can fetch the three files manually and drop them into the expected cache layout:

```bash
mkdir -p ~/.cache/maxim/p4_flowers/flowers-102
cd ~/.cache/maxim/p4_flowers/flowers-102

# Primary source — Oxford VGG
curl -fL -O https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz
curl -fL -O https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat
curl -fL -O https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat

# Verify sizes
ls -l 102flowers.tgz imagelabels.mat setid.mat
# Expected: 102flowers.tgz ~330 MB, imagelabels.mat ~500 B, setid.mat ~14 KB
```

If `curl` also fails with SSL errors, retry with `--insecure` (ACCEPTABLE for this one-shot dataset download because torchvision will re-verify MD5 hashes):

```bash
curl -fL --insecure -O https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz
curl -fL --insecure -O https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat
curl -fL --insecure -O https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat
```

Alternative mirror — if the VGG site is down entirely:

```bash
# HuggingFace mirrors the dataset at nelorth/oxford-flowers
# but the files are stored in a different layout (parquet instead of tgz).
# If the VGG site is dead, the manual fallback is to download from HF
# and reformat — but this is more involved. Try the primary source first.
```

After manual download, verify MD5s (Step 2 script), then unpack the tarball:

```bash
cd ~/.cache/maxim/p4_flowers/flowers-102
tar -xzf 102flowers.tgz
ls jpg/ | head   # should show image_00001.jpg ...
ls jpg/ | wc -l  # should be 8189
```

Then re-run the torchvision verification from Step 3 — if MD5s are right and `jpg/` is populated, it will succeed without trying to re-download.

### Step 5 — Network diagnosis

If the leader can't reach the Oxford VGG site at all:

```bash
# Can we resolve DNS?
getent hosts www.robots.ox.ac.uk

# Can we reach the host?
curl -sI --max-time 10 https://www.robots.ox.ac.uk/ | head -5

# Is it a proxy issue? Check env vars
env | grep -i proxy

# If you see an HTTP_PROXY set, the torchvision download probably needs
# it too — torchvision uses urllib which respects env proxy vars. Worth
# re-running Step 3 with the proxy env vars confirmed present.
```

If the network is genuinely blocked, you'll need to work around it via a sideload — `scp` the Mac peer's cache to the leader:

```bash
# From the MAC PEER, transfer the already-downloaded cache to the leader
# NOTE: this is the rescue path and explicitly allowed per the "get Phase 2E
# unblocked" goal. The Mac peer cache is at ~/.cache/maxim/p4_flowers.
# The leader's username and path may differ — adjust accordingly.
scp -r ~/.cache/maxim/p4_flowers leader_user@leader_host:~/.cache/maxim/
```

After sideload, re-run Step 3's verification script to confirm torchvision is happy.

### Step 6 — Run the audit

Once the dataset loads cleanly:

```bash
cd ~/Scripts/Maxim  # leader checkout
git pull origin main  # make sure you have the latest scripts + this runbook

# Confirm the leader's current LLM state
maxim peer llm --status

# Run the audit in a terminal you can watch
PYTHONPATH=src python scripts/p4_vram_audit.py

# Simultaneously watch nvidia-smi in another terminal
watch -n 1 nvidia-smi
```

Expected console output (approximate):

```
12:34:56 INFO P4 Stage 2 VRAM audit
12:34:56 INFO baseline: Sample(label='baseline', wall_clock_s=0.X, backend='cuda', allocated_mb=0, ...)
12:34:56 INFO loading CLIP ViT-B-32...
12:35:0X INFO after CLIP load: Sample(label='after_clip_load', ..., allocated_mb=6XX)
12:35:0X INFO loading paraphrase-mpnet-base-v2...
12:35:1X INFO after mpnet load: Sample(label='after_mpnet_load', ..., allocated_mb=9XX)
12:35:1X INFO running full mug test encoding...
12:35:3X INFO after mug test encode: Sample(label='after_mug_test_encode', ...)
12:35:3X INFO after torch.cuda.empty_cache: Sample(label='after_cuda_empty_cache', ...)
12:35:3X INFO wrote report docs/experiments/p4_vram_audit.md
```

Expected artifacts:

- `docs/experiments/p4_vram_audit.md` — human-readable report with a **VERDICT: PASS/WARN/FAIL** line in the "Headroom check" section
- `docs/experiments/results/p4_vram_audit.json` — machine-readable dump

### Step 6a — If the audit itself fails (not the dataset)

- **CUDA out of memory during CLIP load** → the baseline VRAM (LLM) + CLIP (~600 MB) doesn't fit in 16 GB. This is VALID FAILURE DATA. Report it — the headroom verdict is FAIL with the baseline numbers you captured. See "If VRAM is tight — downgrade the LLM" below.
- **`FileNotFoundError` on the fixture YAML** → you're running from the wrong directory. `cd ~/Scripts/Maxim` (or wherever the checkout is) and retry.
- **Import error on `maxim.models.vision.clip_encoder`** → `git pull origin main` — the Phase 2A code needs to be present.
- **Import error on `sentence_transformers`** → the leader is missing the `semantic` extra. Install with `maxim peer install semantic` from another machine, OR `pip install sentence-transformers torch` on the leader directly.

### Step 6b — If VRAM is tight — downgrade the LLM

If the audit reports WARN or FAIL, the user has pre-authorized downgrading the leader model to get headroom. From another terminal:

```bash
maxim peer llm qwen2.5-7b           # drop from 14B to 7B (saves ~4-5 GB)
# or
maxim peer llm mistral-7b           # alternative 7B option
```

Wait ~30-60s for the new model to load (`maxim peer llm --status` to check), then re-run the audit. The new baseline should give more headroom. Re-run the audit and report BOTH sets of numbers (14B fail + 7B pass) in the report — Stage 3 needs to know both values.

## Commit results

Once the audit completes successfully (or captures a FAIL-as-data scenario), commit the artifacts directly to main:

```bash
cd ~/Scripts/Maxim   # leader checkout on main

# Sanity check: only the audit files should be new/modified
git status

# Expected new files:
#   docs/experiments/p4_vram_audit.md
#   docs/experiments/results/p4_vram_audit.json

git add docs/experiments/p4_vram_audit.md docs/experiments/results/p4_vram_audit.json

git commit -m "$(cat <<'EOF'
docs(substrate-p4): Phase 2E — VRAM audit from RTX 5080 leader

Authoritative VRAM co-residency measurement on the RTX 5080 with
Qwen-14B (or whichever LLM was loaded — see the report's baseline
section). Complements the Mac peer reference run at
p4_vram_audit_mac_reference.md.

<< briefly summarize the verdict here — PASS/WARN/FAIL and the
   baseline LLM numbers, e.g. "Qwen-14B baseline 8.2 GB, after P4
   encoder stack load 9.1 GB, 6.9 GB free, PASS against 4.4 GB
   headroom rule" >>

Debug runbook: docs/troubleshooting/p4_vram_audit_leader_runbook.md

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"

git push origin main
```

**Important:**

- **Do NOT open a PR** — push directly to main per the user's instructions. This is a docs-only commit that a debug Claude is expected to land.
- **Summarize the verdict in the commit message body** so the Mac peer session can see the outcome at a glance without re-reading the full report.
- **If you hit a FAIL-as-data scenario**, commit that too — the failure IS the measurement, and Stage 3 planning needs to know.

## What the Mac peer session is waiting for

After you push, the Mac peer session will pull main, read `docs/experiments/p4_vram_audit.md`, and use the verdict to:

1. Close Phase 2E in Stage 2's doc
2. Decide if Phase 2F (post-merge Round 2 review) has any new findings to fold
3. Start Phase 3 planning — specifically, whether Stage 3's 20-seed three-arm sweep can run on the 5080 directly or needs a dedicated worktree with `MAXIM_LLM_ENABLED=0`

## Relevant reference files

If anything else is unclear, read these in order:

- [docs/plans/archive/substrate_p4_cross_modal_binding.md](../plans/archive/substrate_p4_cross_modal_binding.md) — the P4 plan (archived)
- [docs/experiments/p4_mug_test_sweep.md](../experiments/p4_mug_test_sweep.md) — Phase 2D results + Option 2 decision
- [docs/experiments/p4_clip_calibration.md](../experiments/p4_clip_calibration.md) — Phase 2B calibration that ran on Mac
- [docs/experiments/p4_vram_audit_mac_reference.md](../experiments/p4_vram_audit_mac_reference.md) — the Mac peer baseline you're trying to reproduce on CUDA
- [scripts/p4_vram_audit.py](../../scripts/p4_vram_audit.py) — the script itself

## Non-goals for this runbook

- Do NOT run any other Stage 2 scripts (the calibration sweep, the mug test sweep) — those are already shipped in Phase 2B/2D with Mac peer results. The VRAM audit is the only thing that NEEDS the 5080.
- Do NOT change any production code — this is a debug + run + report task. If you find a bug in `scripts/p4_vram_audit.py` itself, surface it in your commit message rather than silently patching.
- Do NOT touch the `feat/substrate-p4-stage2` branch — it's already merged to main via PR #129. Work on main directly.
