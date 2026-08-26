# H2 — Loudness bench (roadmap item 18): is sound level measurable on the Reachy Mini?

**Status:** DONE 2026-08-25 — both bench tests answered in one 75 s trace; **no
design shipped** (item 18 deferred to 1.1.1 by decision, see the roadmap).
**Apparatus:** [scripts/orient_backbone/loudness_bench_poll.py](../../scripts/orient_backbone/loudness_bench_poll.py)
against the live daemon at `10.6.0.63:8000` (SDK == daemon 1.8.3, healthy-motor robot).
**Raw data (S4):** [data/h2_loudness_bench.jsonl](data/h2_loudness_bench.jsonl)
(289 samples, 0 errors, ~4 Hz). Operator: one speaker, one room, n = 1 session.

## Why the roadmap's framing was wrong

The roadmap (§"Loudness — blocked outside this repo") said the only level signal would
need either Pollen's daemon code or onboard PCM under `media_backend: default`. Both
are false. The XVF3800 already computes a level, Pollen's SDK already tables it, and
the daemon we run already serves it over REST:

- `AEC_SPENERGY_VALUES` — four floats, per-beam *speech energy* ("indicates whether
  speech is present in the beam as well as the amplitude … higher values indicate
  louder or closer speech" — Seeed's host-control README). Listed read-only in
  `reachy_mini/media/audio_control_utils.py` at `(33, 80, 4, "ro", "float")`, one
  register over from `DOA_VALUE_RADIANS`, which is all `get_DoA()` reads.
- `GET /api/audio/config/parameter/{name}` — any table entry, any client on the
  network; each request opens a short-lived USB handle. Landed in reachy_mini
  `8a382c1f` (2026-05-20), present at tag `v1.8.3`.
- `PP_AGCONOFF` / `PP_AGCGAIN` / `PP_AGCDESIREDLEVEL` / `PP_AGCMAXGAIN` — the
  post-processing AGC, readable and writable the same way.

Snapshot in silence: spE `[0, 101457, 88497, 101457]`, AGC on, gain 53.8 / max 64,
desired level 0.0045, `AUDIO_MGR_MIC_GAIN` 90.

## Protocol (operator-reported, anchored on the AGC trace)

Quiet ~20 s → rising loudness ~20 s → silence ~10 s → loud speech ~20 s → quiet.
The two loud episodes are unmistakable in `PP_AGCGAIN` (t ≈ 10–30 s and 46–58 s).

## Bench test (a) — is a level signal available? YES, two, of different character

| t (2 s bins) | AGC gain | auto-beam spE max | speech flag | phase |
|---|---|---|---|---|
| 0–8 | 41.6 → 45.8 | 90k–270k | 1–3/8 | quiet (ambient floor) |
| 10–14 | 22 → 12.8 | 590k–**1.10M** | 0–4/8 | first raise |
| 16–30 | **12.5** (pinned) | 0–380k, often **0** | 0–2/8 | still loud; VAD mostly 0 |
| 32–44 | 24 → 38 | 0–220k | 0–3/8 | silence, gain releasing |
| 46–58 | 14 → **8.3** → 12 | 360k–**859k** | 7/8 → 1/8 | loud speech |
| 60–74 | 12 → 45.6 | 44k–270k | 0–2/8 | quiet, gain releasing |

- **`PP_AGCGAIN` readback is a clean, graded, inverse loudness envelope.** Quiet
  42–46; first raise → 12.5 within ~4 s; loud speech → 8.3 (lower than the first
  raise — graded, not binary); release back to ~45 over ~15 s. Monotone, no spikes.
  Fast attack (2–4 s), slow release (~15 s): a *room-got-loud* signal on a seconds
  timescale, not an onset detector.
- **`AEC_SPENERGY_VALUES[3]` (auto beam) is a spiky, speech-gated magnitude.**
  Window-max over 2 s separates loud speech (600k–1.1M) from the quiet floor
  (90k–270k) by 3–4×, but per-sample medians overlap and it read **exactly 0** for
  much of 16–24 s — inside the first loud phase, AGC pinned at its floor, `speech_detected`
  0/8. It is what the vendor says it is: speech energy gated by the VAD. Loud
  non-speech (or speech the VAD did not vote for) → 0. Usable as "how loud is the
  speech I am detecting", not as a general level.

## Bench test (b) — does the AGC flatten it? NO for the register, YES for PCM

Speech energy is *largest* exactly where AGC gain is *smallest* (t = 48: gain 9.9,
spE 859k; t = 14: gain 12.8, spE 1.10M). A post-AGC measure would sit near the
constant desired level. So the register is pre-AGC, as its `AEC_` namespace (resource
33, vs the AGC's `PP_` resource 17) suggested. The PCM a host receives is post-AGC by
construction (`PP_AGCDESIREDLEVEL = 0.0045`), so the RMS-over-PCM path the roadmap
contemplated would have been flattened ~5× across these two raises. The roadmap's worry
was correct about PCM and irrelevant to the register.

## Apparatus cost

Per-read latency medians 52 / 46 / 23 ms (spE / AGC / DoA), one 1.26 s outlier;
`/api/state/doa` kept answering at 23 ms throughout, so three extra USB handle
open/closes per 250 ms did not starve the daemon's own DoA reader. Not measured: the
cost inside a live orient session that also streams WebRTC audio.

## Caveats (n = 1, stated so the 1.1.1 plan inherits them)

- One session, one room, one speaker; no calibrated SPL reference — the register is
  relative energy with no units (Seeed's example reads ~2e6).
- The `12.5` plateau recurs often enough to look like an AGC adaptation state rather
  than a hard floor; irrelevant to whether the signal exists, worth a look before
  anyone thresholds on it.
- Speech energy is per-beam; the auto-selected beam (`[3]`) is the one to read, and
  it can disagree with the free-running beam (`[2]`) by 2× on a given sample.

## What this unblocks (1.1.1, not 1.1.0)

Salience in the DoA feed becomes a function of onset + level using two REST reads on
the daemon we already talk to — `PP_AGCGAIN` for the graded envelope,
`AEC_SPENERGY_VALUES[3]` for speech-gated onset magnitude — riding the existing
`percept.salience` field. No PCM, no `media_backend: default`, no vendor dependency,
no new mechanism. The *forced* startle-look reflex tier stays 1.3.
