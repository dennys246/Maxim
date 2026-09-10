# pymaxim 1.2.1 — "Spoken-code loop"

**Released 2026-09-10 (UTC — PyPI `upload_time`).** `pip install --upgrade pymaxim`

A focused patch on 1.2.0 "Oasis": it completes the spoken-code device-pairing loop end to end
and clears the console contract-lag that 1.2.0 shipped with. Infrastructure only — **no
behavioral claim.**

## The spoken-code loop, end to end

1.2.0 shipped the server-side spoken-code pairing surface (A9.1) but at a contract the console
UI hadn't caught up to yet, and with no ready way to make the robot *speak* the code. 1.2.1
closes both halves:

- **Console UI re-vendored to contract 0.5.0.** The 1.2.0 wheel embedded the 0.4.0 bundle
  (pulse had no 0.5.0 build yet), so the server's 0.5.0 contract logged a mismatch warning at
  startup and drew a banner on every screen. 1.2.1 vendors the canonical **maxim-pulse v0.3.0**
  `console-dist` (contract 0.5.0) — both are gone — and the spoken-code pairing screen (the A9.1
  device sign-in) is now present.
- **`maxim.console.make_pairing_announcer`** — the one-call factory that turns a TTS engine + an
  audio sink into the `(code) -> None` announcer `build_app(pairing_announcer=…)` expects, so an
  embedder wires it without composition logic in its own bootstrap. Never logs the code (A7).
- **`maxim.utils.audio.make_device_speak_sink` / `push_audio_to_device`** — the single Reachy
  speaker sink, reachable from a bare SDK handle (the owner must pair *before* any agent exists),
  and the one place the int16→float32 conversion now lives.

A Reachy bootstrap now wires the whole loop in one line:

```python
build_app(pairing_announcer=make_pairing_announcer(tts=TTSEngine(), speak=make_device_speak_sink(reachy_mini)))
```

## A real bug fixed on the way

The Reachy SDK's `push_audio_sample` is typed `NDArray[float32]` and does no conversion, while
Piper TTS returns int16 — so pushing straight through reinterpreted the bytes as float32 (noise
at half duration). The old `MediaLoopMixin.speak` did exactly this on the device path (its local
fallback converted), so **embodied Piper speech on the Reachy speaker was affected too**, not
just the new pairing announce. The conversion now lives once in `utils.audio` and is applied on
both paths. Since A9.1's whole point is that the owner *hears* the code, this closed a silent
failure of the security property.

## Known, owed to hardware (bugs ledger D87)

Two A9.1 audio properties can only be verified on the robot and are tracked in
[docs/bugs/README.md](https://github.com/dennys246/Maxim/blob/main/docs/bugs/README.md): the
device **sample rate** (Piper is 22050 Hz; `push_audio_sample` takes no rate, so the fixed-rate
pipeline may still mis-rate even with the dtype fixed), and the **120 s pairing-TTL budget**
(synthesis + a digit-by-digit repeat, measured on a Pi). Both trigger on the first Reachy A9.1
bring-up.

## Upgrading

`pip install --upgrade pymaxim`. Leaders: `maxim peer update && maxim peer restart`.

Full changelog: [CHANGELOG.md](https://github.com/dennys246/Maxim/blob/main/CHANGELOG.md#121---2026-09-10--spoken-code-loop).
