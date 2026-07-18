# Vision behaviors — the detection-driven path

This is the **mature, shipped** behavior path. Every behavior in
[`default_network/behaviors/`](../../src/maxim/default_network/behaviors/) today is visual: it reads
YOLO detections and proposes a gaze/motion action. Read [README.md](README.md) first for the general
`Behavior` contract; this doc covers the vision-specific input shape and the shipped examples.

## Input: the detection dict

`evaluate(detections, state)` receives `detections: list[dict]` — YOLO detections for the current
frame. Common keys (defensive `.get()` access — not every producer fills every key):

| key | meaning |
|---|---|
| `track_id` | persistent tracker ID (may fragment during head motion) |
| `class_id` | object class integer |
| `conf` | detection confidence |
| `bbox_xyxy` (or `bbox`) | `[x1, y1, x2, y2]` pixel box; center `u = (x1+x2)/2` |

The frame is gated **before** behaviors run: [`_process_tick`](../../src/maxim/default_network/network.py)
returns at `if not detections: return`, so vision behaviors only ever see a non-empty detection list.
This is exactly why the current tick is "visual-only" — a modality with no detections (audio) never
reaches behavior evaluation. (That's the audio path's first blocker; see
[audio_behaviors.md](audio_behaviors.md).)

## Targets are pixel coordinates — and gate your proposal

Vision proposals carry `target=(u, v)` pixel coordinates. This matters beyond aiming: the salience,
IOR (inhibition-of-return via `_gaze_history`), and fear gates all operate in pixel space and
**short-circuit to "allow" when `target` is falsy**. Because vision behaviors supply a real pixel
target, they get novelty/IOR gating for free — the robot doesn't re-fixate the same spot repeatedly.
A behavior that omits the target loses that gating silently.

Use `state.get_tracking_bonus((u, v))` / `state.record_tracking_target((u, v))` for tracking
hysteresis when you commit to a detection.

## Shipped examples (read these before writing a new one)

| behavior | file | what it does |
|---|---|---|
| `OrientingResponse` | [`orienting.py`](../../src/maxim/default_network/behaviors/orienting.py) | look at the most novel object |
| `SocialAttention` | [`social.py`](../../src/maxim/default_network/behaviors/social.py) | track people/faces |
| `MotionTracking` | [`motion.py`](../../src/maxim/default_network/behaviors/motion.py) | follow moving objects |
| `StartleResponse` | [`startle.py`](../../src/maxim/default_network/behaviors/startle.py) | quick reaction to sudden peripheral appearance |
| `TurnAround` | [`turn_around.py`](../../src/maxim/default_network/behaviors/turn_around.py) | rotate body when head hits its yaw limit with interest beyond |
| `IdleScan` / `Microsaccades` / `ReturnToCenter` | [`idle.py`](../../src/maxim/default_network/behaviors/idle.py) | idle-time movement |

### `TurnAround` is the closest precedent for a body-rotation reflex

It is the one shipped behavior that rotates the **body**, and it demonstrates both the setter pattern
and a live pitfall, so study it:

- **Setter pattern** — it needs the current head yaw, which isn't in `detections`, so DN pushes it in
  via `set_head_yaw()` each tick ([`turn_around.py:92`](../../src/maxim/default_network/behaviors/turn_around.py),
  pushed at [`network.py:871`](../../src/maxim/default_network/network.py)). Copy this shape for any
  external state.
- **The yaw sanity guard** — it ignores `|head_yaw| > 90°` as a corrupted (world-frame) reading
  ([`turn_around.py:115`](../../src/maxim/default_network/behaviors/turn_around.py)). This is a scar
  from the head-frame confusion; a body-rotation behavior reading head pose must know whether it's
  world- or body-frame.
- **The dispatch pitfall** — its proposal (`action_type="turn_around"`) routes to
  `_maxim.turn_around()`, which **blocks the DN thread** for the full multi-second turn and dispatches
  `body_yaw` with **no head matrix** (the head=None counter-rotation). That's tolerable for a slow,
  rare "look behind me" gesture but is **wrong for a fast reflex** and wrong for anything that reads a
  head-mounted sensor afterward. A new body reflex should use a **non-blocking** dispatch that ships a
  head matrix — do not reuse `turn_around`.

## Adding a vision behavior — the short version

1. Subclass `Behavior`; key off `class_id` / novelty / `bbox` from `detections`.
2. Return `_create_proposal(action_type="look_at"|"track"|"scan", target=(u, v), ...)`.
3. `look_at` / `track` / `scan` already have motor branches — no new dispatch needed.
4. Register in the DN `behaviors=[...]` list and export from `__init__.py`.
5. Offline-test with a synthetic `detections` list.

Because the vision path's `action_type`s, gates, and input gate all already exist, a new **vision**
behavior really is "just one Behavior." That is *not* true of a new modality — see the audio doc for
what a non-visual behavior additionally has to build.
