- **`robots.yaml` is strict and typed** (#1806 PR 1a).
  - The top level holds only `robots` and `_format_version`.
  - A duplicate key anywhere is refused; YAML used to keep the last one silently. A `<<:` merge override is still
    allowed.
  - A robot id must be a name: `yes:` used to become the id `'True'`.
  - An entry's keys are exactly `type`, `primary`, `config`, `kind`, `hardware_id`, `hostname` and `launch`. Any
    other key, or `hardware_id`/`hostname` under `config:`, is now a `RobotsConfigError`; it used to be dropped
    silently.
  - `type:` must be a string (it was coerced). `primary:` must be an unquoted `true`/`false`: a quoted `"false"` used
    to count as true. Marking more than one entry primary is refused; the first one used to win.
  - A root `_format_version` makes `hardware_id:` required on every entry. A malformed or newer `_format_version`
    is refused.
- **A networked Reachy entry can no longer ask for this machine's camera** (#1806 PR 1a, owner rule).
  - A `media_backend` of `local`, or any alias the SDK resolves to it (`gstreamer`, `gstreamer_no_video`,
    `sounddevice_no_video`, `sounddevice_opencv`), is refused unless the entry is `connection_mode: localhost_only`
    without `tunnel: true`. That covers an absent mode (network), `auto`, and the run's own `media_backend` default.
    Before, such an entry streamed the Maxim host's camera and audio as the robot's.
  - Unknown, mistyped or null `connection_mode`/`media_backend` values and a non-bool `tunnel:` are refused rather
    than left to the SDK, which auto-detects on an unknown backend. Values are matched case-insensitively, as the
    SDK does.
  - Two paths remain until #1806 PR 1b's controller work: a `tunnel: true` entry with `default`/`auto` media can
    still pick this machine's camera (the controller forces such an entry to `localhost_only`), and a manual SSH
    tunnel written as `localhost_only` is accepted.
- **A `hardware_id:` pin is parsed but not yet verified, so a pinned entry refuses to connect** (#1806 PR 1a). The
  run (exit 2, `RobotPinUnverifiable`), the bulk connect and `maxim doctor` report the same refusal.
  `hardware_id: unpinned`, or no line, connects as before. Verifying the pin on connect is #1806 PR 1b, and nothing
  writes `hardware_id:` yet. Until #1806 PR 1c's pinned-registry rule, a run whose robot id matches no entry (with no
  primary) still connects with defaults, unpinned, as before.
- **These checks also hold for a `RobotConfig`/`RobotsConfig` built directly** (the `maxim.hardware` exports), not
  only for a loaded file.
