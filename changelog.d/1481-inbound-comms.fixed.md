- **A message sent to Maxim over SMS, voice or `POST /api/command` now reaches the model, framed as untrusted data**
  ([#1481](https://github.com/dennys246/Maxim/issues/1481); design: `docs/plans/inbound_comms_1481_approach.md`, four
  adversarial rounds). The comms stack published each message on the agent's bus, but the live loop never read that
  bus and the prompt builder never rendered comms, so a message was accepted, logged and never shown, and nothing said
  so.
  - The run's loop now holds `comms:*` messages in a bounded inbox (`runtime/loop_external.py`):
    - at most 20 held;
    - Twilio retries de-duplicated by `MessageSid`;
    - each sender shown as a per-session alias ("sender 1"): a phone number never reaches the live prompt or a log
      (the dormant `ExecAgent` path still would; #1519).
  - The next prompt the loop builds for another reason shows them in an `inbound_messages` section. Each message sits in
    its own untrusted-data frame, with the alias in the frame header, inside a token cap of about 10% of the context.
    The rule line says what the frames hold.
  - A message leaves the inbox only when a turn whose prompt showed it whole is acted on (dispatched). A timeout, an
    empty or fallback reply, a stale or discarded proposal, a dropped section or a failed submit all keep it held. A
    message shown in 3 turns without being acted on is evicted, with a WARNING.
  - Messages never trigger a turn by themselves yet.
  - **`POST /api/command` text is shown as outside data, never as an operator instruction**, even though it needs the
    API token (owner decision 2026-10-28).
  - `bootstrap.build_comms_stack` takes `agent=` (it raised nothing before and took `bus=`), so comms can only be
    wired to the bus the loop watches.
  - Working-memory thoughts still render stored comms text unframed: that is #1519.
