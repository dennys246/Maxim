- **Stored outside text reaches a prompt only inside an untrusted-content frame** (#1519, #1516 F15).
  - Live routes closed: a recalled episode's goal and the ATL names under "Related concepts" (enrichment, so the
    enrichment section, working-memory thoughts and the deliberation transcript), a tool's result text in
    `recent_outcomes`, and the file names in the CWD listing.
  - Latent routes closed: the working-memory recap (the live loop's `enrich()` passes no working memory, so it was
    one wiring change from the prompt), PromptBuilder's Dormant memory sections and ExecAgent's Dormant prompt, which
    also no longer shows an inbound message's sender.
  - Provenance now travels with each working-memory entry: `WorkingMemorySet.add` requires `provenance=` (operator,
    external or internal) and `WMEntry.prompt_text()` renders by it. Only `cli` (typed input, the sim narrator too)
    counts as the operator, shown defanged. Room-mic `voice` (any room-mic line naming the wake word) and
    `transcript` (owner decision 2026-11-21: anyone in the room or on a speakerphone can produce either) and every
    `comms:*` source are external, shown framed. One helper renders it all: `memory/text.py::frame_external`.
  - Cloud redaction no longer turns a frame's closer into `[REDACTED_PATH]`, and a framed line shaped like a section
    header (`=== Notes ===`) can no longer end a private section's redaction early, which had sent the rest of it.
    Only a COMPLETE frame counts: an opener whose closer never comes (a cut) changes nothing, so every private
    header after it is still redacted. A private-shaped header INSIDE a complete frame is no longer redacted: the
    framed body is external text, not the agent's private section.
  - A truncated CWD listing is cut inside its frame, so no file name is ever shown outside it.
  - The enrichment log's ATL line carries concept ids, not names.
