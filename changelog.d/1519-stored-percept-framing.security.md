- **Stored outside text reaches a prompt only inside an untrusted-content frame, and a live voice request is labelled
  on its first prompt** (#1519, #1516 F15). Framing covers stored and recapped text; the turn's own live request is labelled, not framed.
  - Live routes closed: a recalled episode's goal and the ATL names under "Related concepts" (enrichment, so the
    enrichment section, working-memory thoughts and the deliberation transcript), a tool's result text in
    `recent_outcomes`, and the file names in the CWD listing.
  - Latent routes closed: the working-memory recap (the live loop's `enrich()` passes no working memory, so it was
    one wiring change from the prompt), PromptBuilder's Dormant memory sections and ExecAgent's Dormant prompt, which
    also no longer shows an inbound message's sender.
  - Provenance now travels with each working-memory entry: `WorkingMemorySet.add` requires `provenance=` (operator,
    external or internal) and `WMEntry.prompt_text()` renders by it. Only `cli` (typed input, the sim narrator too)
    counts as the operator, shown defanged. Stored room-mic `voice` (any room-mic line naming the wake word) and
    `transcript` (owner decision 2026-11-21: anyone in the room or on a speakerphone can produce either) and every
    `comms:*` source are external, shown framed. One helper renders it all: `memory/text.py::frame_external`.
  - A live voice request stays the turn's request, so voice still commands the robot. On the FIRST prompt of a voice
    turn, its `=== User Request ===` is labelled as heard on the room microphone from an unverified speaker, not
    typed by the operator. The loop carries the source with the request (`LLMRequest.triggering_input_source`) and
    never reads it from the text.
  - Not labelled yet (tracked in #1729, the voice-gate PR, on #1185's `InputSource`):
    - deliberation re-prompts;
    - follow-up prompts;
    - inputs held while a submit is in flight;
    - the deferred `=== New Instructions ===`;
    - conversation history.
  - Gating effectful tools on a voice-sourced turn behind typed confirmation is designed, not built
    (`docs/plans/voice_turn_gate_1519_approach.md`).
  - Cloud redaction no longer turns a frame's closer into `[REDACTED_PATH]`. Only a COMPLETE frame counts, by the
    one pairing rule every frame consumer now shares (`content_safety.frame_spans`): an opener whose closer never
    comes, one repeated before its closer, or one whose span would cross a section header (an opener forged in
    unframed text with a live id) is not a frame, so every private header after it is still redacted. A frame's
    body never holds a section header line: `frame_tool_output` turns every line break `str.splitlines` honours
    (U+2028 included) into a newline and quotes every line the one header predicate
    (`content_safety.is_section_header`: `=== Name ===` or `NAME:`) calls a header, and the redactor finds its
    sections with that same predicate and splitter. A private-shaped header INSIDE a complete frame is no longer
    redacted: the framed body is external text, not the agent's private section.
  - Stored records holding a frame written before this change, whose body had a bare `NAME:` line, lose their keyed
    text: the pairing rule refuses such a frame, so the whole value keys to nothing (accepted, fail-closed, owner
    decision 2026-11-21). New records are quoted correctly.
  - A recent outcome's result is framed as its follow-up is: external for a network tool or for a tool that replays
    stored text (`untrusted_output`). A system-written note (`SYSTEM: …`) is shown unframed as the agent's own
    guidance, and is checked to carry no outside text.
  - A truncated CWD listing is cut inside its frame, so no file name is ever shown outside it.
  - The enrichment log's ATL line carries concept ids, not names.
