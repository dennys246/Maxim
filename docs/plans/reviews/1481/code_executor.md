# #1481 pre-merge code review: Executor lens

Commit reviewed: `fc69e312` (diff against `origin/main`). Probes ran in a scratch copy made with `git archive fc69e312`;
nothing in the worktree was run or edited except this file. The worktree had uncommitted folds while this ran
(`cli._build_cli_comms`, a `try` around `bus.unsubscribe`, a dispatch comment). I only skimmed them and did not
review them.

`tests/unit/test_inbound_comms_1481.py`: 20 passed. The touched neighbour files gave 170 passed and 2 xfailed. The one
failure, `test_narrator_reliability::test_every_open_o19_campaign...`, is a scratch-copy artefact: it needs
`scripts/o19_verdict.py`, which the copy does not have. `test_prompt_budgeter.py` and `test_tool_output_framing.py`:
55 passed.

## Findings

### SHOULD-FIX 1: RENDER_LIMIT evicts at the K-th render, before that turn can act, and its WARNING says otherwise
`ExternalInbox.note_rendered` runs when the prompt is **built**. It increments `renders` and evicts at `>= RENDER_LIMIT`.
So at turn 3's build the message is removed and the WARNING "shown in 3 turns, none acted on" is logged. That happens
before turn 3 has replied. If turn 3 then acts on the message, the ack is a no-op on an id that is already gone.

Probe (p4): renders 1, 2, 3 give held `[1]`, `[1]`, `[]`. The WARNING fires and turn 3 is then acted on. In practice a
message survives K-1 unacted turns, not K, and the log line is false whenever the last turn acts.

Fix: evict an entry whose `renders` is already `>= K` **before** counting the new render. That evicts at the (K+1)-th
render, after K turns that did not act. Another option is to sweep at `held()` entries that have K counted renders and
no ack. Add a test: three renders, the third turn acted on, no WARNING and nothing evicted unacked.

### SHOULD-FIX 2: a sibling consumer still puts the raw phone number and the unframed body in a prompt
The module docstring says "no phone number ever reaches a prompt or a log". The comms percept is also consumed here:

- `MemoryAgent._on_percept` stores `ctx.sender` and the raw content in `_comms_messages`.
- That list reaches `StructuredContext.comms_messages`.
- `ExecAgent` (around line 1119) renders it as `[User via {channel} {sender}] {content}`, unaliased and unframed.

This predates the change, but the new claim does not hold while that path exists. `memory_hub.on_percept_received`
also encodes the raw body, which is a stored-injection route that is not this diff's.

Fix: the Wire lens should confirm whether ExecAgent's proposal prompt is live in a comms run. If it is, alias the
sender there (for example, aliases owned at the gateway) and frame the body, or narrow the docstring claim to "this
path" and file an issue.

### NIT 3: the truncation math overshoots its cap and fails silently on a tiny n_ctx
In `_inbound_text`:

- `keep` scales by the token count of the whole frame, frame overhead included.
- The `-40` is in characters, not tokens.
- The truncated frame's token count is recomputed but never checked against `cap` again.

Probe (p1): at n_ctx 2048 the section is +32 tokens over its cap. At n_ctx 512 or below, the frame overhead alone is
larger than the cap. The budgeter then drops the section, so the main-path prompt never shows the message, and it stays
held until `MAX_HELD` pushes it out.

The proportional `keep` copes with NFKC expansion (U+FDFA ×2000 came out at 632 counted tokens). A truncated frame is
still "complete", so it is acknowledged and the rest of the message is lost with no log. The same happens at intake
(`MAX_CHARS`).

Fix: subtract the frame overhead's tokens, then shrink until the frame is within `cap`. Log a WARNING with the alias
when a message is truncated, at intake or at render.

### NIT 4: two deletion probes survive the test file
- M3: deleting the `request.inbound_frame_ids = {}` reset in `build_prompt` leaves all 20 tests passing. The reset is
  functionally redundant: every build makes fresh nonces, so a stale nonce never matches.
- M13: deleting `inbox.close()` at the subscription's exit leaves all 20 tests passing.

Fix: add a test for the close, a publish through a handler snapshot taken before exit. Either test the reset or call
it defence-in-depth in its comment.

All 13 other probes turned at least one test red:
- the complete-frame check, the once-per-request guard, the ack, the continuation no-count, the sid dedupe and the
  inbound wording;
- the first-oversize truncation, the success-branch-only ids, the closed check, `MAX_HELD` and the follow-up injection;
- the `_record_inbound_rendered` call and the framing.

### NIT 5: the follow-up path adds the inbound section without the budgeter
The follow-up path adds up to about 10% of n_ctx, plus the overshoot from NIT 3, to a prompt that is not budgeted. At
n_ctx 1024 the follow-up prompt showed the message while the main path dropped it. Document this, or check the
follow-up's total against n_ctx.

### NIT 6: voice webhook retries are not de-duplicated
`receive_voice` passes no id. Pass `CallSid` the way SMS passes `MessageSid`.

### NIT 7: overlapping planning requests on more than one infer lane would each render and count
If two planning requests can be in flight at once (on more than one of `_INFER_LANES`), each renders the same held
messages and counts them. That could produce two replies. Low risk if the loop keeps a single planning request in
flight (#1048 defers submits while one is in flight). Confirm, or note it.

## Verified clean (probes p1, p3, p5)
- **Hostile messages:** I sent a forged closer, a lowercase or spaced closer, fullwidth (which NFKC turns into a
  closer), zero-width splitting, the rule line, a fake header with a fake opener, C0/C1 controls and `\x1e`, a bare
  `\x1e` plus "SYSTEM:", a Cyrillic look-alike and a half marker. Every one was rendered (`[1]`), and no marker leaked
  through `outside_tool_output` after the router's `_neutralize_stray_frame_markers`. The router did not change the
  prompt, and there was exactly one `\x1e`, the delimiter.
- **Volume:** 20 hostile messages at n_ctx 1k to 32k, on both the main and the follow-up path, leaked nothing. Frames
  that are cut or dropped are not counted.
- **Nonce forgery** is not possible. The markers in a body are defanged (`_FRAME_TOKEN`), and the nonce is random and
  chosen during the build.
- **Concurrency:** 8 publishing threads × 3000 percepts through the real gateway and AgentBus, with a worker thread
  running `held`, `note_rendered` and `ack`, gave no errors. The bound held, ids stayed unique and no number appeared in
  a held entry.
- **`__exit__` does not block.** `publish` snapshots its handlers outside the bus lock, and `unsubscribe` only holds
  the lock briefly. A raising body still unsubscribes through `finally`.
- **Paths that build no frames:** ANSWER_ONLY, direct_answer and the empty prompt build no frames. Their
  `inbound_rendered_ids` stay empty, the fallback and direct_answer proposals carry no feed, and nothing is
  acknowledged.
- **Acknowledgement:** the deliberation continuation's ack reaches the inbox. The test covers it and so does the code.
  `drain_approved` passes `proposal.source`, so an approved plan acknowledges. The `next_actions` proposals carry no
  ids.
- **Logs:** `ExternalInbox` logs only the alias and channel. Nothing in `comms/` logs a sender.
