# #1801: URLs reduced at the boundaries, not per call site (approach note)

> **Status:** v29 2026-11-25 (Session A). **DESIGN SETTLED: the stop rule held at round 27.** Code built; code review round 1 folded (entry at the end of this block). On on origin/main after #1775 (b53f804e). This is a security boundary, so the adversarial design pass comes before any
> code (CLAUDE.md "Faster without lowering the bar" (1)).
> - **Round 1 (red-team, a different reader):** 2 DO-NOT-BUILD and 6 SHOULD-FIX, all folded below and tagged R1–R8.
>   - FAIL-OPEN: R1 (the factory cannot see `extra=`), R2 (the sink map was missing family 4), R4 (a later factory
>     that does not chain), R7 (`redact_urls` gaps) and R8 (`sim_log` `data` leaves).
>   - FAIL-CLOSED or test-only: R3 (cost and fidelity of rendering at creation), R5 (the claim about children) and
>     R6 (the factory leaking across tests).
>   - Not converged yet, because a new class turned up (family 4), so a delta round is owed on v2.
> - **Round 2 (delta, a different reader):** 0 DO-NOT-BUILD, 6 SHOULD-FIX and 3 NITs, all folded and tagged S1–S6 and
>   N1–N3.
>   - FAIL-OPEN: S1 and S2 (family 4's map, class R2) and S3 (serialization, a NEW class).
>   - S5 (module placement and an import cycle) is a NEW class too, but FAIL-CLOSED.
>   - Converging (no DO-NOT-BUILD; most findings narrower, inside R2/R3/R7). A delta round 3 is owed on S1–S6 only.
> - **Round 3 (delta, a different reader):** 0 DO-NOT-BUILD, 3 SHOULD-FIX and 3 NITs, folded and tagged F1–F3 and
>   Na–Nc.
>   - FAIL-OPEN: F1 (class S3), F2 (cut before reduce, a NEW class), F3 (class R2) and Na (class R7).
>   - Clean: S5's module move.
>   - Converging: findings are narrower each round, and only one new class (F2) appeared. A delta round 4 is owed on
>     F1–F3 only.
> - **Round 4 (delta, a different reader):** 0 DO-NOT-BUILD, 4 SHOULD-FIX and 2 NITs, folded and tagged G1–G6.
>   - FAIL-OPEN: G1 (the F2 lint), G2 (class R2), G3 (class S3) and G5 (class R7, narrow).
>   - NO new class. Converging. A delta round 5 is owed on G1–G6 only.
> - **Round 5 (delta, a different reader):** 0 DO-NOT-BUILD, 4 SHOULD-FIX and 2 NITs, folded and tagged H1–H6.
>   - FAIL-OPEN: H1 (class F2), H2 (class S3), H4 (class R2) and H6 (class R7, narrow).
>   - No new class; every finding is now a site-list completeness question or a mechanic made precise. A delta
>     round 6 is owed on H1–H6 only.
> - **Round 6 (delta, a different reader):** 0 DO-NOT-BUILD, 3 SHOULD-FIX and 4 NITs, folded and tagged J1–J7.
>   - FAIL-OPEN: J1 (class S3), J2 (class R2) and J3 (class F2). J5 and J6 are latent.
>   - All findings are in covered classes, and each is one mechanic made precise.
> - **Owner, 2026-11-25:** keep the stop rule literal. Design rounds continue until one comes back all fail-closed.
> - **Round 7 (delta, a different reader):** 0 DO-NOT-BUILD, 3 SHOULD-FIX and 3 NITs, folded and tagged K1–K6.
>   - FAIL-OPEN: K1 (class S3, latent), K2 and K3 (class F2). No new class.
>   - A delta round 8 is owed on K1–K6 only.
> - **Round 8 (delta, a different reader):** 0 DO-NOT-BUILD, 2 SHOULD-FIX and 4 NITs, folded and tagged L1–L6.
>   - FAIL-OPEN: L1 (class F2), and L2 (class F2, latent: 0 such sites in src).
>   - K1 holds: no `__dict__`/pickling receiver in `src/`, and nothing reads `record.args`/`record.msg`.
>   - A delta round 9 is owed on L1–L5 only.
> - **Round 9 (delta, a different reader):** 0 DO-NOT-BUILD, 2 SHOULD-FIX and 3 NITs, folded and tagged M1–M5.
>   - FAIL-OPEN: M1 (class F2, latent: 0 sites) and M2 (class F2, how the allowlist is keyed).
>   - L1, L3, L4 and L5 hold. The allowlist key was the only fix site not yet specific enough to build; M2 settles
>     it.
>   - A delta round 10 is owed on M1–M5 only.
> - **Round 10 (delta, a different reader):** 0 DO-NOT-BUILD, 3 SHOULD-FIX (all FAIL-OPEN, all latent: 0 sites in
>   src) and 1 NIT, folded and tagged P1–P4. P1's logger cut is now fixed at the root, in the factory, not in the
>   lint. M1, M3, M4 and M5 hold.
> - **Owner, 2026-11-25 (again, after round 10):** keep the stop rule literal. Every lint-evasion form counts as
>   fail-open.
> - **Round 11 (delta, a different reader):** 0 DO-NOT-BUILD, 2 SHOULD-FIX and 2 NITs, folded and tagged Q1–Q4.
>   - FAIL-OPEN: Q1 and Q2 (class F2) and Q3 (lint evasion, 0 sites).
>   - P1 holds for every format directive probed, and the proxies are lazy.
>   - A delta round 12 is owed on Q1–Q4 only.
> - **Round 12 (delta, a different reader):** 0 DO-NOT-BUILD, 2 SHOULD-FIX and 2 NITs, folded and tagged T1–T4.
>   - FAIL-OPEN: T1 (class F2), T2 (lint evasion, 0 sites) and T4 (`Path` collapsing `://`, a NEW class, latent).
>     T4 is FIXED in `redact_urls`, not merely stated.
>   - Nested containers, Mapping keys, split URLs and Q2/Q4 hold.
>   - A delta round 13 is owed on T1–T4 only.
> - **Round 13 (delta, a different reader):** 0 DO-NOT-BUILD, 3 SHOULD-FIX and 2 NITs, folded and tagged U1–U5.
>   - FAIL-OPEN: U1, U2 and U3 (covered classes).
>   - U5 (`PureWindowsPath`) is tagged fail-closed by the reader but does leak, so it is fixed, not stated.
>   - A delta round 14 is owed on U1–U5 only.
> - **Round 14 (delta, a different reader):** 0 DO-NOT-BUILD, 3 SHOULD-FIX and 3 NITs, folded and tagged V1–V6.
>   - FAIL-OPEN: V1 (the composed regex), V2 (bind-then-cut), V4 and V6 (covered classes). V6 is fixed, not stated.
>   - The regex is now PINNED (V1): the reader's prototype passes 104/104 existing redaction tests.
>   - A delta round 15 is owed on V1–V6 only.
> - **Round 15 (delta, a different reader, probed on Python 3.10.19 and 3.12):** 1 DO-NOT-BUILD, 2 SHOULD-FIX and 1 NIT,
>   folded and tagged W1–W4.
>   - W1 (DO-NOT-BUILD, FAIL-OPEN): v15's pin restricted EVERY scheme to http/ws/ftp, so `postgres://`, `redis://`,
>     `git+ssh://` and `unix://` passed raw, and it dropped the whitespace-password branch. Its "104/104" was false
>     for the regex as written (2 of 104 failed).
>   - W1's fold was VERIFIED BY THE AUTHOR, not by prose: the regex below, patched into a scratch copy, passes all 104
>     redaction tests, gives identical matches on 3.10 and 3.12, and leaks nothing on any form in this note. The
>     patch is at `scratchpad/regex1801/regex_candidate.patch`.
>   - W2 (FAIL-OPEN, lint evasion) is folded; W3/W4 are fail-closed.
>   - A delta round 16 is owed on W1–W4 only.
> - **Round 16 (delta, a different reader; re-ran the author's verification: 104/104 on 3.10.19 and 3.12.12):** 0
>   DO-NOT-BUILD, 3 SHOULD-FIX (FAIL-OPEN, covered classes) and 2 NITs, folded and tagged X1–X5.
>   - X1 and X2 (the whitespace-password branch) are FIXED in the regex, not stated. The author re-verified:
>     104/104, and 3.10 compiles it.
>   - A delta round 17 is owed on X1–X5 only.
> - **Round 17 (delta, a different reader; reproduced 104/104 on 3.10.19 and 3.12):** 0 DO-NOT-BUILD, 5 SHOULD-FIX
>   (4 FAIL-OPEN in covered classes, 1 FAIL-CLOSED) and 1 NIT, folded and tagged Y1–Y6. Y1–Y3 are verified by the
>   author on the scratch patch: 104/104, 3.10 compiles, linear.
> - **Owner exceptions, 2026-11-25** (machine-readable; the code PR's boundary tests cite each by id):
>
>   ```yaml
>   owner_exceptions:
>     - id: E1801-1
>       accepts: "on EITHER branch, userinfo containing whitespace leaks its tail when the whitespace lies more than 640 non-space characters after the scheme separator, or the `@` lies more than 640 characters after that whitespace (the catch-all's two bounds; the earlier per-form bounds are superseded by the owner's simplify decision, 2026-11-25)"
>       why: "a bounded regex is linear; an unbounded one is quadratic on adversarial lines inside the logging path; httpx percent-encodes userinfo, so real transport errors never carry such text"
>       class: fail-open residual, accepted
>       pinned_by: "the catch-all's two bounds, 640/641 each, on each branch (reduced at the bound; past it the leak is asserted). Superseded per-form pins: one test per bound, at the bound and one past it (reduced at the bound; past it the leak is asserted, so a change is noticed): form 1 user 64/65, password-before 256/257, password-after 256/257; form 2 user-before 64/65, user-after 64/65, password 256/257; form 3 before 64/65, after 64/65 (Z6); the same pins on the separator branch (AC1). Each pin plants a distinct marker in EVERY part and compares case-insensitively, since past a bound only part
        of a secret leaks and `url_origin` lowercases a host (AA4)"
>     - id: E1801-2
>       accepts: "a URL followed within the catch-all's bounds by whitespace and ANY `@` is erased to `<url>`, the span through that `@` and the rest of its token (fail-closed; widened by the owner 2026-11-25; measured: 4.8% of 20-line src/ windows holding a URL, mostly `@property` near URL f-strings in tracebacks; 1.1% of doc paragraphs; 0 of 14 URL lines in real sim reports)"
>       why: "fail-closed: no leak and no wrong host; Maxim's own log formats never put a URL next to an `@` (0 grep hits)"
>       class: fail-closed over-match, accepted
>       pinned_by: "tests for `see https://a.com and user:x@y`, `https://h:8080 failed, mail bob@ex.com`, `https://u:p@h/x, contact a@b` and a cross-line case"
>     - id: E1801-3
>       accepts: "an unlisted scheme (outside http(s)/ws(s)/ftp(s) and the stated list: postgres, postgresql, mysql, redis, rediss, mongodb, mongodb+srv, amqp, amqps, ssh, sftp, git+ssh, s3) is not recognised in the lone-`/` Path form or the bare-backslash Windows form (`zzz:\\u…`), so its userinfo can pass raw; any scheme IS recognised with a JSON-escaped slash (`zzz:\\/\\/…`) (owner, AG2)"
>       why: "taking any scheme in the lone-`/` or bare-backslash forms would erase ordinary log text such as `path:/Users/me` and `nearby:\\n` (59 hits in 1.24M real sim-report fields, AG2); main recognises none of these forms today"
>       class: fail-open residual, accepted (no worse than main)
>       pinned_by: "tests that Path('/a/redis://:SECRET@h') and `postgres:\\u:SECRET@h` reduce (listed schemes), and that Path('/a/zzz://:SECRET@h') and `zzz:\\u:SECRET@h` leak (both asserted, so a change is noticed) (AH3)"
>     - id: E1801-4
>       accepts: "a URL start (`scheme://`, or a separator form) inside a whitespace-holding userinfo, between the whitespace and its `@`, ends the outer match there, so the outer userinfo leaks: `https://tok SECRET https://x@h` gives `https://tok SECRET https://h` (owner 2026-11-25, AG1)"
>       why: "tempering (AF1) is what stops one URL's match from swallowing a later URL's userinfo and leaking ITS secret (round 25, 12/12 combinations); the residual it leaves is rarer (a URL-shaped string inside a password with whitespace); main leaks both"
>       class: fail-open residual, accepted
>       pinned_by: "a test asserting the leak for `https://tok SECRET https://x@h` (so a change is noticed) and that each URL in round 25's two-URL set reduces"
>   ```
>   - A delta round 18 is owed on Y1–Y6 only. E1801-1 and E1801-2 count as covered, not as fail-open.
> - **Round 18 (delta, a different reader; 104/104 on 3.10.19 and 3.12):** 0 DO-NOT-BUILD, 3 SHOULD-FIX (2 FAIL-OPEN in
>   covered classes, plus 1 quadratic family) and 3 NITs, folded and tagged Z1–Z6. It also flagged the stale "Gaps,
>   stated" list, whose two gaps lay outside E1801-1's wording. Both are now FIXED in the regex, so the list is
>   gone. The author re-verified the three-form regex on the scratch patch: 104/104, 3.10 compiles, linear in
>   every family (256k characters in at most 218 ms).
>   - A delta round 19 is owed on Z1–Z6 only.
> - **Round 19 (delta, a different reader; 104/104 on 3.10.19 and 3.12):** 1 DO-NOT-BUILD, 2 SHOULD-FIX and 2 NITs,
>   folded and tagged AA1–AA5.
>   - AA1 (DO-NOT-BUILD, FAIL-OPEN): Z2 was a REGRESSION. Excluding `\` from the plain userinfo leaked a
>     JSON-escaped quote in a password (`https://u:p\"SECRET@h/p`), which today's main reduces. The fold
>     (a conditional group) was verified by the author on the scratch patch: 104/104, 0 leaks across every case in
>     this note, 3.10 compiles, at most 179 ms at 256k characters.
>   - A delta round 20 is owed on AA1–AA5 only.
> - **Round 20 (delta, a different reader; 104/104 on 3.10.19 and 3.12):** 1 DO-NOT-BUILD, 1 SHOULD-FIX and 2 NITs,
>   folded and tagged AB1–AB4.
>   - AB1 (DO-NOT-BUILD, FAIL-OPEN): the conditional reopened the escaped-quote leak on the separator form itself
>     (`https:\/\/u:p\"SECRET@h\/p`).
>   - The reader's fold needed a wider E1801-1. The author took a STRUCTURAL fold instead, which needs no exception:
>     a separator-form match can only ever reduce to `<url>` (`url_origin` cannot parse it), so that branch takes the
>     WHOLE non-space token and parses no userinfo at all.
>   - Verified by the author on the scratch patch: 104/104, 0 leaks across 25 cases, 3.10 compiles, separator families
>     about 1 ms at 256k characters.
>   - A delta round 21 is owed on AB1–AB4 only.
> - **Round 21 (delta, a different reader; 104/104 on 3.10.19 and 3.12):** 1 DO-NOT-BUILD, 1 SHOULD-FIX and 1 NIT,
>   folded and tagged AC1–AC3.
>   - AC1 (DO-NOT-BUILD, FAIL-OPEN): AB1's `\S+` stopped at the first space, so whitespace userinfo on the separator
>     branch leaked (`https:\/\/u:p SECRET@h\/p`, a `Path` URL with a space in the password, the Windows form). It
>     was a regression against v20, and AB1's "no in-bounds leak is possible" claim was false.
>   - The fold puts the three bounded whitespace forms on the separator branch too. The author verified it: 104/104,
>     0 leaks across 33 cases, 3.10 compiles, linear.
>   - **Owner, 2026-11-25:** E1801-1 extends to EITHER branch.
>   - A delta round 22 is owed on AC1–AC3 only.
> - **Round 22 (delta, a different reader; 104/104 on 3.10.19 and 3.12.12):** 2 SHOULD-FIX (FAIL-OPEN) and 1 NIT, folded
>   and tagged AD1–AD3.
>   - AD1 (class AC1): separator mixes the lookahead did not offer leaked, as did JSON-in-JSON.
>   - AD2 (a NEW class, on both branches, in bounds): a username holding an unencoded `@` (an email) plus a
>     whitespace password leaked: `https://bob@ex.com:p SECRET@h/p`.
>   - The reader's folded tree was verified by the author: 104/104, 0 leaks across every case set, and 3.10 compiles
>     and matches.
>   - A delta round 23 is owed on AD1–AD3 only.
> - **Round 23 (delta, a different reader; 104/104 on 3.10.19 and 3.12.12; AD1 held against the reader's own
>   families):** 2 SHOULD-FIX (FAIL-OPEN, covered classes), folded and tagged AE1–AE2.
>   - AE1: form 3 lets the user hold an `@`. The author verified it: 104/104, 0 leaks across every case set.
>   - AE2: the lint fails on ANY attribute read of a class-body-labelled name, whatever its base.
>   - A delta round 24 is owed on AE1–AE2 only.
> - **Round 24 (delta, a different reader; 104/104 on 3.10 and 3.12):** 3 SHOULD-FIX (FAIL-OPEN) and 1 NIT.
>   - Regex: a `/` after the whitespace, and an `@` with nothing after it.
>   - Lint: labels reached through `setattr`, `globals()` and other modules, which would need a repo-wide analysis.
> - **Owner, 2026-11-25 (after round 24): SIMPLIFY BOTH MECHANISMS.**
>   - Rounds 15–24 kept finding "one more character a bounded userinfo form excludes", or one more route around a
>     static taint lint, and two regressions came from the author's own folds.
>   - **Regex:** the per-character userinfo forms are REPLACED by one catch-all per branch (§Shape, `redact_urls`).
>     It closes every in-bounds whitespace-userinfo variant at once, at the cost of more fail-closed over-match.
>   - **Lint:** a stated RATCHET over the forms in `src/` today. It is not a boundary: the factory and the choke
>     points are the gate.
>   - The author verified the catch-all on the scratch patch:
>     - 104/104;
>     - 0 leaks across every case set from rounds 15–24 (including round 24's eight);
>     - 3.10 compiles it;
>     - linear in nine adversarial families (at most 222 ms at 256k characters, ratio about 3 for 4× the input).
>   - The superseded per-form text is in git history (3cf7f8db) and in the round entries above.
>   - The next round judges the simplified design whole, not a delta.
> - **Round 25 (whole design, a different reader; 104/104 on 3.10.19 and 3.12; linear across 12 of the reader's
>   own families; no unmapped sink after #1775):** 4 SHOULD-FIX and NITs, tagged AF1–AF8.
>   - Folded here: AF2 (one prefilter), AF4 (the guards made to match the ratchet) and the NITs AF5–AF8.
>   - **Owner decisions, 2026-11-25, asked together:**
>     - **AF1:** the catch-all body is TEMPERED, so it never crosses the start of another URL. The code round must
>       prove it linear with a timing test: the reader's version took 1.08 s at 256k characters on
>       `"/a/https:/b "*n`.
>     - **AF3:** a backslash-holding separator takes ANY scheme of 2+ characters, which spares `C:\`. The lone-`/`
>       `Path` form takes http(s)/ws(s)/ftp(s) plus the stated list. Other schemes in the `Path` form are E1801-3.
>     - **E1801-2** is widened to the real rule, with measured rates.
>   - A round 26 is owed on the folds AF1–AF8 only.
> - **Round 26 (delta, a different reader):** it designed and verified the TEMPERED regex: 104/104 on 3.10.19 and
>   3.12.12, 0 leaks in every case set and in round 25's 12 two-URL cases, and linear (the 1.04 s family now takes
>   about 0.11 s). Two NEW classes came from the owner's choices, plus a stale regex block (AG1–AG4).
>   - **Owner, 2026-11-25:**
>     - AG1: keep tempering and accept its trade-off as E1801-4.
>     - AG2: narrow AF3, so that ANY scheme counts only with a JSON-escaped slash (`\/`) in its separator, and the
>       bare-backslash Windows forms count for the listed schemes only. That stops `nearby:\n\n` and `ab:\d` from
>       being erased (59 hits in 1.24M real sim-report fields). Unlisted Windows-form schemes join E1801-3.
>   - The author applied AG2 and re-verified:
>     - 104/104;
>     - 0 leaks in every case set;
>     - `nearby:\n\n`, `line1:\nline2`, `ab:\d+`, `run:\n`, `C:\Users\me` and `path:/Users/me` left alone;
>     - 3.10 compiles it;
>     - linear: ratio 3.6–5.3 for 4× the input, at most 171 ms at 256k.
>   - A round 27 is owed on AG1–AG4 only.
> - **Round 27 (delta, a different reader): 0 DO-NOT-BUILD, 0 SHOULD-FIX and 3 NITs (fail-closed or docs-only). The
>   STOP RULE HOLDS.**
>   - 104/104 on 3.10.19 and 3.12.12.
>   - 0 leaks outside E1801-1..4 in every case set and in the 12 two-URL combinations.
>   - In a 300k random-string fuzz, the regex never matched where `_URL_START` did not.
>   - Linear: the 5.3 ratio was noise; `a://@\n` takes 34, 132 and 513 ms at 64k, 256k and 1M, about 3.9× per
>     step, and all 11 families fall within 3.1–4.1.
>   - The regex block is character-identical to the patch.
>   - NITs AH1–AH3 are folded.
> - **Owner decisions:**
>   - 2026-11-25: a separate PR that lands right after #1743, not widened into it.
>   - D1–D4 (2026-11-25, asked together; all took the recommended option):
>     - the factory is installed only by `configure_logging`; the second installer is
>       `leader_proxy._ensure_log_buffer`, which runs the same chained install on each `_LogBuffer` install (the R4
>       chain check), so it adds no embedding exposure `configure_logging` does not;
>     - `sim_log` and `log_agentic` are reduced in this PR;
>     - `redact_urls`' whitespace-in-password gap is fixed here;
>     - children are covered by their entry points plus one two-process guard.
>   - D5 (2026-11-25): `ToolOutput.error` is reduced at construction (option a).
> - **Delta code review round 2 (2026-11-25): clean on both lenses** (0 DO-NOT-MERGE, 0 SHOULD-FIX). It also
>   covers two `AgentOutputManager` fixes folded under filing-is-not-fixing (the first-write deadlock, now an
>   RLock; `is None` for the buffer). NITs: the tests were tightened; the private `_numpy_numeric_types` import
>   is kept, as the comment documents (a rename would owe another round).
> - **Code review round 1 (three lenses: Executor, Architecture, Wire integrity), 2026-11-25.** Wire returned a
>   DO-NOT-MERGE on the chat; the rest SHOULD-FIX and NITs, all folded test-first, each guard proven by deletion:
>   - **Owner decision, 2026-11-25 (the Wire DO-NOT-MERGE):** the conversation, `sim_log`'s RESPONSE (the agent's
>     reply) and USER (the person's input) records (`sim_logger.CONVERSATION_SUBSYSTEMS`), reaches the sinks (`/ws`)
>     and the console fan-out (`_EventHub._fanout`) UNREDUCED, so the person sees full links in the chat. Every
>     persisted or logged destination stays reduced: the trail, the sim JSONL, `_bridge_to_log_file`, the terminal
>     and `StructuredFormatter`. Every other subsystem is still reduced at the sinks.
>   - Executor SF1: `ensure_redacting_record_factory`'s check and install run under a module-level `RLock`.
>   - Executor SF2: `_render_subclass_record` catches `Exception` in the Stage-1 form.
>   - Wire SF1: `record_outcome` reduces only the error part of NAc's outcome summary, never a success summary.
>   - Wire SF2: D5's consequences for Hippocampus and NAc keys stated (§Decisions, bio-memory.md §4).
>   - Same pattern, pre-existing: the agentic scrub moved into `AbstractionBuffer.append`, so
>     `AgentOutputManager.write`/`write_to_stream`, which append directly, are reduced too.
>   - Wire NIT: `url_scrubbed` keeps numpy NUMERIC scalars by exact type (never importing numpy).
>   - Architecture NITs: `http` no longer re-exports `_URL_IN_TEXT`/`_URL_PATH_SCHEMES`/`_URL_START`; the unread
>     factory marker is deleted; D1 names `_ensure_log_buffer` as the second installer; `serving_provenance` imports
>     the reducers from `maxim.utils.net`.
>   - Executor NITs: `leader_proxy._reduced_error_body` catches `RecursionError` (a deeply nested JSON body), failing
>     CLOSED to reducing the decoded text; subclass records use a stock `Formatter().formatException` (stated).

## Why

`src/` has 248 `logger.*("... %s", e)` sites that format an exception's text raw. httpx puts the request URL in
its errors, so one transport failure can carry `user:pw@`, a private host path or a `?key=` query. #997 and #1743
reduced URLs by hand at the sites they owned (`utils/http.py::redact_urls`). In #1743's rounds 3 to 6, each round
found one more raw site on the same error path. Per-site wraps do not converge: the boundary has to be where text
leaves the process, not where it is formatted.

**Front-gate question:** can this ride on existing infrastructure? Yes. `redact_urls` is the reducer: it is
idempotent and never raises. `simulation/report.py::_url_scrubbed` is the recursive walker. The URL text helpers
(`url_origin`, the URL regex, `redact_urls`, and `url_scrubbed` moved from report.py) move to the stdlib-only
`utils/net.py`, and `utils/http.py` re-exports them (S5). `http` imports `structured_logging` at module level, so
importing `url_scrubbed` from `http` would be a cycle; it would also load httpx (275 ms) into `utils.logging`,
`sim_logger` and `tools.base`, or, imported lazily inside `getMessage`, while a handler lock is held. The
`("report.py", "_url_scrubbed")` entry in `tests/unit/_orchestrator_source.py::NEW_PRE_SLICE_MODULE_FUNCTIONS` moves
with it. No new module. What the existing per-site wraps cannot
reach is text no call site controls: a third-party library's record, a traceback, a record from a code path
written later.

## Where exception text leaves the process (measured on origin/main 0f50fdc9)

1. **stdlib `logging`**, set up by `utils/logging.py::configure_logging`. `cli.py::main` calls it early for every
   subcommand, the leader and `serve` included, and `api.run` calls it too. Its sinks:
   - the stdout `StreamHandler` and the optional plain `log_file` handler;
   - **`leader_proxy._LogBuffer`**, at INFO+, which is network-served to any cluster-key holder via
     `GET /v1/debug/logs`, tracebacks included;
   - the display handler (`sim_logger._DisplayLoggingHandler`) and `IKWarningHandler`;
   - third-party records (httpx/httpcore at WARNING+);
   - the `MAXIM_LOG_FILE` JSONL handler. Its `StructuredFormatter` writes only `event`/`data`, never `getMessage()`.
     So plain-message text never lands there, but the `extra={"data": ...}` payloads of `log_structured` and the 11+
     `extra=` sites do, raw (R1).
2. **`sim_logger.sim_log`**, which bypasses `logging`. It writes to its sinks (the console `/ws` stream),
   `_log_records` (the report and telemetry), its own JSONL file, `_bridge_to_log_file` (into `MAXIM_LOG_FILE`,
   since #1775) and the terminal (AF6).
3. **`structured_logging.log_agentic`**, which bypasses `logging`. It writes to the `AbstractionBuffer`, whose
   `get_recent` feeds the memory agent's LLM context, to its callbacks, and to stdout under `MAXIM_AGENTIC_CONSOLE`.
4. **Wire and record payloads, which are not logs (R2):**
   - `console/server.py`: the run-failure `_event_hub.publish(..., {"error": f"{type(e).__name__}: {e}"})` goes to
     `/ws`, and two `HTTPException(detail=f"{type(e).__name__}: {e}")` replies go out over HTTP;
   - `runtime/leader_proxy.py`: the `{"error": f"pip upgrade failed: {e}"}` reply;
   - `simulation/report.py`: `llm_summary = f"Roundup failed: {e}"`, written raw into a report.json whose other
     fields are scrubbed;
   - `runtime/executor.py` and 4 other sites build `ToolOutput(error=f"... {e}")`, which reaches the planning LLM's
     prompt and so a cloud provider (D5).

## Shape

- **Family 1, part a: a chained `logging` record factory**, installed by `configure_logging` (D1) and re-checked
  by `leader_proxy._ensure_log_buffer`, the second installer, on each `_LogBuffer` install. The check and the install
  run under one module-level `threading.RLock` (`_factory_install_lock`), so two installs racing on different
  threads cannot drop a chained host factory or log a false WARNING (code review SF1). It leaves `msg`
  and `args` untouched (R3: rendering at creation costs up to 598 µs for one large dict argument, it would render
  all 423 `logger.debug` sites under `MAXIM_LOG_FILE`'s root=DEBUG, and it hollows out tests that read
  `record.args`). Instead it:
  - reduces lazily, when a handler calls `getMessage`, behind the prefilter, so the cost is paid only when a handler
    formats the record. It uses no closure and no bound method: a closure cannot be pickled (S3), and a bound method
    stored in `__dict__` drags the raw record into `SocketHandler.makePickle`'s blob, or fails on `exc_info` (F1).
    The choice is made by the record's own type, not by the chained factory's identity (F1):
    - a record whose `type(record) is logging.LogRecord` (the stock class, whoever built it) has its
      `record.__class__` set to the module-level `_RedactingLogRecord(logging.LogRecord)`. Its `getMessage` reduces
      BEFORE and AFTER formatting (P1):
      - **every argument except an EXACT pass-through type** is wrapped in a module-level proxy whose
        `__str__`, `__repr__` and `__format__` return the reduced text, and so are the VALUES of a single Mapping
        argument (`%(k).Ns`: `LogRecord.__init__` unwraps it to a dict). Then `msg % args` runs. A token-as-username
        URL leaked through Mapping values, `list`/`tuple` (including `e.args`), `bytes`, `httpx.URL` and custom
        `__str__` objects when only `str` and exceptions were proxied (Q1; probed, 0 leaks with this rule). A
        1-character `str` passes through unproxied, so `%c` renders as today (Q4). The pass-through test is by EXACT type,
        never `isinstance` (T1): `type(a) in {NoneType, bool, int, float, complex, Decimal, Fraction}`, or
        a numpy NUMERIC scalar by EXACT type: `type(a) in _NUMPY_NUMERIC`, a frozenset built lazily on first proxy use, WITHOUT importing:
        `np = sys.modules.get("numpy")`, and while it is absent the set is empty and NOT cached, since no numpy scalar
        can exist before numpy is imported. An import inside `getMessage` runs with the handler lock held, and could
        deadlock against a thread part-way through importing numpy that logs (X5; W4). While numpy is part-way
        imported it is in `sys.modules` with `__spec__._initializing` True and no `sctypeDict`. In that case, or if
        the build raises ANY exception (`AttributeError`, a `TypeError` from `np.dtype`, …; Z5), the set is empty and NOT cached, logged once in the Stage-1 form, so the
        record is proxied (safe) rather than lost to `handleError` (Y5). The log is ONE-SHOT and armed only by a
        NON-transient failure, classified with `getattr(getattr(np, "__spec__", None), "_initializing", False)`, so a
        `__spec__` of None cannot raise during the classification (AC3; AB4: spending it on the transient `_initializing` state would hide a later real
        failure). The flag is set BEFORE logging, because `log_swallowed_exception`'s own record is proxied too, and an uncached failed build
        would otherwise rebuild and re-log on every record until a swallowed `RecursionError`. A non-transient
        failure (anything but `_initializing`) is cached as empty (AA5). Built from numpy's own scalar
        types (`np.sctypeDict` values whose `np.dtype(t).kind` is one of `b i u f c`, a non-empty kind), so a subclass
        or a fake class setting `__module__ = "numpy"` is proxied (V4); or
        `type(a) is str and len(a) == 1`. `np.str_`, `np.bytes_` and every ndarray are proxied: `"%.15s" %
        np.str_("https://ghp_TOKEN@github.com")` kept `https://ghp_TOK` under a module-only test (U1). A user class
        that sets `__module__ = "numpy"` without a numeric `dtype` is proxied too. A subclass with its own `__str__`
        (`NInt(int)`, a 1-char `S1(str)`) returning a URL is proxied (probed: it leaked under `isinstance`);
      - the proxy forwards `__float__`, `__int__` and `__index__` through `type(o).__float__(o)` and
        `operator.index(o)`, never `float(o)`, which would make `%f` accept a str. `__int__` falls back to `operator.index(o)` and
        `__float__` to `float(operator.index(o))` (an int from `__float__` raises, W3), as CPython does, so an object with only `__index__` still renders `%d`/`%f` (V5). When
        neither exists, the proxy raises `TypeError` (not `AttributeError`), as the unproxied value would (U4). So `%f`/`%d`/`%x` of a 0-d
        ndarray or a torch tensor render as today instead of losing the line to `handleError` (T3); So a precision cut in the message
        (`%.50s`, `%.50r`, `%.50a`, `{:.50}`), whatever the logger's name, a message held in a variable, or
        `logger.log(level, …)`, truncates text that is already reduced;
      - the result is reduced again (a URL built across arguments, or one in `msg` itself);
      - numbers and None pass through untouched, so `%d`/`%f`/`%x` behave as today, numpy scalars included
        (probed). That keeps any foreign attributes and pickles, and `makePickle` is clean (probed);
    - a record of a real subclass is rendered eagerly, `msg = redact_urls(getMessage())` and `args = ()`. That pays
      R3's cost only on this rare path. The render is wrapped in `try`: a bad format (`warning("x %d", "notint")`)
      must not raise out of the `logger` call, where inside an `except` block it would mask the original exception.
      On ANY failure (`Exception`, code review SF2: an argument whose `__str__` raises `RuntimeError` must not
      escape either) the record is left untouched: no class swap, which would
      drop a subclass's own `getMessage`, or raise `TypeError` on a subclass with `__slots__` (H3). `handleError`
      then reports the bad format at emit, as it does today. The `except` is recorded with the swallow lint, in the
      Stage-1 form (H3). The subclass path also reduces the traceback fields in place, OUTSIDE the message-render
      `try`, so a bad format string cannot leave them raw (K5). Guarded as the stdlib guards it: `if record.exc_info:
      exc_text = redact_urls(logging.Formatter().formatException(exc_info))` (it raises `TypeError` on `None`, L3);
      a STOCK `Formatter().formatException`, so a handler's custom `formatException` is not used for that field,
      and `stack_info` is reduced only when it is set (J5; latent, since there is no `LogRecord`
      subclass in `src/` or the venv). A failed render's `getMessage` raises again at emit and goes to
      `handleError`'s stderr dump: the stated terminal-only gap;
  - makes `exc_text` and `stack_info` data descriptors on `_RedactingLogRecord`. Their `__set__` stores the REDUCED
    value (H2): a descriptor that reduces only on read leaves raw text in `__dict__`, which `SocketHandler.makePickle`
    copies (probed on 3.12–3.14). `LogRecord.__init__` writes `stack_info` (and a caller can pass `exc_text`) into
    `__dict__` BEFORE the factory swaps the class, so right after the swap both are written back through their
    descriptors (`r.stack_info = r.__dict__.get("stack_info")`) (J1). A precompute at creation would miss `logging.makeLogRecord(dict)`: the factory runs with
    `exc_info=None`, and the dict update then installs a raw `exc_text` (G3; latent, since there is no
    `makeLogRecord`/`QueueListener`/`SocketHandler` receiver in `src/`). Every stdlib `Formatter` reads `exc_text`
    after filling it, so the read is reduced. `pickle`, `copy` and `QueueHandler.prepare` come out clean. A gap,
    stated: `makeLogRecord`'s dict update bypasses `__set__`, so that record's `__dict__` stays raw (latent; no
    receiver in `src/`) (H2). The clean-through-`pickle`/`copy`/`QueueHandler.prepare` claim covers `exc_text` and
    `stack_info` ONLY (K1);
  - carries no marker attribute (code review NIT2 deleted the unread one): whether to install is decided by the
    behavioural check alone. A chaining wrapper placed over ours would hide any marker, and the outcome is a
    harmless double wrap (N2).

  It is a factory rather than a handler filter because `basicConfig(force=True)` and `sim_logger`'s reconciliation
  rebuild handlers, while the factory survives both (probed, as did `dictConfig`). The bridge logger with
  `propagate=False` goes through it too.
- **Family 1, part b: the formatter and the network buffer.**
  - `StructuredFormatter.format` passes `data` through `url_scrubbed`, because the factory runs before `extra=` is
    attached and cannot see it (R1).
  - `_LogBuffer.emit` also runs `redact_urls` on its formatted string. This is belt-and-braces on the highest-reach
    sink, and it covers a handler that re-renders `exc_info` itself (R4).
- **The chain check is behavioural (R4).** At the end of `configure_logging` and on each `_LogBuffer` install:
  build a record with a credential URL through the current factory and check that it comes out reduced. On a
  failure, re-install on top and log one WARNING; do not raise.
- **Family 2:** `sim_log` REBINDS its `message` and `data` locals to their `url_scrubbed` values first, before
  `record` is built, so that every reader of the locals gets the reduced values (AB3): `_dispatch_to_sinks`, the
  in-memory trail, the JSONL file, `_bridge_to_log_file` and the terminal render. Reducing before `text_shape` is
  also required, because a shape computed over raw text would be an oracle for the raw text (AB3). Since #1775 (on main), `sim_log` also mirrors every record into `MAXIM_LOG_FILE`
  through `_sim_bridge_logger` with `extra={"data": …}` (reply subsystems by shape only). That fifth destination is
  reduced twice: by `sim_log`'s own pass, and by `StructuredFormatter`'s `data` pass (R1). Deletion probes cover all six
  destinations: sinks, trail, sim JSONL, bridge, terminal and `StructuredFormatter` (AB3). It
  recurses and stringifies leaves such as an exception object in `data` (R8).
  - The prefilter is ONE test, stated identically everywhere: `":/" in s or ":\\" in s` (AF2: the earlier
    `"://"`/`r":\/\/"` forms let the `Path` lone-`/` form and `https:\u…` through). Every case set runs through
    the prefiltered `redact_urls`, not the bare regex. It lives
    inside `redact_urls` itself, so `url_scrubbed` gets it on every string:
    5.1 ms drops to 0.43 ms on large data with no URLs (S6).
  - Shape changes that now reach live `sim_log` sinks: a tuple becomes a list, a set or bytes become a str, and a
    URL-keyed dict collapses its keys (S6).
- **Family 3:** `log_agentic` and `log_structured` pass `data` through `url_scrubbed` before the `LogRecord` is
  built.
- **Family 4 (R2), one choke point per surface:**
  - `console/server.py`: `_EventHub._fanout` scrubs, because it sees both `sink` and `publish` (S2). Every
    `HTTPException` detail goes through one `_error_detail(e)` helper: the two run-failure details, the two
    `ConfigurationError` `detail=str(e)` (from `apply_mesh_setup(leader_url, …)`) and `Bad cursor: {e}`;
  - the leader proxy: when `code >= 400`, `_send_json` runs `url_scrubbed` over every string leaf of the body. That
    covers the 500 reply's `"stderr"` key, not only `"error"`, and it keeps a dict-shaped OpenAI error's structure and
    its int `code`, which `maxim_peer_backend._context_overflow_counts` parses (Nb, G2). The scrub runs before
    `json.dumps`, because `Content-Length` is computed from that string (H5). `_send_bytes` parses an error JSON body,
    scrubs its leaves and re-encodes it, never running `redact_urls` over the raw bytes, where the userinfo branch can
    span compact-JSON field boundaries (H5). An error body that does not parse as JSON, or is not JSON content, has
    `redact_urls` run over its decoded text, decoded and re-encoded as `utf-8` with `surrogateescape` (a
    strict re-encode raises `UnicodeEncodeError` on a `\x80` byte, L4), and reduced before the `Content-Length` header
    is computed (J6, K6; latent, since every oasis error goes through `json_response`). That which covers `hivemind/oasis_endpoints.py`'s `{"error": str(exc)}` 400 path (G2); That covers its 25 4xx/5xx sites,
    including `Invalid JSON body: {e}` and the bare `str(e)` (S2);
  - **errors that bypass `ToolOutput` (S1, F3):**
    - `tool_dispatch.py::record_outcome` reduces `error`. It is the one choke point for `recent_outcomes` and
      `llm_worker`, which both feed the planning prompt. Its callers include the `executor.execute`-raised path
      (`rec_outcome(error=str(e))`) and the agent-fallback path;
    - the parallel path: reduce at the `"error": str(e)` construction in `tool_dispatch.py::_dispatch_parallel`.
      `combined_results`, the LLM context, is built from those dicts, which `record_outcome` never touches (F3);
    - the sub-simulation tool: `simulation/tools.py`'s `sub_report["error"] = str(e)`, returned inside a
      *successful* `ToolOutput`, so outside D5 (F3);
    - success-shaped tool payloads that carry an exception's text (G6): `tools/introspection.py`'s two
      `{"error": str(e)}` inside a successful `ToolResult`, `dm_runtime._deliver_and_wait`, and `simulation/minecraft.py::call_action`'s
      `f"bridge send failed: {exc}"` `detail` (M5). Each is reduced where it is built;
    - `roy_runner.py::_persist_result`: result.json's `"detail": f"{type(e).__name__}: {e}"` (the `health_check`
      probe and preflight), `priming["error"]` and a raw `"url"`. Roy artifacts are committed, so `url_scrubbed` runs
      over the whole payload there (H4). The committed `summary.md`, rendered by `_format_summary(result)` from the
      same raw fields, is written as `redact_urls(_format_summary(result))` (J2). `ToolOutput.output` as a whole is deliberately NOT reduced, because
      fetched content keeps its URLs;
    - the leader proxy's `_install_status` writes (`str(exc)` and pip stderr). They are served in a **2xx** body at
      `/v1/debug/install-status`, so `_send_json`'s 4xx/5xx reduction misses them (F3);
    - `loop_planning.py::_abort_drain`'s `detail` (the refusal message);
    - `state.py::RuntimeState.mark_failure` (`last_error`). It reaches only the already-scrubbed report.json, not a
      prompt (F3 checked); it is reduced anyway, as belt-and-braces;
  - `report.py` scrubs `llm_summary` with the other fields;
  - `ToolOutput.error` is reduced at construction (D5a), in `tools/base.py::ToolOutput`, so the prompt, the
    narrator, the console and memory all receive the reduced text. `ToolOutput` is `@dataclass(slots=True, frozen=True)`, so a
    `__post_init__` sets it with `object.__setattr__` (never zero-arg `super()`, which raises `TypeError` in a
    slotted dataclass) and turns a non-str error into a str. `dataclasses.replace`, pickle and copy all keep the
    reduced value (probed). A third-party tool's error is covered too. `http_fetch` failures also carry the raw URL in
    `metadata={"url": url}`; the code round checks whether that reaches a prompt (N1).
- **`redact_urls` (D3, R7; simplified by the owner, 2026-11-25).** The exact regex, verified on the scratch patch
  (`scratchpad/regex1801/regex_candidate.patch`):

  ```python
  _URL_PATH_SCHEMES = r"(?i:[A-Za-z0-9+.-]*(?:https?|wss?|ftps?|sftp|postgres(?:ql)?|mysql|rediss?|mongodb(?:\+srv)?|amqps?|ssh|s3))"
  # The start of a URL, exactly as the branches below recognise one; every unbounded or bounded body is TEMPERED by
  # it, so one URL's match never crosses the start of another (AF1).
  _URL_START = (
      r"(?<![A-Za-z0-9+.-])(?:[A-Za-z0-9+.-]+://|[A-Za-z0-9+.-]{2,}:[\\/]*\\/|" + _URL_PATH_SCHEMES + r":[\\/]*\\|" + _URL_PATH_SCHEMES + r":/(?![/\\]))"
  )
  _T = r"(?!" + _URL_START + r")"
  _URL_IN_TEXT = re.compile(
      r"(?<![A-Za-z0-9+.-])"
      r"(?:(?:[A-Za-z0-9+.-]{2,}:(?=(?P<sep>(?=[\\/]*\\/)[\\/]+))(?P=sep)"  # any scheme: a JSON-escaped slash in the separator
        r"|" + _URL_PATH_SCHEMES + r":(?=(?P<wsep>(?=[\\/]*\\)[\\/]+))(?P=wsep)"  # listed schemes: any backslash separator (Windows)
      r"|" + _URL_PATH_SCHEMES + r":(?P<psep>/)(?![/\\]))"  # the lone-``/`` Path form, listed schemes only
      r"(?:(?P<swsui>(?:" + _T + r"\S){0,640}\s(?:" + _T + r"[\s\S]){0,640}@)(?:" + _T + r"\S)*|(?:" + _T + r"\S)+)"
      r"|[A-Za-z0-9+.-]+://"  # any scheme, the double-slash form (today's)
      r"(?:(?P<wsui>(?:" + _T + r"\S){0,640}\s(?:" + _T + r"[\s\S]){0,640}@)(?:" + _T + r"\S)*"
      r"|(?:[^\s/@]*@)?(?:" + _T + r"[^\s'\"<>])+))"
  )
  ```

  - **Tempering (AF1):** every body, bounded or not, is tempered by `_URL_START`, the regex's own start, so one
    URL's match never crosses the start of another. The residual is E1801-4.
  - **The catch-all (both branches):** if a whitespace lies within 640 non-space characters after the separator, and
    an `@` within 640 characters after that whitespace, the span through the LAST such `@` and the rest of that
    token become `<url>`. It closes every whitespace-userinfo variant inside the bounds at once, whatever the
    characters around it: `/`, `\`, `@`, quotes, `\r\n` or tabs, an email user, a trailing `@`. That covers every
    in-bounds leak of rounds 15–24. `\S` and `\s` are disjoint, so the first part cannot backtrack ambiguously,
    and the second is bounded, so the scan is linear.
  - **The separator branch** (AF3, AG2), case-insensitive:
    - ANY scheme of 2+ characters when its separator holds a JSON-escaped slash (`\/\/`, `\/`, JSON-in-JSON);
    - the LISTED schemes (`_URL_PATH_SCHEMES`: http(s), ws(s), ftp(s), sftp, postgres(ql), mysql, redis(s),
      mongodb(+srv), amqp(s), ssh, s3, and any scheme ending in one of them, such as `git+ssh`) with any
      backslash-holding separator (Windows `\`/`\\`, mixes), or a lone `/` (the `Path` form).

    Everything else is E1801-3. A fail-closed over-match follows from the `[A-Za-z0-9+.-]*` prefix: any word ENDING
    in a listed scheme counts as one, so `follows:\n1.`, `Windows:\n`, `views:/x` and `mysql:\n` are erased and
    also act as tempering starts (E1801-4). It loses text and leaks nothing: 0 hits in 6.19M decoded real
    sim-report string fields. A test pins it (AH1). If
    the catch-all does not apply, the whole non-space token is taken. It always reduces to `<url>`, because
    `url_origin` cannot parse these forms. The separator is atomic by a lookahead plus a backreference, which keeps
    Python 3.10 (the floor) able to compile it.
  - **The double-slash branch:** every scheme (today's), and if the catch-all does not apply, today's plain userinfo
    `[^\s/@]*@` and host, reduced by `url_origin` (`https://u:p@h/x` gives `https://h`, `https://to"k@h/p` gives
    `https://h`).
  - **Fail-closed over-match, accepted under E1801-2:** a URL followed within the bounds by whitespace and any `@` is
    erased: `https://u:p@h/x, contact a@b`, `see https://a.com and user:x@y`, `https://h:8080 failed, mail
    bob@ex.com`, and across line breaks. At most 640 + 1 + 640 characters plus the following token are lost.
    Maxim's own log formats never put a URL next to an `@` (0 grep hits), so only exception or LLM text triggers
    it. Tests pin each.
  - **The residual past the bounds:** E1801-1, either branch, with the 640/641 pins.
  - **The prefilter:** `":/" in s or ":\\" in s`, inside `redact_urls` (S6, Nc, AF2).
  - `redact_urls` stays idempotent and never raises.
- **The per-site wraps stay** (#997, #1743). They are belt-and-braces, and they cut AFTER reducing, which the
  boundary cannot do for them.

## Sites that cut before reducing (F2)

A boundary reducing text that a site has already cut reduces a cut URL. `https://ghp_TOKEN@github.com` cut to
`https://ghp_TOK` survives as a "host", and a tail cut (`stderr[-500:]`) can drop the scheme, so no URL is seen at
all. The boundary cannot repair this, so each such site reduces before it slices:
- `loop_planning.py::_abort_drain` (`f"…{exc}"[:300]`);
- the two `router.py` warmup `str(e)[:50]` sites;
- `leader_proxy`'s `_install_status["error"] = r.stderr[-500:]`;
- `tools/code_tools.py`'s `result.stderr[-1000:]`.

#1743 already fixes `loop_intake.py`'s `[:100]` and `[:60]`.

Round 4 (G1) found more sites than the grep had:
- `openai_backend.py::_sanitize_error_message` (`msg[:max_len]`) and `sanitized_error[:200]`, which reaches
  `TraceRecord.error` and then the JSONL `data`;
- `leader_proxy.py::_run_pip_upgrade` (`stdout[-500:]` and `"stderr": (result.stderr or "")[-500:]`);
- `_install_status["pip_output"]`;
- `tools/code_tools.py`'s `result.stdout[-2000:]` and `leader_proxy._run_pip_install`'s `stdout[-500:]` (H1);
- `bench/recovery_time.py::run_recovery_benchmark`'s `fix_hint[:500]`. Its value is `_classify_error`'s
  `str(exc)`, and it reaches `bench/cli.py`'s `--output` JSONL. It is reduced at `_classify_error`'s exit, so every
  branch is covered: `_hint` covers only some, and the `return "unhandled_" + type(exc).__name__, str(exc)` branch,
  the likeliest to carry a raw httpx URL, skips it (K2, L1). A test pins the `unhandled_*` branch.

**Guard (simplified by the owner, 2026-11-25): a RATCHET lint, not a boundary.**
`scripts/lint_cut_before_redact.py` is an AST lint over `src/maxim`. It flags a cut of exception text that is not
reduced first. Its taint sources are:
- an `except … as` name;
- a parameter named `e`/`exc`/`err`/`error` or annotated as an exception;
- `.stderr`/`.stdout`;
- `traceback.format_exc()`.

Taint propagates through `str()`, `repr()`, f-strings, `%`/`+`, `.attr`, method calls, subscripts and plain
assignments in the function. A cut is a slice, a format precision, `textwrap.shorten`, or a `split`/`partition`
result. A cut passes only when the sliced value is itself a `redact_urls(...)`/`url_scrubbed(...)` call, or a name
whose every assignment is one.

- **Its scope is the forms in `src/` today, which it catches** (the listed sites in §Sites, all fixed in this PR,
  plus the allowlist with reasons, keyed `file::innermost_function` plus the `ast.unparse` text, with an exact
  `count`; it includes `peer/install_core.py`'s `e.response.text[:500]`, which only reaches `print`, AF8). A new cut in
  those forms fails CI.
- **Its blind spots are stated, not chased.** These include:
  - indirect callees (`getattr`, `methodcaller`, `partial`, aliases, class-body or `setattr` or `globals()` labels,
    other modules);
  - cuts inside helpers;
  - exotic cutting callables.

  Rounds 7–24 showed that a static taint lint can always be evaded once more.
- **The boundary is the gate, not this lint:** the record factory (which reduces BEFORE `msg % args` cuts, P1),
  `StructuredFormatter`, `sim_log`, `log_agentic`, the wire choke points and `ToolOutput.error`. A cut made before any
  of those can only leak through text that never reaches a sink this PR guards. The deletion probes prove each
  sink, not the lint.

## Inputs, and how each could make the boundary lie

| Input | How it could lie | Answer |
|---|---|---|
| message text | a URL without a scheme, `Bearer sk-…`, a percent-encoded whole URL | out of scope (URL userinfo, path and query only), stated in the docstrings |
| whitespace or JSON escaping | `u:p w@`, `\n`, `\t`; `https:\/\/` | fixed in `redact_urls` (R7), with a test per form |
| `extra=` / `data` | attached after the factory runs (R1) | reduced in `StructuredFormatter`, `log_structured`, `log_agentic` and `sim_log` |
| `args` | a non-str value, or a `__str__` that raises | rendered lazily by the stdlib `getMessage`; if it raises, logging's `handleError` prints the raw `Arguments:` to stderr: a terminal-only gap, stated |
| `exc_info` / `stack_info` | the message is the traceback's last line | `exc_text`/`stack_info` data descriptors on `_RedactingLogRecord` that reduce on set, with a write-back after the class swap (H2, J1); the proxy buffer is also reduced after formatting |
| install order | records before `configure_logging`; Python's last-resort stderr handler; a later factory that does not chain | before setup there are no Maxim sinks except stderr (a gap, stated). The behavioural chain check catches a later factory at each install point; one set after the last check point is caught only at the next `configure_logging`, and nothing in the venv calls `setLogRecordFactory` (a gap, stated; N2). `logging.makeLogRecord` and `deepcopy` keep the reduced text (probed) |
| raw `__dict__` (K1) | `msg` as an exception object, `args`, `exc_info` and `extra=` keys stay raw in `__dict__`, so `vars()`, `pickle` and `copy` carry them; `extra=` keys survive even `makePickle` and `QueueHandler.prepare` | a gap, stated: the boundary reduces what a handler RENDERS, not a record's raw attributes. Latent, since there is no pickling or `__dict__` receiver in `src/` (no `SocketHandler`/`QueueListener`/`makeLogRecord`). The `StructuredFormatter` path reduces `data` on render (R1) |
| serialization (S3) | a closure `getMessage` cannot be pickled, so `SocketHandler` and a multiprocessing queue drop the record | a module-level `LogRecord` subclass set by `__class__` swap on a stock record (F1); a real subclass is rendered eagerly. Bound methods are NOT used (F1) |
| embedding | a global factory rewrites the host app's records | installed only by `configure_logging` (D1) and re-checked by `leader_proxy._ensure_log_buffer` |
| children (R5) | the LLM server runs `python -m llama_cpp.server`, and `maxim-diagnostics` never calls `configure_logging` | both write only to the terminal or to DEVNULL (stated). Every `cli.py::main` child is covered |
| tests (R6) | 14 test files call `configure_logging` or `cli.main` and never restore the factory | an autouse `tests/conftest.py` fixture saves and restores `logging.getLogRecordFactory()` |
| double application | a per-site wrap followed by the boundary | `redact_urls` is idempotent |

## Guards (each proven by deleting the mechanism)

> **AF4 (owner's simplify decision):** every lint bullet below that plants an INDIRECT form or a callable outside the
> ratchet's stated cut list is SUPERSEDED: `getattr`/`attrgetter`/`methodcaller`/`partial`/aliases (X3, Y4, Z3, AA3,
> AB2, AC2, AD3, AE2), `textwrap.wrap`/`fill`, `itertools.*`, `deque(maxlen=)`, `reprlib.*`, `__mod__`/`operator.mod`,
> `format_map`/`vformat`, dunder calls (Q3, T2, U3). The lint's planted-form tests are EXACTLY its stated sources and
> cuts (§Guard, the ratchet). Its blind spots are tested only as documented non-detections. Every per-form regex pin is
> superseded by the catch-all's 640/641 pins (AF5).

- A credential URL in `msg`, `args`, `exc_info` and `extra=data`, through every handler `configure_logging`
  installs, plus `_LogBuffer` and the display handler, after a `basicConfig(force=True)` rebuild: only the origin
  survives. `test_followup_parse_log_1751`'s `record.args` assertion still means what it says.
- `sim_log` with a URL in `message` and in `data`, including an exception leaf: all SIX destinations are reduced: sinks, trail, sim JSONL, `_bridge_to_log_file`, terminal and `StructuredFormatter` (AF6).
- `log_agentic` and `log_structured`: `get_recent` and the JSONL record are reduced.
- Family 4: a `ToolOutput(error=<credential URL>)` is reduced; the console run-failure event, each `HTTPException` detail, the proxy's pip reply and report.json's
  `llm_summary`.
- The chain: a non-chaining factory installed after setup is caught and re-wrapped, with one WARNING.
- Two processes (the red-team's split): a child `python -c "from maxim.cli import main; main([...])"` logs a
  credential URL, and the test asserts on the child's output. `/v1/debug/logs` is tested in-process through
  `_ensure_log_buffer`.
- `redact_urls`: one test per whitespace form and for the JSON-escaped form, plus the two over-match prose strings
  (S4).
- S1: an `executor.execute` that raises with a credential URL leaves `recent_outcomes`, the parallel results,
  `_abort_drain`'s refusal and `last_error` reduced.
- S2: a `_send_json(500, {"error": <credential URL>})` and each console `HTTPException` detail come out reduced.
- S3/F1: a record made by Maxim's factory pickles and unpickles, its `SocketHandler.makePickle` blob holds no
  credential, and it renders reduced after a `QueueHandler.prepare`, with and without `exc_info`. Run both with the
  stock chain and behind a chaining third-party factory that returns a plain `LogRecord`.
- F2/G1: `lint_cut_before_redact.py` reports 0 on the branch, and fails on each planted form: `str(e)[:50]`,
  `redact_urls(str(e)[:50])`, `f'{e!r}'[:50]`, `msg = str(e)` then `msg[:50]`, and `proc.stdout[-500:]`. Each fixed site has
  a test with a token placed so that cutting before reducing would leak it.
- H2: a `SocketHandler.makePickle` blob of an exception record holds no credential. H3: a subclass with
  `__slots__` and a bad format string neither raises out of the `logger` call nor loses its own `getMessage`.
- H4/J2: a `roy_runner` result.json and summary.md written after a failed health check hold no credential.
- J1: a `stack_info=True` record carrying a credential comes out clean through `makePickle`, `vars()`, `pickle` and
  `QueueHandler.prepare`.
- AE1: the six colon-less `@`-user cases reduce to `<url>`; `https://bob@ex.com@h/p` and `https://u:p@h/x` still
  give `https://h`. AE2: superseded by the ratchet (AF4).
- AD1: `https:\/u:p SECRET@h`, `https:\\/u:p SECRET@h`, `https:\\\/\\\/u:p SECRET@h` and a double
  `json.dumps` reduce to `<url>`; `http:/ prose` is untouched. AD2: `https://bob@ex.com:p SECRET@h/p` and its JSON form
  reduce to `<url>`; `https://bob@ex.com@h/p` still gives `https://h`. AD3: superseded by the ratchet (AF4).
- AC1: `https:\/\/u:p SECRET@h\/p` (bare and inside a JSON string), `Path("/a/https://u:p SECRET@h/p")`,
  `https:\/\/us SECRET:p@h`, `https:\/\/tok SECRET@h`, `https:\u:p SECRET@h`, `ws:\\u:p SECRET@h` and a tab form
  reduce to `<url>`. AC2: superseded by the ratchet (AF4).
- AB1: `https:\/\/u:p\"SECRET@h\/p`, `https:\u:p\"SECRET@h` and `https:\/\/u:p\<SECRET@h` reduce to `<url>`;
  `http:/ prose` is untouched; `'https:\\"'*n` and `('https:\/\/u:p\"')*n` stay linear. AB2: superseded by the ratchet (AF4). AB4: a transient
  `_initializing` build does not spend the one-shot log.
- AA1: the four quote/backslash cases reduce to `https://h`, and `'https:\\"'*n` stays linear. AA2: the three
  backslash-in-user cases reduce. AA3: superseded by the ratchet (AF4). AA5: a failing
  numpy build logs once, and a stream of records makes no nested build (a `RecursionError` sentinel).
- Z1: `https://u:p\r\nw@h/p`, `https://u\ns:p w@h/p`, `https://u:a b\nc@h/p`, `https://us er:p@ss"x@h/p`,
  `https://tok en@h/p` and `https://D\u:p w@h/x` reduce to `<url>`. Z2: `'https:\\"'*n` stays linear (a timing bound
  in the test). Z3: superseded by the ratchet (AF4). Z5: a `TypeError` in the numpy
  build neither loses the record nor caches.
- Y1–Y3: `https://u:a b@c d@h/p`, a 70-character password tail on either side of the space, and `https://us
  er:pw@h/p` reduce to `<url>`. E1801-1: superseded; the catch-all's 640/641 pins replace every per-form pin (AF5). Y5: a part-way-imported numpy (fake package, `_initializing`)
  neither loses the record nor caches the empty set. Y6: the four over-match pins (E1801-2's `pinned_by`). Y4: superseded by the ratchet (AF4).
- X1/X2: `redis://:PW D@cache:6379/0`, `https://:p w@h/x` and `https://u:p@ss w@h/p` reduce to `<url>`. X3: superseded by the ratchet (AF4). X4: the host:port over-match is pinned. X5: the factory never imports
  numpy (a test asserts `getMessage` leaves `sys.modules` unchanged).
- W1: `postgres://u:PW@db`, `redis://:PW@h`, `git+ssh://u:PW@h`, `unix:///s.sock` and `123://u:p@h` reduce;
  `https://u:p w@h/p` gives `<url>`; the case table matches identically on 3.10 and 3.12. W2: the lint fails on
  `re.split("@", str(e))[0]` and `str.partition(e, "@")[0]`. W3: `%f` of an `__index__`-only object renders.
- V1: `https://to"k@h/p` reduces to the origin, and the existing 104 redaction tests pass with the pinned regex. V6:
  `Path("/a/ftp://u:PW@h")` and `Path("/a/wss://u:PW@h")` render reduced. V2: the lint fails on `head, _, _ =
  str(e).partition("@")`, `.split("@").pop(0)` and a bound `parts[0]`. V4: an `np.float64` subclass with a URL
  `__str__` and a fake `__module__ = "numpy"` class render reduced. V5: an `__index__`-only object renders `%d`.
- U1: `np.str_`, `np.bytes_` and `np.array([url])` carrying a credential URL render reduced at every cut length, and
  `%d`/`%f` of numpy numeric scalars are unchanged. U2/U5: `Path("/a/HTTPS://u:PW@h/p")`, `/a/1https:/u:PW@h/p` and
  `PureWindowsPath`'s `https:\u:PW@h\p` render reduced. U3: the lint fails on `textwrap.wrap(s,20)[1]`,
  `textwrap.fill`, `itertools.islice`, `deque(maxlen=)`, `.partition("@")[0]` and `.split(sep)[0]` over tainted
  text. U4: `%d` of a proxied str raises `TypeError`.
- T1: an `int` subclass and a 1-char `str` subclass whose `__str__` returns a credential URL render reduced at every
  cut length. T2: the lint fails on `str(e)[sl]`, `str(e)[slice(0, 50)]`, `operator.getitem`, `.__getitem__`,
  `.__format__(".50")` and `reprlib.repr` over tainted text. T3: `%f`/`%d`/`%x` of a 0-d ndarray render, and
  `%f` of a str still raises. T4: `%.40s` of `Path("/a/https://u:PW@h/p")` renders reduced.
- Q1: the same credential URL as a Mapping value (`%(k).30s`), in a list, in `e.args`, as `bytes`, as an `httpx.URL`
  and in a custom `__str__` object renders reduced at every cut length; `%d`/`%5.2f` with numpy scalars, `%x`, `%c`
  and `%s None` render unchanged. Q2: an entry with fewer hits than its `count` fails as stale. Q3: the lint fails on
  `"%.50s".__mod__(e)`, `operator.mod`, `format_map` and `vformat` over tainted text.
- P1: `log.warning("failed: %.50s", e)`, `self.log.warning(...)`, `logger.log(30, "%.50r", e)` and a variable message,
  each with a credential URL in `e`, all render reduced; `%d`/`%f` arguments render unchanged. P2: a second identical
  hit beyond an entry's `count` fails. P3: a module-level `except … as e: str(e)[:50]`, a class-body one, a lambda
  `lambda e: str(e)[:50]` and a closure over an outer `e` each fail.
- M1: the lint fails on each planted form: `"{:.50}".format(e)`, `format(e, ".50")`, `"%.*s" % (50, e)`,
  `textwrap.shorten(str(e), 50)`, `(r.stderr or "")[-500:]` and
  `json.dumps({"k": str(e)})[:50]`. It passes `f"{duration:.1f}"`. M2: a new `str(e)[:200]` placed inside an
  allowlisted function fails.
- L1: `_classify_error`'s `unhandled_*` branch with a credential URL leaves `fix_hint` clean. L2: the lint fails on
  `dict(m=str(e))["m"][:50]`, `f"{e!s:.50}"` and `"%.50s" % e`. L3: a subclass record logged with no exception does
  not raise. L4: a non-JSON error body with a `\x80` byte is sent and reduced.
- K3: the lint also fails on a `for` target over `.stderr`, a tuple-unpack, a walrus, `" ".join(str(e).split())[:50]`
  and `json.dumps({"k": str(e)})[:200]`. K2: `fix_hint` holds no credential in the `--output` JSONL.
- J3: the lint fails on the branched assignment, `s += str(ex)`, `[str(ex)][0][:50]` and `("x %s" % ex)[:50]`.
- G3: a `logging.makeLogRecord({... "exc_text": <credential traceback>})` formats reduced. G4: a bad format string
  through a subclass record does not raise out of the `logger` call.
- G2: a `_send_json(500, {"error": ..., "stderr": <credential URL>})` and the oasis 400 body come out reduced, and
  an OpenAI-shaped error keeps its int `code`.
- G6: each success-shaped payload site comes out reduced.
- F3: the parallel path's `combined_results`, the sub-simulation's `sub_report["error"]` and `/v1/debug/install-status`
  all come out reduced.
- S5: `import maxim.utils.logging` (and `structured_logging`, `sim_logger`, `tools.base`) loads no httpx, checked
  with a subprocess and `sys.modules`.

## Decisions

- D1–D4: taken (status block).
- **D5 (taken 2026-11-25: option a).** `ToolOutput.error`, whose text reaches the planning LLM and so a cloud provider.
  - (a) Reduce it at `ToolOutput`'s construction. It is the boundary, but the model then sees `https://host` and not
    the path that failed. The model chose the URL, so the path is in its own action.
  - (b) Reduce it only where it is rendered into a prompt.
  - (c) Leave it, and file it.
  - **Consequences (code review Wire SF2, 2026-11-25):** Hippocampus failure episodes now store the reduced error.
    So `EpisodicMemory.copy_key`'s dedupe (words, tool and outcome by `repr`) merges failures that differ only by
    URL path or query, and a failed fetch is no longer recallable by its path. NAc link keys for ERROR outcomes
    (`outcome_signature`, the first 50 characters of the error) change too: an old link whose first 50 characters
    held a URL stops matching and relearns. A SUCCESS `result_summary` is outside D5's error scope and is NOT reduced
    (code review Wire SF1), so success-path NAc keys are unchanged. Stated in docs/agents/bio-memory.md §4.

## Not in scope (stated)

- Scheme-less URLs, bearer tokens and percent-encoded whole URLs.
- `print`, Rich output, `sys.excepthook`, `handleError`'s stderr dump, and the LLM server child's own output.
