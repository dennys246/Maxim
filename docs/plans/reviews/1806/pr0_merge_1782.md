# #1806 PR 0: delta review of the merge with origin/main (#1782), commit 52022ee1

Lenses: executor, architecture, wire integrity (docs/CODE_REVIEW.md). Scope: the PR-side changes in the merge
(parents 6b3652df = PR #1817 head, and origin/main with #1782). `git diff origin/main 52022ee1 -- src/maxim/utils/http.py`
is empty, as the resolution claims.

Probes ran in a `git archive 52022ee1` copy under the session scratchpad (`r6/copy`), scratch HOME,
`PYTHONPATH=<copy>/src` (import verified to resolve to the copy). Nothing touched the network, the robot or `~/.maxim`.

## Verdict

One SHOULD-FIX: a stale test makes the merge commit red. Nothing else. No new class: the finding is the class question (3)
asked about (a reference to the removed `proxy` API), and a grep for `proxy="never"` misses it because the test
spells it `kw.get("proxy") == "never"`.

## SHOULD-FIX

**S1. `tests/unit/test_motion_path_1499.py::test_the_fresh_read_is_private_and_bounded` still asserts the removed API
and fails on 52022ee1.** Line 916: `assert kw.get("proxy") == "never"`. The controller now passes `strict=True`, so
`kw.get("proxy")` is None. Reproduced in the copy: `1 failed, 393 passed` across the 1806 / doctor / audio /
1782 / 1715 / 1499 test files, and this is the one failure. Failure scenario: CI's `unit-tests` goes red on the pushed
merge. This assertion is also the ONLY pin on the controller readback's `strict=True`; the probe, DoA and
maxim-diagnostics reads are pinned elsewhere. Fix: `assert kw.get("strict") is True  # #1806 PR 0: ...`. Verified in the
copy: with that edit the test passes, and if `strict=True` is then deleted from `controller.py::_read_state_full` the
test fails, so the edited assertion does pin it. `tests/unit/test_motion_path_1499.py` did not show up in the merge's
test list, because the merge left it unchanged.

## The five questions

1. **Does `strict=True` with "private" give the claimed guarantees?** Yes. The merged `utils/http.py` `fetch_url(strict=True,
   address_policy="private")` selects `_external_policy_private_strict`, registered `("private", strict=True)`.
   `get_client` builds it through `_checked_client(..., strict=True)`, which gives:
   - `mounts=None`, so there are no proxy mounts;
   - `trust_env=False` on the client, so httpx reads no env or system proxy (`getproxies`);
   - a default transport from `_checked_transport`, whose `HTTPTransport` keeps its own default `trust_env=True`, so the
     env CA (`SSL_CERT_FILE`/`SSL_CERT_DIR`) is still honoured;
   - `_PrivateOnlyBackend`, which resolves once and dials the address it checked.

   The endpoint is in `_DEADLINE_ENDPOINTS`. `_install_deadline_backend` leaves the checked backend in place because it
   subclasses `_DeadlineBackend`, so each checked dial is also deadline-clamped (#1715). These guarantees are the same as
   the PR's own version: the PR had registered the same endpoint name with the same `("private", True)` spec. The PR's
   tests pass on the merged tree (bypass with a dead proxy set, no mounts and `trust_env` False, public address refused
   on the dialled address, drip refused at about `deadline_s`, deadline backend on every fetch_url endpoint).
2. **Is anything lost by dropping the private-only restriction?** Not for the daemon readers. All four hard-code
   `address_policy="private"`, and each has a test pinning `"private"`: probe (`test_1806_pr0_probe_daemon.py`),
   DoA and maxim-diagnostics (`test_1806_pr0_daemon_reads_direct.py`), and the controller (`test_motion_path_1499.py`,
   once S1 is fixed). The generic loss is that `strict=True` with the default `"any"` is now a valid, unchecked,
   proxyless fetch instead of a `ValueError`. That is #1782's intended meaning ("strict" = no proxy), not a daemon-path
   regression, and no daemon reader relies on the `ValueError`.
3. **Leftover references to `proxy="never"`, `ProxyMode` or `_EXTERNAL_STRICT_POLICY_ENDPOINTS`?** None in src/, scripts/,
   docs/ or changelog.d/. One in tests/: S1. A broad grep (`proxy.{0,12}never|ProxyMode|STRICT_POLICY`) turns up only
   unrelated prose.
4. **Are the fragments and docs still accurate?** Yes. `changelog.d/1806-pr0-robot-config-groundwork.fixed.md` ("dialled
   directly at a private address, never through an environment proxy ... A robot reachable only through a proxy, or at
   a public address, now fails these reads") is true of `strict=True` + "private". The `docs/agents/llm-routing.md`
   entry is accurate (see N1). `docs/user/safety.md` and the other user docs make no proxy claim about daemon reads.
   The security fragment does not touch HTTP.
5. **Does the control test still prove `strict` is load-bearing?** Yes. Probe: making the `elif strict:` branch dead in
   the copy's `fetch_url` fails `test_a_configured_proxy_is_bypassed_...` and `test_a_public_address_is_still_refused_...`.
   `test_control_the_default_mode_lets_the_same_proxy_carry_it` passes on the merged tree, so the dead proxy really does
   carry the non-strict request. Both proxy tests skip under the offline latch, where every client is strict. That was
   true before the merge too, and the skip is stated in the tests.

## NIT

- **N1.** The `docs/agents/llm-routing.md` "A LAN robot daemon" entry lists the probe status read, the controller
  readback and the DoA reader as users. It omits maxim-diagnostics' own `/docs` read (`utils/reachy_diagnostics.py`),
  which is a fourth direct `strict=True` call apart from the probe it also uses. The entry could also say "env CA
  honoured", to match the `fetch_url` docstring. Its guard list could add `tests/unit/test_motion_path_1499.py` (the
  controller pin, after S1).
