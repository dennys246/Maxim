- **`maxim.advisor` checked against AvServ plan 50 §18** (#1783): `docs/user/advisor.md` §Conformance checks the
  observer point by point and names the test behind each one. It states the offline latch's Python-level boundary,
  and lists where the code says more than §18: the `tool` refusal layer, `AdvisorReportWriteFailed`, log records
  left by a non-`ok` turn, `prompt.tokens = 0` on `prompt_build_error`, and three extra report fields. A new guard,
  `test_observer_installs_no_signal_handler`, fails if the observer ever installs a signal handler.
