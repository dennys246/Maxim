"""Calibration scripts.

These are not pytest tests in the conventional sense — they sweep tunable
parameters against fixture data and emit human-readable evidence used to
justify the chosen defaults that ship in production code. The scripts are
expected to be re-runnable so future maintainers can re-derive the
defaults if the underlying embedding model or fixtures change.

Each calibration script SHOULD:

- Be executable as ``python -m tests.calibration.<name>`` and print a
  summary table to stdout that documents the parameter sweep.
- Be optionally collectible as a pytest test (a single ``test_<name>``
  function that asserts the published default still passes the criteria
  used to choose it). The pytest entry point is the regression guard;
  the ``__main__`` entry point is the human-driven sweep.
"""
