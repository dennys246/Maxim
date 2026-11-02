"""``maxim.advisor``: the atonia composition, an observer that proposes and never acts (#1315; Experimental).

Design: ``docs/plans/atonia_gate_approach_notes.md`` §4 (item 1) and §5 (item 5), with §1A winning over both.

- :mod:`maxim.advisor.log` is item 5's proposal and refusal log, the observer's one sanctioned write.
- ``open()`` and ``AdvisorSession`` (item 1) are not built yet.

This package imports nothing at import time, so ``import maxim.advisor`` keeps the observer's import closure
(§4.2) to what each submodule pulls in itself.
"""
