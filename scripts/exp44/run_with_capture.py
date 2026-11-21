"""Exp 44 Pass-1 runner: install class-level paired-prompt capture, then run a
status-quo sim. Thin wrapper — no edits to src/.

Installs ``install_classlevel_capture`` (patches ``PromptBuilder.build_prompt``,
filtered to the AUT's embodied decisions) and then hands off to the target
runner script with the remaining argv.

The capture is class-level and in-process, so the runner must build the AUT in THIS process. Usage::

    PYTHONPATH=src python scripts/exp44/run_with_capture.py \
        --capture-log data/exp44/paired_prompts.jsonl \
        --runner <a runner that builds the AUT in-process> -- <its argv>

Everything after ``--`` is forwarded verbatim to the runner as its argv.

For an llm-primary capture, do not use the default ``--runner``: ``scripts/benchmark_exp42_preference.py`` runs
each sim as a ``maxim --sim`` subprocess with ``MAXIM_AUTO_SPAWN_LLM_SERVER=0``, so a patch here never reaches the
child's AUT and an llm-primary AUT has no backend
(``docs/plans/archive/exp44_substrate_counterfactual_harness.md``, "Wiring reality"). The Exp 44 vehicle is a direct
run with the in-process hook (``orchestrator.py``'s ``MAXIM_EXP44_CAPTURE_LOG``)::

    MAXIM_EXP44_CAPTURE_LOG=data/exp44/paired_prompts.jsonl \
        maxim --sim cradle_pref_a --embodiment bodies/infant_humanoid_chilled \
        --aut-mode llm-primary --aut-model qwen2.5-32b-instruct --interactive false --sim-max-turns 40
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
# Make this module's sibling importable and ensure the repo's src is on the path.
sys.path.insert(0, str(_HERE.parent))
_REPO = _HERE.parents[2]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from capture_paired_prompts import install_classlevel_capture  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-log", required=True, help="paired-prompt JSONL output")
    ap.add_argument(
        "--runner",
        default="scripts/benchmark_exp42_preference.py",
        help="the status-quo sim runner to execute under capture",
    )
    ap.add_argument("--all-prompts", action="store_true", help="capture every prompt, not just embodied AUT ones")
    ap.add_argument("forwarded", nargs=argparse.REMAINDER, help="args after -- forwarded to the runner")
    args = ap.parse_args()

    runner = (_REPO / args.runner) if not Path(args.runner).is_absolute() else Path(args.runner)
    if not runner.exists():
        print(f"runner not found: {runner}", file=sys.stderr)
        return 2

    # argparse.REMAINDER keeps the leading "--"; drop it so the runner sees clean argv.
    fwd = list(args.forwarded)
    if fwd and fwd[0] == "--":
        fwd = fwd[1:]

    cap = install_classlevel_capture(args.capture_log, embodied_only=not args.all_prompts)
    print(f"[exp44] capture installed -> {args.capture_log} (embodied_only={not args.all_prompts})")
    try:
        sys.argv = [str(runner), *fwd]
        runpy.run_path(str(runner), run_name="__main__")
    finally:
        cap.uninstall()
        print(f"[exp44] capture stopped; {cap._n} rows written to {args.capture_log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
