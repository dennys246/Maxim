import datetime
import pathlib
import shlex
import subprocess
import sys

out = pathlib.Path('/tmp/maxim-codex-audit-131')
commands = {
 'exp60': 'scripts/survival_world/exp60_run.py verdict --data docs/experiments/data/exp60_trials.jsonl',
 'exp61': 'scripts/survival_world/exp61_run.py verdict --data docs/experiments/data/exp61_pairs.jsonl --campaign-id exp61-campaign-1',
 'exp62': 'scripts/survival_world/exp62_run.py verdict --data docs/experiments/data/exp62_rows.jsonl --campaign-id exp62-rungA-1',
 'r3': 'scripts/survival_world/r3_run.py report --data docs/experiments/data/r3_bench.jsonl --campaign-id r3-bench-1 --gauntlet docs/experiments/data/r3_gauntlet.json',
 'r3-amended': 'scripts/survival_world/r3_run.py report --data docs/experiments/data/r3_bench.jsonl --campaign-id r3-bench-1 --gauntlet docs/experiments/data/r3_gauntlet.json --amended',
 'exp56': 'scripts/analyze_exp56.py --in docs/experiments/data/exp56_rebaseline_1204/56_four_arm.jsonl --assert-noop-fails',
 'prereg-at-tag': 'scripts/lint_prereg_precedes_data.py --ref v1.3.1',
 'silent-swallows': 'scripts/lint_no_silent_swallows.py',
 'function-ratchet': 'scripts/lint_function_length.py',
 'architecture': '-m maxim --audit-architecture',
 'mypy-ci': '-m mypy src/maxim/__init__.py src/maxim/api.py src/maxim/session.py src/maxim/create.py src/maxim/load.py src/maxim/hivemind/ --ignore-missing-imports --follow-imports=silent',
 'mypy-all': '-m mypy src/maxim --ignore-missing-imports --follow-imports=silent',
 'ruff': '-m ruff check src/ tests/',
 'ruff-format-check': '-m ruff format --check src/ tests/',
 'help': '-m maxim --help',
 'collect': '-m pytest tests/ --collect-only -q',
}
for name, command in commands.items():
    args = [sys.executable, *shlex.split(command)]
    with (out / (name + '.txt')).open('w') as f:
        f.write('$ python ' + command + '\nUTC=' + datetime.datetime.now(datetime.timezone.utc).isoformat() + '\n')
        f.flush()
        try:
            p = subprocess.run(args, stdout=f, stderr=subprocess.STDOUT, timeout=300)
            f.write('\nEXIT_STATUS=' + str(p.returncode) + '\n')
            print(name, p.returncode, flush=True)
        except subprocess.TimeoutExpired:
            f.write('\nTIMEOUT=300s\n')
            print(name, 'TIMEOUT', flush=True)
