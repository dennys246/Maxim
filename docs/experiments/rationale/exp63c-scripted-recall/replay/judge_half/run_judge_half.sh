#!/usr/bin/env bash
# Exp 63c fidelity check, judge half: run the replay with every R3d call also judged by the shipped 63c judge port,
# plus the two probes that must FAIL. Offline: no rig, no LLM. Run from the repo root of the commit under test:
#   bash docs/experiments/rationale/exp63c-scripted-recall/replay/judge_half/run_judge_half.sh <python> <out dir>
set -u
PY=${1:?python}; OUT=${2:?out dir}
REPO=$(git rev-parse --show-toplevel); HERE="$REPO/docs/experiments/rationale/exp63c-scripted-recall/replay"
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
W=$(mktemp -d); cp -r "$HERE/." "$W/"
sed -i "s#^B = .*#B = \"$REPO/docs/experiments/data/63b_carried_recall/\"#" "$W/replay.py"  # the replay's 63b store path
cp "$REPO/scripts/o19_verdict.py" "$W/o19_verdict_head.py"
cp "$HERE/judge_half/judge_half_63c.py" "$W/"
export HOME=$(mktemp -d) MAXIM_OFFLINE=1 PYTHONPATH="$REPO/src"
cd "$W"
{ echo "code=$(git -C "$REPO" rev-parse HEAD) sha256(o19_verdict)=$(sha256sum o19_verdict_head.py | cut -d' ' -f1)"
  "$PY" -c "import maxim; print('maxim imported from', maxim.__file__)"; } | tee "$OUT/provenance.txt"
POL="r p e1 e2 e3 b4 cpart cpart2 pA pB pC pD stuckA stuckB stuckC stuckD"
run() { tag=$1; shift; env "$@" "$PY" judge_half_63c.py o19_verdict_head.py script_v5x intents_v4 $POL > "$OUT/judge_$tag.out" 2>&1; sed -n '/^JUDGE HALF/,$p' "$OUT/judge_$tag.out" | head -2; }
run base TAG=_jb
run neutral TAG=_jn R1MODE=neutral
run neutral_nogoal TAG=_jng R1MODE=neutral R2MODE=nogoal
run naut2 TAG=_j2 NAUT=2
run bare TAG=_jbare R1MODE=bare
run neutral_naut2 TAG=_jn2 R1MODE=neutral NAUT=2
# Probe 1: the port sees no carried records, so every decisive turn must disagree with the simulator.
sed 's/^    carried = set(carried_ids) \& set(records)$/    carried = set()/' judge_half_63c.py > p1.py
# Probe 2: a control word planted in a carried record, so control turn 2 must read contaminated.
sed 's/^    base_records = {m: records\[m\] for m in carried}$/    base_records = {m: records[m] for m in carried}\n    if base_records:\n        _f = next(iter(base_records)); base_records[_f] = json.loads(json.dumps(base_records[_f])); base_records[_f]["context"]["active_goal"] = "a silver bell in the chapel"/' judge_half_63c.py > p2.py
{ echo "probe1 edits applied: $(grep -c 'carried = set()$' p1.py)"
  TAG=_p1 "$PY" p1.py o19_verdict_head.py script_v5x intents_v4 $POL 2>&1 | grep -E "^r3d calls|^TOTAL"
  echo "probe2 edits applied: $(grep -c 'silver bell' p2.py)"
  TAG=_p2 "$PY" p2.py o19_verdict_head.py script_v5x intents_v4 r 2>&1 | grep -E "^  r "; } | tee "$OUT/probes.out"
