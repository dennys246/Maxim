"""Exp 63c power replay (freeze: main with #1620).

python replay.py <script_module> [intents_module] [policies...]
"""

import copy
import importlib
import json
import sys
import uuid

from maxim.memory.types import EpisodicMemory
from simcore import STATS, r3d

B = "/home/dennys/GitSpot/Maxim/.worktrees/exp63c-design/docs/experiments/data/63b_carried_recall/"
S = importlib.import_module(sys.argv[1])
INT = importlib.import_module(sys.argv[2] if len(sys.argv) > 2 else "intents_v3")
import os
R1MODE = os.environ.get("R1MODE", "")
if R1MODE == "neutral":
    INT.R1 = ["I'll look around for a way to escape past the sleeping guard."] * 7 + ["llm_fallback"]
elif R1MODE == "bare":
    INT.R1 = ["I'll look around."] * 7 + ["llm_fallback"]
if os.environ.get("R2MODE") == "nogoal":
    INT.R2 = ["I'll look around."] * 7 + [None]
Z1 = os.environ.get("Z1") == "1"
NAUT = int(os.environ.get("NAUT", "1"))
TAG = os.environ.get("TAG", "")
POLICIES = sys.argv[3:] or ["r", "p", "e1", "e2", "e3", "w", "w+p", "w+e3"]
P1, P2, CUE_OF = S.P1, S.P2, S.CUE_OF
raw = {m["capture_seq"]: m for m in json.load(open(B + "20261107_102003/aut_hippocampus.json"))["memories"]}
ARR_T = raw[0]
# 63b's real tool record per turn (phase 1: seqs 1,3,..,15; phase 2: 17,..,29; turn 8 had no action)
TOOL1 = [raw[s] for s in (1, 3, 5, 7, 9, 11, 13, 15)]
TOOL2 = [raw[s] for s in (17, 19, 21, 23, 25, 27, 29)] + [None]
WORLD = raw[9]["outcome"]["result"]["output"]  # sense_presence text naming 63b's manifest entities
FACTS = [r for (_, r, _) in P1 if r.startswith("fact")]
from maxim.memory.text import key_tokens  # answer keys are stored as key_tokens (prereg; F2 NIT at freeze)
ANS = {r: [k for w in a for k in key_tokens(w)] for (_, r, a) in P1 if r.startswith("fact")}
CUE_TURN = {v: k for k, v in CUE_OF.items()}
clock = [0.0, 0]


def stamp(d):
    clock[0] += 20.0
    clock[1] += 1
    d["id"] = str(uuid.uuid4())
    d["timestamp"] = d["created_at"] = 1794071700.0 + clock[0]
    d["capture_seq"] = clock[1]


def arrival(text):
    d = copy.deepcopy(ARR_T)
    stamp(d)
    d["perception"]["cli_input"] = d["perception"]["transcript"] = text
    d["perception"]["observations"] = {"detections": [], "transcript": text, "cli_input": text, "salience": 0.8, "novelty": 0.7}
    return EpisodicMemory.from_dict(d)


def aut(src, intent, world):
    d = copy.deepcopy(src)
    stamp(d)
    d["context"]["active_goal"] = intent
    d["decision"]["intent"]["goal"] = intent
    if world:
        d["outcome"]["result"] = dict(d["outcome"]["result"] or {})
        d["outcome"]["result"]["scene"] = WORLD
    return EpisodicMemory.from_dict(d)


def run(policy):
    clock[0], clock[1] = 0.0, 0
    parts = policy.split("+")
    world = "w" in parts
    echo = ([x for x in parts if x != "w"] or ["r"])[0]
    carried, role_of, tag = [], {}, {}
    for i, (line, role, _) in enumerate(P1, 1):
        a = arrival(line)
        carried.append(a)
        role_of[a.id], tag[a.id] = role, f"P1.{i}"
        t = aut(TOOL1[i - 1], INT.R1[i - 1], world)
        carried.append(t)
        role_of[t.id], tag[t.id] = "aut", f"P1.{i}.aut"
    cids = {m.id for m in carried}
    new, rows, last_fact, distractor = [], [], None, False
    for j, (line, role) in enumerate(P2, 1):
        a = arrival(line)
        pool = carried + new + [a]
        d = r3d(pool, line, cids, own_ids=(a.id,))
        byid = {m.id: m for m in pool}
        if any(role_of.get(i) == "goal-words distractor" for i, _ in d["shown"]):
            distractor = True
        armed = (not any(d["ng"] & m.content_tokens() for m in carried)) if role == "control" else None
        comp = None
        if j in CUE_OF:
            ak = set(ANS[CUE_OF[j]])
            comp = any(ak <= byid[i].content_tokens() for i, _ in d["shown"] if i in cids)
        best = max(d["pairs"], key=lambda x: x[2] - x[3], default=None)
        win = disp = margin = None
        if best:
            win, disp, margin = tag[best[0]], tag[best[1]], f"{best[2]} v {best[3]}"
        echo_load = {k: sum(1 for m in new if k in m.content_tokens()) for k in sorted(d["ng"])}
        rows.append(
            dict(turn=j, role=role, decisive=d["decisive"], winner=win, displaced=disp, margin=margin,
                 shown=[(tag[i], p) for i, p in d["shown"]], shown_wo=[(tag[i], p) for i, p in d["shown_wo"]],
                 new_pool=len(new), armed=armed, completion=comp, echo_load=echo_load,
                 goal_fill_win=[tag[i] for i in d["goal_fill_win"]])
        )
        new.append(a)
        tag[a.id], role_of[a.id] = f"P2.{j}", "arrival"
        for i, _ in d["shown"]:
            if i in cids and role_of[i].startswith("fact"):
                last_fact = role_of[i]
        if TOOL2[j - 1] is None or (Z1 and j == 1):
            continue
        if echo == "r":
            intent = INT.R2[j - 1]
        elif echo == "p":
            intent = INT.restate(last_fact) if last_fact else INT.R2[j - 1]
        elif echo.startswith("n") and len(echo) == 3:  # sweep: fact <X> restated (base form) in the first <k> phase-2 records
            intent = INT.restate("fact " + echo[1]) if j <= int(echo[2]) else INT.R2[j - 1]
        elif echo.startswith("i") and len(echo) == 3:  # sweep, inflected form
            intent = INT.restate_inflected(["fact " + echo[1]]) if j <= int(echo[2]) else INT.R2[j - 1]
        elif echo == "b4":  # every record restates all four facts verbatim
            intent = "I remember " + "; ".join(INT.BASE[f] for f in FACTS) + "."
        elif echo == "cpart":  # fact C partially (no 'squeak'), with both of cue C's goal words, every record
            intent = "I remember the padlock on the cell hatch near the sleeping guard."
        elif echo == "cpart2":  # same, only 'guard' (no 'sleep')
            intent = "I remember the padlock on the cell hatch near the guard."
        elif echo.startswith("stuck"):  # 63b's pattern: ONE recalled fact restated in every record; forced as if t1 surfaced it
            intent = INT.restate("fact " + echo[-1])
        elif echo.startswith("p") and len(echo) == 2:  # p, but turn 1 (R3a view) surfaced fact <X>
            intent = INT.restate("fact " + echo[-1]) if j == 1 else (INT.restate(last_fact) if last_fact else INT.R2[j - 1])
        else:  # e<n>: n facts per record, adversarially the ones whose cue comes soonest after this turn
            n = int(echo[1])
            upcoming = sorted(FACTS, key=lambda f: (CUE_TURN[f] <= j, CUE_TURN[f]))
            intent = INT.restate_inflected(upcoming[:n])
        for r_ in range(NAUT):
            t = aut(TOOL2[j - 1], intent, world)
            new.append(t)
            tag[t.id], role_of[t.id] = f"P2.{j}.aut" + (f"#{r_}" if r_ else ""), "aut"
    return rows, distractor


OUT = {}
for pol in POLICIES:
    rows, dist = run(pol)
    cues = [r for r in rows if r["turn"] in CUE_OF]
    k = sum(r["decisive"] for r in cues)
    print(f"\n===== policy {pol}: decisive cue turns {[r['turn'] for r in cues if r['decisive']]} k={k}/4; "
          f"goal-words distractor surfaced={dist}")
    for r in rows:
        extra = ""
        if r["role"] == "control":
            extra = f" CTRL armed={r['armed']}" + (" ARMED-DECISIVE(C5)" if r["armed"] and r["decisive"] else "")
        print(f" t{r['turn']} [{r['role']}]{extra} decisive={r['decisive']} winner={r['winner']} "
              f"displaced={r['displaced']} margin={r['margin']} newpool={r['new_pool']} completion={r['completion']}")
        print(f"      shown={r['shown']} | without-carried={r['shown_wo']}")
        print(f"      goal_fill_win={r['goal_fill_win']} echo_load(new records holding each cue non-goal key)={r['echo_load']}")
    OUT[pol] = dict(rows=rows, distractor=dist, k=k)
json.dump(OUT, open(f"replay{TAG}_{sys.argv[1]}_{sys.argv[2] if len(sys.argv) > 2 else 'intents_v3'}.json", "w"), indent=1)
print(f"\nranker pin: calls={STATS['pin_calls']} port mismatches={STATS['pin_mismatch']} "
      f"exact (score,timestamp) tie groups={STATS['tie_groups']}")
