import sys, importlib
from blacklist import BL, src, G
from maxim.memory.text import key_tokens
mod=importlib.import_module(sys.argv[1] if len(sys.argv)>1 else "script_v2")
P1,P2,CUE_OF=mod.P1,mod.P2,mod.CUE_OF
def K(l): return set(key_tokens(l))
def where(k): return [n for n,v in src.items() if k in v]
print("goal keys", sorted(G), "| blacklist", len(BL))
print("\n== PHASE 1")
for i,(l,role,ans) in enumerate(P1,1):
    k=K(l); ng=k-G
    print(f"P1.{i} [{role}] goal={sorted(k&G)} nongoal={sorted(ng)} answer={ans}")
    for x in sorted(ng):
        if x in BL: print(f"     COLLISION {x!r}: {where(x)}")
    for a in ans:
        if a not in k: print(f"     WARN answer key {a!r} not a key of the line (keys as {sorted(set(key_tokens(a)))})")
print("\n== PHASE 2")
facts={role:(i,K(l)-G) for i,(l,role,_) in enumerate(P1,1)}
for j,(l,role) in enumerate(P2,1):
    k=K(l); ng=k-G
    share={f"P1.{i}":sorted(ng&(K(P1[i-1][0])-G)) for i in range(1,9) if ng&(K(P1[i-1][0])-G)}
    print(f"P2.{j} [{role}] goal={sorted(k&G)} nongoal={sorted(ng)} shares_with={share}")
    for x in sorted(ng):
        if x in BL: print(f"     COLLISION {x!r}: {where(x)}")
    if j in CUE_OF:
        fi,fk=facts[CUE_OF[j]]
        ans=P1[fi-1][2]
        print(f"     cue->fact {CUE_OF[j]}: shared non-goal={sorted(ng&fk)} (n={len(ng&fk)}); answer keys the cue lacks: {[a for a in ans if a not in k]}; answer keys in cue: {[a for a in ans if a in k]}")
        other=[f"P1.{i}" for i,(l2,r2,_) in enumerate(P1,1) if r2!=CUE_OF[j] and ng&(K(l2)-G)]
        if other: print(f"     WARN shares non-goal keys with other phase-1 lines: {other}")
        if not (K(P1[fi-1][0])&G): print("     WARN fact has no goal word")
    # earlier phase-2 cue reuse
    for jj in range(1,j):
        sh=ng&(K(P2[jj-1][0])-G)
        if sh and j in CUE_OF: print(f"     note: shares {sorted(sh)} with earlier P2.{jj}")
