"""G1 simulator (cue top 5 stopping at 3, goal top 3 only under 3, distinct_copies before the cut, own_ids
excluded) + a stdlib port of the ranker pinned against the shipped _rank_by_relevance on every call."""
from maxim.memory.hippocampus_retrieval import _rank_by_relevance, distinct_copies
from maxim.memory.text import key_tokens
GOAL="escape a dungeon with a sleeping guard"
G=frozenset(key_tokens(GOAL))
STATS={"pin_calls":0,"pin_mismatch":0,"tie_groups":0}

def port_rank(pool, query, limit):
    q=set(key_tokens(query))
    if not q: return [], 0
    sc=[(len(q & m.content_tokens())/len(q), m.timestamp, m) for m in pool if q & m.content_tokens()]
    # exact (score, timestamp) ties among different records -> more than one valid ordering
    seen={}; ties=0
    for s,t,m in sc:
        seen.setdefault((s,t),[]).append(m)
    ties=sum(1 for v in seen.values() if len(v)>1)
    sc.sort(key=lambda x:(x[0],x[1]), reverse=True)
    out=[]; keys=set()
    for _,_,m in sc:
        k=m.copy_key()
        if k not in keys: out.append(m)
        if k is not None: keys.add(k)
    return out[:limit], ties

def ranked(pool, query, limit=5):
    a=_rank_by_relevance(pool, query, limit, collapse_copies=True)
    b,ties=port_rank(pool, query, limit)
    STATS["pin_calls"]+=1
    if [m.id for m in a]!=[m.id for m in b]: STATS["pin_mismatch"]+=1
    STATS["tie_groups"]+=ties
    return a

def simulate(pool, cue, goal=GOAL, own_ids=()):
    """-> list of (id, path) as _query_hippocampus_traced would log them (graph path inert, D6)."""
    pool=[m for m in pool if m.id not in set(own_ids)]
    out=[]; seen=set()
    for m in ranked(pool, cue, 5):
        if len(out)>=3: break
        if m.id not in seen: seen.add(m.id); out.append((m.id,"cue"))
    if len(out)<3 and goal:
        for m in ranked(pool, goal, 5)[:3]:
            if m.id not in seen: seen.add(m.id); out.append((m.id,"goal"))
    return out[:3]

def r3d(pool, cue, carried_ids, own_ids=(), goal=GOAL):
    """G4 on one turn. Returns dict: decisive, pairs [(carried, displaced, c_ng, d_ng)], shown, shown_wo."""
    shown=simulate(pool, cue, goal, own_ids)
    shown_ids={i for i,_ in shown}
    cue_carried=[i for i,p in shown if p=="cue" and i in carried_ids]
    goal_fill_win=[i for i,p in shown if p=="goal" and i in carried_ids]
    pool_wo=[m for m in pool if m.id not in carried_ids]
    shown_wo=simulate(pool_wo, cue, goal, own_ids)
    displaced=[i for i,p in shown_wo if p=="cue" and i not in shown_ids]
    ng=set(key_tokens(cue))-G
    byid={m.id:m for m in pool}
    pairs=[]
    for c in cue_carried:
        for d in displaced:
            a=len(ng & byid[c].content_tokens()); b=len(ng & byid[d].content_tokens())
            pairs.append((c,d,a,b))
    dec=bool(ng) and any(a>b for _,_,a,b in pairs)
    return dict(decisive=dec, pairs=pairs, shown=shown, shown_wo=shown_wo, displaced=displaced,
                cue_carried=cue_carried, goal_fill_win=goal_fill_win, ng=ng)
