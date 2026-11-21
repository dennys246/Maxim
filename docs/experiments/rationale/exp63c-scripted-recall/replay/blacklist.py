"""Exp 63c preliminary vocabulary blacklist (today's tokenizer, pre-#1620)."""
import ast, json, sys, yaml
from maxim.memory.text import key_tokens, content_tokens
from maxim.memory.types import EpisodicMemory
WT="/home/dennys/GitSpot/Maxim/.worktrees/exp63c-design/"
S=WT+"src/maxim/"
B=WT+"docs/experiments/data/63b_carried_recall/"
GOAL="escape a dungeon with a sleeping guard"
G=set(key_tokens(GOAL))
LOGF={"debug","info","warning","error","exception","critical"}

def strings_in(node):
    """string constants in node, minus docstrings and logging-call args"""
    out=[]; skip=set()
    for n in ast.walk(node):
        if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef,ast.Module)) and n.body and isinstance(n.body[0],ast.Expr) and isinstance(n.body[0].value,ast.Constant) and isinstance(n.body[0].value.value,str):
            skip.add(id(n.body[0].value))
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in LOGF:
            for a in ast.walk(n): skip.add(id(a))
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ("sim_log","sim_imagination"):
            for a in ast.walk(n): skip.add(id(a))
    for n in ast.walk(node):
        if isinstance(n,ast.Constant) and isinstance(n.value,str) and id(n) not in skip:
            out.append(n.value)
    return out

def tool_classes(path, names):
    tree=ast.parse(open(path,encoding="utf-8").read())
    got={}
    for c in tree.body:
        if isinstance(c,ast.ClassDef):
            for st in c.body:
                if isinstance(st,(ast.Assign,ast.AnnAssign)):
                    v=st.value
                    if isinstance(v,ast.Constant) and v.value in names:
                        got[v.value]=strings_in(c)
    return got

def keys_of(strs):
    k=set()
    for s in strs: k|=set(key_tokens(s))
    return k

src={}
# 1. AUT tool templates: discovery.py whole module (sense, sense_presence, sense_tools + helpers), plus each AUT tool class
src["discovery.py (module)"]=keys_of(strings_in(ast.parse(open(S+"tools/discovery.py",encoding="utf-8").read())))
TOOLS={"tools/introspection.py":{"causal_links","concept_query","energy_status","memory_recall","predict_outcome","similarity_search","system_stats","temporal_patterns"},
 "tools/display.py":{"display_mode","request_interaction","set_scene"},
 "tools/narrative.py":{"examine","say","think"},
 "tools/reachy_stubs.py":{"focus_on_sound"},
 "simulation/tools.py":{"respond"},
 "tools/response.py":{"respond","speak"}}
tk=set(); found=[]
for f,names in TOOLS.items():
    g=tool_classes(S+f,names)
    found+= [f"{f}:{n}" for n in g]
    for n,ss in g.items(): tk|=keys_of(ss)|set(key_tokens(n))
tk|=keys_of(["focus_on_sound","Turn the head to face the currently heard sound (observation-only mode - no robot connected).","faced_sound","duration"]); found.append("tools/reachy_stubs.py:focus_on_sound")
src["AUT tool classes ("+str(len(found))+")"]=tk
# think's rendered output: format_thought_response templates
fn=[n for n in ast.walk(ast.parse(open(S+"integration/bio_enrichment.py",encoding="utf-8").read())) if isinstance(n,ast.FunctionDef) and n.name=="format_thought_response"][0]
src["think render (format_thought_response)"]=keys_of(strings_in(fn))
# body affordance tools: body YAML + guard YAML (all scalars and keys)
def yaml_strs(p):
    out=[]
    def w(x):
        if isinstance(x,dict):
            for k,v in x.items(): out.append(str(k)); w(v)
        elif isinstance(x,list):
            for v in x: w(v)
        elif isinstance(x,str): out.append(x)
    w(yaml.safe_load(open(p,encoding="utf-8"))); return out
src["bodies/base_humanoid.yaml"]=keys_of(yaml_strs(S+"_data/components/bodies/base_humanoid.yaml"))
src["npcs/guard.yaml"]=keys_of(yaml_strs(S+"_data/components/npcs/guard.yaml"))
# 2. manifest few-shot
from maxim.simulation.narrator import _MANIFEST_SYSTEM
src["manifest few-shot example"]=set(key_tokens(_MANIFEST_SYSTEM.split("Example output")[1]))
src["manifest prompt (whole)"]=set(key_tokens(_MANIFEST_SYSTEM))
# 3. 63b manifest entities, both phases (identical in both)
ENT=["heavy wooden door","snoring guard","bed a rusty key","exit a flickering lantern","guard","heavy_wooden_door","rusty_key","flickering_lantern","base_humanoid"]
src["63b manifest entities"]=keys_of(ENT)
# 4. stall + diversity templates: top-level functions of stall_detector.py
st=ast.parse(open(S+"simulation/stall_detector.py",encoding="utf-8").read())
ss=[]
for n in st.body:
    if isinstance(n,ast.FunctionDef): ss+=strings_in(n)
from maxim.simulation import stall_detector as SD
for name in dir(SD):
    pass
src["stall/diversity templates"]=keys_of(ss)|set(key_tokens(f"What is your next step toward: {GOAL}?"))
# 5. 63b records
for s,lab in (("20261107_101504","63b phase-1 records"),("20261107_102003","63b phase-2 store (31)")):
    k=set()
    for m in json.load(open(B+s+"/aut_hippocampus.json"))["memories"]:
        k|=EpisodicMemory.from_dict(m).content_tokens()
    src[lab]=k
BL=set().union(*src.values())
if __name__=="__main__":
    print("AUT tool classes found:",found)
    for k,v in src.items(): print(f"{k:42s} {len(v):4d} keys")
    print("BLACKLIST size:",len(BL), "(incl. goal keys)", "non-goal:", len(BL-G))
    json.dump({"size":len(BL),"keys":sorted(BL),"sources":{k:sorted(v) for k,v in src.items()}},open("blacklist.json","w"),indent=1)
