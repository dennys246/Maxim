import json
import os
from pathlib import Path
import traceback
import sys
import maxim
sys.path.insert(0,str(Path.cwd()))
from tests import network_guard
network_guard.install()

def check(name, fn):
    try:
        print(name, 'PASS', repr(fn()))
    except Exception as e:
        print(name, 'FAIL', type(e).__name__, str(e))
        traceback.print_exc()

root=Path(os.environ['MAXIM_DATA_HOME'])
def hippo_roundtrip():
    h=maxim.create.hippocampus(persistence_path=str(root/'hippo.json'))
    mid=h.store_observation('The wolf was near the cave entrance')
    h.save()
    loaded=maxim.load.hippocampus(str(root/'hippo.json'))
    assert mid in loaded._memories
    return {'id_preserved':mid in loaded._memories,'recall_count':len(loaded.recall(query='wolf',limit=3))}
check('documented hippo roundtrip',hippo_roundtrip)

def agent_roundtrip():
    a=maxim.create.agent('scout',personality='cautious and observant',remembers=True,learns=True)
    mid=a.hippocampus.store_observation('I saw movement in the shadows')
    a.nac.record_event('observation','saw_movement')
    before=a.export_memories()
    a.shutdown()
    b=maxim.load.agent('scout')
    after=b.export_memories()
    assert mid in b.hippocampus._memories
    assert before['episodic_memories']==after['episodic_memories']
    b.shutdown()
    return {'before':before,'after':after}
check('documented agent create/export/shutdown/load',agent_roundtrip)

def nac_roundtrip():
    n=maxim.create.nac()
    n.record_event('action','ate_mushroom')
    prediction=n.predict('action','ate_mushroom')
    n.save(str(root/'nac.json'))
    restored=maxim.load.nac(str(root/'nac.json'))
    return {'prediction':prediction,'loaded_type':type(restored).__name__}
check('documented nac example/save/load',nac_roundtrip)
def atl_roundtrip():
    a=maxim.create.atl(persistence_path=str(root/'atl.json'))
    cid,created=a.find_or_create('wolf',category='creature')
    a.save()
    b=maxim.load.atl(str(root/'atl.json'))
    restored,_=b.find_or_create('wolf',category='creature')
    assert cid==restored
    return {'concept_id_preserved':True}
check('documented atl roundtrip',atl_roundtrip)
print('network attempts',network_guard.attempts)
