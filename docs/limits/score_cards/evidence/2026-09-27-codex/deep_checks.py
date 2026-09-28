import collections
import datetime
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import zipfile

out = Path('/tmp/maxim-codex-audit-131')
def read(name): return json.loads((out/name).read_text())

pypi=read('pypi.json'); release=read('github-release.json')
print('RELEASE_TIMELINE', json.dumps({k:release[k] for k in ('tagName','createdAt','publishedAt','targetCommitish')}))
assets={a['name']:a for a in release['assets']}
for f in pypi['releases']['1.3.1']:
    local=out/'artifacts'/f['filename']
    digest=hashlib.sha256(local.read_bytes()).hexdigest()
    print('ARTIFACT',f['filename'],'PyPI_upload',f['upload_time_iso_8601'],'sha256',digest,'MATCH_PYPI',digest==f['digests']['sha256'],'GH_digest',assets[f['filename']].get('digest'))
wheel=next((out/'artifacts').glob('*.whl'))
with zipfile.ZipFile(wheel) as z:
    matches=[]; differences=[]; absent=[]
    for name in z.namelist():
        if not name.startswith('maxim/') or not name.endswith('.py'): continue
        p=Path('src')/name
        if not p.exists(): absent.append(name)
        elif z.read(name)!=p.read_bytes(): differences.append(name)
        else: matches.append(name)
    print('WHEEL_PY_SOURCE',len(matches),'exact matches; differing',differences,'absent',absent)
    print('TAG_PY_NOT_IN_WHEEL',sorted(str(p)[4:] for p in Path('src/maxim').rglob('*.py') if str(p)[4:] not in z.namelist()))
print('PYPI_README_EQUAL',pypi['info']['description'].strip()==Path('README.md').read_text().strip())
print('RELEASE_NOTES_EQUAL',release['body'].strip()==Path('docs/announcements/release_1_3_1.md').read_text().strip())
print('RULESET',json.dumps(read('ruleset-detail.json'),indent=2))
for name in ('tag-nightly-run.json','latest-scheduled-run.json'):
    run=read(name)
    print('RUN',name,{k:run[k] for k in ('databaseId','headSha','event','createdAt','updatedAt','status','conclusion')})
    print('JOBS',[(j['name'],j['conclusion'],j['startedAt'],j['completedAt']) for j in run['jobs']])

spec=importlib.util.spec_from_file_location('prereg',Path('scripts/lint_prereg_precedes_data.py'))
m=importlib.util.module_from_spec(spec); sys.modules[spec.name]=m; spec.loader.exec_module(m)
for pre,dat in [
 ('exp60_drowning_avoidance_prereg.md','exp60_trials.jsonl'),
 ('exp61_shared_fear_prereg.md','exp61_pairs.jsonl'),
 ('exp62_pressure_interoception_prereg.md','exp62_rows.jsonl'),
 ('r3_survival_benchmark_prereg.md','r3_bench.jsonl'),
 ('protocols/exp56_four_arm_sharing_preregistration.md','exp56_rebaseline_1204/56_four_arm.jsonl')]:
    path=Path('docs/experiments/data')/dat
    facts=m.data_facts(path)
    print('PROVENANCE',dat,'prereg_first_main_epoch',m.first_commit_time(Path.cwd(),'v1.3.1',Path('docs/experiments')/pre),'data_facts',{k:getattr(facts,k) for k in facts.__slots__})
    records=[json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    def walk(v,k):
        if isinstance(v,dict):
            if k in v: yield v[k]
            for x in v.values(): yield from walk(x,k)
        elif isinstance(v,list):
            for x in v: yield from walk(x,k)
    for k in ('working_tree_dirty_src_scripts','allow_dirty','executed_git_hash','git_hash'):
        print(k,dict(collections.Counter(str(v) for r in records for v in walk(r,k))))

root=Path('docs/experiments/data/rerun_exp10_2026-09-27')
for line in (root/'SHA256SUMS.uncompressed').read_text().splitlines():
    if not line.strip(): continue
    expected,name=line.split(maxsplit=1); p=root/name.lstrip('*')
    data=p.read_bytes() if p.exists() else gzip.decompress(Path(str(p)+'.gz').read_bytes())
    print('EXP10_HASH',name,hashlib.sha256(data).hexdigest()==expected)
for d in sorted(root.iterdir()):
    if not d.is_dir(): continue
    hip=json.loads((d/'aut_hippocampus.json').read_text())
    report=json.loads((d/'report.json').read_text())
    print('EXP10_SESSION',d.name,'hippo_keys',list(hip),'report_finish',{k:v for k,v in report.items() if k in ('finish_reason','total_turns','turns','termination_reason')})
