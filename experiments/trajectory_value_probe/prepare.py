"""Freeze prefixes and lock candidates BEFORE exposing their historical futures."""
import argparse
from collections import Counter
from datetime import datetime
import gzip
import hashlib
import json
from pathlib import Path
import random
from zoneinfo import ZoneInfo
from .features import digest,fitness,node_features,inspect_code

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments_result/analysis/trajectory_feature_probe_20261103'


def qualifies(row,task):
    if task=='graph_colouring':
        d=row['dynamics']; c=row['coverage']['all']
        return (d.get('slope_4') or 0)>1e-9 and c['attempts']>c['improving']
    f=row['formation']
    return (f.get('fitness_delta') or 0)>1e-9 and (f['repair_used'] or (f.get('ast_distance') or 0)>.1)


def prepare():
    if (OUT/'candidates.locked.json').exists():raise FileExistsError('Candidate selection is already locked')
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=[];rows=[];prefixes={}
    sources=sorted((ROOT/'experiments_result/traceaad_base_schema').glob('*/*/native/state.json'))
    for source in sources:
        run=source.parent.parent; cfg=json.loads((run/'run_config.json').read_text())
        assert cfg['batch']=='20261031_base_schema_co6'
        raw=source.read_bytes();state=json.loads(raw)
        attempts=[];cost=0
        for a in sorted(state['attempts'],key=lambda a:a['id']):
            if not a['completed']:break
            n=len(a['evaluations'])
            if cost+n>200:break
            attempts.append(a);cost+=n
        ids={a['program_id'] for a in attempts if a['program_id'] is not None}|{0}
        archive={p['id']:p for p in state['archive'] if p['id'] in ids}
        key=f"{cfg['task']}_s{cfg['seed']}"
        folder=OUT/'snapshots'/key;folder.mkdir(parents=True,exist_ok=True)
        # Full snapshot is frozen without deriving outcomes beyond the prefix.
        with gzip.open(folder/'state.full.json.gz','wb') as f:f.write(raw)
        prefix={'key':key,'task':cfg['task'],'seed':cfg['seed'],'cutoff_attempt':attempts[-1]['id'],
                'evaluation_cost':cost,'archive':list(archive.values()),'attempts':attempts,'run_config':cfg,
                'source_state_sha256':hashlib.sha256(raw).hexdigest()}
        (folder/'prefix.json').write_text(json.dumps(prefix,ensure_ascii=False))
        versions={}
        for a in attempts:
            if a['repair_used'] and len(a['evaluations'])>1:
                eid=a['evaluations'][0]['id'];path=source.parent/a['evaluations'][0]['path']
                if path.exists():versions[eid]=json.loads(path.read_text())
        (folder/'repair_first_versions.json').write_text(json.dumps(versions,ensure_ascii=False))
        metadata=cfg['task_definition']['metadata']
        local=[]
        for pid,p in archive.items():
            if fitness(p) is None:continue
            f=node_features(pid,archive,attempts,metadata,versions)
            f.update(route=key,task=cfg['task'],seed=cfg['seed'],cutoff_attempt=prefix['cutoff_attempt'])
            local.append(f);rows.append(f)
        prefixes[key]=prefix
        manifest.append({'route':key,'source':str(source.relative_to(ROOT)),'state_sha256':prefix['source_state_sha256'],
            'prefix_sha256':hashlib.sha256((folder/'prefix.json').read_bytes()).hexdigest(),
            'prefix_attempts':len(attempts),'prefix_evaluations':cost,'valid_nodes':len(local),
            'repair_versions_available':len(versions),'task':cfg['task'],'seed':cfg['seed']})
        print(key,'prefix',len(attempts),'valid',len(local),flush=True)
    (OUT/'features.prefix.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    selected=[]
    for task in ('graph_colouring','jssp_construct'):
        for seed in (0,1,2):
            pool=[r for r in rows if r['task']==task and r['seed']==seed]
            target=[r for r in pool if qualifies(r,task)]
            controls=[r for r in pool if not qualifies(r,task)]
            if task=='graph_colouring':
                target.sort(key=lambda r:(-(r['coverage']['all']['same_ast']+r['coverage']['all']['known_inactive_only']),r['node_id']))
            else:
                target.sort(key=lambda r:(-int(r['formation']['repair_used']),r['formation'].get('node_retention') if r['formation'].get('node_retention') is not None else 1,r['node_id']))
            pair=None
            for candidate in target:
                prefix=prefixes[candidate['route']]; archive={p['id']:p for p in prefix['archive']}
                h=inspect_code(archive[candidate['node_id']]['code'])['hash']
                possible=[c for c in controls if inspect_code(archive[c['node_id']]['code'])['hash']!=h]
                if not possible:continue
                control=min(possible,key=lambda c:(abs(c['fitness']-candidate['fitness'])/max(abs(candidate['fitness']),1e-9),c['node_id']))
                pair=(candidate,control);break
            if pair:
                for role,r in zip(('feature_target','fitness_matched_control'),pair):
                    selected.append({**r,'role':role,'candidate_key':f"{r['route']}_P{r['node_id']}",
                        'prefix_file':f"snapshots/{r['route']}/prefix.json"})
                break
    jobs=[]
    for r in selected:
        for op in ('Refine','Explore'):
            for repeat in (0,1):
                jobs.append({'job_id':f"{r['candidate_key']}_{op.lower()}_r{repeat}",'candidate_key':r['candidate_key'],
                             'first_operator':op,'repeat':repeat,'request_budget':6})
    random.Random(20261103).shuffle(jobs)
    stamp=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    protocol_hash=hashlib.sha256((OUT/'PROTOCOL.locked.md').read_bytes()).hexdigest()
    locked={'locked_at':stamp,'protocol_sha256':protocol_hash,'feature_source_sha256':digest((Path(__file__).parent/'features.py').read_text()),
            'candidates':selected,'jobs':jobs,'future_outcomes_read_for_selection':False}
    (OUT/'candidates.locked.json').write_text(json.dumps(locked,ensure_ascii=False,indent=2))
    manifest={'created_at':stamp,'source_batch':'20261031_base_schema_co6','prefix_limit_evaluations':200,
              'candidate_lock_sha256':hashlib.sha256((OUT/'candidates.locked.json').read_bytes()).hexdigest(),'routes':manifest}
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print('LOCKED',[(r['candidate_key'],r['role'],r['fitness']) for r in selected],flush=True)


def reveal():
    locked=json.loads((OUT/'candidates.locked.json').read_text());manifest=json.loads((OUT/'manifest.json').read_text())
    assert hashlib.sha256((OUT/'candidates.locked.json').read_bytes()).hexdigest()==manifest['candidate_lock_sha256']
    outcomes=[]
    for candidate in locked['candidates']:
        folder=OUT/'snapshots'/candidate['route'];state=json.loads(gzip.decompress((folder/'state.full.json.gz').read_bytes()))
        archive={p['id']:p for p in state['archive']};cut=candidate['cutoff_attempt'];pid=candidate['node_id']
        future=[a for a in state['attempts'] if a['completed'] and a['id']>cut]
        direct=[a for a in future if a['parent_id']==pid]
        descendants={pid};chain=[]
        for a in future:
            if a['parent_id'] in descendants:
                if a['program_id'] is not None:descendants.add(a['program_id'])
                chain.append(a)
        def summarize(attempts):
            vals=[fitness(archive.get(a['program_id'])) for a in attempts];vals=[v for v in vals if v is not None]
            return {'attempts':len(attempts),'evaluations':sum(len(a['evaluations']) for a in attempts),
                'best_valid':min(vals) if vals else None,'observed':bool(attempts),
                'events':[{'attempt_id':a['id'],'program_id':a['program_id'],'parent_id':a['parent_id'],'operator':a['operator'],
                           'fitness':fitness(archive.get(a['program_id'])),'evaluations':len(a['evaluations'])} for a in attempts]}
        outcomes.append({'candidate_key':candidate['candidate_key'],'role':candidate['role'],'fitness':candidate['fitness'],
                         'direct':summarize(direct),'future_parent_edge_descendants':summarize(chain)})
    (OUT/'history.revealed.json').write_text(json.dumps({'revealed_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'candidate_lock_sha256':manifest['candidate_lock_sha256'],'results':outcomes},ensure_ascii=False,indent=2))
    print('REVEALED',[(r['candidate_key'],r['direct']['attempts'],r['direct']['best_valid']) for r in outcomes])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','reveal']);args=parser.parse_args()
    (prepare if args.action=='prepare' else reveal)()
