"""Census of static duplicates and a locked random sample for output probes."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
from zoneinfo import ZoneInfo
from .features import edge_features, fitness

OUT=Path(__file__).resolve().parents[2]/'experiments_result/analysis/trajectory_feature_probe_20261103'


def prepare():
    destination=OUT/'prevalence';destination.mkdir(exist_ok=True)
    if (destination/'sample.locked.json').exists():raise FileExistsError('Sample already locked')
    groups=defaultdict(list);counts=defaultdict(Counter);allrows=[]
    for path in sorted((OUT/'snapshots').glob('*/prefix.json')):
        s=json.loads(path.read_text());archive={p['id']:p for p in s['archive']}
        n=s['run_config']['task_definition']['metadata']['problem_size'] if s['task']=='graph_colouring' else None
        for a in s['attempts']:
            if a['operator'] not in ('Refine','Explore','Crossover'):continue
            p=archive.get(a['parent_id']);c=archive.get(a['program_id']);pf=fitness(p);cf=fitness(c)
            key=(s['task'],a['operator']);counts[key]['attempts']+=1
            if pf is None or cf is None:
                counts[key]['missing_valid_pair']+=1;continue
            counts[key]['valid_pairs']+=1
            if cf<pf-1e-9:counts[key]['improved']+=1;continue
            e=edge_features(p,c,n)
            row={'pair_id':f"{s['key']}_A{a['id']}",'route':s['key'],'task':s['task'],'operator':a['operator'],
                 'parent_id':p['id'],'child_id':c['id'],'parent_fitness':pf,'child_fitness':cf,
                 'same_fitness':abs(cf-pf)<=1e-9,**e}
            groups[key].append({**row,'parent_code':p['code'],'child_code':c['code']});allrows.append(row)
            for subset in ('nonimproving',*(['same_fitness'] if row['same_fitness'] else [])):
                counts[key][subset]+=1
                for field in ('same_ast','known_inactive_only','numeric_only'):
                    counts[key][subset+'_'+field]+=int(row[field])
    rng=random.Random(20261103);sample=[];summary=[]
    for key,rows in sorted(groups.items()):
        selected=rng.sample(rows,min(8,len(rows)));sample.extend(selected)
        summary.append({'task':key[0],'operator':key[1],**counts[key],'sample_size':len(selected)})
    plan={'locked_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'seed':20261103,
          'population':'All valid nonimproving parent-child pairs in 18 frozen schema prefixes (at most 200 evaluations each)',
          'sample':'Uniform without replacement, up to 8 pairs per task/operator; all historical seeds pooled',
          'inputs':'First two original training instances; TSP 3 fixed greedy-route states per instance; FSSP NEH and reversed NEH per instance',
          'comparison':'Exact recursive array equality of function outputs on all probes, matched Python/NumPy seeds; not full semantic equivalence or outer-search equality',
          'execution':'Trace added/replaced child statement start lines excluding imports, definitions and docstrings; no hits is sample-specific inactivity',
          'timeout':'30 seconds per program for all fixed probes; failures remain missing and receive bounds',
          'sampled_pairs':sample}
    (destination/'sample.locked.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2))
    (destination/'census.json').write_text(json.dumps(summary,indent=2))
    (destination/'pairs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in allrows))
    (destination/'manifest.json').write_text(json.dumps({'sample_sha256':hashlib.sha256((destination/'sample.locked.json').read_bytes()).hexdigest(),
        'prefixes':{str(p.parent.name):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((OUT/'snapshots').glob('*/prefix.json'))}},indent=2))
    for s in summary:print(json.dumps(s),flush=True)


def full_census():
    """Supplementary all-age census; does not change the locked probe sample."""
    import gzip
    counts=defaultdict(Counter);rows=[]
    for path in sorted((OUT/'snapshots').glob('*/state.full.json.gz')):
        state=json.loads(gzip.decompress(path.read_bytes()))
        prefix=json.loads((path.parent/'prefix.json').read_text());task=prefix['task']
        archive={p['id']:p for p in state['archive']}
        n=prefix['run_config']['task_definition']['metadata']['problem_size'] if task=='graph_colouring' else None
        for a in state['attempts']:
            if not a['completed'] or a['operator'] not in ('Refine','Explore','Crossover'):continue
            p=archive.get(a['parent_id']);c=archive.get(a['program_id']);pf=fitness(p);cf=fitness(c)
            key=(task,a['operator']);counter=counts[key];counter['attempts']+=1
            if pf is None or cf is None:counter['missing_valid_pair']+=1;continue
            counter['valid_pairs']+=1
            if cf<pf-1e-9:counter['improved']+=1;continue
            e=edge_features(p,c,n)
            row={'pair_id':f"{prefix['key']}_A{a['id']}",'task':task,'operator':a['operator'],
                 'parent_id':p['id'],'child_id':c['id'],'same_fitness':abs(cf-pf)<=1e-9,
                 'after_prefix':a['id']>prefix['cutoff_attempt'],**e};rows.append(row)
            for subset in ('nonimproving',*(['same_fitness'] if row['same_fitness'] else [])):
                counter[subset]+=1
                for field in ('same_ast','known_inactive_only','numeric_only'):counter[subset+'_'+field]+=int(row[field])
        print(prefix['key'],'complete',flush=True)
    summary=[{'task':key[0],'operator':key[1],**count} for key,count in sorted(counts.items())]
    (OUT/'prevalence/census.full.json').write_text(json.dumps(summary,indent=2))
    (OUT/'prevalence/pairs.full.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--full-census',action='store_true');a=p.parse_args()
    full_census() if a.full_census else prepare()
