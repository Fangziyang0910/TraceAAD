"""Check the manuscript's citations, shared mathematics, and frozen results."""
from pathlib import Path
import json,re,statistics
P=Path(__file__).resolve().parent;R=P.parent
texts=[(R/'docs'/f).read_text() for f in ['paper.md','paper-中文.md']]
keys=set(re.findall(r'@\w+\{([^,]+),',(P/'references.bib').read_text()))
report={'citations':{},'math_equal':False,'score_cells_checked':0}
for name,t in zip(['en','zh'],texts):
 used=set(re.findall(r'@([A-Za-z][A-Za-z0-9_-]*)',t))
 assert used<=keys,(name,used-keys)
 assert not re.search(r'\[\[(EDITOR|FIGURE)|\bTODO\b|\bTBD\b',t)
 report['citations'][name]=sorted(used)
assert report['citations']['en']==report['citations']['zh']
math=[[''.join(x.split()) for x in re.findall(r'\$\$(.*?)\$\$',t,re.S)] for t in texts]
assert math[0]==math[1],math
report['math_equal']=True
x=json.loads((P/'evidence/snapshot.json').read_text());assert len(x['runs'])==12
for a in x['aggregates']:
 vals=[r['series'][x['common_budget']-1]['best'] for r in x['runs'] if r['task']==a['task']]
 assert vals==a['scores'] and statistics.mean(vals)==a['mean'] and statistics.stdev(vals)==a['sd']
 precision=2 if a['task']=='fssp_gls' else 4
 for v in vals+[a['mean'],a['sd']]:
  for t in texts:assert f'{v:.{precision}f}' in t
  report['score_cells_checked']+=1
for name in ['main','main-zh']:
 log=P/'build'/(name+'.log')
 if log.exists():
  s=log.read_text(errors='replace')
  assert not re.search(r'Overfull|undefined citations|undefined references|Missing character:',s),(name,'typesetting issue')
assert sum(r['attached_nodes'] for r in x['runs']) == sum(r['retained_nodes']+r['removed_nodes'] for r in x['runs'])
report['attached_nodes']=sum(r['attached_nodes'] for r in x['runs'])
report['retained_nodes']=sum(r['retained_nodes'] for r in x['runs'])
report['frozen_evaluations']=sum(r['evaluations'] for r in x['runs'])
(P/'evidence/manuscript_checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False))
