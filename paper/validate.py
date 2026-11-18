"""Check citations, bilingual mathematics, frozen results, and ICML artifacts."""
from pathlib import Path
import hashlib,json,re,statistics,subprocess
P=Path(__file__).resolve().parent;R=P.parent
texts=[(P/f).read_text() for f in ['paper.md','paper-中文.md']]
titles={
 'en':'Reflective Elimination: Accumulating Design Experience in Tree-Structured Algorithm Populations',
 'zh':'反思式淘汰：在树状算法种群中积累设计经验',
}
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
x=json.loads((P/'evidence/representative_snapshot.json').read_text());assert len(x['runs'])==12
assert x['common_budget']==240
assert x['totals']['evaluations']==sum(r['evaluations'] for r in x['runs'])==3519
for a in x['aggregates']:
 vals=[r['series'][x['common_budget']-1]['best'] for r in sorted(x['runs'],key=lambda r:r['seed']) if r['task']==a['task']]
 assert vals==a['scores'] and statistics.mean(vals)==a['mean'] and statistics.stdev(vals)==a['sd']
 precision=2 if a['task']=='fssp_gls' else 4
 for v in vals+[a['mean'],a['sd']]:
  for t in texts:assert f'{v:.{precision}f}' in t,(v,precision)
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
report['representative_snapshot_sha256']=hashlib.sha256((P/'evidence/representative_snapshot.json').read_bytes()).hexdigest()
report['archived_snapshot_sha256']=hashlib.sha256((P/'evidence/snapshot.json').read_bytes()).hexdigest()
assert report['archived_snapshot_sha256']=='29a642d88d410c815c94802f60567d2db73e235dada0483c5448c1c05c4f26eb'
refs=[c for r in x['runs'] for c in r['reflection_calls']]
regions=[c for r in x['runs'] for c in r['regions']]
assert len(refs)==len(regions)==490
assert sum(c['prior_experience'] for c in refs)==147
assert sum(c['input_tokens'] for c in refs)==3959071
assert sum(c['output_tokens'] for c in refs)==88604
assert sum(c['size'] for c in regions)==2301
assert all(c['size']>=4 for c in regions)
assert all(r['config']['native']['topology']=={'max_representatives':24,'cover_distance':.15,'region_size':4} for r in x['runs'])
for language,t in zip(['en','zh'],texts):
 assert '147' in t and '30.0' in t and '4Km-2K-m' in t and '332' in t
 assert t.splitlines()[0]=='# '+titles[language]
 assert 'TREES' in t and 'Tree-structured Reflective Evolutionary Search' in t
 assert 'EvoTrace' not in t and 'ArborEvo' not in t
report['naming']={'method':'TREES','full_name':'Tree-structured Reflective Evolutionary Search','titles':titles}
report['cross_update_experience']={'reflections':490,'with_prior_experience':147,'percentage_rounded':30.0}
case=json.loads((P/'evidence/inheritance_case.json').read_text())
source=next(r for r in x['runs'] if r['source']==case['source'])
assert case['source_prefix_sha256']==source['sha256']
def normalized(s):return re.sub(r'\s+',' ',s).strip()
assert normalized(case['earlier']['experience']) in normalized(case['later']['prompt'])
assert normalized(case['later']['experience']) in normalized(case['following']['prompt'])
assert case['earlier']['evaluation']==97 and case['later']['evaluation']==104
assert case['earlier']['target_id'] in case['later']['region']['branch_ids']
assert case['following']['context']['parent_id']==case['later']['target_id']==66
assert case['following']['program']['evaluation']['fitness']>next(p['evaluation']['fitness'] for p in case['programs'] if p['id']==66)
report['case']={'seed':case['seed'],'reflection_evaluations':[97,104],'following_attempt':98,'complete_prompt_matches':True}
template=P/'vendor/icml2027'
provenance=json.loads((template/'SOURCE.json').read_text())
for file,digest in provenance['files'].items():
 assert hashlib.sha256((template/file).read_bytes()).hexdigest()==digest,file
report['official_template']='ICML 2027; vendored files match their recorded download hashes'

record=json.loads((P/'figures/current-generation-record.json').read_text())
report['figure_hashes']={}
for figure in record['figures']:
 digest=hashlib.sha256((R/figure['file']).read_bytes()).hexdigest()
 assert digest==figure['sha256'],figure['file']
 report['figure_hashes'][figure['file']]=digest

report['layout']={}
for language,name,pdf,source in zip(['en','zh'],['main','main-zh'],['trees.pdf','trees-中文.pdf'],texts):
 tex=(P/(name+'.tex')).read_text()
 assert '% Markdown SHA256: '+hashlib.sha256(source.encode()).hexdigest() in tex,(language,'LaTeX is stale')
 aux=(P/'build'/(name+'.aux')).read_text()
 main_end=re.search(r'\\newlabel\{evotrace-main-end\}\{\{[^}]*\}\{(\d+)\}',aux)
 assert main_end,(language,'missing main-end page marker')
 main_pages=int(main_end[1])
 assert main_pages<=8,(language,main_pages)
 assert tex.index(r'\bibliography{references}')<tex.index(r'\appendix')
 assert r'\clearpage' in tex[tex.index(r'\bibliography{references}'):tex.index(r'\appendix')]
 assert r'\usepackage{icml2027}' in tex and r'\bibliographystyle{vendor/icml2027/icml2027}' in tex
 assert r'\begin{algorithm}[H]' in tex and r'\begin{algorithmic}[1]' in tex
 assert tex.count(r'\begin{figure*}')==4 and tex.count(r'\begin{table*}')==1
 ref_heading='## References' if language=='en' else '## 参考文献'
 appendix_heading='## Appendix A.' if language=='en' else '## 附录 A.'
 assert source.index(ref_heading)<source.index(appendix_heading)
 metadata=subprocess.check_output(['pdfinfo',str(P/pdf)],text=True)
 assert re.search(r'^Title:\s+(.+)$',metadata,re.M)[1]==titles[language],(language,'PDF title metadata')
 assert re.search(r'Page size:\s+612 x 792 pts \(letter\)',metadata),metadata
 total_pages=int(re.search(r'Pages:\s+(\d+)',metadata)[1])
 pdf_text=subprocess.check_output(['pdftotext','-layout',str(P/pdf),'-'],text=True)
 assert 'AUTHORERR' not in pdf_text and '\ufffd' not in pdf_text
 assert 'ArborEvo' not in pdf_text and 'EvoTrace' not in pdf_text
 assert 'TREES' in pdf_text
 appendix_title='A. Background and Related Approaches' if language=='en' else 'A. 研究背景与相关方法'
 appendix_page=next(i+1 for i,page in enumerate(pdf_text.split('\f')) if appendix_title in page)
 assert appendix_page>main_pages
 assert tex.count(r'\section{')>=12
 assert 'F. Case Study' in pdf_text if language=='en' else 'F. 案例' in pdf_text
 report['layout'][language]={'main_end_page':main_pages,'appendix_start_page':appendix_page,'total_pages':total_pages,'paper_size':'Letter','main_columns':2,'appendix_columns':1}
(P/'evidence/manuscript_checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False))
