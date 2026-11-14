"""Regenerate manuscript plots from the frozen evidence snapshot."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent
x=json.loads((p/'evidence/snapshot.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axs=plt.subplots(2,4,figsize=(11,4.4),layout='constrained')
colors=['#0072B2','#D55E00','#009E73'];styles=['-','--',':']
for j,(task,label) in enumerate([('tsp_construct','TSP'),('cvrp_aco','CVRP'),('fssp_gls','FSSP'),('op_aco','OP')]):
 for run in [r for r in x['runs'] if r['task']==task]:
  s=run['seed'];v=[v for v in run['series'] if v['evaluation']<=x['common_budget'] and v['best'] is not None]
  axs[0,j].step([a['evaluation'] for a in v],[a['best'] for a in v],where='post',color=colors[s],ls=styles[s],lw=1.1,label=f'Seed {s}')
  v=run['tree_series'];axs[1,j].step([a['evaluation'] for a in v],[a['nodes'] for a in v],where='post',color=colors[s],ls=styles[s],lw=1.1)
 axs[0,j].set_title(label);axs[0,j].set_xlim(1,x['common_budget']);axs[1,j].set_xlabel('Program evaluations')
 for i in [0,1]:axs[i,j].grid(axis='y',alpha=.18);axs[i,j].tick_params(labelsize=7)
axs[0,0].set_ylabel('Best training fitness (lower is better)');axs[1,0].set_ylabel('Retained algorithms');axs[0,3].legend(frameon=False,fontsize=7)
fig.savefig(p/'figures/early-search.png',dpi=240);fig.savefig(p/'figures/early-search.pdf');plt.close(fig)
