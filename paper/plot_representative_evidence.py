"""Publication figures from frozen representative-cover data and one real case."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P = Path(__file__).resolve().parent
snapshot = json.loads((P / 'evidence/representative_snapshot.json').read_text())
case = json.loads((P / 'evidence/inheritance_case.json').read_text())
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
                     'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 9,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'savefig.dpi': 300})
colors = ['#0072B2', '#D55E00', '#009E73']
styles = ['-', '--', ':']
fig, axes = plt.subplots(2, 4, figsize=(6.75, 3.15), layout='constrained')
for j, (task, label) in enumerate([('tsp_construct', 'TSP'), ('cvrp_aco', 'CVRP'),
                                  ('fssp_gls', 'FSSP'), ('op_aco', 'OP')]):
    for run in [r for r in snapshot['runs'] if r['task'] == task]:
        seed = run['seed']
        points = [v for v in run['series'] if v['evaluation'] <= snapshot['common_budget'] and v['best'] is not None]
        axes[0, j].step([p['evaluation'] for p in points], [p['best'] for p in points],
                         where='post', color=colors[seed], ls=styles[seed], lw=1, label=f'Seed {seed}')
        points = [v for v in run['tree_series'] if v['evaluation'] <= snapshot['common_budget']]
        axes[1, j].step([p['evaluation'] for p in points], [p['nodes'] for p in points],
                         where='post', color=colors[seed], ls=styles[seed], lw=1)
    axes[0, j].set_title(label)
    axes[1, j].set_xlabel('Evaluations')
    axes[1, j].set_ylim(0, 115)
    for row in [0, 1]:
        axes[row, j].set_xlim(0, snapshot['common_budget'])
        axes[row, j].set_xticks([0, 120, 240])
        axes[row, j].grid(axis='y', alpha=.18)
        axes[row, j].tick_params(labelsize=7)
axes[0, 0].set_ylabel('Best training fitness')
axes[1, 0].set_ylabel('Algorithm nodes')
axes[0, 3].legend(frameon=False, fontsize=6.5, loc='upper right')
for extension in ['pdf', 'png']:
    fig.savefig(P / f'figures/representative-search.{extension}')
plt.close(fig)

# Numerical lineage plot. Edges connect actual parents, not improvement steps.
programs = {p['id']: p for p in case['programs']}
programs[case['following']['program']['id']] = dict(case['following']['program'],
                                                  evaluation_index=case['following']['evaluation_index'])
fig, ax = plt.subplots(figsize=(6.75, 2.65), layout='constrained')
for ids, color, label, style in [([13, 17, 19, 36, 70], '#0072B2', 'MST development', '-'),
                                ([17, 20, 66, 98], '#D55E00', 'Constructive tail development', '--')]:
    for u, v in zip(ids, ids[1:]):
        assert programs[v]['parent_id'] == u
    ax.plot([programs[i]['evaluation_index'] for i in ids],
            [programs[i]['evaluation']['fitness'] for i in ids],
            marker='o', ms=3.8, color=color, lw=1.3, ls=style, label=label)
for i in [13, 17, 19, 36, 70, 20, 66, 98]:
    p = programs[i]
    offset = (0, 6) if i not in [19, 66] else (0, -13)
    if i == 20:
        offset = (12, 5)
    ax.annotate(f'$v_{{{i}}}$', (p['evaluation_index'], p['evaluation']['fitness']),
                xytext=offset, textcoords='offset points', ha='center', fontsize=8)
for phase, label, y in [('earlier', '$E_{17\\to70}$ formed', 8.08),
                        ('later', '$E_{13\\to66}$ inherits it', 7.77)]:
    x = case[phase]['evaluation']
    ax.axvline(x, color='#777777', lw=.8, ls=':')
    ax.annotate(label, (x, y), xytext=(-4, 0), textcoords='offset points', ha='right', fontsize=8,
                bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': .6})
ax.set_xlim(10, max(p['evaluation_index'] for p in programs.values())+8)
ax.set_ylim(5.9, 8.55)
ax.set_xlabel('Program evaluations')
ax.set_ylabel('Training tour length')
ax.grid(axis='y', alpha=.15)
ax.legend(loc='upper center', bbox_to_anchor=(.59, 1.04), frameon=False, fontsize=7.5)
for extension in ['pdf', 'png']:
    fig.savefig(P / f'figures/inheritance-case.{extension}')
plt.close(fig)
print('Saved representative-search and inheritance-case as PDF and PNG.')
