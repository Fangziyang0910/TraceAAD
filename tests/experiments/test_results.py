"""A saved winner and its test scores must refer to the same program individual."""

import json
from dataclasses import asdict
from http.server import ThreadingHTTPServer
from threading import Thread
from urllib.request import urlopen

import pytest

from experiments.infra import evaluate
from experiments.infra.artifacts import pick_best_sample
from experiments.visualization import ResultsMonitor
from experiments.visualization.server import make_request_handler
from tests.support import TinyEvaluation, TokenLLM, response
from traceaad import Config, TraceAAD
from traceaad.storage import write_json


@pytest.mark.parametrize("fail", [False, True])
def test_heldout_evaluates_the_frozen_training_winner(tmp_path, monkeypatch, fail):
    config = Config(budget=3, roots=3)
    method = TraceAAD(
        evaluation=TinyEvaluation(),
        llm=TokenLLM(response(2), response(5), response(1)),
        run_dir=tmp_path,
        config=config,
    )
    summary = method.run()
    write_json(
        tmp_path / "run_config.json", {"task": "tsp_construct", "method_params": asdict(config)}
    )
    frozen = (tmp_path / "best_program.py").read_text()

    class TestEvaluation(TinyEvaluation):
        def evaluate_program(self, source, function):
            if fail:
                raise ValueError("the selected program failed on heldout")
            return function(1)

    monkeypatch.setattr(evaluate, "heldout_task", lambda *args: TestEvaluation())
    result = evaluate.evaluate_run(tmp_path, ["eval_50"], condition="traceaad")[0]
    assert summary["best"]["fitness"] == -5 and result["fitness"] == (None if fail else 5)
    assert result["node_id"] == summary["best"]["id"] == 2
    assert (tmp_path / "best_program.py").read_text() == frozen
    monkeypatch.setattr(
        evaluate, "heldout_evaluator", lambda *a, **k: pytest.fail("repeat evaluation")
    )
    assert evaluate.evaluate_run(tmp_path, ["eval_50"])[0] == result


def test_result_reader_and_monitor_follow_individual_ids_and_generation_budget(tmp_path):
    directory = tmp_path / "traceaad" / "tsp_construct" / "run"
    method = TraceAAD(
        evaluation=TinyEvaluation(),
        llm=TokenLLM("{bad", response(2, analysis="first"), response(2, analysis="second")),
        run_dir=directory,
        config=Config(budget=3, roots=2),
    )
    method.run()
    write_json(directory / "run_config.json", {"task": "tsp_construct", "budget": 3})
    chosen, samples = pick_best_sample(directory)
    assert chosen["node_id"] == 1 and chosen["sample_order"] == 2
    assert len(samples) == 2 and samples[0]["node_id"] != samples[1]["node_id"]
    monitor = ResultsMonitor(tmp_path)
    detail = monitor.run_detail("traceaad", "tsp_construct", "run")
    assert detail["budget_used"] == 3
    assert detail["best"]["code"] == method.programs[1].code
    assert detail["curve"][-1]["fitness"] == -2
    before = {p: p.read_bytes() for p in directory.rglob("*") if p.is_file()}
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_request_handler(monitor))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        for path in ["/", "/app.js", "/style.css"]:
            with urlopen(base + path) as result:
                assert result.status == 200 and result.read()
        with urlopen(base + "/api/run?batch=traceaad&task=tsp_construct&name=run") as result:
            assert json.load(result)["best"]["id"] == chosen["node_id"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert {p: p.read_bytes() for p in directory.rglob("*") if p.is_file()} == before


def test_comparison_never_averages_across_models(tmp_path):
    models = {"rep1": "qwen3.8-27b-awq", "rep2": "Qwen3.8-27B", "pilot": "deepseek-v4.1-flash"}
    for name, model in models.items():
        write_json(tmp_path / "eoh" / "tsp_construct" / name / "run_config.json",
                   {"task": "tsp_construct", "budget": 3, "llm": {"model": model}})
    monitor = ResultsMonitor(tmp_path)
    cohorts = {c["id"]: c["runs"] for c in monitor.cohorts()}
    assert cohorts == {"eoh@qwen3.8-27b": 2, "eoh@deepseek-v4.1-flash": 1}
    compared = monitor.compare(["eoh@qwen3.8-27b"])["tasks"][0]["cohorts"][0]["runs"]
    assert sorted(run["name"] for run in compared) == ["rep1", "rep2"]


def dashboard_run(directory, *, seed=0, model='qwen3.8-27b-awq', status='finished'):
    task=directory.parent.name
    write_json(directory/'run_config.json',dict(task=task,budget=1000,seed=seed,repeat=seed+1,
               score_metric='raw_objective',llm=dict(model=model)))
    write_json(directory/'summary.json',dict(status=status,budget=1000,budget_used=1000,
               best=dict(id=1,fitness=10+seed,action='Init')))
    write_json(directory/'selection.json',dict(selected_node=1))
    (directory/'best_program.py').write_text('def f(x):\n    return x\n')


def test_primary_index_keeps_eighteen_runs_and_never_substitutes_relaxed_scores(tmp_path):
    from benchmarks.tasks import CO_TASKS, PRIMARY_SPLITS, scale_of_split
    allowed=[]
    for task in CO_TASKS:
        for seed in range(3):
            name=f'rep{seed+1}'
            run=tmp_path/'eoh'/task/name
            dashboard_run(run,seed=seed,model='Qwen3.8-27B' if seed==2 else 'qwen3.8-27b-awq',
                          status='running' if seed==1 else 'finished')
            allowed.append(f'{task}/{name}')
            if seed==1:
                continue
            scale=str(scale_of_split(task,PRIMARY_SPLITS[task][0]))
            relaxed=dict(task=task,node_id=1,verification='verified',scale=scale,
                         variant='relaxed_time_cpu120_wall600',fitness=1)
            values=[relaxed]
            if not (task=='tsp_construct' and seed==2):
                values.insert(0,{**relaxed,'variant':'','fitness':None if task=='tsp_construct' else 20})
            write_json(run/'heldout.json',values)
    for name,model in [('deepseek_pilot','deepseek-v4.1-flash'),('extra_seed0','Qwen3.8-27B')]:
        dashboard_run(tmp_path/'eoh'/'tsp_construct'/name,model=model)
    dashboard_run(tmp_path/'diagnostic'/'tsp_construct'/'smoke')
    write_json(tmp_path/'primary_runs.json',dict(schema=1,methods={'eoh':allowed}))
    original={p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    monitor=ResultsMonitor(tmp_path)
    assert monitor.batches()==[dict(id='eoh',label='eoh')]
    assert monitor.overview('eoh')['summary']['runs']==18
    cohorts=monitor.cohorts()
    assert len(cohorts)==1 and cohorts[0]['id']=='eoh' and cohorts[0]['runs']==18
    assert cohorts[0]['variant']=='' and cohorts[0]['model_name']=='qwen3.8-27b'
    compared=monitor.compare(['eoh'])
    assert all(len(t['cohorts'][0]['runs'])==3 for t in compared['tasks'])
    tsp=next(t for t in compared['tasks'] if t['key']=='tsp_construct')['cohorts'][0]['runs']
    assert tsp[0]['heldout'] and all(v is None for v in tsp[0]['heldout'].values())
    assert tsp[1]['status']=='running' and not tsp[1]['heldout']
    assert not tsp[2]['heldout']  # a relaxed-only result cannot stand in for the main test
    assert {p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}==original


def test_primary_index_changes_invalidate_cached_views(tmp_path):
    for name in ('main','extra'):
        dashboard_run(tmp_path/'eoh'/'tsp_construct'/name)
    index=tmp_path/'primary_runs.json'
    write_json(index,dict(schema=1,methods={'eoh':['tsp_construct/main']}))
    monitor=ResultsMonitor(tmp_path)
    before=monitor.state_signature('eoh')
    assert monitor.overview('eoh')['tasks'][0]['runs'][0]['name']=='main'
    write_json(index,dict(schema=1,methods={'eoh':['tsp_construct/extra']}))
    assert monitor.state_signature('eoh')!=before
    assert monitor.overview('eoh')['tasks'][0]['runs'][0]['name']=='extra'


def test_compare_api_returns_all_thirteen_methods(tmp_path):
    for i in range(13):
        dashboard_run(tmp_path/f'method{i}'/'tsp_construct'/'rep1')
    monitor=ResultsMonitor(tmp_path)
    names=[c['id'] for c in monitor.cohorts()]
    assert len(names)==13
    server=ThreadingHTTPServer(('127.0.0.1',0),make_request_handler(monitor))
    thread=Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        with urlopen(f'http://127.0.0.1:{server.server_port}/api/compare?cohorts='+','.join(names)) as response:
            result=json.load(response)
        assert len(result['cohorts'])==13
        assert len(result['tasks'][0]['cohorts'])==13
    finally:
        server.shutdown();server.server_close();thread.join()


def test_comparison_defaults_include_traceaad_versions_and_preserve_old_links():
    from pathlib import Path
    import shutil
    import subprocess
    node=shutil.which('node')
    if not node:
        pytest.skip('Node is unavailable for the JavaScript view check')
    source=Path(__file__).resolve().parents[2]/'experiments/visualization/static/app.js'
    check=r'''
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const code=fs.readFileSync(process.argv[1],'utf8');
const part=code.slice(code.indexOf('const CMP ='),code.indexOf('async function loadCompare()'));
const data={cohorts:['traceaad_v13','traceaad_v12_1',...Array.from({length:12},(_,i)=>`baseline${i}`)].map(id=>
 ({id,batch:id,label:id,model_name:'qwen3.8-27b',runs:18,variant:'',heldout_tasks:[]}))};
function scope() {
 const chips={innerHTML:''};
 const context=vm.createContext({S:{batch:'traceaad_v13'},getJSON:async()=>({data}),
   $:()=>chips,esc:String,seriesColor:i=>String(i)});
 vm.runInContext(part,context);
 return {context,chips};
}
(async()=>{
 const fresh=scope();await vm.runInContext('loadCohorts()',fresh.context);
 assert.equal(vm.runInContext('CMP.ids.length',fresh.context),14);
 assert.ok(vm.runInContext('CMP.ids.includes("traceaad_v12_1")',fresh.context));
 vm.runInContext('renderChips()',fresh.context);
 assert.equal((fresh.chips.innerHTML.match(/class="chip /g)||[]).length,14);
 const old=scope();
 vm.runInContext(`CMP.ids=['baseline0@qwen3.8-27b','baseline0::relaxed_time_cpu120_wall600@qwen3.8-27b','baseline1'];CMP.ref=CMP.ids[0]`,old.context);
 await vm.runInContext('loadCohorts()',old.context);
 assert.equal(vm.runInContext('JSON.stringify(CMP.ids)',old.context),'["baseline0","baseline1"]');
 assert.equal(vm.runInContext('CMP.ref',old.context),'baseline0');
})().catch(e=>{console.error(e);process.exitCode=1});
'''
    subprocess.run([node,'-e',check,str(source)],check=True,capture_output=True,text=True)


def test_comparison_marks_partial_training_budget():
    from pathlib import Path
    import shutil
    import subprocess
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is unavailable for the JavaScript view check')
    source = Path(__file__).resolve().parents[2]/'experiments/visualization/static/app.js'
    check = r'''
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const code=fs.readFileSync(process.argv[1],'utf8');
const part=code.slice(code.indexOf('const CMP ='),code.indexOf('function drawAllDots()'));
const nodes={'cmp-summary':{},'cmp-tasks':{}};
const data={cohorts:[{id:'v13',label:'TraceAAD V13'}],tasks:[{key:'tsp_construct',label:'TSP',unit:'距离',scales:[],cohorts:[
 {id:'v13',runs:[{train:12,status:'stopped',budget_used:200,budget:1000}]}]}]};
const context=vm.createContext({$:id=>nodes[id],esc:String,seriesColor:String,fmt:String,drawAllDots:()=>{},data});
vm.runInContext(part,context);
vm.runInContext("CMP.ids=['v13'];CMP.ref='v13';CMP.data=data;renderCompare()",context);
assert.ok(nodes['cmp-tasks'].innerHTML.includes('未满预算'));
assert.ok(!nodes['cmp-tasks'].innerHTML.includes('运行中'));
'''
    subprocess.run([node, '-e', check, str(source)], check=True, capture_output=True, text=True)


def test_traceaad_versions_use_separate_formal_batches(tmp_path):
    for method in ('traceaad_v12_1', 'traceaad_v13', 'traceaad_v14', 'eoh'):
        dashboard_run(tmp_path/method/'tsp_construct'/'rep1')
    dashboard_run(tmp_path/'traceaad_v12_1'/'tsp_construct'/'deepseek_pilot', model='deepseek-v4.1-flash')
    historical = tmp_path/'traceaad_v12_1'/'fssp_gls'/'rep1'
    dashboard_run(historical)
    config = json.loads((historical/'run_config.json').read_text())
    del config['score_metric']
    config['revision'] = 'research-protocol-v6-20261012-raw-objective'
    write_json(historical/'run_config.json', config)
    write_json(tmp_path/'traceaad_v12_1'/'batch_formal.json', dict(batch='formal', status='finished', plan=[
        dict(task=task, run_name='rep1') for task in ('tsp_construct', 'fssp_gls')]))
    write_json(tmp_path/'traceaad_v13'/'batch_formal.json', dict(batch='formal', status='stopped', plan=[
        dict(task='tsp_construct', run_name='rep1')]))
    write_json(tmp_path/'traceaad_v13'/'tsp_construct'/'rep1'/'summary.json',
               dict(status='running', budget=1000, budget_used=200, best=dict(fitness=12)))
    write_json(tmp_path/'primary_runs.json', dict(schema=1,
        methods={m:['tsp_construct/rep1'] for m in ('traceaad_v14', 'eoh')},
        visualization_batches={'traceaad_v12_1':'formal', 'traceaad_v13':'formal'},
        labels={'traceaad_v12_1':'TraceAAD V12.1', 'traceaad_v13':'TraceAAD V13',
                'traceaad_v14':'TraceAAD V14'}))
    monitor=ResultsMonitor(tmp_path)
    assert {b['id']:b['label'] for b in monitor.batches()} == {
        'traceaad_v12_1':'TraceAAD V12.1', 'traceaad_v13':'TraceAAD V13',
        'traceaad_v14':'TraceAAD V14', 'eoh':'eoh'}
    cohorts={c['id']:c for c in monitor.cohorts()}
    assert set(cohorts)=={'traceaad_v12_1', 'traceaad_v13', 'traceaad_v14', 'eoh'}
    assert cohorts['traceaad_v12_1']['runs']==2
    overview = monitor.overview('traceaad_v12_1')
    assert next(t for t in overview['tasks'] if t['key']=='fssp_gls')['unit']=='makespan'
    result = monitor.compare(['traceaad_v12_1', 'traceaad_v13', 'traceaad_v14'])
    assert [c['label'] for c in result['cohorts']] == ['TraceAAD V12.1', 'TraceAAD V13', 'TraceAAD V14']
    runs = next(t for t in result['tasks'] if t['key']=='tsp_construct')['cohorts']
    assert [[r['name'] for r in c['runs']] for c in runs] == [['rep1'], ['rep1'], ['rep1']]
    assert runs[1]['runs'][0]['status']=='stopped'
    assert runs[1]['runs'][0]['budget_used']==200 and runs[1]['runs'][0]['budget']==1000


def test_stop_receipt_does_not_override_a_resumed_journal(tmp_path):
    run = tmp_path/'traceaad_v13_1'/'tsp_construct'/'rep1'
    dashboard_run(run, status='running')
    summary = json.loads((run/'summary.json').read_text())
    summary['budget_used'] = 200
    write_json(run/'summary.json', summary)
    write_json(tmp_path/'traceaad_v13_1'/'batch_formal.json', dict(batch='formal', status='running', plan=[
        dict(task='tsp_construct', run_name='rep1', session='session1')]))
    receipt = tmp_path/'traceaad_v13_1'/'stop_formal.json'
    write_json(receipt, dict(at='2999-01-01T00:00:00+00:00', sessions=['session1']))
    monitor=ResultsMonitor(tmp_path)
    before = monitor.state_signature('traceaad_v13_1')
    assert monitor.overview('traceaad_v13_1')['summary']['stopped']==1
    write_json(receipt, dict(at='2020-01-01T00:00:00+00:00', sessions=['session1']))
    (run/'events.jsonl').write_text('')
    assert monitor.state_signature('traceaad_v13_1') != before
    assert monitor.overview('traceaad_v13_1')['summary']['running']==1
