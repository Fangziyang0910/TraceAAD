"""Exercise the same sockets and isolated workers used by independent runs."""

from concurrent.futures import ThreadPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import tempfile
import threading
import time

import pytest

from core.scheduling import SchedulerError, SchedulerServer, SchedulerSession, scheduler_status
from tests.support import TinyEvaluation
from traceaad.common.instance_evaluation import InstanceProgramEvaluator


@pytest.fixture
def daemon():
    cpus = sorted(os.sched_getaffinity(0))[:2]
    if len(cpus) < 2:
        pytest.skip('two CPU IDs needed for concurrency checks')
    with tempfile.TemporaryDirectory(prefix='trace-sched-') as directory:
        path = str(Path(directory) / 's')
        endpoints = [{'base_url': 'http://one/v1', 'model': 'test', 'slots': 2},
                     {'base_url': 'http://two/v1', 'model': 'test', 'slots': 1}]
        with SchedulerServer(path, cpus, endpoints) as server:
            thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .02})
            thread.start()
            try:
                yield path, server
            finally:
                server.shutdown()
                thread.join(timeout=3)


def eventually(check):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if check():
            return
        time.sleep(.02)
    assert check()


def test_cpu_fairness_growth_and_independence(daemon):
    path, _ = daemon
    with SchedulerSession(path, 'cpu', 'first') as a, SchedulerSession(path, 'cpu', 'second') as b:
        first = a.poll(2)
        assert len(first) == 2
        assert b.poll(2) == []
        with SchedulerSession(path, 'gpu') as gpu:
            slot = gpu.acquire_gpu()
            assert slot
            # New demand gets the next available core before the larger job.
            assert a.poll(2, first[:1]) == []
            assert b.poll(2) == first[:1]
            assert a.poll(0, first[1:]) == []
            assert b.poll(2) == first[1:]
            assert scheduler_status(path)['cpu']['used'] == 2
            assert scheduler_status(path)['gpu']['used'] == 1
    eventually(lambda: scheduler_status(path)['cpu']['used'] == scheduler_status(path)['gpu']['used'] == 0)


def test_gpu_capacity_routing_and_endpoint_cooldown(daemon):
    path, _ = daemon
    sessions = [SchedulerSession(path, 'gpu') for _ in range(4)]
    try:
        slots = [s.acquire_gpu() for s in sessions[:3]]
        assert len(set(slots)) == 3
        assert sum(slot.startswith('0:') for slot in slots) == 2
        assert sessions[3].poll(1) == []
        sessions[0].poll(0, [slots[0]], failed=True)
        assert sessions[3].poll(1) == []  # Endpoint 0 is cooling down.
        index = next(i for i, slot in enumerate(slots) if slot.startswith('1:'))
        sessions[index].poll(0, [slots[index]])
        assert sessions[3].poll(1)[0].startswith('1:')
        assert scheduler_status(path)['gpu']['endpoints'][0]['cooldown_seconds'] > 0
    finally:
        for session in sessions:
            session.close()


def _hold_resources(path, sender):
    cpu = SchedulerSession(path, 'cpu')
    gpu = SchedulerSession(path, 'gpu')
    sender.send((cpu.poll(2), gpu.acquire_gpu()))
    sender.close()
    time.sleep(30)


def test_dead_client_releases_both_pools(daemon):
    path, _ = daemon
    context = multiprocessing.get_context('spawn')
    receiver, sender = context.Pipe(False)
    process = context.Process(target=_hold_resources, args=(path, sender))
    process.start()
    sender.close()
    try:
        assert receiver.poll(10)
        assert len(receiver.recv()[0]) == 2
    finally:
        process.kill()
        process.join(5)
        receiver.close()
        process.close()
    eventually(lambda: scheduler_status(path)['cpu']['used'] == scheduler_status(path)['gpu']['used'] == 0)


def test_existing_daemon_is_not_unlinked(daemon):
    path, server = daemon
    with pytest.raises(OSError):
        SchedulerServer(path, server.cpus, server.endpoints)
    assert scheduler_status(path)['instance_id'] == server.instance_id


def test_gpu_polling_preserves_queue_order(daemon):
    path, _ = daemon
    sessions = [SchedulerSession(path, 'gpu') for _ in range(5)]
    try:
        held = [s.acquire_gpu() for s in sessions[:3]]
        assert sessions[3].poll(1) == sessions[4].poll(1) == []
        assert sessions[3].poll(1) == []
        sessions[0].poll(0, [held[0]])
        assert sessions[4].poll(1) == []
        assert sessions[3].poll(1) == [held[0]]
    finally:
        for session in sessions:
            session.close()


class Instances(TinyEvaluation):
    def __init__(self):
        super().__init__()
        self.n_instance = 4
        self._datasets = list(range(4))
        self.fork_proc = False

    def evaluate_program(self, source, function):
        return sum(function(i) for i in self._datasets) / len(self._datasets)


def _evaluate_instances(path, directory, job, sender):
    code = f'''import json, os, time
def score(x):
    started = time.monotonic()
    time.sleep(.15)
    ended = time.monotonic()
    with open({directory!r} + '/{job}-' + str(x), 'w') as f:
        json.dump([started, ended, list(os.sched_getaffinity(0))], f)
    return x
'''
    try:
        result = InstanceProgramEvaluator(Instances(), scheduler_socket=path).evaluate(code, 'x',
                                                                                timeout_seconds=5, n_workers=4)
        sender.send(result)
    finally:
        sender.close()


def test_evaluations_share_cpu_bound_and_pin_each_instance(daemon, tmp_path):
    path, server = daemon
    context = multiprocessing.get_context('spawn')
    jobs = []
    for job in ('a', 'b'):
        receiver, sender = context.Pipe(False)
        process = context.Process(target=_evaluate_instances, args=(path, str(tmp_path), job, sender))
        process.start()
        sender.close()
        jobs.append((process, receiver))
    try:
        for process, receiver in jobs:
            assert receiver.poll(25)
            result = receiver.recv()
            assert result['fitness'] == 1.5, result
            assert result['evaluations'][0]['scheduler']['peak_workers'] <= 2
            assert {r['cpu_id'] for r in result['evaluations'][0]['instances']} <= set(server.cpus)
            process.join(5)
            assert process.exitcode == 0
        intervals = [json.loads(p.read_text()) for p in tmp_path.iterdir()]
        assert len(intervals) == 8
        for start, _, affinity in intervals:
            assert len(affinity) == 1 and affinity[0] in server.cpus
            running = [row for row in intervals if row[0] <= start < row[1]]
            assert len(running) <= 2
            assert len({row[2][0] for row in running}) == len(running)
    finally:
        for process, receiver in jobs:
            if process.is_alive():
                process.kill()
            process.join(5)
            process.close()
            receiver.close()
    eventually(lambda: scheduler_status(path)['cpu']['used'] == 0)


def test_waiting_for_cpu_does_not_use_instance_budget(daemon):
    path, _ = daemon
    with SchedulerSession(path, 'cpu') as holder:
        held = holder.poll(2)
        with ThreadPoolExecutor(1) as executor:
            future = executor.submit(InstanceProgramEvaluator(Instances(), scheduler_socket=path).evaluate,
                                      'def score(x): return x', 'x', timeout_seconds=2, n_workers=2)
            eventually(lambda: scheduler_status(path)['cpu']['waiting'] > 0)
            time.sleep(2.1)
            holder.poll(0, held)
            result = future.result(timeout=15)
        assert result['fitness'] == 1.5
        assert result['evaluations'][0]['scheduler']['queue_seconds'] >= 2


def test_scheduled_llm_routes_and_records_queue_time(daemon):
    from core.llm import generate
    from experiments.infra.scheduled_llm import ScheduledLLM
    path, _ = daemon
    class Client:
        def __init__(self, **kwargs):
            self.url = kwargs['base_url']
            self.model = kwargs['model']
        def draw_sample_with_details(self, prompt, **kwargs):
            return {'content': self.url, 'usage': {'completion_tokens': 2}, 'model': self.model}
        def count_tokens(self, text):
            return 2
        def count_prompt_tokens(self, text):
            return 3
        def close(self):
            pass
    with SchedulerSession(path, 'gpu') as held:
        held.acquire_gpu()  # endpoint one now has a higher occupancy fraction
        llm = ScheduledLLM(path, client_factory=Client)
        result = generate(llm, 'test')
        assert result['content'] == 'http://two/v1'
        assert result['calls'][0]['scheduler']['endpoint'] == result['content']
        assert result['calls'][0]['scheduler']['request_seconds'] >= 0
        assert llm.count_tokens('text') == 2
        llm.close()


def test_host_partition_and_cli_defaults(daemon, capsys):
    from experiments.traceaad_v10_21.launch_host import main
    from experiments.traceaad_v10_21.run import main as run
    path, _ = daemon
    plans = []
    for ids in (['1', '2'], ['3']):
        main(['--batch', 'partition', '--scheduler-socket', path, '--repeat-ids', *ids])
        payload = json.loads(capsys.readouterr().out)
        plans.append(payload['plan'])
        assert all('--scheduler-socket' in row['command'] and '--eval-workers=2' in row['command'] for row in payload['plan'])
    assert [len(plan) for plan in plans] == [12, 6]
    assert not {r['run_name'] for r in plans[0]} & {r['run_name'] for r in plans[1]}
    assert all(row['seed'] == row['repeat'] - 1 for plan in plans for row in plan)
    run(['--task', 'tsp_construct', '--dry-run', '--scheduler-socket', path])
    payload = json.loads(capsys.readouterr().out)
    assert payload['config']['eval_workers'] == 2
    assert payload['config']['scheduler_socket'] == path
    assert payload['resource_scheduler']['gpu']['capacity'] == 3


def test_cpu_configuration_uses_host_affinity():
    from experiments.infra.scheduler import cpu_ids
    allowed = sorted(os.sched_getaffinity(0))
    assert cpu_ids(count=1) == allowed[:1]
    assert cpu_ids() == allowed
    with pytest.raises(ValueError):
        cpu_ids(count=len(allowed) + 1)
    with pytest.raises(ValueError):
        cpu_ids(spec=str(max(allowed) + 1))


def test_scheduler_disconnect_stops_running_instances(daemon, tmp_path):
    path, server = daemon
    marker = tmp_path / 'worker'
    task = Instances()
    task.n_instance, task._datasets = 1, [0]
    code = f'''import os, time
def score(x):
    with open({str(marker)!r}, 'w') as f:
        f.write(str(os.getpid()))
    time.sleep(30)
    return 1
'''
    with ThreadPoolExecutor(1) as executor:
        future = executor.submit(InstanceProgramEvaluator(task, scheduler_socket=path).evaluate,
                                  code, 'x', timeout_seconds=60)
        eventually(lambda: marker.exists() and marker.read_text())
        pid = int(marker.read_text())
        server.server_close()
        with pytest.raises(SchedulerError):
            future.result(timeout=10)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    eventually(lambda: server.pools['cpu'].status()['used'] == 0)


def test_real_http_clients_share_slots_and_search_uses_both_schedulers(daemon, tmp_path):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from experiments.infra.scheduled_llm import ScheduledLLM
    from traceaad.v10_21 import Config, TraceAADV1021
    path, server = daemon
    class Handler(BaseHTTPRequestHandler):
        active, peak, requests = {}, {}, []
        lock = threading.Lock()
        def log_message(self, *args):
            pass
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            endpoint = self.path.split('/')[1]
            if self.path.endswith('/tokenize'):
                result = {'count': 30}
            else:
                with self.lock:
                    self.active[endpoint] = self.active.get(endpoint, 0) + 1
                    self.peak[endpoint] = max(self.peak.get(endpoint, 0), self.active[endpoint])
                    self.requests.append(payload)
                time.sleep(.1)
                result = {'id': 'offline', 'object': 'chat.completion', 'created': 1, 'model': 'test',
                    'choices': [{'index': 0, 'finish_reason': 'stop', 'message': {'role': 'assistant',
                        'content': 'Design: constant\n```python\ndef score(x):\n    return 2\n```'}}],
                    'usage': {'prompt_tokens': 30, 'completion_tokens': 20, 'total_tokens': 50}}
                with self.lock:
                    self.active[endpoint] -= 1
            body = json.dumps(result).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    http = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=http.serve_forever)
    thread.start()
    for i, endpoint in enumerate(server.endpoints):
        endpoint.update(base_url=f'http://127.0.0.1:{http.server_port}/{i}/v1', no_proxy='127.0.0.1,localhost')
    def request():
        client = ScheduledLLM(path)
        try:
            return client.draw_sample_with_details('hello')
        finally:
            client.close()
    try:
        with ThreadPoolExecutor(6) as executor:
            results = list(executor.map(lambda _: request(), range(6)))
        assert all(result['content'] for result in results)
        assert 1 <= Handler.peak['0'] <= 2 and Handler.peak['1'] == 1
        assert all(r['model'] == 'test' and r['temperature'] == .7 for r in Handler.requests)
        client = ScheduledLLM(path)
        task = TinyEvaluation()
        task.fork_proc = False
        try:
            method = TraceAADV1021(evaluation=task, llm=client, run_dir=tmp_path,
                config=Config(budget=1, roots=1, scheduler_socket=path, eval_workers=2))
            assert method.run()['status'] == 'finished'
            assert method.facts.evaluations[0]['scheduler']['socket'] == path
            assert method.programs[1]['fitness'] == -2
        finally:
            client.close()
    finally:
        http.shutdown()
        http.server_close()
        thread.join(3)
