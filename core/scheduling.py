"""Host-local CPU and model-slot schedulers, shared by independent processes.

Clients keep one Unix connection per evaluation or generation. A disconnected
client releases its allocations. CPU and GPU pools make independent decisions.
The daemon schedules resources; program execution stays in isolated children of
the requesting experiment, and model requests keep the existing HTTP client.
"""

from collections import deque
import json
import os
from pathlib import Path
import socket
import socketserver
import threading
import time
from uuid import uuid4


class SchedulerError(RuntimeError):
    pass


class _Pool:
    def __init__(self, resources, endpoints=()):
        self.resources = tuple(resources)
        self.endpoints = list(endpoints)
        self.jobs = {}
        self.queue = deque()
        self.lock = threading.Lock()
        self.cooldowns = {}
        self.grants = 0

    def _dispatch(self):
        occupied = {r for job in self.jobs.values() for r in job['held']}
        free = set(self.resources) - occupied
        now = time.monotonic()
        while free and self.queue:
            available = [r for r in free if self.cooldowns.get(r.split(':')[0], 0) <= now] if self.endpoints else list(free)
            if not available:
                break
            # Give newly waiting evaluations a turn before extending an already
            # parallel one. Running instances are never preempted.
            key = min(self.queue, key=lambda k: len(self.jobs[k]['held']))
            self.queue.remove(key)
            job = self.jobs[key]
            if self.endpoints:
                def load(resource):
                    endpoint = int(resource.split(':')[0])
                    used = sum(r.startswith(f'{endpoint}:') for r in occupied)
                    return used / self.endpoints[endpoint]['slots'], endpoint, resource
                resource = min(available, key=load)
            else:
                resource = min(available)
            job['held'].add(resource)
            job['offered'].add(resource)
            free.remove(resource)
            occupied.add(resource)
            self.grants += 1
            if len(job['held']) < job['target']:
                self.queue.append(key)

    def poll(self, key, label, target, release, failed=False):
        if type(target) is not int or not 0 <= target <= len(self.resources):
            raise ValueError('target must be between zero and scheduler capacity')
        if self.endpoints and target > 1:
            raise ValueError('one generation may hold only one model slot')
        with self.lock:
            job = self.jobs.setdefault(key, {'label': label, 'held': set(), 'offered': set(), 'target': 0})
            if not set(release) <= job['held']:
                raise ValueError('cannot release another request’s resources')
            for resource in release:
                job['held'].remove(resource)
                job['offered'].discard(resource)
                if self.endpoints and failed:
                    self.cooldowns[resource.split(':')[0]] = time.monotonic() + 5
            job['target'] = target
            # Only grants not yet delivered can be withdrawn when demand falls.
            while len(job['held']) > target and job['offered']:
                resource = job['offered'].pop()
                job['held'].remove(resource)
            if len(job['held']) > target:
                raise ValueError('release running resources before reducing target')
            if len(job['held']) < target:
                if key not in self.queue:
                    self.queue.append(key)
            elif key in self.queue:
                self.queue.remove(key)
            self._dispatch()
            offered = sorted(job['offered'])
            job['offered'].clear()
            return offered

    def remove(self, key):
        with self.lock:
            self.jobs.pop(key, None)
            if key in self.queue:
                self.queue.remove(key)
            self._dispatch()

    def status(self):
        with self.lock:
            self._dispatch()
            jobs = [{'label': j['label'], 'active': len(j['held']),
                     'waiting': max(0, j['target'] - len(j['held']))} for j in self.jobs.values()]
            result = {'capacity': len(self.resources), 'used': sum(j['active'] for j in jobs),
                      'waiting': sum(j['waiting'] for j in jobs), 'grants': self.grants, 'jobs': jobs}
            if self.endpoints:
                held = {r for j in self.jobs.values() for r in j['held']}
                result['endpoints'] = [{'base_url': e['base_url'], 'slots': e['slots'],
                    'used': sum(r.startswith(f'{i}:') for r in held),
                    'cooldown_seconds': max(0, self.cooldowns.get(str(i), 0) - time.monotonic())}
                    for i, e in enumerate(self.endpoints)]
            return result


class _Handler(socketserver.StreamRequestHandler):
    def handle(self):
        key, pool = uuid4().hex, None
        self.server.track(self.request, True)
        try:
            while raw := self.rfile.readline(65537):
                try:
                    if len(raw) > 65536:
                        raise ValueError('scheduler message too large')
                    message = json.loads(raw)
                    if message['op'] == 'status':
                        result = self.server.status()
                    else:
                        kind = message['kind']
                        selected = self.server.pools[kind]
                        if pool is not None and selected is not pool:
                            raise ValueError('a connection belongs to one resource pool')
                        pool = selected
                        result = {'resources': pool.poll(key, str(message.get('label', ''))[:240],
                            message['target'], message.get('release', []), message.get('failed', False))}
                    self.wfile.write((json.dumps({'result': result}) + '\n').encode())
                except (ValueError, KeyError, TypeError) as exc:
                    self.wfile.write((json.dumps({'error': str(exc)}) + '\n').encode())
        except (OSError, ConnectionError):
            pass
        finally:
            if pool is not None:
                pool.remove(key)
            self.server.track(self.request, False)


class SchedulerServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True
    block_on_close = False

    def __init__(self, path, cpus, endpoints):
        self._owns_socket = False
        self.instance_id = uuid4().hex
        self.path = str(Path(path).expanduser().resolve())
        self.cpus, self.endpoints = list(cpus), list(endpoints)
        if not self.cpus or len(set(self.cpus)) != len(self.cpus):
            raise ValueError('CPU IDs must be nonempty and distinct')
        allowed = os.sched_getaffinity(0)
        if not set(self.cpus) <= allowed:
            raise ValueError('configured CPUs are outside this process’s allowed affinity')
        if not endpoints or any(type(e.get('slots')) is not int or e['slots'] < 1
                                or not e.get('base_url') or not e.get('model') for e in endpoints):
            raise ValueError('each endpoint requires base_url, model and positive slots')
        if len({e['model'] for e in endpoints}) != 1:
            raise ValueError('all routed endpoints must serve the same model and tokenizer')
        if len({e['base_url'].rstrip('/') for e in endpoints}) != len(endpoints):
            raise ValueError('combine the slots of duplicate model endpoints')
        self.pools = {'cpu': _Pool(self.cpus),
                      'gpu': _Pool([f'{i}:{slot}' for i, e in enumerate(endpoints) for slot in range(e['slots'])], endpoints)}
        self.connections, self.connection_lock = set(), threading.Lock()
        Path(self.path).parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        # Bind refuses existing sockets; never replace a running scheduler.
        super().__init__(self.path, _Handler)
        os.chmod(self.path, 0o600)

    def server_bind(self):
        super().server_bind()
        self._owns_socket = True

    def track(self, connection, add):
        with self.connection_lock:
            if add:
                self.connections.add(connection)
            else:
                self.connections.discard(connection)

    def status(self):
        return {'host': socket.gethostname(), 'pid': os.getpid(), 'socket': self.path,
                'instance_id': self.instance_id,
                'cpus': self.cpus, 'endpoints': self.endpoints,
                **{kind: pool.status() for kind, pool in self.pools.items()}}

    def server_close(self):
        with self.connection_lock:
            for connection in list(self.connections):
                try:
                    connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
        super().server_close()
        if self._owns_socket:
            Path(self.path).unlink(missing_ok=True)
            self._owns_socket = False


class SchedulerSession:
    """One evaluation/generation connection. Closing it releases all allocations.

    ``poll`` asks for a total concurrent allocation, not additional resources.
    It returns only newly granted CPU IDs or model slots. Callers release a CPU
    after stopping/reaping its instance, and a GPU slot after the HTTP request.
    """

    def __init__(self, path, kind='cpu', label=''):
        self.path, self.kind, self.label = str(path), kind, label
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(5)
        try:
            self.sock.connect(self.path)
        except OSError as exc:
            self.sock.close()
            raise SchedulerError(f'cannot connect to scheduler {self.path}: {exc}') from exc
        self.buffer = b''

    def _request(self, message):
        try:
            self.sock.sendall((json.dumps(message) + '\n').encode())
            while b'\n' not in self.buffer:
                chunk = self.sock.recv(65536)
                if not chunk:
                    raise SchedulerError('scheduler disconnected; stop this work before retrying')
                self.buffer += chunk
            line, self.buffer = self.buffer.split(b'\n', 1)
            response = json.loads(line)
        except (OSError, ValueError) as exc:
            raise SchedulerError(f'scheduler communication failed: {exc}') from exc
        if 'error' in response:
            raise SchedulerError(response['error'])
        return response['result']

    def status(self):
        return self._request({'op': 'status'})

    def poll(self, target, release=(), *, failed=False):
        return self._request({'op': 'poll', 'kind': self.kind, 'label': self.label,
                              'target': target, 'release': list(release), 'failed': failed})['resources']

    def acquire_gpu(self):
        while True:
            resources = self.poll(1)
            if resources:
                return resources[0]
            time.sleep(0.05)

    def close(self):
        self.sock.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def scheduler_status(path):
    with SchedulerSession(path) as session:
        return session.status()
