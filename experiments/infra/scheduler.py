"""Run independent CPU and GPU schedulers on this host; inspect with --status."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal

from core.scheduling import SchedulerServer, scheduler_status
from .base import BACKENDS


def default_socket():
    return f'/tmp/traceaad-{os.getuid()}/scheduler.sock'


def cpu_ids(spec=None, count=None):
    allowed = sorted(os.sched_getaffinity(0))
    if spec:
        chosen = set()
        for part in spec.split(','):
            ends = [int(value) for value in part.split('-')]
            if len(ends) == 1:
                chosen.add(ends[0])
            elif len(ends) == 2 and ends[0] <= ends[1]:
                chosen.update(range(ends[0], ends[1] + 1))
            else:
                raise ValueError('use CPU IDs such as 0-7,12,14')
        if not chosen or not chosen <= set(allowed):
            raise ValueError('CPU IDs must belong to this process’s allowed affinity')
        return sorted(chosen)
    if count is not None and not 1 <= count <= len(allowed):
        raise ValueError(f'CPU count must be 1..{len(allowed)} on this host')
    return allowed if count is None else allowed[:count]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--socket', default=default_socket())
    cpu = parser.add_mutually_exclusive_group()
    cpu.add_argument('--cpu-count', type=int, help='default: all CPUs allowed on this host')
    cpu.add_argument('--cpus', help='explicit allowed CPU IDs, e.g. 0-15,24-31')
    parser.add_argument('--gpu-backend', action='append', default=[], metavar='NAME=SLOTS',
                        help='registered model endpoint and slots; repeat for multiple endpoints')
    parser.add_argument('--gpu-endpoint', action='append', nargs=3, default=[], metavar=('URL', 'MODEL', 'SLOTS'),
                        help='custom OpenAI-compatible endpoint, model name and slots')
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--print-config', action='store_true', help='resolve this host’s configuration without starting')
    args = parser.parse_args(argv)
    if args.status:
        print(json.dumps(scheduler_status(args.socket), indent=2))
        return
    endpoints = []
    for entry in args.gpu_backend:
        name, slots = entry.rsplit('=', 1)
        if name not in BACKENDS:
            raise ValueError(f'unknown model backend: {name}')
        profile = asdict(BACKENDS[name])
        endpoints.append({**profile, 'name': name, 'slots': int(slots)})
    for i, (url, model, slots) in enumerate(args.gpu_endpoint):
        from urllib.parse import urlparse
        parsed = urlparse(url)
        if parsed.scheme not in {'http', 'https'} or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError('model endpoint must be an HTTP(S) URL without embedded credentials')
        endpoints.append({'name': f'custom-{i}', 'base_url': url.rstrip('/'), 'model': model,
                          'slots': int(slots), 'no_proxy': parsed.hostname + ',localhost,127.0.0.1,::1'})
    cpus = cpu_ids(args.cpus, args.cpu_count)
    if not endpoints or any(e['slots'] < 1 for e in endpoints):
        raise ValueError('configure at least one model endpoint with positive slots')
    if len({e['model'] for e in endpoints}) != 1:
        raise ValueError('routed endpoints must serve the same model and tokenizer')
    if args.print_config:
        print(json.dumps({'socket': str(Path(args.socket).expanduser().resolve()), 'cpus': cpus,
                          'endpoints': endpoints}, indent=2))
        return
    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    with SchedulerServer(args.socket, cpus, endpoints) as server:
        print(json.dumps(server.status()), flush=True)
        try:
            server.serve_forever(poll_interval=0.1)
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
