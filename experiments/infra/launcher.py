"""Read a model endpoint once and execute a concrete tmux batch plan."""

import json
from datetime import datetime
import shlex
import subprocess
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener

from .base import BACKENDS, REPO_ROOT
from .env import resolve_llm_api_key
from traceaad.common.storage import write_json


def served_models(backends, *, required=False, min_context=None):
    found = {}
    opener = build_opener(ProxyHandler({}))
    for name in sorted(set(backends)):
        profile = BACKENDS[name]
        key = resolve_llm_api_key(base_url=profile.base_url)
        headers = {'Authorization': 'Bearer '+key} if key and key != 'EMPTY' else {}
        record = {'endpoint': profile.base_url}
        try:
            with opener.open(Request(profile.base_url+'/models', headers=headers), timeout=20) as response:
                models = json.load(response)['data']
            model = next((m for m in models if m['id'] == profile.model), None)
            if model is None:
                raise ValueError(f'configured model {profile.model} is not served')
            record.update(model)
            if model.get('owned_by') == 'llamacpp':
                with opener.open(Request(profile.base_url.removesuffix('/v1')+'/props', headers=headers), timeout=20) as response:
                    props = json.load(response)
                record.update(model_path=props.get('model_path'), build=props.get('build_info'), slots=props.get('total_slots'))
                record.setdefault('max_model_len', props.get('default_generation_settings', {}).get('n_ctx'))
            if min_context is not None and (record.get('max_model_len') or 0) < min_context:
                raise ValueError(f'context shorter than {min_context} or not reported')
        except Exception as exc:
            if required:
                raise RuntimeError(f'{name} endpoint unavailable: {exc}') from exc
            record['error'] = f'{type(exc).__name__}: {exc}'
        found[name] = record
    return found


def check_backends(backends):
    return served_models(backends, required=True)


def is_session_alive(session):
    return subprocess.run(['tmux', 'has-session', '-t', '='+session],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def launch_command(session, command):
    if is_session_alive(session):
        raise RuntimeError(f'tmux session already exists: {session}')
    subprocess.run(['tmux', 'new-session', '-d', '-s', session, '-c', str(REPO_ROOT),
                    shlex.join(command)], cwd=REPO_ROOT, check=True)


def launch_plan(path, manifest, *, min_context=None):
    path = Path(path)
    plan = manifest['plan']
    if path.exists():
        raise ValueError(f'batch manifest already exists: {path}')
    for item in plan:
        if Path(item['run_dir']).exists() or is_session_alive(item['session']):
            raise ValueError(f'existing run or session: {item["run_name"]}')
    models = served_models((item['backend'] for item in plan), required=True, min_context=min_context)
    for name, record in models.items():
        record['assigned_runs'] = sum(item['backend'] == name for item in plan)
    manifest.update(served_models=models, status='launching')
    write_json(path, manifest)
    for item in plan:
        command = item['command']
        if item.get('startup_log'):
            Path(item['startup_log']).parent.mkdir(parents=True, exist_ok=True)
            shell = 'exec '+shlex.join(command)+' >> '+shlex.quote(item['startup_log'])+' 2>&1'
            command = ['bash', '-c', shell]
        launch_command(item['session'], command)
        item.update(status='started_unverified', started_at=datetime.now().astimezone().isoformat())
        write_json(path, manifest)
        print(item['session'], item['backend'], flush=True)
    manifest['status'] = 'started_pending_verification'
    write_json(path, manifest)
    print(path)
