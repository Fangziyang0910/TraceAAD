"""Route resumable runs to any equivalent service without changing search state."""

import json


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.routing.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def prepare_resume(item, profiles):
    """Only change service routing metadata; candidate, RNG and receipts stay intact."""
    state_path = item.run_dir / 'tree_state.json'
    if not state_path.exists():
        return
    profile = profiles[item.backend]
    state = json.loads(state_path.read_text())
    llm = state['mechanism']['llm']
    route = {'base_url': profile.base_url, 'model': profile.model}
    if any(llm.get(k) != v for k, v in route.items()):
        backup = item.run_dir / 'checkpoint_before_service_routing.json'
        if not backup.exists():
            write_json(backup, state)
        llm.update(route)
        write_json(state_path, state)
    config_path = item.run_dir / 'run_config.json'
    if config_path.exists():
        config = json.loads(config_path.read_text())
        config['backend'] = item.backend
        config['llm'].update(route)
        if 'no_proxy' in config['llm']:
            config['llm']['no_proxy'] = profile.no_proxy
        write_json(config_path, config)


def allocate_anywhere(allocate, plan, available, backend_pool):
    # A resumed row may still name its previous backend. Clear only the
    # scheduling copy; return the original rows for its normal persistence.
    by_name = {row['run_name']: row for row in plan}
    candidates = [{**row, 'backend': None} if row['status'] == 'queued' else row for row in plan]
    return [(by_name[row['run_name']], backend)
            for row, backend in allocate(candidates, available, backend_pool)]
