# GPU request capacity

The names passed to `--backend` are routing labels.  A running experiment holds
one LLM request slot for its assigned endpoint.

| Resource group | Backend labels | Endpoint(s) | Maximum parallel request slots |
| --- | --- | --- | ---: |
| `server1` | `server1` | `http://222.201.145.8:8080/v1` | 6 |
| `server3:8000` | `server3` | `http://222.201.145.6:8000/v1` | 9 |
| `server3:8001` | `server3b` | `http://222.201.145.6:8001/v1` | 9 |
| `local` | `local` | `http://127.0.0.1:8001/v1` (RTX 4090 D, llama.cpp) | 3 |

`server3` and `server3b` are two endpoints on the same server, with separate
9-slot pools.  A valid allocation therefore has at most 6 active experiments
on `server1`, 9 on each server3 endpoint, and 3 on `local`.  The 25-run V11.1
batch fits within the 27 total slots only if the assignment table respects each
endpoint limit.

For V11.1, put one backend label on each row of
`experiments/traceaad_v11_1/manual_assignments.json` and start through the
capacity-validating launcher.  Do not start all 25 `run.py` commands by hand.  When
an experiment is moved, retain its `tree_state.json` and append a routing event
to that run's `routing_history.jsonl`; the next process resumes from the last
checkpoint with the new endpoint.

The authoritative constants are `BACKEND_GROUP` and
`BACKEND_GROUP_CAPACITY` in `experiments/infra/base.py`.
