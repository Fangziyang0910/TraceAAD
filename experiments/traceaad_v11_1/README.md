# V11.1 手工启动

使用 `manual_launch.py` 读取逐路 LLM 分配表，直接创建 tmux 会话。每批五个任务、每个任务五个重复。启动前人工检查 `backend`；脚本不管理 CPU/GPU 容量，也不自动重试。

从仓库根目录执行，将 `my_new_batch` 替换为新的批次名。先生成独立分配表：

```bash
./.venv/bin/python -m experiments.traceaad_v11_1.manual_launch \
  --batch my_new_batch --assignments /tmp/my_new_batch.json --write-template
```

编辑该 JSON 中每一行的 `backend`（`server1`、`server3`、`server3b` 或 `local`），确认后启动：

```bash
./.venv/bin/python -m experiments.traceaad_v11_1.manual_launch \
  --batch my_new_batch --assignments /tmp/my_new_batch.json \
  --session-prefix my_new_batch
```

可用 `--delay <秒数>` 手工指定相邻启动间隔。已有批次、结果目录或同名 tmux 会话会阻止重复启动。仓库内 `manual_assignments.json` 记录当前批次的分配，不要直接复用于新的批次。

监控入口为 `python -m experiments.traceaad_v11_1.monitor --port 8765`，只读取 V11.1 的 manifest、结果与 tmux 状态。统一调度器及 V11.1 旧自动 launcher 已移除。
