# V11.1 手工启动

使用 `manual_launch.py` 读取逐路 LLM 分配表，并按 GPU 端点容量创建 tmux 会话。每批五个任务、每个任务五个重复。容量表和请求方式见 [`experiments/infra/GPU_RESOURCES.md`](../infra/GPU_RESOURCES.md)。

实验设置中所有端点均服务同一个 **Qwen3.8-27B** 模型：本地配置名 `Qwen3.8-27B` 与远程配置名 `qwen3.8-27b-awq` 视为同一模型。端点只表示服务地址和容量池，不是模型或方法对照；质量分析不得把 `local` 与远程端点当作不同模型。

从仓库根目录执行，将 `my_new_batch` 替换为新的批次名。先生成独立分配表：

```bash
./.venv/bin/python -m experiments.traceaad_v11_1.manual_launch \
  --batch my_new_batch --assignments /tmp/my_new_batch.json --write-template
```

编辑该 JSON 中每一行的 `backend`（`server1`、`server3`、`server3b` 或 `local`），确认每个端点不超过容量后启动：

```bash
./.venv/bin/python -m experiments.traceaad_v11_1.manual_launch \
  --batch my_new_batch --assignments /tmp/my_new_batch.json \
  --session-prefix my_new_batch
```

启动器会在启动前校验每个端点的槽位上限；`server3` 和 `server3b` 分别有 9 个槽位。已有批次、结果目录或同名 tmux 会话会阻止重复启动。仓库内 `manual_assignments.json` 记录当前批次的分配，不要直接复用于新的批次。

监控入口为 `python -m core.training_monitor --port 8765`，默认只读取 V11.1 的 manifest、结果与 tmux 状态；旧的 `experiments.traceaad_v11_1.monitor` 仍保留为兼容入口。面板优先展示评价预算、Best 改进轨迹、父代/全局前沿改善和候选血统，耗时与 token 只作诊断。统一调度器及 V11.1 旧自动 launcher 已移除。
