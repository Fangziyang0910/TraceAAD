# V10.13 实验启动与监控

当前工作区是 **v10.13-r2**，设计见 [r2 机制定义](../../docs/01-主线版本/TraceAAD-V10.13-r2-机制设计.md)。已运行的 `20260923_v1013` 是旧版，继续使用其原冻结运行时；不要向它注入新代码或用新入口恢复它。下列命令仅为新批次使用示例，本轮实现没有启动新实验。

V10.13 正式批次固定为五个任务 × 四个重复，共 20 路。启动器使用三条独立请求池：`server1` 5 路、`server3`（222.201.145.6:8000）8 路、`server3b`（222.201.145.6:8001）7 路。总数 20 路，均低于各端点容量 6、9、9。

先用新的批次名冻结当前代码（已有目录会拒绝覆盖）；启动器随后校验哈希、语法和 spawn：

```bash
./.venv/bin/python -m experiments.traceaad_v10_13.freeze \
  --batch 20260924_v1013r2 --session-prefix v1013r2
```

再启动并持续监控：

```bash
./.venv/bin/python -m experiments.traceaad_v10_13.launch \
  --batch 20260924_v1013r2 --session-prefix v1013r2 \
  --runtime experiments/traceaad_v10_13/results/runtime_20260924_v1013r2 \
  --watch --interval 30
```

启动器在每次启动前检查端点 `/v1/models`、可用槽位、tmux 会话和结果目录；manifest 持久化每路的 backend、尝试次数、状态和冻结源身份。它不会覆盖已有结果，也不会自动停止其他版本。

旧批次的监控继续服务原批次。新批次如需额外面板，可选一个空闲端口直接运行（以下端口仅为示例）：

```bash
./.venv/bin/python -m experiments.traceaad_v10_13.monitor \
  --host 127.0.0.1 --port 8766 --version 20260924_v1013r2
```

监控面板展示 20 路的评价预算、运行/排队/完成状态、Best 曲线、父代和前沿改善、算子计数、backend、提示 token 和错误计数。endpoint 标签只用于资源审计，不参与方法质量判断。

r2 每路默认 E1000，计入初始化、失败、修复和重复代码的真实评价。可选上下文读取增加一轮 LLM 调用，不占 evaluator 预算，调用和已有 token usage 单独留存。同 E1000 比较不意味着 LLM 成本相同。

r2 的 `evaluations.jsonl` 保存评价回执，`tree_state.json` 保存当前阶段。恢复时若已有评价预约却无完整回执，状态为 `uncertain_evaluation`，启动器标为 blocked，禁止自动重评；保留全部文件并核对是否实际发生评价。不同机制指纹的 checkpoint 会在修改 journal 前被拒绝。截断的 journal 尾部另存 `.torn-*`，不作为成功评价证据。
