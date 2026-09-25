# V10.13 实验

机制见[当前 V10.13 定义](../../docs/01-主线版本/TraceAAD-V10.13-机制设计.md)。默认计划为五个任务、每个任务四次重复。Python 环境和模块执行由 `uv` 管理。

启动新批次：

```bash
uv run python -m experiments.traceaad_v10_13.launch \
  --batch 20260924_v1013r3 --session-prefix v1013r3 \
  --watch --interval 30
```

查看结果：

```bash
uv run python -m experiments.traceaad_v10_13.monitor \
  --host 127.0.0.1 --port 8766 --version 20260924_v1013r3
```

每路默认 E1000。一次评价器调用计一次预算，解析失败不计；节点、轮次、模型调用、当前状态和最终摘要保存在对应结果目录。`20260923_v1013`、`20260924_v1013r2` 和 `20260924_v1013r3` 是旧实现产生的历史批次，说明见[设计演进](../../docs/01-主线版本/V10.13设计演进.md)。
