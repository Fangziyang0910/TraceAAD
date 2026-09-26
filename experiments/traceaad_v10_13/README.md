# V10.13 实验

机制见[当前 V10.13 定义](../../docs/01-主线机制设计/TraceAAD-V10.13-机制设计.md)。默认计划为五个任务、每个任务四次重复。Python 环境和模块执行由 `uv` 管理。

新运行暂用 8 根混合初始化（前 4 根独立，后 4 根参考既有根）。单路运行可用 `--n-roots` 调整根数，或用 `--init-mode independent|sequential|hybrid` 指定方式；根数 8 尚未经根数对照验证。恢复旧的顺序初始化运行时需显式传 `--init-mode sequential`，避免中途切换初始化方式。

启动新批次：

```bash
uv run python -m experiments.traceaad_v10_13.launch \
  --batch 20260925_v1013 --session-prefix v1013 \
  --watch --interval 30
```

查看结果：

```bash
uv run python -m experiments.traceaad_v10_13.monitor \
  --host 127.0.0.1 --port 8766 --batch 20260925_v1013
```

页面显示批次总进度、五个任务的各次重复，以及单路最佳值曲线、近期候选和最佳代码。每路默认 E1000。一次评价器调用计一次预算，解析失败不计；原始记录保存在 `experiments_result/runs/traceaad_v10_13/`。

每路的 `run_config.json` 保存配置，`search.jsonl` 按行保存模型调用、评测开始、候选结果及续跑状态，`logs/run_summary.json` 保存最终摘要和最佳算法。候选结果与续跑状态写在同一条记录中；若进程在评价器调用后、结果落盘前中断，该次调用是否完成无法确认，续跑会要求先人工核实。
