# E2-A：轨迹状态与 Refine/Pivot 响应

本目录执行[E2-A固定协议](../../docs/03-机制验证/04-算子动力学与两步价值/2026-09-07-E2A-轨迹状态与算子响应/README.md)。它复制 E1 cutoff 之后已经完成的 V10.6 suffix，复用训练 probe 做行为画像，再以日志保存的 operator conditional propensity 分析 trajectory state 与 Refine/Pivot 响应。

本次批次快照已执行完成，见[结果与分析](../../docs/03-机制验证/04-算子动力学与两步价值/2026-09-07-E2A-轨迹状态与算子响应/README.md)。行为交互因主任务suffix没有frontier event而判未决，Q_kernel独立suffix audit未复制E1优势。

不调用生成模型、不新增正式 evaluator 账本记录、不修改在线调度器。行为画像是真实 CPU 重放成本，单独记录。

执行顺序：

```bash
.venv/bin/python -m experiments.traceaad_e2_a.prepare
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=12 .venv/bin/python -m experiments.traceaad_e2_a.profile --workers 12
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m experiments.traceaad_e2_a.analyze
.venv/bin/python -m pytest -q tests/experiments/test_traceaad_e2_a.py
```

`prepare` 首次执行后不再移动 cutoff；`profile` 按 task、运行名与节点 ID 复用 E1-A 的成功画像；`analyze` 同时输出 E2-A 行为交互和五任务 Q_kernel suffix audit。
