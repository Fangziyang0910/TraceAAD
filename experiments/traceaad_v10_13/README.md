# V10.13 实验启动与监控

V10.13 正式批次固定为五个任务 × 四个重复，共 20 路。启动器使用三条独立请求池：`server1` 5 路、`server3`（222.201.145.6:8000）8 路、`server3b`（222.201.145.6:8001）7 路。总数 20 路，均低于各端点容量 6、9、9。

先冻结当前代码并执行 spawn 预检：

```bash
./.venv/bin/python -m experiments.traceaad_v10_13.freeze \
  --batch 20260923_v1013
```

再启动并持续监控：

```bash
./.venv/bin/python -m experiments.traceaad_v10_13.launch \
  --batch 20260923_v1013 --runtime experiments/traceaad_v10_13/results/runtime_20260923_v1013 \
  --watch --interval 30
```

启动器在每次启动前检查端点 `/v1/models`、可用槽位、tmux 会话和结果目录；manifest 持久化每路的 backend、尝试次数、状态和冻结源身份。它不会覆盖已有结果，也不会自动停止其他版本。

训练可视化单独启动：

```bash
./experiments/traceaad_v10_13/start_monitor.sh 8765 20260923_v1013
```

也可以直接运行：

```bash
./.venv/bin/python -m experiments.traceaad_v10_13.monitor \
  --host 0.0.0.0 --port 8765 --version 20260923_v1013
```

监控面板展示 20 路的评价预算、运行/排队/完成状态、Best 曲线、父代和前沿改善、算子计数、backend、提示 token 和错误计数。endpoint 标签只用于资源审计，不参与方法质量判断。
