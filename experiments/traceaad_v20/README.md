# V20 与 Base 的同条件入口

默认后端为 `scheduled`。Qwen3.8-27B经server4公共调度器分配到四个端点。`--method base` 调用 `traceaad_base.TraceAAD`；`--method v20` 调用完整V20。

采样对齐 `20261031_base_schema_co6` 的保存配置：温度1.0，top_p=0.8，top_k=20，min_p=0，presence_penalty=1.5，frequency_penalty=0，repetition_penalty和repeat_penalty均为1，关闭思考，输出上限16384，JSON schema输出。

该批次已完成Qwen十八路，每路1000次评价及独立测试。对应关系见[源码与配置核对](../../experiments_result/analysis/v20_qwen_smoke_20261103/README.md)。

初始化5个有效程序。失败与修复重评各计一次评价预算。训练16例共享160秒，内存2 GiB。训练最好固定后执行独立测试。完整体四个开关默认全开，说明见[机制文档](../../docs/机制设计/V20.md)。

在server4仓库根目录执行。单路预览不调用模型：

```bash
source .env.server4
.venv/bin/python -m experiments.traceaad_v20.run \
  --task tsp_construct --run-name v20_dry --batch v20_dry \
  --scheduler-socket /tmp/traceaad-1003/scheduler.sock \
  --env-file .env --dry-run
```

TSP完整体冒烟使用30次评价，跳过独立测试：

```bash
source .env.server4
TRACEAAD_EMBEDDING_BASE_URL=http://127.0.0.1:18002/v1 \
TRACEAAD_EMBEDDING_MODEL=qwen3-embedding-4b-fp8 \
TRACEAAD_EMBEDDING_API_KEY_ENV=SERVER3_API_KEY \
.venv/bin/python -u -m experiments.traceaad_v20.run \
  --task tsp_construct --run-name 20261103_v20_complete_qwen_smoke_tsp_rep1 \
  --batch 20261103_v20_complete_qwen_smoke --budget 30 \
  --scheduler-socket /tmp/traceaad-1003/scheduler.sock \
  --env-file .env --skip-heldout
```

正式批次在冻结源码快照中启动。每组六任务×种子0/1/2，每路1000次评价。完整体与无反思对照：

```bash
source .env.server4
.venv/bin/python -m experiments.traceaad_v20.launch_host \
  --method v20 --experiment traceaad_v20_complete --batch 20261103b_v20_complete_qwen_co6 \
  --repeats 3 --budget 1000 --scheduler-socket /tmp/traceaad-1003/scheduler.sock \
  --env-file .env.server4 --launch

.venv/bin/python -m experiments.traceaad_v20.launch_host \
  --method v20 --experiment traceaad_v20_noreflect --batch 20261103b_v20_noreflect_qwen_co6 \
  --repeats 3 --budget 1000 --scheduler-socket /tmp/traceaad-1003/scheduler.sock \
  --env-file .env.server4 --no-experience-learning --launch
```

快照路径与核对见[操作说明](../infra/AGENT_OPERATIONS.md)。对照Base直接使用已完成的`20261031_base_schema_co6`。

去掉 `--launch` 只输出十八路计划。恢复用单路原命令加 `--resume`：读回已完成的尝试，丢弃未完成的尝试后继续。辅助请求与tokens单列，不占评价预算。
