# Base、V20、V21、V22、V23 与 V24 的同条件入口

默认后端为 `scheduled`。Qwen3.8-27B经server4公共调度器分配到四个端点。`--method base` 调用 `traceaad_base.TraceAAD`；`--method v20` 调用完整V20；`--method v21` 调用只含机制2的V21；`--method v22` 调用V21加机制1的V22；`--method v23`调用Base加机制1的V23。

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

V21不需要嵌入服务。它的对照是同一份代码的Base新批次，用 `--method base` 启动。批次名和结果入口在启动时指定：

```bash
source .env.server4
.venv/bin/python -m experiments.traceaad_v20.launch_host \
  --method v21 --experiment traceaad_v21 --batch BATCH_NAME \
  --repeats 3 --budget 1000 --scheduler-socket /tmp/traceaad-1003/scheduler.sock \
  --env-file .env.server4 --launch
```

快照路径与核对见[操作说明](../infra/AGENT_OPERATIONS.md)。对照是同一快照中的Base新批次。

V22为Base加机制1和机制2，不需要反思或嵌入服务。单路配置预览：

```bash
.venv/bin/python -m experiments.traceaad_v20.run \
  --method v22 --experiment traceaad_v22 --task tsp_construct \
  --run-name v22_dry --batch v22_dry \
  --scheduler-socket /tmp/traceaad-1003/scheduler.sock --dry-run
```

批量入口同样支持`--method v22 --experiment traceaad_v22`。机制与验证范围见[V22](../../docs/机制设计/V22.md)。

V23复用Base的完整生成流程和`analysis → design → code`输出，与V22共用机制1的选择和学习实现，状态独立。正式对照使用：

```bash
.venv/bin/python -m experiments.traceaad_v20.launch_host \
  --method v23 --experiment traceaad_v23 --batch BATCH_NAME \
  --repeats 3 --budget 1000 --initial-size 5 --seconds-per-instance 10 \
  --scheduler-socket /tmp/traceaad-1003/scheduler.sock \
  --env-file .env.server4 --launch
```

机制与验证范围见[V23](../../docs/机制设计/V23.md)。

去掉 `--launch` 只输出十八路计划。恢复用单路原命令加 `--resume`：读回已完成的尝试，丢弃未完成的尝试后继续。辅助请求与tokens单列，不占评价预算。

V24为V21加机制3，不需要嵌入服务。反思请求经同一调度器发出，使用独立schema，不占评价预算。正式对照为V21批次`20261105c_v21_co6`：

```bash
.venv/bin/python -m experiments.traceaad_v20.launch_host \
  --method v24 --experiment traceaad_v24 --batch BATCH_NAME \
  --repeats 3 --budget 1000 --initial-size 5 --seconds-per-instance 10 \
  --scheduler-socket /tmp/traceaad-1003/scheduler.sock \
  --env-file .env.server4 --launch
```

机制与记录见[V24](../../docs/机制设计/V24.md)。
