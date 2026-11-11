# TraceAAD 统一实验入口

保留Base、V21.3、V22.2、V23.2与V24.1。使用 `--method` 选择方法；默认 `base`，未指定 `--experiment` 时使用所选方法的结果目录名。点号与下划线写法均支持，例如 `v22.2` 与 `v22_2`。

| 方法 | 搜索实现 |
| --- | --- |
| `base` | Base 基础演化对照。 |
| `v21.3` | Base加机制2：Refine读形成与修改记录，Crossover读参考形成与接入，Explore读方向及后续发展；`--reference-code`额外给出参考完整源码。 |
| `v22.2` | V21.3加机制1。 |
| `v23.2` | Base加机制1。 |
| `v24.1` | V21.3加机制3：条件经验、展示与跨分支修订。 |

默认后端为 `scheduled`，Qwen3.8-27B 经 server4 公共调度器分配。采样为温度1.0、top_p=0.8、top_k=20、min_p=0、presence_penalty=1.5，关闭思考，默认输出上限16384。完整设置见[scheduling.py](scheduling.py)与运行元数据。

V22.2与V23.2默认开启机制1的轨迹嵌入交叉参考；`--no-trajectory-reference-embedding`在同池随机选参考，作为参考部分消融。Base始终在合格池随机选参考。嵌入服务复用 `SHINKA_EMBEDDING_*` 配置，向量和用量保存在native目录。设计见[机制1](../../docs/演化轨迹叙事/机制设计/机制1-轨迹节点价值评估与预算分配.md)。

默认初始化5个有效程序，每路1000次评价。初始化、失败与修复重评按实际评价计费，辅助模型调用与token单列。训练16例共享160秒、内存2 GiB；训练最好程序固定后，再执行独立测试。条件见[实验协议](../PROTOCOL.md)。

## 检查单路配置

在有公共调度器的主机上执行。`--dry-run`读取配置和调度器状态，不发送模型生成请求：

```bash
.venv/bin/python -m experiments.traceaad.run \
  --method base --task tsp_construct --run-name base_dry --batch base_dry \
  --scheduler-socket /tmp/traceaad-1003/scheduler.sock \
  --env-file .env --dry-run
```

## 批量计划与恢复

批量计划由 `experiments.traceaad.launch_host` 生成。未加 `--launch` 时只输出计划；使用新批次名启动。恢复用单路原命令加 `--resume`：读回已完成的尝试，丢弃未完成的尝试后继续。单路入口拒绝覆盖已有目录。服务器位置见[操作说明](../infra/AGENT_OPERATIONS.md)。

```bash
source .env.server4
.venv/bin/python -m experiments.traceaad.launch_host \
  --method v21.3 --experiment traceaad_v21_3 --batch BATCH_NAME \
  --repeats 3 --budget 1000 --scheduler-socket /tmp/traceaad-1003/scheduler.sock \
  --env-file .env.server4 --launch
```

V22.2与V23.2需要轨迹参考嵌入服务，不需要反思服务。V24.1不需要嵌入服务；反思独立使用4096 tokens输出，输入按服务端聊天tokenizer计数，默认上限44544 tokens。代码见[V21.3实现](../../traceaad_v21_3)、[V22.2实现](../../traceaad_v22_2)、[V23.2实现](../../traceaad_v23_2)与[V24.1实现](../../traceaad_v24_1)。

`validate_v24_1`复查V24.1的材料组织与反思：

```bash
# 仅调tokenizer，不请求生成、不评价程序
.venv/bin/python -m experiments.traceaad.validate_v24_1 \
  --pack-root RESULT_DIR \
  --output experiments_result/traceaad_v24_1_validation/NEW_PACKING_RUN

# 明确指定 --live 才发送模型反思；同一窗口从空经验开始，不回写源档案
.venv/bin/python -m experiments.traceaad.validate_v24_1 \
  --live --case RUN_DIR:ATTEMPT_ID --repeats 2 \
  --output experiments_result/traceaad_v24_1_validation/NEW_REFLECTION_RUN
```

## 四任务阶段与跨版本补位队列

四任务阶段使用TSP、CVRP、FSSP和OP，每版本三种子共12路。批量命令显式传入任务列表；`--enqueue`只写入等待清单，不能与`--launch`同时使用：

```bash
.venv/bin/python -m experiments.traceaad.launch_host \
  --method v21.3 --experiment traceaad_v21_3 --batch BATCH_NAME \
  --tasks tsp_construct cvrp_aco fssp_gls op_aco \
  --repeats 3 --budget 1000 --selection-temperature 1 \
  --scheduler-socket /tmp/traceaad-1003/scheduler.sock --env-file .env.server4 --enqueue
```

`python -m experiments.traceaad.queue`读取`--manifests`中的等待清单，用`--observe`纳入已运行的批次，`--capacity 36`限制整个活动队列的路数。第一个等待版本优先启动，后续版本交替补位。运行进程包含搜索和主测试；训练结束不提前释放位置。错误退出保留记录并标为blocked，不自动重跑，也不阻塞其他等待任务。

队列从进程、tmux和持久清单恢复状态，锁文件阻止重复控制器；重启控制器不重复启动已有运行。队列放在server4的独立tmux会话，SSH断开后继续运行。监控区分queued、deferred和异常退出，等待期间也检查控制会话是否存在。
