# TraceAAD 统一实验入口

当前保留Base、V21.1、V22.2、V23.2与V24.1。使用 `--method` 选择方法；默认 `base`，未指定 `--experiment` 时使用所选方法的结果目录名。点号与下划线写法均支持，例如 `v22.2` 与 `v22_2`。

| 方法 | 搜索实现 |
| --- | --- |
| `base` | 当前 Base 基础演化对照。 |
| `v21.1` | 当前Base加机制2，选择与参考规则完全继承Base。 |
| `v21.2` | 当前Base加新版机制2（轨迹引导的设计推理）；`--reference-code`为参考另给完整源码的对照。 |
| `v22.2` | V21.1加完整新版机制1。 |
| `v23.2` | 当前 Base 加机制一。 |
| `v24.1` | V21.1加当前机制3：条件经验、展示与跨分支修订。 |

默认后端为 `scheduled`，Qwen3.8-27B 经 server4 公共调度器分配。采样为温度1.0、top_p=0.8、top_k=20、min_p=0、presence_penalty=1.5，关闭思考，默认输出上限16384。完整设置见[scheduling.py](scheduling.py)与运行元数据。

V22.2/V23.2的单路与批量入口默认开启机制1的“质量过滤＋轨迹嵌入差异”交叉参考1.0；显式使用`--no-trajectory-reference-embedding`时在同池随机选参考，作为参考部分消融。`--trajectory-reference-embedding`可显式声明完整配置。Base本身始终在合格池随机选参考。嵌入服务复用现有 `SHINKA_EMBEDDING_*` 配置，向量和用量保存在native目录，轨迹收益估计本身不变。统一参考与回退后的选择器规则为v3，旧运行仍用其冻结源码恢复。具体输入、缓存、恢复与冒烟证据见[实现说明](../../docs/机制设计/实现说明.md#可选轨迹嵌入参考10)。

默认初始化5个有效程序，每路1000次评价。初始化、失败与修复重评按实际评价计费，辅助模型调用与token单列。训练16例共享160秒、内存2 GiB；训练最好程序固定后，再执行独立测试。历史批次仍按各自冻结源码与配置解释。

## 检查单路配置

在有公共调度器的主机上执行。`--dry-run`读取配置和调度器状态，不发送模型生成请求：

```bash
.venv/bin/python -m experiments.traceaad.run \
  --method base --task tsp_construct --run-name base_dry --batch base_dry \
  --scheduler-socket /tmp/traceaad-1003/scheduler.sock \
  --env-file .env --dry-run
```

## 批量计划与恢复

六任务各三种子的计划由 `experiments.traceaad.launch_host` 生成。未加 `--launch` 时只输出计划；使用新批次名启动，恢复使用同一方法、目录与参数并增加 `--resume`。单路入口会拒绝覆盖已有目录。条件见[实验协议](../PROTOCOL.md)，服务器位置见[操作说明](../infra/AGENT_OPERATIONS.md)。

V21.1不需要嵌入服务，直接继承当前Base的选择和同池随机参考。它与同条件的新Base形成机制2对照。旧V21批次仍按原冻结源码解释。入口示例：

```bash
source .env.server4
.venv/bin/python -m experiments.traceaad.launch_host \
  --method v21.1 --experiment traceaad_v21_1 --batch BATCH_NAME \
  --repeats 3 --budget 1000 --scheduler-socket /tmp/traceaad-1003/scheduler.sock \
  --env-file .env.server4 --launch
```

快照路径与核对见[操作说明](../infra/AGENT_OPERATIONS.md)。历史对照使用当时的冻结源码与配置。

V22.2 为 V21.1 加机制1；V23.2 在当前 Base 生成流程上加入同一机制1。两者不需要反思服务，默认需要轨迹参考嵌入服务，支持点号和下划线方法名。入口示例只展示配置：

```bash
.venv/bin/python -m experiments.traceaad.run \
  --method v22.2 --experiment traceaad_v22_2 --task tsp_construct \
  --run-name v22_2_dry --batch v22_2_dry \
  --scheduler-socket /tmp/traceaad-1003/scheduler.sock --dry-run
```

V23.2 使用 `--method v23.2 --experiment traceaad_v23_2`。规格见 [V22.2](../../docs/机制设计/V22.2.md) 与 [V23.2](../../docs/机制设计/V23.2.md)。旧入口`v21`、`v22`、`v22.1`、`v22_1`、`v23`、`v23.1`、`v23_1`和`v24`已移除；旧结果继续读取，恢复使用原批次冻结源码。当前 V23.2 已跟随 Base 去掉演化生成中的轨迹上下文，旧批次仍按旧条件解释。新 V23.2 运行保存生成协议标识，不能直接续跑旧日志。

去掉 `--launch` 只输出十八路计划。恢复用单路原命令加 `--resume`：读回已完成的尝试，丢弃未完成的尝试后继续。辅助请求与tokens单列，不占评价预算。

V24.1在V21.1上使用条件经验、完整计算背景和可恢复的待审阅队列，入口为 `--method v24.1` 或 `--method v24_1`。它不需要嵌入服务，沿用原模型调度、评价与最终测试流程；不要用 V24 的运行目录恢复 V24.1。单路配置检查示例：

```bash
.venv/bin/python -m experiments.traceaad.run \
  --method v24.1 --experiment traceaad_v24_1 --task tsp_construct \
  --run-name v24_1_dry --batch v24_1_dry \
  --scheduler-socket /tmp/traceaad-1003/scheduler.sock --dry-run
```

批量计划使用 `launch_host --method v24.1 --experiment traceaad_v24_1`，其余参数同上；只有明确启动新批次时添加 `--launch`。此前d7c238a的小规模真实模型验证对应批量材料与单事件来源。当前V24.1采用`cross_branch_experience`，每周期围绕一个计算问题，回读候选经验的完整案例；经验持续保存，每次生成最多展示6条。当前验证为离线回归与脚本化跨分支闭环。历史`focused_relations`及批量反思运行使用各自冻结源码恢复。规格与状态见[V24.1](../../docs/机制设计/V24.1.md)。

以下脚本可复查当前材料组织与冻结窗口反思；覆盖全部事件不再是聚焦反思的目标。重现旧报告需使用其冻结源码。本轮未执行这些命令：

```bash
# 仅调tokenizer，不请求生成、不评价程序
.venv/bin/python -m experiments.traceaad.validate_v24_1 \
  --pack-root experiments_result/traceaad_v21c \
  --output experiments_result/traceaad_v24_1_validation/NEW_PACKING_RUN

# 明确指定 --live 才发送模型反思；同一窗口从空经验开始，不回写源档案
.venv/bin/python -m experiments.traceaad.validate_v24_1 \
  --live --case RUN_DIR:ATTEMPT_ID --repeats 2 \
  --output experiments_result/traceaad_v24_1_validation/NEW_REFLECTION_RUN
```

反思独立使用4096 tokens输出；输入按服务端聊天tokenizer计数，默认上限44544 tokens。原始提示、回复、计数缓存与来源哈希留在结果目录，提交到仓库的摘要和解释见[修正验证记录](../../docs/记录/2026-11-09-V24.1修正与小规模实验.md)。

当前组合与对照表见[机制索引](../../docs/机制设计/README.md#当前机制组合)。运行配置记录机制编号、生成协议与实际轨迹参考开关；批量计划显式转发开关，避免默认值改变消融条件。此轮未启动统一重跑；旧V24的比较脚本与结果通过原冻结源码复查，当前`validate_v24_1`只检验最新机制3。

## 四任务阶段与跨版本补位队列

当前阶段暂缓图着色与JSSP，使用TSP、CVRP、FSSP和OP，每版本三种子共12路。批量命令显式传入任务列表；`--enqueue`只写入等待清单，不能与`--launch`同时使用：

```bash
.venv/bin/python -m experiments.traceaad.launch_host \
  --method v21.1 --experiment traceaad_v21_1 --batch BATCH_NAME \
  --tasks tsp_construct cvrp_aco fssp_gls op_aco \
  --repeats 3 --budget 1000 --selection-temperature 1 \
  --scheduler-socket /tmp/traceaad-1003/scheduler.sock --env-file .env.server4 --enqueue
```

`python -m experiments.traceaad.queue`读取`--manifests`中的等待清单，用`--observe`纳入已运行Base与V23.2，`--capacity 36`限制整个活动队列的路数。第一个等待版本优先启动，后续版本交替补位。运行进程包含搜索和主测试；训练结束不提前释放位置。错误退出保留记录并标为blocked，不自动重跑，也不阻塞其他等待任务。

队列从进程、tmux和持久清单恢复状态，锁文件阻止重复控制器；重启控制器不重复启动已有运行。相同状态文件必须沿用原队列设置。队列放在server4的独立tmux会话，SSH断开后继续运行。监控区分queued、deferred和异常退出，等待期间也检查控制会话是否存在。历史六任务清单与结果保留，暂停的运行恢复时仍用原冻结源码。
