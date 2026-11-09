# TraceAAD 统一实验入口

当前保留 Base、V21、V22.1、V23.1、V24 与 V24.1。使用 `--method` 选择方法；默认 `base`，未指定 `--experiment` 时使用所选方法的结果目录名。点号与下划线写法均支持，例如 `v22.1` 与 `v22_1`。

| 方法 | 搜索实现 |
| --- | --- |
| `base` | 当前 Base 基础演化对照。 |
| `v21` | 轨迹上下文与算子。 |
| `v22.1` | V21 加机制一。 |
| `v23.1` | 当前 Base 加机制一。 |
| `v24` | V21 加早期机制三。 |
| `v24.1` | V21 加围绕计算问题的条件经验。 |

默认后端为 `scheduled`，Qwen3.8-27B 经 server4 公共调度器分配。采样为温度1.0、top_p=0.8、top_k=20、min_p=0、presence_penalty=1.5，关闭思考，默认输出上限16384。完整设置见[scheduling.py](scheduling.py)与运行元数据。

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

V21不需要嵌入服务。原正式对照为旧 Base 批次 `20261105_base_co6`；当前纯净 Base 是另一生成条件，新对照的实验安排待定。入口示例：

```bash
source .env.server4
.venv/bin/python -m experiments.traceaad.launch_host \
  --method v21 --experiment traceaad_v21 --batch BATCH_NAME \
  --repeats 3 --budget 1000 --scheduler-socket /tmp/traceaad-1003/scheduler.sock \
  --env-file .env.server4 --launch
```

快照路径与核对见[操作说明](../infra/AGENT_OPERATIONS.md)。历史对照使用当时的冻结源码与配置。

V22.1 为 V21 加机制1；V23.1 在当前 Base 生成流程上加入同一机制1。两者不需要反思或嵌入服务，支持点号和下划线方法名。入口示例只展示配置：

```bash
.venv/bin/python -m experiments.traceaad.run \
  --method v22.1 --experiment traceaad_v22_1 --task tsp_construct \
  --run-name v22_1_dry --batch v22_1_dry \
  --scheduler-socket /tmp/traceaad-1003/scheduler.sock --dry-run
```

V23.1 使用 `--method v23.1 --experiment traceaad_v23_1`。规格见 [V22.1](../../docs/机制设计/V22.1.md) 与 [V23.1](../../docs/机制设计/V23.1.md)。旧方法名 `v22`、`v23` 已移除；旧结果继续读取，恢复使用原批次冻结源码。当前 V23.1 已跟随 Base 去掉演化生成中的轨迹上下文，旧批次仍按旧条件解释。新 V23.1 运行保存生成协议标识，不能直接续跑旧日志。

去掉 `--launch` 只输出十八路计划。恢复用单路原命令加 `--resume`：读回已完成的尝试，丢弃未完成的尝试后继续。辅助请求与tokens单列，不占评价预算。

V24为V21加机制3，不需要嵌入服务。反思请求经同一调度器发出，使用独立schema，不占评价预算。正式对照为V21批次`20261105c_v21_co6`：

```bash
.venv/bin/python -m experiments.traceaad.launch_host \
  --method v24 --experiment traceaad_v24 --batch BATCH_NAME \
  --repeats 3 --budget 1000 --initial-size 5 --seconds-per-instance 10 \
  --scheduler-socket /tmp/traceaad-1003/scheduler.sock \
  --env-file .env.server4 --launch
```

机制与记录见[V24](../../docs/机制设计/V24.md)。

V24.1 使用条件经验、完整计算背景和可恢复的待审阅队列，入口为 `--method v24.1` 或 `--method v24_1`。它不需要嵌入服务，沿用原模型调度、评价与最终测试流程；不要用 V24 的运行目录恢复 V24.1。单路配置检查示例：

```bash
.venv/bin/python -m experiments.traceaad.run \
  --method v24.1 --experiment traceaad_v24_1 --task tsp_construct \
  --run-name v24_1_dry --batch v24_1_dry \
  --scheduler-socket /tmp/traceaad-1003/scheduler.sock --dry-run
```

批量计划使用 `launch_host --method v24.1 --experiment traceaad_v24_1`，其余参数同上；只有明确启动新批次时添加 `--launch`。此前d7c238a的小规模真实模型验证对应批量材料与单事件来源。当前V24.1采用`focused_relations`，每周期围绕一个计算问题，只回读相关经验的完整来源关系；本轮没有运行新实验。旧运行目录不能直接按新材料规则恢复。规格与状态见[V24.1](../../docs/机制设计/V24.1.md)。

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
