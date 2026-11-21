# TraceAAD 统一实验入口

保留 Base、EvoTrace V1–V5。使用 `--method` 选择；默认 `base`。EvoTrace 支持点号与下划线两种写法。V21–V24 源码、测试和旧入口已移入本地 `archive/trajectory_methods_20261116/`。

| 方法 | 搜索实现 |
| --- | --- |
| `base` | Base 基础演化对照。 |
| `evotrace.v5` | 独立EvoTrace v5：树状种群、动态压缩与反思淘汰，评价预算按各条子路已走出的进步分配；[设计与用法](../../evotrace_v5/README.md)。参数只有评价预算、初始个体数、种子、采样与反思开关，V3/V4 树参数不适用。 |
| `evotrace.v4` | 独立EvoTrace v4：全树可行个体按排名与同一谱系位置投入抽样主父代，三算子统一建立真实父子派生边，局部满块路径收缩与合并子树剪枝凝练经验；[设计与用法](../../evotrace_v4/README.md)。 |
| `evotrace.v3` | 独立EvoTrace v3：在贪心覆盖代表集上按名次质量与直接投入抽样父代，通过路径收缩与子树剪枝成批凝练边经验与节点经验；[设计与用法](../../evotrace_v3/README.md)。 |
| `evotrace.v2` | 独立EvoTrace v2：按最近使用与质量差异一次选择，补齐必要分叉后联合剪枝和路径收缩；[设计与用法](../../evotrace_v2/README.md)。 |
| `evotrace.v1` | 独立EvoTrace v1：自有搜索、任务、模型与评价实现，使用树状种群、反思简化、加权分配与三尺度上下文。 |

默认后端为 `scheduled`，Qwen3.8-27B 经 server4 公共调度器分配。采样为温度1.0、top_p=0.8、top_k=20、min_p=0、presence_penalty=1.5，关闭思考，默认输出上限16384。完整设置见[scheduling.py](scheduling.py)与运行元数据。

在 server4 的 Eco 冻结源码目录下，下面的命令可预览一条 FSSP 实验计划。单路和批量入口均支持 `--backend opendesign-eco`。模型请求直连 OpenDesign Cloud 的 `deepseek-v4.1-flash-eco`，程序评价仍由公共调度器分配 CPU。私有配置文件提供 `OPENDESIGN_API_KEY`。V4 的生成、修复与反思请求均发送 `reasoning_effort=none`。生成结果中的 `analysis` 是演化轨迹读取的显式输出字段。

```bash
.venv/bin/python -m experiments.traceaad.launch_host \
  --method evotrace.v4 --backend opendesign-eco --experiment evotrace_v4_eco \
  --batch eco_preview --tasks fssp_gls --repeats 1 \
  --scheduler-socket /tmp/traceaad-1003/scheduler.sock \
  --env-file ~/.config/traceaad/opendesign-eco.env
```

V5 用同样的 `run`、`launch_host` 与队列入口，结果写入 `experiments_result/evotrace_v5/`（或 `--experiment` 指定的目录）。`--reflection/--no-reflection` 与 V4 相同；`--rank-half-life` 等 V3/V4 树参数、`--selection-temperature` 和 `--reference-code` 对 V5 报错。恢复沿用该运行保存的条件（种子、初始个体数、采样、`--seconds-per-instance`、反思开关与后端），与保存值不同时拒绝；只有 `--budget` 可以改，用于有意延长预算，原预算记入运行配置的 `budget_extended_from`。训练最好程序固定后的独立测试只调用评价器的单次评价并保存目标值。

EvoTrace的独立入口、配置与恢复约定见[V5](../../evotrace_v5/README.md)、[V3](../../evotrace_v3/README.md)、[V2](../../evotrace_v2/README.md)、[V1](../../evotrace_v1/README.md#配置与恢复)。不同版本分别建立批次并由各自冻结源码恢复；当前运行与保留批次清单以[实验与边界台账](../../docs/记录/README.md)为准。下方真实轨迹harness面向V2冻结记录，不能用V2树快照恢复V3。

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
  --method base --experiment traceaad_base_boltzmann_20261109 --batch BATCH_NAME \
  --repeats 3 --budget 1000 --scheduler-socket /tmp/traceaad-1003/scheduler.sock \
  --env-file .env.server4 --launch
```

## 四任务阶段与跨版本补位队列

四任务阶段使用TSP、CVRP、FSSP和OP，每版本三种子共12路。批量命令显式传入任务列表；`--enqueue`只写入等待清单，不能与`--launch`同时使用：

```bash
.venv/bin/python -m experiments.traceaad.launch_host \
  --method base --experiment traceaad_base_boltzmann_20261109 --batch BATCH_NAME \
  --tasks tsp_construct cvrp_aco fssp_gls op_aco \
  --repeats 3 --budget 1000 --selection-temperature 1 \
  --scheduler-socket /tmp/traceaad-1003/scheduler.sock --env-file .env.server4 --enqueue
```

`python -m experiments.traceaad.queue`读取`--manifests`中的等待清单，用`--observe`纳入已运行的批次，`--capacity 36`限制整个活动队列的路数。第一个等待版本优先启动，后续版本交替补位。运行进程包含搜索和主测试；训练结束不提前释放位置。错误退出保留记录并标为blocked，不自动重跑，也不阻塞其他等待任务。

队列从进程、tmux和持久清单恢复状态，锁文件阻止重复控制器；重启控制器不重复启动已有运行。队列放在server4的独立tmux会话，SSH断开后继续运行。监控区分queued、deferred和异常退出，等待期间也检查控制会话是否存在。

## EvoTrace V2 真实轨迹 harness

`harness`捕获实际请求、响应、代码、评价、树状态和简化分区。案例按阶段与材料可用性选择，不按成绩挑选，覆盖初始化、三算子、两类反思及格式、执行失败和超时修复。只读取已提交的完整日志行，损坏的已提交记录直接报错。

在仓库根目录执行，将`PACK`替换为新的结果目录，`FROZEN_SOURCE`和`CANDIDATE_SOURCE`指向包含`evotrace_v2/`的源码根目录：

```bash
.venv/bin/python -m experiments.traceaad.harness capture \
  --out PACK --source FROZEN_SOURCE
.venv/bin/python -m experiments.traceaad.harness render \
  --out PACK --source FROZEN_SOURCE --arm before --require-original
.venv/bin/python -m experiments.traceaad.harness render \
  --out PACK --source CANDIDATE_SOURCE --arm after
```

`--require-original`逐字核对重建提示与历史实际请求。已有`PACK`固定案例集合；日志继续增长也不重新选择。`prefixes/`封存对应运行的日志，支持以后从同一历史状态恢复。分析其他批次或增加案例时用新`PACK`，并指定`--runs`与`--batch`。`--match`接受案例名正则。

模型调用与评价在接入公共调度器的server4环境执行。两臂分别运行：

```bash
.venv/bin/python -m experiments.traceaad.harness run \
  --out PACK --source CANDIDATE_SOURCE --arm after \
  --workers 2 --repeats 1 --env-file .env --env-file .env.server4
.venv/bin/python -m experiments.traceaad.harness evaluate \
  --out PACK --source CANDIDATE_SOURCE --arm after --workers 2
```

生成与修复沿用四任务实际条件，不自动重选程序或追加修复。接口、解析、源码校验和执行结果分别留存。既有响应只有提示哈希一致时才能复用；改提示使用新arm。失败原件保留，增加`--repeats`会生成新的重复样本。收集会完成所有已选案例，遇到缺失结果、解析/校验失败、执行报错或超时则以非零状态退出；执行子集时`run`与`evaluate`使用相同的`--match`。

`handoff`通过真实`reflect_and_apply`把新反思写回该次历史简化，其他区域沿用同一份历史响应，再渲染读取该经验的Refine或Explore：

```bash
.venv/bin/python -m experiments.traceaad.harness handoff \
  --out PACK --source CANDIDATE_SOURCE --arm after --match reflect_variation
.venv/bin/python -m experiments.traceaad.harness run \
  --out PACK --source CANDIDATE_SOURCE --arm after_handoff \
  --workers 2 --env-file .env --env-file .env.server4
.venv/bin/python -m experiments.traceaad.harness evaluate \
  --out PACK --source CANDIDATE_SOURCE --arm after_handoff --workers 2
```

它会检查新经验是否真正进入后续输入。完整提示、响应与任务评价供逐案例审阅；不按术语、AST距离或关键词给算法思想打分，也不把固定状态的单次胜出解释为完整搜索收益。

需要检查真实的生成—评价—失败修复—重评—入树—反思链路时，使用`step`。它恢复案例之前的尝试、档案和树状态，复用已经获得的生成响应，随后运行真正的`EvoTrace.step()`，实际调用修复和反思模型；新的原始调用、实际评价预算及更新后的树保存在`steps/`：

```bash
.venv/bin/python -m experiments.traceaad.harness step \
  --out PACK --source CANDIDATE_SOURCE --arm after --match developed_Explore \
  --workers 2 --env-file .env --env-file .env.server4
```

`step`校验真实步骤重建的生成提示与已获响应一致。重复执行只复用源码、任务和提示均匹配的已完成步骤；新样本先增加`run --repeats`，再运行相应重复编号。历史回放在原批次的冻结基准仓库内执行，`--source`只切换方法包。

V3父代选择的代表集名次减半跨度由`--rank-half-life`设置，默认10；`run`和`launch_host`均支持。该参数只适用于V3，写入批次及运行配置，并实际传给搜索。恢复时省略该参数会读取原生记录中的值，显式更改会拒绝恢复。精确定义见[V3设计](../../docs/EvoTrace-v3设计.md)。

V3覆盖代表集与反思树简化由三个参数控制：`--max-representatives`（默认最多24个互不被更优近邻覆盖的代表）、`--cover-distance`（默认AST距离小于0.15视为被更优代表覆盖）与`--region-size`（默认4个非骨架节点起批触发路径收缩或子树剪枝）。三参数同样支持`run`和`launch_host`，记录在批次与运行的`topology`字段中。恢复时校验`selection_rule`与原生配置，早期冻结运行由各自快照源码恢复。
