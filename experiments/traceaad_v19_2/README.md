# TraceAAD V19.2 的 server4 实验

本入口把 [`traceaad_v19_2`](../../traceaad_v19_2/README.md) 接入现有公共调度器。V19.2 当前使用有界连续开发、每段一个竞争代表与源码反馈；模型请求分配到任意空闲端点，完整评价申请一个 CPU 核；排队在预算计数与执行计时之前完成。

默认六任务 × 种子0、1、2，共18路，每路1000次评价，初始化5个有效个体。模型温度1.0，输出上限16384 tokens；Qwen关闭原生思考，其余采样参数与现有共享模型池一致，逐请求记录实际端点、模型、用量和排队时间。

生成与修复请求使用同一份 JSON Schema：`analysis`、`design`、`code` 为三个必填字符串，禁止额外字段；实际schema随请求设置保存。

```bash
python -m experiments.traceaad_v19_2.launch_host \
  --batch 20261101_v19_2_candidate \
  --experiment traceaad_v19_2 \
  --scheduler-socket /tmp/traceaad-1003/scheduler.sock \
  --env-file /home/fzy/code/LLM4AD/TraceAAD/.env
```

不加 `--launch` 时只输出计划；加上后以3秒间隔启动18个tmux会话。每路依次完成搜索、固定训练最好个体和独立主测试。训练与测试都采用任务的全部实例总时限，不使用旧入口的单实例计时或平均函数CPU门槛；评价进程允许新增2 GiB数据内存。

`--seconds-per-instance` 设置每实例时间额度，默认10秒，总时限为该额度乘实例数：训练16例共160秒，独立主测试50例共500秒。全部实例共享总时限，切换实例不重置计时。

生成与修复依次输出 `analysis`、`design`、`code`。分析说明采用或修改哪些计算、依据及其决策作用，设计概括完整算法；两项各用2–3句，最多100个英文词或200个中文字。代码不含注释和文档字符串。分析随个体保存，上下文按父子关系组织祖先、有效子代和失败子代的分析与评价。V19.2 的批次名和结果目录由启动参数确定；启动前应先完成计划预览和小规模预检。

原始搜索记录位于每路的 `native/`，公共结果文件供现有看板、批次状态查询和自动同步读取，预算轴明确为评价次数。恢复用单路 `experiments.traceaad_v19_2.run` 的原命令添加 `--resume`；调度排队不产生新模型请求或评价记录。

## 当前修订与对照

修订 `bounded_development_20261101` 使用 schema 5，不能续跑旧 schema 4。默认 `--development-calls 4 --parent-floor 0.05`。`--no-protect-new-segments` 和 `--no-local-history` 分别关闭初始化段优先开发、切回旧提示历史，launch_host 会将这些设置传给每路并记入清单。

每段只有一个最好实现参与段外竞争；episode 内允许工作版本退步，修复与失败占用相同请求额度。全局仍按评价数停止。完整机制、算子总量变化及必要对照见[设计](../../docs/机制设计/V19.2.md)。本次重构没有启动正式在线批次。历史主名单不因机制修订而改写。

对照开关 `--reselect-parent` 保留H、首步算子分布和修复上限，仅在普通episode的后续Refine重新选段代表；初始化保护保持相同。它用于检验接续，不用H=1同时改变多个变量。三个首步权重可显式配置为 `--refine-weight 0.45 --explore-weight 0.30 --crossover-weight 0.25`。真实算子总量仍受失败和repair影响，需按日志核对。
