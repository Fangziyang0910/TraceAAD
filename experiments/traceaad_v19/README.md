# TraceAAD V19 的 server4 实验

本入口把 [`traceaad_v19`](../../traceaad_v19/README.md) 接入现有公共调度器。V19 只替换 Base 的父代机会分配：先固定选择算子，再按该算子的局部轨迹选择父代；模型请求分配到任意空闲端点，完整评价申请一个 CPU 核；排队在预算计数与执行计时之前完成。

默认六任务 × 种子0、1、2，共18路，每路1000次评价，初始化5个有效个体。模型温度1.0，输出上限16384 tokens；Qwen关闭原生思考，其余采样参数与现有共享模型池一致，逐请求记录实际端点、模型、用量和排队时间。

生成与修复请求使用同一份 JSON Schema：`analysis`、`design`、`code` 为三个必填字符串，禁止额外字段；实际schema随请求设置保存。

```bash
python -m experiments.traceaad_v19.launch_host \
  --batch 20261031_v19_co6 \
  --experiment traceaad_v19 \
  --scheduler-socket /tmp/traceaad-1003/scheduler.sock \
  --env-file /home/fzy/code/LLM4AD/TraceAAD/.env
```

不加 `--launch` 时只输出计划；加上后以3秒间隔启动18个tmux会话。每路依次完成搜索、固定训练最好个体和独立主测试。训练与测试都采用任务的全部实例总时限，不使用旧入口的单实例计时或平均函数CPU门槛；评价进程允许新增2 GiB数据内存。

`--seconds-per-instance` 设置每实例时间额度，默认10秒，总时限为该额度乘实例数：训练16例共160秒，独立主测试50例共500秒。全部实例共享总时限，切换实例不重置计时。

生成与修复依次输出 `analysis`、`design`、`code`。分析说明采用或修改哪些计算、依据及其决策作用，设计概括完整算法；两项各用2–3句，最多100个英文词或200个中文字。代码不含注释和文档字符串。分析随个体保存，上下文按父子关系组织祖先、有效子代和失败子代的分析与评价。当前启动配置为 `20261031_v19_co6`，独立结果目录为 `traceaad_v19_schema`；此前各批次保留在各自冻结目录。

原始搜索记录位于每路的 `native/`，公共结果文件供现有看板、批次状态查询和自动同步读取，预算轴明确为评价次数。恢复用单路 `experiments.traceaad_v19.run` 的原命令添加 `--resume`；调度排队不产生新模型请求或评价记录。
