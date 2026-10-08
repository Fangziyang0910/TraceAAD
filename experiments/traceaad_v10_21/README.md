# V10.21：精简提示（问题、接口与成绩）

规则见[时限只作为规则](../../docs/01-搜索方法/时限只作为规则.md)，实验条件见[实验准则](../PROTOCOL.md)。

```bash
# 计划（加 --launch 启动）
uv run python -m experiments.traceaad_v10_21.launch_local --suite co6 --batch <批次> --eval-workers 4
# 单路
uv run python -m experiments.traceaad_v10_21.run --task jssp_construct --backend server3 --seed 0 --repeat 1 --run-name <运行名> --eval-workers 4
# 冻结后的同规模测试
uv run python -m experiments.infra.evaluate experiments_result/traceaad_v10_21/<任务>/<运行> --primary --condition traceaad
```

当前批次是 TSP 的计算平台对照（10 月 9 日从头启动），9 路，种子 0–2，模型请求都发到 server3 的两个端点：`20261009_cpu_server3`（server3，Xeon Gold 6230R，0–51 号各一个物理核）、`20261009_cpu_local_p`（本机 8 个物理性能核，0、2、…、14 号）、`20261009_cpu_local_e`（本机 16 个能效核，16–31 号）。三个调度器各管 server3 两个端点各 2 个槽位，每次评价至多 8 个实例并行。候选函数每实例预算 2 秒 CPU 时间（函数内累计），整个实例安全上限 20 秒。同一段固定计算的 CPU 时间：本机性能核约 2 ms、能效核约 4 ms、server3 约 8 ms。

```bash
# 本机性能核、本机能效核（另一个 socket）、server3
uv run python -m experiments.infra.scheduler --cpus 0,2,4,6,8,10,12,14 --gpu-backend server3=2 --gpu-backend server3b=2
uv run python -m experiments.infra.scheduler --socket /tmp/traceaad-1000/scheduler_e.sock --cpus 16-31 --gpu-backend server3=2 --gpu-backend server3b=2
.venv/bin/python -m experiments.infra.scheduler --cpus 0-51 --gpu-backend server3=2 --gpu-backend server3b=2   # server3
uv run python -m experiments.traceaad_v10_21.launch_host --tasks tsp_construct --batch 20261008_cpu_local_p --eval-workers 8 --scheduler-socket /tmp/traceaad-1000/scheduler.sock --launch
```

## 逐实例执行参数（10月8日新增）

六任务评价有两个时间量：候选函数每实例的计算预算 **2 秒**（`benchmarks/tasks.py` 的 `FUNCTION_SECONDS`，由调用探针按函数内累计时间实时计量，超出即超时），以及整个实例的安全上限 `eval_timeout_seconds`，默认 **20 秒**（`INSTANCE_SECONDS`），只终止卡死的程序。`eval_workers` 默认 **1**。每个 worker 执行一个实例的完整评价，包括候选函数与固定框架；完成后领取下一实例。同一路搜索仍按“生成一次、评价一次”推进。

调用入口为 `InstanceProgramEvaluator.evaluate(code, source_key, timeout_seconds=5, n_workers=4)`，位于 `traceaad/common/instance_evaluation.py`。参数只作用于本次调用，省略时使用构造时的默认值。实际 worker 数不超过实例数。等待 worker 不占实例预算；实例进程启动、程序初始化、候选函数与固定框架均占预算。初始化任务时的数据生成与参考值计算不占预算。任一实例超时或失败，则整次评价失败，并停止本次其余运行中的实例；全部成功后按实例等权平均分数。

每个实例使用独立进程、独立程序全局变量和固定候选随机种子 `(evaluation_seed + 实例编号) % 2**32`。ACO 自身仍使用原来的 `aco_seed + 实例编号`。worker 数不改变随机流与汇总顺序。超时限制改变评价协议标识，worker 数作为执行参数记录。

六任务的七种基线也共用此执行器，预算与安全上限相同、候选种子730241，任务说明末尾带同一句预算，每次评价写入运行目录的 `evaluations.jsonl`；`--eval-workers` 控制实例并发。恢复搜索使用当前配置和评价器，活跃运行会同步保存当前执行参数。已有测试成绩直接读取；新的测试调用使用当前逐实例执行入口。

基线结束时也重评训练前五个不同源码的程序，再冻结有效结果；重评全部失败则记录失败。通用 `evaluate` 与 `heldout` 入口共用已有测试成绩，不重复评价同一冻结程序和规模。

训练、可选验证及冻结程序的测试共用此执行入口。测试默认读取运行记录中的时限与 worker 数；可用 `--workers` 与 `--timeout` 覆盖（`heldout` / `heldout_batch` 入口的时限参数名是 `--timeout-seconds`）。新运行的 `run_config.json` 以 `evaluation_execution` 记录实际执行条件；`task_eval` 中旧整套时限是底层任务构造参数，不再控制该执行入口。

评价记录的 `seconds` 是整次评价墙钟时间，`instance_seconds` 是各实例墙钟耗时之和，`instances` 保留逐实例分数、耗时、调用计数与失败。程序记录的 `eval_wall_seconds` 保存整次墙钟时间；`eval_seconds` 保存实例耗时之和，供提示除以实例数后描述每实例耗时。这样并行加速不会被误读成单实例算法计算量减少。

## 每台机器独立调度 CPU 与模型请求

启动入口 `experiments.infra.scheduler` 可直接运行在本机或 server3。一个常驻进程提供两个独立资源池，共享本机 Unix socket；CPU 分配与 GPU 请求分配各自决策。实验仍在所属机器上完成模型请求、评价和结果保存，不把候选程序传到另一台机器执行。

- CPU 默认检测本进程可用的 CPU 逻辑核。`--cpu-count N` 取其中 N 个；`--cpus 0-15,24-31` 指定核心编号。评价子进程实际绑定到分配的核心，所有接入实验合计不超过这份额度。这里管理的是交给此调度器的评价任务；其他未接入的进程不计入资源池。
- 模型端点用可重复的 `--gpu-backend 名称=槽位数` 配置，或 `--gpu-endpoint URL MODEL SLOTS` 指定地址、模型和容量。槽位限制的是在途生成请求数。所有端点应使用同一模型、权重、tokenizer 和 chat template；启动批次时核验模型名称和上下文容量。
- CPU 调度以实例为单位：空闲时同一评价可同时运行多个实例；其他评价到达后，优先把释放的核心分给占用较少的等待评价。运行中的实例不抢占。接入调度器时，`--eval-workers` 是**每次评价的并发上限**；省略则以上述 CPU 容量为上限，实际并发还受实例数和其他评价的需求限制。
- GPU 有空闲槽位就派发，在可用端点中选择已用槽位比例较低的端点。请求完成或失败即释放槽位；连接故障、429 或服务端错误会让对应端点暂避5秒，后续请求可使用其他端点。重试仍沿用原请求层。token 计数走相同模型的 tokenizer，不占生成槽位。
- CPU 等待队列不计入每实例时限。评价记录增加实际核心编号、峰值 worker 数和无可用 worker 时的等待时间。模型请求记录增加实际端点、槽位、排队时间及请求时间。
- 客户端断开时归还资源；调度器断开时评价入口停止已有实例并报错，不转为无额度执行。常驻调度器重启后需重新启动模型客户端。正常结束与超时沿用子进程清理。

### 例：server3 承担12路，本机承担6路

实验路数、CPU 核心数和模型槽位是三个独立配置。下面按六任务的重复1、2分给 server3（12路），重复3分给本机（6路），保留原种子0、1、2。两个机器上的仓库需包含当前实现与依赖，模型凭据仍使用各机器原有环境配置。

先在各机器的仓库中启动常驻调度器（可放入各自的 tmux 会话）：

```bash
# server3：示例使用96个可用逻辑核、12个模型槽位
uv run python -m experiments.infra.scheduler --socket /tmp/traceaad-server3.sock --cpu-count 96 --gpu-backend server3=6 --gpu-backend server3b=6

# 本机：示例使用24个可用逻辑核、6个模型槽位
uv run python -m experiments.infra.scheduler --socket /tmp/traceaad-local.sock --cpu-count 24 --gpu-backend server3=3 --gpu-backend server3b=3
```

省略 `--cpu-count` 即自动使用所在机器允许的全部 CPU。也可用 `--print-config` 先打印配置，不启动常驻进程。2026-10-08核对时，本机可用32个、server3可用104个逻辑核；实际以启动时检测为准。

在另一个终端打印本机负责的实验计划，加 `--launch` 才启动：

```bash
# server3 仓库：六任务各运行重复1、2，共12路
uv run python -m experiments.traceaad_v10_21.launch_host --batch scheduled_server3 --scheduler-socket /tmp/traceaad-server3.sock --repeat-ids 1 2

# 本机仓库：六任务各运行重复3，共6路
uv run python -m experiments.traceaad_v10_21.launch_host --batch scheduled_local --scheduler-socket /tmp/traceaad-local.sock --repeat-ids 3

# 单路也可接入；此处将该次评价的并发上限设为8
uv run python -m experiments.traceaad_v10_21.run --task jssp_construct --run-name scheduled_single --scheduler-socket /tmp/traceaad-local.sock --eval-workers 8

# 查看资源池容量、占用、等待任务和各模型端点的在途请求
uv run python -m experiments.infra.scheduler --socket /tmp/traceaad-local.sock --status
```

两个调度器若共享模型端点，需要预先分好端点额度。上述示例中，每个端点分别由 server3 管6槽、本机管3槽，合计9槽；两台机器互不协调，不能各自都配置为端点的全部容量。已有未接入调度器的模型请求也需要另外预留容量。

冻结程序的测试默认沿用保存的 CPU 调度入口，可用 `--scheduler-socket 路径` 覆盖，用 `--scheduler-socket ''` 改为直接评价。运行配置保存资源池信息，请求日志保存实际路由；调度不改变评分函数、数据和按实例确定的随机流。
