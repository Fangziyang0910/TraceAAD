# 实验入口

这里说明怎样执行和复查实验。研究问题见 [WWH](../docs/WWH.md)。

| 需要做什么 | 入口 |
| --- | --- |
| 设计实验、核对比较条件 | [实验条件](PROTOCOL.md) |
| 查询状态、恢复、远端操作 | [操作说明](infra/AGENT_OPERATIONS.md) |
| 找到源码、反馈与最终结果 | [结果格式](infra/SEARCH_FORMAT.md) |
| 理解当前 TraceAAD 的运行规则 | [搜索实现](../traceaad/README.md) |
| V13 历史实现与恢复 | [V13 实现](../traceaad/v13/README.md) |
| V13.1 冻结变换、配对诊断与条件信用 | [V13.1 实现](../traceaad/v13_1/README.md) |
| V14 回访拒绝、停滞后无代码发散与发育 | [V14 实现](../traceaad/v14/README.md) |
| V15 曝光策略：等投入分配与信息形式曝光 | [V15 实现](../traceaad/v15/README.md) |
| V16 论文方法：乐观估计、修改说明与跨谱系参考程序 | [V16 实现](../traceaad/v16/README.md) |
| V17 按谱系自身进步继续开发，谱系间导入组件 | [V17 实现](../traceaad/v17/README.md) |
| 检验变换迁移、模型执行与在线闭环 | [变换迁移先导实验](transfer/README.md) |

## 训练可视化

看板集中在 [visualization/](visualization/)，展示训练曲线、运行进度、程序详情与结果对比。在仓库根目录启动：

```bash
uv run python -m experiments.visualization --host 0.0.0.0 --port 8765
```

浏览器打开 `http://10.150.68.218:8765/`；本机也可使用 `http://127.0.0.1:8765/`。用 `--experiment` 选择方法目录，或用 `--batch` 筛选运行批次。看板读取当前结果格式，在进程内保留增量状态，不改写实验记录。

当前主实验范围由[主实验名单](../experiments_result/primary_runs.json)固定：12 个基线和当前 TraceAAD，每个方法六任务 × 种子 0／1／2，共 18 路 Qwen3.8 实验。看板还按名单中的 `visualization_batches` 读取历史 TraceAAD 正式批次，并按版本分别展示。默认对比全部版本和基线；未用完预算的训练结果会标记。名单沿用既有基线盘点，不按成绩或完成情况挑选运行。ReEvo、FunSearch 的 GGUF 第三路属于主实验；EoH 额外的 GGUF 种子 0 六路、DeepSeek 试跑和其他先导实验不进入看板。

主对比使用 `variant` 为空的原主测试。`relaxed_time_cpu120_wall600` 等放宽限制补测保留在原始 `heldout.json` 中，不再生成单独的默认对比对象，也不替代主测试失败或缺失。新增正式批次时需更新主实验名单；没有该名单的结果目录仍可按原方式查看全部记录。

## 1. 检查配置与启动计划

在仓库根目录执行。默认模型凭据使用 `DEEPSEEK_API_KEY`。端点与采样设置见 [base.py](infra/base.py)。下面的 `BATCH_NAME` 表示待替换的新批次名。

检查单任务配置，不调用模型：

```bash
uv run python -m experiments.traceaad.run \
  --task tsp_construct --dry-run
```

查看六任务、每任务三路独立搜索的计划。加 `--launch` 才启动：

```bash
uv run python -m experiments.traceaad.launch_host \
  --batch BATCH_NAME --suite co6 --repeats 3 --budget 1000
```

查看基线排队计划。去掉 `--dry-run` 才执行：

```bash
uv run python -m experiments.infra.local_queue \
  --method eoh --suite co6 --batch BATCH_NAME --slots 3 --dry-run
```

TraceAAD 入口为 [run.py](traceaad/run.py) 与 [launch_host.py](traceaad/launch_host.py)。基线批量启动统一使用 [local_queue.py](infra/local_queue.py)，支持的方法见其 `--help`。各方法的 `run.py` 执行一路搜索。改变条件时使用新批次名。

DeltaEvolve 使用 `--method deltaevolve`；单路入口为 `experiments.deltaevolve.run`，支持 `--dry-run`。论文机制、缺失细节的补充规则与 embedding 配置见[实现说明](../baselines/deltaevolve/README.md)。

OpenEvolve 使用 `--method openevolve`；单路入口为 `experiments.openevolve.run`，支持 `--dry-run`。官方源码版本、成绩换算与串行执行等适配见[实现说明](../baselines/openevolve/README.md)。

HSEvo 使用 `--method hsevo`；单路入口为 `experiments.hsevo.run`，支持 `--dry-run`。预算按函数评价次数计，含和声搜索的每组参数；同分兜底与其他适配见[实现说明](../baselines/hsevo/README.md)。

RefineEvo 使用 `--method refineevo`；单路入口为 `experiments.refineevo.run`，支持 `--dry-run`。预算按生成回复计，含上游重试的解析失败与重复回复；经验检索需要 embedding 凭据，同批经验时序与其他适配见[实现说明](../baselines/refineevo/README.md)。

Clade-AHD 使用 `--method clade_ahd`；单路入口为 `experiments.clade_ahd.run`，支持 `--dry-run`。预算按返回的生成回复计，叶批次统一提交树更新，恢复时读取完整树与随机状态。上游实际行为及调度差异见[实现说明](../baselines/clade_ahd/README.md)。

## 2. 查询结果

将 `MANIFEST_PATH` 替换为批次清单路径，将 `RUN_DIR` 替换为运行目录。

```bash
uv run python -m experiments.infra.batch_status \
  --manifest MANIFEST_PATH --details
```

`batch_status` 只读查询批次状态。具体程序、评价和来源关系从结果文件读取。

## 3. 独立测试

最终程序确定后，再执行同规模测试：

```bash
uv run python -m experiments.infra.evaluate RUN_DIR --primary
```

测试会运行程序并保存结果。已有结果先读取；不要因文档或实现变化自动重测。历史运行使用当时配置与源码快照，当前评价器不能替代历史运行的实际评价规则。

## 结果存在哪里

```text
experiments_result/<方法>/
├── batch_<批次>.json       批次清单
└── <任务>/<运行>/          单路搜索的配置、事件与结果
```

目录名不能证明状态或比较条件。核对时读取清单、配置和实际事件，字段说明见[结果格式](infra/SEARCH_FORMAT.md)。
