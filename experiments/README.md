# 实验入口

这里说明怎样执行和复查实验。研究问题与认识见[研究文档](../docs/README.md)。

| 需要做什么 | 入口 |
| --- | --- |
| 设计实验、核对比较条件 | [实验条件](PROTOCOL.md) |
| 查询状态、恢复、远端操作 | [操作说明](infra/AGENT_OPERATIONS.md) |
| 找到源码、反馈与最终结果 | [结果格式](infra/SEARCH_FORMAT.md) |
| 理解当前 TraceAAD 的运行规则 | [搜索实现](../traceaad/README.md) |

## 训练可视化

看板集中在 [visualization/](visualization/)，展示训练曲线、运行进度、程序详情与结果对比。在仓库根目录启动：

```bash
uv run python -m experiments.visualization --host 127.0.0.1 --port 8765
```

浏览器打开 `http://127.0.0.1:8765/`。用 `--experiment` 选择方法目录，或用 `--batch` 筛选运行批次。看板读取当前结果格式，在进程内保留增量状态，不写入实验目录。不同测试条件按 `variant` 分开展示。

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
