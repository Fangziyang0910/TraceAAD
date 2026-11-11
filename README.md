# TraceAAD

**研究怎样利用算法设计中的试错经验，在有限预算下发现和改进优质算法。**

搜索产生程序，也产生经验。我们关心这些经验怎样改善后续的生成、修改与选择。具体机制由问题与证据决定。

## 从哪里开始

| 你想做什么 | 阅读入口 |
| --- | --- |
| 理解本工作针对的问题与当前叙事 | [树状种群与反思简化](docs/树状种群与反思简化.md) |
| 阅读论文摘要与引言 | [摘要与引言](docs/paper-摘要与引言.md) |
| 查阅演化轨迹叙事与旧版方法 | [演化轨迹叙事](docs/演化轨迹叙事/README.md) |
| 了解 Base 的个体表示、上下文与提示 | [Base 说明](docs/演化轨迹叙事/TraceAAD-Base.md) |
| 运行 Base、V21.3、V22.2、V23.2 或 V24.1 | [统一入口](experiments/traceaad/README.md) |
| 设计、运行或复查实验 | [实验入口](experiments/README.md) |
| 查阅第三方提示的来源版本 | [来源记录](baselines/UPSTREAM.md) |

## 代码与环境

| 目录 | 内容 |
| --- | --- |
| `traceaad_base/` | 不利用轨迹经验的 Base 对照与独立运行入口。 |
| `traceaad_v21_3/` | Base 加机制2：三种算子各自的轨迹上下文。 |
| `traceaad_v22_2/` | V21.3 加机制1：轨迹收益修正的预算分配与轨迹嵌入参考。 |
| `traceaad_v23_2/` | Base 加机制1，与 V22.2 共用实现。 |
| `traceaad_v24_1/` | V21.3 加机制3：条件经验、展示选取与跨分支修订。 |
| `baselines/` | 比较方法。 |
| `benchmarks/` | 任务、数据与评价。 |
| `experiments/` | 启动、查询、同步与分析工具。 |

在仓库根目录安装环境和运行测试。模型凭据由本地环境提供。

```bash
uv sync --frozen
uv run python -m pytest -q tests
```

本机 `archive/` 与 `experiments_result/` 保存封存文档和实验结果，不进入提交。
