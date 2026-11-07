# TraceAAD

**研究怎样利用算法设计中的试错经验，在有限预算下发现和改进优质算法。**

搜索产生程序，也产生经验。我们关心这些经验怎样改善后续的生成、修改与选择。具体机制由问题与证据决定。

## 从哪里开始

| 你想做什么 | 阅读入口 |
| --- | --- |
| 理解本工作针对的问题 | [WWH](docs/WWH.md) |
| 阅读各版本的研究问题与机制设计 | [机制设计文档](docs/机制设计/README.md) |
| 了解 Base 的个体表示、上下文与提示 | [TraceAAD Base 基础方法说明](docs/TraceAAD-Base.md) |
| 了解V21加机制1的V22 | [V22机制](docs/机制设计/V22.md) |
| 查阅此前的V20 | [V20机制](docs/机制设计/V20.md) |
| 运行 Base 的独立实现 | [traceaad_base](traceaad_base/README.md) |
| 了解只含上下文机制的 V21 | [V21 机制](docs/机制设计/V21.md) |
| 运行 Base、V20、V21 或 V22 | [实验入口](experiments/traceaad_v20/README.md) |
| 设计、运行或复查实验 | [实验入口](experiments/README.md) |
| 查阅第三方提示的来源版本 | [来源记录](baselines/UPSTREAM.md) |

## 代码与环境

| 目录 | 内容 |
| --- | --- |
| `traceaad_v22/` | V21加机制1：轨迹价值预测、两步预算分配和轨迹上下文。 |
| `traceaad_v20/` | 此前的轨迹评分、轨迹上下文与经验学习实现。 |
| `traceaad_v21/` | 只含轨迹上下文与算子的方法。 |
| `traceaad_base/` | Base 机制的独立实现与运行入口。 |
| `baselines/` | 比较方法。 |
| `benchmarks/` | 任务、数据与评价。 |
| `experiments/` | 启动、查询、同步与分析工具。 |

在仓库根目录安装环境和运行测试。模型凭据由本地环境提供。

```bash
uv sync --frozen
uv run python -m pytest -q tests
```

旧文档已在本机封存于 `archive/documents_20261016_144049/` 与 `archive/documents_20261104_0900/`。归档不进入新提交；已有 Git 历史保留。
