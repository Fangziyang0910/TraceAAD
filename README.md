# TraceAAD

**研究怎样利用算法设计中的试错经验，在有限预算下发现和改进优质算法。**

搜索产生程序，也产生经验。我们关心这些经验怎样改善后续的生成、修改与选择。具体机制由问题与证据决定。

## 从哪里开始

| 你想做什么 | 阅读入口 |
| --- | --- |
| 理解本工作针对的问题 | [WWH](docs/WWH.md) |
| 理解三个机制的设计问题、定义与作用过程 | [机制设计文档](docs/机制设计/README.md) |
| 了解 Base 的个体表示、上下文与提示 | [TraceAAD Base 基础方法说明](docs/TraceAAD-Base.md) |
| 了解Base加机制1的V23.1 | [V23.1机制](docs/机制设计/V23.1.md) |
| 了解V21加机制1的V22.1 | [V22.1机制](docs/机制设计/V22.1.md) |
| 运行 Base 的独立实现 | [traceaad_base](traceaad_base/README.md) |
| 了解只含上下文机制的 V21 | [V21 机制](docs/机制设计/V21.md) |
| 了解带计算条件的经验机制 V24.1 | [V24.1 机制](docs/机制设计/V24.1.md) |
| 运行 Base、V21、V22.1、V23.1、V24 或 V24.1 | [实验入口](experiments/traceaad/README.md) |
| 设计、运行或复查实验 | [实验入口](experiments/README.md) |
| 查阅第三方提示的来源版本 | [来源记录](baselines/UPSTREAM.md) |

## 代码与环境

| 目录 | 内容 |
| --- | --- |
| `traceaad_v24_1/` | V21加条件经验与完整实现反思；V24的旧设计笔记保留于 `traceaad_v24/`。 |
| `traceaad_v23_1/` | 当前Base加机制1，与V22.1共用轨迹经验分配。 |
| `traceaad_v22_1/` | V21加机制1：真实冠军改善与近期质量匹配的预算分配。 |
| `traceaad_v21/` | 只含轨迹上下文与算子的方法。 |
| `traceaad_base/` | 不利用轨迹经验的 Base 对照与独立运行入口。 |
| `baselines/` | 比较方法。 |
| `benchmarks/` | 任务、数据与评价。 |
| `experiments/` | 启动、查询、同步与分析工具。 |

在仓库根目录安装环境和运行测试。模型凭据由本地环境提供。

```bash
uv sync --frozen
uv run python -m pytest -q tests
```

旧文档已在本机封存于 `archive/documents_20261016_144049/` 与 `archive/documents_20261104_0900/`。归档不进入新提交；已有 Git 历史保留。

2026年11月9日起，当前 Base 改为不利用轨迹经验的基础对照。V23.1 已同步为当前 Base 加机制1；V21、V22.1、V24、V24.1 沿用原有生成行为。历史批次按当时的冻结源码解释，V23.1 的旧结果仍属于旧 Base 生成条件。边界见 [Base 说明](docs/TraceAAD-Base.md)。
