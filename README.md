# TraceAAD

**研究怎样利用算法设计中的试错经验，在有限预算下发现和改进优质算法。**

搜索产生程序，也产生经验。我们关心这些经验怎样改善后续的生成、修改与选择。具体机制由问题与证据决定。

## 从哪里开始

| 你想做什么 | 阅读入口 |
| --- | --- |
| 理解本工作针对的问题 | [WWH](docs/WWH.md) |
| 了解当前程序怎样搜索 | [搜索实现](traceaad/README.md) |
| 设计、运行或复查实验 | [实验入口](experiments/README.md) |
| 查阅第三方提示的来源版本 | [来源记录](baselines/UPSTREAM.md) |

## 代码与环境

| 目录 | 内容 |
| --- | --- |
| `traceaad/` | 当前搜索方法。 |
| `baselines/` | 比较方法。 |
| `benchmarks/` | 任务、数据与评价。 |
| `experiments/` | 启动、查询、同步与分析工具。 |

在仓库根目录安装环境和运行测试。模型凭据由本地环境提供。

```bash
uv sync --frozen
uv run python -m pytest -q tests
```

旧文档已在本机封存于 `archive/documents_20261016_144049/`。归档不进入新提交；已有 Git 历史保留。
