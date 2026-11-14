# TraceAAD

**研究怎样利用算法设计中的试错经验，在有限预算下发现和改进优质算法。**

EvoTrace将反思视为搜索经历的压缩，用树状种群连接候选竞争与发展历史组织。一次种群更新同时确定可开发实现、需要凝练的历史区域，以及承接经验的保留节点与边。后续算法生成从当前种群中读取程序与相关经历。

## 从哪里开始

| 你想做什么 | 阅读入口 |
| --- | --- |
| 理解研究问题与核心思路 | [树状种群与反思简化](docs/树状种群与反思简化.md) |
| 核对当前V3规则、公式与参数 | [V3正式设计](docs/EvoTrace-v3设计.md)；历史[V2设计](docs/EvoTrace-v2设计.md) |
| 阅读完整研究稿、PDF与复现材料 | [论文交付](paper/README.md)、[英文主稿](docs/paper.md)、[中文对照](docs/paper-中文.md) |
| 撰写论文或选择投稿场合 | [顶会顶刊论文写作指南](docs/顶会顶刊论文写作指南.md) |
| 复查V1容量触发基线 | [V1正式设计](docs/EvoTrace-v1设计.md) |
| 追溯机制修订与实验依据 | [本周记录](docs/记录/2026-W46.md) |
| 独立使用 EvoTrace 或接入任务与模型 | [V3](evotrace_v3/README.md)、[V2](evotrace_v2/README.md)、[V1](evotrace_v1/README.md) |
| 查阅演化轨迹叙事与旧版方法 | [演化轨迹叙事](docs/演化轨迹叙事/README.md) |
| 了解 Base 的个体表示、上下文与提示 | [Base 说明](docs/演化轨迹叙事/TraceAAD-Base.md) |
| 运行 Base、V21.3、V22.2、V23.2 或 V24.1 | [统一入口](experiments/traceaad/README.md) |
| 设计、运行或复查实验 | [实验入口](experiments/README.md) |
| 查阅第三方提示的来源版本 | [来源记录](baselines/UPSTREAM.md) |

## 代码与环境

| 目录 | 内容 |
| --- | --- |
| `evotrace_v3/` | 独立EvoTrace v3：个体按质量与历史子树投入全局抽样；简化交接经验并保留各个体自身的投入，按真实发展关系读取上下文。 |
| `evotrace_v2/` | 独立EvoTrace v2：受评入树后更新保留集合，凝练移除区域，将经验交给保留节点和边。 |
| `evotrace_v1/` | 独立EvoTrace v1：搜索循环、树状种群、反思简化、预算分配与三尺度生成，自带模型、评价和记录恢复。 |
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
