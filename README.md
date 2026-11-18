# TraceAAD

**树状种群使算法的优胜劣汰，同时成为积累和复用算法设计经验的过程。**

EvoTrace（论文方法名 **TREES**）以树状种群记录算法的演化关系。淘汰算法时，反思结合相关代码变化、评价结果与已有经验形成算法设计经验，并在路径收缩和子树剪枝后将经验保存在重连边与保留节点上。已有经验在后续更新中继续整合，与相关算法及其演化轨迹共同用于精化、交叉和探索。

## 从哪里开始

| 你想做什么 | 阅读入口 |
| --- | --- |
| 理解研究问题与核心思路 | [树状种群与反思简化](docs/树状种群与反思简化.md) |
| 核对当前 V4 的规则、公式与参数 | [V4 设计](docs/EvoTrace-v4设计.md) |
| 阅读完整研究稿、PDF与复现材料 | [论文与PDF](paper/README.md)、[英文主稿](docs/paper.md)、[中文对照](docs/paper-中文.md) |
| 撰写论文或选择投稿场合 | [顶会顶刊论文写作指南](docs/顶会顶刊论文写作指南.md) |
| 复查历史版本的设计 | [V1](docs/EvoTrace-v1设计.md)、[V2](docs/EvoTrace-v2设计.md)、[V3](docs/EvoTrace-v3设计.md) |
| 了解 Base 的个体表示、上下文与提示 | [Base设计](docs/TraceAAD-Base设计.md) |
| 核对当前对照批次与已验证边界 | [实验与边界台账](docs/记录/README.md) |
| 独立使用 EvoTrace 或接入任务与模型 | [V4](evotrace_v4/README.md)、[V3](evotrace_v3/README.md)、[V2](evotrace_v2/README.md)、[V1](evotrace_v1/README.md) |
| 通过公共调度器运行 EvoTrace 或 Base | [统一入口](experiments/traceaad/README.md) |
| 设计、运行或复查实验 | [V3实验方案](docs/EvoTrace-v3实验方案.md)、[实验入口](experiments/README.md) |
| 查阅第三方提示的来源版本 | [来源记录](baselines/UPSTREAM.md) |

## 代码与环境

| 目录 | 内容 |
| --- | --- |
| `evotrace_v4/` | 独立EvoTrace v4：树即种群（全树存活可行节点竞争评价预算），统一父子生长（含探索基于主父代重构），局部满块反思式淘汰与可追溯派生上下文（仅输出 `design` + `code`）。 |
| `evotrace_v3/` | 独立EvoTrace v3：选择覆盖代表集，依据适应度排名与子代评价次数抽样父代；结合反思执行路径收缩和子树剪枝，并按演化轨迹组织生成上下文。 |
| `evotrace_v2/` | 独立EvoTrace v2：新算法经评价后加入树中并更新保留集合，对待移除区域执行反思，将经验保存在保留节点和边上。 |
| `evotrace_v1/` | 独立EvoTrace v1：搜索循环、树状种群、反思简化、预算分配与三尺度生成，自带模型、评价和记录恢复。 |
| `traceaad_base/` | 不利用轨迹经验的 Base 对照与独立运行入口。 |
| `baselines/` | 比较方法。 |
| `benchmarks/` | 任务、数据与评价。 |
| `experiments/` | 启动、查询、同步与分析工具。 |

在仓库根目录安装环境和运行测试。模型凭据由本地环境提供。

```bash
uv sync --frozen
uv run python -m pytest -q tests
```

本机 `archive/` 与 `experiments_result/` 保存封存文档和实验结果，不进入提交。
