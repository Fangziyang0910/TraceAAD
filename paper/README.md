# EvoTrace研究稿

**EvoTrace: Experience-Bearing Tree-Structured Populations for Automated Algorithm Design**

论文以树状种群结合候选竞争与派生关系，将反思作为树简化中的经验压缩。当前方法采用活跃工作集、LCA分叉保留和成批区域反思，后续改进、交叉与探索从同一棵树读取算法与经验。

## 阅读与编辑

| 材料 | 入口 |
| --- | --- |
| 英文与中文PDF | [英文](evotrace.pdf)、[中文](evotrace-中文.pdf) |
| 完整材料包 | [源码与PDF](EvoTrace-paper-source.zip) |
| Markdown维护源 | [英文](../docs/paper.md)、[中文](../docs/paper-中文.md) |
| 生成的LaTeX | [英文](main.tex)、[中文](main-zh.tex) |
| 当前思路与精确规则 | [研究概述](../docs/树状种群与反思简化.md)、[V3设计](../docs/EvoTrace-v3设计.md) |
| 参考文献维护源 | [共享BibTeX](../docs/paper-references.bib) |
| 冻结证据 | [快照](evidence/snapshot.json)、[逐种子分数](evidence/training_scores.csv)、[核验报告](evidence/manuscript_checks.json) |

## 方法与证据版本

方法部分描述当前的工作集与成批阈值规则。实验部分保留批次 `20261115_115206_evotrace_v3_revised_co4` 在北京时间11月15日15:36的快照：四任务三种子，共2,641次评价，共同横截面为180次评价。

该批次使用旧的 `quality_slots = 32` 与 `recent_evaluations = 40` 规则，以及个体直接投入采样；结果只描述其冻结配置。当前规则已有结构回放与不变量检查，完整搜索及固定状态消融按附录E另行检验。

论文中的搜索曲线读取上述冻结快照。旧机制图未体现当前成批阈值，已从正文与材料包移除；历史图示保留于[版本图目录](../docs/figures/evotrace/README.md)。

## 重建与校验

从Markdown生成双语LaTeX与PDF，并核对引用、公式、冻结分数和排版日志：

```bash
.venv/bin/python paper/build.py
.venv/bin/python paper/validate.py
```

构建需要Pandoc、XeLaTeX和latexmk。Pandoc不在PATH中时可通过 `--pandoc /path/to/pandoc` 指定。构建不刷新实验数据。

已有LaTeX也可在 `paper/` 下通过 `latexmk -xelatex main.tex` 与 `latexmk -xelatex main-zh.tex` 编译。
