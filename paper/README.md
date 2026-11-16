# EvoTrace研究论文

**EvoTrace: Experience-Bearing Tree-Structured Populations for Automated Algorithm Design**

EvoTrace研究如何使算法的优胜劣汰同时成为积累和复用算法设计经验的过程。树状种群记录算法的演化关系，淘汰时结合反思执行路径收缩与子树剪枝，将设计经验保存在重连边和保留节点上。更新后的种群用于预算分配，并为精化、交叉和探索提供算法、演化轨迹与设计经验。

## 阅读与维护

| 材料 | 入口 |
| --- | --- |
| 双语PDF | [英文](evotrace.pdf)、[中文](evotrace-中文.pdf) |
| 可编辑正文 | [英文Markdown](../docs/paper.md)、[中文Markdown](../docs/paper-中文.md) |
| 研究论证与方法规则 | [研究概述](../docs/树状种群与反思简化.md)、[V3设计](../docs/EvoTrace-v3设计.md) |
| 完整源码与PDF | [材料包](EvoTrace-paper-source.zip) |
| 生成的LaTeX | [英文](main.tex)、[中文](main-zh.tex) |
| 参考文献 | [共享BibTeX](../docs/paper-references.bib) |
| 实验安排与完成情况 | [V3实验方案](../docs/EvoTrace-v3实验方案.md)、[实验台账](../docs/记录/README.md) |

正文依次展开研究问题、树状种群的更新与使用、种群更新与经验积累的分析、相关工作及讨论。参考文献位于正文之后，附录另起页置于全文末尾，集中收录配置、结构证明、反思提示与复现条件。未完成的实验安排单独维护在实验方案中。

## 格式与机制图

双语PDF采用[ICML 2027官方模板](https://media.icml.cc/Conferences/ICML2027/styles/icml2027.zip)的匿名审稿模式：Letter纸张、双栏正文、模板规定的标题与图表样式，以及作者—年份引用。附录采用单栏。英文主稿遵循官方正文字体；中文对照稿增加CJK字体支持。官方样式与参考文献样式原文件保存在 `vendor/icml2027/`，下载来源和哈希见 `SOURCE.json`。

两张机制图分别说明[淘汰、经验积累与复用](../docs/figures/evotrace/12-v3-experience-cycle.png)和[连续更新中的经验积累](../docs/figures/evotrace/13-v3-experience-accumulation.png)。前者概括整个搜索过程，后者展示已有经验如何进入下一次反思。树中的实线表示演化关系，经验以独立标注附着在边或节点上。当前图稿、生成提示与检查记录见[机制图目录](../docs/figures/evotrace/README.md)。

## 方法与实验范围

方法部分对应当前V3：代表覆盖选择、分叉骨架、成批路径收缩与子树剪枝，以及节点和边经验的递归整合。

第3节分析早期冻结批次 `20261115_115206_evotrace_v3_revised_co4` 在快照保存时的12次搜索记录，共2,641次评价。该批次依据适应度以及个体最近生成或用作主父代的评价时刻保留个体，并允许单节点区域触发简化；分析内容包括算法淘汰与经验形成、反思开销和种群规模变化。当前代表覆盖规则的完整搜索比较与机制消融按实验方案单独组织，完成状态见实验台账。

冻结材料包括[搜索记录快照](evidence/snapshot.json)、[第180次评价处的分数](evidence/training_scores.csv)及[论文校验结果](evidence/manuscript_checks.json)。

## 重建

以Markdown和共享BibTeX为维护源，生成LaTeX及PDF：

```bash
.venv/bin/python paper/build.py
.venv/bin/python paper/validate.py
```

构建需要Pandoc、XeLaTeX、latexmk及标准TeX包（包括xeCJK、caption、placeins、longtable）。字体为Times New Roman、TeX Gyre Heros、TeX Gyre Cursor；中文另需Noto Serif CJK SC、Noto Sans CJK SC、Noto Sans Mono CJK SC。可用 `--pandoc /path/to/pandoc` 指定Pandoc，也可安装 `requirements.txt` 中的Python依赖。校验需要Poppler的 `pdfinfo` 与 `pdftotext`。

构建读取冻结材料，不刷新正在运行的实验。`main.tex` 与 `main-zh.tex` 为生成文件；修改正文时请编辑Markdown。发布材料包包含维护源、图稿、冻结材料、官方样式、生成的LaTeX与PDF；其 `SHA256.json` 记录包内文件哈希。
