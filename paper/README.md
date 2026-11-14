# EvoTrace论文交付

**EvoTrace: Evolving Tree-Structured Populations through Reflective Compression**

本文以“实现的开发价值与经历的参考价值不同”为出发点，将反思描述为搜索经历的压缩，并由树状种群统一候选竞争、历史整理与经验交接。

## 阅读与编辑

| 材料 | 入口 |
| --- | --- |
| 英文PDF | [evotrace.pdf](evotrace.pdf) |
| 中文PDF | [evotrace-中文.pdf](evotrace-中文.pdf) |
| 完整材料包 | [EvoTrace-paper-source.zip](EvoTrace-paper-source.zip)（PDF、LaTeX、Markdown、图示与冻结数据） |
| 英文内容源 | [docs/paper.md](../docs/paper.md) |
| 中文内容源 | [docs/paper-中文.md](../docs/paper-中文.md) |
| 英文LaTeX | [main.tex](main.tex) |
| 中文LaTeX | [main-zh.tex](main-zh.tex) |
| 参考文献 | [references.bib](references.bib)，维护源为[共享BibTeX](../docs/paper-references.bib) |
| 当前机制规格 | [EvoTrace V2设计](../docs/EvoTrace-v2设计.md) |
| 原始分数与运行条件 | [冻结快照](evidence/snapshot.json)、[逐种子分数](evidence/training_scores.csv) |

英文稿包含摘要、引言、相关工作、完整方法、初步实验、讨论、结论，以及默认设置、结构性质证明、反思提示、复现条件和核心评价设计五个附录。中文稿保持相同的机制、公式、分数和论证结构；实际英文提示模板在英文附录C逐字收录。

## 当前证据状态

文稿已形成全文，但当前实证部分是**早期研究稿，不是完成实验的投稿终稿**。2026年11月14日12:13北京时间冻结的本地记录包含最新版12路的951次评价；共同分数位置为60次评价，各路完整目标为1000次。

- 正文只报告当前源码批次的早期训练分数、结构简化和反思调用事实。
- 最终独立测试、统一条件的基线比较，以及无历史／原始历史／反思经验的受控对照尚未完成。正文没有填入假设结果。
- 已有旧基线在采样温度、评价时限或计数口径上存在差异，不能直接拼成同条件排行榜。
- 当前版不绑定具体会场，采用匿名通用研究稿排版；投稿时按目标会场模板和页数要求适配。

实验的下一步按附录E执行。最优先的是完整搜索的独立测试与固定状态历史表示对照，它们分别决定整体算法表现与压缩经验的作用。

## 重建

直接编译已有LaTeX无需Pandoc：

```bash
cd paper
latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
latexmk -xelatex -interaction=nonstopmode -halt-on-error main-zh.tex
```

以下流程从Markdown重新生成LaTeX和PDF。

依赖：Pandoc、XeLaTeX、BibTeX、latexmk；中文字体使用Noto Serif CJK SC。Markdown是内容维护源，LaTeX和PDF由构建脚本导出。

```bash
python3 paper/build.py --pandoc /path/to/pandoc
```

已安装`pypandoc-binary`时可不传路径。构建同时更新两种语言，使用已有证据快照，不请求模型、不评价候选、不刷新实验。

```bash
# 仅在决定更新文稿观察时，显式重新冻结本地已提交结果。
python3 paper/collect_evidence.py
# 使用已安装matplotlib的Python重画数据图。
python3 paper/plot_evidence.py
```

更新快照后，须同步正文中的时间、分数和过程统计，再编译。`collect_evidence.py`只读本地运行日志，不启动、停止或改变实验。

## 图示与来源

框架图及反思简化图由生图模型生成，沿用已复核的[生成记录](../docs/figures/evotrace/current-generation-record.json)。第三幅数据图由冻结快照绘制，提供PNG和矢量PDF。图中箭头表示派生、读取或经验交接；节点与边经验的来源遵循正式机制。

历史研究记录保存当时的条件和认识，不随现行论文改写。两轮Opus意见、Carbon调用失败的记录、采纳依据和统计修正见[整合与交付记录](../docs/记录/2026-11-14-EvoTrace论文整合与交付.md)。
