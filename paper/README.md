# TREES研究论文

**Reflective Elimination: Accumulating Design Experience in Tree-Structured Algorithm Populations**

**反思式淘汰：在树状算法种群中积累设计经验**

方法名：**TREES（Tree-structured Reflective Evolutionary Search，树状反思式演化搜索）**。

核心主张：**树状种群使算法的优胜劣汰，同时成为积累和复用算法设计经验的过程。** 核心机制称为反思式淘汰（Reflective Elimination）：路径收缩与子树剪枝组织反思范围，将经验留在保留算法及其演化关系上。后续更新继承已有经验，生成算子按设计任务读取相应内容。

## 阅读入口

| 材料 | 入口 |
| --- | --- |
| 双语PDF | [英文](trees.pdf)、[中文](trees-中文.pdf) |
| 可编辑正文 | [英文Markdown](paper.md)、[中文Markdown](paper-中文.md) |
| V3 机制 | [V3实现设计](../docs/EvoTrace-v3设计.md) |
| 完整源码与PDF | [TREES材料包](TREES-paper-source.zip) |
| 生成的LaTeX | [英文](main.tex)、[中文](main-zh.tex) |
| 参考文献 | [共享BibTeX](references.bib) |
| 机制图来源 | [生成记录与校验](figures/current-generation-record.json)、[生成提示](figures/current-generation-prompts.json) |
| 已确定的任务与对照方法 | [四项任务及五个外部对照](../docs/记录/2026-11-16-EvoTrace对比任务筛选.md) |

论文标题突出反思式淘汰如何在树状算法种群中积累设计经验；TREES用于指称完整搜索方法。EvoTrace仍是运行包、数据目录与历史记录的内部代号。论文PDF和材料包统一使用TREES文件名。

## 正文与附录

正文按“演化历史为何需要组织—树如何连接淘汰与经验形成—同一种群如何支持继续搜索—结构性质和实际记录”展开。预算分配与生成算子是该表示的用途。附录各自补充一个明确问题：

- A：研究背景与相关方法，区分程序搜索、谱系、规划与分支记忆。
- B：进一步解释分叉祖先、两类经验及其搜索用途。
- C：证明区域划分、代表不变性、幂等性和算法节点数上界。
- D：任务接口、数据、评价协议、比较口径和记录分析方法。
- E：实现规则、上下文窗口、生成指令及反思提示。
- F：从真实TSP分支追踪经验形成、继承及进入后续生成的过程。

双语PDF采用[ICML 2027官方模板](https://media.icml.cc/Conferences/ICML2027/styles/icml2027.zip)的匿名审稿模式，Letter纸张、双栏正文、作者—年份引用。参考文献之后另起单栏附录。官方样式与下载哈希保存在 `vendor/icml2027/`，不修改模板字号和页边距。

## 当前分析材料

第3节与附录F的数据来自代表覆盖版批次 `20261116_165500_evotrace_v3_rep_cover_co4`。该批次使用2026-11-17减法前的反思提示（含代码差分与比较目录）和生成上下文；第2节与附录E当前描述减法版（`1e6552b3`）机制，精简版批次 `evotrace_v3_lean_co4` 已停止作废，正文细节与实验数据待当前事实反思版批次 `20261117_153134_evotrace_v3_factual_co4` 推进后统一更新。参数为 $K=24$、$\delta=0.15$、$m=4$。12次运行的固定前缀共3,519次程序评价，训练轨迹对齐到240次评价。490次反思中有147次纳入严格更早更新形成的经验。TSP案例追踪第97与104次评价后的更新，以及第98次尝试的实际生成提示；后续子代略有退化，正文和附录均如实报告。

| 材料 | 内容 |
| --- | --- |
| [当前方法快照](evidence/representative_snapshot.json) | 固定前缀哈希、配置、评价序列、树规模、区域与反思统计 |
| [真实案例](evidence/inheritance_case.json) | 所选程序、两次反思的完整输入输出、后续生成提示和结果 |
| [搜索图](figures/representative-search.pdf) | 四项任务的训练目标与树规模 |
| [案例图](figures/inheritance-case.pdf) | 两条实际父子路径、训练分数与反思时刻 |
| [论文校验](evidence/manuscript_checks.json) | 双语数学、引用、数值、图稿、官方模板与版式检查 |

当前稿件报告训练前缀与机制过程。对照方法已确定为EoH、ReEvo、MCTS-AHD、RefineEvo和PathWise；完整跨方法排名及独立测试结果尚未写入论文。此前的 `snapshot.json`、`experience_reuse.json` 与 `early-search` 图保留为旧版本分析档案，未改写，也不再作为正文结果。结构上界 $4Km-2K-m$ 约束维护中的算法节点数，默认值为332。

## 重建与复算

Markdown和共享BibTeX是维护源；LaTeX由构建脚本生成。无需原始搜索目录即可重绘图表、构建和校验论文：

```bash
python3 paper/plot_representative_evidence.py
python3 paper/build.py
python3 paper/validate.py
```

Python依赖见 `requirements.txt`。构建还需要Pandoc、XeLaTeX、latexmk和标准TeX包（含xeCJK、caption、placeins、longtable），校验需要Poppler的 `pdfinfo` 与 `pdftotext`。可通过 `--pandoc /path/to/pandoc` 指定Pandoc。字体为Times New Roman、TeX Gyre Heros、TeX Gyre Cursor，以及中文的Noto Serif CJK SC、Noto Sans CJK SC、Noto Sans Mono CJK SC。

在保有原始搜索记录的仓库中，运行 `python3 paper/collect_representative_evidence.py` 可复算当前统计与案例。已有快照时，脚本只读取并校验记录的固定前缀；运行继续追加数据也不会改变分析范围。脚本不调用评价器或模型。运行 `python3 paper/package.py` 可重新生成材料包，其中 `SHA256.json` 记录全部文件哈希。
