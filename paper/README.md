# TREES研究论文

**Reflective Elimination: Accumulating Design Experience in Tree-Structured Algorithm Populations**

**反思式淘汰：在树状算法种群中积累设计经验**

方法名：**TREES（Tree-structured Reflective Evolutionary Search，树状反思式演化搜索）**。

核心主张：**用有根树维护候选启发式种群，使淘汰低分节点的过程同时成为总结和传递算法设计经验的过程。** 通过代表启发式及其最近公共祖先划分路径区域与子树区域，节点数少于 $m$ 的区域留在树中继续参与父代抽样，节点数达到 $m$ 的区域通过路径收缩与子树剪枝总结为重连边上的边经验和保留父节点上的节点经验，并在后续简化与精化、交叉、探索算子中持续复用。

## 阅读入口

| 材料 | 入口 |
| --- | --- |
| 双语PDF | [英文](trees.pdf)、[中文](trees-中文.pdf) |
| 可编辑正文 | [英文Markdown](paper.md)、[中文Markdown](paper-中文.md) |
| 定型方法说明 | [V4方法设计](../docs/EvoTrace-v4设计.md) |
| 完整源码与PDF | [TREES材料包](TREES-paper-source.zip) |
| 生成的LaTeX | [英文](main.tex)、[中文](main-zh.tex) |
| 参考文献 | [共享BibTeX](references.bib) |
| 机制图来源 | [生成记录与校验](figures/current-generation-record.json)、[生成提示](figures/current-generation-prompts.json) |
| 已确定的任务与对照方法 | [四项任务及五个外部对照](../docs/记录/2026-11-16-EvoTrace对比任务筛选.md) |

论文标题突出反思式淘汰如何在树状算法种群中积累设计经验；TREES用于指称完整搜索方法。EvoTrace仍是代码包、数据目录与实验记录的内部代号。论文PDF和材料包统一使用TREES文件名。

## 正文与附录结构

正文依次展开问题背景与相关工作局限、树状种群与反思式淘汰更新（含结构性质命题）、全树父代选择与三类搜索算子、四项组合优化任务上的搜索过程分析，以及讨论、局限与结论。六个附录各自补充一项具体内容：

- A：研究背景、代表性基于LLM的AHD方法对比表，以及与树搜索和全局经验池方法的具体区别。
- B：补充说明最近公共祖先保留、全树可行节点父代抽样、边经验与节点经验（含虚拟根经验）分工，以及 `analysis`、`code`、`design` 的输出顺序。
- C：证明区域互不相交划分、代表集与骨架不变性、更新幂等性、原有经验保留，以及在树算法节点数紧上界 $4Km-2K-m$（含紧性构造）。
- D：四项基准任务接口、评价器限制、独立测试协议、外部对照基线与早期搜索记录统计口径。
- E：AST结构距离、抽样与单次修复细节、三类算子的提示上下文结构与指令原文，以及区域反思提示词原文。
- F：从TSP搜索记录追踪边经验生成、原节点移除后的跨轮合并，以及进入后续精化提示词的具体案例。

双语PDF采用[ICML 2027官方模板](https://media.icml.cc/Conferences/ICML2027/styles/icml2027.zip)的匿名审稿模式，Letter纸张、双栏正文、作者—年份引用，参考文献之后另起单栏附录。官方样式与下载哈希保存在 `vendor/icml2027/`，不修改模板字号和页边距。

## 当前分析材料

摘要、第1–2节、第4–6节及附录A–C、E已对齐当前TREES定型实现（见 `docs/EvoTrace-v4设计.md` 与 `evotrace_v4/`）；第3节、附录D.5与附录F暂保留早期代表覆盖版批次 `20261116_165500_evotrace_v3_rep_cover_co4` 的冻结前缀统计，待12路1,000次评价正式批次 `20261118_175341_evotrace_v4_aad_co4` 及独立测试评测完成后统一更新主结果、泛化对比、消融实验与案例。早期搜索记录采用 $K=24$、$\delta=0.15$、$m=4$，12次运行共包含3,519次程序评价，训练曲线统一对齐至第240次评价；490次区域反思中有147次（$30.0\%$）继承了严格更早更新步生成的反思文本。TSP案例追踪第97次与第104次评价后的两次路径收缩及第98次生成尝试，并如实报告后续子代目标值的轻微退化。

| 材料 | 内容 |
| --- | --- |
| [当前方法快照](evidence/representative_snapshot.json) | 固定前缀哈希、配置、评价序列、树规模、区域与反思统计 |
| [真实案例](evidence/inheritance_case.json) | 所选程序、两次反思的完整输入输出、后续生成提示和结果 |
| [搜索图](figures/representative-search.pdf) | 四项任务的训练目标值与在树节点数曲线 |
| [案例图](figures/inheritance-case.pdf) | 两条TSP分支的训练目标值与反思更新时刻 |
| [论文校验](evidence/manuscript_checks.json) | 双语数学、引用、数值、图稿、官方模板与版式检查 |

外部对照基线固定为EoH、ReEvo、MCTS-AHD、RefineEvo和PathWise；完整1,000次评价跨方法对比及独立测试结果将在正式批次跑完后写入。此前的 `snapshot.json`、`experience_reuse.json` 与 `early-search` 图保留为历史分析档案，不再作为正文结果。节点数紧上界 $4Km-2K-m$ 在默认参数 $K=24,m=4$ 下为332。

## 构建与校验

Markdown和共享BibTeX是唯一维护源，LaTeX由构建脚本自动生成。无需原始搜索目录即可重绘图表、构建双语PDF并运行完整校验：

```bash
python3 paper/plot_representative_evidence.py
python3 paper/build.py
python3 paper/validate.py
```

Python依赖见 `requirements.txt`。构建需要Pandoc、XeLaTeX、latexmk及标准TeX宏包（含xeCJK、caption、placeins、longtable），校验需要Poppler的 `pdfinfo` 与 `pdftotext`。可通过 `--pandoc /path/to/pandoc` 指定Pandoc路径。字体使用Times New Roman、TeX Gyre Heros、TeX Gyre Cursor，以及中文的Noto Serif CJK SC、Noto Sans CJK SC、Noto Sans Mono CJK SC。

在保有原始搜索记录的环境中，运行 `python3 paper/collect_representative_evidence.py` 可复算统计快照与案例；运行 `python3 paper/package.py` 可重新打包源码与PDF材料包并写入 `SHA256.json`。
