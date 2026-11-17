# TREES机制图

本目录维护当前论文方法图与历史版本图示记录。论文方法名为TREES（Tree-structured Reflective Evolutionary Search），核心更新为反思式淘汰（Reflective Elimination）；EvoTrace保留为内部目录与历史版本代号。当前代表集覆盖与成批反思树简化规则见[V3设计](../../EvoTrace-v3设计.md)。

## 当前 V3 论文图（11月16日重构）

两张图围绕“算法的优胜劣汰同时成为积累和复用算法设计经验的过程”展开。总览图呈现淘汰与反思如何衔接后续生成；第二张图进一步说明经验如何跨多次更新积累。每个示例移除区域均含4个算法节点，与当前默认简化阈值一致。图稿同步拷贝至 `paper/figures/`，在ICML双栏正文中跨栏展示。

| 图 | 论文对应文件 | 作用 |
| --- | --- | --- |
| [淘汰、经验积累与复用](12-v3-experience-cycle.png) | `paper/figures/framework.png` | 选择算法后，路径收缩与子树剪枝分别形成边经验和节点经验；更新后的种群参与预算分配并为三种算子提供上下文 |
| [连续更新中的经验积累](13-v3-experience-accumulation.png) | `paper/figures/reflective-simplification.png` | 既有边经验参与更长路径的反思，既有节点经验与后来分支的设计和结果继续整合 |

完整生成提示见[生成提示](current-generation-prompts.json)，源文件、尺寸、SHA-256与语义检查见[生成记录](current-generation-record.json)。当前图稿为PNG；原生尺寸分别为2061×763与1920×819像素。图中实线箭头表示树的演化关系；橙色标注表示设计经验，虚线箭头表示其进入生成上下文。总览底部箭头表示下一轮种群更新。

## 历史版本存档

以下文件保留早期迭代版本图示与生成记录，仅供版本追溯（其中 V1 图仍由[V1设计](../../EvoTrace-v1设计.md)引用）：

| 文件 | 适用范围 |
| --- | --- |
| [10-v3-framework.png](10-v3-framework.png) | 11月16日重构前V3总览 |
| [11-v3-reflective-simplification.png](11-v3-reflective-simplification.png) | 11月16日重构前反思简化示意 |
| [09-v3-framework.png](09-v3-framework.png) | 11月15日早期V3总览（含近期窗口留存与旧 S/V 记号，未区分代表集与纯分叉锚点及起批规模 $m$） |
| [07-framework.png](07-framework.png) | 11月14日历史V2总览 |
| [08-reflective-simplification.png](08-reflective-simplification.png) | 11月14日历史V2简化示意（单节点即触发，沿用旧 S/V 记号） |
| [01-main-framework.png](01-main-framework.png) | 11月13日V1总览 |
| [01-overview.png](01-overview.png) | 早期流程框总览 |
| [02-v1-selection.png](02-v1-selection.png) | V1容量选择与结构闭包 |
| [03-v2-selection.png](03-v2-selection.png) | 早期V2选择示意 |
| [04-reflective-simplification.png](04-reflective-simplification.png) | V1经验归属示意 |
| [05-operator-context.png](05-operator-context.png) | 早期V2上下文视图 |
| [05-operator-context-v1.png](05-operator-context-v1.png) | V1上下文视图 |
| [06-v2-history.png](06-v2-history.png) | 早期V2统计示意 |

[历史图浏览](gallery.html)保留原稿供查阅。早期生成记录见[generation-prompts.json](generation-prompts.json)、[refinement-prompts.json](refinement-prompts.json)、[finalization-prompts.json](finalization-prompts.json)、[generation-record.json](generation-record.json)、[main-framework-prompts.json](main-framework-prompts.json)、[main-framework-refinement.json](main-framework-refinement.json)及[main-framework-review.json](main-framework-review.json)。
