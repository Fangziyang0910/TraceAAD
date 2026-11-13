# EvoTrace机制图

本组图用内置`image_gen`生成，并逐张按V1/V2源码核对节点、箭头、保留集合与经验归属。图内使用英文，机制文档提供中文图注；所有PNG原图保存在本目录。

正式文档：[V1设计](../../EvoTrace-v1设计.md)、[V2设计](../../EvoTrace-v2设计.md)。交互查看：[机制图浏览](gallery.html)。

| 文件 | 表达重点 |
| --- | --- |
| [01-main-framework.png](01-main-framework.png) | 当前主图：一棵持续演化的种群树、局部反思简化与经验回流 |
| [01-overview.png](01-overview.png) | 原流程框总览，保留作历史版本 |
| [02-v1-selection.png](02-v1-selection.png) | V1容量选择与结构闭包 |
| [03-v2-selection.png](03-v2-selection.png) | V2一次选择与LCA补全 |
| [04-reflective-simplification.png](04-reflective-simplification.png) | 联合剪枝与路径收缩的经验归属 |
| [05-operator-context.png](05-operator-context.png) | V2三类算子的树上下文 |
| [05-operator-context-v1.png](05-operator-context-v1.png) | V1三类算子的树上下文 |
| [06-v2-history.png](06-v2-history.png) | V2评价时钟与历史统计 |

青色表示保留算法与主结构，灰色表示移除节点，蓝色表示计算模块或近期使用标记，橙色表示经验及其交接。S表示节点选择经验，V表示边变异经验。V1选择图省略虚拟根；V2选择示例单独绘制方形虚拟根r。

三类算子分别提供V1与V2版本，V2图标明Explore读取内部变异经验的扩展。当前主图以一棵连通演化树形成视觉主体，用局部放大说明经历转为经验，再将树的不同视图与经验接入生成；P/R来源与新算法的单主父代关系明确区分。路径/旁支图明确区分B原有选择经验与本轮新剪旁支X的去向。

生成提示保存在[generation-prompts.json](generation-prompts.json)，修订提示保存在[refinement-prompts.json](refinement-prompts.json)及[finalization-prompts.json](finalization-prompts.json)。[generation-record.json](generation-record.json)记录生成来源、采用轮次、实际像素尺寸、SHA256与逐图核对点。图中集合和树形是机制示例，不是性能实验数据。

主图重构提示见[main-framework-prompts.json](main-framework-prompts.json)，采用稿修订见[main-framework-refinement.json](main-framework-refinement.json)，构图取舍和机制复核见[main-framework-review.json](main-framework-review.json)。当前主图保留种群树、局部简化和经验参与后续设计这三层关系，参数细节由正文和子图展开。
