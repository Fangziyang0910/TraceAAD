# EvoTrace机制图

## 当前V2

两张图由内置`image_gen`生成，按11月14日实现核对结构边、经验归属和读取关系，已接入中英文论文与V2正式设计。

| 图 | 作用 |
| --- | --- |
| [整体框架](07-framework.png) | 种群生长、反思简化与后续设计在同一棵树上循环 |
| [动态反思简化](08-reflective-simplification.png) | 同一保留集合下的区域划分、S/V交接与旧经验合并 |

完整提示与两次框架修订见[生成提示](current-generation-prompts.json)，来源、尺寸、哈希与核对点见[生成记录](current-generation-record.json)。两张PNG均为1752×897像素。图中的树形是机制示例。

## 历史版本

以下文件保留11月13日的设计与生成记录，V1仍引用对应机制图。当前规则见[V2正式设计](../../EvoTrace-v2设计.md)。

| 文件 | 适用范围 |
| --- | --- |
| [01-main-framework.png](01-main-framework.png) | 历史总览；局部示例把本轮旁支X交给S_A，不符合当前V2的并轨归属 |
| [01-overview.png](01-overview.png) | 早期流程框总览 |
| [02-v1-selection.png](02-v1-selection.png) | V1容量选择与结构闭包 |
| [03-v2-selection.png](03-v2-selection.png) | 早期V2选择示意；不作为当前时钟与代表规则的规格 |
| [04-reflective-simplification.png](04-reflective-simplification.png) | V1经验归属；B的本轮旁支X进入S_A，区别于当前V2进入V_AD |
| [05-operator-context.png](05-operator-context.png) | 11月14日修复前的V2上下文；未区分跨路线局部视图 |
| [05-operator-context-v1.png](05-operator-context-v1.png) | V1上下文视图 |
| [06-v2-history.png](06-v2-history.png) | 早期V2使用与历史统计，包含已撤回的统计口径 |

[历史图浏览](gallery.html)保留原稿供版本追溯。V1图仍由[V1设计](../../EvoTrace-v1设计.md)引用；当前V2采用上方两张新图。

生成依据见[generation-prompts.json](generation-prompts.json)、[refinement-prompts.json](refinement-prompts.json)、[finalization-prompts.json](finalization-prompts.json)与[generation-record.json](generation-record.json)。原总览的提示与核对见[main-framework-prompts.json](main-framework-prompts.json)、[main-framework-refinement.json](main-framework-refinement.json)及[main-framework-review.json](main-framework-review.json)。这些核对记录只对应生成当时的版本。
