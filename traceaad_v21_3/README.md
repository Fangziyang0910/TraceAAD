# TraceAAD V21.3

V21.3的Refine读取连续的形成路径、同父代各类尝试和当前完整代码。路径默认保留最近3步与更早成绩变化较大的步骤，给真实差分和评价变化；同父代尝试最多3条有效、2条失败。历史不重复卡片字段、完整代码或模型自述。Crossover使用父代代码与参考Design，`--reference-code`可额外展示参考完整源码；Explore只读任务与当前代码。

演化生成仍是单次`analysis → code → design`，分析只说明本次改变的计算。初始化与修复沿用原有提示。协议为`refine_evolution_history_v1`，旧运行使用其冻结源码恢复。

规格见[V21.3](../docs/机制设计/V21.3.md)，入口为`--method v21.3`或`v21_3`。
