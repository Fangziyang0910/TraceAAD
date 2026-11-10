# TraceAAD V21.3

V21.3只在Refine中默认使用轨迹：当前完整代码，加最多3条有效、2条失败的同父代Refine差分与结果，不给祖先或历史Design/analysis。Crossover使用父代代码与参考Design，`--reference-code`可额外展示参考完整源码；Explore只读任务与当前代码。

演化生成仍是单次`analysis → code → design`，分析只说明本次改变的计算。初始化与修复沿用原有提示。协议为`refine_local_evidence_v1`，旧运行使用其冻结源码恢复。

规格见[V21.3](../docs/机制设计/V21.3.md)，入口为`--method v21.3`或`v21_3`。
