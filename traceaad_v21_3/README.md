# TraceAAD V21.3

V21.3保留三种轨迹用途：Refine读当前算法的形成路径与同起点尝试，Crossover读参考的形成与历史接入，Explore读既有方向及其后续发展。

材料使用简单有界的最近窗口，保留真实diff、评价结果和必要Design，不传递历史analysis或原因标签。Explore给出起点与较好后代的设计和实际算子路径，不把开发统计当作思想空间覆盖或独立贡献。

演化生成保持单次`analysis → code → design`。`--reference-code`额外给参考完整平铺源码；形成diff本身也可能包含参考代码。初始化与修复沿用共同模板，协议为`operator_trajectory_context_v1`。V22.2与V24.1的生成侧继承本包。

规格见[V21.3](../docs/机制设计/V21.3.md)，入口为`--method v21.3`或`v21_3`。
