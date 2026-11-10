# TraceAAD V21.1

V21.1在当前[Base](../traceaad_base/)上加入[机制2](../docs/机制设计/机制2-轨迹引导的上下文与算子.md)：轨迹上下文、算子任务与`analysis → code → design`输出。父代、算子和交叉参考选择直接继承Base，没有额外的计算结构去重。

初始化和修复复用当前Base公共实现，轨迹材料保留机制2的规范化代码与实际修改。机制说明见[V21.1](../docs/机制设计/V21.1.md)，入口为`--method v21.1`或`v21_1`，运行说明见[实验入口](../experiments/traceaad/README.md)。旧V21结果按各自冻结源码解释。
