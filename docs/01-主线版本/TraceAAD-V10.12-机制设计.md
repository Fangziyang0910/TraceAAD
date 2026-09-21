# TraceAAD V10.12：短形成轨迹加双档案 profile card

V10.12 以 V10.11 generic 为唯一基线，只修改生成上下文：保留 V10.11 的短形成轨迹，并在每次生成时额外加入两张档案 profile card。profile card 只包含一个已评价算法节点的 `Idea` 与 `fitness`，不包含来源算子、代码长度、复杂度或完整代码。

## 保留不变

- ESS-8 质量分配与 V10.11 的父代选择；
- 四个算子各 25%；
- Pivot 的 1:1 质量/均匀混合；
- Fuse 的 donor 完整代码；
- 当前程序、最近八条短形成轨迹、解析、一次修复、任务契约、模型和真实评价预算。

## 新增上下文

每次生成在 V10.11 上下文之后加入两张 profile card：

```text
profile card = {idea, fitness}
```

两张卡从已有档案按 `rand_ctx` 的 rank-softmax（`tau=8`）无放回抽样，排除当前父代。卡片是补充参考，不替换形成轨迹，也不改变父代分配。四个算子都使用同一规则，以便先检验“档案参考能否补充形成关系”这一单一假设。

## 机制假设

短形成轨迹保留 TSP/VRPTW 所需的连续改进关系；两张档案卡为 CVRP/OBP 提供当前谱系之外的机制材料。限制为两张卡是为了控制提示噪声、上下文长度和抽样方差。该版本不引入 task router、算子专属历史、动态温度、行为簇或新的预算控制器。

## 实验契约

五任务、三重复、每路 1000 次真实评价；与 V10.11 generic 和 rand_ctx 对照。报告训练 best、E500/E1000 frontier、parent-to-frontier 转化、有效率、提示 token、运行错误和同规模测试。若 TSP/VRPTW 同规模测试任一相对 generic 恶化超过 1%，或上下文超限/有效率明显恶化，则不增加 profile card 数量；只有至少一个档案受益任务改善且构造型任务基本保持，才考虑后续算子条件化。

实现与结果目录：`llm4ad/method/traceaad_v10_12/`、`experiments/traceaad_v10_12/`、`experiments/traceaad_v10_12/results/`。
