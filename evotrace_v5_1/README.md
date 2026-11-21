# EvoTrace V5.1

独立实现包（仅依赖 Python 标准库），对应设计文档 [docs/EvoTrace-v5.1设计.md](../docs/EvoTrace-v5.1设计.md)。

## 核心机制

1. **有界树状种群（$K=24$）**：`Init` 与 `Explore` 挂接于虚拟根 $v_0$ 开启独立谱系，`Refine` 与 `Crossover` 挂接于主父代下继承谱系；规范 AST 精确去重。
2. **路径收缩与固定容量淘汰（反思式淘汰）**：
   - **路径收缩**：无分叉单链上后代严格优于祖先时，后代取代祖先并继承原有节点经验 $E_u$（双非空时轻量合并），**不生成成功路径反思**；
   - **固定容量淘汰**：当 $|\mathcal{V}| > K$ 时移出非全局最优且出度 $\le 1$ 的最差节点（必然终止、无死锁，分叉祖先因出度 $\ge 2$ 自动保留）；**剪除实父代下的叶节点是唯一产生新反思的时机**，从代码差异中拆解未兑现想法与疑似实现缺陷写入父代 $E_P$。
3. **适应度加权父代选择**：沿用 EvoTrace V1 / ShinkaEvolve 的中位数/MAD Sigmoid 质量权重（$\lambda=10$），主父代按 $q(u)/(1+n_u)$ 抽样，跨谱系交叉供体按 $q(q)$ 抽样。
4. **算子上下文隔离**：`Refine (0.45)` 唯一消费同谱系节点经验（无成功路径叙事，退步节点附 1 步原始 diff）；`Crossover (0.25)` 跨谱系双源码无反思；`Explore (0.30)` 仅展示至多 5 条不同谱系代表的设计简述，无源码、无反思；取消 `analysis` 硬字数截断。

## 运行方式

```bash
# 独立 CLI（支持同 run 随机屏蔽开关 --reflection-mask-prob 0.5）
python -m evotrace_v5_1 --task tsp_construct --run-dir runs/v5_1_demo \
  --base-url http://localhost:8000/v1 --model Qwen3.8-27B --budget 1000

# 统一实验入口
python -m experiments.traceaad.run --method evotrace.v5.1 --task tsp_construct \
  --run-name demo --batch v51_test --scheduler-socket /path/to/scheduler.sock

# 同 run 随机屏蔽机理验证（50% Refine 展示节点经验，50% 屏蔽）
python -m experiments.traceaad.run --method evotrace.v5.1 --reflection-mask-prob 0.5 \
  --task tsp_construct --run-name demo_mask --batch v51_mask --scheduler-socket /path/to/scheduler.sock
```
