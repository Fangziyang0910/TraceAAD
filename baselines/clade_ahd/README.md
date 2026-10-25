# Clade-AHD 基线

本实现把作者的 Clade-AHD 搜索机制接到 TraceAAD 的任务、模型接口和生成预算上。方法名为 `clade_ahd`，与已有 `mcts_ahd` 独立运行。

来源：[Mriya0306/Clade-AHD@ffd566b70070331dbd2759a2cb828543e43da2ac](https://github.com/Mriya0306/Clade-AHD/tree/ffd566b70070331dbd2759a2cb828543e43da2ac)。论文为 [arXiv:2602.00549](https://arxiv.org/abs/2602.00549)。本地以这份公开代码的实际执行路径为准，未声称该提交复现了论文 v2 的全部结果。上游没有独立许可证文件；原 README 的研究用途声明保存在 [UPSTREAM_README.md](UPSTREAM_README.md)。

## 保留的机制

- `upstream/bayesian_node.py`、`score_normalizer.py`、`pop_greedy.py`、`prob_rank.py` 原样复制。
- `upstream/mcts.py` 保留子树分数聚合、Thompson Sampling、退火、衰减回传和冻结逻辑。仅补上嵌套 `freeze_subtree` 中缺失的 `nonlocal frozen_count`，避免冻结后代时抛出 `UnboundLocalError`。
- `upstream/evolution.py` 提取作者的提示构造、输入检查、回复解析和描述重写方法，方法体不变。模型传输与任务契约由本地提供。任务契约复用 MCTS-AHD 的适配器，实际提示和解析器使用 Clade-AHD 自己的实现。
- 初始化先 `i1`，再通过 `e1` 生成其余根节点。根部渐进扩展使用跨子树 `e1`，非根节点使用 `e2`。叶节点执行 `e2, m1, m1, m2, m2, s1`，单节点路径跳过 `s1`。
- 保留按目标值去重的精英集合、按算法描述去重并按目标值升序排序的 `s1` 路径。`e1` 的父代数从 2–5 中随机取值，再限制到已有参考数量。

## 实际默认值与上游细节

| 设置 | 本地默认及来源 |
| --- | --- |
| 初始化 / 精英集合 | 4 / 10，来自 `cfg/config.yaml`。 |
| 节点先验 | Beta(1, 1)。 |
| 子树伪评价数 | 10。 |
| 回传衰减 | 0.8，采用 YAML 的值；README 示例为 0.9。 |
| 退火 | 开启，指数 1.0。 |
| 冻结 | 开启，阈值 0.1，最少访问 10 次；README 示例阈值为 0.3。 |
| 渐进扩展指数 / 选择深度阈值 | 0.5 / 10。 |
| 归一化 | 普通 min–max。作者构造函数创建 adaptive normalizer，但 `run()` 随后替换为 `mcts.normalizer`。 |

作者主循环先调用新节点的 `update_belief(score)`，再调用从该节点自身开始的 `backpropagate`，因此新节点重复接收同一份证据。本地保留这一行为。初始化与叶批次还先设 `visits=1`，所以新节点随后为 3 次访问；渐进扩展的新节点为 2 次访问。根子树列表中的重复自身引用也保留，因其会影响后续 `e1` 采样。

## 本地适配

共同条件见[实验协议](../../experiments/PROTOCOL.md)。原始 `fitness` 越小越好；归一化仅用于内部信念更新。树和精英集合保留上游五位小数的目标值，公共记录与最终选择保留评价器原始成绩。

每个返回的生成回复消耗一次预算，包括解析重试、重复程序和失败程序。描述重写及没有返回回复的接口失败单独记录，不消耗生成预算。最后一个生成回复仍会完成描述重写和评价。退火进度使用“已返回生成数 / 总预算”；这是统一生成预算下对上游 `eval_times / fe_max` 的适配。

同时保留实际评价次数 `_evaluation_count` 与上游混合计数 `_upstream_evals`。后者在初始化及渐进扩展的每次评价尝试时增加，在叶批次的每个接纳节点上增加。冻结仍在批次结束且该计数整除 5 时检查。三种计数分开保存，不把上游计数当成实际评价成本。

作者默认并行执行叶节点算子，并按请求完成顺序更新树。本地串行请求，各槽位都读取同一份批次开始时的树和精英集合，批次结束后按算子顺序提交结果。保留同批候选互不可见的条件，但不声称与原并发调度产生相同随机轨迹。没有其他父代可供 `e2` 采样时跳过该槽位，避免上游对空列表反复重试。

检查点保存整棵树的父子关系、子树采样列表、信念、访问次数、冻结状态、归一化历史、精英集合、初始化进度、计数和 Python/NumPy 随机状态，不保存模型客户端。完整初始化节点与搜索轮次边界可以恢复相同轨迹。轮次中断时，从上一个完整边界重启；已返回回复的预算和日志中的候选保留，未提交的树更新不重放。最终选择仍从全部已记录的有效候选中取训练最优。

## 运行

在仓库根目录先检查配置：

```bash
uv run python -m experiments.clade_ahd.run --task tsp_construct --budget 20 --dry-run
```

去掉 `--dry-run` 执行一路搜索。六任务正式计划使用现有队列：

```bash
uv run python -m experiments.infra.local_queue \
  --method clade_ahd --suite co6 --batch BATCH_NAME --slots 3 --dry-run
```

结果写入 `experiments_result/clade_ahd/<task>/<run_name>/`。恢复与独立测试使用[公共入口](../../experiments/README.md)。新增离线测试见 [test_clade_ahd.py](../../tests/method/test_clade_ahd.py)。
