# HSEvo 基线

本实现以作者源码 [datphamvn/HSEvo@f3afffab](https://github.com/datphamvn/HSEvo/tree/f3afffabc2b458fddf70a42a8678758039671297)（2026-07-06）为准，对应 AAAI 2025 论文 *HSEvo: Elevating Automatic Heuristic Design with Diversity-Driven Harmony Search and Genetic Algorithm Using LLMs*。HSEvo 在 ReEvo 的流程上改了反思方式，并加入和声搜索调参数。本地参考副本位于 `reference_code/HSEvo/`，未下载其中的实例数据文件（`.pkl`、`.npy`、`.npz`）。

## 代码来源

许可证为 MIT，随代码保留。

| 官方部件 | 本地位置 | 用法 |
| --- | --- | --- |
| `prompts/common/` 的 10 个提示 | `prompts/common/` | 逐字复制。另加两个同分提示，见下文。 |
| 任务外部知识 | `prompts/{cvrp_aco,op_aco,tsp_construct}/` | 复制 `cvrp_aco`、`op_aco`、`tsp_constructive`，与 ReEvo 的版本相同。其他任务没有外部知识。 |
| `utils/utils.py` 的代码提取、`filter_code`、`extract_to_hs` | `sampler.py` | 保留原逻辑。全局导入与参数范围的执行方式有改动，见下文。 |
| `hsevo.py` 的 `HSEvo` | `hsevo.py` | 按原方法顺序改写为串行循环。 |

## 保留的官方机制

默认参数取自 `cfg/config.yaml`。

| 机制 | 设置 |
| --- | --- |
| 初始化 | 先评价种子，再生成 30 个初始个体。第 i 个提示使用第 i mod 10 个科学家角色，系统消息与用户消息都带角色。初始提示不含外部知识，因为上游的 `long_term_reflection_str` 始终为空。 |
| 选择 | 选择池为当前种群；精英不在种群中时把精英加入。只从有效个体中随机抽 10 对，两者分数必须不同。 |
| 快速反思 | 选中的 20 个个体按分数从好到差列出，调用一次，得到 Analysis 与 Experience。 |
| 综合反思 | 用有效反思列表、无效反思列表和当前 Experience 调用一次。结果前接外部知识，供交叉与变异使用。 |
| 交叉 | 每对一个提示，含 Better/Worse code、Analysis 和综合反思。当代种群替换为交叉结果，失败个体也保留。 |
| 变异 | 对精英生成 int(10 × 0.5) = 5 个个体，加入种群。 |
| 反思记账 | 交叉与变异后精英改变，则本代 Experience 记入有效列表，否则记入无效列表。 |
| 和声搜索 | 每代最多尝试 3 次。选当前种群中未尝试过的最好个体。LLM 把其中的阈值和权重改为带默认值的参数，并给出 `parameter_ranges`。初始和声记忆 5 组，再迭代 5 次（HMCR 0.7、PAR 0.5、bandwidth 0.2）。每组参数评价一次，只把最好的一个加入种群。 |
| 失败个体 | 留在种群中，`obj = inf`。 |
| 终止 | 种群全部无效、有效个体不足 2 个或选择 1000 次仍凑不齐时，上游结束 `evolve` 并返回最好程序。本地同样结束搜索，`summary.json` 的 `termination` 写明原因。 |

## 本地适配

| 问题 | 本地规则与理由 |
| --- | --- |
| 预算 | 上游 `max_fe` 计函数评价次数，含种子、无法提取代码的回复和每组和声参数。本地预算 1000 用同一口径。反思与参数提取调用不占预算，单独记录。上游只在每代开始检查预算，最后一代会超出；本地每次评价前检查，用完即停。 |
| 和声搜索的退还 | 上游在 5 组初始和声全部失败时，从计数中减去这 5 次评价。协议规定返回的程序都计入预算，本地不退还。 |
| 全部同分 | 上游要求父代分数不同，全部同分时选择失败并终止。本地按 ReEvo 的规则兜底：只在可选个体全部同分时允许同分配对，并记录 `selection_fallback`。此时快速反思与交叉改用 `user_flash_reflection_tie.txt` 与 `crossover_tie.txt`，不标注排名与优劣。其他情况与上游相同。 |
| 全局导入 | 上游在每段代码前加入 numpy、random、math、scipy、torch 的导入。共同评价环境没有 torch，任务也不用它，本地去掉 torch 一行。 |
| 参数范围的执行 | 上游在主进程中 `exec` LLM 写的 `parameter_ranges` 代码块。本地在隔离子进程中执行，限时 10 s，并要求每项恰好两个数值。上游遇到其他形状会抛出非 `RuntimeError` 的异常，使整个运行中断；本地只把这次和声搜索记为失败。 |
| 入口函数 | 上游评价脚本按 `name`、`name_v1`、`name_v2`、`name_v3` 的顺序查找函数。本地用同一顺序，再改名为任务入口。 |
| 和声搜索的个体选择 | 上游用候选列表的下标去取种群元素。重试时因此常再次选中同一段代码。本地保留这一行为。 |
| 种子 | 上游种子来自各任务的 `seed_func.txt`。本地用任务模板函数，初始提示中改名为 `_v1`，与上游格式一致。 |
| 采样参数 | 上游 temperature 1，初始种群再加 0.3；输出上限 32768。本地使用共同模型配置，输出上限 16384，与 ReEvo、OpenEvolve 相同。 |
| 并行与故障 | 上游并发调用 LLM，并同时启动多个评价子进程。本地串行生成与评价，不改变采样分布。上游接口失败 30 次后退出程序；本地由公共接口重试，非临时错误时跳过该个体，不占预算。反思调用失败时保留上一次的记忆。 |
| 不使用的部件 | SWDI 与 CDI 多样性指标、逐次写出的提示与回复文件、tiktoken 计数。调用与事件由公共记录器保存。 |

## 恢复

检查点保存在三类边界：开始前、种子与每个初始个体评价后、每代结束后。检查点含种群、精英、两类反思列表、两段记忆和随机状态。中断于初始化时，恢复后从下一个科学家角色继续。中断于一代中间时，恢复后从这一代开头重来，随机状态与检查点一致；已消耗的预算不退还，这一代已评价的程序仍留在记录中，参与最终选择。其他边界沿用[DeltaEvolve 的公共恢复规则](../deltaevolve/README.md#实验框架适配)。

## 使用

仅检查配置，不调用模型：

```bash
uv run python -m experiments.hsevo.run --task op_aco --dry-run
```

单路运行：

```bash
uv run python -m experiments.hsevo.run \
  --task op_aco --budget 1000 --seed 0 --run-name BATCH_op_hsevo_rep1
```

六任务三路计划：

```bash
uv run python -m experiments.infra.local_queue \
  --method hsevo --suite co6 --batch BATCH --repeats 3 --slots 3 --dry-run
```

恢复使用 `experiments.infra.resume_baseline --run-dir RUN_DIR`，加 `--execute` 才继续。[离线检查](../../tests/method/test_hsevo.py)覆盖提示顺序与内容、和声搜索、预算口径、同分兜底、终止、检查点恢复，以及六任务的完整运行入口。它们只验证机制和框架行为，不证明搜索效果。
