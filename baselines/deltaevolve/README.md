# DeltaEvolve 论文复现

本实现依据 [ICML 2026 正式版](https://proceedings.mlr.press/v306/jiang26g.html)的 §4、Algorithm 1、Figures 5–6 和 13–14、Table 4。2026-10-16 查询作者仓库时，仅有 Coming Soon，未使用作者实现。逐项文献核对见[研究记录](../../docs/记录/2026-10-16-DeltaEvolve复现核对.md)。

## 已实现的论文机制

一次回复同时生成 SEARCH/REPLACE 补丁、L1 FROM/TO 摘要、L2 修改条目。每项 L2 包含 COMPONENT、OLD LOGIC、NEW LOGIC、HYPOTHESIS。系统重建完整子程序并评价，再绑定实际结果。没有额外的事后反思调用、数值动量、novelty 拒绝或元提示更新。

每次选取父程序、至多 3 个高分历史和 2 个多样性历史。父程序展示完整代码，历史按 Algorithm 1 展示 L1/L2。默认 3 个岛、population_size=40、archive_size=20；每 10 次返回生成结果后执行一次迁移，迁移率 0.1。模型采样参数使用公共配置，输出上限为 8192。

系统按最小化 raw fitness 排序。提示中的 FITNESS SCORE 为其相反数，避免与原论文最大化口径混淆。当前父程序显示数值 FITNESS SCORE，历史显示相对其原父程序的 improved/degraded/unchanged。HYPOTHESIS 是模型提出的理由，不是评价后验证的组件贡献。

## 补充决定

以下是本地选择，不是作者已经披露的实现事实。主要参数及 embedding 服务设置同时保存在 `run_config.json`。

| 未明确项目 | 本地规则与理由 |
| --- | --- |
| 父程序的概率与权重 | 均匀选岛。以 0.1 概率从全局 archive 最好的前 10% 均匀取父程序；0.2 概率从当前岛均匀取；0.7 概率按当前岛成绩排名的 `1/rank` 权重取。对应表 4 的三个比例，避免权重受负分和任务量纲影响。所有岛中节点均有非零选择概率。 |
| MAP-Elites 描述子 | 对完整代码做文本 embedding。单位向量经过固定种子 0 的两个单位高斯随机投影，再计算 `tanh(sqrt(d) * projection @ embedding)`；两个坐标各分为 10 箱。使用固定描述子，避免新增节点导致旧网格坐标漂移。 |
| embedding 来源 | `experiments.deltaevolve.embedding.CodeEmbedding` 复用现有 Shinka 服务客户端，默认 DashScope `text-embedding-v4`，读取 `DASHSCOPE_API_KEY`。端点与模型沿用 `SHINKA_EMBEDDING_BASE_URL`、`SHINKA_EMBEDDING_MODEL`；`SHINKA_EMBEDDING_API_KEY_ENV` 指定凭据环境变量名，默认 `DASHSCOPE_API_KEY`。仅复用服务客户端和补丁正则，不启用 ShinkaEvolve 的搜索机制。模型和网格均为本地补充。 |
| 长程序 embedding | 按最多 6000 个 UTF-8 字节分块，保持字符完整，逐块调用后按字节长度加权平均，再由网格归一化。短程序只调用一次。保留完整代码，不截掉尾部；分块 usage 累加，失败时也记录此前成功块的用量。[服务文档](https://www.alibabacloud.com/help/en/model-studio/text-embedding-synchronous-api)限制每段 8192 tokens，6000 字节是避免依赖另一个 tokenizer 的保守上限。分块聚合是本地补充，不改变其他基线的 embedding。 |
| 网格与容量 | 每岛每格保留成绩最好的程序，成绩相同取较早个体；每岛最多 40 个占用格，超限淘汰成绩最差者。全局另保留最好的 20 个程序，不要求其来自不同格。所有有效节点的原记录仍保存在检查点和事件中，历史检索候选受当前网格与 archive 限制。 |
| 高分与多样性历史 | 高分从全局 archive 选择；多样性从所有岛占用格的并集中选择，每格取最好的程序，按格坐标到父程序的平方欧氏距离从大到小排序。同距离按成绩、个体编号排序。排除父程序和已选高分节点，不重复展示。 |
| 新旧窗口 | 默认 `w=10`，单位为返回生成结果的次数。严格使用 `node.iteration > next_iteration - w`；新节点展示 L2，否则展示 L1。窗口值来自本地选择，10 与论文迁移周期同量级。格式失败也推进生成次数，接口故障不推进。 |
| 正文与伪代码冲突 | 历史层级遵循 Algorithm 1，不采用“所有选中节点都显示 L2”的另一处文字表述。历史按高分组、再按多样性组展示，组内不另按时间排序。 |
| 反馈冲突 | 历史采用式 (7) 明确描述的定性变化，不复制 Figures 13–14 中的逐实例数值反馈；父程序保留模板要求的评价反馈，用负 raw fitness 表示。完整原始成绩与逐实例评分保留在事件中。 |
| 改善判断 | 子程序与原父程序 raw fitness 严格比较，无额外容差；等分为 unchanged。多项修改共享一次程序评价，不拆分归因。 |
| 初始化 | 评价任务自带有效模板一次，所有岛共享该种子，不重复评价。种子没有 delta，不进入历史 delta 材料，但可以作为父程序。种子无效则停止，不调用模型伪造初始经验。 |
| 无效与退化候选 | 缺失日志、字段为空或重复、补丁不匹配或歧义、语法错误均拒绝。有效且退化的子程序仍可参加网格竞争和归档；执行失败程序仅记录，不作为父程序或历史。没有独立失败经验池。 |
| 日志格式兼容 | 提示保留附录 C 格式。解析接受行首 1–6 个 `#`、标题空白和 CRLF，统一为同一标记；真实模型出现过 `###DELTA-SUMMARY-START`。只兼容展示格式，仍要求唯一、完整且按 SUMMARY→PLAN-DETAILS 排列的块，以及非空的必需字段。 |
| 补丁与接口 | 依次应用所有补丁，每段 SEARCH 必须精确匹配唯一非空区域。必须保留入口名称和参数签名。完整模块的 imports、辅助函数、常量与前后初始化代码随程序保存；使用 AST 规范化并编译检查，不使用可能截掉模块内容的补全截取器。任务固定求解框架继续由评价器持有。 |
| 迁移 | 同步截取各岛最好的 `ceil(当前岛大小 × 0.1)` 个节点，复制至下一个岛，构成环；目标仍按格内最优与容量接纳。源岛保留原节点。不修改迁移节点的原父节点与出生时间。 |
| 长度 | 不将正文“通常 20–40 tokens”变成硬截断，保留模型实际生成的 L1/L2。没有额外压缩调用。 |
| 失败与重试 | 每个返回结果占一次预算，包括解析失败、重复、退化、执行失败。解析失败不计入已完成的搜索评价次数；最终重评另行累加。接口故障不占生成预算，其实际调用仍记录；调用重试复用公共客户端规则。有效回复之后不额外生成修复回复；下一次正常采样继续使用剩余预算。按现有协议不对失败程序重复评价 3 次。 |
| embedding 故障 | 记录错误并停止，不退化为随机或词法多样性。评价后、embedding 前保存检查点；恢复时仅重试 embedding，不重复生成和评价。embedding 返回的 usage 与耗时作为独立事件保存，不混入生成 token 汇总。服务内部重试不逐请求展开；事件记录整次客户端调用及其最终返回用量。 |
| 评价中断 | 种子和成功解析的生成程序均在评价前保存待评价状态。恢复完成该次评价，不重新生成，也不再次扣预算。已消耗但尚未提交的生成仍推进历史年龄，避免恢复后窗口变年轻。 |

## 实验框架适配

- 默认使用本项目后端与单一生成模型。指定 `--secondary-model` 后，在同一端点按 0.8/0.2 随机选择主模型与副模型；没有自动替换成论文模型。模型和实际概率写入配置。若需论文模型组合，应显式设置主模型与副模型。
- 默认预算为共享的 1000 次尝试，包括一次种子评价；论文默认是 100 轮演化。`--budget 101` 对应一次种子加 100 次返回生成结果。
- 任务、数据、实例评价与资源约束沿用[共同协议](../../experiments/PROTOCOL.md)。实例并发使用 `--eval-workers`；不复制论文的 4 路程序并行评价，以保持现有调度与提交边界。
- 搜索结束后直接固定训练成绩最好的有效个体，不增加训练重评或重新选择。独立测试、可视化和同步读取标准结果文件。新增方法不修改既有实验结果。
- 完整检查点保存数据库、原始 delta、待评价程序、待完成 embedding 节点、生成次数、投影与随机状态。中断于未提交步骤时，沿用公共恢复规则：已消耗预算不返还，回到最后检查点；不承诺模型服务请求的精确重放。评价已返回但还未提交检查点的极短窗口中断时，该评价可能重做。

## 使用

仅检查配置，不请求模型或 embedding，也不创建运行目录：

```bash
uv run python -m experiments.deltaevolve.run --task tsp_construct --dry-run
```

单路运行；embedding 凭据沿用项目环境配置：

```bash
uv run python -m experiments.deltaevolve.run \
  --task tsp_construct --budget 1000 --seed 0 --run-name BATCH_tsp_deltaevolve_rep1
```

使用已部署的本地 OpenAI 兼容 embedding 服务时，显式设置端点、模型与凭据来源。下例对应 server3 的 FP8 服务，凭据从已有 `SERVER3_API_KEY` 读取。服务使用 vLLM 0.31.0 的 `fp8_per_tensor` 在线量化，原始 BF16 权重文件保持原样：

```bash
SHINKA_EMBEDDING_BASE_URL=http://127.0.0.1:8002/v1 \
SHINKA_EMBEDDING_MODEL=qwen3-embedding-4b-fp8 \
SHINKA_EMBEDDING_API_KEY_ENV=SERVER3_API_KEY \
uv run python -m experiments.deltaevolve.run \
  --task tsp_construct --budget 1000 --seed 0 --run-name BATCH_tsp_deltaevolve_rep1
```

代码按对称的文档表示处理，不添加检索 query 指令。更换 embedding 模型会改变网格位置，应建立新批次，不在已有检查点恢复时换模型。

六任务三路计划：

```bash
uv run python -m experiments.infra.local_queue \
  --method deltaevolve --suite co6 --batch BATCH --repeats 3 --slots 3 --dry-run
```

恢复使用 `experiments.infra.resume_baseline --run-dir RUN_DIR`，加 `--execute` 才继续。测试覆盖补丁与完整模块保存、历史披露、网格与迁移、预算、调用费用、评价中断、embedding 失败的恢复，以及六任务的完整运行入口。它们只验证机制和框架行为，不证明搜索效果或复现论文数值。

本轮完整检查与真实服务验证见[2026-10-17 记录](../../docs/记录/2026-10-17-DeltaEvolve实现检查.md)。
