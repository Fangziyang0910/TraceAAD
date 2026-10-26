# TraceAAD V13

V13 实现[设计轨迹与算法变换方案](../../docs/TraceAAD：从设计轨迹中学习可复用的算法变换.md)的搜索闭环。入口独立于原版 TraceAAD，评价器、任务说明、结果格式、最终选择和资源调度沿用仓库公共实现。

## 一轮搜索

1. 按近期的实际改进率分配路线预算，在路线内按训练成绩抽取父代。
2. 从共享谱系 DAG 读取父代的压缩摘要、近期差分、旁支结果和动态试错清单。交叉合并两条祖先路径，共同祖先只出现一次。
3. 选择 Open、Transform 或 Crossover。Open 保留最低机会；Transform 先排除已用操作、未满足前提及相关代码尚未改变的谱系失败。
4. Transform 检索候选规格，在**实际接收父代**上匹配计算角色，选择一个匹配，输出局部编辑块。代码检查保证编辑位于绑定区间且不重叠；独立模型调用核对实际差分是否满足操作、依赖、参数和保留条件。
5. 运行训练评价，并生成实际修改摘要与新节点的谱系摘要。所有生成程序都保存，包括未通过实施核对的程序。
6. 从有改进的 Open/Crossover 修改或跨路线重复修改触发归纳。等价操作附加证据；孤立负例不能创建新变换。一个有改进的来源样本就能提出候选操作，不要求基本操作新颖，也不要求已证明跨程序有效。来源锚点直接从父子差分提取，模型只负责抽象规格；接收父代仍须独立匹配、绑定。
7. 按新使用记录修订适用条件、拆分规格或组合操作。发现操作前提需要“正例已用、至少一个反例未用”的观察对照。旧版本与负例始终保留。

实现位置：

| 机制 | 代码 |
| --- | --- |
| DAG、局部编辑约束、谱系过滤、采样权重 | [knowledge.py](knowledge.py) |
| 模型输入与输出约定、压缩轨迹视图 | [prompts.py](prompts.py) |
| 搜索、变换库更新、阶段恢复和记账 | [search.py](search.py) |
| 默认策略与校验 | [config.py](config.py) |
| 单路与批次入口 | [实验入口](../../experiments/traceaad_v13/) |

## 实验前固定的具体选择

设计文档没有固定数值，V13 使用下列可记录的初始设置。这些是待实验验证的策略，不是已证明最优的超参数。

| 项目 | 默认值或规则 |
| --- | --- |
| 有效根程序 | 5；初始化最多 16 次尝试，且计入总预算 |
| 路线 | 模型按计算思路命名，复用已有类别，最多 8 条 |
| 路线保底 | 预算份额的 25% 均分给已有有效路线；其余按近期收益分配 |
| Open 保底 | 35%；概率混合与累计尝试配额共同实现 |
| 近期统计 | 最近 20 条相关记录；改进率用 `(1 + 改进次数) / (2 + 尝试次数)` 平滑 |
| 父代选择 | 路线内复用公共搜索器的温度与成绩采样规则 |
| 检索 | 计算词重合度排序，最多 8 项；保留一个随机探索位置，再由模型匹配 |
| 变换选择 | 同路线有效实施的成功率；不足时用全库记录，乘以降级权重 |
| 历史细节 | 最近 8 个祖先节点；摘要每节点生成一次，试错清单实时汇总 |
| 变换库 | 最多 64 项；满库继续使用、修订和附加证据，不删除旧项 |
| 反馈修订 | 每积累 4 条新记录触发一次；各类结果最多展示 2 个实际程序例子 |
| 上下文 / 输出 | 24000 / 8192 tokens；超限先移除可选上下文，必要程序上下文仍超限则记录该阶段失败 |
| 最终程序 | 训练集上最好有效程序；独立测试不参与搜索或挑选 |

路线与生成方式的预算分配把每次消耗的机会都计入分母。变换自身的性能统计则只使用正确实施后的评价，避免把无法匹配或实施出错解释为操作的性能无效。

规格修订是有版本的语义约束。局部编辑边界由代码确定，操作含义及依赖是否成立由模型核对。模型核对不等于形式化证明。初版复用现有模型池完成各阶段，没有另设“小模型摘要器”。继承配置中的 `operators`、`attempts_shown`、`progress_shown` 是旧版字段，V13 不使用；三种方式的实际概率写入每次尝试。

规格字段的内部格式统一为文本。模型用非空文本列表表达依赖、参数或检查项时，按行合并，保留内容；缺字段、空字段及未知嵌套结构仍拒绝。修改类别标识具体编辑操作，不能用整个算法家族代替。

## 结果分类与预算

| 结果 | 性能成功率 | 后续处理 |
| --- | --- | --- |
| `not_applicable`：当前父代无匹配 | 不计入 | 留下结构不符记录，供修订条件 |
| `implementation_failed`：局部编辑、核对或执行正确性失败 | 不计入 | 留下诊断，不据此封禁该变换 |
| `no_improvement`：实施后的有效程序无改进，或计算超限 | 计入失败 | 记录接收背景；相关代码未变时过滤谱系重复尝试 |
| `improved`：实施后的训练成绩改善 | 计入成功 | 留下正例和谱系来源 |

运行错误、接口不合法和输出不合法归为实施失败；超时保留为满足计算约束失败的结果。根初始化及无法交付代码的开放生成另记 `root`、`delivery_failed`。程序的原始状态、错误类别和评价记录仍保留。

`--budget` 是**候选尝试上限**，遵循当前[实验协议](../../experiments/PROTOCOL.md)，包括初始化、不匹配和失败编辑，不能称为相同有效评价次数。`--max-model-calls` 独立限制全部模型请求，默认 `8 × budget`。摘要、归纳、匹配、实施、核对、修订、失败请求及中断的不确定请求都占请求预算。任何一个上限触发后，冻结已知最好有效程序。

每条模型请求保留接口 usage。摘要中的 `measured_input_tokens`、`measured_output_tokens` 只累加接口已提供的 token；`missing_usage` 和 `uncertain_calls` 单独报告。公共结果字段 `input_tokens`、`output_tokens` 延用原有缺失 usage 估算口径。硬上限限制的是请求数，token 总量是观察量。相同候选预算不能解释为相同模型计算量；后续可以按已有日志比较公共调用数或 token 前缀。

## 运行与恢复

以下命令在仓库根目录运行。单任务预览不调用模型：

```bash
.venv/bin/python -m experiments.traceaad_v13.run \
  --task tsp_construct --backend server3 --budget 1000 --dry-run
```

server4 的共享资源池使用相同 socket 分配 CPU 和任意空闲模型端点：

```bash
source /home/fzy/code/LLM4AD/TraceAAD/.env.server4
.venv/bin/python -m experiments.traceaad_v13.run \
  --task tsp_construct --backend server3 --seed 0 --repeat 1 \
  --run-name v13_tsp_rep1 --budget 1000 --max-model-calls 8000 \
  --eval-workers 8 --scheduler-socket /tmp/traceaad-1003/scheduler.sock
```

恢复使用**同一源码快照、原目录、相同参数**再次执行原命令。每次请求前保留预算槽位，响应到达后先提交原文，再推进阶段。已提交响应不会再次调用模型；收到响应前中断则保留不确定槽位，后续重试另计一次。网络故障保存 `service_unavailable` 状态后退出，恢复会继续原阶段。

待评价的程序先保存，再执行评价；评价完成写入独立回执，然后提交搜索状态。已落盘的评价回执可复用。如果进程在评价中、或回执落盘前被终止，未完成的评价可能重做。文件锁防止同一路同时运行。相同运行目录不接受不同版本或配置。

搜索结束后进行独立测试：

```bash
.venv/bin/python -m experiments.infra.evaluate \
  experiments_result/traceaad_v13/tsp_construct/v13_tsp_rep1 --primary
```

六任务、三次重复的批次预览：

```bash
.venv/bin/python -m experiments.traceaad_v13.launch_host \
  --batch v13_co6 --suite co6 --repeats 3 --budget 1000 \
  --max-model-calls 8000 --backends server3 --eval-workers 8 \
  --scheduler-socket /tmp/traceaad-1003/scheduler.sock
```

确认预览后，在同一命令末尾加 `--launch` 才启动。批次入口只启动搜索；各路冻结后使用上述公共测试入口评价。

## 产物与迁移

标准 `run_config.json`、`events.jsonl`、`programs.jsonl`、`calls.jsonl.gz`、`resume.json`、`summary.json`、`selection.json` 和 `best_program.py` 与公共读取器兼容。额外的 `evaluation_receipts/` 保留可复用评价回执。

`resume.json` 的 `state.v13` 包含当前阶段、变换库及版本、全部尝试、路线、请求计数和未完成工作。每次尝试有父代、参考父代、变换、绑定、结果及阶段错误。实际差分可以从父子代码精确重建。

`transformations.json` 是可导入的变换库，附带使用记录和实际来源代码。新目录启动时传入 `--library-path /absolute/path/transformations.json`，保留库而重新生成程序种群；旧路线编号清空、旧使用记录标记为 external，不会把旧程序编号误当作新谱系。当前仅允许**同任务**导入。若要检验完全相同的固定初始父代，使用已有的[迁移实验入口](../../experiments/transfer/README.md)；V13 正式搜索的初始化由模型生成，单独相同随机种子不保证模型生成相同程序。

## 检查

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q \
  tests/method/test_v13.py tests/experiments/test_v13_entry.py
```

测试覆盖完整阶段链、DAG 与晚到旁支失败、受限编辑、实施失败记账、预算截断、已提交响应恢复、评价恢复、规格版本/拆分/组合/前提、库导入、六任务入口、结果读取与预算前缀选择。真实模型试跑的证据与正式启动准备单独保存在 `experiments_result/traceaad_v13/preflight/`。
