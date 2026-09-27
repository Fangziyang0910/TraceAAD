# TraceAAD V10.14-2

这是 V10.14 的联合减法修订。依据见[诊断与证据边界](../../docs/03-机制探索与验证/2026-09-27-V10.14-搜索停滞与机制诊断.md)，代码在 [`traceaad/v10_14_2/`](../../traceaad/v10_14_2/)。原 V10.14 代码和结果保留；新版禁止混入旧目录续跑。

## 冻结的默认协议

- 父代：所有有效可执行、不同实际源码的候选，包括同分和退步候选。源码重复形成的锚点仍存档，以最早锚点代表该源码竞争，共享父代使用次数，不复制探索权重。
- 选择：历史 V9.14（`b5bd6bbf`）的 `fitness + 1/sqrt(count+1)`，没有归一化。最大者胜出，同分按次数少、ID 早。新实现的 count 计每次实际付费候选尝试作为父代的使用，包含 bootstrap、无效、重复和 Repair；根提案无父代不计。次数与候选预算在外部调用前一起持久化。记录原始质量范围和 bonus，不承诺每个新子代有一轮预算。
- 动作：可用的 Refine / Tune / Pivot / Transfer 等概率。Tune 需要显式数值赋值；Transfer 需要另一份有效源码，donor 在全异源码档案均匀抽取。Pivot 可从普通父代出发。`sampled_action` 与 `executed_action`、`repair_of` 分开；若 donor 超出证据额度，实际回退到 Refine 并记录 `reference_fallback`，不冒充一次成功 Transfer。
- 默认关闭 `online_revalidation`、`fixed_three_step_commitment` 和 `behavior_eligibility_gate`。区域与描述继续归档，默认不决定父代／参考资格；全局最好与最终候选单独保留。
- 初始化仍为 8 次 hybrid 提案（前 4 独立、后 4 参考此前有效根），每个有效唯一根最多一次 Refine bootstrap。小预算缩减提案数，失败与重复不免费补根。
- Idea：最多 500 个**实际 tokenizer token**，不要求填满；支持 `Idea:`、`**Idea:**`、`**Idea**:` 多行。`idea` 保存完整原文，`idea_metadata` 保存显示文本、原始／显示 token 数和截断标记。说明缺失不拒绝完整代码；tokenizer 元数据失效也不把有效代码判成算法失败。生产禁止字符估算客户端，thinking 保持 `False`。
- 上下文：当前程序意图与形成事件包含 Idea、真实 diff、旧／新分数及有限探针反馈。保持 4,000 token / 6 事件 / 深度 6 的证据上限，记录入选与排除 ID；不宣称所有历史都进入模型。原始完整响应、源码和 diff 不因展示预算被覆盖。
- 完整交付：完整单块保留 imports、常量、类、helper、主函数；明确 `Final implementation:` / `Final code:` 标记优先，否则使用最后一个含目标函数的完整候选块，记录从 0 起的块索引。截断／不闭合块仍拒绝。
- 确定性分块只接受每块前明确的 `Candidate: name; Part 1/N` … `Part N/N`，同名、顺序完整、无冲突和未解依赖。其他分块不盲拼；主函数引用一个只出现在另一草稿的 helper 时不会自动猜补。模板依赖按缺失全局名称闭包补齐，保留提交的新定义，多函数模板需显式 `# Target: function_name`。任务模板没有的名称保持错误，进入正常评价和修复。

## 评价与诊断

任务评分的数据、超时、种子面板 `[730241]` 及 ACO 的 `1234 + 730241` 流与 V10.14 一致。搜索尝试 1000，搜索评分上限 1000，ACO 每路 4 workers；模型输出上限 8192、输入上限 24320，T=1 / top_p=.95 / top_k=20 / thinking=False。不注入已知任务答案，不加全局突破奖励或统一复杂度惩罚。

行为探针均从训练数据生成；先真实评分，再用单独安全进程（最多 5 秒）诊断。探针超时／失败不撤销已取得的有效分数；探针调用、错误和秒数另存于评价结果，独立选择不跑探针。

| 任务 | 诊断内容 | 限制 |
|---|---|---|
| OBP | 两种容量和早／中／尾部 best-fit 公共状态；选中索引、原箱索引、容量、剩余量、平局数、残量直方图及有序状态 hash | 同容量不等于全局等价；基线轨迹不是候选自己的轨迹 |
| ACO | 用实际 solver 的可行性更新捕获固定轨迹；用 evaluator 的 `max(prior+1e-9,1e-9)`、真实掩码、alpha/beta、初始信息素、OP dummy sink 计算概率 | 只覆盖初始信息素下有限状态，未覆盖所有强化后分布 |
| TSP | 固定训练实例上最近邻轨迹的早期、中段、尾部与最后一步 | 不是完整语义等价测试 |
| VRPTW | 沿用训练可行状态与时间／容量信息 | 仍是有限样本，默认无资格否决权 |

成功的同源码／同协议／同种子面板可缓存，失败不缓存。默认冻结训练质量前五个不同源码候选，在独立选择集上选最终程序：ACO `val_50`，生成任务 seed `20260927`。最终测试 seed / test split 不参与生成和选择。

## 运行与开关

```bash
uv run python -m experiments.traceaad_v10_14_2.run \
  --task tsp_construct --budget 1000 --dry-run
uv run python -m experiments.traceaad_v10_14_2.run \
  --task tsp_construct --run-name v1014_2_trial --budget 1000
```

结果根目录 `experiments_result/traceaad_v10_14_2/`；方法 ID `v1014_2`，显示名 V10.14-2。配置、模型、有效实现及评价源码哈希参与恢复校验。外部调用结果未知时不自动重放，不隐藏新增费用。

可选消融明确同时打开开关和预算比例：`--online-revalidation --recheck-fraction .1`，`--fixed-three-step-commitment --trial-fraction .2`；行为资格消融用 `--behavior-eligibility-gate`。这些是实验配置，不用于本批默认运行。连续开发可通过全档案正常再次选父发生，无须强制三步票。精确闭合与事实结构保留，离线能力不需打开在线开关。

复制既有五任务四重复的后端／seed 映射，先检查计划：

```bash
uv run python -m experiments.traceaad_v10_14_2.launch_batch \
  --from-batch experiments_result/traceaad_v10_14/batch_20260927_v1014.json \
  --batch 20260928_v1014_2 --dry-run
```

正式启动需去掉 `--dry-run`。此入口拒绝已有目录／session／manifest，也拒绝旧批次 session 尚活跃的情况；不自动停止旧实验、不重试失败路次、不迁移后端。清单先写入，再逐路启动并更新状态。

监控：`uv run python -m experiments.monitor --host 0.0.0.0 --port 8765 --experiment traceaad_v10_14_2`。横轴是候选尝试，不是纯评价次数。

搜索及独立选择完成后：

```bash
uv run python -m experiments.infra.evaluate <run_dir> --output-dir <heldout_dir>
```

## 验证与研究边界

```bash
uv run python -m pytest -q tests
ruff check traceaad/v10_14_2 experiments/traceaad_v10_14_2 \
  tests/method/test_traceaad_v10142.py tests/experiments/test_traceaad_v10142_experiments.py
uv run python -m experiments.traceaad_v10_14_2.replay \
  --snapshot experiments_result/traceaad_v10_14/analysis_20260927_diagnosis \
  --output <replay_report.json>
```

测试覆盖原始多行 Idea 经持久化重读进入下一轮提示、真实 token 展示限额、完整模块／依赖／多备选／冲突／未定义名称、全档案竞争及重复共享次数、恢复、独立动作、Repair 计费、探针失败隔离与五任务真实小规模评价／选择。批量历史响应回放不调用模型或评价器、不改写旧成绩；其 token 统计明确为未测量，实际 tokenizer 另做在线只读核验。

本批是联合机制修订；即使优于旧批次，也不能把差异单独归给选父、Pivot 或 Idea。必须完整观察四路分布、结构命中与开发、实际成本和独立测试，后续再做预先固定口径的消融。300 候选只能用于早期退化筛查，不作为迟发突破的终局判决。
