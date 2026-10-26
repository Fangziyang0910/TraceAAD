# 固定父代上的变换迁移实验

检验同一批历史修改作为案例、结构化建议、受约束变换使用时，是否改变实施质量和单步收益。入口是 `python -m experiments.transfer.run`，汇总入口是 `python -m experiments.transfer.analyze`。

## 20261025 TSP 先导实验

在生成前固定 3 个来源修改、6 个接收父代、3 组条件、每格 2 次重复，共 108 次生成。来源为已完成的 TraceAAD v12.0 TSP 第一路，接收父代来自第二、三路。具体程序及原始代码复制到 `inputs.json`；历史原件只读。

来源修改由研究者按代码选取：26→42、80→89、281→284，分别涉及局部连接估计、部分评分权重的距离尺度调整和一步后续连接估计。它们不是历史成功修改的随机代表样本。接收父代按历史训练成绩分三层，每路每层随机取一个，排除与来源运行及已选候选重复的 AST，并限制代码长度。运行不同不保证算法思想独立，重复生成也不增加历史运行数量。

| 组 | 看到的知识 | 输出与实施 |
| --- | --- | --- |
| raw | 来源父子源码、差异、训练成绩 | 自由修改，完整代码 |
| advice | 相同案例及归纳规格 | 把规格作为建议，自由修改，完整代码 |
| constrained | 与 advice 相同的案例及规格 | 明确绑定，仅输出局部替换块；不适用可弃权 |

后两组共享规格。所有组均只生成一次，输出上限同为 8192 tokens，无修复或按结果重试。输出格式、弃权和受约束实施合在一个处理条件中，因此本实验不能把差异全部归因于某一句提示。输入长度和实际总 token 不同，本实验不是严格的同 token 比较。归纳和盲审调用单独记账。

第一版归纳规格把变量名、源码公式与抽象角色混在一起，并有节点 ID 的措辞错误。正式生成前保留 `draft_specs_v1/`，仅用原始来源案例重新归纳，明确角色绑定、允许的参数适配和输出语义。正式规格由 `library_manifest.json` 冻结。不得根据接收程序的迁移结果修改同批规格。

三组调用按固定随机顺序交错，经现有公共 Qwen 模型池执行，逐请求记录实际端点。评价使用当前统一 TSP 条件，16 个训练实例、50 个同规模测试实例；每实例平均函数 CPU 2 秒、墙钟 120 秒。接收父代和来源父子在新条件下重评，历史成绩不被覆盖。所有成功交付的代码都评价训练和测试，测试结果不反馈给生成器。

主要结果是所有尝试中的测试改进率。交付失败、运行失败、不适用和无变化不能从分母删除。另列有效程序的收益分布、弃权比例和代码实施忠实度。三组同一来源、父代、重复构成配对；结果按父代和来源分解。这是 6 个父代、2 条接收运行上的描述性先导实验，不报告把 108 次尝试当独立程序样本的显著性。

盲审只接收父代、规格和随机编号的子代代码，不看到组名、生成解释或成绩。盲审给出是否实施操作、是否保留必要计算、是否有无关修改及代码依据。该结果是模型辅助判断，关键案例仍需阅读代码核对。

## 执行与恢复

从仓库根目录执行。`prepare` 使用新目录；其他阶段可恢复，已有请求回复和评价不会重复执行。阶段锁防止同一阶段被两个进程同时写入。先设置现有模型和 CPU 调度器环境变量，凭据仍由原有环境加载。

```bash
.venv/bin/python -m experiments.transfer.run prepare --out experiments_result/transfer/NEW_BATCH
.venv/bin/python -m experiments.transfer.run induce --out experiments_result/transfer/NEW_BATCH
# 检查归纳规格后再生成；修改条件必须建立新的批次。
.venv/bin/python -m experiments.transfer.run generate --out experiments_result/transfer/NEW_BATCH --workers 4
.venv/bin/python -m experiments.transfer.run evaluate --out experiments_result/transfer/NEW_BATCH --workers 8
.venv/bin/python -m experiments.transfer.run audit --out experiments_result/transfer/NEW_BATCH --workers 4
.venv/bin/python -m experiments.transfer.analyze --out experiments_result/transfer/NEW_BATCH
```

`inputs.json`、`protocol.json`、`library_manifest.json` 固定输入和条件；`calls/` 保存完整请求、回复、实际用量和路由；`candidates/` 保存交付状态与源码；`evaluations/` 保存逐实例结果；`audits/` 保存盲审及编号对应；`results.csv`、`analysis.json` 为可重建汇总。源码快照、逐文件哈希和执行环境随批次保存。

本批的后续诊断独立保存在 `diagnostic_no_history/`（12 次只看父代的生成）、`diagnostic_self_neighbor/`（修正一个已生成程序的自邻居错误，额外两次评价）和 `diagnostics.json`（来源程序直接复用、实际端点、适用判断重复一致性）。`control.py` 运行无历史对照，`diagnose.py` 重建事后诊断，均不修改主实验。`manual_review.json` 保存本编码代理对四个受约束改动的逐份核对；它不是独立人工审阅。

完整结果与解释见[2026-10-25 实验记录](../../docs/记录/2026-10-25-变换迁移先导实验.md)。原始模型盲审存在 4 个回复格式或标签错误，另有布尔判定与代码证据冲突。不得把 `analysis.json` 的 `faithful` 字段直接当作已验证的事实；结合逐份核对解释。

设计与分析使用 Scientific Agent Skills 中的 experimental-design 和 statistical-analysis 指引处理配对、重复层级和失败分母。参考：Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. [arXiv:2609.00065](https://doi.org/10.48550/arXiv.2609.00065)。

## 20261025 最小在线闭环与模型执行对照

入口为 `python -m experiments.transfer.loop`。复用当前的父代选择、轨迹提示、Refine／Explore／Crossover 和实例评价；不修改正式搜索入口。两组都从相同的历史父代开始，分别使用弱、中、强两父代组合，每组一路。这三个配对块同时改变父代与随机种子，不能当成同一起点的三次重复。

- `open`：按现有权重做开放修改。
- `bank`：空库开始。开放修改产生训练改进后，用一次调用归纳变换，最多保存六个。隔次生成提供变换库，结合当前父代轨迹及已评价分支选择，不能匹配则开放修改。每次实施后更新尝试数、改进数和最近三次结果。其余生成继续开放修改。

每路固定 **32 次实际模型调用**，包括归纳、格式失败和接口失败；不免费重试、不另设修复。输出上限均为 8192，实际输入及输出 tokens 单列，不能称为等 token 预算。已有父代的历史生成成本在两组中均作为共同前提；重新评价的 CPU 成本单列。主比较看最好训练成绩随总调用的变化，结束后固定训练赢家，只测试该程序。变换正确性不能由模型自己填写的 `checks` 字段认定。

运行目录包含 `run_config.json`、逐调用原始响应、逐程序评价、`state.json` 中的全部程序与库更新、`selection.json` 和 `summary.json`。恢复通过已缓存的调用与评价重放决策，核对提示和源码哈希；同一路由文件锁防止重复启动。仅兼容同一源码快照。基础设施 tokenization 请求不算生成调用。

另用 `python -m experiments.transfer.execution` 做 24 次执行诊断：三个明确指定的操作 × 原变量名／改名 × 两次生成 × Qwen／DeepSeek。操作由实际源码差异人工写明，目标都是已知可适用父代，专门隔离执行能力，不检验归纳或跨结构迁移。候选使用同一局部替换输出。按冻结参考程序，在 300 个固定输入上比较返回节点；涵盖不同节点数、排列与距离尺度。零差异只是这些输入上的行为一致，不等于完整语义证明。参考程序与未经修改父代的自检结果也保存。

两项实验使用相同 JSON 处理：允许裸 JSON 或单个完整 JSON 代码围栏，不从混杂文本中猜测或修复内容。原先导实验及其结果保持原条件。

批次协议、源码快照、输入与结果：`experiments_result/transfer/20261025_online_loop/`。运行前冻结协议，不因中间成绩扩展预算。

本批最终比较使用 `qwen_open_{0,1,2}` 与 `qwen_bank_{0,1,2}_corrected`。原 bank 目录保留纠错前记录。修正仅将模型反馈中的完整评价失败对象缩成失败类别，三个合规前缀分别为 18、16、15 次调用；额外请求不计入六条比较曲线，但在整批成本中列出。详见 `feedback_correction.json` 和[实验记录](../../docs/记录/2026-10-25-在线变换闭环与模型执行对照.md)。

`python -m experiments.transfer.loop_analysis experiments_result/transfer/20261025_online_loop` 核对并汇总最终六路。分析同时保存最终测试、真实 tokens、函数 CPU、变换应用与赢家路径，以及明确标为事后的共同 token 上限比较。旧快照在冻结赢家后遇到测试接口返回值解包错误，使用保存的 `finalize.py` 与最终评价驱动完成测试；当前 `loop.py` 已修正，且可通过 `finish(run_dir)` 续做已冻结赢家的测试，不调用生成模型。
