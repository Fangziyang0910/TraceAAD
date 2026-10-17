# 第三方提示来源

本页保留第三方提示的来源版本，供追溯使用。来源版本不等于对本地实现忠实性的验证。

## 来源与本地适配

| 本地内容 | 来源版本 | 原来源记录说明的适配 |
| --- | --- | --- |
| `pathwise/prompts/` | [PathWise · 9d014a36](https://github.com/oguzhangungordu/PathWise/tree/9d014a368d920d46ef4538da4c484503549214cb) | 按该版本复制提示。本地适配器提供任务描述、函数签名与计分方向。 |
| `reevo/prompts/` 中的任务知识 | [ReEvo · 6dce1825](https://github.com/ai4co/reevo/tree/6dce18257da5e11db2d138e417a2fffc5c72d05f/prompts) | 任务名称映射到本地基准，没有新增任务知识。 |
| `shinka_evo/prompts/` | [ShinkaEvolve · 8adc053a](https://github.com/SakanaAI/ShinkaEvolve/tree/8adc053a2ce4511ad2ac310e004c530a73fb974a) | 按该版本复制提示。本地适配器提供任务信息，类型导入指向本地程序记录。 |
| `deltaevolve/prompt.py` | [DeltaEvolve · ICML 2026 正式版](https://proceedings.mlr.press/v306/jiang26g.html)，附录 C Figures 5–6 | 根据 PDF 转录提示的实质内容，统一空白与标点，代入任务和目标方向。依据论文独立实现，未使用作者代码；补充决定见[实现说明](deltaevolve/README.md)。 |

链接保留完整提交标识。表中适配说明来自封存前的来源记录；本次未逐文件重新比对上游内容。

## 复查一次实验

先定位运行配置和源码快照，再对照该方法的上游版本。任务接口、提示适配和搜索流程分别核对，不能仅凭来源版本判断三者一致。

共同实验条件见[实验条件](../experiments/PROTOCOL.md)。DeltaEvolve 的新增来源与转录核对于 2026-10-16。

## 2026-10-17 机制移植

目标是在本项目任务、预算和统一模型配置下保留官方方法机制。方法内部以固定官方源码为准。已删除公共回复解析器；`core.code.function_from_source` 仅表示已被方法接受的源码。

- FunSearch 的函数体截取移入 `funsearch/parsing.py`，来源 `google-deepmind/funsearch@cc53f274237d7ab05c19df939edbc1f9616a7c19`。
- EoH 提示和 `_extract` 来自 `FeiLiu36/EoH@472545785c936dcfc863d2bc0d6109cf23c7ce62` 当前主线；采用其逐个更新、四算子、种群 5、父代 2 的默认设置。
- ReEvo 的 common 提示和提取函数来自 `ai4co/reevo@6dce18257da5e11db2d138e417a2fffc5c72d05f`，提示保持原文。
- MCTS-AHD 提示来自 `zz1358m/MCTS-AHD-master@ee9c4f424503c65a5fd2b899e6620ce86079fedb`，任务适配器提供函数契约。
- CALM 保留 `whxru/CALM@3c40dbd78f2f6439bcc7cc2cbac334ab28a6f8f3` 的唯一函数提取和操作识别，关闭 GRPO。
- PathWise 解析和父代码清理来自 `oguzhangungordu/PathWise@9d014a368d920d46ef4538da4c484503549214cb`；Action 的 `kappa` 接到本地字段 `rationale`。
- `shinka_evo/edit/` 的补丁引擎来自 `SakanaAI/ShinkaEvolve@8adc053a2ce4511ad2ac310e004c530a73fb974a`，只调整包导入，许可证随代码保留。当前运行入口使用单模型。

验证范围及尚未公开的 DeltaEvolve 细节见[机制移植记录](../docs/记录/2026-10-17-基线实现差异核对.md)。
