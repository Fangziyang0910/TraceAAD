# LLM 算法设计中的 Idea 长度与输出约束

更新：2026-09-13

本文只检查公开论文和官方代码中的原始提示词、输出格式和解析方式，用来判断 TraceAAD 的 Idea 是否应该使用“词数上限”，以及怎样设置更简单的约束。

## 结论

相关方法通常不把自然语言说明写成较大的字数配额，而是限定为一句话或极短摘要；代码部分则明确要求函数实现，或者干脆只返回代码。可复用的做法是：

1. 生成阶段要求 Idea 用一段简短的话概括核心机制，描述新方法做了什么以及最关键的变化；不要求逐项解释决策规则、推理过程或所有计算。
2. 解析阶段使用 token 数作为统一的异常阈值，而不是把正常输出卡在一个很窄的词数上。500 token 可以作为异常上限；超过后进入修复流程。
3. Idea 与 Code 分开提取。Idea 只作为搜索记录和后续上下文，Code 仍按目标函数的语法结构提取、编译和评估。
4. 对明显超长的 Idea 记录实际长度和错误类型；不要静默截断，因为截断可能删除机制的关键部分。

这支持当前 V10.11 的方向，但需要注意：放宽到 500 token 会降低 `idea_length_error`，不会自动消除模型的自我讨论。提示词必须把输出对象定义为“核心机制摘要”，而不是“解释你的全部设计”。

## MCTS-AHD：一句话 Idea，代码函数，另行做短摘要

MCTS-AHD 的动作提示词对初始化、交叉和变异都采用同一类输出契约：先“describe the design idea and main steps ... in one sentence”，并要求描述放在代码外的括号/花括号中；随后实现指定名称、指定输入和指定输出的 Python 函数，最后要求不要给额外解释。树路径动作也保持“一句话描述 + 函数实现”。

MCTS-AHD 还使用一个独立的 thought-alignment 调用，在得到代码后重新描述算法；这个调用要求“less than 3 sentences”，并提示参考代码、突出最关键的设计。也就是说，长篇 Idea 不是搜索节点的必要格式，详细说明被压缩成最多三句话的事后摘要。

来源：

- [MCTS-AHD 论文（Proceedings of Machine Learning Research，附录 E.1/E.2）](https://raw.githubusercontent.com/mlresearch/v267/main/assets/zheng25o/zheng25o.pdf)
- [MCTS-AHD 官方论文页面](https://proceedings.mlr.press/v267/zheng25o.html)

## EoH：一句话描述 + 函数代码，解析允许前置描述

EoH 原论文的五类提示词使用“describe your new algorithm and main steps in one sentence”，并规定描述放在花括号中；实现部分明确要求 Python 函数。EoH 的官方实现从代码块提取函数，并从代码块前提取算法描述；描述没有设置 200 词这类硬上限，缺少花括号时还会把代码前文本作为描述的回退值。

这说明两个设计点：长度控制应靠输出对象的简洁定义（一句话/短摘要），而不是堆叠多个解释项目；解析器可以保留轻量回退，但必须把代码结构作为独立的主契约。

来源：

- [EoH 论文（ICML 2024，官方论文 PDF）](https://raw.githubusercontent.com/mlresearch/v235/main/assets/liu24bs/liu24bs.pdf)
- [EoH 官方实现](https://github.com/FeiLiu36/EoH)，提示和提取逻辑见 `eoh/src/eoh/eoh/evolution.py`

## ReEvo：生成器只输出代码，反思器少于 20 词

ReEvo 把生成和反思分成两个角色。生成器的系统提示是“Your response outputs Python code and nothing else”，用户提示也要求“Output code only”。反思器看到较好和较差的代码后，只被要求给出设计提示，并明确规定“using less than 20 words”。

ReEvo 的做法将代码契约和语言提示分离：代码生成不携带自由文本，短文本只承担反思信号。这对 TraceAAD 的启示是，Idea 若继续保留，应当是短的机制标签/摘要，不应兼作模型的思考草稿。

来源：[ReEvo 论文附录 B（NeurIPS 2024 官方 PDF）](https://proceedings.neurips.cc/paper_files/paper/2024/file/4ced59d480e07d290b6f29fc8798f195-Paper-Conference.pdf)，其中 Prompt 1、4、5 分别给出代码-only 生成和少于 20 词的反思约束。

## FunSearch：只演化骨架中的函数，不维护 Idea 字段

FunSearch 的官方实现将 LLM 输出视为函数体的程序延续：提示包含带有 `@funsearch.evolve` 的目标函数和 `@funsearch.run` 的评估入口，评估器从生成文本中截取目标函数体，再放回固定程序骨架。成功与否由编译、运行和数值评分决定。

因此 FunSearch 没有独立 Idea 长度问题；它把解释性信息放在固定代码结构和程序版本中。这与“模板负责上下文，模型负责目标函数”的 TraceAAD 方向一致，也说明自由文本越少，格式错误面越小。

来源：

- [FunSearch 官方代码](https://github.com/google-deepmind/funsearch)，函数提取与骨架重建见 `implementation/evaluator.py`
- [FunSearch 论文（Nature）](https://www.nature.com/articles/s41586-023-06924-6)

## 对 TraceAAD 的具体建议

### 提示词

Idea 输出说明保留一个对象和一个动作即可：

> Idea: 用一段简短的话概括这个候选方法的核心机制和相对当前方法的主要变化。

随后直接给出 Code 区域及目标函数模板。不要再要求说明所有决策规则、关键计算、设计过程或选择理由；这些项目会把摘要任务变成自我讨论。

### 解析

- 用 tokenizer 统计 Idea token 数，阈值设为 500；这只是异常上限，不是目标长度。
- Idea 缺失或空白仍然进入修复；Idea 在 500 token 以内即视为可解析，不因略长于 200 词而拒绝。
- 仍然严格提取 Code 的目标函数定义并检查签名；Idea 放宽不应放宽代码结构契约。
- 日志至少保存 `idea_token_count`、`idea_length_status`（正常/超限）和修复结果，便于区分模型超长与解析问题。

### 研究判断

从这些方法的公开提示词看，TraceAAD 的 200 词硬上限比主流做法更像一个人为的格式配额；“一句话/短摘要”更贴近 EoH 和 MCTS-AHD 的实际设计。将异常阈值放宽到 500 token 是合理的工程调整，但不能把它解释成性能改进；需要在相同模型、预算和搜索设置下比较 Idea 长度分布、修复成功率、运行错误率和最终 fitness。
