# TraceAAD V10.10

从 V10.9（1091）复制后移除新结构试用和直接子代试错上下文，保留质量分配与形成历史，加入宽容解析、具体错误记录和最多一次修复。2026-09-11 修订：初始化收缩为 sequential informed initialization，删除 Init 阶段 AST 去重与提示中的防御性约束；正常搜索上下文改为统一短形成路径；解析协议改为代码优先——程序单独决定候选能否评价，说明按 tagged/prose/missing 三种来源记录且不再设标签、位置或字数门槛，VRPTW 非法构造携带具体条件、修复反馈附真实时限与候选代码定位，契约写明运行环境版本与整次评价时限（解析策略 `code_first_description_extracted_v1`）。设计见 [V10.10 机制设计](../../docs/01-主线版本/TraceAAD-V10.10-机制设计.md)。正式批次 `20260910_v1010_formal`（2026-09-10 启动）运行的是修订前机制；历次修订改变机制与检查点指纹，需另起新批次。

解析修订的离线重放（零 LLM 调用）：`python experiments/traceaad_v10_10/analysis/replay_parser.py`，对 15 路全部已持久化响应重放新旧解析并输出分类、修复链对账与文件快照，报告写入本地 `results/analysis/`。2026-09-11 重放：旧 7,772 次解析失败中 7,676 次被新解析接受（代码通过语法/接口/编译校验），30 次暴露真实代码错误，66 次仍不可提取；新旧均接受的 14,641 次代码文本零差异，零回归。

默认五任务 × 三重复，seed 0/1/2，每路 1000 次真实评价、8 个有效根、32K 总上下文、16K 输出上限。修复后实际调用评价器也计入这 1000 次，LLM 修复调用在 `llm_calls.jsonl` 中标记 `stage=repair`，保留 usage、耗时与 `repair_of`。解析失败与搜索阶段父代/donor 重复过滤不占评价次数；初始化不做重复过滤。

输入不再固定限制为 16128 tokens。普通生成和错误修复都使用 `min(16384, 32768 - 256 - 实际输入tokens)` 作为本次输出上限；完整输入不做裁剪。服务的总窗口仍需容纳输入和实际输出。

当前配方：Refine、Tune、Fuse、Pivot 各 25%。父代选择在全档案上使用 ESS-8 Boltzmann 质量分布，Pivot 与均匀抽样 1:1 混合，selection counts 只记录不参与概率。四个算子统一使用「当前程序完整代码 + 成绩 + 最近八条短形成路径」，每步历史为该步生成算法的 Idea、算子与前后 fitness；Fuse 额外加入本轮 donor 的完整代码与成绩。算子指令按「设计对象 + 认知操作 + 结果目标」书写（如 Fuse 要求识别宿主不足与 donor 计算的对应关系后定向迁移整合），组装顺序为任务接口、当前程序（Fuse 中标题 Host Algorithm）、当前算法的形成路径、donor（仅 Fuse，标题 Donor Algorithm）、算子指令、输出契约。输出契约顺序中性：单个代码块的完整实现 + 描述该代码所实现算法的 Idea，可前置、后置或省略。初始化为 sequential informed initialization，按生成顺序展示全部已有根的完整代码与实测 fitness，不做重复规避；donor 只抽一次。历史不展开祖先代码、diff 或历史 donor，一次组装后检查总容量。删除在线 AST 修改分类，原始代码可供离线分析。

直接启动并监控批次：

```bash
uv run python -m experiments.traceaad_v10_10.launch \
  --batch v1010_formal --session-prefix v1010 --watch
```

调度继承 V10.9 的空槽分配及断点恢复；新 results 路径和会话前缀与 V10.9 隔离。

单路入口可通过以下命令查看：

```bash
uv run python -m experiments.traceaad_v10_10.run --help
```

共享监控将两个正式批次分开注册：`v10_10_new` 为新版 `20260911_v1010_formal`（会话前缀 `v1010f`），`v10_10` 为旧版 `20260910_v1010_formal`（前缀 `v1010`）。已停止的 thinking 批次不纳入这两个视图。

```bash
uv run python -m experiments.traceaad_v10_6.monitor \
  --version v10_10_new --session-prefix v1010f --port 8765
```

新版监控地址为 `http://127.0.0.1:8765/?version=v10_10_new`，旧版为 `http://127.0.0.1:8765/?version=v10_10`。页面默认展示新版，可在顶部切换，也可用两个浏览器标签页同时查看。重启现有监控可运行 `bash experiments/traceaad_v10_6/start_monitor.sh --restart`。

遇错后，普通失败候选先按原规则落日志；候选代码提取或校验失败，以及导入、运行、超时或非法结果才触发下一候选的一次修复——响应缺标签、说明为空或说明过长不触发修复。修复保留同一父代、donor 和原操作标签，增加 `repair_of`，不再次消耗父节点的独立设计请求计数。修复失败或重复即结束本次修复链。反馈只含清除路径后最多 2000 字符的核心异常类型和消息，runtime/exec 错误附候选代码最深栈帧行号与函数名，超时附真实时限数字；不附调用栈或父代整份代码；不按本地提示长度跳过修复。评价准备故障或框架异常记录后终止。没有剩余评价预算时不发修复请求，未知评价仍阻断。
