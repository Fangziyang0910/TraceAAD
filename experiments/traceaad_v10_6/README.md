# TraceAAD V10.6

实现[完整机制设计](../../docs/01-主线版本/TraceAAD-V10.6-机制设计.md)。每次先选择父代、再选择算子，生成并评价一个子代，然后重新分配预算。第一次调用先输出设计Idea和完整代码，第二次调用依据最终代码重新生成约500 words的实现Idea；历史只使用后者。

两次调用统一使用公共任务说明与函数模板。初始化与V9.16一样展示签名和docstring，函数体由模型生成。VRPTW公共模板已恢复为旧基线文本；V10.6专用任务描述、模板措辞替换与具体评测时限注入已移除。评价配置保持原状。

当前协议标识为 `idea_code_then_implementation_idea`。此前已启动的20260906正式批次属于旧单次调用配方，新代码拒绝将其检查点作为新协议恢复。旧批次现已按用户要求停止并删除其运行数据；新正式批次`20260906_215231_revised`已启动15路并完成初始协议核查，见[修订启动记录](launch_20260906_revised.md)。

| 项目 | 正式配置 |
| --- | --- |
| 规模 | 五任务 × 三重复，seed=0/1/2，每路1000次真实评价 |
| 初始化 | 从头生成8个有效根，评价计入1000次 |
| 分配 | 保留V10.5联合概率：m=.925p0+.075/N，父代选定后按条件概率选择R/P/F |
| 基础权重 | ESS目标min(N,max(.1N,2))，再除以sqrt(选择次数+1) |
| 算子 | 请求边际R/P/F=.50/.15/.35；Fuse无合法donor时执行Refine |
| 历史 | 最近最多8条真实形成事件，上限8192实际tokens |
| 摘要 | 每条进入上下文最多1024实际tokens；超长时保留完整前缀段落，原文留档 |
| 上下文 | 总32768，输出预留16384，余量256；完整输入最多16128 |
| 采样 | temperature=1、top_p=.95、top_k=20，thinking关闭 |

运行入口：

```bash
.venv/bin/python -m experiments.traceaad_v10_6.run --task tsp_construct --backend server3 --repeat 1 --seed 0 --run-name example_tsp_v106_rep1
.venv/bin/python -m experiments.traceaad_v10_6.launch --batch UNIQUE_BATCH --session-prefix v106 --watch
./experiments/traceaad_v10_6/start_monitor.sh [PORT] [--restart]
```

调度器放在持久tmux会话中。每一路实验另建会话，完成后由调度器更新`results/batch_BATCH.json`。`--dry-run`查看当时可启动的分配。同任务重复优先分散到不同后端，再按占用/容量分配；只使用空闲槽位，server1/server3/server3b/local逻辑容量为6/9/9/3。后端启动检查失败仅暂停该后端分配，下轮重试。恢复沿用原backend、run_dir和seed；每路最多启动5次，未知评价提交状态单独阻断。

实际模型服务使用现有配置：远端为Qwen3.8-27B-AWQ-INT4，本地为Qwen3.8-27B-UD-Q4_K_XL GGUF。两者量化及服务实现不同，分析需保留backend分组，不能仅凭共同别名视作完全相同。启动时进程参数保存为`results/backend_snapshot_20260906.json`。

完整候选模块直接进入统一评价器，保留辅助函数、装饰器、类和模块级语句。第一次响应需要设计Idea及闭合代码块；`finish_reason=length`时，已经完整闭合且通过静态检查的代码仍进入第二次调用和评价。第二次响应为空、格式不符、截断或传输失败时，代码继续评价，实现Idea标不可用。校准请求超出完整输入预算时同样保留代码，记录`context_exceeded`。每次真实评价（包括初始化、运行失败、超时）占一个预算；模型调用单独记成本。

复用V10.5的持久化响应、评价提交及收据机制。两次调用分别保存响应，已有代码响应不因校准中断而重新生成，已有校准响应也直接复用；已有收据不重复评价。未持久化返回结果的远程请求可能需要重试未完成阶段。无法确认是否执行过的评价保留unknown reservation并停止该路。V10.6 checkpoint版本106，保存生成协议、公共任务内容及源文件指纹。

每路产物：`run_config.json`保存配置；`tree_state.json`保存档案、预算和RNG，节点idea为第二次调用结果；`pending_candidate.json`保存处理中操作及两次响应；`llm_calls.jsonl`分别以generation/thought_alignment记录请求、响应、usage与耗时；`events.jsonl`保存分配、前置design_idea、上下文和评价结果；`evaluations.jsonl`保存评价收据；`tokenizer_calls.jsonl`保存精确计数；`logs/run_summary.json`保存终态。运行中以checkpoint及pending状态判断进度，不能将尚未生成终态summary误判为失败。

必要验证：

```bash
.venv/bin/python -m pytest -q tests/method/test_traceaad_v106.py tests/experiments/test_traceaad_v106_launch.py
.venv/bin/python -m pytest -q tests/method/test_traceaad_v105.py tests/experiments/test_traceaad_v105_launch.py tests/tools/test_openai_api.py tests/method/test_traceaad_v103_schema.py tests/method/test_traceaad_v104.py tests/experiments/test_traceaad_v104_launch.py
.venv/bin/python -m pytest -q tests/task/test_task_contracts.py tests/task/test_task_contract_fixes.py tests/task/test_vrptw_heldout_protocol.py
```

2026-09-06本轮：针对性验证46项、旧版及共享接口回归55项、任务与评价协议89项通过，共190项。覆盖公共任务历史文本、两次调用及历史取值、校准失败与各阶段中断恢复、评价幂等及旧协议拒绝恢复。本地检查使用可控模型替身与本地评价；随后新正式批次已验证真实两次调用及入档协议，摘要语义质量仍待研究核查。

历史单次调用配方曾通过32项针对性检查、55项回归与五任务真实冒烟；这些联调结论不自动适用于新的两次调用。旧正式批次的启动观察见[启动记录](launch_20260906.md)。
