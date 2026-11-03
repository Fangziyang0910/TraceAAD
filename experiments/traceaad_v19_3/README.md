# V19.3实验入口

方法实现见[traceaad_v19_3](../../traceaad_v19_3/README.md)。本入口适配任务注册、资源调度、公开结果文件，以及训练赢家固定后的独立测试。

11月1日正式批次为`20261101_v19_3_co6`：server4运行六任务×种子0、1、2，每路1000次实际评价。采用现有Qwen共享池、Qwen tokenizer、49152-token上下文窗口和`trajectory`反思。配置、源码与真实启动核验见[启动记录](../../experiments_result/traceaad_v19_3/launch_20261101/README.md)。

先预览条件，预览不会请求模型或运行评价：

```bash
.venv/bin/python -m experiments.traceaad_v19_3.run \
  --task tsp_construct --run-name v193_preview --batch v193_preview \
  --budget 1000 --reflection-mode trajectory --dry-run
```

单路直接模型接口运行：

```bash
.venv/bin/python -m experiments.traceaad_v19_3.run \
  --task tsp_construct --run-name my_v193_tsp_rep1 --batch my_v193 \
  --base-url "$TRACEAAD_BASE_URL" --model "$TRACEAAD_MODEL" \
  --api-key-env DEEPSEEK_API_KEY --env-file .env \
  --budget 1000 --reflection-mode trajectory
```

`--scheduler-socket`提供CPU租约；不指定直接模型接口时，也通过调度器取得模型租约。调度器路径保留其Qwen采样设置；直接接口可用`--request-options`配置采样。两条路径都为每次生成与反思传入独立schema。模型的tokenizer及context window须与实际模型匹配。

默认所有实例共享总墙钟上限，每实例10秒作为总额度的计算单位，当前训练共160秒；内存2 GiB。这一条件明确记录为`whole_task`，与逐实例CPU预算分开解释。超时修复显示实际总时限，普通生成历史仅显示超时状态。

搜索结束直接固定训练成绩最好的有效程序，同分取较早记录，再独立测试。`native/`是权威搜索状态；公开的`programs.jsonl`、`events.jsonl`、`summary.json`和`best_program.py`由报告适配器生成。报告分别显示生成／反思请求、token用量、缺失usage、评价数和经验池大小。

批量预览：

```bash
.venv/bin/python -m experiments.traceaad_v19_3.launch_host \
  --batch my_v193 --tasks tsp_construct cvrp_aco --repeats 3 \
  --budget 1000 --reflection-mode trajectory
```

添加模型接口或调度器参数及`--launch`才启动tmux任务。`local`、`siblings`、`trajectory`三种反思范围应使用相同任务、种子、评价条件和经验维护配置，并按总模型成本另外对齐。

冻结源记录的迁移检验按[设计§5.1](../../docs/机制设计/V19.3.md#51-反思单元的作用)执行：源阶段只提取经验，不能用冻结的结果奖惩事后新检索的经验。当前入口实现在线搜索与反思范围对照，冻结源—目标实验需按所选记录另行组织。
