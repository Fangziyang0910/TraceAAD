# TraceAAD V19.3

V19.3在同一任务的搜索中共享正负设计经验。生成读取相关经验；评价后，反思模型结合最新尝试、近期主链与同父代旁支提炼新经验。研究问题见[设计文档](../docs/机制设计/V19.3.md)，完整提示见[提示附录](../docs/机制设计/V19.3-上下文与提示词.md)。

## 运行

在仓库根目录运行完整离线演示：

```bash
.venv/bin/python -m traceaad_v19_3 demo --run-dir /tmp/v193-demo --budget 12
```

演示使用脚本化回复，经过实际评价、共享修复、独立反思、经验池和持久化。演示结果用于检查流程。

接入模型与仓库任务：

```bash
.venv/bin/python -m traceaad_v19_3 run \
  --task tsp_construct --run-dir /tmp/v193-tsp \
  --base-url "$TRACEAAD_BASE_URL" --model "$TRACEAAD_MODEL" \
  --api-key-env DEEPSEEK_API_KEY --budget 1000
```

外部任务使用`--task-factory module:function`，返回本包的`Task`或兼容评价器。包内的搜索、评价、模型客户端、存储和提示只依赖本包；仓库benchmark仅在指定`--task`时加载。可以使用`pip install ./traceaad_v19_3`安装。

token计数默认使用本地缓存的`deepseek-ai/DeepSeek-V4.1-Flash` tokenizer。其他模型通过`--tokenizer`指定对应的本地`tokenizer.json`或已缓存仓库，并设置正确的`--context-window`。运行时不自动下载tokenizer。自定义模型的`complete`应接受`temperature`、`max_tokens`、`timeout`和本次`response_format`，返回文本或`ModelReply`。

恢复时使用相同命令并添加`--resume`。配置、任务、模型、提示、tokenizer和源码须匹配。已落盘回复继续处理；没有回复的在途请求保留已用计数。检查点不兼容V19.2。

## 反思对照

| 参数 | 反思可见内容 |
| --- | --- |
| `--reflection-mode local` | 最新尝试及其生成、修复版本 |
| `--reflection-mode siblings` | 最新尝试及同父代的其他尝试 |
| `--reflection-mode trajectory` | 最新尝试、近期主链与同父代旁支，默认 |
| `--reflection-mode none` | 关闭反思调用，用于生成流程对照 |

前三项共用经验结构、生成提示、检索、维护和输出上限。默认最近3次主链、2次旁支；正负池分别取3条BM25相关经验。查询使用当前及近期`design`，兼容中英文。一次反思最多保存3条有效经验，原始回复与舍弃项保持可查。

## 预算与记录

默认1000次实际评价，初始化、运行失败和修复重评都计入。父代按有效去重程序的平方倒数名次选择，加5%均匀项。同分对称。初始化顺序取得5份有效且源码不同的程序。重复程序照常评价、保留实际来源，不增加选择权重。

每次尝试最多一次共享修复。生成默认16384输出tokens，反思2048。生成、格式修复、程序修复与反思分别记录请求、耗时和提供方用量；缺失usage保持缺失。

`state.json`保存权威状态；`programs/`保留独立代码版本；`calls/`保存完整提示、角色、schema、回复及用量；`evaluations/`保存实际评价；`attempts/`包含实际展示的经验、上下文覆盖、效用与反思状态。`experiences.json`是运行结束后的全局池视图。训练最好程序保存为`best.py`。

全局经验效用沿用RefineEvo源码规则，作为维护启发式。一次尝试依据最终已评价结果更新一次，仅更新实际展示的条目。反思失败保留算法结果。

正式任务的记录、调度与独立测试入口见[实验说明](../experiments/traceaad_v19_3/README.md)。
