# V19.4实验

入口统一执行搜索、固定训练赢家、独立主测试。每路1000次实际评价，初始化为五次基础尝试；温度1.0，生成与反思输出上限均为16384。整任务时限按每实例10秒形成，训练160秒、主测试500秒，内存2 GiB。主比较沿用现有Qwen共享池与实际Qwen tokenizer。

## 准备

在准备运行的源码根目录执行。以下操作写配置和启动脚本，不启动搜索：

```bash
.venv/bin/python -m experiments.traceaad_v19_4.prepare \
  --output experiments_result/traceaad_v19_4/preparation_20261102f \
  --batch-prefix 20261102f_v19_4_co6 \
  --scheduler-socket /tmp/traceaad-1003/scheduler.sock \
  --env-file /home/fzy/code/LLM4AD/TraceAAD/.env \
  --tokenizer /path/to/qwen_tokenizer.json
```

准备内容包括：

| 格 | 设置 | 路次 |
| --- | --- | --- |
| `main` | 新最好概率，响应向邻域与近期总体收缩 | 六任务×三个种子，18路 |
| `rank` | 个体排名平方，不使用轨迹 | 同上 |
| `direct` | 新最好概率，响应直接向近期总体收缩，不用邻域 | 同上 |
| `preflight` | 完整机制，初始化一次、每任务5次评价，跳过主测试 | 六任务×一个种子，6路 |

三组正式配置使用相同的生成、反思、经验保留、初始化与评价规则。预检用于检查实际流程，预算与正式组分开解释。

准备目录保存各格计划、源码哈希、tokenizer及各自启动脚本。脚本先校验源码再调用`launch_host --launch`，已存在批次或运行目录不会被覆盖。准备目录不是批次清单，也不会登记为主实验。

## 执行

先执行准备目录的`launch_preflight.sh`。预检记录显示正常生成、评价、反思与经验呈现后，再执行`launch_main.sh`。`launch_rank.sh`和`launch_direct.sh`启动相应对照；它们各自有独立结果目录。使用公共调度器时，模型和CPU资源通过现有租约分配。

单任务只读检查：

```bash
.venv/bin/python -m experiments.traceaad_v19_4.run \
  --task tsp_construct --run-name preview --batch preview --dry-run
```

直接查看计划：

```bash
.venv/bin/python -m experiments.traceaad_v19_4.launch_host \
  --batch 20261102f_v19_4_co6_main --budget 1000 --repeats 3 \
  --selection-mode trajectory --tokenizer /path/to/qwen_tokenizer.json
```

默认只输出预览。默认选择读取候选的局部轨迹（`trajectory`），`--selection-mode rank|direct|neighborhood`改为数值选择；反思范围使用`--reflection-mode local|siblings|trajectory|none`。`--no-generation-history`仅隐藏生成侧的形成和已有尝试，检索查询及反思保持不变。

## 记录和恢复

外层配置、事件、程序、选择、成本和主测试接入现有结果格式。`native/`保存不可变代码版本、实际评价、响应、调用原文、经验和检查点。模型角色和实际usage分开记录。经验的呈现次数与结果仅用于分析，检索不按单次结果删除经验。

恢复使用原`run.py`命令和`--resume`，源码、任务、模型身份、tokenizer与配置必须一致。训练赢家同分取较早记录，测试结果不用于再次选择。该版本的完整设计见[V19.4](../../docs/机制设计/V19.4.md)。
