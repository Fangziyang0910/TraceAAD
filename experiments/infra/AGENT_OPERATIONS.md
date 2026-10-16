# 实验执行入口

实时状态以批次清单、日志和实际进程为准。实验条件见[规范](../../docs/08-实验规范与共识/README.md)，成绩见[结果](../../docs/02-实验结果/01-搜索结果概览.md)。

## 状态查询

```bash
uv run python -m experiments.infra.batch_status --manifest <批次清单> --details
uv run python -m experiments.infra.batch_status --ssh B3-server3 --repo /home/fzy/code/LLM4AD/TraceAAD --manifest <批次清单> --details
```

`--json` 返回结构化数据，`--wait-seconds 45` 等待状态变化，每次最多等待 60 秒。`recorded` 只表示存在记录，`stale` 需要检查进程。本地副本只表示已同步状态。ETA 仅包含搜索。预算读取方法事件和恢复点，不能用有效生成尝试数代替。

## server3

| 对象 | 入口 |
| --- | --- |
| SSH | `B3-server3` |
| 仓库 | `/home/fzy/code/LLM4AD/TraceAAD` |
| Python | 仓库内 `.venv/bin/python`，不解析 symlink |
| CPU 调度器 | tmux `traceaad_scheduler`，socket `/tmp/traceaad-1005/scheduler.sock`，物理核 0–51 |
| 实验数据 | `experiments_result/` |

运行设置 `HF_HUB_OFFLINE=1`，凭据从远端 `.env` 读取。连接失败时检查 `ssh -G B3-server3`。调度器重启前确认旧进程已退出，再删除遗留 socket。结束 tmux 不会自动删除 socket。

server3 用作执行机，保留统一 `.venv` 和在跑数据。代码同步使用现有入口，不在搜索进程运行时覆盖公共源码：

```bash
uv run python -m experiments.infra.remote push
uv run python -m experiments.infra.remote pull --experiment <方法> --batch <批次>
uv run python -m experiments.infra.remote sync --experiment <方法> --batch <批次>
uv run python -m experiments.infra.remote watch --experiment <方法> --batch <批次>
uv run python -m experiments.infra.remote prune --experiment <方法> --batch <批次>
```

`push` 镜像代码，删除远端多余代码但保留 `.venv` 与结果。依赖变化时执行 `uv sync --frozen`。`sync` 每 60 秒只拉取，连接失败后重试，整批搜索结束后停止，不等待后续独立测试。可在本机 tmux 中运行。`watch` 每 60 秒拉取并清理已完成档案。`prune` 仅清理已完成、已有独立测试且与本地逐字节一致的数据。固定档案先固定清单，再同步并检查哈希。增长中的日志不用于固定档案的一致性判断。

## 看板与结果读取

```bash
uv run python -m experiments.monitor --host 0.0.0.0 --port 8765
```

看板地址 `http://127.0.0.1:8765/`。`#b=<方法>&rb=<批次>` 固定批次，`rb=latest` 跟随最新批次。API 为 `/api/state?batch=<方法>&run_batch=<批次>`，跨方法结果为 `/api/compare?cohorts=eoh,reevo`。修改 Python 后重启服务，修改 HTML、CSS、JS 后刷新页面。

日常读取 `events.jsonl`、`programs.jsonl`、`resume.json`、`summary.json`、`selection.json`、`heldout.json`。原始请求在 `calls.jsonl` 或其 gzip 档案。基线失败类型与超时率读 `evaluations.jsonl`。测试结果须与最终选中程序的哈希和节点一致。`.cache/history.json` 可删除重建。格式见[SEARCH_FORMAT](SEARCH_FORMAT.md)。

## 历史原始目标值补全

`experiments.infra.migrations.raw_search` 处理已结束的相对评分批次。已有逐实例成绩直接换算，缺失时按该条历史评价的 CPU 规则、限时和种子补评。仅有平均改进率不能精确换算。程序和最终选择不变。超时补评按用户要求取消 CPU 时间限制、将单实例墙钟上限放宽到至少 600 秒重试。此前缓存的超时也重试。结果来源记为 `cpu_reevaluation_relaxed`，保留原失败记录和实际条件，不改变历史有效性。其他失败按原条件重试一次。仍失败则保留缺失，不能写成零分。

20261011 四种基线的三项相对评分任务共 12 路，在 server3 的 `experiments_result/raw_objective_conversion/` 独立目录执行，tmux 为 `raw_objective_conversion`，6 路 × 4 worker（最多 24 个 CPU 槽） 共享现有调度器。批次状态读该目录 `status.json`，逐路读 `raw_search.json`，逐次成绩与来源读 `raw_search_evaluations.jsonl`。本机 tmux `raw_objective_sync` 每 60 秒同步，补全结束或失败后停止。原同期对比表在同步时更新，缺失值不参与排名。

每路补评结束后替换程序、评价和种群的公开分数字段，清除曲线缓存。`score_source` 区分精确换算与 CPU 补评。原件保存在 `logs/raw_search_originals/`。历史决策分数使用 `historical_*` 字段，恢复状态与调用原文保留当时含义，不代表按原始目标值重新搜索。已转换的档案用于分析。复现历史搜索使用原件及其配置。

<a id="中断后的基线恢复"></a>

## 基线恢复

恢复入口支持 EoH、ReEvo、FunSearch、MCTS-AHD、CALM、PathWise 和 ShinkaEvo。在执行机仓库根目录运行：

```bash
# 只读核对配置、预算与搜索状态
HF_HUB_OFFLINE=1 .venv/bin/python -m experiments.infra.resume_baseline --run-dir <运行目录>
# 缺少评分向量的 FunSearch 旧档案，先做 CPU 重评
HF_HUB_OFFLINE=1 .venv/bin/python -m experiments.infra.resume_baseline --run-dir <运行目录> --prepare-profiles
# 单路或整批续跑
HF_HUB_OFFLINE=1 .venv/bin/python -m experiments.infra.resume_baseline --run-dir <运行目录> --execute
HF_HUB_OFFLINE=1 .venv/bin/python -m experiments.infra.resume_baseline --manifest <批次清单> --execute
```

批次入口复用原 tmux 会话，跳过已存在的会话和已固定路次。满预算但未固定的路次仅做最终选择。日志为 `logs/resume_search.log`。`launched` 仅确认会话创建，仍须核对进程和新增记录。

| 方法 | 恢复状态与边界 |
| --- | --- |
| EoH | 最近种群和后续程序，从下一算子继续 |
| ReEvo | 最近完整种群、当前最好程序、长期反思，从新一代继续 |
| MCTS-AHD | 按事件和真实父代提示重建搜索树，从新一轮继续 |
| FunSearch | 岛归属与逐实例评分向量，从新提示批继续 |
| CALM | 自动检查点保存程序、种子、评分向量、提示回访与数值微调状态 |
| PathWise | 自动检查点保存初始化、种群、节点、父代与反思状态 |
| ShinkaEvo | 自动检查点保存岛副本、archive、embedding、bandit、meta 与代数 |

前三种方法及 FunSearch 的旧日志恢复为 `journal_boundary`，原随机状态与执行栈未保存，按原 seed 重新开始边界后的执行。后三种方法的新实验保存 `search_checkpoint.json.gz`，恢复 Python／NumPy 随机状态。执行栈不保存。未完成阶段从保存边界继续，已有程序支出与 token 累计不退回。旧档案缺少必需状态时拒绝恢复，不能只凭最好程序补造状态。

FunSearch 已有评分向量时直接读取。缺失时用重评分数与向量重建数据库，保留岛归属。失败程序排除，最终训练排序使用重评分数，数据分别保存在 `recovery_profiles.jsonl` 和 `recovery_evaluations.jsonl`。空岛、父代不唯一或岛重置来源缺失时拒绝启动。CPU 补全结束不会自动恢复模型请求，准备状态见 `recovery_ready.json`。

新启动与恢复共用 `.recovery.lock`，进程退出后释放。带 `writer_lock` 的遗留 running 状态可再次恢复。旧进程没有该标记时，先核实退出再修正状态。批次更新用 `.batch.lock` 串行写入。未完成 JSONL 尾行在写入前移到 `logs/interrupted_tails/`，压缩档案原子替换。只读核对不修改原件。恢复事实与代码哈希在 `recovery.jsonl`。

## 本地模型与排队

本地后端为 `http://127.0.0.1:8001/v1`、`Qwen3.8-27B`，tmux `qwen3_8-27b-mtp`，3 槽。健康检查：

```bash
curl -s --noproxy '*' 127.0.0.1:8001/health
uv run python -m experiments.infra.local_queue --method funsearch --suite co6 --batch <批次> --slots 3
```

服务启动参数见 `~/models/gguf/qwen3.8-27b/update_ud_q4_k_xl.sh`。本地评价使用物理性能核 `0,2,4,6,8,10,12,14`。模型线程用 `taskset -a -cp 16-31 <pid>` 固定到能效核，重启后重新绑定。排队器只在内存跟踪在跑任务，中途退出后等现有任务结束再重启。旧实验见[归档目录](../../archive/local_models_20261011/README.md)。

## 本地 SGLang 决策接口

2026-10-16 建立独立环境 `/home/fang/.local/share/sglang-decision/venv`，使用 Python 3.12.11、SGLang 0.5.21、PyTorch 2.13.0+cu130、sglang-kernel 0.4.7、TypeSafe SDK 0.7.5。完整依赖固定在同目录上一级的 `requirements.lock.txt`。项目的 Python 3.14 环境用于发送测试请求。

| 对象 | 入口 |
| --- | --- |
| GPU | 本机 RTX 4090 D，24GB |
| 模型 | `/home/fang/models/hf/Qwen3.8-27B-AWQ-INT4`，来源 `cyankiwi/Qwen3.8-27B-AWQ-INT4` |
| 固定版本 | `6e134bae811fb5adac50ee042ae5f029ac6779aa`。五个权重分片校验 SHA256，10 个配置与分词文件核对 HF 文件摘要 |
| 服务 | `http://127.0.0.1:30000`，模型名 `Qwen3.8-27B-AWQ-INT4` |
| 接口 | `POST /v1/systemone`，也提供原生 `/v1/decisions` 与 `/v1/score` |
| 启动脚本 | `/home/fang/.local/share/sglang-decision/launch-qwen38.sh` |
| 进程与日志 | tmux `sglang-decision`。`/home/fang/.local/share/sglang-decision/server.log` |

原有 GGUF 在此版本报 `GGUF model with architecture qwen35 is not supported yet.`，因此使用 AWQ 权重。启动脚本通过 `--json-model-override-args '{"language_model_only":true}'` 关闭视觉塔，不启用 MTP。保留 BF16 激活和 KV、FP32 线性注意力状态。配置上限 32K、并发 2、状态缓存 2、显存比例 0.91，关闭 CUDA graph 和前缀缓存。实际 KV 池为 38,703 token。模型线程固定在能效核 16–31，启动脚本自动设置。`--language-model-only` 的命令行开关在此版本拒绝 Qwen。`--language-only` 仅跳过视觉权重加载，仍创建视觉模块，不能代替上述配置。

启动脚本已把独立环境加入 PATH，首次推理需要的 `ninja` 和 CUDA JIT 编译由该环境完成。只有健康检查返回 200 后再发送测试。HTTP 端口开放时可能仍在预热。

```bash
tmux new-session -d -s sglang-decision '/home/fang/.local/share/sglang-decision/launch-qwen38.sh > /home/fang/.local/share/sglang-decision/server.log 2>&1'
curl --noproxy '*' -fsS --max-time 5 http://127.0.0.1:30000/health
```

TypeSafe 客户端的 `base_url` 指向服务根路径，不带 `/v1`。本地未设置 API key，SDK 使用占位 key 即可。长状态、多题请求把 SDK 超时提高到 300 秒。用独立环境的 Python 调用：

```python
from typesafe_sdk import Choice, TypeSafeClient

client = TypeSafeClient(
    base_url="http://127.0.0.1:30000", api_key="unused",
    model="Qwen3.8-27B-AWQ-INT4", timeout=300, compression=False,
)
result = client.system_one(
    "The current best program has been refined 20 times without improvement.",
    {"operator": Choice(
        instructions="Which kind of change should the next attempt make?",
        criteria={
            "Refine": "Improve the implementation within the same core idea.",
            "Explore": "Try a different core computation.",
            "Crossover": "Bring in a useful computation from another program.",
        },
    )},
)
print(result.answers)
```

真实状态重放使用 `experiments.infra.decision_probe`，每个 JSONL 输入行包含 `id` 和完整 `request`。回复、耗时与原请求写入新的输出文件，已有文件拒绝覆盖。脚本检查选项编号、概率归一化、选择结果及生成 token 为零。

```bash
uv run python -m experiments.infra.decision_probe \
  --requests experiments_result/decision_probe/20261016_sglang_qwen38/requests.jsonl \
  --output /tmp/qwen38_decision_replay.jsonl
```

输入、旧 Jev 缓存与本机回复保存在 `experiments_result/decision_probe/20261016_sglang_qwen38/`。版本与启动配置见 `manifest.json`。这是决策接口诊断，不是完整搜索结果。决策接口不训练模型。返回的是答案标签之间归一化的偏好，不能直接解释为某次改动实际成功的概率。
