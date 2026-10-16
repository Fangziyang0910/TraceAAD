# 实验操作

运行前核对[实验条件](../PROTOCOL.md)。本页的主机与服务位置用于查找；实时状态需要检查进程、日志和健康接口。

## 查询状态

命令在仓库根目录执行。将 `MANIFEST_PATH` 替换为批次清单路径，将 `RUN_DIR` 替换为运行目录。

```bash
# 本地查询
uv run python -m experiments.infra.batch_status \
  --manifest MANIFEST_PATH --details

# 远端查询：清单路径按远端位置填写
uv run python -m experiments.infra.batch_status \
  --ssh B3-server3 --repo /home/fzy/code/LLM4AD/TraceAAD \
  --manifest MANIFEST_PATH --details

# 本地看板
uv run python -m experiments.visualization --host 127.0.0.1 --port 8765
```

| 输出或选项 | 含义 |
| --- | --- |
| `--json` | 输出结构化状态。 |
| `--wait-seconds 45` | 最多等待 45 秒，检查状态变化。 |
| `recorded` | 找到记录，不等于已确认进程仍在运行。 |
| `stale` | 需要进一步检查进程。 |

看板地址为 `http://127.0.0.1:8765/`。本地数据只反映已同步的状态。生成预算读取事件与恢复点，不用有效程序数代替。文件说明见[结果格式](SEARCH_FORMAT.md)。

## server3 入口

| 对象 | 位置 |
| --- | --- |
| SSH | `B3-server3` |
| 仓库 | `/home/fzy/code/LLM4AD/TraceAAD` |
| Python | 仓库内 `.venv/bin/python`；保留该路径，不解析符号链接。 |
| CPU 调度器 | tmux `traceaad_scheduler`；socket `/tmp/traceaad-1005/scheduler.sock`。 |
| 实验数据 | 仓库下 `experiments_result/`。 |

远端运行设置 `HF_HUB_OFFLINE=1`，凭据从远端环境加载。连接配置用 `ssh -G B3-server3` 检查。重启调度器前确认旧进程已退出；结束 tmux 后可能仍有遗留 socket。

## 同步与清理

将 `METHOD_NAME` 替换为实际结果目录名，将 `BATCH_NAME` 替换为批次名。当前 TraceAAD 默认结果目录名为 `traceaad`，历史批次按原目录填写。

```bash
uv run python -m experiments.infra.remote pull \
  --experiment METHOD_NAME --batch BATCH_NAME

uv run python -m experiments.infra.remote sync \
  --experiment METHOD_NAME --batch BATCH_NAME
```

| 动作 | 实际作用 |
| --- | --- |
| `pull` | 拉取一次批次结果。 |
| `sync` | 默认每 60 秒拉取一次；清单内所有搜索达到终止状态后停止。失败终止也算终止。 |
| `push` | 镜像本地代码，删除远端多余代码；保留排除的环境、结果与 `archive/`。 |
| `prune` | 先拉取，再按完成状态、测试文件存在性和校验结果删除远端运行目录。 |
| `watch` | 循环拉取并执行 `prune`。 |

同步代码的命令为：

```bash
uv run python -m experiments.infra.remote push
```

远端搜索使用公共源码时，不要覆盖它。`push` 检测到依赖文件变化后，会在远端执行 `uv sync --frozen`。具体排除与同步规则见 [remote.py](remote.py)。

需要清理时才使用 `prune` 或 `watch`。默认检查 `heldout.json` 存在，但不验证测试内容；`--no-heldout` 会跳过存在性检查。删除前使用 rsync 校验远端文件是否已在本地一致保存，排除 `.cache/`。这不是对结果科学有效性的检查。

增长中的日志副本仍可能变化。复查实验时，以已提交的事件和对应配置为准。

## 恢复搜索

先只读检查基线的恢复条件，再加 `--execute` 续跑：

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m experiments.infra.resume_baseline \
  --run-dir RUN_DIR

HF_HUB_OFFLINE=1 .venv/bin/python -m experiments.infra.resume_baseline \
  --run-dir RUN_DIR --execute
```

支持的方法与恢复边界见 [resume_baseline.py](resume_baseline.py)。CALM、PathWise 和 ShinkaEvo 读取完整检查点；其他基线从当前日志中保存的种群、树或数据库记录恢复。日志恢复不还原完整随机状态。FunSearch 使用已记录的逐实例评分；缺失时停止恢复，不补评旧数据。

TraceAAD 使用原入口、原目录和相同参数读取恢复点。操作前核对源码与配置，确认旧进程已退出。历史运行的评价规则以对应源码快照为准。恢复继续已提交的状态；最好程序不能重建缺失的搜索状态。锁与日志尾部由现有入口处理。

## 模型与计算资源

后端标签、端点和请求容量见 [base.py](base.py) 的 `BACKENDS`。通过现有启动器分配请求槽，不在文档中复制容量表。

提供调度器 socket 时，实例评价申请 CPU 核心；未提供时，按配置的 worker 数并发执行。DeepSeek 直接请求网关，提供 socket 时仍可使用调度器分配 CPU。

| 资源 | 配置线索 |
| --- | --- |
| 本机常用 CPU 核 | `0,2,4,6,8,10,12,14`。 |
| server3 常用物理核 | `0–51`。 |
| 本地生成服务 | `http://127.0.0.1:8001/v1`；脚本 `~/models/gguf/qwen3.8-27b/update_ud_q4_k_xl.sh`。 |

实际可用资源需要现场核对。本地基线排队入口为 `experiments.infra.local_queue`，先用 `--dry-run` 查看计划。它只在内存跟踪任务，跳过已有运行目录与活动会话，不自动创建重试目录。失败运行通过恢复入口处理。排队器中断后，先检查已启动的任务，再重新排队。

## 历史数据与修复

旧实验数据保存在本机 `archive/local_models_20261011/`。需要核对指定历史事实时，再定向读取其中的元数据与转换记录。分析或恢复前复制到新位置；已有硬链接在写入前需要分离共享 inode。

相对评分转换前的原件通常位于运行目录的 `logs/raw_search_originals/`。仅有平均相对改进，不能精确还原原始目标均值。`score_source` 区分换算与重新评价；放宽 CPU 或墙钟限制的结果保留其条件。

旧格式转换、评分换算和专项归档脚本已清理。相关历史结果与处理记录保留在对应实验目录中，当前读取器按程序个体编号读取结果。
