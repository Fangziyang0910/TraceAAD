# 实验原始结果

## 当前格式（2026-10-07）

当前读取和展示均使用 `traceaad-results-v2` 的最小化目标。601 路已结束档案已转换存储；三路仍运行的 CVRP 使用只读最小化视图，后台会在结束后转换其文件。所有对外读取的 `fitness` 与 `score` 都越小越好。路径长度和箱数为正值，OP 为负奖励，CO-Bench 为负归一化质量。7,150 份派生分析和历史数值文件也已转换。目录仍是 `<实验>/<任务>/<运行>/`；字段见[结果格式](../experiments/infra/SEARCH_FORMAT.md)。

此次只转换目标的表示，没有重新评价或改变程序选择。候选编号、预算、突破时刻、源码身份、模型请求与回复保留原记录。数值备份和核验报告保存在 `.archive/minimize_20261007/`，每路有 `.minimize/receipt.json`。旧存储格式的原始证据仍在 `.archive/storage_20261006/`。恢复原始证据时解到独立目录，导入当前布局时显式转换；备份须包含当前目录和 `.archive/`。

## 之前的整理记录

`<实验>/<任务>/<运行名>/` 保存单路结果。TraceAAD 版本搜索统一使用 `run_config.json`（配置）、`search.jsonl`（模型调用、候选、评价与状态）和 `logs/run_summary.json`（终局摘要）。历史版本的 `search.jsonl` 每条记录包含 `kind`、`source` 和原始 `data`；同一来源保持原有行序，不表示不同来源之间有确定的发生顺序。无法从摘要或完整历史评价记录确认完成的运行标为 `status: unknown`；重建的摘要记录来源与完成依据。旧版状态快照保存在日志中，不能直接当作 V10.13 的续跑状态。

独立的 held-out 结果和实验级计划、汇总仍在对应实验目录中。

实验目录是本机数据，Git 不跟踪。V10.14 的正式批次为 `traceaad_v10_14/batch_20260928_v1014.json`，包含 20 个运行目录与 76 项 held-out。正式结论见 [V10.14 实验分析](../docs/03-现象与检验/2026-09-29-搜索结果与运行波动.md)；共享运行入口见 [实验脚本](../experiments/README.md)。迁移或备份原始数据时应连同运行目录的 `run_config.json` 一起保存。

2026-10-05 清理了本机非正式批次 `traceaad_initialization/`、`task_pilot/`、`traceaad_bc/` 和 `traceaad_v10_18_smoke/`。共删除 1,864 个结果文件，按文件分配块及硬链接计数释放约 717.2 MiB；[清理明细](local_cleanup_20261005/nonformal_cleanup.json)保存在本机。相关文档保留此前的历史汇总，这些批次的原始结果已不在本机。

同日继续清理了已中止的 `traceaad_v10_15_2/`、`traceaad_v10_15_3/`、`traceaad_v10_15_4/`，以及 V11.1 旧启动批次的 13 路残留和过期清单。共删除 297 个文件，释放约 1,414.7 MiB；V11.1 当前保留 25 路已完成的正式结果。[清理明细](local_cleanup_20261005/stopped_batches_cleanup.json)保存在本机。

V9.7 的旧实现将结束摘要写到 `logs/summary.json`，本机整理后的 12 路档案缺少这份原始文件，原 `logs/run_summary.json` 使用 `unknown` 占位。逐路核对历史候选记录后，确认每路均完成配置中的 1,000 次评价；已从评价记录和 `best_history.jsonl` 重建摘要，页面显示 12/12 完成。原始日志 SHA-256 均未变化；[原摘要备份](local_cleanup_20261005/v97_summary_reconstruction/original_summaries.json)和[重建依据](local_cleanup_20261005/v97_summary_reconstruction/report.json)保存在本机。

## 已完成档案的去重

以下为迁移前的方法，仅适用于尚未迁移且路径仍与恢复清单一致的档案。已经迁移的档案从上方压缩原档恢复。

`experiments.infra.deduplicate_archives` 可以把内容相同的 JSON 文件改为硬链接，保留每个路径和原始内容。只用于已完成、后续不再写入的档案。默认仅输出计划；执行时必须保存恢复清单：

```bash
python3 -m experiments.infra.deduplicate_archives <已完成的档案目录> \
  --apply --manifest experiments_result/local_cleanup_20261003/archive_dedup_manifest.json
```

硬链接共享后续写入。编辑档案或恢复写入任务前，先恢复为独立文件；恢复会核验内容，拒绝覆盖已改变的文件：

```bash
python3 -m experiments.infra.deduplicate_archives \
  --restore experiments_result/local_cleanup_20261003/archive_dedup_manifest.json
```

本轮清理范围和核验结果见[档案整理记录](../experiments/infra/RESULT_STORAGE.md)。恢复清单与原始数据保存在本机，迁移时一并保存；使用归档工具备份时，可保留硬链接关系，避免重新展开全部副本。
