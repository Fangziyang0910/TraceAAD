# 实验原始结果

## 当前格式（2026-10-06）

30 个批次的 564 路训练结果，以及 25 份诊断用搜索副本，已统一为 `traceaad-results-v1`。目录仍是 `<实验>/<任务>/<运行>/`。训练轨迹、源码、模型调用、恢复点和测试成绩各存一份，具体字段见[结果格式](../experiments/infra/SEARCH_FORMAT.md)。本机原有 1,827 条 held-out 记录的比较值保持一致。

原始文件压缩存入 `.archive/storage_20261006/<实验>/<任务>/<运行>.tar.gz`，工作目录中的旧日志、分片样本和逐规模测试文件已清理。588 份原档已检查完整性。剩余 1 路正在运行的 V10.19 CVRP 由临时转换器同步，完成后自动归档。迁移明细见本机的 `.archive/storage_20261006/report.json` 和 `storage_summary.json`。

这次涉及的已封存原始内容约 45.5 GiB；当前格式文件约 3.8 GiB，保留的压缩原档约 13.0 GiB。上述是文件内容大小，不是精确的磁盘释放量。备份时同时保存当前运行目录、批次清单和 `.archive/`。

需要原格式证据时，把对应压缩包解到单独目录，不覆盖当前结果。现有硬链接恢复清单仅描述迁移前布局；已经转换的档案应从压缩原档恢复，见[结果存储与恢复](../experiments/infra/RESULT_STORAGE.md)。

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
