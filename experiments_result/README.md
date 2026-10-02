# 实验原始结果

`<实验>/<任务>/<运行名>/` 保存单路结果。TraceAAD 版本搜索统一使用 `run_config.json`（配置）、`search.jsonl`（模型调用、候选、评价与状态）和 `logs/run_summary.json`（终局摘要）。历史版本的 `search.jsonl` 每条记录包含 `kind`、`source` 和原始 `data`；同一来源保持原有行序，不表示不同来源之间有确定的发生顺序。没有原始终局摘要的运行标为 `status: unknown`。旧版状态快照保存在日志中，不能直接当作 V10.13 的续跑状态。

独立的 held-out 结果和实验级计划、汇总仍在对应实验目录中。

实验目录是本机数据，Git 不跟踪。V10.14 仅保留最终正式批次 `traceaad_v10_14_3/batch_20260928_v1014_3_template2.json` 及其 20 个运行目录；其余 V10.14 中断批次、替换批次和测试工件已删除。正式结论见 [V10.14 实验分析](../docs/03-机制探索与验证/2026-09-29-V10.14-正式实验结果与分析.md)；共享运行入口见 [实验脚本](../experiments/README.md)。迁移或备份原始数据时应连同运行目录的 `run_config.json` 一起保存。

## 已完成档案的去重

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

本轮清理范围和核验结果见[档案整理记录](../docs/02-实验结果/2026-10-03-本地实验档案整理.md)。恢复清单与原始数据保存在本机，迁移时一并保存；使用归档工具备份时，可保留硬链接关系，避免重新展开全部副本。
