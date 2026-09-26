# 实验原始结果

`<实验>/<任务>/<运行名>/` 保存单路结果。TraceAAD 版本搜索统一使用 `run_config.json`（配置）、`search.jsonl`（模型调用、候选、评价与状态）和 `logs/run_summary.json`（终局摘要）。历史版本的 `search.jsonl` 每条记录包含 `kind`、`source` 和原始 `data`；同一来源保持原有行序，不表示不同来源之间有确定的发生顺序。没有原始终局摘要的运行标为 `status: unknown`。旧版状态快照保存在日志中，不能直接当作 V10.13 的续跑状态。

独立的 held-out 结果和实验级计划、汇总仍在对应实验目录中。

实验目录是本机数据，Git 不跟踪。正式结论和逐路数值见 [实验结果](../docs/02-实验结果/)；共享运行入口见 [实验脚本](../experiments/README.md)。迁移或备份原始数据时应连同运行目录的 `run_config.json` 一起保存。
