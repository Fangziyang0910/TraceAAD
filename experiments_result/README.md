# 实验原始结果

`runs/<实验>/<任务>/<运行名>/` 保存运行配置、训练轨迹、最佳程序及测试结果；有些历史实验还在该实验目录下保存单独的 held-out 批次或离线画像原始数据。实验级计划与汇总（如 `batch_*.json`、`schedule.json`、`diversity_summary.json`）也保存在实验目录下。

`runs/` 是本机数据，Git 不跟踪。正式结论和逐路数值见 [实验结果](../docs/02-实验结果/)；共享运行入口见 [实验脚本](../experiments/README.md)。迁移或备份原始数据时应连同运行目录的 `run_config.json` 一起保存。
