# E2-B'：Pivot 两步 option value

本目录执行固定锚点随机干预：在 experienced 与 fitness-matched fresh 锚点上配对比较 Refine→Refine 和 Pivot→Refine。第一步有效的候选无论涨跌，都获得一次使用正常 V10.6 形成历史和 Implementation Summary 的 Refine。

实验配置见 e2b_config.json，详细识别边界见机制实验文档。运行产物写入被 Git 忽略的 experiments/traceaad_e2_b/raw/traceaad_e2_b_20260907。

执行入口：

    .venv/bin/python -m experiments.traceaad_e2_b.e2b prepare
    .venv/bin/python -m experiments.traceaad_e2_b.e2b run-shard --backend local --shard-index 0 --num-shards 1
    .venv/bin/python -m experiments.traceaad_e2_b.e2b profile --workers 12
    .venv/bin/python -m experiments.traceaad_e2_b.e2b analyze

run-shard 可安全续跑；不同 shard 按 anchor block 划分，一个 anchor 的两条路径始终由同一 backend 执行。
