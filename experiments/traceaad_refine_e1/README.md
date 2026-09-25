# E1-A：V10.6 Refine 局部响应的历史回放

独立离线实验；不修改正式调度器、不调用生成模型、不写入搜索评价账本。完整研究协议与实验数据保存在 `experiments/traceaad_refine_e1/raw/refine_e1_20260907/`。

2026-09-07 批次快照已完成，E1-A 离线回放与 E1-A.1 质量条件行为增量检验均未通过筛选门槛。`report.py` 中的研究判断针对该批次；更换数据时须重新审读，不能沿用这些判断。参数保存在 `e1a1_config.json`，结果写入同一日志目录及文档目录。


## 实际执行协议

复制 15 路 `20260906_215231_revised` 运行在指定截点的 checkpoint 和对应已完成事件。每路截点与状态存于 `experiments/traceaad_refine_e1/raw/refine_e1_20260907/snapshot.json`。排除 smoke 和未完成 pending 请求。

所有 Refine parent 均安排画像，历史邻域只含当前时刻以前**曾被选择为 Refine parent**的节点，且主比较排除当前 parent。这样避免用最终“哪些节点以后会被选中”泄漏未来，也把画像成本限制在研究的父代集合。这里的密度是历史父代观测子集密度，不是全 archive 密度。M2/M3 采用同一可见节点规则。

画像核心复用删除前版本 `d234021251800e1ef9979993c799717ddbb0e577:experiments/analysis/behavesim_profiler.py` 的 PSTraj 与 DTW 代码。每面板四个 probe，构造任务保留12步，ACO保留5步；蚂蚁数和迭代数沿用各任务评价器。OP 面板 B 改为独立生成的训练探针（seed=20260907），CVRP 面板 B 改为训练集第4–7号实例，避免旧 v3 使用 validation 的口径。先验证两次画像相同，以及与正式评价器在相同输入、随机流上的得分一致，再批量执行。

embedding 对照固定为 [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)，revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`。通过 ONNX 在 CPU 上计算；对实现摘要与 AST canonicalized code 分别按240 tokens切块，池化所有块、分别归一化，再等权平均并归一化，避免仅保留长代码前缀。它是轻量通用表示基线，不代表所有代码表示模型。

预测采用各 run 独立的扩展窗口：严格先预测再读当前标签；每10次 Refine 用此前全部样本重拟合标准化逻辑回归，C=1，不调参；前50次为预热。没有利用别的 run 的未来反馈拟合模型。M3静态量、排除同代码、允许同父历史、质量距离邻域作辅助对照。主指标为逐 run 的配对 Brier 差，log loss 检查方向，PR-AUC、校准、top20% lift及超当前best为辅助。仅以原始Brier池化会让较长run占更高权重，因此同时报告run宏平均与按任务结果。

影子预测只描述旧策略访问状态上的信息价值，不是新控制器的离策略终局估计。事先固定筛选门槛为宏平均Brier相对改善至少2%、至少3/5任务同向且平均log loss不变差；即使通过也仍需E1-B确认。门槛不是显著性或因果结论。

## 命令

从仓库根目录执行。首次建立独立的embedding环境：

```bash
uv venv --python .venv/bin/python experiments/traceaad_refine_e1/raw/refine_e1_20260907/venv
uv pip install --python experiments/traceaad_refine_e1/raw/refine_e1_20260907/venv/bin/python numpy==2.4.6 onnxruntime==1.29.0 tokenizers==0.22.2 huggingface-hub==0.36.2
```

主流程：

```bash
.venv/bin/python -m experiments.traceaad_refine_e1.prepare
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 .venv/bin/python -m experiments.traceaad_refine_e1.profile validate
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 .venv/bin/python -m experiments.traceaad_refine_e1.seed_check
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 .venv/bin/python -m experiments.traceaad_refine_e1.profile profile --workers 12
experiments/traceaad_refine_e1/raw/refine_e1_20260907/venv/bin/python -m experiments.traceaad_refine_e1.embed
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m experiments.traceaad_refine_e1.replay
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m experiments.traceaad_refine_e1.auxiliary
.venv/bin/python -m experiments.traceaad_refine_e1.report
.venv/bin/python -m experiments.traceaad_refine_e1.e1a1
.venv/bin/python -m pytest -q tests/experiments/test_refine_e1.py
```

embedding 使用隔离的本地环境（Python3.11、onnxruntime、tokenizers、huggingface-hub、numpy），不修改项目依赖文件。首次运行需下载固定版本模型。`profile` 按运行名与节点 ID 复用已完成缓存，`embed` 按显式文本 ID 复用缓存；画像失败也记录，不以零距离替代。正式搜索与本实验的CPU/耗时分别记录。
