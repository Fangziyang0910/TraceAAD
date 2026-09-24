# TraceAAD

科研仓库：以 LLM 驱动的自动算法设计（AAD）机制研究。协作规范见 [AGENTS.md](AGENTS.md)，文档导航见 [docs/README.md](docs/README.md)，实验导航见 [experiments/README.md](experiments/README.md)。

## 结构

| 目录 | 内容 |
| --- | --- |
| `traceaad/` | 机制模块，每版本一个子包：祖先链 `v10_3`→`v10_8`、`v10_10`、六文件族 `v10_11`/`v10_12`/`v11_0`/`v11_1`、消融 `rand_ctx`、组合 `bc`、早期对照 `v10_1`/`v10_2` |
| `baselines/` | 对比方法：`eoh` `reevo` `pathwise` `mcts_ahd` `calm` `shinka_evo`（冻结契约，只增不改）+ 共享设施 `observability.py`、`sampling.py`（SampleTrimmer）、`profiler/`（ProfilerBase） |
| `benchmarks/` | 五任务（冻结契约）：`tsp_construct` `cvrp_aco` `op_aco` `vrptw_construct` `online_bin_packing` + `generated_data_config.py` |
| `core/` | 共享核心（3 文件，保留 LLM4AD 署名）：`code.py` 代码表示、`evaluate.py` SecureEvaluator 安全评价、`llm.py` LLM 客户端 |
| `experiments/` | 各版本/基线的运行入口、freeze 冻结副本机制、监控与共享 `infra/` |
| `tests/` / `docs/` | 测试与研究文档 |

## 运行约定

- 一律从仓库根以 `.venv/bin/python -m experiments.<version>.run …` 运行（项目不安装自身，`uv` 仅管理依赖；`pyproject.toml` 中 `package = false`）。
- 正式搜索必须先冻结运行时（`freeze.py`），并使用返回的 runtime 以隔离 cwd/PYTHONPATH 启动；`verify_runtime` 做哈希与 spawn 预检。背景见 2026-09-22 worker 导入事故诊断（独立诊断已清理）。
- 测试：`.venv/bin/python -m pytest -q`。

## 相关工作（本地代码与论文）

相关工作代码与论文在仓库外的工作区 `/home/fang/code/LLM4AD/`（`reference_code/`、`papers/`）。
