# 测试

运行：`uv run python -m pytest -q tests`。

V12.0 机制检查：`uv run pytest -q tests/method/test_traceaad_v12_*.py`。接续机制、基础搜索与短程开发各自维护一组测试；公共评价只经 `experiments.infra.evaluate` 检查，旧入口与旧提示快照已移除。

- `task/`：任务数据、候选函数与评价口径。
- `method/`：搜索机制、预算和续跑等行为。
- `experiments/`：实验配置、调度和结果产物。
- `base/`：共享的模型接口与安全评价。

新增测试优先使用小规模、固定随机种子的输入，检查可观察的结果和失败行为。实验协议中必须固定的设置可以直接断言；普通实现细节和源码措辞不作为测试目标。需要共用的假对象放在 `tests/support.py`，测试文件之间不相互导入。
