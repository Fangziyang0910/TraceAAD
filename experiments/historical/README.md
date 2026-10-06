# 历史研究脚本

这里保存已经结束的初始化对照与格式研究。当前搜索和训练监控不依赖这些模块。

- `initialization/analyze.py`：E32 三臂初始化汇总。
- `initialization/diversity.py` 与 `behavior.py`：训练集 A/B 行为探针。
- [format_study](format_study/README.md)：代码前文字的格式、长度与位置对照。

初始化分析命令：

```bash
uv run python -m experiments.historical.initialization.analyze
uv run python -m experiments.historical.initialization.diversity --help
```

初始化输入仍是 `experiments_result/traceaad_initialization/`；行为脚本使用它原来指定的历史批次。部分输入已经按要求从本机清理。复查这些研究时，先从原始归档恢复对应数据，不要用当前批次代替。脚本中的旧格式读取只服务这些历史研究；当前运行格式见 [SEARCH_FORMAT](../infra/SEARCH_FORMAT.md)。
