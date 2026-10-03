# 生成格式实验：写代码前的决策与 Design

结论与解读见 [研究记录](../../docs/03-现象与检验/2026-10-01-写代码前的决策与Design格式.md)。原始结果在 `experiments_result/format_study/`。

所有实验都在同一批父代上做配对比较，只改变回复格式，并用训练评价器评分（超时放宽为 120 s，比较格式而不研究超时）。

```bash
# DeepSeek v4.1 flash（thinking 关闭）：说明的长度与位置（Refine），参考卡长度（Explore）
FORMAT_STUDY_KEY=... uv run python -m experiments.format_study.deepseek_length_order ab
# Qwen（server3/3b）第一轮：6 种格式，Refine
uv run python -m experiments.format_study.qwen_round1 20
# Qwen 第二轮：分析长度/内容与 Design 长度（Refine），再迁移到 Explore/Crossover
uv run python -m experiments.format_study.qwen_round2 refine 25
uv run python -m experiments.format_study.qwen_round2 transfer 15 design_only few options target_options
# 分析
uv run python -m experiments.format_study.analysis deepseek
uv run python -m experiments.format_study.analysis qwen experiments_result/format_study/qwen_round2/refine.jsonl bytask
```

父代取自 V10.15-2（DeepSeek 实验）与 V10.15-4（Qwen 实验）的档案；随机种子写在各脚本中。
