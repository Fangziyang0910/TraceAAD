# 思考强度与代码交付诊断

使用 EoH、ReEvo 六任务的首请求作为12个配对输入，依次运行high、low、off。每种模式每个提示一次，提示、temperature、top_p保持原请求字段；客户端不发送max_tokens，服务端默认上限由实际finish_reason检查。原搜索实验保持停止。

```bash
uv run python -m experiments.diagnosis_reasoning_effort.run \
  --inputs experiments_result/diagnosis_deepseek_high_20261011/paired_inputs.json \
  --output experiments_result/diagnosis_deepseek_high_20261011/new_comparison \
  --workers 12
```

输入文件保存原提示、提示hash、原采样字段及任务模板。每个回复保存calls.jsonl、token_usage.json、result.json和可解析时的candidate.py。manifest.json逐次记录进展。错误／缺失API usage单独标记，不以0补算。

2026-10-11结果位于`experiments_result/diagnosis_deepseek_high_20261011/uncapped_high_low_off/`；汇总为comparison.json和comparison.csv，完整API回复包括原生reasoning保留在各case/calls.jsonl。

| 模式 | 平均思考token | 思考范围 | 平均最终回答token | 平均耗时 | 中位耗时 | 代码可解析 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| high | 16828.5 | 5336–26456 | 1207.8 | 146.0秒 | 132.3秒 | 11/12 |
| low | 10133.2 | 2034–19769 | 1279.0 | 72.6秒 | 75.2秒 | 12/12 |
| off | 0 | 0 | 3604.0 | 43.6秒 | 10.5秒 | 11/12 |

三组36次均finish_reason=stop，无输出截断。high的一例最终content为空，代码位于reasoning字段。off的ReEvo图着色出现多语言乱码、入口代码未完整交付；ReEvo-CVRP已有可提取代码，但随后产生大量乱码，最终回答20125 tokens、耗时327秒。故off的均值受单个长回复影响。

后续单独检查这两个off提示，将temperature从ReEvo初始化的1.3改为模型推荐1.0；两例分别10.3、13.6秒返回可解析代码。这两次不混入主36次比较，记录在`off_temperature_1_control/`。网关没有提供这两次API usage，token计数留空，仅比较交付和耗时。这提示温度是需要核对的变量，不能凭两次生成认定唯一原因。

这是生成长度、延迟和交付的诊断；没有运行完整搜索或比较求解成绩。每个提示每种模式仅一次，统计描述这批实际回复，不宣称一般质量优势。
