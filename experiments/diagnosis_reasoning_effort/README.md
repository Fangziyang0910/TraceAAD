# 思考模式与生成交付诊断

诊断已完成，用于确定[主实验的模型与思考设置](../../docs/08-实验规范与共识/README.md)。输入为EoH／ReEvo六任务的12个原始首请求，每提示分别生成high、low、off一次；客户端不发送max_tokens。每次保存提示hash、实际采样、完整回答与reasoning、API usage及解析结果。

## 证据与结果

| 检验 | 控制条件与观察 | 记录 |
| --- | --- | --- |
| 原阶段采样下的模式配对 | 保留EoH温度1.0、ReEvo初始化1.3；36次均stop。high有最终回答为空，off两例出现多语言混杂回复 | [逐例CSV](../../experiments_result/diagnosis_deepseek_high_20261011/uncapped_high_low_off/comparison.csv)、[汇总](../../experiments_result/diagnosis_deepseek_high_20261011/uncapped_high_low_off/comparison.json) |
| 异常提示温度复测 | 同两个off提示分别用1.0和0，各一次，均正常交付。温度是首要待核变量，尚未证明唯一原因 | [温度1.0](../../experiments_result/diagnosis_deepseek_high_20261011/off_temperature_1_control/manifest.json)、[温度0](../../experiments_result/diagnosis_deepseek_high_20261011/off_temperature_0_control/manifest.json) |
| 官方采样下的模式配对与评价 | 统一1.0／0.95，共36次生成；可解析代码在固定训练集评价，思路格式与代码有效性分开记录 | [成本、逐任务成绩及评价复核](../../experiments_result/diagnosis_deepseek_high_20261011/official_sampling_evaluated/REPORT.md) |

原始回复、代码和逐实例评价位于对应mode/case目录。缺失API usage或reasoning明细按缺失记录，不补成零。两例worker退出状态0用原代码单独复核，初次失败保留。每提示每模式一次，结果描述本批交付、成本与成绩，不代表完整搜索收益。

## 复现入口

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m experiments.diagnosis_reasoning_effort.run \
  --inputs experiments_result/diagnosis_deepseek_high_20261011/paired_inputs.json \
  --output experiments_result/diagnosis_deepseek_high_20261011/new_evaluated_comparison \
  --temperature 1.0 --evaluate --workers 12 --eval-workers 8 \
  --scheduler-socket /tmp/traceaad-1005/scheduler.sock
```

去掉--temperature覆盖和--evaluate可复现原阶段采样下的生成诊断。结果目录必须是新目录；输入保存在本机，远端复现需先同步输入。评价使用现行六任务协议，生成耗时与评价耗时分别记录；代码不作修复，也不从reasoning回收最终程序。
