# 第三方提示来源

本页保留三组提示的来源版本，供追溯使用。来源版本不等于对本地实现忠实性的验证。

## 来源与本地适配

| 本地内容 | 来源版本 | 原来源记录说明的适配 |
| --- | --- | --- |
| `pathwise/prompts/` | [PathWise · 9d014a36](https://github.com/oguzhangungordu/PathWise/tree/9d014a368d920d46ef4538da4c484503549214cb) | 按该版本复制提示。本地适配器提供任务描述、函数签名与计分方向。 |
| `reevo/prompts/` 中的任务知识 | [ReEvo · 6dce1825](https://github.com/ai4co/reevo/tree/6dce18257da5e11db2d138e417a2fffc5c72d05f/prompts) | 任务名称映射到本地基准，没有新增任务知识。 |
| `shinka_evo/prompts/` | [ShinkaEvolve · 8adc053a](https://github.com/SakanaAI/ShinkaEvolve/tree/8adc053a2ce4511ad2ac310e004c530a73fb974a) | 按该版本复制提示。本地适配器提供任务信息，类型导入指向本地程序记录。 |

链接保留完整提交标识。表中适配说明来自封存前的来源记录；本次未逐文件重新比对上游内容。

## 复查一次实验

先定位运行配置和源码快照，再对照该方法的上游版本。任务接口、提示适配和搜索流程分别核对，不能仅凭来源版本判断三者一致。

本次只整理来源文档，没有修改提示或许可文件。共同实验条件见[实验条件](../experiments/PROTOCOL.md)。
