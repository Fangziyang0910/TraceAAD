# TraceAAD V19.4

V19.4以演化轨迹支持算法改进、全局经验学习与节点选择。研究定义见[机制设计](../docs/机制设计/V19.4.md)，输入与角色指令见[提示规范](../docs/机制设计/V19.4-上下文与提示词.md)。本包拥有独立的搜索、评价、模型客户端、状态与提示实现。

节点价值是子代成为新最好的概率：改进响应率乘以近期改进幅度超过该节点与当前最好差距的比例。响应率由近期256次开发评价估计，依次向一跳父子邻域和近期总体改进率收缩，各层强度2；保留5%均匀项。一次失败后修复产生两份评价事件。生成侧只给当前算法代码，历史以设计、生成分析与成绩呈现；轨迹与经验作为参考，指令要求分析改进方向并提出更好的算法思想。反思分两级。产生新源码或运行失败时进行短期反思：只读文字，说明这次尝试的思想以后怎样能实现得更好，形成一两句暂定提示，保存在该尝试上并随轨迹呈现。每累积十条新的短期反思（`--long-term-interval`），长期反思读取最近三十条短期反思及其实际结果（`--long-term-window`），重新形成完整的正负经验池：正经验是有希望的思想，负经验是实现思想时的陷阱。它不读取上一版经验池和当前最好算法。生成从当前版本的经验池按共同BM25语料检索；各版本、合并来源与使用结果均保留。

在仓库环境运行完整离线示例：

```bash
.venv/bin/python -m traceaad_v19_4 demo --run-dir /tmp/traceaad-v194-demo
```

示例使用脚本化回复和演示专用词数计数；加`--tokenizer`可检查实际tokenizer。正式搜索使用匹配模型的tokenizer。

单独安装可使用`pip install ./traceaad_v19_4`。接口接受用户提供的`Task`、模型和评价器；内置benchmark适配只在选择`--task`时加载。tokenizer必须是匹配所用模型的本地文件或缓存。

```bash
.venv/bin/python -m traceaad_v19_4 run --task tsp_construct \
  --run-dir /tmp/v194-tsp --base-url http://127.0.0.1:8001/v1 \
  --model Qwen3.8-27B --tokenizer /path/to/qwen_tokenizer.json \
  --budget 1000
```

该单独入口使用任务自身的评价条件。与现有主实验比较时，使用[实验入口](../experiments/traceaad_v19_4/README.md)统一整任务时限和独立测试。

起点选择默认为`--selection-mode trajectory`：按\(1/(\text{rank}+1)\)抽取`--candidate-count`个候选（默认4），模型并排读取它们的局部轨迹后选择。数值对照由`--selection-mode neighborhood|direct|rank`选择。`--reflection-mode trajectory|siblings|local|none`控制短期反思范围；`--no-generation-history`只去掉生成侧的原始演化材料，保留经验检索查询与反思规则。其余配置保持一致。

初始化执行五次基础尝试，已有有效算法时转入开发；空档案继续初始化。每次尝试共享一次修复，连续十次无评价即停止。每个有效新实现可以成为父代，同分和退步仍入档案。

原始记录包括`programs/`、`evaluations/`、`responses/`、`attempts/`、`calls/`、`experiences.json`和`state.json`。起点选择保存候选编号、预先抽定的备用节点、比较文字和选择状态；数值对照保存选中节点的质量、直接及邻域评价编号、响应与概率。响应在实际评价完成时冻结。恢复使用原目录、相同源码与配置，schema为1。
