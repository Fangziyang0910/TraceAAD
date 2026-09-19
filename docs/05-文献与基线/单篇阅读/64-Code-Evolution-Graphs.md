# Code Evolution Graphs

- 论文：*Code Evolution Graphs: Understanding Large Language Model Driven Design of Algorithms*；本地来源：LaTeX 源码目录 `Code_Evolution_Graphs_Understanding_Large_Language_Model_Driven_Design_of_Algorithms/`；研究对象：LLM 进化框架中代码结构与复杂度的演化轨迹，诊断工具而非新方法。

## 1. 核心问题与方法

CEG 定义为有向图 $G=(V,E)$：节点 $v_i=(f_i,x_i,m_i)$——归一化性能、标准化 AST 特征向量（约 20–26 个：AST 图度量如节点/边数、度分布、度/深熵、聚类、直径、assortativity；复杂度如圈复杂度、token 数、参数个数，Lizard + 自定义提取）、元数据（id、评价序号、父 id）；边 = 进化谱系父子关系。分析用 PCA（22 个 AST 图特征）/t-SNE 投影与 token 数曲线。任务：BBOB（24 函数 5D、3 实例 × 3 次）、OBP、TSP（GLS 特征启发式）；框架 LLaMEA（1+1 与 HPO 变体）与 EoH（4+20）。

## 2. 论文宣称的机制贡献（逐项）

- 不同 LLM 有可测的"代码指纹"：不同模型生成的解在特征空间占据互不重叠的区域；同一配置的独立运行也基本不重叠。
- 代码复杂度随反复提示单调上涨（token 数最明显，LLaMEA 1+1 与 GPT-4o 上最强）——LLM 变异默认让代码变长。
- 复杂度与性能的关系按任务反转：BBO 与 OBP 上正相关（更复杂=更好），TSP 上负相关（更简单=更好）。

## 3. 实验究竟支持了什么

|主张|论文证据|证据等级|判断|
|---|---|---|---|
|模型代码指纹不重叠|多框架×多 LLM 的 PCA/t-SNE 投影；只有 Random Search 的运行重叠|直接支持|支持"多 LLM 组合即跨结构探索"；缺多 LLM 协同搜索的直接实验。|
|复杂度单调上涨|token 数/圈复杂度随评价序号曲线|直接支持|与 Behavior Space 的 L4 结论互补：需要显式简化机制对冲。|
|复杂度-性能关系任务反转|逐任务特征-fitness 相关表|部分支持|相关性非因果；作者自认静态 AST 特征看不到超参与运行行为（配置差的算法与调好的算法 AST 特征相同）。|
|停滞模式可诊断|TSP 上 LLaMEA-HPO 前 20 次评价近优、其后无改进；TSP 上 EoH 无任何特征与 fitness 强相关（改动在发生但不影响性能）|直接支持|给出两种失败模式的判别：改动无效型停滞 vs 早熟收敛到顶；前者应换簇/换意图，后者特征-性能相关性衰减是在线可测信号。|

## 4. 机制的底层逻辑（阅读分析，不是作者已证明结论）

"加代码"是 LLM 变异的默认利用方向之一；复杂度增长本身无方向性善恶，任务决定其价值。EoH（4+20）的 CEG 连通密集但变异在 PCA 空间彼此靠近——结构上"看不到明显探索"，与 Mutation Without Variation 的骨架坍缩互证。

## 5. 对 LLM4AD / TraceAAD 可学习之处

- 可学习点：在线监控"代码特征-fitness 相关性"的衰减作为换路线信号。前提：特征提取廉价稳定。风险：特征与行为错位。最小验证：对停滞 run 与持续改进 run 分别画相关性曲线，检验分界。
- 可学习点：把简化算子（EoH-M3、LLaMEA refine&simplify）视为对冲复杂度漂移的必要组件。最小验证：关/开简化算子对比最终复杂度与 held-out 表现。
