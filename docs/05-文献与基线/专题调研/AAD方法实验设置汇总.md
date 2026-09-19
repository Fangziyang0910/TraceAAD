# AAD / AHD 方法实验设置汇总

## 范围

汇总 `../papers/` 中**做自动算法设计（AAD）或自动启发式设计（AHD）**、且有实验报告的工作。

证据来自各论文实验章节与附录；找不到的项标「论文未明确」。各论文协议不统一，数字不宜直接横向比较。

**预算口径与真实性核查原则**：
不同论文对搜索规模的控制轴截然不同。**宁可保留未知，不做无依据的统一**。严禁将“候选数、samples、LLM calls、代数、token、运行时间”强行换算并直接等同于真实启发式评估次数（eval）：
- 例如 A₂DEPT 的细项明确是 500 次 LLM 调用（calls），并非 500 次算法程序评测；
- EoH / MEoH 的“代数 × 算子 × 种群数”只能算得名义生成上限，无法自动证明实际评价次数相同（未披露语法错误拦截、超时丢弃或缓存去重是否消耗评价）；
- HSEvo 主报 425K tokens，InstSpecHH 主报 800 LLM queries/子类，CORAL 主报 100 iter 或 3h 墙钟，均无法直接折算为 eval。

因此，预算汇总严格区分为“按原文记录的原始指标”与“真实评价核查”，避免在不可比的基准上形成虚假公平结论。

**Tasks 口径**：下表与各节「Tasks」均统计**主文实验 + 附录实验**中出现的问题/设定；同问题不同嵌入框架（如 TSP-构造 / TSP-GLS / TSP-ACO）分开列出。仅做跨规模测试或增加同库实例、不构成新题时，写在括号内。

## 总览

### 1. 各方法预算口径与评价真实性核查表

| 方法 | 原文预算数值 | 原始单位 | 是否能确定真实评价次数 | 失败、修复、初始化是否计入 |
|---|---|---|---|---|
| FunSearch | 约 10⁶ | samples (programs) | 可推导（分布式架构评估约 10⁶ 样本） | 语法/运行错误丢弃不计有效样本；初始化计入（原文 §Methods） |
| EoH | BPP: 2000; TSP/FSSP: 1000 | generations × operators × pop | 未知（仅为名义上限，实际真实 eval 未逐一记录） | 未披露（语法错误/超时丢弃是否计入未明确） |
| ReEvo | 100 | evals (evaluations) | 已明确（以 100 次启发式评估为显式预算上限） | 初始种群（30）计入；运行失败异常由框架处理（原文 §4.1） |
| MEoH | BPP: 2000; TSP: 1000 | generations × operators × pop | 未知（名义上限，实际有效评价次数未记录） | 未披露 |
| HSEvo | 约 425K | tokens | 未知（以 LLM token 耗尽为停止条件，未按 eval 计数） | 未披露 |
| MCTS-AHD | T = 1000 | evals (iterations/evaluations) | 已明确（固定 1000 次模拟/评估） | 语法错误是否重试或扣减未披露 |
| PathWise | n_e = 500（辅以上限 6h 墙钟） | evals | 已明确（以 500 次环境反馈/评估为预算） | 计入初始与规划展开 |
| CALM | T = 500 rounds（声称对齐约 1000 eval） | rounds / samples | 可推导（每轮采样 G 个 response 评估并打分） | 不可行输出给予层级负奖励并计入训练，但不进入种群（原文 §3.2） |
| CDEoH | 200 | samples (采样预算) | 可推导（以 200 个生成样本评估为上限） | 未披露 |
| QUBE | OBP: 80K; Cap set: 2M; TSP: 2K | samples | 可推导（沿用 FunSearch 样本流） | 语法错误丢弃，沿用 FunSearch 机制 |
| RedAHD | ≤ 1000 | evals | 已明确（对齐 MCTS-AHD 的 1000 次评估） | 未披露 |
| RoCo | 400 | calls / evals（原文称 400 calls/代，最大评估 400） | 已明确（以 400 次评估为上限） | 初始种群 30 计入 |
| HiFo-Prompt | 约 200 | LLM requests | 未知（原始单位为 request，未证明与真实 eval 1:1） | 未披露 |
| A₂DEPT | N = 500（父预算 K = 5） | LLM calls | 未知（原文明确为 500 次 LLM 调用，真实 eval 未披露） | 包含自修复循环，未明确修复调用是否消耗 N |
| AHD Agent | 约 30（w/SR 扩展至 100） | evals / steps | 已明确（强化学习在线设计时评估严格受限） | 离线训练阶段交互消耗已另外分账 |
| EoH-S | N_max = 2000 | evals (LLM4AD 框架设定) | 可推导（LLM4AD 框架计数器） | 框架过滤无效程序 |
| PoH | 10 iterations, w=5, d=5 | iterations / proposals | 可推导（主文对比表显式标注 60 eval） | 未披露 |
| Hercules | 100 | evals | 已明确（对齐 ReEvo 的 100 次评估协议） | 初始种群 15 计入 |
| MoH | 1000 | evals | 已明确（双层演化总评估 1000 次） | 未披露 |
| InstSpecHH | ≤ 800 / 子类 | LLM queries / 子类 | 未知（原始单位是 query，全题 4500/675 子类总真实 eval 极大） | 未披露 |
| EvoTune | 9.6k / 16k / 22.4k | samples | 可推导（island 生成样本并评估） | 离策略 DPO 训练另外消耗，已分账 |
| Fine-tune AAD | N_max = 2000 | evals | 可推导（微调后用 EoH/FunSearch 评测，名义上限 2000） | 数据集收集阶段生成/评估未计入搜索 |
| AlphaEvolve | 任务定制 | iterations / 墙钟 / runs | 未知（无统一定量 eval 报告） | 异步并行与验证器过滤 |
| ShinkaEvolve | Circle 150; AIME 75; ALE 50; MoE 20 | generations / evals | 已明确（各域设定固定轮数/评估上限） | 未披露 |
| DeltaEvolve | 100 | iterations | 未知（3 islands × 种群 40，真实 eval 显著高于 100） | 未披露 |
| CORAL | 100 iter 或 3h | iterations / 墙钟时间 | 未知（多智能体自由交互与调用工具） | 未披露 |
| BaSE | 512 | LLM calls | 未知（原始为 LLM 调用，以有效 FLOPs 为对齐轴） | 未披露 |
| AutoEP | 非启发式评估预算 | 在线 ELA + 调参 steps | 不适用（优化对象为超参，非生成代码评估） | 不适用 |

### 2. 任务范围、LLM 底座与对比基准总览

| 方法 | Tasks（主文+附录） | LLM | 常见对比 |
|---|---|---|---|
| FunSearch | Cap set；Admissible sets；Online BPP；附录 Corners、Shannon capacity | Codey/StarCoder | First/Best Fit；文献下界 |
| EoH | Online BPP（构造）；TSP-GLS；FSSP-GLS（测 TSPLib/Taillard） | GPT-3.5 | FunSearch；手工/NCO |
| ReEvo | TSP-GLS；TSP/CVRP/OP/MKP/Offline BPP（ACO）；DPP（GA）；TSP-构造；TSP/CVRP（NCO attention） | GPT-3.5 | KGLS、DeepACO、EoH、POMO/LEHD |
| MEoH | Online BPP（构造）；TSP-GLS（测 TSPLIB；含多目标变体） | GPT-3.5 | FunSearch、EoH |
| HSEvo | Online BPP（构造）；TSP-GLS；OP-ACO | GPT-4o-mini | FunSearch、EoH、ReEvo |
| MCTS-AHD | 主文：TSP/KP/Online BPP（构造）；TSP/CVRP/MKP/Offline BPP（ACO）；BO-CAF；附录：ASP（构造）；TSP-GLS | GPT-3.5/4o-mini | FunSearch、EoH、ReEvo、HSEvo、NCO |
| PathWise | 主文：TSP/KP（构造）；TSP/CVRP/MKP/OP/Offline BPP（ACO）；附录：Online BPP（构造）；TSP-GLS | GPT-4o-mini/GPT-5-nano | 上列 + MCTS-AHD |
| CALM | Online BPP（构造）；TSP（构造）；CVRP-ACO；OP-ACO | Qwen2.5-7B | FunSearch…MCTS-AHD、EvoTune |
| CDEoH | Online BPP（构造）；TSP（构造） | DeepSeek-Chat | EoH、FunSearch、ReEvo |
| QUBE | Online BPP；Cap set；TSP-GLS | OpenCoder-8B 等 | FunSearch、EoH |
| RedAHD | TSP；CVRP；KP；MKP；Online BPP；Offline BPP（测 TSPLib） | GPT-4o-mini | MCTS-AHD 协议下多基线 |
| RoCo | TSP/OP/CVRP/MKP/Offline BPP（ACO）；TSP-GLS；白盒/黑盒 | GPT-4o-mini | EoH、ReEvo、HSEvo、MCTS-AHD |
| HiFo-Prompt | 主文：TSP（构造+GLS）；Online BPP；FSSP；附录：BO-CAF（测 TSPLib/Taillard） | Qwen2.5-Max | FunSearch…MCTS-AHD |
| A₂DEPT | MIS；CVRP；CFLP；FJSP；CEVRPTW；MRCPSP（构造为主；另评 GLS/开放 AAD） | DeepSeek / Gemini | FunSearch…MCTS-AHD、Gurobi |
| AHD Agent | TSP/CVRP（构造+ACO）；OVRP（构造）；OP/MKP（ACO）；CAF | Qwen3-4B | ReEvo、EoH、MCTS-AHD、CALM |
| EoH-S | Online BPP；TSP；CVRP（测 BPPLib/TSPLib/CVRPLib） | DeepSeek-V3 | FunSearch、EoH、ReEvo、MCTS-AHD 等 |
| PoH | TSP-GLS；FSSP-GLS（测 TSPLib/Taillard） | GPT-4 | EoH、ReEvo、NCO、手工 |
| Hercules | TSP-GLS；TSP-构造；ACO：BPP/MKP/OP/TSP；附录 ACO-CVRP；NCO attention：POMO/LEHD×TSP/CVRP | 多 LLM | Random、EoH、ReEvo |
| MoH | TSP-构造/GLS/KGLS；Online BPP；CVRP-ACO；Offline BPP-ACO；附录 Acrobot、QAP（测 TSPLib/Cluster） | GPT-4o-mini | FunSearch…MCTS-AHD |
| InstSpecHH | Online BPP（子类）；CVRP（子类） | DeepSeek 系 | EoH、ReEvo、手工 |
| EvoTune | Online BPP；TSP；Flatpack；Hash Code datacenter/rides；LLM-SR×2 | Llama/Phi/Granite 1–2B | FunSearch-style |
| Fine-tune AAD | ASP；TSP（构造）；CVRP（构造） | Llama/Pangu + DPO | base LLM；EoH/FunSearch |
| AlphaEvolve | 矩阵乘；数学构造套件（50+，附录全表）；Borg 调度；Gemini kernel；TPU RTL；FlashAttention/XLA | Gemini Flash+Pro | 任务 SOTA |
| ShinkaEvolve | Circle packing；AIME scaffold；ALE-Bench LITE；MoE LBL | 多模型 UCB | AlphaEvolve、OpenEvolve、EoH |
| DeltaEvolve | BBOB；Hexagon Packing；Symbolic Regression；PDE Solver；Efficient Convolution | LLM 集成 | Parallel Sampling、AlphaEvolve |
| CORAL | Math×6（Circle Packing、Signal Processing、Erdős Overlap、MMD-16-2、MMD-14-3、3rd Autocorr）；System×5（EPLB、PRISM、LLM-SQL、Txn Sched.、Cloudcast）；Stress×2（Kernel Engineering、Polyominoes） | Claude Opus 等 | OpenEvolve、ShinkaEvolve、EvoX |
| BaSE | Circle Packing；MinMaxDist；Heilbronn Triangle | Qwen3 / Llama | OpenEvolve、ShinkaEvolve、greedy |
| AutoEP | TSP；CVRP；FSSP；UAV-IoT | Qwen3-30B | PT、GLEET、EoH、ReEvo |

## FunSearch

- **论文目录**: `Mathematical_discoveries_from_program_search_with_large_language_models/`
- **Tasks**:
  - **主文**：Cap set；Admissible sets；Online BPP（OR1–OR4；Weibull）
  - **附录/SI**：Corners problem；Shannon capacity of cycle graphs
  - **合计**：Cap set；Admissible sets；Online BPP；Corners；Shannon capacity
- **配置**: LLM=Codey（附录对比 StarCoder）；约 15 samplers + 150 CPU evaluators；样本量级约 10⁶；island 模型，按 signature 聚类；prompt 内 k=2 个程序
- **对比方法**: Cap set 文献下界；BPP：First Fit、Best Fit
- **指标**: Cap set 大小；BPP 相对 L2 下界的 excess bins
- **备注**: 在固定 skeleton 内进化 `priority` 函数；后续 AHD 论文常作基准

## EoH

- **论文目录**: `EoH/`（与 `Evolution_of_Heuristics_...` 同源）
- **Tasks**:
  - **主文**：Online BPP（构造评分）；TSP-GLS；FSSP-GLS
  - **附录**：无新题（同三题更多规模 / TSPLib / Taillard）
  - **合计**：Online BPP；TSP；FSSP
- **配置**: GPT-3.5-turbo；20 代；种群 20（BPP）/10（TSP、FSSP）；E1/E2 父代 p=5；约 2000 LLM 查询（BPP）；TSP/FSSP 局部搜索 ≤1000 迭代、≤60s/实例
- **对比方法**: BPP：First/Best Fit、FunSearch；TSP：NI、FI、OR-Tools、AM、POMO、LEHD；FSSP：GUPTA、CDS、NEH、NEHFF、PFSPNet、PFSPNet_NEH
- **指标**: BPP excess bins；TSP gap%；FSSP 相对 makespan
- **备注**: BPP 为构造评分函数；TSP/FSSP 嵌 **GLS**

## ReEvo

- **论文目录**: `ReEvo/`
- **Tasks**:
  - **主文**：TSP-GLS；TSP/CVRP/OP/MKP/Offline BPP（ACO）；DPP（GA 交叉/变异）；TSP-构造（TSPLIB）；TSP/CVRP（NCO attention reshape，POMO/LEHD）
  - **附录**：无新题（设置、问题定义、生成启发式）
  - **合计**：TSP；CVRP；OP；MKP；Offline BPP；DPP
- **配置**: gpt-3.5-turbo；temperature=1（初始化 +0.3）；种群 10；初始代 30；最大评估 **100**；交叉率 1、变异率 0.5；每 COP 设置 3 次；在 64 个独立测试实例上评价最优启发式
- **对比方法**: KGLS、EoH、NeuOpt、GNNGLS、NeuralGLS；ACO 专家启发式与 DeepACO；DevFormer；GHPP；POMO/LEHD
- **指标**: gap%、目标值、时间；ACO 进化曲线
- **备注**: 强调样本效率（≤100 eval）；白盒/黑盒 prompt

## MEoH

- **论文目录**: `MEoH/`
- **Tasks**:
  - **主文**：Online BPP（构造）；TSP-GLS
  - **附录**：无新题（同两题多目标/3 目标变体；更大随机实例与 TSPLIB）
  - **合计**：Online BPP；TSP
- **配置**: GPT-3.5-turbo；20 代；种群 20/10；交叉 5 父代；TSP GLS ≤1000 迭代、≤60s；3 次重复
- **对比方法**: FunSearch、EoH
- **指标**: gap 与运行时间双目标；HV、IGD
- **备注**: 单次运行产出非支配启发式集

## HSEvo

- **论文目录**: `HSEvo/`
- **Tasks**:
  - **主文**：Online BPP（构造）；TSP-GLS；OP-ACO
  - **附录**：无新题
  - **合计**：Online BPP；TSP；OP
- **配置**: gpt-4o-mini；temperature=1；最大约 **425K tokens**；初始种群 30、其后 10；Harmony Search 相关超参见附录；TSP-GLS 超时 100s，其余 50s；3 次
- **对比方法**: FunSearch、EoH、ReEvo
- **指标**: Obj；多样性 CDI/SWDI
- **备注**: 嵌 GLS（TSP）与 ACO（OP）

## MCTS-AHD

- **论文目录**: `MCTS-AHD/`
- **Tasks**:
  - **主文**：TSP/KP/Online BPP（构造）；TSP/CVRP/MKP/Offline BPP（ACO）；BO-CAF
  - **附录**：ASP（构造）；TSP-GLS
  - **合计**：TSP；KP；Online BPP；Offline BPP；CVRP；MKP；ASP；BO-CAF
- **配置**: GPT-3.5-turbo 与 GPT-4o-mini；评估预算 **T=1000**；N_I=4；单启发式在 D 上 60s；3 次
- **对比方法**: 手工启发式、GHPP、POMO、DeepACO；FunSearch、EoH、ReEvo、HSEvo
- **指标**: Obj、Gap%
- **备注**: 后续多篇工作沿用其数据与对比协议

## PathWise

- **论文目录**: `PathWise/`
- **Tasks**:
  - **主文**：TSP/KP（构造）；TSP/CVRP/MKP/OP/Offline BPP（ACO）
  - **附录**：Online BPP（构造）；TSP-GLS
  - **合计**：TSP；KP；Online BPP；Offline BPP；CVRP；MKP；OP
- **配置**: GPT-4o-mini、GPT-5-nano；temperature=1.0；N_a=2, N_w=2, N_p=6, I_max=3；主文评估预算 **n_e=500**；60s；3 次；单 run 搜索上限约 6h
- **对比方法**: 手工/NCO；FunSearch、EoH、ReEvo、HSEvo、MCTS-AHD
- **指标**: Obj、Gap%、MRGI；进化曲线
- **备注**: 策略–世界模型–批评者 + entailment graph；training-free

## CALM

- **论文目录**: `CALM/`
- **Tasks**:
  - **主文**：Online BPP（构造）；TSP（构造）；CVRP-ACO；OP-ACO
  - **附录**：无新题
  - **合计**：Online BPP；TSP；CVRP；OP
- **配置**: INT4 Qwen2.5-7B-Instruct + GRPO；T=500 rounds；与 baseline 对齐约 1000 次启发式评估；API 变体用 GPT-4o-mini；3 次
- **对比方法**: Best-Fit、GC、ACO；POMO、DeepACO；FunSearch…MCTS-AHD；EvoTune（Optuna 复现）
- **指标**: Gap%、Obj
- **备注与对照规范**:
  - 联合演化启发式与 LLM 权重（强化学习闭环）。
  - **对照命名规范**：若外层对比未运行在线 GRPO 参数更新（使用固定底座模型），**必须始终标为 `CALM w/o GRPO` 或 `CALM（搜索框架版本）`**。
  - **严禁在结果叙述中简化成“击败了完整 CALM”**：消融实验显示 GRPO 微调是其性能跃升的核心支柱；未包含微调的基线仅代表其 verbal operators 与搜索流水线，不能代表完整 CALM 系统的上限。

## CDEoH

- **论文目录**: `CDEoH_Category_Driven_Automatic_Algorithm_Design_With_Large_Language_Models/`
- **Tasks**:
  - **主文**：Online BPP（构造）；TSP（构造）
  - **附录**：无新题
  - **合计**：Online BPP；TSP
- **配置**: DeepSeek-Chat；LLM4AD；最大采样预算 **200**；种群 10；保留 4 类 top-1，λ=0.7；每设置 10 次
- **对比方法**: EoH、FunSearch、ReEvo；消融 nocategory / noreflection
- **指标**: OBP excess gap；TSP 相对 LKH gap
- **备注**: 类别驱动、无交叉、强调并行

## QUBE

- **论文目录**: `QUBE_Enhancing_Automatic_Heuristic_Design_via_Quality_Uncertainty_Balanced_Evolution/`
- **Tasks**:
  - **主文**：Online BPP；Cap set（n=8）；TSP-GLS（TSP20/50/100）
  - **附录**：无新题（OR/Weibull 与 TSP 设定细节）
  - **合计**：Online BPP；Cap set；TSP
- **配置**: OpenCoder-8B（消融 Deepseek-coder-6.7B）；temperature=1.0；OBP 80K / Cap 2M / TSP 2K samples；islands=10（TSP=1）；10 次取最优
- **对比方法**: FunSearch、FunSearch*、EoH
- **指标**: excess ratio；cap set 规模
- **备注**: FunSearch island + UIQ 亲本选择与 reset

## RedAHD

- **论文目录**: `RedAHD_Reduction_Based_End_to_End_Automatic_Heuristic_Design_with_Large_Language_Models/`
- **Tasks**:
  - **主文**：TSP；CVRP；KP；MKP；Online BPP；Offline BPP
  - **附录**：无新题（TSPLib 等为同任务外推）
  - **合计**：TSP；CVRP；KP；MKP；Online BPP；Offline BPP
- **配置**: GPT-4o-mini，temperature=1；默认嵌简化 EoH；M=3，M_init=10，l=3；60s；3 次
- **对比方法**: IC：Greedy、POMO、FunSearch*、EoH*、MCTS-AHD*、手工；ACO：ACO、DeepACO、EoH*、ReEvo*、MCTS-AHD*
- **指标**: Obj 或 Gap%
- **备注**: 先学问题约简（LR）再在约简空间设计启发式

## RoCo

- **论文目录**: `RoCo_Role_Based_LLMs_Collaboration_for_Automatic_Heuristic_Design/`
- **Tasks**:
  - **主文**：TSP/OP/CVRP/MKP/Offline BPP（ACO）；TSP-GLS；白盒/黑盒
  - **附录**：无新题（五题定义与 GLS/ACO 设定）
  - **合计**：TSP；OP；CVRP；MKP；Offline BPP
- **配置**: GPT-4o-mini；协作 3 轮；API 预算 400 calls/代；最大评估 400；种群 10；初始种群 RoCo/ReEvo/HSEvo=30；角色温度 explorer 1.3 / exploiter 0.8 / 其余 1.0；3 次
- **对比方法**: ACO、DeepACO、EoH、ReEvo、HSEvo、MCTS-AHD
- **指标**: Obj
- **备注**: explorer / exploiter / critic / integrator 多角色

## HiFo-Prompt

- **论文目录**: `HiFo_Prompt_Prompting_with_Hindsight_and_Foresight_for_LLM_based_Automatic_Heuristic_Des/`
- **Tasks**:
  - **主文**：TSP（构造 + GLS）；Online BPP；FSSP
  - **附录**：BO-CAF；同题测 TSPLib / Taillard
  - **合计**：TSP；Online BPP；FSSP；BO-CAF
- **配置**: Qwen2.5-Max；种群 8；代数 CO=8 / BO=4；约 **200** LLM requests；3 次
- **对比方法**: LKH3、First/Best Fit、NEH；POMO、LEHD、PFSPNet_NEH；FunSearch…MCTS-AHD
- **指标**: Gap%、Obj、Time
- **备注**: Insight Pool（hindsight）+ Navigator（foresight）

## A₂DEPT

- **论文目录**: `A2DEPT_Large_Language_Model_Driven_Automated_Algorithm_Design_via_Evolutionary_Program_T/`
- **Tasks**:
  - **主文**：MIS；CVRP；CFLP；FJSP；CEVRPTW；MRCPSP
  - **附录**：无新题（FrontierCO / ESOGU / PSPLIB 细节；另评 GLS 与开放 AAD）
  - **合计**：MIS；CVRP；CFLP；FJSP；CEVRPTW；MRCPSP
- **配置**: DeepSeek v3.2 与 Gemini 2.5 Flash；预算 **N=500** LLM calls；父预算 K=5；整集墙钟 120s；标准 3 次、高约束 20 次
- **对比方法**: Gurobi、ILS；神经求解器；FunSearch、EoH、ReEvo、MCTS-AHD；另比 GLS/AAD 与 OPRO
- **指标**: Gap%±std；规模归一化适应度
- **备注**: 开放式完整求解器程序树（AAD），非固定模板内 AHD

## AHD Agent

- **论文目录**: `AHD_Agent_Agentic_Reinforcement_Learning_for_Automatic_Heuristic_Design/`
- **Tasks**:
  - **主文**：TSP/CVRP（构造 + ACO）；OVRP（构造）；OP/MKP（ACO）；CAF
  - **附录**：无新题（八域接口与设定）
  - **合计**：TSP；CVRP；OVRP；OP；MKP；CAF
- **配置**: Qwen3-4B + GRPO（500 steps）；设计时评估预算约 **30**（w/SR=100）；固定流程 baseline 100、CALM 150；5 次；对比亦用 GPT-4o、DeepSeek-V4-Flash
- **对比方法**: 手工 GC/ACO；ReEvo、EoH、MCTS-AHD；CALM
- **指标**: Validation Gap%；Eval 次数与成本
- **备注**: 工具增强多轮 agent + RL

## EoH-S

- **论文目录**: `EoH_S_Evolution_of_Heuristic_Set_using_LLMs_for_Automated_Heuristic_Design/`
- **Tasks**:
  - **主文**：Online BPP；TSP；CVRP（测试含 BPPLib / TSPLib / CVRPLib）
  - **附录**：无新题
  - **合计**：Online BPP；TSP；CVRP
- **配置**: DeepSeek-V3；LLM4AD；N_max=**2000**；种群 10；最终保留 10 个互补启发式；3 次
- **对比方法**: Random、1+1 EPS、FunSearch、EoH、MEoH、CALM、ReEvo、MCTS-AHD（含 Top10）
- **指标**: gap；互补性能指数 CPI
- **备注**: 演化互补启发式集合，而非单一最优启发式

## PoH（Planning of Heuristics）

- **论文目录**: `Planning_of_Heuristics_Strategic_Planning_on_Large_Language_Models_with_Monte_Carlo_Tree/`
- **Tasks**:
  - **主文**：TSP-GLS；FSSP-GLS
  - **附录**：无新题（测 TSPLib / Taillard 与提示细节）
  - **合计**：TSP；FSSP
- **配置**: GPT-4；base temp=0.0，optimizer=1.0；MCTS：10 iterations，width=5，depth=5，exploration=2.5；搜索评估用 TSP200 + 800 GLS iterations
- **对比方法**: OR-Tools、AM、POMO、LEHD、GNNGLS、KGLS、NeuralGLS、NeuOpt、EoH、ReEvo；FSSP 手工与 PFSPNet；搜索消融 MC/Beam/Greedy
- **指标**: Opt. gap%；相对 makespan；Time
- **备注**: 状态=启发式，动作=改进建议的 MCTS 规划

## Hercules / Hercules-P

- **论文目录**: `Efficient_Heuristics_Generation_for_Solving_Combinatorial_Optimization_Problems_Using_La/`
- **Tasks**:
  - **主文**：TSP-GLS；TSP-构造（测 TSPLIB）；ACO：BPP/MKP/OP/TSP；NCO attention：POMO×TSP/CVRP、LEHD×TSP/CVRP
  - **附录**：ACO-CVRP（黑/白盒）；其余为同题 case / 换 LLM
  - **合计**：TSP；BPP；MKP；OP；CVRP
- **配置**: temperature=1；最大评估 100；种群 15；3 次；协议大体对齐 ReEvo；多 LLM（GPT-4o-mini、GPT-3.5、Llama3.1-405b 等）
- **对比方法**: Random、EoH、ReEvo；KGLS、常规 ACO、POMO/LEHD
- **指标**: Gain（相对 seed）；gap；搜索时间与 token
- **备注**: Hercules-P 加性能预测以减评估

## MoH（Meta-Optimization of Heuristics）

- **论文目录**: `Generalizable_Heuristic_Generation_Through_LLMs_with_Meta_Optimization/`
- **Tasks**:
  - **主文**：TSP-构造；TSP-GLS；TSP-KGLS；Online BPP；CVRP-ACO；Offline BPP-ACO
  - **附录**：Acrobot；QAP；同题测 TSPLib / TSP200-Cluster
  - **合计**：TSP；Online BPP；CVRP；Offline BPP；Acrobot；QAP
- **配置**: GPT-4o-mini；外环 T=10；启发式与 meta-optimizer 种群各 10；评估预算 **1000**；TSP 在 20/50/100/200 上搜索，在 500/1000 上做跨规模测试；3 次
- **对比方法**: Concorde、OR-Tools、Nearest Neighbor、First/Best Fit；FunSearch、EoH、ReEvo、HSEvo、MCTS-AHD
- **指标**: Gap%、Obj
- **备注**: 强调跨尺度泛化与双层 meta-optimizer

## InstSpecHH

- **论文目录**: `LLM_Driven_Instance_Specific_Heuristic_Generation_and_Selection/`
- **Tasks**:
  - **主文**：Online BPP（4500 子类）；CVRP（675 子类）；intra/inter 7:3
  - **附录**：无新题
  - **合计**：Online BPP；CVRP
- **配置**: 种群 10；LLM queries ≤800；InstSpecHH 用 DeepSeek-R1-Distill-Qwen-14B，EoH/ReEvo 用 DeepSeek-V3；选择重复 5 次
- **对比方法**: Best/First Fit、Closest Priority；EoH、ReEvo；选择策略 Random/Closest/LLM/Classifier
- **指标**: Obj、Opt. Gap、LLM query 数、在线时间
- **备注**: 离线按子类建启发式池 + 在线算法选择

## EvoTune

- **论文目录**: `Algorithm_Discovery_With_LLMs_Evolutionary_Search_Meets_Reinforcement_Learning/`
- **Tasks**:
  - **主文**：Online BPP；TSP；Flatpack；Hash Code datacenter；Hash Code rides；LLM-SR material stress；LLM-SR bacterial growth
  - **附录**：无新题（扩展任务完整协议；validation / perturbed / test）
  - **合计**：Online BPP；TSP；Flatpack；HC-datacenter；HC-rides；LLM-SR×2
- **配置**: Llama3.2-1B / Phi-3.5-Mini / Granite-3.1-2B；6 islands；预算点 9.6k/16k/22.4k samples；周期性 DPO；主实验 10 seeds
- **对比方法**: FunSearch-style（无 RL）；LLM-SR 上另比更大闭源模型
- **指标**: Optimality gap；top-50 reward；unique scores
- **备注**: 演化搜索与离策略 DPO 交替

## Fine-tuning LLM for AAD（DAR + DPO）

- **论文目录**: `Fine-tuning-LLM-Automated-Algorithm-Design/`
- **Tasks**:
  - **主文**：ASP（n=15,w=10）；TSP（构造）；CVRP（构造）
  - **附录**：无新题
  - **合计**：ASP；TSP；CVRP
- **配置**: DPO + LoRA；Llama-1B/8B、Pangu-1B/7B；搜索侧 EoH/FunSearch N_max=2000，EoH 种群 20；3 次
- **对比方法**: base LLM；Top-k 采样；EoH/FunSearch 微调前后
- **指标**: 相对 best-known 的 gap；top-k 分布
- **备注**: 方法目标是提升 LLM 作为算法设计器的能力，而非新搜索框架

## AlphaEvolve

- **论文目录**: `AlphaEvolve/`
- **Tasks**:
  - **主文**：矩阵乘张量分解；数学构造套件（约 50+，含自相关/不确定性/Erdős 最小重叠/kissing/packing/Heilbronn 等）；Borg 调度；Gemini kernel tiling；TPU RTL；FlashAttention/XLA IR
  - **附录**：矩阵乘 54 组尺寸全表；数学题完整列表（三则自相关、不确定性、Erdős 最小重叠、有限集和差、正六边形装箱、最大/最小距离比、Heilbronn、11 维 kissing、圆 packing 等）
  - **合计**：矩阵乘；数学构造套件；Borg；Gemini kernel；TPU RTL；FlashAttention/XLA
- **配置**: Gemini 2.0 Flash + Pro；MAP-Elites 风格 + island；评估可高度并行；无统一 COP 代数/种群
- **对比方法**: 各任务既有 SOTA；消融 No evolution / No context 等
- **指标**: 张量秩、数学界、调度资源、runtime、面积/功耗等
- **备注**: 闭源、任务定制协议；偏科学发现与系统优化

## ShinkaEvolve

- **论文目录**: `ShinkaEvolve/`
- **Tasks**:
  - **主文**：Circle packing（26 圆）；AIME agent scaffold；ALE-Bench LITE（10 题）；MoE load-balancing loss
  - **附录**：无新题（同四域实现细节）
  - **合计**：Circle packing；AIME；ALE-Bench LITE；MoE LBL
- **配置**: 任务相关：Circle 150 代 / AIME 75 / ALE 50；多模型 UCB；温度集合 [0,0.5,1.0]
- **对比方法**: AlphaEvolve、OpenEvolve、LLM4AD/EoH；任务基线 scaffold / ALE-Agent
- **指标**: 半径和；AIME 准确率；ALE score；训练 CE/LBL 等
- **备注**: 开放框架；非标准 COP AHD 协议

## DeltaEvolve

- **论文目录**: `DeltaEvolve_Accelerating_Scientific_Discovery_through_Momentum_Driven_Evolution/`
- **Tasks**:
  - **主文**：BBOB；Hexagon Packing；Symbolic Regression；PDE Solver；Efficient Convolution
  - **附录**：无新题
  - **合计**：BBOB；Hexagon Packing；Symbolic Regression；PDE Solver；Efficient Convolution
- **配置**: LLM 集成（0.8 高吞吐 + 0.2 推理）；种群 40、archive 20、3 islands、max iter=100；种子 11/42/100
- **对比方法**: Parallel Sampling、Greedy Refine、AlphaEvolve（OpenEvolve 复现）
- **指标**: Best Score；Token Consumption
- **备注**: 强调相对 AlphaEvolve 的 token 效率

## CORAL

- **论文目录**: `CORAL_Towards_Autonomous_Multi_Agent_Evolution_for_Open_Ended_Discovery/`
- **Tasks**:
  - **主文**：Math×6 — Circle Packing、Signal Processing、Erdős Overlap、MMD-16-2、MMD-14-3、3rd Autocorr；System×5 — EPLB、PRISM、LLM-SQL、Txn Sched.、Cloudcast；Stress×2 — Kernel Engineering、Polyominoes
  - **附录**：无新题
  - **合计**：上述 13 项
- **配置**: Claude Code + Opus 4.6（主）；开源栈 MiniMax M2.5 + OpenCode；预算 3h 或 100 iter；多智能体 4 agents；4 次
- **对比方法**: OpenEvolve、ShinkaEvolve、EvoX；Bo4 / 1-Agent 对照
- **指标**: Final Score；Improvement Rate；#Evals
- **备注**: 自主 agent 共演化 vs 固定演化流水线

## BaSE（Compute Allocation）

- **论文目录**: `Compute_Allocation_in_Evolutionary_Search_From_Depth_Breadth_to_Multi_Armed_Bandits/`
- **Tasks**:
  - **主文**：Circle Packing（n=26）；MinMaxDist（n=16）；Heilbronn Triangle（n=11）
  - **附录**：无新题
  - **合计**：Circle Packing；MinMaxDist；Heilbronn Triangle
- **配置**: Qwen3 / Llama-3.1；扫 depth–breadth 网格；Bandit 跨轨迹分配；以有效 FLOPs 为成本轴
- **对比方法**: Greedy depth–breadth、best-of-N、OpenEvolve、CodeEvolve、ShinkaEvolve
- **指标**: 归一化 fitness；fitness–FLOPs envelope
- **备注**: 主贡献是固定预算下深度–广度与分配器，任务本身是算法/构造发现

## AutoEP

- **论文目录**: `AutoEP_LLMs_Driven_Automation_of_Hyperparameter_Evolution_for_Metaheuristic_Algorithms/`
- **Tasks**:
  - **主文**：TSP（TSPLIB）；点名 CVRP / FSSP / UAV
  - **附录**：CVRP（VRPLIB）；FSSP（Taillard）；UAV-IoT 轨迹
  - **合计**：TSP；CVRP；FSSP；UAV-IoT
- **配置**: AutoEP 用 Qwen3-30B；EoH/ReEvo 对比用 GPT-3.5；重复 30 次；在线 ELA + 多 LLM 协同调参
- **对比方法**: PT、GLEET、BEA；DACT、LEHD；EoH、ReEvo（含插件式组合）
- **指标**: Opt.gap%；运行时间
- **备注**: 主贡献是元启发**超参在线控制**，非直接演化启发式代码。本地 `manuscript.tex` 曾与综述源混淆；实验以 arXiv:2509.23189 为准

## 未纳入正文的论文（简述）

| 类型 | 目录示例 | 原因 |
|------|----------|------|
| 综述 | `A_Systematic_Survey_...`、`When_Large_Language_Model_Meets_Optimization` 等 | 无本方法实验协议 |
| 分析 | `Fitness-Landscape-...`、`Understanding_the_Importance_...`、`traj_evo_search`、`Behaviour_Space_...`、`Code_Evolution_Graphs_...` | 分析既有 AHD，非新方法主实验 |
| Benchmark | `CO-Bench`、`FrontierOR`、`Reasoning_in_a_Combinatorial_...` | 评测平台 |
| 端到端求解 | `Large_Language_Models_as_End_to_end_...`、`Bridging_...`、`MEGO_...`、`Structure_Aware_...` | 求解实例或结构优化，优化目标不是一类问题上的算法函数 |
| 其他 | `SE_Agent_...`、`PhyloEvolve`、`Evolution_through_Large_Models`、`Code_Repair_...` | SE / 内核 / 开放机器人 / 代码修复 |

早期相关但协议偏演示或通用算子：`Language_Model_Crossover_...`、`Evolving_Code_with_A_Large_Language_Model`（符号回归等），未展开。

## 协议差异（读结果时注意）

1. **生成器成本、搜索系统开销与候选求解成本必须彻底分开**：
   - **生成器成本（Generator Cost）**：LLM 本身的推理消耗，包括 LLM API 调用次数、Token 输入/输出量、生成模型推理延迟与 GPU 显存/FLOPs。它反映“产出一个候选算法消耗了多少大模型算力”。
   - **搜索系统墙钟与调度开销（Search Wall-clock & Orchestration Overhead）**：外层搜索流水线的系统耗时，包括端到端运行总墙钟时间（Wall-clock time）、并发 evaluator 数量、并行加速比、任务调度与通信开销。它反映“完成一次完整自动算法设计过程需要多少物理时间与并发资源”。
   - **被生成算法在问题实例上的求解运行成本（Candidate Solver Execution Cost）**：被设计出来的启发式代码在测试/验证实例上运行的开销，包括单个实例的 CPU/GPU 运行秒数、局部搜索迭代步数（如 GLS 1000 步）、ACO 蚂蚁代数、渐近时间复杂度等。它反映“该算法在实际应用中好不好用、快不快”。
   - **方法论警示**：Token 消耗大不代表算法求解慢，求解速度快也不代表生成器开销小。三者属于不同层级，必须分项独立报告，严禁混同为单一成本指标。

2. **不能把不同预算强行换算成 eval（宁可保留未知，不做无依据的统一）**：
   - 严格区分原文单位与真实评价次数：候选数、samples、LLM calls、代数不能直接读成真实 evaluator 调用次数。
   - 例如 A₂DEPT 的细项明确是 500 次 LLM 调用（calls），未经语法与可行性筛选前并非 500 次算法评估；EoH/MEoH 的“代数 × 5 算子 × 种群数”仅为名义生成上限，无法自动证明实际评价次数相同（语法错误拦截、超时丢弃或缓存去重是否计入通常未披露）；HSEvo（425K tokens）、InstSpecHH（800 queries/子类）、CORAL（100 iter 或 3h）等更无法简单折算为 eval。在跨论文分析时，必须保留其原始单位与未知状态。

3. **嵌入框架与 LLM 底座不同**：
   - 同一 TSP 可嵌构造、GLS 或 ACO，各框架的 Baseline 与求解质量不在同一量级，绝对 gap 不可直接横向比较；
   - 模型底座（GPT-3.5 / 4o-mini / DeepSeek / Qwen / Gemini 等）的先验与代码生成能力差异巨大，脱离基座模型单谈算法框架性能是不严谨的。

4. **反复观察 held-out 的用途定性与防泄漏原则**：
   - 部分文献与机制设计（如 `05-EoH.md` 建议“每次新增 best 立即 held-out 重测”）倡导在搜索进行中频繁触发 held-out 实例测试以监控泛化性能。
   - **用途定性**：若这些 held-out 重测的结果被研究者即时观察，并实质性地用于调试提示词模板、调整算子比例、修改代码结构、选择超参数或筛选方法版本，则该集合**已经实际承担了‘开发验证集（Development/Validation Set）’的角色**。
   - **严禁虚假声称**：在这种情况下，**绝不能继续宣称该集合是‘完全未参与研究决策的严格独立测试集（Pristine Held-out Test Set）’**。
   - **规范做法**：真正的最终泛化能力评测，必须在算法搜索逻辑与所有超参数完全冻结的前提下，仅对最终产出的最佳候选在独立的 Test 集上执行单次盲态评估；过程中的中间重测应如实披露为验证集观测。

5. **CALM 对照命名与叙述规范**：
   - CALM 的性能由 verbal operators（搜索框架）与在线 GRPO 微调共同支撑。当外部基准仅使用固定预训练模型复现其搜索流程时，**必须始终标为 `CALM w/o GRPO` 或 `CALM（搜索框架版本）`**。
   - **严禁在结果叙述中简化成“击败了完整 CALM”**：消融实验已证实去掉 GRPO 会导致性能显著下滑，未微调版本仅代表其启发式变异算子逻辑，不能代表完整 CALM 联合系统的能力边界。

6. **本仓库正式比较基准与口径**：
   - 本仓库正式比较已固定：**评估预算 = 1000 eval**；主表对比方法为 **EoH、ReEvo、MCTS-AHD、PathWise、CALM w/o GRPO**；主实验 task 为 **Online BPP、TSP-构造、CVRP-ACO、OP-ACO、VRPTW-构造**（见 实验配置）。
   - 其中 CALM 当前阶段跑 **CALM w/o GRPO（搜索框架版本）**，微调阶段再补 **完整 CALM（w/ GRPO）**。任务协议取舍：OBP 保持多容量现状；TSP-构造维持降采样（同标准重跑）；CVRP-ACO 测试含 **CVRP200×64**；OP-ACO 保持现状。统一比较时再固定嵌入框架 × LLM，并独立报告 LLM 调用与 token（生成器成本）以及算法运行时间（求解成本）。
