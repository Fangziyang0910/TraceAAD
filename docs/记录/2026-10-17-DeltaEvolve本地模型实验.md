# DeltaEvolve：本地模型与六任务实跑

2026-10-17，北京时间。批次：`20261017_qwen_co6`。

## 我为什么保留 Qwen3-Embedding-4B

我需要为 DeltaEvolve 的代码多样性网格提供稳定的本地表示。发布日期本身不能决定这个选择。[F2LLM-v2 论文表 2](https://arxiv.org/html/2603.19223v1#S4.T2)报告 4B 模型在 12 项代码任务上得分 80.15；[Perplexity 论文表 1](https://arxiv.org/html/2602.11151v2#S3.T1)报告 Qwen3-Embedding-4B 为 80.07、pplx-4B INT8 为 78.73。F2LLM 更新，但这些报告不足以证明它在本次机制中有明确收益。

我也核对了 2026 年 8 月的 [UEmbed-4B](https://huggingface.co/Alibaba-NLP/UEmbed-4B) 和 [WeMM-Embedding-4B](https://huggingface.co/tencent/WeMM-Embedding-4B)。它们主要增加多模态检索能力，公开模型卡未给出能决定本次代码表示选择的直接比较。因此，我选择部署 Qwen3-Embedding-4B；本批没有进行 embedding 模型对照。

这个判断只决定当前批次的实现条件。代码检索分数不直接衡量代码到代码的多样性，也不直接证明 DeltaEvolve 的搜索效果。

## 实际条件

| 项目 | 本批设置 |
| --- | --- |
| 任务与重复 | `co6` 六任务，每任务两路；随机种子 0、1。 |
| 生成预算 | 每路 1000 次尝试，含一次种子评价；解析失败等按已有规则扣预算。 |
| 生成模型 | server3 的 `Qwen3.8-27B-AWQ-INT4`，服务名 `qwen3.8-27b-awq`。GPU0、GPU1 各一份，各承接六路搜索；同任务的两路分配到不同端点。 |
| 生成服务 | 沿用原启动脚本：vLLM 0.31.0、TP=1、上下文 32768、最大并发序列 12、显存比例 0.75、FP8 KV cache、MTP 2。 |
| 请求参数 | 关闭原生思考；temperature=0.7、top_p=0.95、输出上限 8192；其余使用实际保存的 Qwen 后端配置。 |
| 本地 embedding | GPU0 共置 `Qwen/Qwen3-Embedding-4B`，BF16，2560 维；仅监听 `127.0.0.1:8002`。最后 token 池化并做 L2 归一化。 |
| embedding 服务 | vLLM pooling runner；显存比例 0.20、上下文与批 token 上限 8192、最大并发序列 1、eager 执行。并发请求由服务排队。 |
| embedding 输入 | 原始代码，不添加检索 query 指令。沿用 6000 UTF-8 字节分块和字节加权平均，之后由网格归一化；不截断代码。 |
| 数据与评价 | 训练 16 实例；CPU 调度器共用 0–51 核，单路配置 52 workers，实际核心由调度器分配。其余遵循[共同协议](../../experiments/PROTOCOL.md)。 |
| 最终选择与测试 | 训练前至多 5 个个体重评；成功选定最终程序后，自动执行同规模 50 实例独立测试。测试不参与搜索与选择。 |

本地 embedding 的端点、服务名与凭据变量名均写入每路配置。实际密钥不写入批次配置。模型通过 Qwen 的 ModelScope 仓库下载；全部文件按仓库提供的大小和 SHA-256 校验。文件版本与校验值保存在批次附件中。

## 启动检查与原始证据

本地与 server3 的 17 项 DeltaEvolve 检查均通过。真实服务检查覆盖六任务模板、长代码分块、12 个并发 embedding 请求、重复输入的一致性，以及同时调用两路生成服务。六任务短代码均返回有限、非零的 2560 维单位向量。12 路搜索于北京时间 10:43 启动，种子评价均合法。启动复查时，12 个进程均存活，每路已收到 1–4 个真实生成回复；实际配置与预定配置一致，214 个源码与检查文件也与快照校验值一致。检查时 GPU0 剩余显存约 1 GiB，GPU1 约 6.8 GiB；两路生成服务与本地 embedding 同时正常运行。

首批回复中，部分模型将 SEARCH/REPLACE 补丁放在 delta 日志后面。论文模板明确要求将 delta 放在回复末尾；当前解析器按这个规则拒绝这些回复，并照常计入生成预算。这是本次观察到的输出格式失败，不能把这些候选算作执行评价失败。

| 证据 | 位置（仓库内） |
| --- | --- |
| 批次清单、实际服务配置与逐路命令 | `experiments_result/deltaevolve/batch_20261017_qwen_co6.json`。 |
| 逐路启动验证 | `experiments_result/deltaevolve/launch_logs/20261017_qwen_co6_verification.json`。 |
| 源码与检查条件 | `experiments_result/deltaevolve/launch_logs/20261017_qwen_co6_source.tar.gz`，214 个文件，SHA-256 为 `5bd23fdfa1f2fa18762aeed12952ceef3b0a78a2a49f1a3ecf7053e1bbc113bf`。逐文件校验值见同目录 `20261017_qwen_co6_source_checksums.json`。 |
| 模型文件版本 | 同目录 `20261017_qwen_co6_embedding_download_manifest.json`。 |
| 单路数据 | `experiments_result/deltaevolve/<task>/20261017_qwen_co6_<short>_deltaevolve_rep<1或2>/`。 |

server3 执行目录为 `/home/fzy/code/LLM4AD/TraceAAD`。启动时已开启现有结果同步，每 60 秒拉取一次，不删除远端数据。该同步入口在所有搜索终止后退出；最终独立测试是否完成仍以 `heldout.json` 和对应进程为准。

这些检查支持本批的部署、请求和任务链路正常。该初始批次随后按下述 FP8 切换决定中止，没有完整搜索收益和独立测试结果。

## 切换为 FP8 embedding 后重新开始

我随后将 embedding 换成 FP8，生成模型继续使用原来的两路 Qwen3.8-27B AWQ INT4。原服务的进程与启动时间没有改变。曾准备下载 27B BF16 权重，但已按用户补充指示停止，没有切换生成模型。

我用 vLLM 0.31.0 的 `fp8_per_tensor` 在加载时量化已有 Qwen3-Embedding-4B 权重，实际使用 `CutlassFP8ScaledMMLinearKernel`。线性层权重采用 FP8 E4M3，激活按 token 动态量化；未量化层保持 BF16。模型文件没有被覆盖。输出仍为 2560 维，池化、归一化和完整代码分块规则保持一致。

显存目标比例从 0.20 调为 0.14，避免权重节省的空间全部变成更多 KV 缓存。模型加载占用由 7.56 GiB 降到 4.38 GiB，完整服务进程由 9694 MiB 降到 6768 MiB，即从 9.47 GiB 降到 6.61 GiB，节省约 2.86 GiB。GPU0 的空闲显存因此增加到约 4 GiB。

为检查量化是否破坏表示，我先在切换前保存了旧批次中 33 个不同程序的 BF16 向量，再在 FP8 服务上处理完全相同的代码。对应向量的余弦相似度最小为 0.99508、平均为 0.99614；程序间余弦相似度误差平均为 0.00746、最大为 0.02468；前三近邻平均保留率为 95.96%。六任务模板、长代码分块、12 请求并发与两路生成接口也通过了真实服务检查。

这个小样本检查支持 FP8 表示接近 BF16，不证明网格单元或搜索轨迹完全相同。因此，我正常中断旧批次的 12 路搜索并保留原始记录与完整检查点；没有将旧网格直接续接到新精度上。旧路次停止时各消耗 6–16 次预算，没有执行最终独立测试。

新批次为 `20261017_qwen_co6_fp8`。六任务各两路、种子 0 和 1、每路预算 1000，其余搜索与评价条件沿用上表。所有路次从种子重新开始，训练结束后仍自动执行独立测试。启动复查时，12 路均存活、种子均合法，每路已收到 1–3 个真实生成回复，实际配置与预定配置一致。当前实验代码与旧批次源码快照一致；本地使用说明随后更新为 FP8 服务名，未覆盖远端运行中的源码。

| FP8 批次证据 | 位置（仓库内） |
| --- | --- |
| 批次清单与完整 FP8 服务配置 | `experiments_result/deltaevolve/batch_20261017_qwen_co6_fp8.json`。 |
| 精度与显存检查 | `experiments_result/deltaevolve/launch_logs/20261017_qwen_co6_fp8_service_verification.json`。 |
| 逐路启动检查 | 同目录 `20261017_qwen_co6_fp8_verification.json`。 |
| 旧批次停止记录 | 同目录 `20261017_qwen_co6_stop.json`。 |

新批次仍在运行。这些检查验证部署、表示与任务链路，没有比较 FP8 和 BF16 的完整搜索收益。

按用户要求，一次性预检、向量比较脚本与临时结果，以及取消的 27B BF16 下载碎片已清理。上文的向量比较统计是清理前保存的结论，33 个程序的临时参考向量不再保留。正式批次的配置、调用、候选、评价、检查点、源码快照、部署配置及启动记录继续保留。
