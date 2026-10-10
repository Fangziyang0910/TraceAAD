# 决策模型的数据与训练入口

**2026-10-09 晚：server2 的 GPU 5、6 正在训练“普通算子加自动 Repair”的 9B 请求模型。** 旧目标两轮均已完成，但收益排序没有超过简单状态规则。当前真实数据、标签设计与推进条件见[训练计划](../../docs/04-研究认识与构想/2026-10-09-决策模型训练计划.md)，结果与认识见[实际请求收益](../../docs/03-现象与检验/2026-10-09-算子选择与实际请求收益.md)。

## server2 当前运行

目录 `/home/fzy/code/traceaad-decision`；Python `/home/fzy/venvs/traceaad-decision/bin/python`。完整底座在 `models/clef-flash`，从 ModelScope 下载并保存文件哈希。两张卡各运行一轮，不合并显存。

| GPU | 初始化 | tmux 会话 | 日志 |
| --- | --- | --- | --- |
| 5 | 发布版 Clef-Flash | `traceaad-request-base-gpu5-20261009` | `logs/request-v1023-base.log` |
| 6 | 旧 AAD checkpoint 292 | `traceaad-request-warm-gpu6-20261009` | `logs/request-v1023-warm.log` |

数据为 `data/current-v1023-requests-fit`：训练 1,468、标定 1,321、测试 1,380。训练有 50 个正收益请求。测试只做结构和编码检查，不计算模型测试指标。最大输入 4,852 token；父程序与参考全文保留，预设的历史片段不需进一步缩短。两种初始化的两步硬件检查通过，峰值保留显存 12,510 MiB。

独立审计 4,169 条请求的代码、历史、收益与成本，零错配；4,102 个有效程序的原始实例平均分与档案一致。两步硬件检查后的实际参数核验确认 LoRA 和决策头都更新。报告为本地本轮档案的 `request-v1023-{data-audit,score-vector-audit,encoder-check}.json`。

实际设置：4-bit LoRA、BF16、rank 16／alpha 16、语言模型学习率 `5e-5`、决策头 `1e-4`、batch 1、累积 16、8,192 token、2 epoch、种子 3407，各 184 步。输出为 `runs/request-v1023-base-seed3407` 与 `runs/request-v1023-warm-seed3407`；每 50 步保存 checkpoint，结束后标定并保存 `adapter`。运行参数与数据哈希写入 `run_config.json`。

```bash
ssh B3-server2 'tail -n 4 /home/fzy/code/traceaad-decision/logs/request-v1023-base.log'
ssh B3-server2 'tail -n 4 /home/fzy/code/traceaad-decision/logs/request-v1023-warm.log'
ssh B3-server2 'tmux list-sessions'
```

当前使用正确但较慢的 PyTorch 因果卷积参考实现。原来两个 `aad-outcome-v1-warm-server2-seed*` 训练均已完成 292 步并保存标定 adapter；在固定 427 案例中的增益捕获为 29.3%／22.5%，简单状态表为 46.9%。完整结果保留在本地 `campaign_20261009/checkpoint292-calibration-screen-seed*/`。

## server3 当前采集

根目录 `/home/fzy/code/traceaad-decision-data/campaign_20261009`。采集使用冻结的 `runtime-v1023` 和已有 CPU／模型槽调度器；不改正在运行的其他实验。原始结果以每条 `chains/<run>/<state>/<action>/repN/result.json` 及其逐步事件为准。

| 批次 | 会话 | 规模 | 日志 |
| --- | --- | --- | --- |
| 固定试采 | `traceaad-decision-pilot-v1023-20261009` | 72 状态，最多 3,456 新候选，4 worker | `logs/pilot-v1023-blocked.log` |
| 训练增样 | `traceaad-decision-enriched-20261009` | 100 训练状态，最多 4,800 新候选，4 worker | `logs/enriched-v1023.log` |
| 独立确认 | `traceaad-decision-confirm-tsp-20261009` | 一个 TSP 标定状态，各算子追加四重复，最多 48 候选 | `logs/confirm-tsp.log` |

每状态比较三个算子、每算子四条独立重复、每条四次候选。四次开发内先执行初始算子，再使用 Refine／Repair；部署用的标签只取初始请求及自动 Repair，不能把这个开发块当作实际 V10.23 Develop。确认的 repeat ID 为 4–7，独立保存，不混入固定试采。

固定试采已核验 72 状态、216 个实际生成提示，最大 15,920 token。训练增样已核验 100 状态、300 个实际提示，最大 7,697 token；每个父程序最多两个起点。预检没有生成候选。历史正收益只用于训练增样选起点，重新运行的结果才作为标签。标定／测试的固定状态不做成功筛选；进度日志不显示测试结果。

## 自动导出、训练与实际选择评价

本地 tmux `traceaad-decision-research-20261009` 已启动当前后台流程：

```bash
# 当前会话已经在运行；这个入口用于复现或检查中断后的流程。
.venv/bin/python -u -m experiments.decision_model.campaign experiments_result/decision_model/campaign_20261009
```

流程先等两轮当前训练和固定试采完成，在 server3 的 `analysis-runtime-v1023` 导出请求标签，检查全单元完成和提示哈希，然后在 server2 完整编码。它比较两轮模型、未微调底座与三个简单规则的标定侧实际选择收益。训练增样完成后，流程用独立重复标签替换被选中的历史单次样本，再从相同初始化训练，并重做同状态标定比较。它不自动开始完整搜索，也不计算决策测试指标。

首轮选择评价后，流程还在授权 GPU 上启动真实模型的 HTTP 检查：核对生成条件，确认无令牌返回 401，并给同一标定状态的三个动作评分。检查不生成候选，完成后退出自己的服务；报告为 `request-http-check.json`。

本地 `pipeline-status.json`、`pipeline.log` 记录推进状态；`campaign-status.json` 保存整体账本及启动事实。每批导出时同时拉回完整原始重放档案，保存程序、调用及逐步评价；冻结生成代码在 `runtime-v1023` 副本，审计与研究脚本在 `research_scripts/`。已有事实记录不等于进程当前仍活着，要同时看会话和日志。异常时流程写 `needs_attention` 并停止，保留档案，不覆盖已有结果。

实际收益报告文件为 `request-observed-{warm,base,released}-calibration.json` 和随后 `request-paired-{warm,base}-calibration.json`。主字段 `equal_task_macro_gain_per_candidate`：每任务先累计所选动作增益、除以实际候选消耗，再对任务等权。简单规则只拟合训练来源；`matched_sample_best` 使用观察结果，仅作为描述，不是可部署对照。loss／accuracy 不作为验收指标。

独立运行数据入口：

```bash
# 在项目环境：从已有冻结前缀重新归并自动 Repair，无新生成。
.venv/bin/python -m experiments.decision_model.request_data SOURCE REQUEST_DATA

# 在项目环境：仅在所有固定单元完成后导出重复标签。
.venv/bin/python -m experiments.decision_model.export_replays DATASET COLLECTION PAIRED_DATA --request-only

# 训练增样只有 train，用 --splits train 导出，再与完整数据合并。
.venv/bin/python -m experiments.decision_model.augment_data REQUEST_DATA PAIRED_DATA ENRICHED_DATA AUGMENTED_DATA
```

在服务器训练环境中，完整编码与训练的模板如下。`OUTPUT` 必须是新目录；恢复已有训练时使用原数据与 checkpoint：

```bash
cd /home/fzy/code/traceaad-decision
HF_HUB_OFFLINE=1 /home/fzy/venvs/traceaad-decision/bin/python fit_data.py RAW FIT --tokenizer models/clef-flash --max-seq-length 8192
CUDA_VISIBLE_DEVICES=5 HF_HUB_OFFLINE=1 /home/fzy/venvs/traceaad-decision/bin/python train.py FIT OUTPUT --model models/clef-flash --load-in-4bit --max-seq-length 8192 --seed 3407
# 同一个 OUTPUT 和数据；具体 checkpoint 按实际档案选择。
CUDA_VISIBLE_DEVICES=5 HF_HUB_OFFLINE=1 /home/fzy/venvs/traceaad-decision/bin/python train.py FIT OUTPUT --model models/clef-flash --load-in-4bit --max-seq-length 8192 --seed 3407 --resume OUTPUT/checkpoints/checkpoint-50
```

独立确认已完成 48 次候选，追加四次 Explore 有两次即时改善；结果保存在 `confirmation-tsp-before464/request-outcomes.json`，不混入固定标定估计。

HTTP 服务和 V10.23 算子接入已有原型，使用收益／成本比评分。实际 HTTP 检查已排入后台流程，尚未完成；完整搜索接入与搜索收益尚未验证。下面保留环境安装与早期格式说明；早期算子硬标签与 BF16 示例不是当前训练设置。

## 环境与机器

每台机器使用独立的 Python 3.12 环境，四台均已安装并通过基础 GPU 检查。依赖锁定在 [requirements.txt](requirements.txt)：Unsloth 固定到 [v0.1.905-beta 的源码提交](https://github.com/unslothai/unsloth/commit/d94ca0ee6b54891378c6de65af9eba48f1afae59)（包内版本 2026.10.3），unsloth-zoo 为 2026.10.3、PyTorch 为 2.11.0 的 CUDA 12.8 版、Transformers 为 5.17.0。实测 PyPI 的 Unsloth 2026.10.1 缺少新的 `predict` 接口，因此不用它重建环境。训练命令直接使用这个环境的 Python。

| 机器 | SSH | Python 路径 | 训练入口目录 | 驱动报告的 GPU |
| --- | --- | --- | --- | --- |
| 本地 | 无 | `/home/fang/venvs/traceaad-decision/bin/python` | `/home/fang/code/LLM4AD/TraceAAD/experiments/decision_model` | 1 × RTX 4090 D，约 24 GB |
| server1 | `B3-server1` | `/home/fzy/venvs/traceaad-decision/bin/python` | `/home/fzy/code/traceaad-decision` | 2 × RTX 4080 SUPER，各约 32 GB |
| server2 | `B3-server2` | `/home/fzy/venvs/traceaad-decision/bin/python` | `/home/fzy/code/traceaad-decision` | 7 × RTX 3090，各约 24 GB |
| server3 | `B3-server3` | `/home/fzy/venvs/traceaad-decision/bin/python` | `/home/fzy/code/traceaad-decision` | 4 × RTX 4090，各约 48 GB |

硬件在 2026-10-09 实测，显存按驱动报告填写。环境检查执行 `verify.py`，四台均已验证决策模型 API、BF16 反向传播和 8-bit 优化器的一步更新；服务器结果保存在入口目录的 `environment.json`，本地副本在 `experiments_result/decision_model/environment/`。

server1 另用 `trl-internal-testing/tiny-Qwen3_5ForConditionalGeneration` 与合成 JSONL 检查了完整入口：训练一步、从 checkpoint 恢复到第二步、标定、保存 adapter、重新加载与预测。已确认恢复后决策头参数发生更新，预测概率有限且和为 1。记录在本地 `experiments_result/decision_model/smoke-qwen35/`；远端完整检查产物位于 `runs/environment-smoke-qwen35/`。这只验证执行，不评价 AAD 决策质量，也不代表真实 9B 的显存与吞吐量。

可用显存会随其他任务变化。安装时本地模型服务占用了约 23 GB，server3 各卡只剩约 9–12 GB；环境就绪不等于当时有空闲显存训练。server1 的 GPU0 当时空闲约 31 GB，最适合先检查 9B。先运行 `nvidia-smi`，选择空闲卡，再设置 `CUDA_VISIBLE_DEVICES`。一个训练进程只使用一张卡；两张卡分别运行两个训练，不会自动合并显存。

```bash
# 在任一服务器
cd /home/fzy/code/traceaad-decision
CUDA_VISIBLE_DEVICES=0 /home/fzy/venvs/traceaad-decision/bin/python verify.py --output environment.json

# 在本地
cd /home/fang/code/LLM4AD/TraceAAD
CUDA_VISIBLE_DEVICES=0 /home/fang/venvs/traceaad-decision/bin/python experiments/decision_model/verify.py
```

## 选择底座

主训练先考虑 `Cloudflare/clef-flash`。它已经是基于 Qwen3.5-9B 的决策模型，适合在已有决策能力上学习 AAD。原始 `Qwen/Qwen3.5-9B` 则会初始化新的决策头，作为同一模型系列的对照。24 GB 卡先用 `Qwen/Qwen3.5-4B` 检查数据和训练流程；Unsloth 打包的同系列模型也可以加载，但需分别记录权重来源。模型不同的结果分别报告，不能把 4B 的流程检查当作 9B 的效果。[Clef-Flash 模型卡](https://huggingface.co/Cloudflare/clef-flash)

初始配置为 BF16 LoRA：rank 16、alpha 16、语言模型学习率 `2e-4`、决策头学习率 `1e-4`、每卡 batch 1、梯度累积 16、2 个 epoch、上下文 2048 token。语言模型的大部分参数冻结，LoRA 和决策头更新。先跑 10 个优化器步，再按实际显存与输入长度调到 4096 或 8192 token。

Unsloth 给出的 Qwen3.5 BF16 LoRA 参考显存为 4B 约 10 GB、9B 约 22 GB；决策头、输入长度和软件版本会改变实际占用。9B 优先放在 server1 或空闲的 server3 卡上。Qwen3.5 的专用指南提示量化差异较大，因此这里先使用 BF16。[Qwen3.5 微调指南](https://unsloth.ai/docs/models/qwen3.5/fine-tune)

## 下载底座

优先使用 ModelScope。2026-10-09 已通过仓库 API 核验 [Cloudflare/clef-flash](https://modelscope.cn/models/Cloudflare/clef-flash)、[Qwen/Qwen3.5-9B](https://modelscope.cn/models/Qwen/Qwen3.5-9B) 和 [Qwen/Qwen3.5-4B](https://modelscope.cn/models/Qwen/Qwen3.5-4B) 均存在。四台机器已安装 ModelScope 1.40.1。Clef-Flash 完整权重已在 server2、server3 就绪；server1 保留部分下载，本地只有配置。其他两个底座的完整权重未准备。

在训练入口目录执行。下面使用已实测的 CLI 语法，不需要 Hugging Face 连接：

```bash
cd /home/fzy/code/traceaad-decision
/home/fzy/venvs/traceaad-decision/bin/modelscope download \
  Cloudflare/clef-flash --local-dir models/clef-flash
# 原始 Qwen 底座的对照，按需要下载
/home/fzy/venvs/traceaad-decision/bin/modelscope download \
  Qwen/Qwen3.5-9B --local-dir models/Qwen3.5-9B
/home/fzy/venvs/traceaad-decision/bin/modelscope download \
  Qwen/Qwen3.5-4B --local-dir models/Qwen3.5-4B
```

本地换成 `/home/fang/venvs/traceaad-decision/bin/modelscope`，从仓库根目录执行。`--model` 接受本地完整模型目录。只需要下载一次；用 rsync 把同一个目录复制到其他机器的训练工作目录，路径保持为 `models/<模型名>`。恢复时 adapter 记录的底座路径也必须可用。

正式数据实验选定模型 revision 后，在下载命令中增加 `--revision <提交>`，记录来源、revision 与文件哈希，之后保留同一份权重。训练命令可加 `HF_HUB_OFFLINE=1`；先下载全仓库，不要把本次只有配置文件的 `download-check/` 当作完整底座。备用的 Hugging Face 下载入口为 `hf download Cloudflare/clef-flash --local-dir models/clef-flash`。[ModelScope 下载说明](https://modelscope.cn/docs/models/download)

## 准备训练文件

原始来源是搜索决策时的状态和同状态对照结果。保存 `events.jsonl`、`programs.jsonl` 与原始调用，通过程序编号还原当前代码和当时已有的历史。字段取值方式与标签依据见[训练计划](../../docs/04-研究认识与构想/2026-10-09-决策模型训练计划.md#数据从哪里来)。

先按完整搜索运行分组，写成三个 JSONL 文件。每行一个决策状态，不是一次后续生成。

```text
data/operator-v1/
  train.jsonl
  calibration.jsonl
  test.jsonl
```

下面只说明字段格式，`gold` 是示例，不能直接当作真实训练样本。JSONL 实际保存时，将一个对象写在一行。

```json
{
  "run_id": "tsp_construct/source-batch/rep0",
  "state_id": "parent-206-before-generation",
  "state": {
    "task": "问题与函数接口",
    "current_code": "当前程序全文",
    "current_fitness": 5.95,
    "past_attempts": "在决策当时已经完成的最近改动及训练评价结果",
    "reference_code": "预先固定的 Crossover 参考程序",
    "remaining_candidates": 400
  },
  "questions": {
    "operator": {
      "type": "choice",
      "instructions": "Which first step is most useful for the fixed development budget?",
      "criteria": {
        "Refine": "Keep the core computation and improve it",
        "Explore": "Change the core computation used to produce the output",
        "Crossover": "Borrow useful computation from the supplied reference"
      }
    }
  },
  "gold": {"operator": "Refine"}
}
```

同一运行、同一状态的全部重复和改写版本必须进入同一划分。`run_id` 使用“任务／来源批次／重复编号”，保持完整身份。`state` 只能包括当时可见的信息；此次对照的结果、后来最好的代码和组合优化任务的测试集成绩只用于分析，不能放进输入。

`gold` 来自固定预算下的重复比较，不能直接复制原搜索选择的动作。Unsloth 也接受 `{"operator": {"probabilities": {"Refine": ..., "Explore": ..., "Crossover": ...}}}` 形式的软标签。这些值应由重复对照构造，表示该对照设计下的经验选择分布。[官方训练格式与 API](https://unsloth.ai/docs/basics/train-your-own-decision-model-with-unsloth)

先检查文件。此命令不加载模型，也不占 GPU：

```bash
cd /home/fzy/code/traceaad-decision
/home/fzy/venvs/traceaad-decision/bin/python train.py data/operator-v1 runs/check --check-data
```

脚本拒绝空文件、缺失的运行身份、不完整标签，以及跨划分重复的搜索运行。模型分词后还会拒绝跳过或截断的样本。超长状态先减少无关历史，保留当前代码和本次选择需要的参考材料；仍然超长时增大 `--max-seq-length`。

## 开始训练

先在 server1 上检查 9B 的实际占用：

```bash
ssh B3-server1
cd /home/fzy/code/traceaad-decision
nvidia-smi
CUDA_VISIBLE_DEVICES=0 /home/fzy/venvs/traceaad-decision/bin/python train.py \
  data/operator-v1 runs/operator-v1-9b-smoke \
  --model models/clef-flash --max-steps 10 --max-seq-length 2048
```

确认损失有限、数据没有截断、显存有余量后，使用新的输出目录跑完整训练：

```bash
CUDA_VISIBLE_DEVICES=0 /home/fzy/venvs/traceaad-decision/bin/python train.py \
  data/operator-v1 runs/operator-v1-9b \
  --model models/clef-flash --max-seq-length 4096
```

本地命令只换解释器和入口路径。24 GB 卡的第一次流程检查可使用 4B：

```bash
cd /home/fang/code/LLM4AD/TraceAAD
CUDA_VISIBLE_DEVICES=0 /home/fang/venvs/traceaad-decision/bin/python \
  experiments/decision_model/train.py \
  experiments_result/decision_model/data/operator-v1 \
  experiments_result/decision_model/runs/operator-v1-4b-smoke \
  --model models/Qwen3.5-4B --max-steps 10 --max-seq-length 2048
```

长训练放进 tmux：先 `tmux new -s decision-train`，执行训练命令，用 `Ctrl-b d` 离开，用 `tmux attach -t decision-train` 返回。同一张 GPU 上的多个进程并不会自动排队；启动前检查占用。

本次 server1 直连 Hugging Face 超时，按上节先从 ModelScope 下载完整目录，随后离线训练：

```bash
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  /home/fzy/venvs/traceaad-decision/bin/python -u train.py \
  data/operator-v1 runs/operator-v1-offline \
  --model models/clef-flash --max-seq-length 2048
```

初次训练会编译 CUDA 内核，第一步可能较慢。微型检查中 `causal_conv1d` 使用了 PyTorch 回退实现；训练已执行成功，真实 9B 的吞吐量和显存仍需用 10 步检查测量。

## 检查结果与继续训练

输出包括 `checkpoints/checkpoint-*`、`adapter/` 和 `metrics.json`。checkpoint 每 50 步保存，保留最近两个；`adapter/` 保存 LoRA、决策头和标定参数。普通训练只报告 calibration 集，先用它比较模型和设置。

中断后用相同数据、底座、上下文和训练配置恢复。把下例编号换为实际存在的 checkpoint：

```bash
CUDA_VISIBLE_DEVICES=0 /home/fzy/venvs/traceaad-decision/bin/python train.py \
  data/operator-v1 runs/operator-v1-9b \
  --model models/clef-flash --max-seq-length 4096 \
  --resume runs/operator-v1-9b/checkpoints/checkpoint-50
```

模型和设置冻结后，最终那次训练可增加 `--evaluate-test`，报告此前保留的决策 test 集。不要每次试参数都加这个选项。决策 test 集由保留的搜索运行组成，区别于组合优化任务的 held-out 实例。

离线主要看选择的质量损失：模型所选算子与同状态最好算子的短程结果相差多少；同时看超过父代、超过当前前沿的比例、失败率和重复率。准确率与标定用于解释模型，最后仍需在完整搜索中比较同候选预算的质量。[具体比较](../../docs/04-研究认识与构想/2026-10-09-决策模型训练计划.md#怎样判定模型确实有用)

## 调用训练后的模型

训练产物可以直接加载；其中的 `state` 和 `questions` 与训练时格式相同：

```python
from unsloth import FastDecisionModel
import json

model, tokenizer = FastDecisionModel.from_pretrained("runs/operator-v1-9b/adapter")
FastDecisionModel.for_inference(model)
record = json.load(open("decision-request.json", encoding="utf-8"))
answers = FastDecisionModel.predict(model, tokenizer, record["state"], record["questions"])
operator = answers["operator"]["answer"]
print(operator, answers["operator"]["probabilities"])
```

之后由 TraceAAD 按选择的算子构造提示，再调用原有生成模型。第一轮只替换算子选择，起点选择、上下文规则、评价器和候选预算保持实验约定。

## 重建环境与同步

重建同一环境时使用锁定依赖和 CUDA 12.8 索引。server1 与 server2 的训练目录内另有新版 `uv`，下面从该目录执行；server3 把 `./uv` 换成 `/home/fzy/.local/bin/uv`。本地使用 `uv`，并把环境路径换为 `/home/fang/venvs/traceaad-decision`。已有环境直接使用，只有不存在时才执行 `venv` 命令。

```bash
./uv venv --python 3.12 /home/fzy/venvs/traceaad-decision
./uv pip install --python /home/fzy/venvs/traceaad-decision/bin/python \
  -r requirements.txt --torch-backend cu128
./uv pip check --python /home/fzy/venvs/traceaad-decision/bin/python
```

从本地更新入口时只复制训练文件，不覆盖远端数据和模型：

```bash
scp experiments/decision_model/train.py experiments/decision_model/verify.py \
  experiments/decision_model/requirements.in experiments/decision_model/requirements.txt \
  experiments/decision_model/README.md \
  B3-server1:/home/fzy/code/traceaad-decision/
```

替换 SSH 别名即可更新另两台服务器。将数据目录和需要恢复的完整运行目录分别用 rsync 复制；训练不依赖搜索仓库的 Python 环境。三台服务器的评价数据采集仍使用现有实验入口及调度器，不能用训练环境临时改变评价协议。

```bash
# 示例：把本地已准备的数据送到 server1
rsync -a experiments_result/decision_model/data/operator-v1/ \
  B3-server1:/home/fzy/code/traceaad-decision/data/operator-v1/
# 示例：把包含 checkpoint 和 adapter 的完整运行复制到 server2
rsync -a B3-server1:/home/fzy/code/traceaad-decision/runs/operator-v1-9b/ \
  /tmp/operator-v1-9b/
rsync -a /tmp/operator-v1-9b/ \
  B3-server2:/home/fzy/code/traceaad-decision/runs/operator-v1-9b/
```

换机器恢复时，数据、底座、分词器、LoRA 配置和上下文长度必须相同。底座权重也要在目标机器下载或复制缓存。LoRA 文件可用于推理；继续训练需完整 checkpoint，不能只复制 `adapter/`。
