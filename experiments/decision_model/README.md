# 决策模型的数据与训练入口

**2026-10-09 状态：server2 的 GPU 5、6 已启动两轮初始训练。** 两步真实长输入检查通过，峰值保留显存 12,820 MiB，决策头确实更新。server1 队列与新增数据采集继续暂停，断点保留。当前使用“给定改进请求的结果预测”数据结构与 8,192 token 的 4-bit LoRA 配置，见[实际方案与执行记录](../../docs/04-研究认识与构想/2026-10-09-决策模型训练计划.md#10-月-9-日实际方案与暂停状态)。下文的算子硬标签例子与初始 BF16 参数保留作早期接口说明，不是本轮的数据与配置。

## server2 当前运行

训练目录 `/home/fzy/code/traceaad-decision`；底座 `models/clef-flash` 已从 ModelScope 下载齐全并固定文件哈希。实际数据 `data/bootstrap-fit`，训练 2,328 条、标定 2,304 条，保留测试 2,493 条不用于本轮评价。两张卡各跑一轮独立训练，不合并显存。

| GPU | 种子 | tmux 会话 | 日志 |
| --- | --- | --- | --- |
| 5 | 3408 | `traceaad-decision-server2-gpu5-20261009` | `logs/train-server2-gpu5.log` |
| 6 | 3407 | `traceaad-decision-server2-gpu6-20261009` | `logs/train-server2-gpu6.log` |

两轮的实际参数相同：4-bit LoRA、BF16 计算、rank 16、语言模型学习率 `5e-5`、决策头 `1e-4`、batch 1、累积 16、8,192 token、2 epoch，共 292 个优化步骤。输出分别为 `runs/aad-outcome-v1-warm-server2-seed3408` 和 `runs/aad-outcome-v1-warm-server2-seed3407`，每 50 步保存 checkpoint，结束后标定并保存 adapter。`run_config.json` 保存种子、参数和数据哈希。

```bash
ssh B3-server2 'tail -n 8 /home/fzy/code/traceaad-decision/logs/train-server2-gpu5.log'
ssh B3-server2 'tail -n 8 /home/fzy/code/traceaad-decision/logs/train-server2-gpu6.log'
```

启动时的硬件核验、会话和路径保存在远端 `runs/server2-training-status.json`；它记录启动结果，后续是否仍运行须结合 tmux 会话与最新训练日志。当前使用 PyTorch 的参考因果卷积，数值正确但吞吐较慢，尚未安装 `causal_conv1d` 优化内核。

第 100 步模型的[中间检查](../../docs/02-实验结果/2026-10-09-决策模型100步中间检查.md)已完成：使用标定集全部 43 个前沿改善案例和每任务 64 个随机负例，按抽样比例加权。按任务分别取预期增益最高 10% 请求，模型捕获增益的六任务平均为 21.6%，简单状态表为 46.9%，尚未体现模型增量。推理在 server3 GPU 3 完成后退出，server2 训练继续；本地记录在 `experiments_result/decision_model/campaign_20261009/checkpoint100-calibration-screen/`。这是已观察请求的排序诊断，没有实际更换动作或运行完整搜索。

这里训练一个供 TraceAAD 使用的决策模型。第一阶段只选择 Refine、Explore 或 Crossover，代码仍由现有生成模型修改。研究问题、数据采集对照和推进条件见[训练计划](../../docs/04-研究认识与构想/2026-10-09-决策模型训练计划.md)。

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

优先使用 ModelScope。2026-10-09 已通过仓库 API 核验 [Cloudflare/clef-flash](https://modelscope.cn/models/Cloudflare/clef-flash)、[Qwen/Qwen3.5-9B](https://modelscope.cn/models/Qwen/Qwen3.5-9B) 和 [Qwen/Qwen3.5-4B](https://modelscope.cn/models/Qwen/Qwen3.5-4B) 均存在。四台机器已安装 ModelScope 1.40.1，并成功下载 Clef-Flash 的配置文件；完整 9B／4B 权重尚未下载。

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

原始来源是搜索决策时的状态和同状态对照结果。保存 `events.jsonl`、`programs.jsonl` 与原始调用，通过程序编号还原当前代码和当时已有的历史。字段取值方式与标签依据见[训练计划](../../docs/04-研究认识与构想/2026-10-09-决策模型训练计划.md#数据与标签)。

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

离线主要看选择的质量损失：模型所选算子与同状态最好算子的短程结果相差多少；同时看超过父代、超过当前前沿的比例、失败率和重复率。准确率与标定用于解释模型，最后仍需在完整搜索中比较同候选预算的质量。[具体比较](../../docs/04-研究认识与构想/2026-10-09-决策模型训练计划.md#判断模型是否有用)

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
