# TraceAAD V10.9

设计见[机制文档](../../docs/01-主线版本/V10.9/机制设计.md)，问题依据见[V10.8 机制分析](../../docs/01-主线版本/V10.8/思想形成与精炼/分析.md)。

五任务（TSP、CVRP、OP、OBP、VRPTW）× 三重复，seed 0/1/2，每路 1000 次真实评价、8 个有效根、32K 总上下文和 16K 输出上限。评价器、数据与模型服务沿用现有配置；V10.9 的全部策略固定，不按某任务的运行成绩修改。本批从零搜索，不注入历史最好程序。

初始化版本 1091：首根独立生成，后续根参考已有根的完整代码和成绩；整程序裁剪，拒绝根的 AST 复制，保留同分不同代码。共用提示、不注入任务种子、不增加每根强制开发。旧 1090 批次保留归档，见[本轮重启记录](restart_initialization_20260909.md)。

## 冻结与排队

```bash
.venv/bin/python -m experiments.traceaad_v10_9.freeze --batch 20260909_v109_initaware --session-prefix v109
```

冻结程序打印 `runtime` 和 `launch_command`；进入该 runtime，用打印的绝对解释器和命令启动。`--watch` 持续补位，默认每 30 秒最多启动一路。调度只占现有模型服务的空闲逻辑槽，不重启 GPU 服务或抢占实验。15 路先全部进入持久化清单，初次启动及恢复都可使用任一空闲后端，各后端等价。

快照仅复制算法与运行依赖，不复制凭据、历史日志。若原仓库存在私有 `.env`，冻结目录只建立指向它的本地链接，供调度器和 tmux 子实验通过现有读取机制使用；不把其内容写入清单。五任务的数据由冻结的生成器与固定种子产生；运行结果链接回本目录 results。`runtime_manifest.json` 保存各源码哈希，启动器核查源码身份，批次绑定快照摘要。不要在运行中的冻结目录修改源码。Python 环境与模型服务沿用现有部署，未打包为独立镜像。

```bash
.venv/bin/python -m experiments.traceaad_v10_9.launch --batch preview --dry-run
```

上述预览不会启动实验。正式启动要求存在已冻结的运行目录。调度清单 `results/batch_<batch>.json` 区分 queued、running、finished、blocked、stopped；同批有排他锁，每路最多尝试启动三次（包含首次启动），不重做状态不明的评价。未解决的 blocked/stopped 不算完成。其他旧调度器不共享该锁，启动前重新检查空位并逐路启动，以降低竞争窗口；这些槽是服务并发配额，不是 GPU 显存百分比。

## 核验

```bash
.venv/bin/python -m pytest -q tests/method/test_traceaad_v109.py tests/method/test_traceaad_v108.py tests/experiments/test_traceaad_v109_launch.py
```

检查质量集中、近期试用边界、代码复制过滤、参数修改类别、完整证据/容量裁剪、真实评价计数及恢复、15 路计划与任意空闲后端恢复。实现审查、冻结身份与真实启动状态见[启动记录](launch_20260909.md)。
