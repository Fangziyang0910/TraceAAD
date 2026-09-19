# V10.6 修订配方启动记录

新批次：`20260906_215231_revised`。启动：2026-09-06 21:53（Asia/Shanghai）；核对快照：2026-09-06T21:54:44.339986+08:00。

用户授权停止旧 V10.6、删除被停止实验的数据，并启动新配方。旧调度器与15路实验会话已停止，其进程树已确认退出；15个旧正式运行目录、旧manifest/lock、旧启动快照及调度日志已删除。旧批次为`20260906_formal`。历史研究文档、先前独立冒烟和模型部署信息保留。

新配方使用公共任务说明与模板，VRPTW恢复为旧基线文本；第一次调用Idea→Code，第二次重新生成实现Idea，历史与节点使用后者。协议为`idea_code_then_implementation_idea`。每路从头生成8个有效根，1000次真实评价，五任务各三重复，seed=0/1/2。预算分配及R/P/F=.50/.15/.35沿用既定配置。

15路均已启动并产生真实评价。当前核查确认：两阶段调用已记录，公共任务说明在两次请求中保持一致，专用时限句已移除；节点代码与第一阶段响应一致，节点Idea与第二阶段响应一致，fitness与单次评价收据吻合，运行源码指纹无漂移。此检查验证流程，未证明摘要语义正确或新机制提高搜索表现。

| 任务 | 重复 | 后端 | 已确认评价 | 有效节点 |
| --- | ---: | --- | ---: | ---: |
| tsp_construct | 1 | server3 | 1 | 1 |
| cvrp_aco | 1 | server3b | 1 | 1 |
| op_aco | 1 | server1 | 1 | 0 |
| online_bin_packing | 1 | server3 | 2 | 2 |
| vrptw_construct | 1 | server3b | 2 | 2 |
| tsp_construct | 2 | server1 | 1 | 1 |
| cvrp_aco | 2 | server3 | 1 | 1 |
| op_aco | 2 | server3b | 1 | 1 |
| online_bin_packing | 2 | local | 3 | 3 |
| vrptw_construct | 2 | server3 | 1 | 1 |
| tsp_construct | 3 | server3b | 1 | 1 |
| cvrp_aco | 3 | server1 | 1 | 1 |
| op_aco | 3 | server3 | 1 | 1 |
| online_bin_packing | 3 | server3b | 2 | 2 |
| vrptw_construct | 3 | server1 | 1 | 1 |

快照合计20次评价；以上均为启动观察，实时进度以checkpoint/监控为准。后端分配：{'server3': 5, 'server3b': 5, 'server1': 4, 'local': 1}。

调度会话：`v106_sched`；实验会话：`v106_任务_r重复`。监控会话`v106_monitor`继续服务，已读取新批次15路数据。模型服务保持运行。

- [批次manifest](results/batch_20260906_215231_revised.json)
- [协议核对快照](results/startup_audit_20260906_215231_revised.json)
- [调度日志](results/launcher_20260906_215231_revised.log)
- [运行说明](README.md)
- [机制设计](../../docs/01-主线版本/TraceAAD-V10.6-机制设计.md)
