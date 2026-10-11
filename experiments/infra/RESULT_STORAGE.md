# 实验结果存储

现用结果位于`experiments_result/<实验>/<任务>/<运行>/`，格式见[搜索实验与结果](SEARCH_FORMAT.md)。当前目录保留DeepSeek实验。

旧本地模型结果已封存到[archive/local_models_20261011](../../archive/local_models_20261011/README.md)，原目录结构保留。状态统一为archived，归档前的配置、总结和批次清单保存在original_metadata，成绩、代码、模型回复与事件不改写。路径映射与校验记录分别在archive_manifest.json、integrity.jsonl。

更早的原始格式压缩档位于`archive/local_models_20261011/.archive/storage_20261006/<实验>/<任务>/<运行>.tar.gz`，查阅时解到独立目录：

```bash
mkdir -p /tmp/traceaad-original-run
tar -xzf <原档.tar.gz> -C /tmp/traceaad-original-run
```

归档供只读查阅，看板不再向归档写入派生缓存；.cache单独标记，不作为原始实验证据。需要恢复或复算时复制到新目录，并按原条件使用original_metadata；不要在封存目录直接续跑。原有硬链接保留，追加写入须先分离共享inode。archive目录被Git与代码镜像排除，后续push不会上传或清除封存数据。
