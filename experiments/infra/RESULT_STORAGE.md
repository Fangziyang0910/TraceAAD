# 实验结果存储

每路结果在 `experiments_result/<实验>/<任务>/<运行>/`，文件格式见[搜索实验与结果格式](SEARCH_FORMAT.md)。所有 fitness 越小越好。

原始格式的封存档在 `experiments_result/.archive/storage_20261006/<实验>/<任务>/<运行>.tar.gz`，需要查阅时解到单独目录：

```bash
mkdir -p /tmp/traceaad-original-run
tar -xzf <原档.tar.gz> -C /tmp/traceaad-original-run
```

追加写入会先分离共享 inode，不改变已封存的副本。
