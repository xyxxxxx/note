# tf.train

## Checkpoint

## CheckpointManager

## ClusterDef

协议消息类型。

## ClusterSpec

`ClusterSpec` 实例表示进行 TensorFlow 分布式计算的集群的规格。集群由一组 job 构成，而每个 job 又包含若干 task。

为了创建有 2 个 job 和 5 个 task 的一个集群，我们传入从 job 名称到网络地址列表的映射：

```python
cluster_spec = tf.train.ClusterSpec({"worker": ["worker0.example.com:2222",
                                                "worker1.example.com:2222",
                                                "worker2.example.com:2222"],
                                     "ps": ["ps0.example.com:2222",
                                            "ps1.example.com:2222"]})
```

### as_cluster_def()

返回基于此集群的 `tf.train.ClusterDef` 协议缓冲区。

### as_dict()

返回从 job 名称到其包含的 task 的字典。

### job_tasks()

返回指定 job 中的从 task ID 到网络地址的映射。

### num_tasks()

返回指定 job 中的 task 数量。

### task_address()

返回指定 job 中指定索引的 task 的网络地址。

### task_indices()

返回指定 job 中的有效 task 索引列表。
