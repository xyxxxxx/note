# tf.distribute

分布式训练。

## CentralStorageStrategy

中央存储策略。

## cluster_resolver.ClusterResolver

所有集群解析器实现的基类。集群解析器是 TensorFlow 与各种集群管理系统（K8s、GCE、AWS 等）进行交流的手段，也为 TensorFlow 建立分布式训练提供必要的信息。

通过集群解析器和集群管理系统的交流，我们能够自动发现并解析各 TensorFlow 工作器的 IP 地址，进而能够自动从底层机器的故障中恢复，或者对 TensorFlow 工作器进行伸缩。

### cluster_spec()

获取集群的当前状态并返回一个 `tf.train.ClusterSpec` 实例。

### num_accelerators()

返回每个工作器可用的加速器核心（GPU 和 TPU）数量。

### task_id

返回 `ClusterResolver` 实例指明的 task ID。

一般在 TensorFlow 分布式环境中，每个 task 有一个相应的 task ID，即在该 task 类型中的索引。这在用户需要根据 task 索引运行特定代码时十分有用，例如：

```python
cluster_spec = tf.train.ClusterSpec({
    "ps": ["localhost:2222", "localhost:2223"],
    "worker": ["localhost:2224", "localhost:2225", "localhost:2226"]
})

# SimpleClusterResolver is used here for illustration; other cluster
# resolvers may be used for other source of task type/id.
cluster_resolver = SimpleClusterResolver(cluster_spec, task_type="worker", task_id=0)

if cluster_resolver.task_type == 'worker' and cluster_resolver.task_id == 0:
    # Perform something that's only applicable on 'worker' type, id 0. This
    # block will run on this particular instance since we've specified this
    # task to be a 'worker', id 0 in above cluster resolver.
else:
    # Perform something that's only applicable on other ids. This block will
    # not run on this particular instance.
```

若 task ID 在当前分布式环境中不适用，则返回 `None`。

### task_type

返回 `ClusterResolver` 实例指明的 task 类型。

一般在 TensorFlow 分布式环境中，每个 task 有一个相应的 task 类型。这在用户需要根据 task 类型运行特定代码时十分有用，例如：

```python
cluster_spec = tf.train.ClusterSpec({
    "ps": ["localhost:2222", "localhost:2223"],
    "worker": ["localhost:2224", "localhost:2225", "localhost:2226"]
})

# SimpleClusterResolver is used here for illustration; other cluster
# resolvers may be used for other source of task type/id.
cluster_resolver = SimpleClusterResolver(cluster_spec, task_type="worker", task_id=1)

if cluster_resolver.task_type == 'worker':
    # Perform something that's only applicable on workers. This block
    # will run on this particular instance since we've specified this task to
    # be a worker in above cluster resolver.
elif cluster_resolver.task_type == 'ps':
    # Perform something that's only applicable on parameter servers. This
    # block will not run on this particular instance.
```

TensorFlow 中有效的 task 类型包括：

* `'worker'`：常规的用于训练/测试的工作器
* `'chief'`：被分配了更多任务的工作器
* `'ps'`：参数服务器
* `'evaluator'`：测试检查点保存的模型

若 task 类型在当前分布式环境中不适用，则返回 `None`。

## cluster_resolver.SimpleClusterResolver

集群解析器的简单实现。

```python
tf.distribute.cluster_resolver.SimpleClusterResolver(cluster_spec, master='', task_type=None, task_id=None, environment='', num_accelerators=None, rpc_layer=None)
```

## cluster_resolver.TFConfigClusterResolver

读取 `TF_CONFIG` 环境变量的集群解析器。

```python
tf.distribute.cluster_resolver.TFConfigClusterResolver(task_type=None, task_id=None, rpc_layer=None, environment=None)
```

## coordinator.ClusterCoordinator

用于创建容错资源并分派需要执行的函数给远程 TensorFlow 服务器。

目前此类仅支持与分布式策略 `ParameterServerStrategy` 一起使用。

**处理 task 故障**

此类有内置的对于工作器故障的容错机制，即当部分工作器由于任何原因变得对于协调器不可用时，训练过程由剩余的工作器继续完成。……

### create_per_worker_dataset()

通过在工作器的设备上调用 `dataset_fn()` 创建工作器的数据集。

```python
strategy = tf.distribute.experimental.ParameterServerStrategy(
    cluster_resolver=...)
coordinator = tf.distribute.experimental.coordinator.ClusterCoordinator(
    strategy=strategy)

@tf.function
def worker_fn(iterator):
  return next(iterator)

def per_worker_dataset_fn():
  return strategy.distribute_datasets_from_function(
      lambda x: tf.data.Dataset.from_tensor_slices([3] * 3))

per_worker_dataset = coordinator.create_per_worker_dataset(
    per_worker_dataset_fn)
per_worker_iter = iter(per_worker_dataset)
remote_value = coordinator.schedule(worker_fn, args=(per_worker_iter,))
assert remote_value.fetch() == 3
```

### done()

返回是否所有分派的函数都已经执行完毕。

如果先前分派的函数引发了错误，此方法将引发这些错误中的任意一个。

当此方法返回或引发错误时，可以保证没有任何函数仍在执行。

### fetch()

阻塞直到获取 `RemoteValue` 实例的结果。

```python
strategy = ...
coordinator = tf.distribute.experimental.coordinator.ClusterCoordinator(
    strategy)

def dataset_fn():
  return tf.data.Dataset.from_tensor_slices([1, 1, 1])

with strategy.scope():
  v = tf.Variable(initial_value=0)

@tf.function
def worker_fn(iterator):
  def replica_fn(x):
    v.assign_add(x)
    return v.read_value()
  return strategy.run(replica_fn, args=(next(iterator),))

distributed_dataset = coordinator.create_per_worker_dataset(dataset_fn)
distributed_iterator = iter(distributed_dataset)
result = coordinator.schedule(worker_fn, args=(distributed_iterator,))
assert coordinator.fetch(result) == 1
```

### join()

阻塞直到所有分派的函数执行完毕。

如果先前分派的函数引发了错误，此方法将引发这些错误中的任意一个，并清除已经收集的错误。

当此方法返回或引发错误时，可以保证没有任何函数仍在执行。

### schedule()

分派函数到某个工作器以异步执行。

```python
schedule(fn, args=None, kwargs=None)
# fn       要异步执行的函数
# args     `fn`的位置参数
# kwargs   `fn`的关键字参数
```

此方法是非阻塞的，其将函数 `fn` 排进执行队列并立即返回一个 `RemoteValue` 实例。可以对该实例调用 `fetch()` 方法以等待函数执行结束并从远程工作器获取返回值，或者调用 `join()` 方法以等待所有分派的函数执行结束。

此方法保证 `fn` 将在工作器上执行至少一次；如果在执行的过程中相应的工作器出现故障，则会分派到另一工作器上重新执行。

如果先前分派的函数引发了错误，此方法将引发这些错误中的任意一个，并清除已经收集的错误。这时，部分先前分派的函数可能并没有开始执行（或执行完毕），用户可以通过对返回的 `RemoteValue` 实例调用 `fetch()` 方法来检查函数的当前执行状态。

当此方法引发错误时，可以保证没有任何函数仍在执行。

目前尚不支持为函数指定工作器，或设定函数执行的优先级。

## CrossDeviceOps

跨设备归约和广播算法的基类。`ReductionToOneDevice`, `NcclAllReduce` 和 `HierarchicalCopyAllReduce` 是 `CrossDeviceOps` 的子类，实现了具体的归约算法。

此类的主要目标是被传入到 `MirroredStrategy` 以在不同的跨设备通信实现中进行选择。

## DistributedDataset

表示分布式数据集。

当使用 `tf.distribute` API 进行分布式训练时，通常也需要分布输入数据，这时我们选择 `DistributedDataset` 实例，而不是非分布式情况下的 `tf.data.Dataset` 实例。

有两个 API 用于创建 `DistributedDataset` 实例：`Strategy.experimental_distribute_dataset(dataset)` 和 `Strategy.distribute_datasets_from_function(dataset_fn)`。如果你现有一个 `tf.data.Dataset` 实例，并且适用常规的分批和自动 sharding（即 `tf.data.experimental.AutoShardPolicy` 选项）时，使用前一个 API；如果你不是在使用一个 `tf.data.Dataset` 实例，或者你想要自定义分批和 sharding，那么你可以将这些逻辑包装到 `dataset_fn` 函数中并使用后一个 API。

`DistributedDataset` 实例的主要用法是迭代以产生分布式输入数据，是一个 `DistributedValues` 实例。

```python
>>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
>>> dataset = tf.data.Dataset.from_tensor_slices([5., 6., 7., 8.]).batch(2)
>>> dist_dataset = strategy.experimental_distribute_dataset(dataset)
>>> for x in dist_dataset:
    print(x)
PerReplica:{
  0: tf.Tensor([5.], shape=(1,), dtype=float32),
  1: tf.Tensor([6.], shape=(1,), dtype=float32)
}
PerReplica:{
  0: tf.Tensor([7.], shape=(1,), dtype=float32),
  1: tf.Tensor([8.], shape=(1,), dtype=float32)
}
>>> dataset_iterator = iter(dist_dataset)   # 创建迭代器
>>> next(dataset_iterator)
PerReplica:{
  0: <tf.Tensor: shape=(1,), dtype=float32, numpy=array([5.], dtype=float32)>,
  1: <tf.Tensor: shape=(1,), dtype=float32, numpy=array([6.], dtype=float32)>
}
```

   

## DistributedIterator

## DistributedValues

表示分布式值的基类。

`DistributedValues` 的实例在迭代 `DistributedDataset` 实例、调用 `Strategy.run()` 或在分布式策略内创建变量时被创建；不应直接实例化此基类。`DistributedValues` 实例对每一个模型副本包含一个值，这些值可以是自动同步、手动同步或从不同步，取决于子类的具体实现。

```python
# 由`DistributedDataset`实例创建
>>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
>>> dataset = tf.data.Dataset.from_tensor_slices([5., 6., 7., 8.]).batch(2)
>>> dataset_iterator = iter(strategy.experimental_distribute_dataset(dataset))
>>> next(dataset_iterator)
PerReplica:{
  0: <tf.Tensor: shape=(1,), dtype=float32, numpy=array([5.], dtype=float32)>,
  1: <tf.Tensor: shape=(1,), dtype=float32, numpy=array([6.], dtype=float32)>
}
```

```python
# 由`Strategy.run()`返回
>>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
>>> @tf.function
def f():
    ctx = tf.distribute.get_replica_context()
    return ctx.replica_id_in_sync_group
>>> strategy.run(f)
PerReplica:{
  0: <tf.Tensor: shape=(), dtype=int32, numpy=0>,
  1: <tf.Tensor: shape=(), dtype=int32, numpy=1>
}
```

```python
# 作为`Strategy.run()`的输入
>>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
>>> dataset = tf.data.Dataset.from_tensor_slices([5., 6., 7., 8.]).batch(2)
>>> dataset_iterator = iter(strategy.experimental_distribute_dataset(dataset))
>>> distributed_values = next(dataset_iterator)
>>> @tf.function
def f(x):
    return x * 2.0
>>> strategy.run(f, args=(distributed_values,))
PerReplica:{
  0: <tf.Tensor: shape=(1,), dtype=float32, numpy=array([10.], dtype=float32)>,
  1: <tf.Tensor: shape=(1,), dtype=float32, numpy=array([12.], dtype=float32)>
}
```

```python
# 归约分布式值
>>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
>>> dataset = tf.data.Dataset.from_tensor_slices([5., 6., 7., 8.]).batch(2)
>>> dataset_iterator = iter(strategy.experimental_distribute_dataset(dataset))
>>> distributed_values = next(dataset_iterator)
>>> strategy.reduce(tf.distribute.ReduceOp.SUM,
                    distributed_values,
                    axis=0)
<tf.Tensor: shape=(), dtype=float32, numpy=11.0>
```

## get_replica_context()

返回当前的 `ReplicaContext` 实例。

```python
>>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
>>> tf.distribute.get_replica_context()           # 在默认分布式策略下,返回默认模型副本上下文实例
<tensorflow.python.distribute.distribute_lib._DefaultReplicaContext object at 0x7f1b8057e190>
>>> with strategy.scope():                        # 在`MirroredStrategy`下,返回`None`
    print(tf.distribute.get_replica_context())
None
>>> def f():
    return tf.distribute.get_replica_context()
>>> strategy.run(f)
PerReplica:{                                      # `strategy.run()`返回镜像模型副本上下文实例.是为此函数的通常用法
  0: <tensorflow.python.distribute.mirrored_run._MirroredReplicaContext object at 0x7f1b805c0610>,
  1: <tensorflow.python.distribute.mirrored_run._MirroredReplicaContext object at 0x7f1b805c0390>
}
```

## get_strategy()

返回当前的 `Strategy` 实例。

```python
>>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
>>> print(tf.distribute.get_strategy())           # 默认分布式策略实例
<tensorflow.python.distribute.distribute_lib._DefaultDistributionStrategy object at 0x7fb547e94bd0>
>>> with strategy.scope():
  print(tf.distribute.get_strategy())             # `MirroredStrategy`实例
<tensorflow.python.distribute.mirrored_strategy.MirroredStrategy object at 0x7fb54795bf50>
```

## has_strategy()

返回当前是否为非默认的 `Strategy` 实例。

```python
>>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
>>> print(tf.distribute.has_strategy())
False
>>> with strategy.scope():
  print(tf.distribute.has_strategy())
True
```

## HierarchicalCopyAllReduce

hierarchical copy all-reduce 算法的实现。

## in_cross_replica_context()

返回当前是否为跨模型副本的上下文。

```python
>>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
>>> print(tf.distribute.in_cross_replica_context())
False
>>> with strategy.scope():
  print(tf.distribute.in_cross_replica_context())
True
>>> def f():
  return tf.distribute.in_cross_replica_context()
>>> strategy.run(f)
False
```

## InputContext

此类是一个上下文类，其实例包含了关于模型副本和输入流水线的信息，用以传入到用户的输入函数。

### get_per_replica_batch_size()

```python
get_per_replica_batch_size(global_batch_size)
```

返回每个模型副本的批次规模。

### input_pipeline_id

输入流水线的 ID。

### num_input_pipelines

输入流水线的数量。

### num_replicas_in_sync

同步的模型副本的数量。

## MirroredStrategy

单机多卡同步训练。此策略下模型的参数是 `MirroredVariable` 类型的变量，在所有的模型副本中通过 all-reduce 模式保持同步。

```python
tf.distribute.MirroredStrategy(devices=None, cross_device_ops=None)
# devices             设备列表.若为`None`或空列表,则使用所有可用的GPU;若没有发现GPU,则使用可用的CPU
#                     注意TensorFlow将一台机器上的多核CPU视作单个设备,并且在内部使用线程并行
# cross_device_ops    `CrossDeviceOps`的子类的实例,默认使用`NcclAllReduce()`.通常在NCCL不可用或者有可用的
#                     能够充分利用特殊硬件的特殊实现时自定义此参数
```

```python
>>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
>>> with strategy.scope():
  x = tf.Variable(1.)       # 在`MirroredStrategy`下创建的变量是一个`MirroredVariable`
>>> x
MirroredVariable:{
  0: <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=1.0>,
  1: <tf.Variable 'Variable/replica_1:0' shape=() dtype=float32, numpy=1.0>
}
```

### cluster_resolver

## experimental.MultiWorkerMirroredStrategy

多机多卡同步训练。此策略下模型的参数是 `MirroredVariable` 类型的变量，在所有的模型副本中保持同步。

```python
```

## NcclAllReduce

Nvidia NCCL all-reduce 算法的实现。默认使用的 all-reduce 算法。

## OneDeviceStrategy

在单个设备上运行。在此策略下创建的变量和通过 `strategy.run()` 调用的函数都会被放置在指定设备上。此策略通常用于在使用其它策略实际分布训练到多个设备/机器之前，测试代码对于 `tf.distribute.Strategy` API 的使用。

```python
tf.distribute.OneDeviceStrategy(device)
```

```python
>>> strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
>>> with strategy.scope():
  v = tf.Variable(1.0)
  print(v.device)
/job:localhost/replica:0/task:0/device:GPU:0
>>> def step_fn(x):
  return x * 2
>>> result = 0
>>> for i in range(10):
  result += strategy.run(step_fn, args=(i,))
>>> print(result)
90
```

## partitioners.FixedShardsPartitioner

将数据（张量）分割为固定份数。

```python
>>> partitioner = FixedShardsPartitioner(num_shards=2)
>>> partitioner(tf.TensorShape([10, 3]), tf.float32, axis=0)
[2, 1]
>>> partitioner(tf.TensorShape([10, 3]), tf.float32, axis=1)
[1, 2]
```

## partitioners.MinSizePartitioner

在保证每一份的最小规模的前提下，将数据（张量）分割为尽量多的份数。

```python
>>> partitioner = MinSizePartitioner(min_shard_bytes=24, max_shards=4)   # 每一份最小为24字节
>>> partitioner(tf.TensorShape([10, 3]), tf.float32, axis=0)
[4, 1]              # 最多分为4份
>>> partitioner = MinSizePartitioner(min_shard_bytes=24, max_shards=8)
>>> partitioner(tf.TensorShape([10, 3]), tf.float32, axis=0)
[5, 1]              # 分为5份,每份最小为24字节
```

## partitioners.Partitioner

所有分割器的基类。

### \__call__

分割指定的张量形状并返回分割结果。

```python
__call__(shape, dtype, axis=0)
# shape     要分割的张量的形状,是`tf.TensorShape`实例
# dtype     要分割的张量的数据类型
# axis      沿此轴进行分割
```

## ReduceOp

表示一组值的归约方法。`ReduceOp.SUM` 表示求和，`ReduceOp.MEAN` 表示求平均值。

## ReductionToOneDevice

一种 `CrossDeviceOps` 实现，其复制所有值到一个设备上进行归约，再广播归约结果到目标 rank。

```python
tf.distribute.ReductionToOneDevice(reduce_to_device=None, accumulation_fn=None)
# reduce_to_device   进行归约的中间设备
# accumulation_fn
```

## ReplicaContext

此类具有在模型副本上下文中调用的一系列 API，其实例通常由 `get_replica_context()` 得到，用于在 `strategy.run()` 传入的函数中调用以获取模型副本的信息。

```python
>>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
>>> def f():
  replica_context = tf.distribute.get_replica_context()
  return replica_context.replica_id_in_sync_group
>>> strategy.run(f)
PerReplica:{
  0: <tf.Tensor: shape=(), dtype=int32, numpy=0>,
  1: <tf.Tensor: shape=(), dtype=int32, numpy=1>
}
```

### all_gather

### all_reduce

### merge_call

### num_replicas_in_sync

返回进行梯度汇总的模型副本的数量。

### replica_id_in_sync_group

返回模型副本的索引。

### strategy

返回当前的 `Strategy` 实例。

## Server

进程内的 TensorFlow 服务器，用于分布式训练。

服务器从属于一个集群（通过 `tf.train.ClusterSpec` 指定），并对应指定名称的 job 中的一个特定 task。服务器可以与同一集群中的所有其它服务器进行通信。

```python
tf.distribute.Server(server_or_cluster_def, job_name=None, task_index=None, protocol=None, config=None, start=True)
# server_or_cluster_def    `tf.train.ServerDef`或`tf.train.ClusterDef`协议缓冲区,或`tf.train.ClusterSpec`对象,
#                          用于描述要创建的服务器或其从属的集群
# job_name                 服务器从属的job的名称.默认为`server_or_cluster_def`中的相应值(如果指定了该值)
# task_index               服务器对应的task的索引.默认为`server_or_cluster_def`中的相应值(如果指定了该值);
#                          若job仅有一个task,则默认为0
# protocol                 服务器使用的协议,可以是`'grpc'`或`'grpc+verbs'`.默认为`server_or_cluster_def`中的
#                          相应值(如果指定了该值);其余情况下默认为`'grpc'`
# config
# start                    若为`True`,则在创建服务器之后立即启动
```

### create_local_server()

在本地主机创建一个新的单进程集群。

此方法是一个便利的包装器，用于创建一个服务器，其 `tf.train.ServerDef` 指定了一个单进程集群，集群在名为 `'local'` 的 job 下包含单个 task。

```python
@staticmethod
create_local_server(config=None, start=True)
# config                   
# start                    若为`True`,则在创建服务器之后立即启动它
```

### join()

阻塞直到服务器关闭。

### start()

启动服务器。

## Strategy

在一组设备上进行分布式计算的策略。

### distribute_datasets_from_function()

接收一个输入函数并返回一个分布式数据集（`tf.distribute.DistributedDataset` 实例）。用户传入的输入函数应接收一个 `tf.distribute.InputContext` 实例，返回一个 `tf.data.Dataset` 实例，并进行由用户自定义的分批和分割操作，`tf.distribute` 不会对返回的数据集再进行任何修改。相对于 `experimental_distribute_dataset()`，此方法不仅更加灵活（允许用户自定义分批和分割操作），而且在用于分布式训练时显示出了更好的伸缩性和性能。

```python
>>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
>>> def dataset_fn(input_context):                # 定义输入函数
    global_batch_size = 4
    batch_size = input_context.get_per_replica_batch_size(global_batch_size)
    dataset = tf.data.Dataset.range(8).batch(global_batch_size)
    # dataset = dataset.shard(                      # 手动分割数据集,适用于`MultiWorkerMirroredStrategy`
    #     input_context.num_input_pipelines, input_context.input_pipeline_id)
    dataset = dataset.unbatch().batch(batch_size) # 手动再分批
    dataset = dataset.prefetch(2)                 # 手动添加预取;每个设备预取2个批次(而不是全局总共预取2个批次)
    return dataset
>>> dist_dataset = strategy.distribute_datasets_from_function(dataset_fn)
>>> for x in dist_dataset:
    print(x)
PerReplica:{
  0: tf.Tensor([0 1], shape=(2,), dtype=int64),
  1: tf.Tensor([2 3], shape=(2,), dtype=int64)
}
PerReplica:{
  0: tf.Tensor([4 5], shape=(2,), dtype=int64),
  1: tf.Tensor([6 7], shape=(2,), dtype=int64)
}
```

### experimental_distribute_dataset()

将数据集（`tf.data.Dataset` 实例）转换为分布式数据集（`tf.distribute.DistributedDataset` 实例），`tf.data.Dataset` 实例的批次规模就是分布式训练的全局批次规模。如果你没有特定的想要去分割数据集的方法，则推荐使用此方法。

```python
>>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
>>> dataset = tf.data.Dataset.from_tensor_slices([5., 6., 7., 8.]).batch(2)
>>> dist_dataset = strategy.experimental_distribute_dataset(dataset)
>>> @tf.function
def f(x):
  return x * 2.0
>>> for x in dist_dataset:
    print(strategy.run(f, args=(x,)))
PerReplica:{
  0: tf.Tensor([10.], shape=(1,), dtype=float32),
  1: tf.Tensor([12.], shape=(1,), dtype=float32)
}
PerReplica:{
  0: tf.Tensor([14.], shape=(1,), dtype=float32),
  1: tf.Tensor([16.], shape=(1,), dtype=float32)
}
```

此方法在底层进行了三个关键操作：分批、分割和预取。

**分批**

对输入数据集进行重新分批，新的批次规模等于全局批次规模除以同步的模型副本数量。例如：

* 输入：`tf.data.Dataset.range(10).batch(4, drop_remainder=False)`

  原分批：`[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]`

  模型副本数量为 2 时分批：副本 1：`[0, 1], [4, 5], [8]`；副本 2：`[2, 3], [6, 7], [9]`

* 输入：`tf.data.Dataset.range(8).batch(4)`

  原分批：`[0, 1, 2, 3], [4, 5, 6, 7]`

  模型副本数量为 3 时分批：副本 1：`[0, 1], [4, 5]`；副本 2：`[2, 3], [6, 7]`；副本 3：`[], []`
  
* 输入：`tf.data.Dataset.range(10).batch(5)`
  
  原分批：`[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]`
  
  模型副本数量为 3 时分批：副本 1：`[0, 1], [5, 6]`；副本 2：`[2, 3], [7, 8]`；副本 3：`[4], [9]`

> 上面的例子仅用于展示一个全局批次是如何划分到多个模型副本中的，实际使用时不应对划分结果有任何的假定，因为划分结果可能会随着具体实现而发生变化。

重新分批操作的空间复杂度与模型副本的数量呈线性关系，因此当模型副本数量较多时输入流水线可能会引发 OOM 错误。

**分割**

对输入数据集进行自动分割（在 `MultiWorkerMirroredStrategy` 下），每个模型副本被（不重复不遗漏地）分配原数据集的一个子集，具体到每个 step 中，每个模型副本被（不重复不遗漏地）分配全局批次的一个子集并处理。

自动分割有三种策略可供选择，通过以下方式进行设定：

```python
>>> dataset = tf.data.Dataset.range(16).batch(4)
>>> options = tf.data.Options()                                                      # 设定为`DATA`
>>> options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
>>> dataset = dataset.with_options(options)
```

* `DATA`：将数据集的样本自动分割到所有模型副本。每个模型副本会读取整个数据集，保留分割给它的那一份，丢弃所有其它的份。此策略通常用于输入文件数量小于模型副本数量的情形，例如将 1 个输入文件分布到 2 个模型副本中：
  + 全局批次规模为 4
  + 文件：`[0, 1, 2, 3, 4, 5, 6, 7]`
  + 副本 1：`[0, 1], [4, 5]`；副本 2：`[2, 3], [6, 7]`
* `FILE`：将输入文件分割到所有模型副本。每个模型副本会读取分配给它的输入文件，而不会去读取其它文件。此策略通常用于输入文件数量远大于模型副本数量的情形（并且数据均匀分布在各文件中），例如将 2 个输入文件分布到 2 个模型副本中：
  + 全局批次规模为 4
  + 文件 1：`[0, 1, 2, 3]`；文件 2： `[4, 5, 6, 7]`
  + 副本 1：`[0, 1], [2, 3]`；副本 2：`[4, 5], [6, 7]`
* `AUTO`：默认选项。首先尝试 `FILE` 策略，如果没有检测到基于文件的数据集则尝试失败；然后尝试 `DATA` 策略。
* `OFF`：关闭自动分割，每个模型副本会处理所有样本：
  + 文件：`[0, 1, 2, 3, 4, 5, 6, 7]`
  + 副本 1：`[0, 1], [2, 3], [4, 5], [6, 7]`；副本 2：`[0, 1], [2, 3], [4, 5], [6, 7]`

**预取**

默认对输入数据集增加一个 `prefetch()` 变换，参数 `buffer_size` 取模型副本的数量。

### experimental_distribute_values_from_function()

### gather()

```python
>>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
INFO:tensorflow:Using MirroredStrategy with devices ('/job:localhost/replica:0/task:0/device:GPU:0', '/job:localhost/replica:0/task:0/device:GPU:1')
>>> distributed_values = strategy.experimental_distribute_values_from_function(lambda _: tf.identity(tf.constant([[1], [2]])))

>>> with strategy.scope():
    distributed_values = tf.Variable([[1], [2]])

strategy.gather(distributed_values, axis=0)

INFO:tensorflow:Gather to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
<tf.Tensor: shape=(4, 1), dtype=int32, numpy=
array([[1],
       [2],
       [1],
       [2]], dtype=int32)>
```

### num_replicas_in_sync

返回进行梯度汇总的模型副本的数量。

### reduce()

### run()

```python
run(fn, args=(), kwargs=None, options=None)
```

在每个模型副本上调用 `fn`，使用给定的参数。

### scope()

返回一个上下文管理器，

```python
strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
# Variable created inside scope:
with strategy.scope():
    mirrored_variable = tf.Variable(1.)
mirrored_variable

# Variable created outside scope:
regular_variable = tf.Variable(1.)
regular_variable
```
