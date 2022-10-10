# Horovod

[Horovod](https://horovod.ai/) 是一套面向 TensorFlow、Keras、PyTorch 和 Apache MXNet 的分布式深度学习训练框架。Horovod 的目标是让分布式深度学习快速且易用。

## 安装

参见 [Horovod Installation Guide](https://github.com/horovod/horovod/blob/master/docs/install.rst)。

建议使用官方镜像 [Dockerfile.cpu](https://github.com/horovod/horovod/blob/master/Dockerfile.cpu) 和 [Dockerfile.gpu](https://github.com/horovod/horovod/blob/master/Dockerfile.gpu)。

## 基本概念

Horovod 基于下列 MPI 概念，这里结合实例进行解释。假设我们有 4 台机器，各有 4 个 GPU，在每个 GPU 上执行训练脚本的一个副本，那么：

* **size**：进程数，这里为 16
* **rank**：进程的唯一 ID，这里为 0-15
* **local rank**：进程在本机的唯一 ID，这里为 0-3
* **allreduce**, **allgather**, **broadcast**：参见 [MPI 集体通信模式](../distributed/mpi.md#集体通信模式)

## 脚本示例

* [Keras 示例](https://github.com/horovod/horovod/blob/master/examples/tensorflow2/tensorflow2_keras_mnist.py)。
* [PyTorch 示例](https://github.com/horovod/horovod/blob/master/examples/pytorch/pytorch_mnist.py)。
* [Lightning 示例](https://github.com/horovod/horovod/blob/master/examples/pytorch/pytorch_lightning_mnist.py)。

## 运行

下面的示例命令演示了如何启动分布式训练：

1. 单机多卡，例如在有 4 个 GPU 的单台机器上运行：

   ```shell
   $ horovodrun -np 4 -H localhost:4 python train.py
   ```

2. 多机多卡，例如在各有 4 个 GPU 的 4 台机器上运行：

   ```shell
   $ horovodrun -np 16 -H server1:4,server2:4,server3:4,server4:4 python train.py
   ```

## 弹性训练

!!! abstract "参考"
    * [Elastic Horovod](https://horovod.readthedocs.io/en/stable/elastic_include.html)

Horovod 的弹性训练允许在运行过程中动态地增加或减少工作器的数量，而无需重启训练或从保存到持久存储的检查点继续。

**使用场景**

* 运行自动伸缩的任务，随着训练的进行可以动态地获得或释放资源
* 任务运行在可以抢占或发现的实例上，这些实例可能在没有警告的情况下变得可用或不可用
* 存在不可靠的节点，在部分节点失效时训练能够继续

**要求**

* TensorFlow >= 1.15 或 PyTorch >= 1.0
* Horovod >= 0.20.0 并且有 Gloo 支持
* 运行过程中发现可用主机的方法。

### 使用状态同步修改训练脚本

弹性训练与一般分布式训练的最大不同在于，需要在工作器加入或移除时追踪和同步各工作器的状态，为此对训练脚本进行如下修改：

1. 将主要的训练过程（初始化之后的所有操作）包装到一个被 `hvd.elastic.run` 装饰的函数中。

   被装饰的函数的第一个参数是一个 `hvd.elastic.State` 实例，在执行被装饰的函数前，该状态实例会被同步到各工作器中。这将确保新增加的工作器以及状态不一致的工作器在训练开始前都同步到相同的状态。

   在此函数之前不可调用 Horovod 集体通信操作（Broadcast, All-Reduce, All-Gather 等）。

2. 将所有需要在各工作器间保持同步的变量（模型参数、优化器状态、训练参数等）放置到 `hvd.elastic.State` 实例中。

   Horovod 提供了 TensorFlow、Keras 和 PyTorch 的标准的状态类型实现，但在有些情况下也需要重载基类 `hvd.elastic.State` 以处理自定义广播操作。

3. 周期性地调用 `state.commit()` 以在内存中备份当前状态。

   这有助于防止当工作器意外出错时状态被破坏。例如，如果训练在参数更新的过程中出错，部分参数的梯度更新被应用而部分参数的梯度仍在执行 All-Reduce 操作，那么此时将引发一个 `HorovodInternalError`，并且所有的参数都将恢复到上一次提交的值。

   提交的代价可能十分昂贵（对于大型模型），因此你需要在提交的频率与回滚的距离之间寻求平衡。例如，如果你每 10 个 step 提交一次，那么相比每个 step 都提交，备份的花销降低为原来的十分之一，但是当训练出错时，你需要回滚到至多 10 个已处理的批次之前。

   Horovod 会通过（我们称为）工作器的*优雅移除*来回避此类回滚。当主进程发现一个主机被标记为移除时，它向所有工作器推送一个通知，在下次 `state.commit()` 或更轻量的 `state.check_host_updates()` 被调用时引发一个 `HostsUpdatedInterrupt` 以更新当前主机和工作器，此时参数不会恢复到上一次提交的值。

   通常情况下，如果你的硬件是比较可靠的，并且你的调度系统会在计划移除主机时给予主进程足够的警告，那么你可以安全地以比较低的频率调用 `state.commit()`，并在每个 step 结束时调用 `state.check_host_updates()`。

4. 使用 `hvd.elastic.State` 实例注册回调以因应训练过程中工作器成员的变化。

   例如根据新的全局规模重新调整学习率，或者重新划分数据集等操作通常在这些回调中完成。
   
   回调在 Horovod 重新初始化之后、状态在各工作器之间同步之前调用。

`HorovodInternalError`（出错）或 `HostsUpdatedInterrupt`（增加/移除请求）之后的重置过程如下：

1. 抓取 `hvd.elastic.run` 装饰器内的异常，若为 `HorovodInternalError`，恢复到上一次提交的状态。
2. 通过一轮新的协调组织重新初始化 Horovod 上下文。
3. 通过广播新的 0 号工作器的状态同步各工作器的状态。上一个步骤中，越老的工作器被指定为 0 号工作器的优先级越高，以确保广播的状态是最新的。
4. 继续训练，执行底层的训练函数。

### 脚本示例

* [Keras 示例](https://horovod.readthedocs.io/en/stable/elastic_include.html#elastic-keras)
* [PyTorch 示例](https://horovod.readthedocs.io/en/stable/elastic_include.html#elastic-pytorch)

### 使用 horovodrun 运行

弹性训练通过 `horovodrun` 命令行工具启动，启动时最大的不同是不再显式地指定主机，而是在运行过程中动态地发现主机。最通常的使  Horovod 发现可用主机的方法是在 `--host-discovery-script` 选项下提供一个脚本：

```shell
$ horovodrun -np 8 --host-discovery-script discover_hosts.sh python train.py
```

此主机发现脚本需要有用户执行权限，并且以 `<hostname>:<slots>` 的格式每行返回一个主机和它的可用槽位，例如：

```shell
$ ./discover_hosts.sh
host-1:4
host-2:4
host-3:4
```

如果此主机发现脚本执行失败（由于权限问题）或……

### 实践过程中的思考

## API

### 共有

#### allgather

All-Gather 操作。所有收集的张量在第一个维度进行拼接，因此各进程提供的张量必须具有相同的形状，除了第一个维度的规模可以不同。

```python
horovod.tensorflow.allgather(tensor, name=None, ignore_name_scope=False)
# tensor         收集的数据,是`tf.Tensor`,`tf.Variable`或`tf.IndexedSlices`类型
```

```python
horovod.tensorflow.keras.allgather(value, name=None)
# value          收集的数据,是`tf.Tensor`,`tf.Variable`或`tf.IndexedSlices`类型
```

```python
horovod.keras.allgather(value, name=None)
# value          收集的数据,是`tf.Tensor`,`tf.Variable`或`tf.IndexedSlices`类型
```

```python
horovod.torch.allgather(tensor, name=None)
# tensor         收集的数据,是`torch.Tensor`类型
```

#### allreduce

All-Reduce 操作。

```python
horovod.tensorflow.allreduce(
    tensor, average=None, device_dense='', device_sparse='',
    compression=<class 'horovod.tensorflow.compression.NoneCompressor'>,
    op=None, prescale_factor=1.0, postscale_factor=1.0, name=None
)
# tensor         归约的数据,是`tf.Tensor`,`tf.Variable`或`tf.IndexedSlices`类型
# compression    用于减少数据通信量的压缩算法
```

```python
horovod.tensorflow.keras.allreduce(
    value, name=None, average=None, prescale_factor=1.0, postscale_factor=1.0,
    op=None, compression=<class 'horovod.tensorflow.compression.NoneCompressor'>
)
# value          归约的数据,是`tf.Tensor`,`tf.Variable`或`tf.IndexedSlices`类型
# compression    用于减少数据通信量的压缩算法
```

```python
horovod.keras.allreduce(
    value, name=None, average=True, prescale_factor=1.0, postscale_factor=1.0
)
# value          归约的数据,是`tf.Tensor`,`tf.Variable`或`tf.IndexedSlices`类型
```

```python
horovod.torch.allreduce(
    tensor, average=None, name=None, 
    compression=<class 'horovod.torch.compression.NoneCompressor'>, op=None,
    prescale_factor=1.0, postscale_factor=1.0
)
# tensor         归约的数据,是`torch.Tensor`类型
# compression    用于减少数据通信量的压缩算法
```

#### broadcast

Broadcast 操作。

```python
horovod.tensorflow.broadcast(tensor, root_rank, name=None, ignore_name_scope=False)
# tensor        广播的数据,是`tf.Tensor`,`tf.Variable`或`tf.IndexedSlices`类型
# root_rank     发送数据的进程的秩
```

```python
horovod.tensorflow.keras.broadcast(value, root_rank, name=None)
# value         广播的数据,是`tf.Tensor`,`tf.Variable`或`tf.IndexedSlices`类型
# root_rank     发送数据的进程的秩
```

```python
horovod.keras.broadcast(value, root_rank, name=None)
# value         广播的数据,是`tf.Tensor`,`tf.Variable`或`tf.IndexedSlices`类型
# root_rank     发送数据的进程的秩
```

```python
horovod.torch.broadcast(tensor, root_rank, name=None)
# tensor        广播的数据,是`torch.Tensor`类型
# root_rank     发送数据的进程的秩
```

#### Compression

可选的 All-Reduce 操作中用于减少数据通信量的压缩算法。

##### NoneCompressor

##### FP16Compressor

#### cuda_built()

若 Horovod 编译时包含了 CUDA 支持，返回 `True`。

#### DistributedOptimizer

返回一个包装了原优化器的分布式优化器，其负责各进程间的通信，计算梯度值和应用参数更新则委托原优化器完成。

```python
horovod.tensorflow.DistributedOptimizer(
    optimizer, name=None, use_locking=False, device_dense='', device_sparse='',
    compression=<class 'horovod.tensorflow.compression.NoneCompressor'>,
    sparse_as_dense=False, backward_passes_per_step=1,
    op=<MagicMock name='mock().horovod_reduce_op_average()' id='140316232634960'>,
    gradient_predivide_factor=1.0, average_aggregated_gradients=False,
    num_groups=0, groups=None
)
# optimizer      用于计算梯度和应用参数更新的优化器
# compression    All-Reduce操作中用于减少数据通信量的压缩算法
```

```python
horovod.tensorflow.keras.DistributedOptimizer(
    optimizer, name=None, device_dense='', device_sparse='',
    compression=<class 'horovod.tensorflow.compression.NoneCompressor'>,
    sparse_as_dense=False, gradient_predivide_factor=1.0,
    op=<MagicMock name='mock().horovod_reduce_op_average()' id='140316232634960'>,
    backward_passes_per_step=1, average_aggregated_gradients=False,
    num_groups=0, groups=None
)
# optimizer      用于计算梯度和应用参数更新的优化器
# compression    All-Reduce操作中用于减少数据通信量的压缩算法
# backward_passes_per_step        
# average_aggregated_gradients   
```

```python
horovod.keras.DistributedOptimizer(
    optimizer, name=None, device_dense='', device_sparse='',
    compression=<class 'horovod.tensorflow.compression.NoneCompressor'>,
    sparse_as_dense=False, gradient_predivide_factor=1.0,
    op=<MagicMock name='mock().horovod_reduce_op_average()' id='140316232634960'>,
    num_groups=0, groups=None
)
# optimizer      用于计算梯度和应用参数更新的优化器
# compression    All-Reduce操作中用于减少数据通信量的压缩算法
```

```python
horovod.torch.DistributedOptimizer(
    optimizer, named_parameters=None,
    compression=<class 'horovod.torch.compression.NoneCompressor'>,
    backward_passes_per_step=1,
    op=<MagicMock name='mock().horovod_reduce_op_average()' id='140316224808592'>,
    gradient_predivide_factor=1.0, num_groups=0, groups=None,
    sparse_as_dense=False
)
# optimizer          用于计算梯度和应用参数更新的优化器
# named_parameters   参数名称到值的映射,用于allreduce操作的命名.一般就是`model.named_parameters()`
# compression        All-Reduce操作中用于减少数据通信量的压缩算法
```

#### elastic.run()

用于运行弹性训练过程的装饰器。参见[弹性训练](#弹性训练)。

#### gloo_enabled()

若 Gloo 在当前运行时可用，返回 `True`。

#### gloo_built()

若 Horovod 编译时包含了 Gloo 支持，返回 `True`。

#### init()

初始化 Horovod。

```python
horovod.tensorflow.init(comm=None)
# comm     通讯器,给定的通讯器将被复制并使用副本,默认使用`MPI_COMM_WORLD`通讯器
```

#### is_initialized()

若 Horovod 已经初始化，返回 `True`。

#### local_rank()

返回当前进程的本地 Horovod rank。

#### local_size()

返回当前进程所在节点上的 Horovod 进程数。

#### mpi_threads_supported()

若支持 MPI 多线程，返回 `True`。

#### mpi_enabled()

若 MPI 在当前运行时可用，返回 `True`。

#### mpi_built()

若 Horovod 编译时包含了 MPI 支持，返回 `True`。

#### nccl_built()

若 Horovod 编译时包含了 NCCL 支持，返回 `True`。

#### rank()

返回当前进程的 Horovod rank。

#### shutdown()

关闭 Horovod。

#### size()

返回 Horovod 进程数。

#### start_timeline()

创建时间线（日志）文件并开始记录。

```python
horovod.tensorflow.start_timeline(file_path, mark_cycles=False)
# file_path    时间线文件的路径
# mark_cycles  若为`True`,时间线中将标记循环
```

#### stop_timeline()

停止记录时间线并关闭文件。

### horovod.tensorflow

#### alltoall

All-to-all 操作。所有发送和接收的张量在第一个维度进行切分和拼接，因此各进程提供的张量必须具有相同的形状，除了第一个维度的规模可以不同。

```python
horovod.tensorflow.alltoall(
    tensor, splits=None, name=None, ignore_name_scope=False
)
# tensor       分发的数据,是`tf.Tensor`,`tf.Variable`或`tf.IndexedSlices`类型
# splits       指示数据分发的数组,索引为i的整数n表示`tensor`接下来的n个元素向秩为i的进程发送.若为`None`,则`tensor`
#              的所有元素将被均分并发送到每个进程
```

#### cross_rank()

返回当前进程所在节点的 rank。

#### cross_size()

返回与当前进程具有相同本地 rank 的进程数。

#### is_homogeneous()

若集群的所有节点上的进程数相同，返回 `True`。

### horovod.tensorflow.keras

#### broadcast_global_variables()

根进程向所有（其它）进程广播所有全局变量。

```python
horovod.tensorflow.keras.broadcast_global_variables(root_rank)
# root_rank    发送数据的进程的秩
```

#### callbacks.BroadcastGlobalVariablesCallback

根进程向所有（其它）进程广播所有全局变量，以确保所有的进程的模型初始化是一致的。

```python
horovod.tensorflow.keras.callbacks.BroadcastGlobalVariablesCallback(
    root_rank, device
)
# root_rank    发送数据的进程的秩
```

#### callbacks.MetricAverageCallback

在 epoch 结束后对所有进程的指标求平均，常配合 `ReduceLROnPlateau`, `TensorBoard` 和其它指标相关的回调使用（必须在回调列表中位于这些回调之前）。

#### callbacks.LearningRateScheduleCallback

> 建议使用 Keras 的相关回调而非此回调。

计划学习率。

#### callbacks.LearningRateWarmupCallback

> 建议使用 Keras 的相关回调而非此回调。

学习率 warmup。

#### load_model()

使用 Horovod 分布式优化器加载保存的 Keras 模型。分布式优化器将包装原优化器，使用其计算梯度值和应用参数更新。

```python
horovod.tensorflow.keras.load_model(
    filepath, custom_optimizers=None, custom_objects=None,
    compression=<class 'horovod.tensorflow.compression.NoneCompressor'>
)
# filepath    模型的保存路径或h5格式的文件对象
```

### horovod.keras

#### broadcast_global_variables()

根进程向所有（其它）进程广播所有全局变量。

```python
horovod.keras.broadcast_global_variables(root_rank)
# root_rank    发送数据的进程的秩
```

#### load_model()

见 `horovod.tensorflow.keras.load_model()`。

#### callbacks.BroadcastGlobalVariablesCallback, callbacks.MetricAverageCallback, callbacks.LearningRateScheduleCallback, callbacks.LearningRateWarmupCallback

见 `horovod.tensorflow.keras.callbacks.BroadcastGlobalVariablesCallback`, `horovod.tensorflow.keras.callbacks.MetricAverageCallback`, `horovod.tensorflow.keras.callbacks.LearningRateScheduleCallback`, `horovod.tensorflow.keras.callbacks.LearningRateWarmupCallback`。

### horovod.torch

#### allgather_async()

All-Gather 操作的异步版本，返回此操作的用于 `poll()` 和 `synchronize()` 调用的柄。

#### allreduce_async()

All-Reduce 操作的异步版本，返回此操作的用于 `poll()` 和 `synchronize()` 调用的柄。

#### alltoall()

All-to-all 操作。所有发送和接收的张量在第一个维度进行切分和拼接，因此各进程提供的张量必须具有相同的形状，除了第一个维度的规模可以不同。

```python
horovod.torch.alltoall(tensor, splits=None, name=None)
# tensor       分发的数据,是`torch.Tensor`类型
# splits       指示数据分发的数组,索引为i的整数n表示`tensor`接下来的n个元素向秩为i的进程发送.若为`None`,则`tensor`
#              的所有元素将被均分并发送到每个进程
```

#### alltoall_async()

All-to-all 操作的异步版本，返回此操作的用于 `poll()` 和 `synchronize()` 调用的柄。

#### broadcast_async()

Broadcast 操作的异步版本，返回此操作的用于 `poll()` 和 `synchronize()` 调用的柄。

#### broadcast_object()

#### broadcast_optimizer_state()

从根进程广播优化器状态到所有其它进程。

```python
horovod.torch.broadcast_optimizer_state(optimizer, root_rank)
# optimizer      优化器
# root_rank      进程的rank,该进程的优化器将被广播到所有其它进程
```

#### broadcast_parameters()

从根进程广播参数状态到所有其它进程，主要用于广播 `model.state_dict()`, `model.named_parameters()` 和 `model.parameters()`。

```python
horovod.torch.broadcast_parameters(params, root_rank)
# params         模型参数
# root_rank      进程的rank,该进程的优化器将被广播到所有其它进程
```

#### cross_rank()

返回当前进程所在节点的 rank。

#### cross_size()

返回与当前进程具有相同本地 rank 的进程数。

#### join()

阻塞直到所有进程调用此方法，返回最后调用此方法的进程的 rank。

#### poll()

若异步操作完成，返回 `True`，此时调用 `synchronize()` 将不再阻塞。

```python
horovod.torch.poll(handle)
# handle      异步操作返回的柄
```

#### synchronize()

同步异步操作直到其完成，返回该操作的结果。

```python
horovod.torch.synchronize(handle)
# handle      异步操作返回的柄
```
