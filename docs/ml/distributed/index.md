# 分布式训练

分布式训练将模型训练的工作负载分配到多个处理器，这些处理器，称为工作器节点，并行工作以加速模型训练。

## 基本概念

!!! abstract "参考"
    * [Module: tf.distribute](https://www.tensorflow.org/api_docs/python/tf/distribute)

### 机器，设备

**机器（machine）**：计算机，通常指物理机。

**设备（device）**：CPU 或加速器（GPU、TPU 等），通常指物理的 CPU 或加速器（GPU、TPU 等）。TensorFlow、PyTorch 等机器学习框架可以在这些设备上执行运算。单台机器上可以有多个设备（CPU 和多个 GPU）。

### 数据并行 vs 模型并行

**数据并行（data parallelism）**：在多个设备上运行相同的模型副本，输入数据被分为多份并分别传入各个模型副本。

**模型并行（model parallelism）**：将一个（大型）模型的各个部分拆分到多个设备上运行，相当于多个设备组成一个模型的流水线。

![](https://s2.loli.net/2022/09/29/YRJU5DeNv4EHsAi.png)

此外也可以采用数据并行和模型并行的混合方法。

一般情况下会采用数据并行的方式，各模型副本独立，规模伸缩性好。

模型并行一般用于一台设备的内存或一个 GPU 的显存已经无法容纳的大型模型，此时模型被拆分放置在多台机器或多个 GPU 上，模型的各部分之间存在依赖，健壮性差，网络通信开销大，规模伸缩性差。

!!! info "并行范式"
    [并行技术](https://www.colossalai.org/zh-Hans/docs/concepts/paradigms_of_parallelism)更加详细地介绍了现有的各种并行训练方法。

### 模型副本，工作器，rank，参数服务器，变量

**模型副本（replica）**

数据并行中的模型副本。

**工作器（worker）**

具有一个或多个物理设备的物理机器，其中运行着一个或多个模型副本。通常一个工作器对应一台机器，但在模型并行的情形下，一个工作器可能对应多台机器。

也称为工作节点（worker node）。

**rank**

模型副本在数据并行中的序号。

**local rank**

模型副本在当前工作器上的序号。

**world size**

数据并行中的模型副本总数。

**参数服务器（parameter server）**

保存模型参数和变量的唯一副本的机器，用于一些策略。所有的模型副本都必须在 step 开始时从参数服务器获取参数，在 step 结束时向参数服务器发送一个更新。

![](https://s2.loli.net/2022/09/29/5TR7CrwtEZPnqil.png)

**分布式变量（distributed variable）**

在不同设备上代表不同的值，它包含一个模型副本 ID 到值的映射。

**镜像变量（mirrored variable）**

在不同设备上代表同一个值，变量的更新会应用到每一个副本以保持同步。

### 同步更新 vs 异步更新

数据并行中模型参数更新的方式分为以下两种：

**同步（synchronous）更新**：所有模型副本的梯度计算完成之后，归约起来统一更新模型参数，再将新参数广播给所有模型副本。

**异步（asynchronous）更新**：每个模型副本的梯度计算完成之后，向模型参数发送一个更新，模型参数应用这个更新并将新参数发送给该模型副本。

![](https://s2.loli.net/2022/09/29/jdb2fwaps5KzA1S.png)

此外也可以采用同步更新和异步更新的混合方法，例如为模型副本分组，组内同步，组间异步。

同步更新需要等待最后完成计算的模型副本。

异步更新无需等待，但是会涉及到梯度过时、噪声大等更复杂的问题。因此一般情况下会采用同步更新的方式。

## 分布式策略

!!! abstract "参考"
    * [使用 TensorFlow 进行分布式训练](https://www.tensorflow.org/guide/distributed_training)

### 多工作器同步训练

在（一个或多个工作器的）多个设备上的同步训练。

参见：

* TensorFlow2：[tf.distribute.MirroredStrategy](https://www.tensorflow.org/api_docs/python/tf/distribute/MirroredStrategy)，[tf.distribute.MultiWorkerMirroredStrategy](https://www.tensorflow.org/api_docs/python/tf/distribute/MultiWorkerMirroredStrategy)
* PyTorch：[DistributedDataParallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)，[教程](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
* [Horovod](https://horovod.readthedocs.io/en/stable/index.html)

### 参数服务器训练

Parameter server training is a common data-parallel method to scale up a machine learning model on multiple machines. A parameter server training cluster consists of workers and parameter servers. Variables are created on parameter servers and they are read and updated by workers in each step. By default, workers read and update these variables independently without synchronizing with each other. Under this configuration, it is known as asynchronous training.

参见：

* TensorFlow2：[tf.distribute.experimental.ParameterServerStrategy](https://www.tensorflow.org/api_docs/python/tf/distribute/experimental/ParameterServerStrategy)
* PyTorch：[教程](https://pytorch.org/tutorials/intermediate/rpc_param_server_tutorial.html)
