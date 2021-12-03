# 机器，设备

**机器（machine）**：计算机，通常指物理机。

**设备（device）**：CPU、GPU 和 TPU，通常指物理的 CPU、GPU 和 TPU。TensorFlow、PyTorch 等机器学习框架可以在这些设备上执行运算。单台机器上可以有多个设备（CPU 和多个 GPU）。

# 数据并行 vs 模型并行

**数据并行（data parallelism）**：在多台机器上运行相同的模型副本，样本被分为多份并分别传入各个模型副本。

**模型并行（model parallelism）**：将一个（大型）模型的各个部分拆分到多台机器上运行，相当于多台机器组成一个模型的流水线。

![para](https://fyubang.com/2019/07/08/distributed-training/para.png)

此外也可以采用数据并行和模型并行的混合方法。

一般情况下会采用数据并行的方式，各模型副本独立，规模伸缩性好。

模型并行一般用于一个 GPU 的显存已经无法容纳的大型模型，此时模型被拆分放置在多个 GPU 或机器上，模型的各部分之间存在依赖，健壮性差，网络通信开销大，规模伸缩性差。

# 同步更新 vs 异步更新

数据并行中模型参数更新的方式分为以下两种：

**同步（synchronous）更新**：所有模型副本的梯度计算完成之后，归约起来统一更新模型参数，再将新参数广播给所有模型副本。

**异步（asynchronous）更新**：每个模型副本的梯度计算完成之后，向模型参数发送一个更新，模型参数应用这个更新并将新参数发送给该模型副本。

![syn_dp](https://fyubang.com/2019/07/08/distributed-training/syn_dp.jpg)

![asyn_dp](https://fyubang.com/2019/07/08/distributed-training/asyn_dp.jpg)

此外也可以采用同步更新和异步更新的混合方法，例如为模型副本分组，组内同步，组间异步。

同步更新需要等待最后完成计算的模型副本。

异步更新无需等待，但是会涉及到梯度过时、噪声大等更复杂的问题。因此一般情况下会采用同步更新的方式。

# worker, parameter server

**replica**

数据并行中的模型副本。

**host**

**worker**

具有一个或多个物理设备的物理机器，其中运行着一个或多个 replica。通常一个 worker 对应一台机器，但在模型并行的情形下，一个 worker 可能对应多台机器。

**参数服务器（parameter server）**

保存模型参数唯一副本的机器，用于一些策略。所有的 replica 都必须在 step 开始时从参数服务器获取参数，在 step 结束时向参数服务器发送一个更新。

**分布变量（distributed variable）**

在不同设备上代表不同的值，它包含一个 replica 到值的映射。

**镜像变量（distributed variable）**

在不同设备上代表同一个值，变量的更新会应用到每一个副本以保持同步。

