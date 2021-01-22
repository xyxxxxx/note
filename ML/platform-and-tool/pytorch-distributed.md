[toc]



# 数据并行训练

**分布式数据并行训练(Distributed Data-Parallel Training, DDP)**是一种广泛采用的单程序多数据的训练范例。DDP中，模型会被复制到多个进程中，而每个模型副本都会传入不同的输入数据集（可能是对同一数据集的切分）。DDP负责梯度通信，以保证各模型副本同步和梯度计算叠加。

PyTorch提供了几种数据并行训练的选项。对于逐渐从简单到复杂、从原型到生产的各种应用，常见的发展轨迹为：

1. 使用**单机训练**：如果数据和模型可以在单个GPU中完成训练，并且训练速度不成问题
2. 使用**单机多卡数据并行**：如果机器上有多个GPU，并且想通过最少的代码修改来加速训练
3. 使用**单机多卡分布式数据并行**：如果你想进一步加速训练，通过多增加一些代码
4. 使用**多机分布式数据并行和启动脚本**：如果应用需要在多机之间伸缩
5. 使用torchelastic以启动分布式训练：如果可能出错或者在训练过程中动态地增减资源



## `torch.nn.DataParallel`

DataParallel以最小的代码障碍实现单机多GPU数据并行，它只需要在应用中增加一行代码。



在PyTorch中使用GPU非常简单，只需要把模型放到GPU中：

```python
device = torch.device("cuda:0")
model.to(device)
```

再复制所有的张量到GPU中：

```python
mytensor = my_tensor.to(device)
```

注意`my_tensor.to(device)`返回的是`my_tensor`在GPU中的一个新副本，因此你需要将其赋给一个新的张量并使用该张量。

PyTorch默认只使用一个GPU，你可以使用`DataParallel`来让模型并行运行在多个GPU上：

```Python
model = nn.DataParallel(model)
```

下面是一个详细的例子：

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Parameters and DataLoaders
input_size = 5
output_size = 2

batch_size = 30
data_size = 100

class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),
                         batch_size=batch_size, shuffle=True)

class Model(nn.Module):
    # Our model

    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("\tIn Model: input size", input.size(),
              "output size", output.size())

        return output
    
model = Model(input_size, output_size)
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  model = nn.DataParallel(model)

model.to(device)

for data in rand_loader:
    input = data.to(device)
    output = model(input)
    print("Outside: input size", input.size(),
          "output_size", output.size())
```

如果机器没有GPU或只有一个GPU，那么`In Model`和`Outside`的输入是相同的：

```
    In Model: input size torch.Size([30, 5]) output size torch.Size([30, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
	In Model: input size torch.Size([30, 5]) output size torch.Size([30, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
	In Model: input size torch.Size([30, 5]) output size torch.Size([30, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
	In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
Outside: input size torch.Size([10, 5]) output_size torch.Size([10, 2])
```

如果有两个GPU，那么每个GPU各有一个模型副本，各处理`input`的二分之一：

```python
# on 2 GPUs
Let's use 2 GPUs!
    In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
    In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
    In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
    In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
    In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
    In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
    In Model: input size torch.Size([5, 5]) output size torch.Size([5, 2])
    In Model: input size torch.Size([5, 5]) output size torch.Size([5, 2])
Outside: input size torch.Size([10, 5]) output_size torch.Size([10, 2])
```

类似地，如果有8个GPU：

```python
Let's use 8 GPUs!
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
    In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
    In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
    In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
    In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
    In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
Outside: input size torch.Size([10, 5]) output_size torch.Size([10, 2])
```



## `torch.nn.parallel.DistributedDataParallel`

### 比较`DataParallel`和`DistributedDataParallel`

首先让我们了解为什么你应该考虑使用`DistributedDataParallel`而非`DataParallel`:

+ 首先，`DataParallel`是单进程多线程，只能单机运行，而`DistributedDataParallel`是多进程，可以单机或多机运行。 `DataParallel`通常比`DistributedDataParallel` 慢，即便是单机运行，因为线程间的GIL争夺、每次迭代都复制模型，以及切分输入和汇总输入带来的花销。
+ 如果你的模型太大以至于不能在单个GPU上训练，那么就必须用模型并行来将其切分到多个GPU中。`DistributedDataParallel`兼容模型并行而`DataParallel`不能。当DDP结合模型并行时，每个DDP进程都会使用模型并行，并且所有的进程共同使用数据并行。
+ 如果你的模型需要跨机器或者不适用于数据并行范例，请参考RPC API。



### 示例1

让我们看一个简单的`torch.nn.parallel.DistributedDataParallel`例子：

```python
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP


def example(rank, world_size):
    # create default process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    # create local model
    model = nn.Linear(10, 10).to(rank)
    # construct DDP model
    ddp_model = DDP(model, device_ids=[rank])
    # define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    # forward pass
    outputs = ddp_model(torch.randn(20, 10).to(rank))
    labels = torch.randn(20, 10).to(rank)
    # backward pass
    loss_fn(outputs, labels).backward()
    # update parameters
    optimizer.step()

def main():
    world_size = 2
    mp.spawn(example,
        args=(world_size,),
        nprocs=world_size,
        join=True)

if __name__=="__main__":
    main()
```

其中，local model是一个线性层，将其用DDP包装后，对DDP模型进行一次前馈计算、反向计算和更新参数。在这之后，local model的参数会被更新，并且所有进程的模型都完全相同。



### 示例2

首先设置进程组。

```python
import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
```

然后创建一个玩具模型，用DDP包装，并输入一些随机数据。请注意，DDP构造函数广播rank0进程的模型状态到所有其它进程，因此不必担心不同的进程有不同的模型初始值。

```python
class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    # create model and move it to GPU with id rank
    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(rank)
    loss_fn(outputs, labels).backward()
    optimizer.step()

    cleanup()


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)
```

可以看到，DDP包装了底层分布式通信的细节并提供了一个简洁的API。梯度同步通信发生在反向计算过程中，并叠加反向计算。当`backward()`返回时，`param.grad`已经包含了同步的梯度张量。



### 内部设计

+ 前提：DDP依赖c10d `ProcessGroup`用于进程间通信，因此应用在构建DDP之前必须先创建`Process Group`实例
+ 构造：DDP构造函数引用本地模块，并广播rank0进程的`state_dict()`到组内的所有进程以确保所有模型副本都从同样的状态开始。随后每个DDP进程创建一个本地`Reducer`，其在之后的反向计算过程中负责梯度同步。为了提高通信效率，`Reducer `组织参数梯度为桶结构，每次reduce一个桶。……



### 注意事项

对于基础使用，DDP仅需要多一点的LoC来创建进程组；而当DDP应用到更高级的用例中，则还有一些注意事项。



**不一致的进程速度**

在DDP中，构造函数、前向计算和反向计算是分布式同步点。不同的进程应当在大致相同的时间到达这些同步点，否则快的进程会先到而等待落后的进程。因此用户应负责进程之间的负载均衡。

有时由于网络延迟、资源争夺、无法预测的负载峰值等原因，不一致的进程速度也无法避免。但为了防止这些情形下的超时，在调用`init_process_group`时请确保`timeout`传入了一个足够大的值。



**保存和加载检查点**

使用`torch.save`和`torch.load`在检查点保存和恢复模型是非常常见的操作。使用DDP时的一种优化方法是，保存模型仅在一个进程中进行，而加载模型则加载到所有进程，这样可以减少写的花销。

当加载模型时，你需要提供一个合适的`map_location`参数以防止进程进入其它的设备。当`map_location`参数缺失时，`torch.load`会首先将模型加载到CPU，再将每一个参数复制到它被保存的地方，这将导致同一机器上的所有进程会使用相同的设备。对于更高级的错误恢复和弹性支持，请参考TorchElastic。

```python
def demo_checkpoint(rank, world_size):
    print(f"Running DDP checkpoint example on rank {rank}.")
    setup(rank, world_size)

    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    CHECKPOINT_PATH = tempfile.gettempdir() + "/model.checkpoint"
    if rank == 0:
        # All processes should see same parameters as they all start from same
        # random parameters and gradients are synchronized in backward passes.
        # Therefore, saving it in one process is sufficient.
        torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)

    # Use a barrier() to make sure that process 1 loads the model after process
    # 0 saves it.
    dist.barrier()
    # configure map_location properly
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    ddp_model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=map_location))

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(rank)
    loss_fn = nn.MSELoss()
    loss_fn(outputs, labels).backward()
    optimizer.step()

    # Not necessary to use a dist.barrier() to guard the file deletion below
    # as the AllReduce ops in the backward pass of DDP already served as
    # a synchronization.

    if rank == 0:
        os.remove(CHECKPOINT_PATH)

    cleanup()
```



**结合DDP和模型并行**

DDP兼容多GPU模型。当用巨量数据训练大型模型时，用DDP包装多GPU模型十分有用。

```python
class ToyMpModel(nn.Module):
    def __init__(self, dev0, dev1):
        super(ToyMpModel, self).__init__()
        self.dev0 = dev0
        self.dev1 = dev1
        self.net1 = torch.nn.Linear(10, 10).to(dev0)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 5).to(dev1)

    def forward(self, x):
        x = x.to(self.dev0)
        x = self.relu(self.net1(x))
        x = x.to(self.dev1)
        return self.net2(x)
```

当传递一个多GPU模型到DDP时不能设置`device_ids`和`output_device`，输入和输出数据会被放在合适的设备中。

```python
def demo_model_parallel(rank, world_size):
    print(f"Running DDP with model parallel example on rank {rank}.")
    setup(rank, world_size)

    # setup mp_model and devices for this process
    dev0 = rank * 2
    dev1 = rank * 2 + 1
    mp_model = ToyMpModel(dev0, dev1)
    ddp_mp_model = DDP(mp_model)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_mp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    # outputs will be on dev1
    outputs = ddp_mp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(dev1)
    loss_fn(outputs, labels).backward()
    optimizer.step()

    cleanup()


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    if n_gpus < 8:
      print(f"Requires at least 8 GPUs to run, but got {n_gpus}.")
    else:
      run_demo(demo_basic, 8)
      run_demo(demo_checkpoint, 8)
      run_demo(demo_model_parallel, 4)
```







### 详解

```python
torch.distributed.init_process_group( )
```







# 一般分布式训练

许多训练范例不用于数据并行，例如参数服务器范例、分布式管道范例等。torch.distributed.rpc的目标就是支持一般的分布式训练场景。







# TorchElastic