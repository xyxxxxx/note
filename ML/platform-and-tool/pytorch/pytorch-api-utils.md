[toc]

# torch.utils.data

PyTorch 数据加载功能的核心是 `torch.utils.data.Dataloader` 类，





## DataLoader详解

DataLoaderPyTorch 数据加载功能的核心类，其将一个数据集表示为一个 Python 可迭代对象。

```python
DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None, *, prefetch_factor=2,
           persistent_workers=False)
```

下面将详细介绍各参数。

### dataset

需要加载的 `DataSet` 对象。PyTorch 支持两种类型的数据集：

* **映射数据集**：实现了 `__getitem__()` 和 `__len()__` 方法，表示一个从键到数据样本的映射。例如调用 `dataset[idx]` 时，可以读取第 `idx` 个图像和相应的标签。
* **迭代数据集**：`IterableDataset` 的子类的对象，实现了 `__iter__()` 方法，表示一个数据样本的可迭代对象。此种数据集非常适用于随机读取非常昂贵的情形（如使用磁盘）。例如调用 `iter(dataset)` 时，可以返回一个从数据库、远程服务器或实时生成的日志的数据流。



### 加载顺序和Sampler

对于迭代数据集，加载数据的顺序完全由用户定义的可迭代对象控制。这使得区块读取和 batch 的实现更加简单快速。

对于映射数据集，`torch.utils.data.Sampler` 类用于指定加载数据过程的索引或键顺序，它们表示数据集索引的可迭代对象。例如在常规的 SGD 中，一个 `Sample` 对象可以随机产生索引的一个排列，每次 yield 一个索引；或者 yield 多个索引，实现 mini-batch SGD。

一个顺序或乱序的 sampler 基于 `DataLoader` 的 `shuffle` 参数构建。或者，也可以通过传入参数自定义一个 `Sampler` 对象，每次 yield 下一个样本的索引。

> `sampler`与迭代数据集不兼容，因为这种数据集没有键或索引。



### 加载单个和批次数据

`DataLoader` 支持自动整理单个的数据样本为 batch，通过参数 `batch_size`，`drop_last` 和 `batch_sampler`。



#### 自动分批

最常见的情形，对应于拿来一个 mini-batch 的数据，将它们整理为 batched 样本的情形。

当 `batch_size`（默认为 1）不为 `None`，`dataloader` 会 yield batched 样本而非单个样本。`batch_size` 和 `drop_last` 参数用于指定 `dataloader` 如何获取数据集的键的 batch。对于映射数据集，用户也可以指定 `batch_sampler`，其每次 yield 一个键的列表。

> `batch_size`和`drop_last`参数用于从`sampler`构建`batch_sampler`。对于映射数据集，`sampler`由用户提供或者根据`shuffle`参数构造。

> 当使用多进程从迭代数据集拿数据时，`drop_last`参数丢弃每个worker的数据集副本的最后一个数量不满的batch。

根据 `sampler` yield 的索引拿到一个样本列表后，作为 `collate_fn` 参数传入的函数就用于整理样本列表为 batch。

这种情形下，从映射数据集加载就大致相当于：

```python
for indices in batch_sampler:    yield collate_fn([dataset[i] for i in indices])
```

从迭代数据集加载就大致相当于：

```python
dataset_iter = iter(dataset)for indices in batch_sampler:    yield collate_fn([next(dataset_iter) for _ in indices])
```

自定义 `collate_fn` 可以用于自定义整理过程，即填充顺序数据到 batch 的最大长度。



#### 禁用自动分批

在有些情况下，用户可能想要手动处理分批，或仅加载单个样本。例如，直接加载 batched 数据会使得花销更小（从数据库批量读取，从磁盘批量读取，读取主存的连续块等），或者 batch size 取决于数据本身，或者模型被设计为在单个样本上运行。在这些情景下，更好的做法是不使用自动分批（和 `collate_fn` 函数），而让 `dataloader` 直接返回 `dataset` 对象的成员。

当 `batch_size` 和 `batch_sampler` 都为 `None` 时，自动分批就被禁用。每一个从 `dataset` 获得的样本都由 `collate_fn` 参数传入的函数处理。

当自动分批被禁用时，默认的 `collate_fn` 函数仅将 numpy 数组转化为 PyTorch 张量，而不做其它改变。

这种情形下，从映射数据集加载就大致相当于：

```python
for index in sampler:    yield collate_fn(dataset[index])
```

从迭代数据集加载就大致相当于：

```python
for data in iter(dataset):    yield collate_fn(data)
```



#### 使用`collate_fn`函数

`collate_fn` 的作用根据启动或禁用自动分批而略有差异。





### 单进程和多进程数据加载



### num_workers





## BatchSampler

包装另一个 sampler 并 yield 一个 mini-batch 的索引。

```python
>>> from torch.utils.data import SequentialSampler>>> from torch.utils.data import BatchSampler>>> sampler = SequentialSampler(range(10))>>> sampler = BatchSampler(sampler, batch_size=3, drop_last=False)>>> for i in sampler:...   print(i)... [0, 1, 2][3, 4, 5][6, 7, 8][9]>>> sampler = SequentialSampler(range(10))>>> sampler = BatchSampler(sampler, batch_size=3, drop_last=True)>>> for i in sampler:...   print(i)... [0, 1, 2][3, 4, 5][6, 7, 8]
```

```python
>>> from torch.utils.data import RandomSampler>>> from torch.utils.data import BatchSampler>>> sampler = RandomSampler(range(10), replacement=True, num_samples=100)>>> sampler = BatchSampler(sampler, batch_size=3, drop_last=False)>>> for i in sampler:...   print(i)... [1, 4, 5][6, 8, 9]# ...[5, 7, 0][9]
```





## ChainDataset





## ConcatDataset





## DataLoader

数据加载器，其结合数据集和采样器，返回一个给定数据集上的可迭代对象。

```python
torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, collate_fn=None, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None, multiprocessing_context=None, generator=None, *, prefetch_factor=2, persistent_workers=False)# dataset        加载的数据集# batch_size     批次规模# shuffle        若为`True`,则每完成一次迭代都重新打乱数据# sampler        采样器,其定义了从数据集中采样的策略,可以是任何实现了`__len__`方法的可迭代对象.与`shuffle`互斥# batch_sampler  与`sampler`类似,但每次返回一个批次的索引.与`batch_size`,`shuffle`,`sample`,`drop_last`互斥# num_workers    用于加载数据的工作进程(子进程)数量,`0`表示在主进程中加载数据# collate_fn# pin_memory     若为`True`,则数据加载器将在返回张量之前将它们复制到# drop_last      若为`True`,则丢弃末尾的达不到批次规模的剩余样本;若为`False`,则剩余样本将组成一个较小的批次# timeout        从工作进程收集一个批次数据的超时时间# worker_init_fn# generator# prefetch_factor# persistant_workers
```

```python
>>> from torchvision import datasets, transforms>>> transform = transforms.Compose(    [transforms.ToTensor(),     transforms.Normalize((0.5), (0.5))])>>> train_set = datasets.MNIST(root='./data',                           train=True,                           download=True,                           transform=transform)>>> train_set, val_set = torch.utils.data.random_split(train_set, [48000, 12000])>>> train_loader = torch.utils.data.DataLoader(train_set,                                           batch_size=32,                                           shuffle=True)>>> type(train_loader)<class 'torch.utils.data.dataloader.DataLoader'>>>> train_loader.dataset                                          # 采样的数据集<torch.utils.data.dataset.Subset object at 0x1559ae8e0>>>> train_loader.sampler                                          # 当`shuffle=True`时创建并使用的随机采样器<torch.utils.data.sampler.RandomSampler object at 0x152229f40>>>> len(train_loader.sampler)48000>>> list(train_loader.sampler)                                    # 每次采样的顺序不同[1883, 2208, 28103, 25083, 3052, 44262, 2523, 12614, 44167, 44528, 43330, 4986, 38242, 5401, 20988, 10679, 26630, 5071, 39648, 12959, 37922, 47678, 16923, 39058, 411, 24899, 3682, 21712, 9970, 20472, 18930, 3124, 12951, ...]>>> train_loader.batch_size                                       # 批次规模32>>> len(train_loader)                                             # 数据加载器规模,即产生多少个批次1500
```





## Dataset

表示数据集的抽象类。

所有映射数据集应继承此类。所有子类应覆写 `__getitem__()` 方法，以支持由键拿到数据样本。子类也可以覆写 `__len__()` 方法，用于由于 `Sampler` 的许多实现和 `DataLoader` 的默认选项而返回数据集的大小。

> 对于映射数据集，`DataLoader` 默认构造一个产生整数索引的索引采样器。如果映射数据集的索引或键不是整数，则需要提供一个自定义采样器。

```python
# 自定义映射数据集>>> 
```



```python
# torchvision提供的MNIST数据集>>> from torchvision import datasets, transforms>>> transform = transforms.Compose(    [transforms.ToTensor(),     transforms.Normalize((0.5), (0.5))])>>> train_set = datasets.MNIST(root='./data',                           train=True,                           download=True,                           transform=transform)>>> type(train_set)<class 'torchvision.datasets.mnist.MNIST'>>>> isinstance(train_set, torch.utils.data.Dataset)            # 是映射数据集True>>> isinstance(train_set, torch.utils.data.IterableDataset)    # 而非迭代数据集False>>> train_set                                                  # 数据集概况Dataset MNIST    Number of datapoints: 60000    Root location: ./data    Split: Train    StandardTransformTransform: Compose(               ToTensor()               Normalize(mean=0.5, std=0.5)           )>>> len(train_set)                                             # 数据集大小60000>>> train_set.data                                             # 数据张量tensor([[[0, 0, 0,  ..., 0, 0, 0],         [0, 0, 0,  ..., 0, 0, 0],         [0, 0, 0,  ..., 0, 0, 0],         ...,         [0, 0, 0,  ..., 0, 0, 0],         [0, 0, 0,  ..., 0, 0, 0],         [0, 0, 0,  ..., 0, 0, 0]]], dtype=torch.uint8)>>> train_set.data.shape                                       # 数据张量形状torch.Size([60000, 28, 28])>>> train_set.targets                                          # 标签张量tensor([5, 0, 4,  ..., 5, 6, 8])>>> train_set.targets.shape                                    # 标签张量形状torch.Size([60000])>>> train_set.classes                                          # 标签类型['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four', '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']>>> train_set.train                                            # 是否为训练数据True>>> train_set.transforms                                       # 预处理步骤StandardTransformTransform: Compose(               ToTensor()               Normalize(mean=0.5, std=0.5)           )>>> train_set.root                                             # 数据集存放的路径'./data'
```









## distributed.DistributedSampler

分布式采样器，将采样限定在数据集的一个子集中。常用于分布式训练（与 `torch.nn.parallel.DistributedDataParallel` 或 `horovod.torch` 结合使用），其中每个进程传入一个 `DistributedSampler` 实例作为 `Dataloader` 的采样器，并加载一个原始数据集的一个独占的子集。

```python
torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=None, rank=None, shuffle=True, seed=0, drop_last=False)# dataset           采样的数据集# num_replicas      参与分布式训练的进程数,默认从当前进程组获取`WORLD_SIZE`# rank              当前进程的rank,默认从当前进程组获取`RANK`# shuffle           若为`True`,则采样器打乱索引的顺序# seed              当`shuffle=True`时采样器打乱使用的随机种子.此参数应在进程组的各进程中保持一致# drop_last         若为`True`,则采样器将会丢弃末尾的样本以使样本平分到各进程;若为`False`,则采样器将会#                   添加起始的样本以使样本平分到各进程
```

```python
>>> from torch.utils.data import DistributedSampler>>> sampler0 = DistributedSampler(range(10), num_replicas=3, rank=0, shuffle=False, drop_last=True)>>> sampler1 = DistributedSampler(range(10), num_replicas=3, rank=1, shuffle=False, drop_last=True)>>> sampler2 = DistributedSampler(range(10), num_replicas=3, rank=2, shuffle=False, drop_last=True)>>> list(sampler0)[0, 3, 6]               # 丢弃索引为10的样本以使样本平分到各进程>>> list(sampler1)[1, 4, 7]>>> list(sampler2)[2, 5, 8]>>> sampler0 = DistributedSampler(range(10), num_replicas=3, rank=0, shuffle=False, drop_last=False)>>> sampler1 = DistributedSampler(range(10), num_replicas=3, rank=1, shuffle=False, drop_last=False)>>> sampler2 = DistributedSampler(range(10), num_replicas=3, rank=2, shuffle=False, drop_last=False)>>> list(sampler0)[0, 3, 6, 9]>>> list(sampler1)[1, 4, 7, 0]            # 添加索引为0,1的样本以使样本平分到各进程>>> list(sampler2)[2, 5, 8, 1]>>> sampler0 = DistributedSampler(range(10), num_replicas=3, rank=0, shuffle=True, drop_last=True)>>> sampler1 = DistributedSampler(range(10), num_replicas=3, rank=1, shuffle=True, drop_last=True)>>> sampler2 = DistributedSampler(range(10), num_replicas=3, rank=2, shuffle=True, drop_last=True)>>> list(sampler0)      # 随机划分,默认随机种子为`0`因此每次划分的结果相同[4, 5, 0]>>> list(sampler1)[1, 3, 8]>>> list(sampler2)[7, 9, 6]
```



### set_epoch()

此方法在每个 epoch 开始、创建 `DataLoader` 迭代器之前调用，以使得每个 epoch 被打乱的顺序不同。

```python
>>> sampler = DistributedSampler(dataset) if is_distributed else None>>> loader = DataLoader(dataset, shuffle=(sampler is None),...                     sampler=sampler)>>> for epoch in range(start_epoch, n_epochs):...     if is_distributed:...         sampler.set_epoch(epoch)...     train(loader)
```

```python
>>> sampler0 = DistributedSampler(range(10), num_replicas=3, rank=0, shuffle=True, drop_last=True)>>> sampler1 = DistributedSampler(range(10), num_replicas=3, rank=1, shuffle=True, drop_last=True)>>> sampler2 = DistributedSampler(range(10), num_replicas=3, rank=2, shuffle=True, drop_last=True)>>> list(sampler0)[4, 5, 0]>>> list(sampler1)[1, 3, 8]>>> list(sampler2)[7, 9, 6]>>> sampler0.set_epoch(0)   # 相当于设置随机种子为`0`>>> list(sampler0)[4, 5, 0]>>> list(sampler1)[1, 3, 8]>>> list(sampler2)[7, 9, 6]>>> sampler0.set_epoch(1)>>> list(sampler0)[5, 2, 9]>>> list(sampler1)[1, 3, 8]>>> list(sampler2)[7, 9, 6]
```



## get_worker_info()





## IterableDataset

迭代数据集。

所有迭代数据集应继承此类。当数据来源于一个流时，这种形式的数据集尤为有用。

所有子类应覆写 `__iter__()` 方法，用于返回数据集的样本的一个迭代器。

当 `Dataloader` 使用迭代数据集时，`Dataloader` 的迭代器会产生数据集的每一个样本。当 `num_worker>0` 时，每一个工作进程都会有数据集对象的一份单独的副本



```python
# 自定义迭代数据集>>> class MyIterableDataset(torch.utils.data.IterableDataset):     def __init__(self, start, end):         super(MyIterableDataset).__init__()         assert end > start, "this example code only works with end > start"         self.start = start         self.end = end     def __iter__(self):         worker_info = torch.utils.data.get_worker_info()         if worker_info is None:        # 主进程读取数据,返回完整的迭代器             iter_start = self.start             iter_end = self.end         else:                          # 工作进程读取数据,划分数据集并返回相应子集的迭代器             per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))             worker_id = worker_info.id             iter_start = self.start + worker_id * per_worker             iter_end = min(iter_start + per_worker, self.end)         return iter(range(iter_start, iter_end))    # `Dataloader`调用数据集实例的`__iter__()`方法,                                                     # 使用其返回的迭代器>>> ds = MyIterableDataset(start=1, end=11)     # range(1, 11) as dataset  >>> list(torch.utils.data.DataLoader(ds, num_workers=0))   # 主进程读取数据[tensor([1]), tensor([2]), tensor([3]), tensor([4]), tensor([5]), tensor([6]), tensor([7]), tensor([8]), tensor([9]), tensor([10])]                 # 包装为`torch.tensor`>>> list(torch.utils.data.DataLoader(ds, num_workers=2))   # 2个工作进程读取数据(error)>>> list(torch.utils.data.DataLoader(ds, num_workers=8))   # 更多工作进程读取数据(error)
```







## random_split

将数据集随机划分为多个指定规模的数据集。使用 `torch.generator` 以产生可重复的结果。

```python

```





## RandomSampler

随机采样器。若 `replacement=False`，则为无放回随机采样；若 `replacement=True`，则为有放回随机采样，并且可以指定 `num_samples`（采样数）。

```python
>>> from torch.utils.data import RandomSampler>>> sampler = RandomSampler(range(5))>>> list(sampler)
```

```python
>>> sampler = RandomSampler(range(10), replacement=True, num_samples=100)>>> list(sampler)
```



## Sampler

所有采样器的基类。

每个采样器子类必须提供一个 `__iter__()` 方法，用于迭代数据集中所有样本的索引，和一个 `__len__()` 方法，用于返回实例化的迭代器的长度。



## SequentialSampler

顺序采样器，并且总是以相同的顺序。

```python
>>> from torch.utils.data import SequentialSampler>>> sampler = SequentialSampler(range(5))>>> list(sampler)[0, 1, 2, 3, 4]
```



## Subset

数据集的指定索引的样本构成的子集。

```python
torch.utils.data.Subset(dataset, indices)# dataset      数据集# indices      指定索引
```





# torch.utils.tensorboard

`torch.utils.tensorboard` 模块用于记录 PyTorch 模型和指标到本地目录下，以供 TensorBoard 进行可视化。此模块支持标量（SCALAR）、图像（IMAGE）、直方图（HISTOGRAM）、图（GRAPH）和投影（PROJECTOR）等全部功能。

`SummaryWriter` 类是记录模型数据的主要入口，例如：

```python
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

# Writer will output to ./runs/ directory by default
writer = SummaryWriter()

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = datasets.MNIST('mnist_train', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
model = torchvision.models.resnet50(False)
# Have ResNet model take in grayscale rather than RGB
model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
images, labels = next(iter(trainloader))

grid = torchvision.utils.make_grid(images)
writer.add_image('images', grid, 0)
writer.add_graph(model, images)
writer.close()
```

运行 TensorBoard 以展示这些数据：

```shell
tensorboard --logdir=runs
```



## SummaryWriter

`SummaryWriter` 类提供了用于在指定目录下创建日志文件并写入数据的高级 API。日志文件更新以异步的方式进行，这表示训练进程可以在训练循环中直接调用方法写入数据而不会造成训练速度的减慢。

```python
torch.utils.tensorboard.writer.SummaryWriter(log_dir=None, comment='', purge_step=None, max_queue=10, flush_secs=120, filename_suffix='')
# log_dir      保存目录的路径,默认为'./runs/CURRENT_DATETIME_HOSTNAME'.请使用层级目录结构以能够更容易地
#              在多次运行之间进行比较
# comment      为默认的`log_dir`添加的后缀.若指定了`log_dir`则此参数无效
# purge_step   
# max_queue    挂起的数据写入的队列规模,达到此规模后再次调用`add`类方法将强制全部写入磁盘.
# flush_secs   每`flush_secs`秒将挂起的数据写入全部写入磁盘
# filename_suffix   为`log_dir`目录下所有日志文件的文件名添加的后缀
```

```python
from torch.utils.tensorboard import SummaryWriter

# 使用默认保存路径
writer = SummaryWriter()
# folder location: runs/May04_22-14-54_s-MacBook-Pro.local/

# 指定保存路径
writer = SummaryWriter("runs/exp1")
# folder location: runs/exp1

# 为默认保存路径添加后缀
writer = SummaryWriter(comment="LR_0.1_BATCH_16")
# folder location: runs/May04_22-14-54_s-MacBook-Pro.localLR_0.1_BATCH_16/
```



### add_scalar()

添加标量数据。

```python
add_scalar(tag: str, scalar_value: float, global_step: int = None, walltime: float = None, new_style: bool = False)
# tag           数据的标签
# scalar_value  标量数据的值
# global_step   当前的全局步数
# walltime      重载默认的真实经过时间(`time.time()`)
# new_style
```

```python
from torch.utils.tensorboard import SummaryWriter

with SummaryWriter() as w:
    x = range(100)
    for i in x:
        w.add_scalar('y=2x', i * 2, i)
        time.sleep(random.uniform(1, 2))
```

![](https://i.loli.net/2021/06/23/Wsg31JEZYM8HuNw.png)

```python
from torch.utils.tensorboard import SummaryWriter
import numpy as np

with SummaryWriter() as w:
    for i in range(100):
        w.add_scalar('Loss/train', np.random.random(), i)      # 层级标签用于TensorBoard将数据分组
        w.add_scalar('Loss/test', np.random.random(), i)
        w.add_scalar('Accuracy/train', np.random.random(), i)
        w.add_scalar('Accuracy/test', np.random.random(), i)
```

![](https://i.loli.net/2021/06/24/wYDNcyXJjeo3CLn.png)



### add_scalars()

添加一组标量数据，绘制在同一幅图上。

```python
add_scalars(main_tag: str, tag_scalar_dict: dict, global_step: int = None, walltime: float = None)
# main_tag          一组数据的标签
# tag_scalar_value  标量数据的名称到相应值的字典
# global_step       当前的全局步数
# walltime          重载默认的真实经过时间(`time.time()`)
```

```python
import numpy as np
from torch.utils.tensorboard import SummaryWriter

with SummaryWriter() as w:
    r = 5
    for i in range(100):
        w.add_scalars('run_14h', {'xsinx':i*np.sin(i/r),
                                  'xcosx':i*np.cos(i/r),
                                  'tanx': np.tan(i/r)}, i)
# This call adds three values to the same scalar plot with the tag
# 'run_14h' in TensorBoard's scalar section.
```

![](https://i.loli.net/2021/06/24/wuHtp7NFABfiJX3.png)



### add_histogram()

添加直方图，即特定的统计分布数据。

```python
add_histogram(tag: str, values, global_step: int = None, bins: str = 'tensorflow', walltime: float = None, max_bins=None)
# tag           数据的标签
# values        统计分布数据,是`torch.Tensor`或`numpy.ndarray`类型
# global_step   当前的全局步数
# bins
# walltime      重载默认的真实经过时间(`time.time()`)
# max_bins
```

```python
import numpy as np
from torch.utils.tensorboard import SummaryWriter

with SummaryWriter() as w:
    for i in range(10):
        x = np.random.randn(1000)
        w.add_histogram('distribution centers', x + i, i)
```

![](https://i.loli.net/2021/06/24/WmyORqGBrlSDvns.png)

![](https://i.loli.net/2021/06/24/DKRyk6OZ1Ivqjp4.png)



### add_image()

添加图像。

```python
add_image(tag: str, img_tensor, global_step: int = None, walltime: float = None, dataformats: str = 'CHW')
# tag           数据的标签
# img_tensor    图像张量,是`torch.Tensor`或`numpy.ndarray`类型
# global_step   当前的全局步数,默认为0
# walltime      重载默认的真实经过时间(`time.time()`)
# dataformats   数据格式,例如'CHW'表示`img_tensor`的3个维度分别为通道,高,宽
```

```python
import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor()])

trainset = torchvision.datasets.MNIST('./data',
                                      download=True,
                                      train=True,
                                      transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=32)

images, labels = iter(trainloader).next()
img_grid = torchvision.utils.make_grid(images)

with SummaryWriter() as w:
    w.add_image('mnist_images', img_grid)
```

![](https://i.loli.net/2021/06/24/ysXNFqGpBWTthHQ.png)



### add_images()

添加一组图像。

```python
add_images(tag: str, img_tensor, global_step: int = None, walltime: float = None, dataformats: str = 'NCHW')
# tag           数据的标签
# img_tensor    图像张量,是`torch.Tensor`或`numpy.ndarray`类型
# global_step   当前的全局步数,默认为0
# walltime      重载默认的真实经过时间(`time.time()`)
# dataformats   数据格式,例如'NCHW'表示`img_tensor`的4个维度分别为批次索引,通道,高,宽
```

```python
import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor()])

trainset = torchvision.datasets.MNIST('./data',
                                      download=True,
                                      train=True,
                                      transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=32)

images, labels = iter(trainloader).next()

with SummaryWriter() as w:
    w.add_images('mnist_images', images)
```

![](https://i.loli.net/2021/06/24/5IcGTEBPWdJbwxZ.png)



### add_figure()

解析 `matplotlib.pyplot.figure` 实例为图像并添加。



### add_video()

添加视频。



### add_audio()

添加音频。



### add_text()

添加文本。

```python
add_text(tag: str, text_string: str, global_step: int = None, walltime: float = None)
# tag           数据的标签
# text_string   文本字符串
# global_step   当前的全局步数,默认为0
# walltime      重载默认的真实经过时间(`time.time()`)
```

```python
from torch.utils.tensorboard import SummaryWriter

with SummaryWriter() as w:
    w.add_text('note', 'abcabcabcabc')
```

![](https://i.loli.net/2021/06/24/D6Jn9iQOZmIotyH.png)



### add_graph()

添加模型的结构图。

```python
add_graph(model: torch.nn.Module, input_to_model=None, verbose: bool = False)
# model           PyTorch模型实例
# input_to_model  模型的任意合法输入
# verbose         若为`True`,在命令行中打印图结构
```

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms


transform = transforms.Compose(
    [transforms.ToTensor()])

trainset = torchvision.datasets.MNIST('./data',
                                      download=True,
                                      train=True,
                                      transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=32)

images, labels = iter(trainloader).next()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dense1 = nn.Linear(576, 64)
        self.dense2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.dense1(x))
        output = self.dense2(x)

        return output

model = Net()

with SummaryWriter() as w:
    w.add_graph(model, images)
```

![](https://i.loli.net/2021/06/24/z5jAu6O9E2dZFp8.png)



### add_embedding()

添加嵌入投影数据。

```python
add_embedding(mat, metadata: list = None, label_img: torch.Tensor = None, global_step: int = None, tag: str = 'default', metadata_header=None)
# mat           所有数据点(词)的嵌入向量组成的二维张量,是`torch.Tensor`或`numpy.ndarray`类型
# metadata      所有数据点(词)的名称
# label_img     所有数据点(词)对应的图像组成的张量
# global_step   当前的全局步数,默认为0
# tag           嵌入的标签
```

```python
# bug exists
import keyword

import torch
from torch.utils.tensorboard import SummaryWriter

meta = []
while len(meta) < 100:
    meta = meta + keyword.kwlist # get some strings
meta = meta[:100]

for i, v in enumerate(meta):
    meta[i] = v + str(i)

label_img = torch.rand(100, 3, 10, 32)
for i in range(100):
    label_img[i] *= i / 100.0

with SummaryWriter() as w:
    w.add_embedding(torch.randn(100, 5), metadata=meta, label_img=label_img)
```





### add_pr_curve()



### add_mesh()



### add_hparams()

添加一组超参数和指标，用于在 TensorBoard 中进行比较。

```python
add_hparams(hparam_dict: dict, metric_dict: dict, hparam_domain_discrete: dict = None, run_name: str = None)
# hparam_dict      超参数的名称到相应值的字典
# metric_dict      指标的名称到相应值的字典.注意此字典会同时添加到SCALARS和HPARAMS面板中
# hparam_domain_discrete   定义超参数可取的全部离散值
# run_name         当次运行的名称,默认为当前的时间戳
```

```python
from torch.utils.tensorboard import SummaryWriter

with SummaryWriter() as w:
    for i in range(5):
        w.add_hparams({'lr': 0.1*i, 'bsize': i},
                      {'metric/accuracy': 10*i, 'metric/loss': 10*i})
```

![](https://i.loli.net/2021/06/24/2Iyv8BlpZLhgi45.png)

![](https://i.loli.net/2021/06/24/osPzdgOGyARZUpW.png)



### close()

关闭流。



### flush()

将挂起的数据写入全部写入磁盘。

