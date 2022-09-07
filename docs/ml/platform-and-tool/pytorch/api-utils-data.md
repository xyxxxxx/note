# torch.utils.data

## DataLoader详解

PyTorch 数据加载功能的核心是 `torch.utils.data.Dataloader` 类，其代表数据集的一个 Python 可迭代对象，并支持：

+ 映射风格和可迭代对象风格的数据集
+ 自定义数据加载顺序
+ 自动分批
+ 单进程和多进程数据加载
+ 自动内存锁页

这些选项通过 `DataLoader` 的初始化参数进行配置，其具有签名：

```python
DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None, *, prefetch_factor=2,
           persistent_workers=False)
```

下面各节将详细介绍这些选项的效果和使用方法。

### 数据集类型

`dataset` 是 `DataLoader` 最重要的初始化参数，将从其表示的数据集对象加载数据。PyTorch 支持两种类型的数据集：

* **映射数据集**：实现了 `__getitem__()` 和 `__len()__` 方法，表示一个从索引/键到数据样本的映射。例如访问 `dataset[idx]` 时，会读取第 `idx` 个图像和相应的标签。参见 [`Dataset`](#Dataset)。
* **迭代数据集**：`IterableDataset` 的子类的实例，实现了 `__iter__()` 方法，表示数据样本的一个可迭代对象。此类数据集特别适用于随机读取非常昂贵的情形（如使用磁盘）。例如调用 `iter(dataset)` 时，会返回一个来自数据库、远程服务器甚至实时生成的日志的数据读取流。参见 [`IterableDataset`](#IterableDataset)。

### 加载顺序和采样器

对于迭代数据集，数据加载的顺序完全由用户定义的可迭代对象控制。这使得区块读取和动态批次规模的实现更加简单。

对于映射数据集，`torch.utils.data.Sampler` 类用于指定数据加载使用的索引/键顺序，其代表数据集的索引/键的可迭代对象。例如在常规的 SGD 中，一个 `Sample` 实例可以随机产生索引的一个排列，每次产出一个索引，或每次产出多个索引以实现小批次 SGD。

一个顺序或乱序的采样器基于 `DataLoader` 的 `shuffle` 参数构造。或者，用户也可以为 `sample` 参数传入一个自定义 `Sampler` 实例，每次产出下一个样本的索引/键。

> 采样器与迭代数据集不兼容，因为这种数据集没有键或索引。

### 加载单个和批次数据

`dataloader` 支持自动整理单个数据样本为批次，通过指定参数 `batch_size`、`drop_last` 和 `batch_sampler`。

#### 自动分批（默认）

最常见的情形，对应于拿一个批次的样本，并将它们整理为一个批次的数据（即第一个维度代表批次的张量）。

当 `batch_size`（默认为 1）不为 `None` 时，`dataloader` 会产出分批的样本而不是单个样本。`batch_size` 和 `drop_last` 参数用于指定 `dataloader` 如何获得数据集的索引/键的批次。对于映射数据集，用户也可以指定 `batch_sampler`，其每次产出一个索引/键的列表。

> `batch_size` 和 `drop_last` 参数用于从 `sampler` 构造一个 `batch_sampler`。对于映射数据集，`sampler` 由用户提供或者根据 `shuffle` 参数构造。对于迭代数据集，`sampler` 就是数据集迭代的顺序。

> 当使用多进程从迭代数据集拿数据时，`drop_last` 参数丢弃每个工作器的数据集副本的最后一个不完整批次。

使用 `sampler` 产出的键/索引拿到一个样本列表后，作为 `collate_fn` 参数传入的函数就用于整理样本列表为批次。

这种情形下，从映射数据集加载就大致相当于：

```python
for indices in batch_sampler:
    yield collate_fn([dataset[i] for i in indices])
```

从迭代数据集加载就大致相当于：

```python
dataset_iter = iter(dataset)
for indices in batch_sampler:
    yield collate_fn([next(dataset_iter) for _ in indices])
```

自定义 `collate_fn` 可以用于自定义整理过程，例如填充顺序数据到批次的最大长度。

#### 禁用自动分批

在有些情况下，用户可能想要手动处理分批，或仅加载单个样本。例如，直接加载分批数据会使得花销更小（从数据库批量读取，从磁盘批量读取，读取内存的连续块等），或者批次规模取决于数据本身，或者模型被设计为在单个样本上运行。在这些情景下，更好的做法是不使用自动分批（和 `collate_fn` 函数），而让 `dataloader` 直接返回 `dataset` 的每个样本。

当 `batch_size` 和 `batch_sampler` 都为 `None` 时，自动分批就被禁用。每一个从 `dataset` 获得的样本都由 `collate_fn` 函数处理。

当自动分批被禁用时，默认的 `collate_fn` 函数仅将 NumPy 数组转换为 PyTorch 张量，而不做其它改变。

这种情形下，从映射数据集加载就大致相当于：

```python
for index in sampler:
    yield collate_fn(dataset[index])
```

从迭代数据集加载就大致相当于：

```python
for data in iter(dataset):
    yield collate_fn(data)
```

#### 使用 `collate_fn` 函数

`collate_fn` 函数的使用根据自动分批是否启用而略有差异。

当自动分批禁用时，`collate_fn` 对每个单个数据样本调用，并且输入样本产出自 `dataloader` 迭代器。在这种情况下，默认的 `collate_fn` 仅将 NumPy 数组转换为 PyTorch 张量。

当自动分批启用时，`collate_fn` 每次对一个数据样本列表调用，整理产出自 `dataloader` 迭代器的输入样本为一个批次。在这种情况下，默认的 `collate_fn` 的行为描述如下：

例如，若每个数据样本由一个三通道图像和一个整数类别标签组成，即数据集的每个元素返回元组 `(image, class_index)`，默认的 `collate_fn` 整理这样的元组的列表为分批的图像张量和分批的类别标签张量组成的单个元组。特别地，默认的 `collate_fn` 有下列特性：

+ 总是在起始位置添加一个新的维度作为批次维度。
+ 自动将 NumPy 数组和 Python 数值转换为 PyTorch 张量。
+ 保留数据结构，例如，若每个样本是一个字典，那么输出有相同的键但值为分批的数据张量的字典。

用户可以使用自定义的 `collate_fn` 以实现自定义分批，例如沿第一个维度以外的维度整理，填充变长序列，或为自定义数据类型添加支持。

### 单进程和多进程数据加载

`dataloader` 默认使用单进程数据加载。

在 Python 进程中，全局解释器锁（GIL）阻止了真正的线程间并行。为了防止计算代码阻塞在数据加载上，PyTorch 转变为执行多进程数据加载，只需要简单地设置 `num_workers` 参数为正整数。

#### 单进程数据加载

在此模式下，数据加载在 `DataLoader` 初始化的同一进程中完成，因此数据加载可能阻塞计算。但当用于进程间共享数据的资源（例如共享内存或文件描述符）有限，或当整个数据集小到可以整个加载进内存时，此模式可能是更好的选择。此外，单进程加载经常显示更多可读的错误轨迹因而有助于调试。

#### 多进程数据加载

设置 `num_workers` 参数为正整数将开启有指定数量的加载工作进程的多进程数据加载。

在此模式下，每当创建 `dataloader` 的一个迭代器，`num_workers` 数量的工作进程被创建。这时，`dataset`、`collate_fn` 和 `worker_init_fn` 被传递到每个工作进程，并用于初始化和拿取数据。这意味着数据集访问以及其内部 IO、变换（包括 `collate_fn`）都运行在工作器进程上。

`torch.utils.data.get_worker_info()` 在工作进程中返回的各种有用的信息（包括工作器 ID、数据集副本、初始种子等等），而在主进程中返回 `None`。用户可以在数据集代码中使用此函数和 `worker_init_fn` 来分别配置每个数据集副本，并决定代码是否运行在某个工作进程上。例如，这可以用于分割数据集。

对于映射数据集，主进程使用 `sampler` 生成索引并将索引分派到各个工作进程，因此任何的随机打乱都在主进程中完成。

对于迭代数据集，由于每个工作进程都有 `dataset` 的一个副本，朴素的多进程加载往往会造成重复的数据。用户可以使用 `torch.utils.data.get_worker_info()` 和 `worker_init_fn` 来分别配置每一个数据集副本。由于类似的原因，在多进程加载中，`drop_last` 参数丢弃每一个工作进程的迭代数据集副本的最后一个不完整批次。

当迭代完成时，或当迭代器被垃圾回收时工作进程被关闭。

> 通常不推荐在多进程加载中返回 CUDA 张量，由于多进程中使用和分享 CUDA 的许多细节问题。作为替代，我们推荐使用自动内存锁定（即设置 `pin_memory=True`），它能够使数据更加快速地传输到 GPU。

> Python 多进程在不同平台上启动子进程的行为略有区别，为兼容所有平台，使用多进程数据加载时应：
>
> + 将脚本的大部分代码包装到 `if __name__ == '__main__':` 块中以防止其在工作进程中再次运行。数据集和 `dataloader` 的创建逻辑也可以放到这里。
> + 确保任何自定义的 `collate_fn`、`worker_init_fn` 或 `dataset` 声明在全局作用域中，这保证了它们可以被工作进程访问。

> 默认情况下，每个工作进程的随机数种子被设置为 `base_seed + worker_id`，其中 `base_seed` 是由主进程使用其随机数生成器或指定的 `generator` 生成的 `long` 类型整数。然而，其它库的随机种子可能在初始化工作进程时被复制，从而导致各工作进程返回相同的随机数。
>
> 在 `worker_init_fn` 中，你可以通过 `torch.utils.data.get_worker_info().seed` 或 `torch.initial_seed()` 访问每个工作进程的 PyTorch 随机种子，或在数据加载前为其它库设置种子。

### 内存锁页

对于数据加载，为 `dataloader` 设置 `pin_memory=True` 会自动将拿取的数据张量放置在锁页内存中，因此能够使数据更加快速地传输到 GPU。

默认的内存锁页逻辑仅识别张量和包含张量的映射和可迭代对象。默认情况下，若内存锁页逻辑看到一个自定义类型的批次，或批次的每个元素都是自定义类型（会在 `collate_fn` 返回自定义类型结果的情况下发生），则内存锁页逻辑不会识别它们，并返回没有内存锁页的该批次。若要为自定义批次或数据类型启用内存锁页，就需要为该类型定义 `pin_memory()` 方法。例如：

```python
class SimpleCustomBatch:
    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.inp = torch.stack(transposed_data[0], 0)
        self.tgt = torch.stack(transposed_data[1], 0)

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.inp = self.inp.pin_memory()
        self.tgt = self.tgt.pin_memory()
        return self

def collate_wrapper(batch):
    return SimpleCustomBatch(batch)

inps = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
tgts = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
dataset = TensorDataset(inps, tgts)

loader = DataLoader(dataset, batch_size=2, collate_fn=collate_wrapper,
                    pin_memory=True)

for batch_ndx, sample in enumerate(loader):
    print(sample.inp.is_pinned())
    print(sample.tgt.is_pinned())
```

## BatchSampler

包装另一个 sampler 并产出一个 mini-batch 的索引。

```python
>>> from torch.utils.data import SequentialSampler
>>> from torch.utils.data import BatchSampler
>>> sampler = SequentialSampler(range(10))
>>> sampler = BatchSampler(sampler, batch_size=3, drop_last=False)
>>> for i in sampler:
...   print(i)
...
[0, 1, 2][3, 4, 5][6, 7, 8][9]
>>> sampler = SequentialSampler(range(10))
>>> sampler = BatchSampler(sampler, batch_size=3, drop_last=True)
>>> for i in sampler:
...   print(i)
...
[0, 1, 2][3, 4, 5][6, 7, 8]
```

```python
>>> from torch.utils.data import RandomSampler
>>> from torch.utils.data import BatchSampler
>>> sampler = RandomSampler(range(10), replacement=True, num_samples=100)
>>> sampler = BatchSampler(sampler, batch_size=3, drop_last=False)
>>> for i in sampler:
...   print(i)
...
[1, 4, 5][6, 8, 9]       # ...[5, 7, 0][9]
```

## ChainDataset

## ConcatDataset

## DataLoader

数据加载器，其结合数据集和采样器，提供给定数据集的一个可迭代对象。

```python
class torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, collate_fn=None, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None, multiprocessing_context=None, generator=None, *, prefetch_factor=2, persistent_workers=False)
# dataset        加载的数据集
# batch_size     批次规模,即每个批次加载多少个样本
# shuffle        若为True,则每完成一次迭代都重新打乱数据
# sampler        采样器,其定义了从数据集中采样的策略,可以是任何实现了`__len__`方法的可迭代对象.与`shuffle`互斥
# batch_sampler  与`sampler`类似,但每次返回一个批次的索引.与`batch_size`,`shuffle`,`sample`,`drop_last`互斥
# num_workers    用于加载数据的工作进程数量,`0`表示在主进程中加载数据
# collate_fn     将样本列表合并为一个批次的张量的函数.当从映射数据集分批加载数据时调用
# pin_memory     若为True,则数据加载器将在返回张量之前将它们复制到CUDA锁页内存中
# drop_last      若为True,则丢弃末尾的达不到批次规模的剩余样本;若为False,则剩余样本将组成一个较小的批次
# timeout        从工作进程收集一个批次数据的超时时间
# worker_init_fn    工作进程初始化函数,其接收一个索引参数,在每个工作进程上调用(设置随机种子之后,加载数据之前)
# generator         随机数生成器,被`RandomSampler`用于生成随机索引,或在多进程加载数据时用于为各工作进程生成`base_seed`
# prefetch_factor   每个工作进程预先加载的样本数量
# persistant_workers   若为True,则数据集被使用过一次之后数据加载器不会关闭工作进程,即保持工作进程存活
```

> 若多进程使用 `spawn` 启动方法，则 `worker_init_fn` 不能是一个不可序列化对象，例如匿名函数。

> `len(dataloader)` 的结果取决于使用的采样器的长度。当 `dataset` 是 `IterableDataset` 实例时，返回的是基于 `len(dataset) / batch_size` 的估计值，其中根据 `drop_last` 进行合适的舍入。这代表了 PyTorch 所能作出的最佳估计，因为 PyTorch 信任用户能够正确处理多进程加载以避免重复数据。
>
> 尽管如此，如果多个工作进程的分割结果都有不完整的最后批次，此估计仍可能是不准确的，因为：若设置 `drop_last=False`，则一个完整批次可能分成多个工作进程中的多个不完整批次；若设置 `drop_last=True`，则一个完整批次可能在多个工作进程中被丢弃。遗憾的是，PyTorch 通常无法检测到这些情形。

```python
# 完整的使用示例
>>> from torchvision import datasets, transforms
>>> transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5), (0.5))])
>>> train_set = datasets.MNIST(root='./data',
                               train=True,
                               download=True,
                               transform=transform)
>>> train_set, val_set = torch.utils.data.random_split(train_set, [48000, 12000])
>>> train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=32,
                                               shuffle=True)
>>> type(train_loader)
<class 'torch.utils.data.dataloader.DataLoader'>
>>> train_loader.dataset               # 采样的数据集
<torch.utils.data.dataset.Subset object at 0x1559ae8e0>
>>> train_loader.sampler               # 当`shuffle=True`时创建并使用的随机采样器
<torch.utils.data.sampler.RandomSampler object at 0x152229f40>
>>> len(train_loader.sampler)
48000
>>> list(train_loader.sampler)         # 每次采样的顺序不同
[1883, 2208, 28103, 25083, 3052, 44262, 2523, 12614, 44167, 44528, 43330, 4986, 38242, 5401, 20988, 10679, 26630, 5071, 39648, 12959, 37922, 47678, 16923, 39058, 411, 24899, 3682, 21712, 9970, 20472, 18930, 3124, 12951, ...]
>>> train_loader.batch_size            # 批次规模
32
>>> len(train_loader)                  # 数据加载器规模,即产生多少个批次
1500
```

```python
>>> train_set = 
```

## Dataset

表示数据集的抽象类。

所有映射数据集应继承此类。所有子类应覆写 `__getitem__()` 方法，以支持由键拿到数据样本。子类也可以覆写 `__len__()` 方法，`Sampler` 的许多实现和 `DataLoader` 的默认选项都需要此方法来返回数据集的大小。

> 对于映射数据集，`DataLoader` 默认构造一个产出整数索引的索引采样器。如果映射数据集的索引或键不是整数，则需要提供一个自定义采样器。

```python
# 自定义映射数据集
>>> 
```

```python
# torchvision提供的MNIST数据集
>>> from torchvision import datasets, transforms
>>> transform = transforms.Compose(
        [transforms.ToTensor(), 
         transforms.Normalize((0.5), (0.5))])
>>> train_set = datasets.MNIST(root='./data',
                               train=True,
                               download=True,
                               transform=transform)
>>> type(train_set)
<class 'torchvision.datasets.mnist.MNIST'>
>>> isinstance(train_set, torch.utils.data.Dataset)            # 是映射数据集
True
>>> isinstance(train_set, torch.utils.data.IterableDataset)    # 而非迭代数据集
False
>>> train_set                                                  # 数据集概况
Dataset MNIST
    Number of datapoints: 60000
    Root location: ./data
    Split: Train
    StandardTransform
Transform: Compose(
               ToTensor()
               Normalize(mean=0.5, std=0.5)
           )
>>> train_set[0]                                               # 循秩访问样本
(tensor([[[-1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
          -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
          -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
          -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000],
         ...
         [-1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
          -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
          -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
          -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000]]]), 5)
>>> len(train_set)                                             # 数据集大小
60000
>>> train_set.data                                             # (原始)数据张量
tensor([[[0, 0, 0,  ..., 0, 0, 0],
         [0, 0, 0,  ..., 0, 0, 0],
         [0, 0, 0,  ..., 0, 0, 0],
         ...,
         [0, 0, 0,  ..., 0, 0, 0],
         [0, 0, 0,  ..., 0, 0, 0],
         [0, 0, 0,  ..., 0, 0, 0]]], dtype=torch.uint8)
>>> train_set.data.shape                                       # 数据张量形状
torch.Size([60000, 28, 28])
>>> train_set.targets                                          # 标签张量
tensor([5, 0, 4,  ..., 5, 6, 8])
>>> train_set.targets.shape                                    # 标签张量形状
torch.Size([60000])
>>> train_set.classes                                          # 标签类型
['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four', '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']
>>> train_set.train                                            # 是否为训练数据
True
>>> train_set.transforms                                       # 预处理步骤
StandardTransform
Transform: Compose(
               ToTensor()
               Normalize(mean=0.5, std=0.5)
           )
>>> train_set.root                                             # 数据集存放的路径
'./data'
```

## distributed.DistributedSampler

分布式采样器，将采样限定在数据集的一个子集中。常用于分布式训练（与 `torch.nn.parallel.DistributedDataParallel` 或 `horovod.torch` 结合使用），其中每个进程传入一个 `DistributedSampler` 实例作为 `Dataloader` 的采样器，并加载一个原始数据集的一个独占的子集。

```python
torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=None, rank=None, shuffle=True, seed=0, drop_last=False)
# dataset           采样的数据集
# num_replicas      参与分布式训练的进程数,默认从当前进程组获取`WORLD_SIZE`
# rank              当前进程的rank,默认从当前进程组获取`RANK`
# shuffle           若为True,则采样器打乱索引的顺序
# seed              当`shuffle=True`时采样器打乱使用的随机种子.此参数应在进程组的各进程中保持一致
# drop_last         若为True,则采样器将会丢弃末尾的样本以使样本平分到各进程;若为False,则采样器将会
#                   添加起始的样本以使样本平分到各进程
```

```python
>>> from torch.utils.data import DistributedSampler
>>> sampler0 = DistributedSampler(range(10), num_replicas=3, rank=0, shuffle=False, drop_last=True)
>>> sampler1 = DistributedSampler(range(10), num_replicas=3, rank=1, shuffle=False, drop_last=True)
>>> sampler2 = DistributedSampler(range(10), num_replicas=3, rank=2, shuffle=False, drop_last=True)
>>> list(sampler0)[0, 3, 6]               # 丢弃索引为10的样本以使样本平分到各进程
>>> list(sampler1)[1, 4, 7]
>>> list(sampler2)[2, 5, 8]
>>> sampler0 = DistributedSampler(range(10), num_replicas=3, rank=0, shuffle=False, drop_last=False)
>>> sampler1 = DistributedSampler(range(10), num_replicas=3, rank=1, shuffle=False, drop_last=False)
>>> sampler2 = DistributedSampler(range(10), num_replicas=3, rank=2, shuffle=False, drop_last=False)
>>> list(sampler0)[0, 3, 6, 9]
>>> list(sampler1)[1, 4, 7, 0]            # 添加索引为0,1的样本以使样本平分到各进程
>>> list(sampler2)[2, 5, 8, 1]
>>> sampler0 = DistributedSampler(range(10), num_replicas=3, rank=0, shuffle=True, drop_last=True)
>>> sampler1 = DistributedSampler(range(10), num_replicas=3, rank=1, shuffle=True, drop_last=True)
>>> sampler2 = DistributedSampler(range(10), num_replicas=3, rank=2, shuffle=True, drop_last=True)
>>> list(sampler0)                        # 随机划分,默认随机种子为`0`因此每次划分的结果相同[4, 5, 0]
>>> list(sampler1)[1, 3, 8]
>>> list(sampler2)[7, 9, 6]
```

### set_epoch()

此方法在每个 epoch 开始、创建 `DataLoader` 迭代器之前调用，以使得每个 epoch 被打乱的顺序不同。

```python
>>> sampler = DistributedSampler(dataset) if is_distributed else None
>>> loader = DataLoader(dataset, shuffle=(sampler is None),...                     sampler=sampler)
>>> for epoch in range(start_epoch, n_epochs):...     if is_distributed:...         sampler.set_epoch(epoch)...     train(loader)
```

```python
>>> sampler0 = DistributedSampler(range(10), num_replicas=3, rank=0, shuffle=True, drop_last=True)
>>> sampler1 = DistributedSampler(range(10), num_replicas=3, rank=1, shuffle=True, drop_last=True)
>>> sampler2 = DistributedSampler(range(10), num_replicas=3, rank=2, shuffle=True, drop_last=True)
>>> list(sampler0)[4, 5, 0]
>>> list(sampler1)[1, 3, 8]
>>> list(sampler2)[7, 9, 6]
>>> sampler0.set_epoch(0)   # 相当于设置随机种子为`0`
>>> list(sampler0)[4, 5, 0]
>>> list(sampler1)[1, 3, 8]
>>> list(sampler2)[7, 9, 6]
>>> sampler0.set_epoch(1)
>>> list(sampler0)[5, 2, 9]
>>> list(sampler1)[1, 3, 8]
>>> list(sampler2)[7, 9, 6]
```

## get_worker_info()

## IterableDataset

迭代数据集。

所有迭代数据集应继承此类。此种形式的数据集尤其适用于流数据。

所有子类应覆写 `__iter__()` 方法，用于返回数据集样本的一个迭代器。

当 `Dataloader` 使用迭代数据集时，`Dataloader` 的迭代器会产出数据集的每一个样本。当 `num_worker > 0` 时，每一个工作进程都会有数据集对象的一份单独的副本，因此我们经常

```python
# 自定义迭代数据集
>>> class MyIterableDataset(torch.utils.data.IterableDataset):     def __init__(self, start, end):         super(MyIterableDataset).__init__()         assert end > start, "this example code only works with end > start"         self.start = start         self.end = end     def __iter__(self):         worker_info = torch.utils.data.get_worker_info()         if worker_info is None:        # 主进程读取数据,返回完整的迭代器             iter_start = self.start             iter_end = self.end         else:                          # 工作进程读取数据,划分数据集并返回相应子集的迭代器             per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))             worker_id = worker_info.id             iter_start = self.start + worker_id * per_worker             iter_end = min(iter_start + per_worker, self.end)         return iter(range(iter_start, iter_end))    # `Dataloader`调用数据集实例的`__iter__()`方法,                                                     # 使用其返回的迭代器
>>> ds = MyIterableDataset(start=1, end=11)     # range(1, 11) as dataset  
>>> list(torch.utils.data.DataLoader(ds, num_workers=0))   # 主进程读取数据[tensor([1]), tensor([2]), tensor([3]), tensor([4]), tensor([5]), tensor([6]), tensor([7]), tensor([8]), tensor([9]), tensor([10])]                 # 包装为`torch.tensor`
>>> list(torch.utils.data.DataLoader(ds, num_workers=2))   # 2个工作进程读取数据(error)
>>> list(torch.utils.data.DataLoader(ds, num_workers=8))   # 更多工作进程读取数据(error)
```

## random_split

将数据集随机划分为多个指定规模的数据集。使用 `torch.generator` 以产生可重复的结果。

```python

```

## RandomSampler

随机采样器。若 `replacement=False`，则为无放回随机采样；若 `replacement=True`，则为有放回随机采样，并且可以指定 `num_samples`（采样数）。

```python
>>> from torch.utils.data import RandomSampler
>>> sampler = RandomSampler(range(5))
>>> list(sampler)
```

```python
>>> sampler = RandomSampler(range(10), replacement=True, num_samples=100)
>>> list(sampler)
```

## Sampler

所有采样器的基类。

每个采样器子类必须提供一个 `__iter__()` 方法，用于迭代数据集中所有样本的索引，和一个 `__len__()` 方法，用于返回实例化的迭代器的长度。

## SequentialSampler

顺序采样器，并且总是以相同的顺序。

```python
>>> from torch.utils.data import SequentialSampler
>>> sampler = SequentialSampler(range(5))
>>> list(sampler)[0, 1, 2, 3, 4]
```

## Subset

数据集的指定索引的样本构成的子集。

```python
torch.utils.data.Subset(dataset, indices)
# dataset      数据集
# indices      指定索引
```

## SubsetRandomSampler

## WeightedRandomSampler

## TensorDataset
