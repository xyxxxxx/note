# torch.distributed

`torch.distributed` 包为跨多个计算节点的多进程并行提供了 PyTorch 支持和通信原语。建立在此功能上的 `torch.nn.parallel.DistributedDataParallel` 类提供了 PyTorch 模型的同步训练的包装器，它与 `torch.multiprocessing` 提供的并行方法以及 `torch.nn.DataParallel` 的不同之处在于它支持在由网络连接的多台机器上运行，以及用户必须显式地为每个进程启动一个训练脚本的一个单独的副本。

即使是单台机器上的同步训练，`torch.nn.parallel.DistributedDataParallel` 包装器也相对于其它数据并行的方法具有优势，因为其每个进程都拥有单独的 Python 解释器，消除了 GIL 锁对于性能的限制。

## 后端

`torch.distributed` 支持三种后端：GLOO、MPI 和 NCCL，它们各自的适用条件请参考[官方文档](https://pytorch.org/docs/stable/distributed.html#backends)。

## 初始化

### init_process_group()

初始化默认的分布式进程组，同时初始化 `torch.distributed` 包。阻塞进程直到所有进程已经加入。

```python
torch.distributed.init_process_group(backend, init_method=None, timeout=datetime.timedelta(0, 1800), world_size=-1, rank=-1, store=None, group_name='', pg_options=None)
# backend      使用的后端,可以是`'mpi'`,`'gloo'`或`'nccl'`,取决于构建时的设置.如果使用NCCL后端并且一台机器上
#              有多个进程,那么每个进程必须对其使用的每个GPU有排他的访问权,否则进程间共享GPU可能会造成死锁
# init_method  指明如何初始化进程组的URL.如果`init_method`和`store`都没有指定,则默认为'env://'.与`store`互斥
# world_size   参与任务的进程数
# rank         当前进程的rank
# store        对于所有worker可见的键值存储,用于交换连接/地址信息.与`init_method`互斥
# timeout      进程组执行操作的超时时间,默认为30min,对于gloo后端适用
# group_name   进程组名称
# pg_options
```

目前支持以下三种初始化方法：

+ **TCP 初始化**

  此方法需要指定一个属于 rank 0 进程的所有进程都可以访问的网络地址，各进程的 rank，以及 `world_size`。

  ```python
  import torch.distributed as dist
  
  # Use address of one of the machines
  dist.init_process_group(backend, init_method='tcp://10.1.1.20:23456',
                          rank=args.rank, world_size=4)
  ```

+ **共享文件系统初始化**

  此方法需要指定一个对所有进程可见的共享文件系统，以及 `world_size`。URL 应以 `file://` 开头，并且包含一个到已经存在的目录下的不存在的文件的路径。

  ```python
  import torch.distributed as dist
  
  # rank should always be specified
  dist.init_process_group(backend, init_method='file:///mnt/nfs/sharedfile',
                          world_size=4, rank=args.rank)
  ```

+ **环境变量初始化**

  此方法从环境变量中读取配置，允许用户完全自定义配置信息。需要设置的变量有：

  + `MASTER_ADDR`：rank 0 进程所在节点的网络地址。
  + `MASTER_PORT`：rank 0 进程所在节点的一个空闲端口号，rank 0 进程将监听此端口并负责建立所有链接。
  + `WORLD_SIZE`：进程数，rank 0 进程据此确定要等待来自多少个进程的连接。可以设为环境变量或直接传入初始化函数。
  + `RANK`：当前进程的 rank，进程据此确定自己是否是 rank 0 进程。可以设为环境变量或直接传入初始化函数。

  此方法为默认方法。

  ```python
  os.environ['MASTER_ADDR'] = '127.0.0.1'
  os.environ['MASTER_PORT'] = '29500'
  dist.init_process_group(backend, rank=args.rank, world_size=4)
  ```

### is_available()

返回 `torch.distributed` 包是否可用。

```python
>>> torch.distributed.is_available()
True
```

### is_initialized()

返回默认进程组是否已经初始化。

### is_mpi_available()

返回 MPI 后端是否可用。

### is_nccl_available()

返回 NCCL 后端是否可用。

## 初始化后

### Backend

可用后端的枚举类，包括成员 `GlOO`, `MPI`, `NCCL`，对应的值分别为字符串 `'gloo'`, `'mpi'`, `'nccl'`（即 `Backend.GLOO == 'gloo'`）。

可以直接调用此类以解析字符串，例如 `Backend('GLOO')` 将返回 `'gloo'`。

### get_backend(), get_rank(), get_world_size()

返回指定进程组的后端、rank 和进程数，默认为当前进程组。

## 分布式键值存储

分布式键值存储用于在一个进程组的各进程之间共享数据或者初始化进程组，共有 3 种类型：`TCPStore`、`FileStore` 和 `HashStore`。

### Store

存储的基类。

#### add()

#### delete_key()

#### get()

#### num_keys()

#### wait()

### FileStore

### HashStore

### TCPStore

基于 TCP 的分布式键值存储实现。存储服务器保存数据，而存储客户端则可以通过 TCP 连接到存储服务器。只能有一个存储服务器。

```python
class torch.distributed.TCPStore(host_name: str, port: int, world_size: int = -1, is_master: bool = False, timeout: timedelta = timedelta(seconds=300), wait_for_worker: bool = True)
# host_name
# port
# world_size
# is_master
# timeout
# wait_for_worker
```

## 进程组

默认情况下集体通信操作在 world 上执行，并要求所有进程都进入该分布式函数调用。然而，更加细粒度的通信有利于一些工作负载，这时就可以使用进程组。`new_group()` 函数可以对所有进程的任意子集创建新的进程组，返回的不透明组局柄可以用作所有集体通信方法的 `group` 参数。

### new_group()

创建一个新的分布式进程组。

此函数要求所有进程都进入（调用）此函数，即使进程不会成为该进程组的成员。此外，所有进程中的 `ranks` 参数应该是相同的，包括 rank 的顺序。

```python
torch.distributed.new_group(ranks=None, timeout=datetime.timedelta(0, 1800), backend=None, pg_options=None)
# ranks       进程组成员的rank列表.若为None,则设定为所有进程的rank
# timeout     进程组执行操作的超时时间,默认为30min,对于gloo后端适用
# backend     使用的后端,可以是`gloo`和`nccl`,取决于构建时的设置.默认与world使用相同的后端
# pg_options  
```

```python
def run(rank, size):
    group = dist.new_group([0, 1])
    tensor = torch.tensor([1.])
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    print('Rank ', rank, ' has data ', tensor)
```

```
Rank  1  has data  tensor([2.])
Rank  0  has data  tensor([2.])
Rank  3  has data  tensor([1.])
Rank  2  has data  tensor([1.])
```

## 点对点通信

### send()

同步地发送一个张量。

```python
torch.distributed.send(tensor, dst, group=None, tag=0)
# tensor    发送的张量
# dst       目标rank
# group     工作的进程组.若为`None`,则设为world
# tag       用于与远程`recv`匹配的标签
```

```python
def run(rank, size):
    tensor = torch.tensor([0.])
    if rank == 0:
        tensor += 1
        # rank 0 sends the tensor to rank 1
        dist.send(tensor=tensor, dst=1)
    elif rank == 1:
        # rank 1 receives tensor from rank 0
        dist.recv(tensor=tensor, src=0)
    print('Rank ', rank, ' has data ', tensor)
```

```
Rank  2  has data  tensor([0.])
Rank  3  has data  tensor([0.])
Rank  1  has data  tensor([1.])
Rank  0  has data  tensor([1.])
```

### recv()

同步地接收一个张量。

```python
torch.distributed.recv(tensor, src=None, group=None, tag=0)
# tensor    放置接收数据的张量
# src       源rank.若为`None`,则接收来自任意进程的数据
# group     工作的进程组.若为`None`,则设为world
# tag       用于与远程`send`匹配的标签
```

### isend()

异步地发送一个张量。

`isend()` 和 `irecv()` 返回一个分布式请求对象，支持下面两个方法：

+ `is_completed()`：当操作结束时返回 `True`
+ `wait()`：阻塞进程直到操作结束。当 `wait()` 返回后，`is_completed()` 一定返回 `True`

```python
torch.distributed.isend(tensor, dst, group=None, tag=0)
# tensor    发送的张量
# dst       目标rank
# group     工作的进程组.若为`None`,则设为world
# tag       用于与远程`recv`匹配的标签
```

```python
def run(rank, size):
    tensor = torch.tensor([0.])
    if rank == 0:
        tensor += 1
        # rank 0 sends the tensor to rank 1
        req = dist.isend(tensor=tensor, dst=1)
        print('Rank 0 started sending')
    elif rank == 1:
        # rank 1 receives tensor from rank 0
        req = dist.irecv(tensor=tensor, src=0)
        print('Rank 1 started receiving')
    if rank == 0 or rank == 1:
        req.wait()
    print('Rank ', rank, ' has data ', tensor)
```

```
Rank 1 started receiving
Rank 0 started sending
Rank  3  has data  tensor([0.])
Rank  2  has data  tensor([0.])
Rank  0  has data  tensor([1.])
Rank  1  has data  tensor([1.])
```

### irecv()

异步地接收一个张量。

```python
torch.distributed.irecv(tensor, src=None, group=None, tag=0)
# tensor    放置接收数据的张量
# src       源rank.若为`None`,则接收来自任意进程的数据
# group     工作的进程组.若为`None`,则设为world
# tag       用于与远程`send`匹配的标签
```

## 集体通信

每个集体通信操作函数都支持下面两种操作类型：

+ **同步操作**：当函数返回时，相应的集体通信操作会确保已经完成。但对于 CUDA 操作则不能确保其已经完成，因为 CUDA 操作是异步的。因此对于 CPU 集体通信操作，对其返回值的操作结果会符合预期；对于 CUDA 集体通信操作，在同一个 CUDA 流上对其返回值的操作结果会符合预期；在运行在不同 CUDA 流上的情形下，用户必须自己负责同步。
+ **异步操作**：函数返回一个分布式请求对象，支持下面两个方法：
  + `is_completed()`：对于 CPU 操作，当操作结束时返回 `True`；对于 GPU 操作，当操作成功进入排进一个 CUDA 流并且输出可以在默认流上使用（而无需进一步同步）时返回 `True`。
  + `wait()`：对于 CPU 操作，阻塞进程直到操作结束；对于 GPU 操作，阻塞进程直到操作成功进入排进一个 CUDA 流并且输出可以在默认流上使用（而无需进一步同步）。

```python

```

### broadcast()

Broadcast 操作。参与到此集体通信操作的所有进程的 `tensor` 必须具有相同的形状。

```python
torch.distributed.broadcast(tensor, src, group=None, async_op=False)
# tensor    若当前进程的rank是`src`,则为发送的张量,否则为放置接收数据的张量
# src       源rank
# group     工作的进程组.若为`None`,则设为world
# async_op  是否为异步操作
```

```python
def run(rank, size):
    if rank == 0:
        tensor = torch.tensor([1.])
    else:
        tensor = torch.tensor([0.])
    dist.broadcast(tensor, 0)
    print('Rank ', rank, ' has data ', tensor)
```

```
Rank  0  has data  tensor([1.])
Rank  1  has data  tensor([1.])
Rank  2  has data  tensor([1.])
Rank  3  has data  tensor([1.])
```

### reduce()

Reduce 操作。原位操作，rank 为 `dst` 的进程的 `tensor` 将放置最终归约结果，其它进程的 `tensor` 将放置中间结果。

```python
torch.distributed.reduce(tensor, dst, op=<ReduceOp.SUM: 0>, group=None, async_op=False)
# tensor      归约的张量兼放置归约结果的张量(原位操作,rank为`dst`的进程将放置最终结果,其它进程将放置中间结果)
# dst         目标rank
# op          归约操作,是`torch.distributed.ReduceOp`枚举类的实例之一
# group       工作的进程组.若为`None`,则设为world
# async_op    是否为异步操作
```

```python
def run(rank, size):
    tensor = torch.tensor([rank], dtype=torch.float32)
    dist.reduce(tensor, 0, op=dist.ReduceOp.SUM)
    print('Rank ', rank, ' has data ', tensor)
```

```
Rank  3  has data  tensor([3.])        # 3.
Rank  2  has data  tensor([5.])        # 3. + 2.
Rank  1  has data  tensor([6.])        # 3. + 2. + 1.
Rank  0  has data  tensor([6.])        # 3. + 2. + 1. + 0.
```

### all_reduce()

All-Reduce 操作。

```python
torch.distributed.all_reduce(tensor, op=<ReduceOp.SUM: 0>, group=None, async_op=False)
# tensor      归约的张量兼放置归约结果的张量(原位操作)
# op          归约操作,是`torch.distributed.ReduceOp`枚举类的实例之一
# group       工作的进程组.若为`None`,则设为world
# async_op    是否为异步操作
```

```python
def run(rank, size):
    tensor = torch.tensor([rank], dtype=torch.float32)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print('Rank ', rank, ' has data ', tensor)
```

```
Rank  0  has data  tensor([6.])
Rank  3  has data  tensor([6.])
Rank  2  has data  tensor([6.])
Rank  1  has data  tensor([6.])
```

### gather()

Gather 操作。

```python
torch.distributed.gather(tensor, gather_list=None, dst=0, group=None, async_op=False)
# tensor       收集的张量
# gather_list  放置收集数据的张量列表,必须包含正确数量和形状的张量元素(仅限rank为`dst`的进程)
# dst          目标rank
# group        工作的进程组.若为`None`,则设为world
# async_op     是否为异步操作
```

```python
def run(rank, size):
    tensor = torch.tensor([rank], dtype=torch.float32)
    gather_list = [torch.zeros(1) for _ in range(4)] if rank == 0 else []
    dist.gather(tensor, gather_list=gather_list, dst=0)
    print('Rank ', rank, ' has data ', tensor)
    print('Rank ', rank, ' has list ', gather_list)
```

```
Rank  0  has data  tensor([0.])
Rank  1  has data  tensor([1.])
Rank  2  has data  tensor([2.])
Rank  1  has list  []
Rank  2  has list  []
Rank  3  has data  tensor([3.])
Rank  3  has list  []
Rank  0  has list  [tensor([0.]), tensor([1.]), tensor([2.]), tensor([3.])]
```

### all_gather()

All-Gather 操作。

```python
torch.distributed.all_gather(tensor_list, tensor, group=None, async_op=False)
# tensor_list  放置收集数据的张量列表,必须包含正确数量和形状的张量元素
# tensor       收集的张量
# group        工作的进程组.若为`None`,则设为world
# async_op     是否为异步操作
```

```python
def run(rank, size):
    tensor = torch.tensor([rank], dtype=torch.float32)
    tensor_list = [torch.zeros(1) for _ in range(4)]
    dist.all_gather(tensor_list, tensor)
    print('Rank ', rank, ' has data ', tensor)
    print('Rank ', rank, ' has list ', tensor_list)
```

```
Rank  0  has data  tensor([0.])
Rank  1  has data  tensor([1.])
Rank  2  has data  tensor([2.])
Rank  3  has data  tensor([3.])
Rank  0  has list  [tensor([0.]), tensor([1.]), tensor([2.]), tensor([3.])]
Rank  2  has list  [tensor([0.]), tensor([1.]), tensor([2.]), tensor([3.])]
Rank  1  has list  [tensor([0.]), tensor([1.]), tensor([2.]), tensor([3.])]
Rank  3  has list  [tensor([0.]), tensor([1.]), tensor([2.]), tensor([3.])]
```

### all_to_all()

All-to-All 操作。

```python
torch.distributed.all_to_all(output_tensor_list, input_tensor_list, group=None, async_op=False)
# output_tensor_list   放置分发数据的张量列表
# input_tensor_list    分发的张量列表
# group         工作的进程组.若为`None`,则设为world
# async_op      是否为异步操作
```

```python
def run(rank, size):
    input_tensor_list = [torch.tensor([rank * size + i], dtype=torch.float32) for i in range(size)]
    output_tensor_list = [torch.zeros(1) for _ in range(4)]
    dist.all_to_all(output_tensor_list=output_tensor_list, input_tensor_list=input_tensor_list)
    print('Rank ', rank, ' has input list ', input_tensor_list)
    print('Rank ', rank, ' has output list ', output_tensor_list)
```

```
RuntimeError: ProcessGroup does not support alltoall
```

### scatter()

Scatter 操作。

```python
torch.distributed.scatter(tensor, scatter_list=None, src=0, group=None, async_op=False)
# tensor        放置分发数据的张量
# scatter_list  分发的张量列表
# scr           源rank
# group         工作的进程组.若为`None`,则设为world
# async_op      是否为异步操作
```

```python
def run(rank, size):
    tensor = torch.zeros(1)
    scatter_list = [torch.tensor([i], dtype=torch.float32) for i in range(size)] if rank == 0 else []
    dist.scatter(tensor, scatter_list=scatter_list)
    print('Rank ', rank, ' has list ', scatter_list)
    print('Rank ', rank, ' has data ', tensor)
```

```
Rank  2  has list  []
Rank  1  has list  []
Rank  3  has list  []
Rank  3  has data  tensor([3.])
Rank  1  has data  tensor([1.])
Rank  2  has data  tensor([2.])
Rank  0  has list  [tensor([0.]), tensor([1.]), tensor([2.]), tensor([3.])]
Rank  0  has data  tensor([0.])
```

### barrier()

同步所有进程，即阻塞进入此函数的进程，直到进程组的所有进程全部进入此函数。

```python
torch.distributed.barrier(group=None, async_op=False, device_ids=None)
# group         工作的进程组.若为`None`,则设为world
# async_op      是否为异步操作
# device_ids    GPU设备的id列表,仅对NCCL后端有效
```

### ReduceOp

可用归约操作的枚举类，包括成员：`SUM`, `PRODUCT`, `MIN`, `MAX`, `BAND`, `BOR`, `BXOR`。

注意 `BAND`, `BOR`, `BXOR` 不适用于 `NCCL` 后端；`MAX`, `MIN`, `PRODUCT` 不适用于复张量。

## RPC

### init_rpc()

初始化诸如本地 RPC 代理和分布式 autograd 的 RPC 原语，这会立刻使当前进程准备好发送和接收 RPC。

```python
torch.distributed.rpc.init_rpc(name, backend=None, rank=-1, world_size=None, rpc_backend_options=None)
# name        此进程的全局唯一名称(例如`Master`,`ps`,`Worker1`等)
# backend     RPC后端实现的类型,默认为`Backend.TENSORPIPE`
# rank        此进程的全局唯一rank
# world_size  组内的工作器数量
# rpc_backend_options
```

### rpc_sync()

```python
torch.distributed.rpc.rpc_sync(to, func, args=None, kwargs=None, timeout=-1.0)
# to          目标工作器的名称/rank/`WorkerInfo`实例
# func        函数,例如Python可调用对象、Python内置函数和被注解的TorchScript函数
# args        传递给`func`的参数元组
# kwargs      传递给`func`的关键字参数字典
# timeout     此远程调用的超时时间(秒).0表示永不超时.若此参数没有提供,则使用初始化期间设定的默认值
```

进行一次阻塞的 RPC 调用以在工作器 `to` 上运行函数 `func`。RPC 消息的发送和接收相对于 Python 代码的执行是并行的。此方法是线程安全的。

```python
# on worker 0
>>> import os
>>> os.environ['MASTER_ADDR'] = '127.0.0.1'
>>> os.environ['MASTER_PORT'] = '29500'
>>> import torch
>>> import torch.distributed.rpc as rpc
>>> rpc.init_rpc("worker0", rank=0, world_size=2)
>>> ret = rpc.rpc_sync("worker1", torch.add, args=(torch.ones(2), 3))
>>> ret
tensor([4., 4.])
>>> rpc.shutdown()
```

```python
# on worker 1
>>> import os
>>> os.environ['MASTER_ADDR'] = '127.0.0.1'
>>> os.environ['MASTER_PORT'] = '29500'
>>> import torch
>>> import torch.distributed.rpc as rpc
>>> rpc.init_rpc("worker1", rank=1, world_size=2)
>>> rpc.shutdown()
```

!!! warning "警告"

    使用 `rpc_sync()`、`rpc_async()` 等 API 时，函数的参数张量和返回值张量都必须是 CPU 张量（否则当两个进程的设备列表不一致时可能会引起崩溃）。如有必要，应用可以显式地在调用进程中将张量移动到 CPU，再在被调用进程中将其移动到想要的设备中。

### rpc_async()

```python
torch.distributed.rpc.rpc_async(to, func, args=None, kwargs=None, timeout=-1.0)
# to          目标工作器的名称/rank/`WorkerInfo`实例
# func        函数,例如Python可调用对象、Python内置函数和被注解的TorchScript函数
# args        传递给`func`的参数元组
# kwargs      传递给`func`的关键字参数字典
# timeout     此远程调用的超时时间(秒).0表示永不超时.若此参数没有提供,则使用初始化期间设定的默认值
```

进行一次非阻塞的 RPC 调用以在工作器 `to` 上运行函数 `func`，并立即返回一个 `Future` 实例。RPC 消息的发送和接收相对于 Python 代码的执行是并行的。此方法是线程安全的。

```python
# on worker 0
>>> import os
>>> os.environ['MASTER_ADDR'] = '127.0.0.1'
>>> os.environ['MASTER_PORT'] = '29500'
>>> import torch
>>> import torch.distributed.rpc as rpc
>>> rpc.init_rpc("worker0", rank=0, world_size=2)
>>> fut1 = rpc.rpc_async("worker1", torch.add, args=(torch.ones(2), 3))
>>> fut2 = rpc.rpc_async("worker1", min, args=(1, 2))
>>> result = fut1.wait() + fut2.wait()
>>> result
tensor([5., 5.])
>>> rpc.shutdown()
```

```python
# on worker 1
>>> import os
>>> os.environ['MASTER_ADDR'] = '127.0.0.1'
>>> os.environ['MASTER_PORT'] = '29500'
>>> import torch
>>> import torch.distributed.rpc as rpc
>>> rpc.init_rpc("worker1", rank=1, world_size=2)
>>> rpc.shutdown()
```

!!! warning "警告"

    `rpc_async()` API 会等到要通过网络发送参数时才复制这些参数（包括其中的张量），这会由另一个线程完成，取决于 RPC 后端的类型。调用进程应确保参数张量的内容保持不变直到返回的 `Future` 实例得到返回值。

### remote()

```python
torch.distributed.rpc.remote(to, func, args=None, kwargs=None, timeout=-1.0)
# to          目标工作器的名称/rank/`WorkerInfo`实例
# func        函数,例如Python可调用对象、Python内置函数和被注解的TorchScript函数
# args        传递给`func`的参数元组
# kwargs      传递给`func`的关键字参数字典
# timeout     此远程调用的超时时间(秒).0表示永不超时.若此参数没有提供,则使用初始化期间设定的默认值
```

进行一次远程调用以在工作器 `to` 上运行函数 `func`，并立即返回一个指向返回值的 `RRef` 实例。工作器 `to` 是返回的 `RRef` 实例的所有者，而调用进程则是使用者。所有者管理 `RRef` 实例的全局引用计数，并在计数归零（即全局不再有任何对它的引用）时销毁该实例。

```python
# on worker 0
>>> import os
>>> os.environ['MASTER_ADDR'] = '127.0.0.1'
>>> os.environ['MASTER_PORT'] = '29500'
>>> import torch
>>> import torch.distributed.rpc as rpc
>>> rpc.init_rpc("worker0", rank=0, world_size=2)    # 阻塞
>>> rref1 = rpc.remote("worker1", torch.add, args=(torch.ones(2), 3))
>>> rref2 = rpc.remote("worker1", torch.add, args=(torch.ones(2), 1))
>>> x = rref1.to_here() + rref2.to_here()
>>> x
tensor([6., 6.])
>>> rpc.shutdown()    # 阻塞
```

```python
# on worker 1
>>> import os
>>> os.environ['MASTER_ADDR'] = '127.0.0.1'
>>> os.environ['MASTER_PORT'] = '29500'
>>> import torch
>>> import torch.distributed.rpc as rpc
>>> rpc.init_rpc("worker1", rank=1, world_size=2)    # 阻塞
>>> rpc.shutdown()    # 阻塞
```

!!! warning "警告"

    `remote()` API 会等到要通过网络发送参数时才复制这些参数（包括其中的张量），这会由另一个线程完成，取决于 RPC 后端的类型。调用进程应确保参数张量的内容保持不变直到返回的 `RRef` 实例被其所有者确认，可以通过 `torch.distributed.rpc.RRef.confirmed_by_owner()` API 进行检查。

### shutdown()

关闭 RPC 代理并随后销毁。这将阻止本地代理接收外部请求，并通过停止所有 RPC 线程关闭 RPC 框架。

```python
torch.distributed.rpc.shutdown(graceful=True)
# graceful       若为`True`,则会阻塞直到(1)不再有挂起的`UserRRefs`系统消息,同时删除这些消息
#                (2)所有本地和远程RPC进程到达此方法并且等待所有外部工作结束
```

```python
# on worker 0
>>> import os
>>> os.environ['MASTER_ADDR'] = '127.0.0.1'
>>> os.environ['MASTER_PORT'] = '29500'
>>> import torch
>>> import torch.distributed.rpc as rpc
>>> rpc.init_rpc("worker0", rank=0, world_size=2)
>>> ret = rpc.rpc_sync("worker1", torch.add, args=(torch.ones(2), 3))
>>> rpc.shutdown()    # 阻塞
```

```python
# on worker 1
>>> import os
>>> os.environ['MASTER_ADDR'] = '127.0.0.1'
>>> os.environ['MASTER_PORT'] = '29500'
>>> import torch
>>> import torch.distributed.rpc as rpc
>>> rpc.init_rpc("worker1", rank=1, world_size=2)
>>> rpc.shutdown()    # 阻塞
```

### WorkerInfo

包装了工作器信息的结构，其中包含工作器的名称和 ID。此类的实例不应直接构造，而应通过 `get_worker_info()` 函数获取；实例可用于传入诸如 `rpc_sync()`、`rpc_async()`、`remote()` 等函数以避免每次远程调用都要复制字符串。

### get_worker_info()

获取指定工作器的 `WorkerInfo`。使用此 `WorkerInfo` 实例以避免每次远程调用都传递昂贵的字符串。

```python
torch.distributed.rpc.get_worker_info(worker_name=None)
# worker_name    工作器的字符串名称
```

### BackendType

可用后端的枚举类。

PyTorch 内置一个 `BackendType.TENSORPIPE` 后端；可以使用 `register_backend()` 函数来注册更多的后端。

#### BackendType.TENSORPIPE

默认使用的 TensorPipe 代理利用了 TensorPipe 库，其提供了一个原生的点对点的通信原语，从根本上解决了 Gloo 的一些局限性，因而特别适用于机器学习。相比 Gloo，其优势在于它是异步的，允许大量转移操作同时进行，而不会互相阻塞或影响运行速度。进程对之间的管道只有在需要时才会打开；当一个进程故障时只有它相关的管道会被关闭，而其他管道都会照常工作。此外，TensorPipe 还支持多种传输方式（TCP、共享内存、NVLink、InfiniBand 等），能够自动检测这些方式的可用性并为每个管道选择最佳的传输方式。

TensorPipe 后端自 PyTorch v1.6 版本被引入。目前它仅支持 CPU 张量，GPU 支持将在不久之后到来。它和 Gloo 一样使用基于 TCP 的连接。它还可以自动切分大型张量，在多个套接字和线程上多路复用以达成高带宽。

### RpcBackendOptions

包装了传入到 RPC 后端的选项的抽象结构。此类的实例可被传入到 `init_rpc()` 以使用特定配置初始化 RPC，例如 RPC 超时时间和使用的 `init_method`。

#### init_method

指定如何初始化进程组的 URL，默认为 `env://`。

#### rpc_timeout

指定所有 RPC 操作的超时时间的浮点数。

### TensorPipeRpcBackendOptions

TensorPipe 代理的后端选项，继承自 `RpcBackendOptions`。

```python
torch.distributed.rpc.TensorPipeRpcBackendOptions(*, num_worker_threads=16, rpc_timeout=60.0, init_method='env://', device_maps=None, devices=None, _transports=None, _channels=None)
# num_worker_threads   TensorPipe代理执行请求时使用的线程池中线程的数量
# rpc_timeout          默认的RPC请求的超时时间(秒).如果RPC没有在这一时间范围内完成,则引发一个
#                      相应的异常.调用进程可以为`rpc_sync()`和`rpc_async()`的RPC单独重载
#                      这一超时时间
# init_method          同`init_process_group()`的`init_method`参数
# device_maps          从调用进程到被调用进程的设备放置映射.其中键是被调用工作器的名称,值是映射
#                      调用进程的设备到被调用进程的设备的字典(设备用`int`,`str`或`torch.device`
#                      表示)
# devices              RPC代理使用的所有本地CUDA设备.默认从`device_maps`初始化得到.
```

```python
>>> import os
>>> os.environ['MASTER_ADDR'] = '127.0.0.1'
>>> os.environ['MASTER_PORT'] = '29500'
>>> import torch.distributed.rpc as rpc
>>> rpc.init_rpc(
    "worker1",
    rank=0,
    world_size=2,
    rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=8,
        rpc_timeout=20    # 20 second timeout
    )
)
```

#### device_maps, devices, init_method, num_worker_threads, rpc_timeout

见 `TensorPipeRpcBackendOptions` 初始化函数。

#### set_device_map()

设定调用进程和被调用进程之间的设备映射。此函数可以被多次调用以逐渐增加设备放置的设置。

```python
set_device_map(to, device_map)
# to           被调用进程的名称
# device_map   从当前进程到被调用进程的设备放置映射
```

```python
>>> # on both workers
>>> def add(x, y):
>>>     print(x)    # `tensor([1., 1.], device='cuda:1')` on worker 1
>>>     return x + y, (x + y).to(2)
>>>
>>> # on worker 0
>>> options = TensorPipeRpcBackendOptions(
>>>     num_worker_threads=8,
>>>     device_maps={"worker1": {0: 1}}
>>>     # maps worker0's cuda:0 to worker1's cuda:1
>>> )
>>> options.set_device_map("worker1", {1: 2})
>>> # maps worker0's cuda:1 to worker1's cuda:2
>>>
>>> rpc.init_rpc(
>>>     "worker0",
>>>     rank=0,
>>>     world_size=2,
>>>     backend=rpc.BackendType.TENSORPIPE,
>>>     rpc_backend_options=options
>>> )
>>>
>>> x = torch.ones(2)
>>> rets = rpc.rpc_sync("worker1", add, args=(x.to(0), 1))
>>> # The first argument will be moved to cuda:1 on worker1. When
>>> # sending the return value back, it will follow the invert of
>>> # the device map, and hence will be moved back to cuda:0 and
>>> # cuda:1 on worker0
>>> print(rets[0])  # `tensor([2., 2.], device='cuda:0')` on worker 0
>>> print(rets[1])  # `tensor([2., 2.], device='cuda:1')` on worker 0
```

#### set_devices()

设定 TensorPipe 代理使用的本地设备。

```python
set_devices(devices)
# devices      TensorPipe 代理使用的本地设备,是`int`,`str`或`torch.device`列表
```

### RRef

!!! warning "警告"

    RRef 目前尚不支持 CUDA 张量。

一个 `RRef` 实例是对远程工作器上的一个某种类型的值的引用。这种处理方式使被引用的远程值仍位于其所有者上。RRef 可用于多机训练，通过保有对存在于其他工作器上的 `nn.Modules` 实例的引用，并在训练过程中调用适当的函数以获取或修改其参数。

#### backward()

以此 `RRef` 实例为根进行反向计算。……

```python
backward(self, dist_autograd_ctx_id=-1, retain_graph=False)
# dist_autograd_ctx_id   要获取的梯度所位于的分布式autograd上下文的id
# retain_graph           
```

#### confirmed_by_owner()

返回此 `RRef` 实例是否被其所有者确认。所有者的 `RRef` 实例总是返回 `True`，而使用者的 `RRef` 实例只有当所有者知道此实例时才返回 `True`。

#### is_owner()

返回当前进程是否是此 `RRef` 实例的所有者。

#### local_value()

若当前进程是所有者，则返回对本地值的一个引用，否则引发一个异常。

#### owner()

返回此 `RRef` 实例的所有者的 `WorkerInfo` 实例。

#### owner_name()

返回此 `RRef` 实例的所有者的名称。

#### remote()

创建一个辅助代理以简单地启动一个 `remote`，其以此 `RRef` 实例的所有者为目标工作器运行此 `RRef` 实例所引用对象的指定方法。更具体地，`rref.remote().func_name(*args, **kwargs)` 相当于：

```python
>>> def run(rref, func_name, args, kwargs):
>>>   return getattr(rref.local_value(), func_name)(*args, **kwargs)
>>>
>>> rpc.remote(rref.owner(), run, args=(rref, func_name, args, kwargs))
```

```python
>>> import torch.distributed.rpc as rpc
>>> rref = rpc.remote("worker1", torch.add, args=(torch.zeros(2, 2), 1))
>>> rref.remote().size().to_here()      # returns torch.Size([2, 2])
>>> rref.remote().view(1, 4).to_here()  # returns tensor([[1., 1., 1., 1.]])
```

#### rpc_async()

创建一个辅助代理以简单地启动一个 `rpc_async`，其以此 `RRef` 实例的所有者为目标工作器运行此 `RRef` 实例所引用对象的指定方法。更具体地，`rref.rpc_async().func_name(*args, **kwargs)` 相当于：

```python
>>> def run(rref, func_name, args, kwargs):
>>>   return getattr(rref.local_value(), func_name)(*args, **kwargs)
>>>
>>> rpc.rpc_async(rref.owner(), run, args=(rref, func_name, args, kwargs))
```

```python
>>> import torch.distributed.rpc as rpc
>>> rref = rpc.remote("worker1", torch.add, args=(torch.zeros(2, 2), 1))
>>> rref.rpc_async().size().wait()      # returns torch.Size([2, 2])
>>> rref.rpc_async().view(1, 4).wait()  # returns tensor([[1., 1., 1., 1.]])
```

#### rpc_sync()

创建一个辅助代理以简单地启动一个 `rpc_sync`，其以此 `RRef` 实例的所有者为目标工作器运行此 `RRef` 实例所引用对象的指定方法。更具体地，`rref.rpc_sync().func_name(*args, **kwargs)` 相当于：

```python
>>> def run(rref, func_name, args, kwargs):
>>>   return getattr(rref.local_value(), func_name)(*args, **kwargs)
>>>
>>> rpc.rpc_sync(rref.owner(), run, args=(rref, func_name, args, kwargs))
```

```python
>>> import torch.distributed.rpc as rpc
>>> rref = rpc.remote("worker1", torch.add, args=(torch.zeros(2, 2), 1))
>>> rref.rpc_sync().size()      # returns torch.Size([2, 2])
>>> rref.rpc_sync().view(1, 4)  # returns tensor([[1., 1., 1., 1.]])
```

#### to_here()

阻塞调用，从所有者复制此 `RRef` 实例的值到本地进程并返回该值。如果当前进程就是所有者，则返回对本地值的一个引用。

## RemoteModule

!!! warning "警告"

    RemoteModule 目前尚不支持 CUDA 张量。

`RemoteModule` 实例只能在 RPC 初始化之后创建，其将用户指定的一个模块创建在指定的远程 RPC 进程上。`RemoteModule` 实例的行为就像是常规的 `nn.Module` 实例，除了其 `forward` 方法在远程进程中执行。它负责 autograd 记录以确保反向计算过程中将梯度传播回相应的远程模块。

它基于原模块类的 `forward` 方法的签名产生了 `forward_async` 和 `forward` 方法。`forward_async` 异步运行并返回一个 `Future`。`forward_async` 和 `forward` 方法的参数与原 `forward` 方法相同。

例如，如果原模块类返回一个 `nn.Linear` 的实例，其有 `forward` 方法签名：

`def forward(input: Tensor) -> Tensor`

那么产生的 `RemoteModule` 实例将有如下的方法签名：

`def forward(input: Tensor) -> Tensor`

`def forward_async(input: Tensor) -> Future[Tensor]`

```python
class torch.distributed.nn.api.remote_module.RemoteModule(
    remote_device, module_cls, args, kwargs
)
# remote_device   模块放置在目标工作器上的设备,格式应为`"<worker_name>/<device>"`,例如
#                 `"trainer0/cpu"``"trainer0"``"ps0/cuda:0"`.设备字段默认为"cpu"
# module_cls      要远程创建的模块的类
# args            要传给`module_cls`的参数
# kwargs          要传给`module_cls`的关键字参数
```

```python
>>> # On worker 0:
>>> import torch
>>> import torch.nn as nn
>>> import torch.distributed.rpc as rpc
>>> from torch.distributed.nn.api.remote_module import RemoteModule
>>>
>>> rpc.init_rpc("worker0", rank=0, world_size=2)
>>> remote_linear_module = RemoteModule(
>>>     "worker1/cpu", nn.Linear, args=(20, 30),
>>> )
>>> input = torch.randn(128, 20)
>>> ret_future = remote_linear_module.forward_async(input)
>>> ret = ret_future.wait()
>>> rpc.shutdown()

>>> # On worker 1:
>>> import torch
>>> import torch.distributed.rpc as rpc
>>>
>>> rpc.init_rpc("worker1", rank=1, world_size=2)
>>> rpc.shutdown()
```

### get_module_rref()

返回引用远程模块的 `RRef` 实例。

### remote_parameters()

返回引用远程模块的参数的 `RRef` 实例列表。通常与 `DistributedOptimizer` 一起使用。

## 分布式 Autograd 框架

!!! warning "警告"

    分布式 autograd 目前尚不支持 CUDA 张量。

### backward()

使用给定的根开始分布式反向计算。当前实现的 [FAST 模式算法](https://pytorch.org/docs/master/rpc/distributed_autograd.html#fast-mode-algorithm)假定在同一个分布式 autograd 上下文各工作器之间发送的所有 RPC 消息都是反向计算过程中 autograd 图的一部分。

我们使用给定的根来发现 autograd 图并计算其中的依赖关系。此方法会阻塞直到整个 autograd 计算完成。

我们在每一个进程中的适当的 `torch.distributed.autograd.context` 上下文中累积梯度。调用 `torch.distributed.autograd.backward()` 时传入的 `context_id` 用于查找使用的 autograd 上下文，如果没有有效的 autograd 上下文对应于给定的 ID，则抛出一个错误。你可以使用 `get_gradients()` API 获取累积的梯度。

```python
torch.distributed.autograd.backward(context_id: int, roots: List[Tensor], retain_graph=False) -> None
# context_id     用于获取梯度的autograd上下文ID
# roots          代表autograd计算的根的张量.所有的张量都应该为标量
# retain_graph   若为False,计算图在梯度计算完成后(backward()返回后)即被释放.注意在几
#                乎所有情形下将其设为True都是不必要的,因为总有更好的解决方法
```

```python
>>> import torch.distributed.autograd as dist_autograd
>>> with dist_autograd.context() as context_id:
>>>   pred = model.forward()             # 在远程进程中
>>>   loss = loss_func(pred, loss)
>>>   dist_autograd.backward(context_id, loss)
```

### context

使用分布式 autograd 时用于包装前向和反向计算的上下文对象。`with` 语句中生成的 `context_id` 必须在所有工作器上唯一地识别一次分布式反向计算。每个工作器保存与该 `context_id` 相关的元数据，这些对于正确执行分布式 autograd 计算是必须的。

```python
>>> import torch.distributed.autograd as dist_autograd
>>> with dist_autograd.context() as context_id:
>>>   t1 = torch.rand((3, 3), requires_grad=True)
>>>   t2 = torch.rand((3, 3), requires_grad=True)
>>>   loss = rpc.rpc_sync("worker1", torch.add, args=(t1, t2)).sum()
>>>   dist_autograd.backward(context_id, [loss])
```

### get_gradients()

获取张量到其相应梯度的一个映射，其中梯度累积在给定的 `context_id` 对应的上下文中。

```python
>>> import torch.distributed.autograd as dist_autograd
>>> with dist_autograd.context() as context_id:
...   t1 = torch.rand((3, 3), requires_grad=True)
...   t2 = torch.rand((3, 3), requires_grad=True)
...   loss = rpc.rpc_sync("worker1", torch.add, args=(t1, t2)).sum()
...   dist_autograd.backward(context_id, [loss])
...   grads = dist_autograd.get_gradients(context_id)
...   print(grads[t1])
...   print(grads[t2])
... 
tensor([[1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.]])
tensor([[1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.]])
>>> grads
{tensor([[0.9787, 0.3666, 0.9716],
         [0.6967, 0.4684, 0.0524],
         [0.8899, 0.0569, 0.2332]], requires_grad=True): tensor([[1., 1., 1.],
                                                                 [1., 1., 1.],
                                                                 [1., 1., 1.]]), 
 tensor([[0.6230, 0.7423, 0.5838],
         [0.0084, 0.6071, 0.9528],
         [0.3312, 0.6938, 0.6464]], requires_grad=True): tensor([[1., 1., 1.],
                                                                 [1., 1., 1.],
                                                                 [1., 1., 1.]])}
```

## 分布式优化器

### DistributedOptimizer

