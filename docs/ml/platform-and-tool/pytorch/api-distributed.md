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
torch.distributed.rpc.init_rpc(name, backend=None, rank=- 1, world_size=None, rpc_backend_options=None)
# name        此进程的全局唯一名称(例如`Master`,`ParameterServer2`,`Worker1`等)
# backend     RPC后端实现的类型,默认为`Backend.TENSORPIPE`
# rank        此进程的全局唯一rank
# world_size  当前组的工作器数量
# rpc_backend_options
```

### rpc_sync()

```python
torch.distributed.rpc.rpc_sync(to, func, args=None, kwargs=None, timeout=-1.0)
# to          目标工作器的名称/rank/`WorkerInfo`实例
# func        函数,例如Python可调用对象,Python内置函数和被注解的TorchScript函数
# args        传递给`func`的参数元组
# kwargs      传递给`func`的关键字参数字典
# timeout     此RPC调用的超时时间(秒).0表示永不超时.若此参数没有提供,则使用初始化期间设定的默认值
```

进行一次阻塞的 RPC 调用以在工作器 `to` 上运行函数 `func`。RPC messages are sent and received in parallel to execution of Python code. 此方法是线程安全的。

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

> 使用 `rpc_sync()`、`rpc_async()` 等 API 时，函数传入的张量和返回的张量必须是 CPU 张量（否则当两个进程的设备列表不一致时可能会引起崩溃）。如有必要，应用可以显式地在调用进程中将张量移动到 CPU，再在被调用进程中将其移动到想要的设备中。

### rpc_async()

```python
torch.distributed.rpc.rpc_async(to, func, args=None, kwargs=None, timeout=-1.0)
# to          目标工作器的名称/rank/`WorkerInfo`实例
# func        函数,例如Python可调用对象,Python内置函数和被注解的TorchScript函数
# args        传递给`func`的参数元组
# kwargs      传递给`func`的关键字参数字典
# timeout     此RPC调用的超时时间(秒).0表示永不超时.若此参数没有提供,则使用初始化期间设定的默认值
```

进行一次非阻塞的 RPC 调用以在工作器 `to` 上运行函数 `func`，并立即返回一个 `Future` 实例。RPC messages are sent and received in parallel to execution of Python code. 此方法是线程安全的。

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

> API `rpc_async()`、 `remote()` 会到要通过网络发送参数时才复制这些参数（包括其中的张量），这会由另一个线程完成，取决于 RPC 后端的类型。调用进程应确保参数张量的内容保持不变直到返回的 `Future` 实例得到返回值。

### remote()

```python
torch.distributed.rpc.remote(to, func, args=None, kwargs=None, timeout=-1.0)
# to          目标工作器的名称/rank/`WorkerInfo`实例
# func        函数,例如Python可调用对象,Python内置函数和被注解的TorchScript函数
# args        传递给`func`的参数元组
# kwargs      传递给`func`的关键字参数字典
# timeout     此RPC调用的超时时间(秒).0表示永不超时.若此参数没有提供,则使用初始化期间设定的默认值
```

进行一次远程 RPC 调用以在工作器 `to` 上运行函数 `func`，并立即返回一个指向返回值的 `RRef` 实例。工作器 `to` 是返回的 `RRef` 实例的所有者，而调用进程则是使用者。所有者管理 `RRef` 实例的全局引用计数，并在计数归零（即全局不再有对它的任何引用）时销毁该实例。

```python
# on worker 0
>>> import os
>>> os.environ['MASTER_ADDR'] = '127.0.0.1'
>>> os.environ['MASTER_PORT'] = '29500'
>>> import torch
>>> import torch.distributed.rpc as rpc
>>> rpc.init_rpc("worker0", rank=0, world_size=2)
>>> rref1 = rpc.remote("worker1", torch.add, args=(torch.ones(2), 3))
>>> rref2 = rpc.remote("worker1", torch.add, args=(torch.ones(2), 1))
>>> x = rref1.to_here() + rref2.to_here()
>>> x
tensor([6., 6.])
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
>>> rpc.shutdown()     # 等待worker0结束工作,然后关闭
```

### WorkerInfo

包装了工作器信息的结构，其中包含工作器的名称和 ID。此类的实例不应直接构造，而应通过 `get_worker_info()` 函数返回获取；实例可用于传入诸如 `rpc_sync()`、`rpc_async()`、`remote()` 等函数以避免每次远程调用都要复制字符串。

### get_worker_info()

获取指定工作器的 `WorkerInfo`。使用此 `WorkerInfo` 实例以避免每次远程调用都传递非常昂贵的字符串。

```python
torch.distributed.rpc.get_worker_info(worker_name=None)
# worker_name    工作器的字符串名称
```

### BackendType

可用后端的枚举类。

PyTorch 内置一个 `BackendType.TENSORPIPE` 后端；使用 `register_backend()` 函数以注册更多的后端。

#### BackendType.TENSORPIPE

默认使用的 TensorPipe 代理利用了 TensorPipe 库，其提供了一个原生的点对点的通信原语，从根本上解决了 Gloo 的一些缺陷，因而特别适用于机器学习。相比 Gloo，其优势在于大量的异步操作可以同时进行，不会互相阻塞或影响运行速度。节点对之间的管道只有在需要时才会打开；当一个节点故障时只有它相关的管道会被关闭，而其它管道都会照常工作。此外，TensorPipe 还支持多种传输方式（TCP、共享内存、NVLink、InfiniBand 等），能够自动检测这些方式的可用性并为每个管道选择最佳的传输方式。

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
# rpc_timeout          默认的RPC请求的超时时间(秒).调用进程可以单独重载`rpc_sync()`和`rpc_async()`的超时时间
# init_method          同`init_process_group()`的`init_method`参数
# device_maps          从调用进程到被调用进程的设备放置映射.其中键是被调用工作器的名称,值是映射调用进程的设备到
#                      被调用进程的设备的字典(设备用`int`,`str`或`torch.device`表示)
# devices              RPC代理使用的所有本地CUDA设备.默认从`device_maps`初始化得到.
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

> RRef 目前尚不支持 CUDA 张量。

一个 `RRef` 实例是对远程工作器上的一个某种类型的值的引用。这种处理方式使被引用的远程值仍位于其所有者上。RRef 可用于多机训练，通过保有对存在于其它工作器上的 `nn.Modules` 实例的引用，并在训练过程中调用适当的函数以获取或修改其参数。

#### backward()

```python
backward(self, dist_autograd_ctx_id=-1, retain_graph=False)
# dist_autograd_ctx_id   要获取的梯度所位于的分布式autograd上下文的id
# retain_graph           
```

#### confirmed_by_owner()

返回此 `RRef` 实例是否被所有者确认。所有者的 `RRef` 实例总是返回 `True`，而使用者的 `RRef` 实例只有当所有者知道此实例时才返回 `True`。

#### is_owner()

返回当前进程是否是此 `RRef` 实例的所有者。

#### local_value()

若当前进程是所有者，则返回对本地值的一个引用，否则引发一个异常。

#### owner()

返回此 `RRef` 实例的所有者的 `WorkerInfo` 实例。

#### owner_name()

返回此 `RRef` 实例的所有者的名称。

#### remote()

#### rpc_async()

#### rpc_sync()

#### to_here()

阻塞调用，从所有者复制此 `RRef` 实例的值

## autograd

### backward()

### context

### get_gradients()

## optim

### DistributedOptimizer

# torch.multiprocessing

`torch.multiprocessing` 是对原始 `multiprocessing` 模块的一个包装。It registers custom reducers，that use shared memory to provide shared views on the same data in different processes.一旦张量被移动到共享内存（shared_memory）中，它就能被发送到其它进程而无需复制。

此 API 100% 兼容原始模块，你完全可以将 `import multiprocessing` 改为 `import torch.multiprocessing` 以将所有张量送入队列或通过其它机制分享。

由于此 API 与原始 API 的相似性，请参考 `multiprocessing` 包的文档以获取更多细节。

## 共享CUDA张量

在进程间共享 CUDA 张量使用 `spawn` 或 `forkserver` 启动方法。

不同于 CPU 张量，只要接收进程持有的是张量的副本，发送进程就需要保留原始张量。

1. 消费者应尽快释放内存

   ```python
   ## Good
   x = queue.get()
   # do somethings with x
   del x
   ```

   ```python
   ## Bad
   x = queue.get()
   # do somethings with x
   # do everything else (producer have to keep x in memory)
   ```

2. 保持生产者进程运行直到所有消费者退出。这将防止生产者释放消费者依然在使用的张量内存。

   ```python
   ## producer
   # send tensors, do something
   event.wait()
   ```

   ```python
   ## consumer
   # receive tensors and use them
   event.set()
   ```

3. 不要直接转发张量：

   ```python
   # not going to work
   x = queue.get()
   queue_2.put(x)
   ```

   ```python
   # you need to create a process-local copy
   x = queue.get()
   x_clone = x.clone()
   queue_2.put(x_clone)
   ```

   ```python
   # putting and getting from the same queue in the same process will likely end up with segfault
   queue.put(tensor)
   x = queue.get()
   ```

   

## 共享策略（CPU张量）

### `file_descriptor`

### `file_system`

## spawn()

可以通过创建 `Process` 实例以启动若干子进程，执行特定函数，然后调用 `join` 等待其完成。此方法在处理单个子进程时工作得很好，但处理多个子进程时就显露了潜在问题，亦即：以特定顺序 `join` 进程默认了它们会按照该顺序终止。如果事实上没有按照这个顺序，例如在 `join` 第一个进程时后面的进程终止，则不会被注意到。此外，在这一过程中也没有误差传播的原生工具支持。

下面的 `spawn` 函数解决了上述问题，其支持误差传播、任意顺序终止，并且当检测到错误时可以动态地终止进程。

```python
torch.multiprocessing.spawn(fn, args=(), nprocs=1, join=True, daemon=False, start_method='spawn')
# fn        启动进程的作为进入点的调用函数,此函数必须定义在模块的顶级作用域以能够被序列化和启动,
#           这也是`multiprocessing`规定的必要条件
# args      传递给`fn`的参数列表.注意`fn`的第一个参数应为`rank`,由`spawn()`自动传入,此参数传递的参数列表对应
#           `fn`的第二个及以后的所有参数
# nprocs    启动的进程数
# join      join所有进程并阻塞
# daemon    启动进程的守护进程标识.若为`True`,则将创建守护进程

# 若join=True,返回None;否则返回ProcessContext
```

启动 `nprocs` 个进程，以使用 `args` 参数运行 `fn` 函数。

如果进程中的任意一个以非零退出状态退出，则剩余进程将被杀掉，并且抛出一个进程退出原因的异常。

