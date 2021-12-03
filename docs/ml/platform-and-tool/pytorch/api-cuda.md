# torch.cuda

`torch.cuda` 包加入了对于 CUDA 张量类型的支持，其实现了与 CPU 张量同样的函数，但使用 GPU 进行计算。

## 设备管理

### current_device()

返回当前设备的索引。

```python
>>> torch.cuda.current_device()
0
```

### device()

改变当前设备的上下文管理器。

```python
print(torch.cuda.current_device())   # 0
cuda = torch.device('cuda')          # 当前CUDA设备,此时为`'cuda:0'`
cuda0 = torch.device('cuda:0')
cuda2 = torch.device('cuda:2')
x = torch.tensor([1., 2.], device=cuda0)
y = torch.tensor([1., 2.]).cuda()    # `x`和`y`位于设备`'cuda:0'`

with torch.cuda.device(1):           # `'cuda:1'`的上下文管理器
    a = torch.tensor([1., 2.], device=cuda)     # 当前CUDA设备,此时为`'cuda:1'`
    b = torch.tensor([1., 2.]).cuda()           # `a`和`b`位于设备`'cuda:1'`
    c = a + b                                   # `c`同样位于设备`'cuda:1'`

    z = x + y
    # z.device is device(type='cuda', index=0)

    d = torch.randn(2, device=cuda2)
    e = torch.randn(2).to(cuda2)                # 将张量从CPU移动到'cuda:2'
    f = torch.randn(2).cuda(cuda2)              # 同上
    # d.device, e.device, and f.device are all device(type='cuda', index=2)
```

### device_count()

返回可用的 GPU 数量。

```python
>>> torch.cuda.device_count()
1
```

### get_device_capability()

返回设备的 CUDA 计算能力。

```python
>>> torch.cuda.get_device_capability(0)
(3, 7)
```

### get_device_name()

返回设备的名称，默认返回当前设备（由 `current_device()` 给出）的名称。

```python
>>> torch.cuda.get_device_name(0)
'Tesla T4'
```

### get_device_properties()

返回设备的属性。

```python
>>> torch.cuda.get_device_properties(0)
_CudaDeviceProperties(name='Tesla K80', major=3, minor=7, total_memory=11441MB, multi_processor_count=13)
```

### is_available()

返回一个布尔值，表明当前 CUDA 是否可用。

```python
>>> torch.cuda.is_available()
True
```

### set_device()

设定当前 CUDA 设备。

```python
>>> torch.cuda.set_device(1)
```

## 流

### current_stream()

返回指定设备的当前选择的流（`Stream` 实例）。

### default_stream()

返回指定设备的默认流（`Stream` 实例）。

### set_stream()

设定当前流。建议使用 `stream()` 上下文管理器而非此函数。

### stream()

选择一个指定流的上下文管理器（`StreamContext` 实例）的包装器。

### Stream

CUDA 流的包装器。

CUDA 流是一个线性执行序列，从属于一个具体设备，各个流之间相互独立。

#### query()

检查是否所有提交的工作都已经完成。

#### record_event()

记录一个事件。

#### synchronize()

等待此流的所有内核完成。

#### wait_event()

所有提交的之后的工作等待一个事件。

#### wait_stream()

与另一个流同步。

### StreamContext

选择一个指定流的上下文管理器。

此上下文中的所有排队的 CUDA 内核将会添加到指定流的队列中。

### synchronize()

等待指定设备上的所有流的所有内核完成。

## 显存管理

### empty_cache()

### list_gpu_processes()

返回指定设备的当前运行进程及显存使用的可读打印结果。

此函数可帮助处理 OOM 异常。

```python
>>> print(torch.cuda.list_gpu_processes(0))
```

### memory_allocated()

返回指定设备的当前张量占用显存，以字节为单位。

```python
>>> torch.cuda.memory_allocated(0)
512
```

### memory_summary()

返回指定设备的当前显存分配统计数据的可读打印结果。

```python
>>> print(torch.cuda.memory_summary(0))
```

### memory_reserved()

返回指定设备的当前缓存分配器管理的显存，以字节为单位。

```python
>>> torch.cuda.memory_reserved(0)
```

### memory_snapshot()

返回所有设备的显存分配器状态的快照。

读懂此函数的返回值需要熟悉显存分配器的内部工作。

### memory_stats()

返回指定设备的显存分配器统计数据的字典。

## 集合通信

### comm.broadcast()

广播张量到指定的 GPU 设备。

```python
torch.cuda.comm.broadcast(tensor, devices=None, *, out=None)
# tensor   要广播的张量,可以位于CPU或GPU
# devices  GPU设备的可迭代对象,广播到这些设备上
# out      ……不可与`devices`同时指定
```

## 随机数生成

### get_rng_state_all()

### set_rng_state_all()

### manual_seed_all()

### seed_all()

