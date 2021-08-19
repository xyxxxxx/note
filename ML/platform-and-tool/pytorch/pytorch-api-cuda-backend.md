# torch.cuda

## current_device()

返回当前设备的索引。

```python
>>> torch.cuda.current_device()
0
```



## device()

改变当前设备的上下文管理器。

```python
>>> torch.cuda.device(0)
<torch.cuda.device at 0x7efce0b03be0>
```



## device_count()

返回可用的 GPU 数量。

```python
>>> torch.cuda.device_count()
1
```



## get_device_name()

获取设备的名称，默认返回当前设备（由 `current_device()` 给出）的名称。

```python
>>> torch.cuda.get_device_name(0)
'Tesla T4'
```



## is_available()

返回一个布尔值，表示当前 CUDA 是否可用。

```python
>>> torch.cuda.is_available()
True
```



## set_device()

设定当前 CUDA 设备。

```python
>>> torch.cuda.set_device(1)
```



