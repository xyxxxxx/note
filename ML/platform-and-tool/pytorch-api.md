[toc]

# torch

## 张量

### device

`torch.device` 实例代表 `torch.Tensor` 被分配到的设备。



### dtype

`torch.dtype` 实例代表 `torch.Tensor` 的数据类型。PyTorch 有如下数据类型：

| Data type                | dtype                                 | Legacy Constructors      |
| ------------------------ | ------------------------------------- | ------------------------ |
| 32-bit floating point    | `torch.float32` or `torch.float`      | `torch.*.FloatTensor`    |
| 64-bit floating point    | `torch.float64` or `torch.double`     | `torch.*.DoubleTensor`   |
| 64-bit complex           | `torch.complex64` or `torch.cfloat`   |                          |
| 128-bit complex          | `torch.complex128` or `torch.cdouble` |                          |
| 16-bit floating point 1  | `torch.float16` or `torch.half`       | `torch.*.HalfTensor`     |
| 16-bit floating point 2  | `torch.bfloat16`                      | `torch.*.BFloat16Tensor` |
| 8-bit integer (unsigned) | `torch.uint8`                         | `torch.*.ByteTensor`     |
| 8-bit integer (signed)   | `torch.int8`                          | `torch.*.CharTensor`     |
| 16-bit integer (signed)  | `torch.int16` or `torch.short`        | `torch.*.ShortTensor`    |
| 32-bit integer (signed)  | `torch.int32` or `torch.int`          | `torch.*.IntTensor`      |
| 64-bit integer (signed)  | `torch.int64` or `torch.long`         | `torch.*.LongTensor`     |
| Boolean                  | `torch.bool`                          | `torch.*.BoolTensor`     |







### Tensor

#### bool()





#### cpu()

返回张量的一个位于内存中的副本。



#### cuda()

返回张量的一个位于显存中的副本。可以通过 `device` 参数指定 CUDA 设备，默认为当前 CUDA 设备。

如果张量已经位于当前 CUDA 设备的显存中，则直接返回该张量对象。



#### detach()

返回一个张量，其与输入张量共享内存，但在计算图之外，不参与梯度计算。

```python
# 1
>>> a = torch.tensor([1, 2, 3.], requires_grad=True)
>>> out = a.sigmoid()
>>> out.sum().backward()
>>> a.grad
tensor([1., 1., 1.])

# 2
>>> a = torch.tensor([1, 2, 3.], requires_grad=True)
>>> out = a.sigmoid()
>>> c = out.detach()
>>> out.sum().backward()  # 可以计算梯度 
>>> a.grad
tensor([0.1966, 0.1050, 0.0452])

# 3
>>> a = torch.tensor([1, 2, 3.], requires_grad=True)
>>> out = a.sigmoid()
>>> c = out.detach()
>>> c.sum().backward()    # c不能计算梯度
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn

# 4
>>> a = torch.tensor([1, 2, 3.], requires_grad=True)
>>> out = a.sigmoid()
>>> c = out.detach()
>>> c.zero_()
>>> out.sum().backward()    # out的值被修改而不能计算梯度
RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: ……
```



#### expand()

将张量在某些维度上以复制的方式扩展。注意内存共享问题。

```python
>>> a = torch.tensor([[1], [2], [3]])
>>> a.size()
torch.Size([3, 1])
>>> a.expand(-1, 4)       # -1 表示此维度保持不变
tensor([[ 1,  1,  1,  1],
        [ 2,  2,  2,  2],
        [ 3,  3,  3,  3]])
>>> a1 = x.expand(-1, 4)  # 共享内存
>>> a1[0][0] = 0
>>> a1
tensor([[0, 0, 0, 0],     # 共享内存
        [2, 2, 2, 2],
        [3, 3, 3, 3]])
>>> a
tensor([[0],
        [2],
        [3]])
```



#### get_device()

对于 CUDA 张量，返回其所位于的 GPU 设备的序号；对于 CPU 张量，抛出一个错误。

```python
>>> x = torch.randn(3, 4, 5, device='cuda:0')
>>> x.get_device()
0
>>> x.cpu().get_device()
# RuntimeError: get_device is not implemented for type torch.FloatTensor
```



#### item()

对于只有一个元素的张量，返回该元素的值。

```python
>>> a = torch.tensor([[[1]]])
>>> a.shape
torch.Size([1, 1, 1])
>>> a.item()
1
```



#### new_full(), new_ones(), new_zeros()

`new_full()` 返回一个指定形状和所有元素值的张量，并且该张量与调用对象有同样的 `torch.dtype` 和 `torch.device`。

```python
>>> a = torch.randn((2,), dtype=torch.float64)
>>> a.new_full((2, 3), 3.141592)
tensor([[ 3.1416,  3.1416,  3.1416],
        [ 3.1416,  3.1416,  3.1416]], dtype=torch.float64)
```

`new_ones(),new_zeros()` 返回一个指定形状的全 1/全 0 张量，并且该张量与调用对象有同样的 `torch.dtype` 和 `torch.device`。

```python
>>> a = torch.randn((2,), dtype=torch.float64)
>>> a.new_ones((2,3))
tensor([[1., 1., 1.],
        [1., 1., 1.]], dtype=torch.float64)
>>> a.new_zeros((2,3))
tensor([[0., 0., 0.],
        [0., 0., 0.]], dtype=torch.float64)
```



#### permute()

返回将调用对象的所有维度重新排序得到的张量。

```python
>>> a = torch.arange(24).view(2,3,4)
>>> a
tensor([[[ 0,  1,  2,  3],
         [ 4,  5,  6,  7],
         [ 8,  9, 10, 11]],

        [[12, 13, 14, 15],
         [16, 17, 18, 19],
         [20, 21, 22, 23]]])
>>> a.permute(2, 0, 1)
tensor([[[ 0,  4,  8],
         [12, 16, 20]],

        [[ 1,  5,  9],
         [13, 17, 21]],

        [[ 2,  6, 10],
         [14, 18, 22]],

        [[ 3,  7, 11],
         [15, 19, 23]]])
```



#### repeat()

将张量在某些维度上重复。

```python
>>> a = torch.arange(24).view(2, 3, 4)
>>> a.repeat(1, 1, 1).shape     # x自身
torch.Size([2, 3, 4])
>>> a.repeat(2, 1, 1).shape     # 在维度0上重复2次
torch.Size([4, 3, 4])
>>> a.repeat(2, 1, 1)
tensor([[[ 0,  1,  2,  3],
         [ 4,  5,  6,  7],
         [ 8,  9, 10, 11]],

        [[12, 13, 14, 15],
         [16, 17, 18, 19],
         [20, 21, 22, 23]],

        [[ 0,  1,  2,  3],
         [ 4,  5,  6,  7],
         [ 8,  9, 10, 11]],

        [[12, 13, 14, 15],
         [16, 17, 18, 19],
         [20, 21, 22, 23]]])
>>> a1 = a.repeat(2, 1, 1)
>>> a1[0][0][0] = 1
>>> a1
tensor([[[ 1,  1,  2,  3],       # 不共享内存
         [ 4,  5,  6,  7],
         [ 8,  9, 10, 11]],

        [[12, 13, 14, 15],
         [16, 17, 18, 19],
         [20, 21, 22, 23]],

        [[ 0,  1,  2,  3],
         [ 4,  5,  6,  7],
         [ 8,  9, 10, 11]],

        [[12, 13, 14, 15],
         [16, 17, 18, 19],
         [20, 21, 22, 23]]])
```



#### T

返回将调用对象的所有维度反转后的张量。

```python
>>> a = torch.randn(3, 4, 5)
>>> a.T.shape
torch.Size([5, 4, 3])
```



#### to()

返回调用对象更改 `torch.dtype` 和 `torch.device` 后的张量。

```python
>>> a = torch.randn(2, 2)  # Initially dtype=float32, device=cpu
>>> a.to(torch.float64)
tensor([[-0.5044,  0.0005],
        [ 0.3310, -0.0584]], dtype=torch.float64)
>>> cuda0 = torch.device('cuda:0')
>>> a.to(cuda0)
tensor([[-0.5044,  0.0005],
        [ 0.3310, -0.0584]], device='cuda:0')
>>> a.to(cuda0, dtype=torch.float64)
tensor([[-0.5044,  0.0005],
        [ 0.3310, -0.0584]], dtype=torch.float64, device='cuda:0')
```



#### topk()

返回一维张量的最大的 k 个数。对于二维张量，返回每行的最大的 k 个数。

```python
>>> a = torch.arange(6)
>>> a.topk(1)
torch.return_types.topk(values=tensor([5]),indices=tensor([5]))
>>> a.topk(3)
torch.return_types.topk(values=tensor([5, 4, 3]),indices=tensor([5, 4, 3]))

>>> a = torch.arange(6).view(2,3)
>>> v, i = a.topk(1)
>>> v
tensor([[2],
        [5]]) 
>>> i
tensor([[2],
        [2]])

```





## 张量创建

### arange()

根据给定的初值，末值和步长创建一维张量。与 Python 的 `range()` 用法相同。

```python
>>> torch.arange(5)
tensor([ 0,  1,  2,  3,  4])
>>> torch.arange(1, 4)
tensor([ 1,  2,  3])
>>> torch.arange(1, 2.5, 0.5)
tensor([ 1.0000,  1.5000,  2.0000])
```



### from_numpy()

由 NumPy 数组（`numpy.ndarray` 实例）创建张量。返回张量与 NumPy 数组共享内存。

```python
>>> a = numpy.array([1, 2, 3])
>>> t = torch.from_numpy(a)
>>> t
tensor([ 1,  2,  3])
>>> t[0] = -1         # 返回张量与NumPy数组共享内存
>>> a
array([-1,  2,  3])
```



### full()

返回指定形状的用指定值填充的张量。

```python
>>> torch.full((2, 3), 3.141592)
tensor([[ 3.1416,  3.1416,  3.1416],
        [ 3.1416,  3.1416,  3.1416]])
```



### linspace()

根据给定的初值，末值和项数创建一维张量。

```python
>>> torch.linspace(0, 10, steps=5)
tensor([ 0.0000,  2.5000,  5.0000,  7.5000, 10.0000])
>>> torch.linspace(0, 10, steps=1)
tensor([0.])
```



### logspace()

根据给定的底数和指数的初值，末值和项数创建一维张量。

```python
>>> torch.logspace(0, 10, steps=5)           # 默认底数为10
tensor([1.0000e+00, 3.1623e+02, 1.0000e+05, 3.1623e+07, 1.0000e+10])
>>> torch.logspace(0, 10, steps=5, base=2)
tensor([1.0000e+00, 5.6569e+00, 3.2000e+01, 1.8102e+02, 1.0240e+03])
>>> torch.logspace(0, 10, steps=1)
tensor([1.])
```



### ones()

返回指定形状的全 1 张量。

```python
>>> torch.ones(2, 3)
tensor([[ 1.,  1.,  1.],
        [ 1.,  1.,  1.]])
>>> torch.ones(2, 3, dtype=int)
tensor([[1, 1, 1],
        [1, 1, 1]])
```



### set_default_tensor_type()

设置默认的浮点数据类型。

```python
>>> torch.tensor([1.2, 3]).dtype
torch.float32
>>> torch.set_default_tensor_type(torch.DoubleTensor)
>>> torch.tensor([1.2, 3]).dtype
torch.float64
```



### tensor()

由数据创建张量。

```python
torch.tensor(data, *, dtype=None, device=None, requires_grad=False, pin_memory=False) → Tensor
# data           张量的初始数据,可以是数字,列表,元组,`numpy.ndarray`类型等.
# dtype          张量的数据类型,默认从`data`中推断.
# device         张量位于的设备,默认使用数据类型相应的设备:若为CPU数据类型,则使用CPU;若为CUDA数据类型,
#                则使用当前的CUDA设备
# requires_grad  若为`True`,则autograd应记录此张量参与的运算
# pin_memory     若为`True`,则张量将被分配到锁页内存中.仅对位于CPU的张量有效.
```

```python
>>> torch.tensor([[0.1, 1.2], [2.2, 3.1], [4.9, 5.2]])
tensor([[ 0.1000,  1.2000],
        [ 2.2000,  3.1000],
        [ 4.9000,  5.2000]])

>>> torch.tensor([0, 1])   # 从数据推断类型
tensor([ 0,  1])

>>> torch.tensor([[0.11111, 0.222222, 0.3333333]],
...              dtype=torch.float64,
...              device=torch.device('cuda:0'))   # 创建一个`torch.cuda.DoubleTensor`
tensor([[ 0.1111,  0.2222,  0.3333]], dtype=torch.float64, device='cuda:0')

>>> torch.tensor(3.14159)  # 创建一个标量(0维张量)
tensor(3.1416)

>>> torch.tensor([])       # 创建一个空张量(形状为(0,))
tensor([])
```



### zeros()

返回指定形状的全 0 张量。

```python
>>> torch.zeros(2, 3)
tensor([[ 0.,  0.,  0.],
        [ 0.,  0.,  0.]])
```



## 张量操作

> 以下操作均不是原位操作，即返回一个新的张量，而不改变原张量。

### cat()

沿指定轴拼接张量。

```python
>>> a = torch.randn(2, 3)
>>> a
tensor([[ 0.6580, -1.0969, -0.4614],
        [-0.1034, -0.5790,  0.1497]])
>>> torch.cat((a, a, a), 0)
tensor([[ 0.6580, -1.0969, -0.4614],
        [-0.1034, -0.5790,  0.1497],
        [ 0.6580, -1.0969, -0.4614],
        [-0.1034, -0.5790,  0.1497],
        [ 0.6580, -1.0969, -0.4614],
        [-0.1034, -0.5790,  0.1497]])
>>> torch.cat((a, a, a), 1)
tensor([[ 0.6580, -1.0969, -0.4614,  0.6580, -1.0969, -0.4614,  0.6580,
         -1.0969, -0.4614],
        [-0.1034, -0.5790,  0.1497, -0.1034, -0.5790,  0.1497, -0.1034,
         -0.5790,  0.1497]])
```



### index_select()

选择张量的沿指定轴的若干索引，返回相应的子张量。

```python
>>> a = torch.arange(24).view(2, 3, 4)
>>> a
tensor([[[ 0,  1,  2,  3],
         [ 4,  5,  6,  7],
         [ 8,  9, 10, 11]],

        [[12, 13, 14, 15],
         [16, 17, 18, 19],
         [20, 21, 22, 23]]])
>>> torch.index_select(a, 2, torch.tensor([0, 2]))
tensor([[[ 0,  2],
         [ 4,  6],
         [ 8, 10]],

        [[12, 14],
         [16, 18],
         [20, 22]]])
```



### flatten()

将张量展开为向量。

```python
>>> a = torch.tensor([[[1, 2],
                       [3, 4]],
                      [[5, 6],
                       [7, 8]]])
>>> torch.flatten(a)
tensor([1, 2, 3, 4, 5, 6, 7, 8])
>>> torch.flatten(a, start_dim=1)
tensor([[1, 2, 3, 4],
        [5, 6, 7, 8]])
```



### masked_select()

根据布尔掩码返回由张量的部分元素组成的一维张量。

```python
>>> a = torch.randn(3, 4)
>>> a
tensor([[ 0.3552, -2.3825, -0.8297,  0.3477],
        [-1.2035,  1.2252,  0.5002,  0.6248],
        [ 0.1307, -2.0608,  0.1244,  2.0139]])
>>> mask = a.ge(0.5)       # 比较运算产生与张量形状相同的掩码
>>> mask
tensor([[False, False, False, False],
        [False, True, True, True],
        [False, False, False, True]])
>>> torch.masked_select(a, mask)
tensor([ 1.2252,  0.5002,  0.6248,  2.0139])   # `True`对应的元素组成的一维张量
```



### movedim()

移动张量指定轴的位置。

```python
>>> a = torch.arange(24).view(2, 3, 4)
>>> torch.movedim(a, 1, 0).shape
torch.Size([3, 2, 4])
>>> torch.movedim(a, 2, 0).shape
torch.Size([4, 2, 3])
```



### reshape()

改变张量的形状。

```python
>>> a = torch.arange(10).view(2, -1)
>>> a
tensor([[0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9]])
>>> torch.reshape(a, (5, 2))      # 保持各元素的顺序
tensor([[0, 1],
        [2, 3],
        [4, 5],
        [6, 7],
        [8, 9]])
>>> torch.reshape(a, (-1,))
tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
```



### split()

划分张量为多个部分，每个部分是原始张量的一个视图。

```python
>>> a = torch.arange(36).reshape(6, 6)
>>> a0, a1, a2 = torch.split(a, 2, dim=0)         # 沿轴0 每2个索引划分
>>> a0
tensor([[ 0,  1,  2,  3,  4,  5],
        [ 6,  7,  8,  9, 10, 11]])
>>> a0, a1 = torch.split(a, 3, dim=1)             # 沿轴1 每3个索引划分
>>> a0
tensor([[ 0,  1,  2],
        [ 6,  7,  8],
        [12, 13, 14],
        [18, 19, 20],
        [24, 25, 26],
        [30, 31, 32]])
>>> a0, a1 = torch.split(a, 4, dim=1)             # 最后一个部分不足4个索引 
>>> a1
tensor([[ 4,  5],
        [10, 11],
        [16, 17],
        [22, 23],
        [28, 29],
        [34, 35]])
>>> a0, a1, a2 = torch.split(a, [1, 2, 3], dim=1) # 沿轴1划分
>>> a0
tensor([[ 0],
        [ 6],
        [12],
        [18],
        [24],
        [30]])
```



### squeeze()

移除张量的规模为 1 的维度。返回张量与输入张量共享内存。亦为 `torch.Tensor` 方法。

```python
>>> a = torch.randn(1,2,1,3,1,4)
>>> a.shape
torch.Size([1, 2, 1, 3, 1, 4])
>>> torch.squeeze(a).shape           # 移除所有规模为1的维度
torch.Size([2, 3, 4])
>>> torch.squeeze(a, 0).shape        # 移除指定规模为1的维度
torch.Size([2, 1, 3, 1, 4])
```



### stack()

沿新轴拼接形状相同的张量。

```python
>>> a1 = torch.randn(3, 4)
>>> a2 = torch.randn(3, 4)
>>> torch.stack((a1, a2), 0).shape
torch.Size([2, 3, 4])
>>> torch.stack((a1, a2), 1).shape
torch.Size([3, 2, 4])
```



### t()

转置二维张量。

```python
>>> a = torch.arange(12).view(3, 4)
>>> a
tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
>>> torch.t(a)
tensor([[ 0,  4,  8],
        [ 1,  5,  9],
        [ 2,  6, 10],
        [ 3,  7, 11]])
```



### take()

根据索引返回由张量的部分元素组成的一维张量，其中将张量视作展开处理。

```python
>>> a = torch.arange(12).view(3, 4)
>>> torch.take(a, torch.tensor([0, 2, 5]))
tensor([ 0,  2,  5])
```



### take_along_dim()





### tile()

重复张量的元素。亦为 `torch.Tensor` 方法。

```python
>>> a = torch.tensor([1, 2, 3])
>>> a.tile((2,))
tensor([1, 2, 3, 1, 2, 3])
>>> a = torch.tensor([[1, 2], [3, 4]])
>>> torch.tile(a, (2, 2))
tensor([[1, 2, 1, 2],
        [3, 4, 3, 4],
        [1, 2, 1, 2],
        [3, 4, 3, 4]])
```



### transpose()

交换张量的指定两个维度。亦为 `torch.Tensor` 方法。

```python
>>> a = torch.arange(12).view(3, 4)
>>> a
tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
>>> torch.transpose(a, 0, 1)
tensor([[ 0,  4,  8],
        [ 1,  5,  9],
        [ 2,  6, 10],
        [ 3,  7, 11]])
```



### unbind()

移除张量的一个维度，返回沿此维度的所有切片组成的元组。

```python
>>> torch.unbind(torch.tensor([[1, 2, 3],
                           [4, 5, 6],
                           [7, 8, 9]]))
(tensor([1, 2, 3]), tensor([4, 5, 6]), tensor([7, 8, 9]))
```



### unsqueeze()

在张量的指定位置插入一个规模为 1 的维度。返回张量与输入张量共享内存。亦为 `torch.Tensor` 方法。

```python
>>> a = torch.randn(2, 3, 4)
>>> torch.unsqueeze(a, 0).shape
torch.Size([1, 2, 3, 4])
>>> a.unsqueeze(3).shape
torch.Size([2, 3, 4, 1])
```





## 数学运算

### abs()

对张量的所有元素应用绝对值函数。亦为 `torch.Tensor` 方法。

```python
>>> torch.abs(torch.tensor([-1, -2, 3]))
tensor([1,  2,  3])
>>> torch.tensor([-1, -2, 3]).abs()
tensor([1, 2, 3])
```



### add(), sub()

张量加法/减法。亦为 `torch.Tensor` 方法。`+,-` 符号重载了这些方法。

```python
>>> a = torch.arange(12).view(3, 4)
>>> a
tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
>>> a + 1                    # 张量+标量: 扩张的张量加法
tensor([[ 1,  2,  3,  4],
        [ 5,  6,  7,  8],
        [ 9, 10, 11, 12]])
>>> a + torch.tensor([1])    # 同前
tensor([[ 1,  2,  3,  4],
        [ 5,  6,  7,  8],
        [ 9, 10, 11, 12]])
>>> a + torch.arange(4)      # 张量+子张量: 扩张的张量加法
tensor([[ 0,  2,  4,  6],
        [ 4,  6,  8, 10],
        [ 8, 10, 12, 14]])
>>> a + a                    # 张量+张量: 张量加法
tensor([[ 0,  2,  4,  6],
        [ 8, 10, 12, 14],
        [16, 18, 20, 22]])

```



### all(), any()





### argmax(), argmin()

返回张量沿指定维度的最大值的索引。

```python
>>> a = torch.randn(4)
>>> a
tensor([-1.6751, -0.7609,  0.8919, -0.0545])
>>> torch.argmax(a)
tensor(2)
>>> torch.argmin(a)
tensor(0)

>>> a = torch.randn(4, 4)
>>> a
tensor([[ 1.3398,  0.2663, -0.2686,  0.2450],
        [-0.7401, -0.8805, -0.3402, -1.1936],
        [ 0.4907, -1.3948, -1.0691, -0.3132],
        [-1.6092,  0.5419, -0.2993,  0.3195]])
>>> torch.argmax(a, dim=1)
tensor([ 0,  2,  0,  1])
>>> torch.argmin(a, dim=1)
tensor([ 2,  3,  1,  0])
```



### bmm()

批量矩阵乘法。

```python
>>> m1 = torch.randn(10, 3, 4)
>>> m2 = torch.randn(10, 4, 5)
>>> res = torch.bmm(m1, m2)    # 相同索引的矩阵对应相乘
>>> res.size()
torch.Size([10, 3, 5])
```



### ceil()

对张量的所有元素应用向上取整函数。

```python
>>> a = torch.randn(4)
>>> a
tensor([-0.6341, -1.4208, -1.0900,  0.5826])
>>> torch.ceil(a)
tensor([-0., -1., -1.,  1.])
```



### clamp()

对张量的所有元素应用下限和上限。

```python
>>> a = torch.randn(4)
>>> a
tensor([-1.7120,  0.1734, -0.0478, -0.0922])
>>> torch.clamp(a, min=-0.5, max=0.5)
tensor([-0.5000,  0.1734, -0.0478, -0.0922])

>>> a = torch.randn(4)
>>> a
tensor([-0.0299, -2.3184,  2.1593, -0.8883])
>>> torch.clamp(a, min=0.5)
tensor([ 0.5000,  0.5000,  2.1593,  0.5000])
```

`torch.clamp(x,min=0)` 即为 ReLU 激活函数。



### equal()

判断两个张量是否相等。

```python
>>> one1 = torch.ones(2,3)
>>> one2 = torch.ones(2,3)
>>> one1 == one2
tensor([[True, True, True],
        [True, True, True]])
>>> one1.equal(one2)
True
```



### exp()

对张量的所有元素应用自然指数函数。亦为 `torch.Tensor` 方法。

```python
>>> a = torch.arange(10.).view(2,5)
>>> a
tensor([[0., 1., 2., 3., 4.],
        [5., 6., 7., 8., 9.]])
>>> torch.exp(a)
tensor([[1.0000e+00, 2.7183e+00, 7.3891e+00, 2.0086e+01, 5.4598e+01],
        [1.4841e+02, 4.0343e+02, 1.0966e+03, 2.9810e+03, 8.1031e+03]])
```



### floor()

对张量的所有元素应用向下取整函数。

```python
>>> a = torch.randn(4)
>>> a
tensor([-0.8166,  1.5308, -0.2530, -0.2091])
>>> torch.floor(a)
tensor([-1.,  1., -1., -1.])
```



### log(), log10(), log2()

对张量的所有元素应用对数函数。亦为 `torch.Tensor` 方法。

```python
>>> a
tensor([[0., 1., 2., 3., 4.],
        [5., 6., 7., 8., 9.]])
>>> torch.log(a)
tensor([[  -inf, 0.0000, 0.6931, 1.0986, 1.3863],
        [1.6094, 1.7918, 1.9459, 2.0794, 2.1972]])
>>> torch.log2(a)
tensor([[  -inf, 0.0000, 1.0000, 1.5850, 2.0000],
        [2.3219, 2.5850, 2.8074, 3.0000, 3.1699]])
>>> torch.log10(a)
tensor([[  -inf, 0.0000, 0.3010, 0.4771, 0.6021],
        [0.6990, 0.7782, 0.8451, 0.9031, 0.9542]])
```



### matmul()

张量乘法。亦为 `torch.Tensor` 方法。`@` 符号重载了此方法。

```python
# 向量×向量: 内积
>>> v1 = torch.tensor([1, 2, 3])
>>> torch.matmul(v1, v1)
tensor(14)

# 矩阵×向量, 向量×矩阵, 矩阵×矩阵: 矩阵乘法
>>> m1 = torch.arange(1, 10).view(3, 3)
>>> m1
tensor([[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]])
>>> torch.matmul(m1, v1)                 # 向量会自动补全维度
tensor([14, 32, 50])                     # 3x3 x 3(x1) = 3(x1)
>>> torch.matmul(v1, m1)
tensor([30, 36, 42])                     # (1x)3 x 3x3 = (1x)3

# 矩阵序列×向量: 扩张的矩阵乘法
>>> bm1 = m1.view(1, 3, 3).repeat(2, 1, 1)
>>> bm1
tensor([[[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]],

        [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]]])
>>> torch.matmul(bm1, v1)
tensor([[14, 32, 50],                     # [2x]3x3 x 3(x1) = [2x]3(x1)
        [14, 32, 50]])

# 矩阵序列×矩阵: 扩张的矩阵乘法
>>> m2 = torch.ones(3, 3, dtype=torch.int64)
>>> torch.matmul(bm1, m2)
tensor([[[ 6,  6,  6],                    # [2x]3x3 x 3x3 = [2x]3x3
         [15, 15, 15],
         [24, 24, 24]],

        [[ 6,  6,  6],
         [15, 15, 15],
         [24, 24, 24]]])

# 矩阵序列×矩阵序列: 逐元素的矩阵乘法
>>> bm2 = m2.view(1, 3, 3).repeat(2, 1, 1)
>>> bm2[1] = 2
>>> bm2
>>> bm2
tensor([[[1, 1, 1],
         [1, 1, 1],
         [1, 1, 1]],

        [[2, 2, 2],
         [2, 2, 2],
         [2, 2, 2]]])
>>> torch.matmul(bm1, bm2)
tensor([[[ 6,  6,  6],
         [15, 15, 15],
         [24, 24, 24]],

        [[12, 12, 12],
         [30, 30, 30],
         [48, 48, 48]]])

# 矩阵序列×向量序列: 逐元素的矩阵乘法 不适用,会被识别为 矩阵序列×矩阵
# 请将向量序列扩展为矩阵序列
```



### max(), min(), mean(), std()

返回张量所有元素统计量。亦为 `torch.Tensor` 方法。

```python
>>> a = torch.arange(10.).view(2, -1)
>>> a
tensor([[0., 1., 2., 3., 4.],
        [5., 6., 7., 8., 9.]])

>>> torch.max(a)
tensor(9.)
>>> torch.max(a, 0)
torch.return_types.max(
values=tensor([5., 6., 7., 8., 9.]),
indices=tensor([1, 1, 1, 1, 1]))
>>> torch.max(a, 1)
torch.return_types.max(
values=tensor([4., 9.]),
indices=tensor([4, 4]))

>>> torch.min(a)
tensor(0.)
>>> torch.min(a, 0)
torch.return_types.min(
values=tensor([0., 1., 2., 3., 4.]),
indices=tensor([0, 0, 0, 0, 0]))
>>> torch.min(a, 1)
torch.return_types.min(
values=tensor([0., 5.]),
indices=tensor([0, 0]))

>>> torch.mean(a)
tensor(4.5000)
>>> torch.mean(a, 0)
tensor([2.5000, 3.5000, 4.5000, 5.5000, 6.5000])
>>> torch.mean(a, 1)
tensor([2., 7.])

>>> torch.std(a, unbiased=False)
tensor(2.8723)
>>> torch.std(a, 0, unbiased=False)
tensor([2.5000, 2.5000, 2.5000, 2.5000, 2.5000])
>>> torch.std(a, 1, unbiased=False)
tensor([1.4142, 1.4142])

# 统计学的标准差
>>> torch.std(a)
tensor(3.0277)
>>> torch.std(a, 0)
tensor([3.5355, 3.5355, 3.5355, 3.5355, 3.5355])
>>> torch.std(a, 1)
tensor([1.5811, 1.5811])
```



### mm()

矩阵乘法。亦为 `torch.Tensor` 方法。

```python
>>> m1 = torch.randn(1, 3)
>>> m2 = torch.randn(3, 1)
>>> torch.mm(m1, m2)
tensor([[0.0717]])
```



### mul(), div(), fmod(), pow()

张量逐元素乘法/除法/除法取余/乘方。亦为 `torch.Tensor` 方法。`*, /, %, **` 符号重载了此方法。

```python
>>> a = torch.arange(12).view(3, 4)
>>> a
tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
>>> a * 100                       # 张量*标量: 张量的数乘
tensor([[   0,  100,  200,  300],
        [ 400,  500,  600,  700],
        [ 800,  900, 1000, 1100]])
>>> a * torch.arange(4)           # 张量*子张量: 张量的扩张逐元素乘法
tensor([[ 0,  1,  4,  9],
        [ 0,  5, 12, 21],
        [ 0,  9, 20, 33]])
>>> a * a                         # 张量*张量: 张量的逐元素乘法
tensor([[  0,   1,   4,   9],
        [ 16,  25,  36,  49],
        [ 64,  81, 100, 121]])

```



### sigmoid()

Sigmoid 激活函数。亦为 `torch.Tensor` 方法。见 `torch.nn.Sigmoid`。

```python
>>> input = torch.randn(2)
>>> input
tensor([1.7808, -0.9893])
>>> torch.sigmoid(input)
tensor([0.8558, 0.2710])
```



### sign()

对张量的所有元素应用符号函数。

```python
>>> a = torch.tensor([0.7, -1.2, 0., 2.3])
>>> a
tensor([ 0.7000, -1.2000,  0.0000,  2.3000])
>>> torch.sign(a)
tensor([ 1., -1.,  0.,  1.])
```



### sin(), cos(), tan(), arcsin(), arccos(), arctan(), sinh(), cosh(), tanh(), arcsinh(), arccosh(), arctanh()

对张量的所有元素应用三角函数和双曲函数。

```python
>>> a
tensor([[0., 1., 2., 3., 4.],
        [5., 6., 7., 8., 9.]])
>>> torch.sin(a)
tensor([[ 0.0000,  0.8415,  0.9093,  0.1411, -0.7568],
        [-0.9589, -0.2794,  0.6570,  0.9894,  0.4121]])
>>> a.sin()
tensor([[ 0.0000,  0.8415,  0.9093,  0.1411, -0.7568],
        [-0.9589, -0.2794,  0.6570,  0.9894,  0.4121]])
```



### sqrt()

对张量的所有元素应用平方根函数。

```python
>>> a = torch.randn(4)
>>> a
tensor([-2.0755,  1.0226,  0.0831,  0.4806])
>>> torch.sqrt(a)
tensor([    nan,  1.0112,  0.2883,  0.6933])
```



### square()

对张量的所有元素应用平方函数，相当于 `** 2`。







### tanh()

tanh 激活函数。亦为 `torch.Tensor` 方法。

```python
>>> input = torch.randn(2)
>>> input
tensor([-1.5400,  0.3318])
>>> torch.tanh(input)
tensor([-0.9121,  0.3202])
```



## 随机数

### manual_seed()

设置产生随机数的种子。

```python
>>> torch.manual_seed(1)
```



### rand()

返回指定形状的随机张量，其中每个元素服从 (0, 1) 区间的均匀分布。

```python
>>> torch.rand(2, 3)
tensor([[ 0.8237,  0.5781,  0.6879],
        [ 0.3816,  0.7249,  0.0998]])
```



### randint()

返回指定形状的随机张量，其中每个元素等可能地取到 [low, high) 区间内的各个整数。

```python
>>> torch.randint(10, (3, 3))
tensor([[3, 7, 1],
        [5, 8, 4],
        [2, 3, 5]])
>>> torch.randint(6, 10, (3, 3))
tensor([[6, 9, 7],
        [7, 6, 6],
        [8, 6, 7]])
```



### randn()

返回指定形状的随机张量，其中每个元素服从标准正态分布。

```python
>>> torch.randn(2, 3)
tensor([[ 1.5954,  2.8929, -1.0923],
        [ 1.1719, -0.4709, -0.1996]])
```





# torch.nn

## 线性层

### Linear

全连接层。

```python
>>> m = nn.Linear(20, 4)
>>> input = torch.randn(128, 20)
>>> m(input).size()
torch.Size([128, 4])
```



## 卷积层

### Conv1d

一维卷积层。

```python
torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
# in_channels     输入通道数
# out_channels    输出通道数
# kernel_size     卷积核大小(高/宽)
# stride          卷积步长(高/宽)
# padding         填充的行/列数(上下/左右)
# padding_mode    填充模式
# dilation        卷积核元素的间隔
```



```python
>>> m1 = nn.Conv1d(1, 32, 3, 1)                 # 卷积核长度为3,步长为1
>>> m2 = nn.Conv1d(1, 32, 3, 3)                 # 步长为3
>>> m3 = nn.Conv1d(1, 32, 3, 3, padding=(1,1))  # 左右各用1个零填充
>>> input = torch.rand((100, 1, 28))
>>> output1, output2, output3= m1(input), m2(input), m3(input)
>>> output1.shape
torch.Size([100, 32, 26])
>>> output2.shape
torch.Size([100, 32, 9])
>>> output3.shape
torch.Size([100, 32, 10])
```



### Conv2d

二维卷积层。

```python
torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
# in_channels     输入通道数
# out_channels    输出通道数
# kernel_size     卷积核大小(高/宽)
# stride          卷积步长(高/宽)
# padding         填充的行/列数(上下/左右)
# padding_mode    填充模式
# dilation        卷积核元素的间隔
```



```python
>>> m1 = nn.Conv2d(1, 32, 3, 1)                 # 卷积核大小为(3,3),步长为1
												                        # 将1个通道(卷积特征)映射到32个通道(卷积特征)
>>> m2 = nn.Conv2d(1, 32, (3,5), 1)             # 卷积核大小为(3,5)
>>> m3 = nn.Conv2d(1, 32, 3, 3)                 # 步长为3
>>> m4 = nn.Conv2d(1, 32, 3, 3, padding=(1,1))  # 上下/左右各用1/1行零填充
>>> input = torch.rand((100, 1, 28, 28))
>>> m1(input).shape
torch.Size([100, 32, 26, 26])
>>> m2(input).shape
torch.Size([100, 32, 26, 24])
>>> m3(input).shape
torch.Size([100, 32, 9, 9])
>>> m4(input).shape
torch.Size([100, 32, 10, 10])
```



## 汇聚层（池化层）

### MaxPool1d

一维最大汇聚层。见 torch.nn.functional.max_pool1d。

```python
>>> m1 = nn.MaxPool1d(2, stride=1)
>>> m2 = nn.MaxPool1d(2, stride=2)
>>> input = torch.randn(1, 8)
>>> output1 = m1(input)
>>> output2 = m2(input)
>>> input
tensor([[ 0.3055,  0.5521,  1.9417, -0.7325,  0.3202, -1.4555,  1.7270,  3.1311]])
>>> output1
tensor([[0.5521, 1.9417, 1.9417, 0.3202, 0.3202, 1.7270, 3.1311]])
>>> output2
tensor([[0.5521, 1.9417, 0.3202, 3.1311]])
```



### MaxPool2d

二维最大汇聚层。见 torch.nn.functional.max_pool2d。

```python
>>> m1 = nn.MaxPool2d(2, stride=1)
>>> m2 = nn.MaxPool2d(2, stride=2)
>>> input = torch.randn(1, 1, 4, 4)
>>> output1 = m1(input)
>>> output2 = m2(input)
>>> input
tensor([[[[-0.5308,  1.2014, -1.3582,  1.1337],
          [ 0.2359,  0.9501,  1.1915,  0.3432],
          [-1.4260, -0.1276, -2.2615,  0.8555],
          [-0.8545,  0.5436,  1.6482,  1.2749]]]])
>>> output1
tensor([[[[1.2014, 1.2014, 1.1915],
          [0.9501, 1.1915, 1.1915],
          [0.5436, 1.6482, 1.6482]]]])
>>> output2
tensor([[[[1.2014, 1.1915],
          [0.5436, 1.6482]]]])
```



## 循环层

### GRU

GRU 层。

```python
>>> rnn = nn.GRU(5, 10, 2)           # GRU可以视作简化的LSTM,各参数含义与LSTM相同
>>> input = torch.randn(20, 64, 5)
>>> h0 = torch.randn(2, 64, 10)
>>> output, hn = rnn(input, h0)
>>> output.shape
torch.Size([20, 64, 10])
>>> hn.shape
torch.Size([2, 64, 10])
```



### LSTM

> 参考：[理解Pytorch中LSTM的输入输出参数含义](https://www.cnblogs.com/marsggbo/p/12123755.html)

LSTM 层。

```python
>>> rnn = nn.LSTM(5, 10, 2, [dropout=0.5]) # 输入向量x的维数为5,隐状态h的维数为10,堆叠2层
										   # 在每层(最上层除外)的输出位置增加一个dropout层
									       # 多层LSTM中,上层的输入是下层的隐状态
>>> input = torch.randn(20, 64, 5)         # 一批64个序列,每个序列有20个5维向量
>>> h0 = torch.randn(2, 64, 10)            # 第一个参数为层数与方向数的乘积,单向和双向LSTM
										   #   的方向数分别为1和2
    									   # 第二个参数为输入序列的数量
>>> c0 = torch.randn(2, 64, 10)            # 第三个参数为隐状态维数
>>> output, (hn, cn) = rnn(input, (h0, c0)) # 输入h,c的初值,输出h,c的终值
											# 若不输入初值,则默认为0
>>> output.shape
torch.Size([20, 64, 10])                   # 从前往后输出最上层的所有隐状态
>>> hn.shape
torch.Size([2, 64, 10])                    # 输出每一(层,方向)的最终隐状态
                                           # 对于单向LSTM, hn[-1]==output[-1]


>>> rnn = nn.LSTM(5, 10, 2, bidirectional=True)  # 双向LSTM,相当于将输入向量正向和反向各
                                                 #   输入一次
>>> input = torch.randn(20, 64, 5)
>>> h0 = torch.randn(4, 64, 10)                  # 层数*方向数=4
>>> c0 = torch.randn(4, 64, 10)
>>> output, (hn, cn) = rnn(input, (h0, c0))
>>> output.shape
torch.Size([20, 64, 20])                   # 输出最上层的所有隐状态,拼接正向与反向的输出
>>> hn.shape
torch.Size([4, 64, 10])                    # 每一(层,方向)的最终隐状态
```

> 尤其需要注意的是，这里接受的输入张量的形状为`(seq_len, batch, input_size)`，而常见的输入的形状为`(batch, seq_len, input_size)`，为此需要使用`transpose()`或`permute()`方法交换维度。参见[For beginners: Do not use view() or reshape() to swap dimensions of tensors!](https://discuss.pytorch.org/t/for-beginners-do-not-use-view-or-reshape-to-swap-dimensions-of-tensors/75524)



## 嵌入层

### Embedding

嵌入层。

```python
>>> embedding = nn.Embedding(10, 3)  # 词汇表规模 = 10, 嵌入维数 = 3, 共30个参数
                                     # 注意10表示词汇表规模,输入为0~9的整数而非10维向量
>>> input = torch.LongTensor([[1,2,4,5],[4,3,2,9]])
>>> embedding(input)
tensor([[[-0.0251, -1.6902,  0.7172],
         [-0.6431,  0.0748,  0.6969],
         [ 1.4970,  1.3448, -0.9685],
         [-0.3677, -2.7265, -0.1685]],

        [[ 1.4970,  1.3448, -0.9685],
         [ 0.4362, -0.4004,  0.9400],
         [-0.6431,  0.0748,  0.6969],
         [ 0.9124, -2.3616,  1.1151]]])
```



## 丢弃层

### Dropout

以给定概率将张量中的每个数置零，剩余的数乘以 $$1/(1-p)$$。每次使用 Dropout 层的结果是随机的。

```python
>>> m = nn.Dropout(0.5)
>>> input = torch.randn(4, 4)
>>> output = m(input)
>>> input
tensor([[-1.1218,  0.1338, -0.0065, -1.6416],
        [ 0.8897, -1.6002, -0.6922,  0.0689],
        [-1.3392, -0.5207, -0.2739, -0.9653],
        [ 0.6608,  0.9212,  0.0579,  0.9670]])
>>> output
tensor([[-0.0000,  0.2677, -0.0000, -3.2831],
        [ 1.7795, -3.2004, -1.3843,  0.0000],
        [-0.0000, -0.0000, -0.0000, -0.0000],
        [ 0.0000,  1.8425,  0.1158,  0.0000]])
```



### Dropout2d

以给定概率将张量 $$(N,C,H,W)$$ 的每个通道置零,剩余的通道乘以 $$1/(1-p)$$。每次使用 Dropout 层的结果是随机的。

```python
>>> m = nn.Dropout2d(0.5)
>>> input = torch.randn(1, 8, 2, 2)
>>> output = m(input)
>>> input
tensor([[[[ 1.7200, -0.7948],
          [-0.1551, -0.8467]],

         [[-1.0479, -0.6172],
          [-0.8419, -0.8668]],

         [[ 0.4776,  1.7682],
          [ 1.0376,  0.8871]],

         [[-0.8826,  1.5624],
          [ 1.4573, -0.0573]],

         [[-1.4288, -0.6288],
          [ 1.2000,  1.3250]],

         [[ 1.8099,  0.7262],
          [-0.5595,  1.4562]],

         [[ 0.7452, -2.1875],
          [ 0.0116,  0.5224]],

         [[ 0.1152,  0.1012],
          [ 0.5634, -0.1202]]]])
>>> output
tensor([[[[ 0.0000, -0.0000],
          [-0.0000, -0.0000]],

         [[-2.0957, -1.2344],
          [-1.6837, -1.7336]],

         [[ 0.9551,  3.5364],
          [ 2.0752,  1.7741]],

         [[-1.7651,  3.1247],
          [ 2.9146, -0.1147]],

         [[-0.0000, -0.0000],
          [ 0.0000,  0.0000]],

         [[ 3.6198,  1.4524],
          [-1.1190,  2.9124]],

         [[ 0.0000, -0.0000],
          [ 0.0000,  0.0000]],

         [[ 0.0000,  0.0000],
          [ 0.0000, -0.0000]]]])
```



## 激活函数

### ReLU

ReLU 激活函数层。见 `torch.nn.functional.relu`。

```python
>>> m = nn.ReLU()
>>> input = torch.randn(2)
>>> output = m(input)
>>> input
tensor([ 1.2175, -0.7772])
>>> output
tensor([1.2175, 0.0000])
```



### Sigmoid

Logistic 激活函数层。见 `torch.sigmoid`。

```python
>>> m = nn.Sigmoid()
>>> input = torch.randn(2)
>>> output = m(input)
>>> input
tensor([ 1.7808, -0.9893])
>>> output
tensor([0.8558, 0.2710])
```



### Softmax, LogSoftmax

Softmax 层。torch.nn.LogSoftmax 相当于在 Softmax 的基础上为每个输出值求（自然）对数。

```python
>>> m1 = nn.Softmax(dim=0)
>>> m2 = nn.LogSoftmax(dim=0)
>>> input = torch.arange(4.0)
>>> output1 = m1(input)
>>> output2 = m2(input)
>>> input
tensor([0., 1., 2., 3.])
>>> output1
tensor([0.0321, 0.0871, 0.2369, 0.6439])
>>> output2            # logsoftmax() = softmax() + log()
tensor([-3.4402, -2.4402, -1.4402, -0.4402])
```





## 损失函数

### CrossEntropyLoss

交叉熵损失函数。见 `torch.nn.NLLLoss`。

```python
torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')
```





```python
>>> loss = nn.CrossEntropyLoss()
>>> a1 = torch.tensor([[0.1, 0.8, 0.1]])    # prediction
>>> a2 = torch.tensor([1])                  # label
>>> loss(a1, a2)
tensor(0.6897)
>>> a2 = torch.tensor([0])
>>> loss(a1, a2)
tensor(1.3897)

# CrossEntropyLoss() = softmax() + log() + NLLLoss() = logsoftmax() + NLLLoss()
>>> loss = nn.CrossEntropyLoss()
>>> input = torch.tensor([[ 0.4377, -0.3976, -1.3221],
                          [ 1.8402, -0.1696,  0.4744],
                          [-3.4641, -0.2303,  0.3552]])
>>> target = torch.tensor([0, 1, 2])
>>> loss(input, target)
tensor(1.0896)

>>> loss = nn.NLLLoss()
>>> input = torch.tensor([[ 0.4377, -0.3976, -1.3221],
                          [ 1.8402, -0.1696,  0.4744],
                          [-3.4641, -0.2303,  0.3552]])
>>> input = input.softmax(dim=1)
>>> input = input.log()
>>> target = torch.tensor([0, 1, 2])
>>> loss(input, target)
tensor(1.0896)

>>> loss = nn.NLLLoss()
>>> input = torch.tensor([[ 0.4377, -0.3976, -1.3221],
                          [ 1.8402, -0.1696,  0.4744],
                          [-3.4641, -0.2303,  0.3552]])
>>> logsoftmax = nn.LogSoftmax(dim=1)
>>> input = logsoftmax(input)
>>> target = torch.tensor([0, 1, 2])
>>> loss(input, target)
tensor(1.0896)
```



### MSELoss

均方差损失函数。

```python
>>> a1 = torch.arange(10.0)
>>> a2 = a1+2
>>> loss = nn.MSELoss()
>>> b = loss(a1, a2)
>>> b
tensor(4.)
>>> loss = nn.MSELoss(reduction='sum')
>>> b = loss(a1, a2)
>>> b
tensor(40.)
```



### NLLLoss

见 `torch.nn.CrossEntropyLoss`。

```python
>>> loss = nn.NLLLoss()
>>> input = torch.tensor([[ 0.4377, -0.3976, -1.3221],
                          [ 1.8402, -0.1696,  0.4744],
                          [-3.4641, -0.2303,  0.3552]])
>>> input = input.softmax(dim=1)
>>> input = input.log()
>>> target = torch.tensor([0, 1, 2])
>>> loss(input, target)
tensor(1.0896)

```



### L1Loss

平均绝对误差损失函数。

```python
>>> a1 = torch.arange(10.0)
>>> a2 = a1+2
>>> loss = nn.L1Loss()
>>> b = loss(a1, a2)
>>> b
tensor(2.)
>>> loss = nn.MSELoss(reduction='sum')
>>> b = loss(a1, a2)
>>> b
tensor(20.)
```





## 模组

### Module



named_parameters



parameters



## 数据并行模组

### DataParallel





### parallel.DistributedDataParallel





## 实用功能

### Flatten

将张量展开为向量，用于顺序模型。

```python

```







# torch.nn.functional



## max_pool1d()

一维最大汇聚函数。见 `torch.nn.MaxPool1d`。

```python
>>> input = torch.randn(1, 8)
>>> output1 = F.max_pool1d(input, 2, 1)
>>> output2 = F.max_pool1d(input, 2)
>>> input
tensor([[ 0.3055,  0.5521,  1.9417, -0.7325,  0.3202, -1.4555,  1.7270,  3.1311]])
>>> output1
tensor([[0.5521, 1.9417, 1.9417, 0.3202, 0.3202, 1.7270, 3.1311]])
>>> output2
tensor([[0.5521, 1.9417, 0.3202, 3.1311]])
```



## max_pool2d()

二维最大汇聚函数。见 `torch.nn.MaxPool2d`。

```python
>>> input = torch.randn(1, 1, 4, 4)
>>> input
tensor([[[[-0.5308,  1.2014, -1.3582,  1.1337],
          [ 0.2359,  0.9501,  1.1915,  0.3432],
          [-1.4260, -0.1276, -2.2615,  0.8555],
          [-0.8545,  0.5436,  1.6482,  1.2749]]]])
>>> F.max_pool2d(input, 2, 1)
tensor([[[[1.2014, 1.2014, 1.1915],
          [0.9501, 1.1915, 1.1915],
          [0.5436, 1.6482, 1.6482]]]])
>>> F.max_pool2d(input, 2)
tensor([[[[1.2014, 1.1915],
          [0.5436, 1.6482]]]])
```



## one_hot()

将向量转换为 one-hot 表示。

```python
>>> F.one_hot(torch.arange(0, 5))
tensor([[1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1]])

>>> F.one_hot(torch.arange(0, 5) % 3, num_classes=5)
tensor([[1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0]])
```



## relu()

ReLU 激活函数。见 `torch.nn.ReLU`。

```python
>>> input = torch.randn(2)
>>> input
tensor([1.2175, -0.7772])
>>> F.relu(input)
tensor([1.2175, 0.0000])
```



## sigmoid() (deprecated)

Sigmoid 激活函数。见 `torch.nn.Sigmoid,torch.sigmoid`。

```python
>>> input = torch.randn(2)
>>> input
tensor([1.7808, -0.9893])
>>> F.sigmoid(input)
tensor([0.8558, 0.2710])
```



## softmax()

softmax 回归。

```python
>>> input = torch.arange(5.)
>>> F.softmax(input)
tensor([0.0117, 0.0317, 0.0861, 0.2341, 0.6364])

>>> input = torch.tensor([[0.1,0.2,0.3,1],[0.2,0.3,0.4,2]])
>>> F.softmax(input, dim=0)  # for every column
tensor([[0.4750, 0.4750, 0.4750, 0.2689],
        [0.5250, 0.5250, 0.5250, 0.7311]])
>>> F.softmax(input, dim=1)  # for every row
tensor([[0.1728, 0.1910, 0.2111, 0.4251],
        [0.1067, 0.1179, 0.1303, 0.6452]])
```



## tanh()

tanh 激活函数。

```python

```



# tensor attributes

## torch.dtype

| Data type                | dtype                                 | CPU tensor             | GPU tensor                  |
| ------------------------ | ------------------------------------- | ---------------------- | --------------------------- |
| 32-bit floating point    | `torch.float32` or `torch.float`      | `torch.FloatTensor`    | `torch.cuda.FloatTensor`    |
| 64-bit floating point    | `torch.float64` or `torch.double`     | `torch.DoubleTensor`   | `torch.cuda.DoubleTensor`   |
| 16-bit floating point    | `torch.float16` or `torch.half`       | `torch.HalfTensor`     | `torch.cuda.HalfTensor`     |
| 16-bit floating point    | `torch.bfloat16`                      | `torch.BFloat16Tensor` | `torch.cuda.BFloat16Tensor` |
| 32-bit complex           | `torch.complex32`                     |                        |                             |
| 64-bit complex           | `torch.complex64`                     |                        |                             |
| 128-bit complex          | `torch.complex128` or `torch.cdouble` |                        |                             |
| 8-bit integer (unsigned) | `torch.uint8`                         | `torch.ByteTensor`     | `torch.cuda.ByteTensor`     |
| 8-bit integer (signed)   | `torch.int8`                          | `torch.CharTensor`     | `torch.cuda.CharTensor`     |
| 16-bit integer (signed)  | `torch.int16` or `torch.short`        | `torch.ShortTensor`    | `torch.cuda.ShortTensor`    |
| 32-bit integer (signed)  | `torch.int32` or `torch.int`          | `torch.IntTensor`      | `torch.cuda.IntTensor`      |
| 64-bit integer (signed)  | `torch.int64` or `torch.long`         | `torch.LongTensor`     | `torch.cuda.LongTensor`     |
| Boolean                  | `torch.bool`                          | `torch.BoolTensor`     | `torch.cuda.BoolTensor`     |

类型转换：

```python
>>> a = np.array([1.,2,3])
>>> t64 = torch.tensor(a)
>>> t64                                     # t64.dtype = torch.float64
tensor([1., 2., 3.], dtype=torch.float64)
>>> t32 = torch.tensor(a, dtype=torch.float32)
>>> t32                                     # t32.dtype = torch.float32
tensor([1., 2., 3.])
>>> t32 = t64.float()                       # 类型转换
>>> t32
tensor([1., 2., 3.])
# 注意nn中各层接受的tensor类型一般为float32
```

不同类型张量进行运算的结果类型：

```python
>>> float_tensor = torch.ones(1, dtype=torch.float)
>>> double_tensor = torch.ones(1, dtype=torch.double)
>>> complex_float_tensor = torch.ones(1, dtype=torch.complex64)
>>> complex_double_tensor = torch.ones(1, dtype=torch.complex128)
>>> int_tensor = torch.ones(1, dtype=torch.int)
>>> long_tensor = torch.ones(1, dtype=torch.long)
>>> uint_tensor = torch.ones(1, dtype=torch.uint8)
>>> double_tensor = torch.ones(1, dtype=torch.double)
>>> bool_tensor = torch.ones(1, dtype=torch.bool)
# zero-dim tensors
>>> long_zerodim = torch.tensor(1, dtype=torch.long)
>>> int_zerodim = torch.tensor(1, dtype=torch.int)

>>> torch.add(5, 5).dtype
torch.int64
# 5 is an int64, but does not have higher category than int_tensor so is not considered.
>>> (int_tensor + 5).dtype
torch.int32
>>> (int_tensor + long_zerodim).dtype
torch.int32
>>> (long_tensor + int_tensor).dtype
torch.int64
>>> (bool_tensor + long_tensor).dtype
torch.int64
>>> (bool_tensor + uint_tensor).dtype
torch.uint8
>>> (float_tensor + double_tensor).dtype
torch.float64
>>> (complex_float_tensor + complex_double_tensor).dtype
torch.complex128
>>> (bool_tensor + int_tensor).dtype
torch.int32
# Since long is a different kind than float, result dtype only needs to be large enough
# to hold the float.
>>> torch.add(long_tensor, float_tensor).dtype
torch.float32
```

不同类型张量相乘的可行性：

```python
# allowed:
>>> float_tensor *= double_tensor
>>> float_tensor *= int_tensor
>>> float_tensor *= uint_tensor
>>> float_tensor *= bool_tensor
>>> float_tensor *= double_tensor
>>> int_tensor *= long_tensor
>>> int_tensor *= uint_tensor
>>> uint_tensor *= int_tensor

# disallowed (RuntimeError: result type can't be cast to the desired output type):
>>> int_tensor *= float_tensor
>>> bool_tensor *= int_tensor
>>> bool_tensor *= uint_tensor
>>> float_tensor *= complex_float_tensor
```



## torch.device

`torch.device` 对象表示 `torch.Tensor` 被分配在哪个设备上。

`torch.device` 包含了设备类型（`cpu` 或 `cuda`）和可选的设备序号。如果设备序号没有指定，对象总是会代表设备类型的当前设备。例如，构造 `torch.Tensor` 时选择设备 `cuda`，即为选择设备 `cuda:X`，其中 X 为 `torch.cuda.current_device()` 的结果。

```python
>>> torch.device('cuda:0')
device(type='cuda', index=0)
# or
>>> torch.device('cuda', 0)
device(type='cuda', index=0)

>>> torch.device('cuda')  # current cuda device
device(type='cuda')

>>> torch.device('cpu', 0)
device(type='cpu', index=0)

>>> torch.device('cpu')
device(type='cpu')
```





# torch.view

view 相当于 numpy 中的 resize 功能，即改变张量的形状。

```python
>>> a = torch.arange(12)
>>> b = a.view(2, 6)
>>> c = b.view(1, -1) # -1位置的参数会根据元素总数和其它维度的长度计算
>>> a
tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
>>> b
tensor([[ 0,  1,  2,  3,  4,  5],
        [ 6,  7,  8,  9, 10, 11]])
>>> c
tensor([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11]])
```





# torch.autograd

`torch.autograd` 提供了实现自动微分的类和函数。



## backward

```python
torch.autograd.backward(tensors: Union[torch.Tensor, Sequence[torch.Tensor]], grad_tensors: Union[torch.Tensor, Sequence[torch.Tensor], None] = None, retain_graph: Optional[bool] = None, create_graph: bool = False, grad_variables: Union[torch.Tensor, Sequence[torch.Tensor], None] = None) → None
# tensors       计算此张量对所有叶节点的梯度
# grad_tensors  tensors是非标量时,与此张量作内积以转换为标量.形状必须与tensors相同
# retain_graph  若为False,计算图在梯度计算完成后(backward()返回后)即被释放.注意在几
#                  乎所有情形下将其设为True都是不必要的,因为总有更好的解决方法
# create_graph  若为True,可以计算更高阶梯度
```

计算 `tensors` 对所有计算图叶节点的梯度。

图使用链式法则进行微分。如果 `tensors` 不是一个标量（即拥有多于一个元素）且需要计算梯度，函数就需要指定 `grad_tensors`，`tensors` 与 `grad_tensors` 作内积从而转换为一个标量。`grad_tensors` 必须与 `tensors` 的形状相同。

该函数会在叶节点累计梯度，因此你需要在每次迭代时先将其归零。



## grad

```python
torch.autograd.grad(outputs: Union[torch.Tensor, Sequence[torch.Tensor]], inputs: Union[torch.Tensor, Sequence[torch.Tensor]], grad_outputs: Union[torch.Tensor, Sequence[torch.Tensor], None] = None, retain_graph: Optional[bool] = None, create_graph: bool = False, only_inputs: bool = True, allow_unused: bool = False) → Tuple[torch.Tensor, ...]
# outputs       计算outputs对inputs的梯度
# inputs
# grad_outputs  类似于backward()的grad_tensors
# retain_graph  同backward()
# create_graph  同backward()
# only_inputs   若为True,则只计算对inputs的梯度;若为False,则计算对所有叶节点的梯度,
#                  并将梯度累加到.grad上
# allow_unused  若为False,则指定的inputs没有参与outputs的计算将视作一个错误
```

计算 `outputs` 对 `inputs` 的梯度并返回。



## no_grad

禁用梯度计算的上下文管理器。

在此模式下，所有运算的结果都有 `requires_grad = False`，即使输入有`requires_grad = True`。当你确定不会调用`tensor.backward()`时，禁用梯度计算可以降低结果本来为`requires_grad = True` 的运算这一部分的内存消耗。

可以作为装饰器使用。

```python
>>> x = torch.tensor([1], requires_grad=True)
>>> with torch.no_grad():
...   y = x * 2
>>> y.requires_grad
False
>>> @torch.no_grad()
... def doubler(x):
...     return x * 2
>>> z = doubler(x)
>>> z.requires_grad
False
```



## enable_grad

启用梯度计算的上下文管理器。

用于在 `no_grad` 或 `set_grad_enabled` 禁用梯度计算的环境下启用梯度计算。

可以作为装饰器使用。

```python
>>> x = torch.tensor([1], requires_grad=True)
>>> with torch.no_grad():
...   with torch.enable_grad():
...     y = x * 2
>>> y.requires_grad
True
>>> y.backward()
>>> x.grad
>>> @torch.enable_grad()
... def doubler(x):
...     return x * 2
>>> with torch.no_grad():
...     z = doubler(x)
>>> z.requires_grad
True
```



## set_grad_enabled

设置梯度计算开或关的上下文管理器。

根据其参数 `mode` 启用或禁用梯度计算。可以用作上下文管理器或函数。

```python
>>> x = torch.tensor([1], requires_grad=True)
>>> is_train = False
>>> with torch.set_grad_enabled(is_train):
...   y = x * 2
>>> y.requires_grad
False
>>> torch.set_grad_enabled(True)
>>> y = x * 2
>>> y.requires_grad
True
>>> torch.set_grad_enabled(False)
>>> y = x * 2
>>> y.requires_grad
False
```



## Function

> 参考：pytorch-tutorial-torch.autograd的简单入门-Function

记录运算历史，定义运算导数公式。

对（`requires_grad=True` 的）张量的每一次运算都会创建一个新的 `Function` 对象，用于执行计算、记录过程。运算历史会保留在由 `Function` 对象构成的有向无环图的形式中，其中边表示数据的依赖关系（`input <- output`）。当调用`backward`时，计算图按照拓扑序进行处理，依次调用每个`Function`对象的`backward()` 方法，传递返回的梯度值。

`Function` 类的一般使用方法是创建子类并定义新操作，这是扩展 `torch.autograd` 的推荐方法。

```python
class Exp(Function):

    # 执行运算
    # 第一个参数必须接受一个context,用于保存张量,在backward中取回
    @staticmethod
    def forward(ctx, i):
        result = i.exp()
        ctx.save_for_backward(result)  # 保存张量,这里为e**i
        return result
    
    # 定义该运算的导数公式
    # 第一个参数必须接受一个context,用于取回张量;属性ctx.needs_input_grad是一个
    #    布尔类型的元组,表示哪些输入需要计算梯度
    # 之后的每一个参数对应(损失)对相应输出的梯度
    # 返回的每一个变量对应(损失)对相应输入的梯度
    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_tensors    # 取回张量
        return grad_output * result    # 计算梯度并返回

# Use it by calling the apply method:
output = Exp.apply(input)
```





# torch.cuda

## current_device()

返回当前选择设备的索引。

```python
>>> torch.cuda.current_device()
0
```



## device()

改变当前选择设备的上下文管理器。

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





# torch.distributed

`torch.distributed` 包为跨多个计算节点的多进程并行提供了 PyTorch 支持和通信原语。建立在此功能上的 `torch.nn.parallel.DistributedDataParallel` 类提供了 PyTorch 模型的同步训练的包装器，它与 `torch.multiprocessing` 提供的并行方法以及 `torch.nn.DataParallel` 的不同之处在于它支持在由网络连接的多台机器上运行，以及用户必须显式地为每个进程启动一个训练脚本的一个单独的副本。

即使是单台机器上的同步训练，`torch.nn.parallel.DistributedDataParallel` 包装器也相对于其它数据并行的方法具有优势，因为其每个进程都拥有单独的 Python 解释器，消除了 GIL 锁对于性能的限制。



## 后端

`torch.distributed` 支持三种后端：gloo、mpi 和 nccl，它们各自的适用条件请参考[官方文档](https://pytorch.org/docs/stable/distributed.html#backends)。



## 初始化

### init_process_group()

初始化默认的分布式进程组，同时初始化 `torch.distributed` 包。

```python
torch.distributed.init_process_group(backend, init_method=None, timeout=datetime.timedelta(0, 1800), world_size=-1, rank=-1, store=None, group_name='', pg_options=None)
# backend      使用的后端,可以是`'mpi'`,`'gloo'`或`'nccl'`
# init_method  指明如何初始化进程组的URL.如果`init_method`和`store`都没有指定,则默认为'env://'.与`store`互斥
# world_size   参与任务的进程数
# rank         当前进程的rank
# store        对于所有worker可见的键值存储,用于交换连接/地址信息.与`init_method`互斥
# timeout      进程组执行操作的超时时间,默认为30min
# group_name   进程组名称
```



当前主要有以下三种初始化方法：

+ **TCP 初始化**

  此方法需要指定 rank 0 进程所在节点的地址以及各进程的 rank。

  ```python
  import torch.distributed as dist
  
  # Use address of one of the machines
  dist.init_process_group(backend, init_method='tcp://10.1.1.20:23456',
                          rank=args.rank, world_size=4)
  ```

  

+ **共享文件系统初始化**



+ **环境变量初始化**

  此方法从环境变量中读取配置，允许用户完全自定义配置信息。需要设置的变量有：

  + `MASTER_ADDR`：rank 0 进程所在节点的地址
  + `MASTER_PORT`：rank 0 进程所在节点的空闲端口号
  + `WORLD_SIZE`：进程数
  + `RANK`：当前进程的 rank

  此方法为默认方法。



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

所有可用后端的近似于枚举类。



#### GLOO

gloo 后端。



#### MPI

MPI 后端。



#### NCCL

NCCL 后端。



### get_backend(), get_rank(), get_world_size()

返回指定进程组的后端、rank 和进程数，默认为当前进程组。



## 分布式键值存储

分布式键值存储用于在一个进程组的各进程之间共享数据或者初始化进程组，共有 3 种类型：`TCPStore`、`FileStore` 和 `HashStore`。



### FileStore



### HashStore



### Store

存储的基类。



#### add()



#### delete_key()



#### get()



#### num_keys()



#### wait()



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

### new_group()





## 点对点通信







## 集体通信

### all_gather()



### all_reduce()



### all_to_all()



### broadcast()



### gather()



### reduce()



### scatter()





# torch.multiprocessing

`torch.multiprocessing` 是对原始 `multiprocessing` 模块的一个包装。It registers custom reducers，that use shared memory to provide shared views on the same data in different processes.一旦张量被移动到共享内存（shared_memory）中，它就能被发送到其它进程而无需复制。

此 API100% 兼容原始模块，你完全可以将 `import multiprocessing`改为`import torch.multiprocessing` 以将所有张量送入队列或通过其它机制分享。

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



## `spawn()`

可以通过创建 `Process` 实例以启动若干子进程，执行特定函数，然后调用 `join` 等待其完成。此方法在处理单个子进程时工作得很好，但处理多个子进程时就显露了潜在问题，亦即：以特定顺序 `join` 进程默认了它们会按照该顺序终止。如果事实上没有按照这个顺序，例如在 `join` 第一个进程时后面的进程终止，则不会被注意到。此外，在这一过程中也没有误差传播的原生工具支持。

下面的 `spawn` 函数解决了上述问题，支持误差传播、任意顺序终止，并且当检测到错误时可以动态地终止进程。



```python
torch.multiprocessing.spawn(fn, args=(), nprocs=1, join=True, daemon=False, start_method='spawn')
# fn        启动进程的作为进入点的调用函数,此函数必须定义在模块的top level以能够被序列
#           化和启动,这也是multiprocessing规定的必要条件
# args      传递给fn的参数
# nprocs    启动的进程数
# join      对于所有进程join并阻塞
# daemon    启动进程的守护进程标识.若为True,将创建守护进程

# 若join=True,返回None;否则返回ProcessContext
```

启动 `nprocs` 个进程，以使用 `args` 参数运行 `fn` 函数。

如果进程中的任意一个以非零退出状态退出，则剩余进程将被杀掉，并且抛出一个进程退出原因的异常。











# torch.optim

```python
optimizer = optim.SGD(model.parameters(), lr=0.01)
              # 梯度下降法   需要学习的参数  学习率
```



## Adam

实现 Adam 算法。

```python
torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
# params        要优化的参数的可迭代对象,或定义了参数组的字典
# lr            学习率
# betas         用于计算梯度的running average和其平方的系数
# eps           添加到分母的项,用于提升数值稳定性
# weight_decay  权重衰退(L2惩罚)
# amsgrad       是否使用此算法的AMSGrad变体
```



## Optimizer

所有优化器的基类。



### load_state_dict()

加载优化器状态。



### state_dict()

返回优化器状态。返回的字典包含两项：

+ `state`：包含当前优化状态的字典
+ `param_groups`：包含所有参数组的字典



### step

执行单步优化。



### zero_grad







## SGD

实现随机梯度下降。

```python
torch.optim.SGD(params, lr=<required parameter>, momentum=0, dampening=0, weight_decay=0, nesterov=False)
# params        要优化的参数的可迭代对象,或定义了参数组的字典
# lr            学习率
# momentum      动量系数
# weight_decay  权重衰退(L2惩罚)
# dampening     
# nesterov      启用Nesterov动量
```

```python
>>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
>>> optimizer.zero_grad()
>>> loss_fn(model(input), target).backward()
>>> optimizer.step()
```





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
for indices in batch_sampler:
    yield collate_fn([dataset[i] for i in indices])
```

从迭代数据集加载就大致相当于：

```python
dataset_iter = iter(dataset)
for indices in batch_sampler:
    yield collate_fn([next(dataset_iter) for _ in indices])
```

自定义 `collate_fn` 可以用于自定义整理过程，即填充顺序数据到 batch 的最大长度。



#### 禁用自动分批

在有些情况下，用户可能想要手动处理分批，或仅加载单个样本。例如，直接加载 batched 数据会使得花销更小（从数据库批量读取，从磁盘批量读取，读取主存的连续块等），或者 batch size 取决于数据本身，或者模型被设计为在单个样本上运行。在这些情景下，更好的做法是不使用自动分批（和 `collate_fn` 函数），而让 `dataloader` 直接返回 `dataset` 对象的成员。

当 `batch_size` 和 `batch_sampler` 都为 `None` 时，自动分批就被禁用。每一个从 `dataset` 获得的样本都由 `collate_fn` 参数传入的函数处理。

当自动分批被禁用时，默认的 `collate_fn` 函数仅将 numpy 数组转化为 PyTorch 张量，而不做其它改变。

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



#### 使用`collate_fn`函数

`collate_fn` 的作用根据启动或禁用自动分批而略有差异。





### 单进程和多进程数据加载



### num_workers





## BatchSampler

包装另一个 sampler 并 yield 一个 mini-batch 的索引。

```python
>>> from torch.utils.data import SequentialSampler
>>> from torch.utils.data import BatchSampler
>>> sampler = SequentialSampler(range(10))
>>> sampler = BatchSampler(sampler, batch_size=3, drop_last=False)
>>> for i in sampler:
...   print(i)
... 
[0, 1, 2]
[3, 4, 5]
[6, 7, 8]
[9]

>>> sampler = SequentialSampler(range(10))
>>> sampler = BatchSampler(sampler, batch_size=3, drop_last=True)
>>> for i in sampler:
...   print(i)
... 
[0, 1, 2]
[3, 4, 5]
[6, 7, 8]
```

```python
>>> from torch.utils.data import RandomSampler
>>> from torch.utils.data import BatchSampler
>>> sampler = RandomSampler(range(10), replacement=True, num_samples=100)
>>> sampler = BatchSampler(sampler, batch_size=3, drop_last=False)
>>> for i in sampler:
...   print(i)
... 
[1, 4, 5]
[6, 8, 9]
# ...
[5, 7, 0]
[9]

```





## ChainDataset





## ConcatDataset





## DataLoader

数据加载器，其结合数据集和采样器，返回一个给定数据集上的可迭代对象。

```python
torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, collate_fn=None, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None, multiprocessing_context=None, generator=None, *, prefetch_factor=2, persistent_workers=False)
# dataset        加载的数据集
# batch_size     批次规模
# shuffle        若为`True`,则每完成一次迭代都重新打乱数据
# sampler        采样器,其定义了从数据集中采样的策略,可以是任何实现了`__len__`方法的可迭代对象.与`shuffle`互斥
# batch_sampler  与`sampler`类似,但每次返回一个批次的索引.与`batch_size`,`shuffle`,`sample`,`drop_last`互斥
# num_workers    用于加载数据的工作进程(子进程)数量,`0`表示在主进程中加载数据
# collate_fn
# pin_memory     若为`True`,则数据加载器将在返回张量之前将它们复制到
# drop_last      若为`True`,则丢弃末尾的达不到批次规模的剩余样本;若为`False`,则剩余样本将组成一个较小的批次
# timeout        从工作进程收集一个批次数据的超时时间
# worker_init_fn
# generator
# prefetch_factor
# persistant_workers
```

```python
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
>>> train_loader.dataset                                          # 采样的数据集
<torch.utils.data.dataset.Subset object at 0x1559ae8e0>
>>> train_loader.sampler                                          # 当`shuffle=True`时创建并使用的随机采样器
<torch.utils.data.sampler.RandomSampler object at 0x152229f40>
>>> len(train_loader.sampler)
48000
>>> list(train_loader.sampler)                                    # 每次采样的顺序不同
[1883, 2208, 28103, 25083, 3052, 44262, 2523, 12614, 44167, 44528, 43330, 4986, 38242, 5401, 20988, 10679, 26630, 5071, 39648, 12959, 37922, 47678, 16923, 39058, 411, 24899, 3682, 21712, 9970, 20472, 18930, 3124, 12951, ...]
>>> train_loader.batch_size                                       # 批次规模
32
>>> len(train_loader)                                             # 数据加载器规模,即产生多少个批次
1500
```





## Dataset

表示数据集的抽象类。

所有映射数据集应继承此类。所有子类应覆写 `__getitem__()` 方法，以支持由键拿到数据样本。子类也可以覆写 `__len__()` 方法，用于由于 `Sampler` 的许多实现和 `DataLoader` 的默认选项而返回数据集的大小。

> 对于映射数据集，`DataLoader` 默认构造一个产生整数索引的索引采样器。如果映射数据集的索引或键不是整数，则需要提供一个自定义采样器。

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
>>> len(train_set)                                             # 数据集大小
60000
>>> train_set.data                                             # 数据张量
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
# shuffle           若为`True`,则采样器打乱索引的顺序
# seed              当`shuffle=True`时采样器打乱使用的随机种子.此参数应在进程组的各进程中保持一致
# drop_last         若为`True`,则采样器将会丢弃末尾的样本以使样本平分到各进程;若为`False`,则采样器将会
#                   添加起始的样本以使样本平分到各进程
```

```python
>>> from torch.utils.data import DistributedSampler

>>> sampler0 = DistributedSampler(range(10), num_replicas=3, rank=0, shuffle=False, drop_last=True)
>>> sampler1 = DistributedSampler(range(10), num_replicas=3, rank=1, shuffle=False, drop_last=True)
>>> sampler2 = DistributedSampler(range(10), num_replicas=3, rank=2, shuffle=False, drop_last=True)
>>> list(sampler0)
[0, 3, 6]               # 丢弃索引为10的样本以使样本平分到各进程
>>> list(sampler1)
[1, 4, 7]
>>> list(sampler2)
[2, 5, 8]

>>> sampler0 = DistributedSampler(range(10), num_replicas=3, rank=0, shuffle=False, drop_last=False)
>>> sampler1 = DistributedSampler(range(10), num_replicas=3, rank=1, shuffle=False, drop_last=False)
>>> sampler2 = DistributedSampler(range(10), num_replicas=3, rank=2, shuffle=False, drop_last=False)
>>> list(sampler0)
[0, 3, 6, 9]
>>> list(sampler1)
[1, 4, 7, 0]            # 添加索引为0,1的样本以使样本平分到各进程
>>> list(sampler2)
[2, 5, 8, 1]

>>> sampler0 = DistributedSampler(range(10), num_replicas=3, rank=0, shuffle=True, drop_last=True)
>>> sampler1 = DistributedSampler(range(10), num_replicas=3, rank=1, shuffle=True, drop_last=True)
>>> sampler2 = DistributedSampler(range(10), num_replicas=3, rank=2, shuffle=True, drop_last=True)
>>> list(sampler0)      # 随机划分,默认随机种子为`0`因此每次划分的结果相同
[4, 5, 0]
>>> list(sampler1)
[1, 3, 8]
>>> list(sampler2)
[7, 9, 6]
```



### set_epoch()

此方法在每个 epoch 开始、创建 `DataLoader` 迭代器之前调用，以使得每个 epoch 被打乱的顺序不同。

```python
>>> sampler = DistributedSampler(dataset) if is_distributed else None
>>> loader = DataLoader(dataset, shuffle=(sampler is None),
...                     sampler=sampler)
>>> for epoch in range(start_epoch, n_epochs):
...     if is_distributed:
...         sampler.set_epoch(epoch)
...     train(loader)
```

```python
>>> sampler0 = DistributedSampler(range(10), num_replicas=3, rank=0, shuffle=True, drop_last=True)
>>> sampler1 = DistributedSampler(range(10), num_replicas=3, rank=1, shuffle=True, drop_last=True)
>>> sampler2 = DistributedSampler(range(10), num_replicas=3, rank=2, shuffle=True, drop_last=True)
>>> list(sampler0)
[4, 5, 0]
>>> list(sampler1)
[1, 3, 8]
>>> list(sampler2)
[7, 9, 6]

>>> sampler0.set_epoch(0)   # 相当于设置随机种子为`0`
>>> list(sampler0)
[4, 5, 0]
>>> list(sampler1)
[1, 3, 8]
>>> list(sampler2)
[7, 9, 6]

>>> sampler0.set_epoch(1)
>>> list(sampler0)
[5, 2, 9]
>>> list(sampler1)
[1, 3, 8]
>>> list(sampler2)
[7, 9, 6]
```



## get_worker_info()





## IterableDataset

迭代数据集。

所有迭代数据集应继承此类。当数据来源于一个流时，这种形式的数据集尤为有用。

所有子类应覆写 `__iter__()` 方法，用于返回数据集的样本的一个迭代器。

当 `Dataloader` 使用迭代数据集时，`Dataloader` 的迭代器会产生数据集的每一个样本。当 `num_worker>0` 时，每一个工作进程都会有数据集对象的一份单独的副本



```python
# 自定义迭代数据集
>>> class MyIterableDataset(torch.utils.data.IterableDataset):
     def __init__(self, start, end):
         super(MyIterableDataset).__init__()
         assert end > start, "this example code only works with end > start"
         self.start = start
         self.end = end

     def __iter__(self):
         worker_info = torch.utils.data.get_worker_info()
         if worker_info is None:        # 主进程读取数据,返回完整的迭代器
             iter_start = self.start
             iter_end = self.end
         else:                          # 工作进程读取数据,划分数据集并返回相应子集的迭代器
             per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
             worker_id = worker_info.id
             iter_start = self.start + worker_id * per_worker
             iter_end = min(iter_start + per_worker, self.end)
         return iter(range(iter_start, iter_end))    # `Dataloader`调用数据集实例的`__iter__()`方法,
                                                     # 使用其返回的迭代器

>>> ds = MyIterableDataset(start=1, end=11)     # range(1, 11) as dataset  


>>> list(torch.utils.data.DataLoader(ds, num_workers=0))   # 主进程读取数据
[tensor([1]), tensor([2]), tensor([3]), tensor([4]), tensor([5]), tensor([6]), tensor([7]), tensor([8]), tensor([9]), tensor([10])]                 # 包装为`torch.tensor`
>>> list(torch.utils.data.DataLoader(ds, num_workers=2))   # 2个工作进程读取数据
(error)
>>> list(torch.utils.data.DataLoader(ds, num_workers=8))   # 更多工作进程读取数据
(error)
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
>>> list(sampler)
[0, 1, 2, 3, 4]
```



## Subset

数据集的指定索引的样本构成的子集。

```python
torch.utils.data.Subset(dataset, indices)
# dataset      数据集
# indices      指定索引
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





# torchvision

## datasets

所有的数据集都是 `torch.utils.data.Dataset` 的子类，即实现了 `__getitem__` 和 `__len__` 方法，因此它们可以被传入到一个 `torch.utils.data.DataLoader` 对象。`DataLoader` 可以使用 `torch.multiprocessing` 并行加载多个样本。例如：

```python
imagenet_data = torchvision.datasets.MNIST('path/to/imagenet_root/')
data_loader = torch.utils.data.DataLoader(imagenet_data,
                                          batch_size=4,
                                          shuffle=True,
                                          num_workers=args.nThreads)
```



### CIFAR10

CIFAR10 数据集。

```python
torchvision.datasets.CIFAR10(root: str, train: bool = True, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False)
# root         同`torchvision.datasets.MNIST`
# ...
```



### FashionMNIST

Fashion-MNIST 数据集。

```python
torchvision.datasets.FashionMNIST(root: str, train: bool = True, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False)
# root         同`torchvision.datasets.MNIST`
# ...
```



### MNIST

MNIST 数据集。

```python
torchvision.datasets.MNIST(root: str, train: bool = True, transform: Union[Callable, NoneType] = None, target_transform: Union[Callable, NoneType] = None, download: bool = False) → None
# root         数据集的根目录的路径
# train        若为`True`,使用训练集,否则使用测试集
# transform    对图片数据应用的变换函数
# download     若为`True`,在线下载数据集并将其放置在根目录下.若数据集已经下载则不再重复下载
```

```python
>>> trainset = torchvision.datasets.MNIST('./data',             # 加载MNIST数据集的训练数据
                                          download=True,
                                          train=True,
                                          transform=None)
>>> trainset.data.shape
torch.Size([60000, 28, 28])    # 60000个单通道28×28图像
>>> trainset[0]
(<PIL.Image.Image image mode=L size=28x28 at 0x19B9D2040>, 5)   # 图像为`PIL.Image.Image`实例,标签为5
>>> trainset[0][0],show()      # 展示图像
```

```python
>>> transform = transforms.Compose(
    [transforms.ToTensor()])                                    # 转换图像为张量
>>> trainset = torchvision.datasets.MNIST('./data',             # 加载MNIST数据集的训练数据
                                          download=True,
                                          train=True,
                                          transform=None)
>>> trainset.data.shape
torch.Size([60000, 28, 28])    # 60000个单通道28×28图像
>>> trainset[0]
(tensor([[[0.0000, 0.0000, ..., 0.0000],                        # 图像为张量,标签为5
					...
          [0.0000, 0.0000, ..., 0.0000]]]),
 5)
```



## transforms





## utils

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

print(images.shape)    # torch.Size([32, 1, 28, 28]): 32个单通道28×28图像
```



### make_grid()

对一组图像进行排列以供展示。

```python
torchvision.utils.make_grid(tensor: Union[torch.Tensor, List[torch.Tensor]], nrow: int = 8, padding: int = 2, normalize: bool = False, value_range: Optional[Tuple[int, int]] = None, scale_each: bool = False, pad_value: int = 0, **kwargs) → torch.Tensor
# tensor       代表一个批次的图像的4维张量(批次规模×通道数×高×宽),或由相同大小的图像组成的列表
# nrow         每一行排列的图像数量
# padding      图像边框的填充宽度
# normalize    若为`True`,将图像归一化到(0, 1)区间
# value_range
# scale_each
# pad_value    图像边框的填充值
```

```python
img_grid = torchvision.utils.make_grid(images)

print(img_grid.shape)   # torch.Size([3, 122, 242]): 三通道,32个图像排列为4行8列,故高为4×28+5×2=122,
                        #                            宽为8×28+9×2=242
```



### save_image()

保存指定张量为图像文件。

```python
torchvision.utils.save_image(tensor: Union[torch.Tensor, List[torch.Tensor]], fp: Union[str, pathlib.Path, BinaryIO], format: Optional[str] = None, **kwargs) → None
# tensor       要保存为图像的张量
# fp           文件名或文件对象
# format       文件格式.若`fp`为文件名且`format`为`None`,则文件格式从文件名的扩展名确定;若`fp`为文件对象,
#              则必须指定`format`
```

```python
torchvision.utils.save_image(img_grid, 'sample.jpg')
```

