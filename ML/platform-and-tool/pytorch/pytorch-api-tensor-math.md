[toc]

# torch

## 张量

### device

`torch.device` 实例代表张量被分配到的设备，其包含了设备类型（`'cpu'` 或 `'cuda'`）和可选的该类型的设备序号。如果设备序号没有指定，则默认为该类型的当前设备（由 `torch.cuda.current_device()` 给出）。

张量的设备可以通过 `device` 属性得到。

`torch.device` 实例可以通过字符串或者字符串加上设备序号来构造：

```python
>>> torch.device('cpu')
device(type='cpu')
>>> torch.device('cpu', 0)    # 0号CPU设备.多核CPU通常被视为一个设备,因此CPU通常只有0号设备.
device(type='cpu', index=0)
>>> torch.device('cuda:0')    # 0号CUDA设备
device(type='cuda', index=0)
>>> torch.device('cuda')      # 当前CUDA设备
device(type='cuda')
>>> torch.device('cuda', 0)
device(type='cuda', index=0)  # 0号CUDA设备
>>> torch.device(0)
device(type='cuda', index=0)  # 0号CUDA设备
```



### dtype

`torch.dtype` 实例代表张量的数据类型。PyTorch 有如下数据类型：

| Data type                 | dtype                                 | Legacy Constructors      |
| ------------------------- | ------------------------------------- | ------------------------ |
| 32-bit floating point     | `torch.float32` or `torch.float`      | `torch.*.FloatTensor`    |
| 64-bit floating point     | `torch.float64` or `torch.double`     | `torch.*.DoubleTensor`   |
| 64-bit complex            | `torch.complex64` or `torch.cfloat`   |                          |
| 128-bit complex           | `torch.complex128` or `torch.cdouble` |                          |
| 16-bit floating point [1] | `torch.float16` or `torch.half`       | `torch.*.HalfTensor`     |
| 16-bit floating point [2] | `torch.bfloat16`                      | `torch.*.BFloat16Tensor` |
| 8-bit integer (unsigned)  | `torch.uint8`                         | `torch.*.ByteTensor`     |
| 8-bit integer (signed)    | `torch.int8`                          | `torch.*.CharTensor`     |
| 16-bit integer (signed)   | `torch.int16` or `torch.short`        | `torch.*.ShortTensor`    |
| 32-bit integer (signed)   | `torch.int32` or `torch.int`          | `torch.*.IntTensor`      |
| 64-bit integer (signed)   | `torch.int64` or `torch.long`         | `torch.*.LongTensor`     |
| Boolean                   | `torch.bool`                          | `torch.*.BoolTensor`     |

[1]：有时被称为 binary16：使用一个符号位、5 个指数位和 10 个有效数字位，可以表示更大的精度。

[2]：有时被称为脑浮点：使用一个符号位、8 个指数位和 7 个有效数字位，可以表示更大的范围。



当参与数学运算的张量的数据类型不同时，我们将数据类型转换为满足以下规则的最小数据类型：

+ 若标量操作数的数据类型比张量操作数的等级更高（复数 > 浮点数 > 整数 > 布尔值），则转换为有足够大小能够容纳该类别操作数的类型。
+ 若零维张量操作数的数据类型比多维张量操作数的等级更高，则转换为有足够大小能够容纳该类别操作数的类型。
+ 若标量操作数/零维张量操作数/多维张量操作数的数据类型比其余标量操作数/零维张量操作数/多维张量操作数的等级更高或规模更大，则转换为有足够大小能够容纳该类别操作数的类型。

下面给出一些示例：

```python
# 零维张量
>>> long_zerodim = torch.tensor(1, dtype=torch.long)
>>> int_zerodim = torch.tensor(1, dtype=torch.int)
# 多维张量
>>> float_tensor = torch.ones(1, dtype=torch.float)
>>> double_tensor = torch.ones(1, dtype=torch.double)
>>> complex_float_tensor = torch.ones(1, dtype=torch.complex64)
>>> complex_double_tensor = torch.ones(1, dtype=torch.complex128)
>>> int_tensor = torch.ones(1, dtype=torch.int)
>>> long_tensor = torch.ones(1, dtype=torch.long)
>>> uint_tensor = torch.ones(1, dtype=torch.uint8)
>>> double_tensor = torch.ones(1, dtype=torch.double)
>>> bool_tensor = torch.ones(1, dtype=torch.bool)

# 整数标量操作数被推断为`torch.int64`类型,因此结果为`torch.int64`类型
>>> torch.add(5, 5).dtype
torch.int64
# 整数标量操作数被推断为`torch.int64`类型,不比张量操作数(`torch.int32`)的等级更高,
# 因此结果为张量操作数的`torch.int32`类型
>>> (int_tensor + 5).dtype
torch.int32
# 浮点数标量操作数被推断为`torch.float32`类型,比张量操作数(`torch.int32`)的等级更高,
# 因此结果为标量操作数的`torch.float32`类型
>>> (int_tensor + 5.).dtype
torch.float32
# 浮点数标量操作数(`torch.float32`)的数据类型比整数标量操作数(`torch.int32`)的等级更高,
# 因此结果为浮点数张量操作数的`torch.float32`类型
>>> torch.add(5, 5.).dtype
torch.float32
# 零维张量操作数(`torch.int64`)的数据类型不比多维张量操作数(`torch.int32`)的等级更高,
# 因此结果为多维张量操作数的`torch.int32`类型
>>> (int_tensor + long_zerodim).dtype
torch.int32
# 浮点数张量操作数(`torch.float32`)的数据类型比整数张量操作数(`torch.int64`)的等级更高,
# 因此结果为浮点数张量操作数的`torch.float32`类型
>>> torch.add(long_tensor, float_tensor).dtype
torch.float32
# 张量操作数(`torch.int64`)的数据类型比张量操作数(`torch.int32`)的规模更大,
# 因此结果为能够容纳两者的`torch.int64`类型.下同
>>> (long_tensor + int_tensor).dtype
torch.int64
>>> (float_tensor + double_tensor).dtype
torch.float64
>>> (complex_float_tensor + complex_double_tensor).dtype
torch.complex128
# 布尔类型可以转换为各种整数类型
>>> (bool_tensor + long_tensor).dtype
torch.int64
>>> (bool_tensor + uint_tensor).dtype
torch.uint8
>>> (bool_tensor + int_tensor).dtype
torch.int32
```



### Tensor

PyTorch 张量。



#### \_\_getitem\_\_(), \_\_setitem\_\_(), \_\_delitem\_\_()

```python
>>> x = torch.tensor([[1, 2, 3], [4, 5, 6]])
>>> x[1]
tensor([4, 5, 6])
>>> x[1][2]
tensor(6)
>>> x[0][1] = 8
>>> x
tensor([[1, 8, 3],
        [4, 5, 6]])
```



#### apply_()





#### backward()

以张量作为计算图的根节点启动反向计算。

```python
Tensor.backward(gradient=None, retain_graph=None, create_graph=False, inputs=None)
# gradient
# retain_graph
# create_graph
# inputs
```







#### bool()

等价于 `self.to(torch.bool)`。



#### cpu()

返回张量的一个位于内存中的副本。



#### cuda()

返回张量的一个位于显存中的副本。可以通过 `device` 参数指定 CUDA 设备，默认为当前 CUDA 设备。

如果张量已经位于当前 CUDA 设备的显存中，则直接返回该张量对象。



#### data



#### data_ptr()

返回张量的第一个元素的内存地址。



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



#### device

返回张量所位于的设备。

```python
>>> x = torch.tensor([1, 2, 3])
>>> x.device
device(type='cpu')
>>> x = torch.tensor([1, 2, 3], device='cuda:0')
>>> x.device
device(type='cuda', index=0)
```



#### dim()

返回张量的维数。



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



#### expand_as()



#### get_device()

对于 CUDA 张量，返回其所位于的 GPU 设备的序号；对于 CPU 张量，抛出一个错误。

```python
>>> x = torch.randn(3, 4, 5, device='cuda:0')
>>> x.get_device()
0
>>> x.cpu().get_device()
# RuntimeError: get_device is not implemented for type torch.FloatTensor
```



#### grad

调用 `loss.backward()` 后计算的损失对此张量的梯度值的累积。由于多次调用 `loss.backward()` 后梯度值会进行累加，因此需要在必要时调用 `optimizer.zero_grad()` 以清零。

```python
>>> w = torch.tensor([1.], requires_grad=True)  # 参数
>>> b = torch.tensor([1.], requires_grad=True)
>>> 
>>> x = torch.tensor([2.])  # 样本1
>>> y = torch.tensor([4.])
>>> z = w @ x + b
>>> l = (y - z)**2
>>> l.backward()            # 反向计算
>>> w.grad
tensor([-4.])
>>> b.grad
tensor([-2.])
>>> 
>>> x = torch.tensor([3.])  # 样本2
>>> y = torch.tensor([5.])
>>> z = w @ x + b
>>> l = (y - z)**2
>>> l.backward()
>>> w.grad
tensor([-10.])              # 梯度值累加
>>> b.grad
tensor([-4.])
>>> 
>>> optimizer = torch.optim.SGD([w, b], lr=1e-2)  # 创建更新参数的优化器
>>> optimizer.step()
>>> w
tensor([1.1000], requires_grad=True)              # 参数被更新
>>> b
tensor([1.0400], requires_grad=True)
>>> optimizer.zero_grad()   # 清零所有参数的累积梯度值
>>> w.grad
tensor([0.])                # 归零
>>> b.grad
tensor([0.])
```



#### grad_fn



#### is_cuda()

若张量保存在 GPU（显存）上，则返回 `True`。



#### is_leaf()

若张量是计算图中的叶节点，则返回 `True`。



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

`new_full()` 返回一个指定形状和所有元素值的张量，并且该张量与调用对象有相同的 `torch.dtype` 和 `torch.device`。

```python
>>> a = torch.randn((2,), dtype=torch.float64)
>>> a.new_full((2, 3), 3.141592)
tensor([[ 3.1416,  3.1416,  3.1416],
        [ 3.1416,  3.1416,  3.1416]], dtype=torch.float64)
```

`new_ones()`/`new_zeros()` 返回一个指定形状的全 1/全 0 张量，并且该张量与调用对象有相同的 `torch.dtype` 和 `torch.device`。

```python
>>> a = torch.randn((2,), dtype=torch.float64)
>>> a.new_ones((2,3))
tensor([[1., 1., 1.],
        [1., 1., 1.]], dtype=torch.float64)
>>> a.new_zeros((2,3))
tensor([[0., 0., 0.],
        [0., 0., 0.]], dtype=torch.float64)
```



#### numpy()

将张量作为 `numpy.ndarray` 实例返回。

```python
>>> a = torch.randn(4)
>>> a
tensor([ 0.7129, -1.6347,  0.4912, -2.3418])
>>> a.numpy()
array([ 0.7128906 , -1.6347297 ,  0.49121562, -2.3418238 ], dtype=float32)
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



#### requires_grad

若需要计算对此张量的梯度，返回 `True`。



#### requires_grad_()

设置是否需要计算对此张量的梯度。

```python
>>> w = torch.randn(4)
>>> w
tensor([-0.1482, -0.2680,  1.4278,  1.7212])
>>> w.requires_grad_()          # 相当于 w.requires_grad = True
tensor([-0.1482, -0.2680,  1.4278,  1.7212], requires_grad=True)
>>> w.requires_grad_(False)     # 相当于 w.requires_grad = False
tensor([-0.1482, -0.2680,  1.4278,  1.7212])
```



#### retain_grad()

对于计算图中需要计算梯度但不是叶节点的张量，启用 `grad` 属性。

```python
>>> w = torch.tensor([1.], requires_grad=True)
>>> b = torch.tensor([1.], requires_grad=True)
>>> x = torch.tensor([2.])
>>> y = torch.tensor([4.])
>>> z = w @ x + b
>>> z.retain_grad()
>>> l = (y - z) ** 2
>>> l.backward()
>>> z.grad
tensor([-2.])    # 保留了对此张量的梯度值
```



#### scatter()



#### select()



#### size()

返回张量的形状。



#### T

反转张量的所有维度。

```python
>>> a = torch.randn(3, 4, 5)
>>> a.T.shape
torch.Size([5, 4, 3])
```



#### to()

更改张量的 `dtype` 或 `device` 属性。

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



#### type()

以字符串的形式返回张量的数据类型，或将张量转换为特定数据类型。

如果目标数据类型就是当前数据类型，则直接返回当前张量实例，否则创建新的张量实例并返回。

```python
>>> a = torch.arange(5)
>>> a
tensor([0, 1, 2, 3, 4])
>>> a.type()
'torch.LongTensor'
>>> a.type(torch.float32)
tensor([0., 1., 2., 3., 4.])
```



#### view()





#### zero_()

用 0 填充张量。



## 张量类型操作

### get_default_dtype()

返回默认的浮点类型。

```python
>>> torch.get_default_dtype()
torch.float32
>>> torch.set_default_dtype(torch.float64)
>>> torch.get_default_dtype()
torch.float64
```



### is_complex()

若输入张量的数据类型是复类型，即 `torch.complex64`、`torch.complex128` 两者之一，则返回 `True`。

```python
>>> x = torch.tensor([1.+2j, 2, 3])
>>> torch.is_complex(x)
True
```



### isfinite()



### is_floating_point()

若输入张量的数据类型是浮点类型，即 `torch.float64`、`torch.float32`、`torch.float16` 和 `torch.bfloat16` 其中之一，则返回 `True`。

```python
>>> x = torch.tensor([1., 2, 3])
>>> torch.is_tensor(x)
True
```



### isinf()



### isnan()



### is_tensor()

若实例是 PyTorch 张量，则返回 `True`。

```python
>>> x = torch.tensor([1, 2, 3])
>>> torch.is_tensor(x)
True
```



### set_default_dtype()

设置默认的浮点类型。

```python
>>> torch.set_default_dtype(torch.float32)
>>> torch.tensor([1.2, 3]).dtype    # Python浮点数推断为`torch.float32`类型
torch.float32
>>> torch.tensor([1.2, 3j]).dtype   # Python复数推断为`torch.complex64`类型
torch.complex64
>>> torch.set_default_dtype(torch.float64)
>>> torch.tensor([1.2, 3]).dtype    # Python浮点数推断为`torch.float64`类型
torch.float64
>>> torch.tensor([1.2, 3j]).dtype   # Python复数推断为`torch.complex128`类型
torch.complex128
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



### tensor()

由数据创建张量。

```python
torch.tensor(data, *, dtype=None, device=None, requires_grad=False, pin_memory=False) -> Tensor
# data           张量的初始数据,可以是数字,列表,元组,`numpy.ndarray`实例等.
# dtype          张量的数据类型,默认从`data`中推断(推断的结果一定是CPU数据类型)
# device         张量位于的设备,默认使用数据类型相应的设备:若为CPU数据类型,则使用CPU;若为CUDA数据类型,
#                则使用当前的CUDA设备
# requires_grad  若为`True`,则autograd应记录此张量参与的运算
# pin_memory     若为`True`,则张量将被分配到锁页内存中.仅对位于CPU的张量有效
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
>>> torch.zeros(2, 3, dtype=int)
tensor([[0, 0, 0],
        [0, 0, 0]])
```



## 张量操作

> 以下函数均不是原位操作，即返回一个新的张量，而不改变原张量。

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



### chunk()





### index_select()

选择张量沿指定轴的若干索引，返回相应的子张量。

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



### flip()



### fliplr(), flipud()





### index_select()



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



### narrow()



### nonezero()



### permute()

重新排序张量的各个维度。

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



### roll()



### rot90()

旋转多维张量 90° 若干次，在指定的两个轴所决定的平面内。

```python
torch.rot90(input, k, dims) → Tensor
# input   要旋转的多维张量
# k       旋转90°的次数.若大于0,则从第一个轴旋转至第二个轴;若小于0,则从第二个轴旋转至第一个轴
# dims    指定的两个轴
```

```python
>>> a = torch.arange(4).view(2, 2)
>>> a
tensor([[0, 1],
        [2, 3]])
>>> a.rot90(1, [0, 1])
tensor([[1, 3],
        [0, 2]])
>>> a.rot90(2, [0, 1])
tensor([[3, 2],
        [1, 0]])
>>> 
>>> a = torch.arange(8).view(2, 2, 2)
>>> a
tensor([[[0, 1],
         [2, 3]],

        [[4, 5],
         [6, 7]]])
>>> a.rot90(1, [0, 1])
tensor([[[2, 3],
         [6, 7]],

        [[0, 1],
         [4, 5]]])
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

移除张量的规模为 1 的维度。返回张量与输入张量共享内存。

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

重复张量的元素。

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

交换张量的指定两个维度。

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

在张量的指定位置插入一个规模为 1 的维度。返回张量与输入张量共享内存。

```python
>>> a = torch.randn(2, 3, 4)
>>> torch.unsqueeze(a, 0).shape
torch.Size([1, 2, 3, 4])
>>> a.unsqueeze(3).shape
torch.Size([2, 3, 4, 1])
```





## 数学运算

> 下列所有数学运算函数都是 `torch.Tensor` 方法，即张量可以调用下列函数的同名方法，相当于将张量自身作为函数的第一个张量参数。张量的这些方法同时有非原位操作和原位操作两个版本，后者的方法名增加了后缀 `_`，例如：
>
> ```python
> >>> a = torch.tensor([-1, -2, 3])
> >>> a.abs()
> tensor([1, 2, 3])      # 返回新张量
> >>> a
> tensor([-1, -2,  3])   # 原张量不变
> >>> 
> >>> a = torch.tensor([-1, -2, 3])
> >>> a.abs_()
> tensor([1, 2, 3])      # 返回原张量
> >>> a
> tensor([1, 2, 3])      # 原张量被修改
> ```



### abs()

对张量的所有元素应用绝对值函数。

```python
>>> a = torch.randn(4)
>>> a
tensor([ 0.2736, -1.2038,  0.9149, -0.2633])
>>> a.abs()
tensor([0.2736, 1.2038, 0.9149, 0.2633])
```



### add(), sub()

张量加法/减法。符号 `+, -` 重载了这些方法。

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





### amax(), amin()

返回张量沿指定维度的最大值/最小值。

```python
>>> a = torch.randn(4)
>>> a
tensor([-0.2656,  0.5158, -0.9120,  0.1984])
>>> a.amax()
tensor(0.5158)
>>> a.amin()
tensor(-0.9120)

>>> a = torch.randn(4, 4)
>>> a
tensor([[-0.3866, -0.6860,  1.6496, -1.7280],
        [-0.2662,  1.0654, -0.1922,  0.4900],
        [ 1.0274, -1.1158, -2.4285, -0.1089],
        [ 0.1412,  1.2908,  0.7853, -0.9393]])
>>> a.amax(1)             # 沿轴1的最大值
tensor([1.6496, 1.0654, 1.0274, 1.2908])
>>> a.amax((0, 1))        # 沿轴0和轴1的最大值
tensor(1.6496)
>>> a.amin(1)
tensor([-1.7280, -0.2662, -2.4285, -0.9393])
>>> a.amin((0, 1))
tensor(-2.4285)
```

> `amax()`/`amin()` 和 `max()`/`min()` 的区别在于：
>
> + `amax()`/`amin()` 支持沿多个维度归约
> + `amax()`/`amin()` 不返回索引
> + `amax()`/`amin()` 将梯度均摊到多个最大值上，而 `max()`/`min()` 只将梯度传播给某一个最大值



### angle()

对张量的所有元素（复数）计算角度（以弧度为单位）。

```python
>>> c = torch.tensor([1 + 1j, -2 + 2j, 3 - 3j])
>>> c.angle()
tensor([ 0.7854,  2.3562, -0.7854])
>>> c.angle().rad2deg()
tensor([ 45., 135., -45.])
```



### argmax(), argmin()

返回张量沿指定维度的最大值/最小值的索引。

```python
>>> a = torch.randn(4)
>>> a
tensor([-1.6751, -0.7609,  0.8919, -0.0545])
>>> a.argmax()
tensor(2)
>>> a.argmin()
tensor(0)

>>> a = torch.randn(4, 4)
>>> a
tensor([[ 1.3398,  0.2663, -0.2686,  0.2450],
        [-0.7401, -0.8805, -0.3402, -1.1936],
        [ 0.4907, -1.3948, -1.0691, -0.3132],
        [-1.6092,  0.5419, -0.2993,  0.3195]])
>>> a.argmax(1)             # 沿轴1的最大值的索引
tensor([ 0,  2,  0,  1])
>>> a.argmin(1)
tensor([ 2,  3,  1,  0])
```



### argsort()



### bincount()



### bmm()

批量矩阵乘法。

```python
>>> m1 = torch.randn(10, 3, 4)
>>> m2 = torch.randn(10, 4, 5)
>>> torch.bmm(m1, m2).size()     # 相同索引的矩阵对应相乘
torch.Size([10, 3, 5])
```



### ceil()

对张量的所有元素应用向上取整函数。

```python
>>> a = torch.randn(4)
>>> a
tensor([-0.6341, -1.4208, -1.0900,  0.5826])
>>> a.ceil()
tensor([-0., -1., -1.,  1.])
```



### clamp()

对张量的所有元素应用下限和上限。

```python
>>> a = torch.randn(4)
>>> a
tensor([-1.7120,  0.1734, -0.0478, -0.0922])
>>> a.clamp(min=-0.5, max=0.5)
tensor([-0.5000,  0.1734, -0.0478, -0.0922])

>>> a = torch.randn(4)
>>> a
tensor([-0.0299, -2.3184,  2.1593, -0.8883])
>>> a.clamp_(min=0)       # ReLU激活函数
tensor([0.0000, 1.0605, 0.0000, 0.5617])
```



### conj()





### count_nonzero()

计数张量沿指定轴的非零值。若没有指定轴，则计数张量的所有非零值。

```python
>>> a = torch.zeros(3, 3)
>>> a[torch.randn(3, 3) > 0.5] = 1
>>> a
tensor([[0., 1., 1.],
        [0., 0., 0.],
        [0., 0., 1.]])
>>> x.count_nonzero(0)   # 沿轴0
tensor([0, 1, 2])
>>> x.count_nonzero(1)   # 沿轴1
tensor([2, 0, 1])
>>> a.count_nonzero()    # 总数
tensor(3)
```



### cross()



### cummax(), cummin()





### cumsum(), cumprod()





### deg2rad(), rad2deg()

对张量的所有元素从角度值/弧度值转换为弧度值/角度值。

```python
>>> c = torch.tensor([1 + 1j, -2 + 2j, 3 - 3j])
>>> c.angle()
tensor([ 0.7854,  2.3562, -0.7854])
>>> c.angle().rad2deg()
tensor([ 45., 135., -45.])
```



### det()





### diag()





### diff()





### dot()

计算两个向量（一维张量）的点积。

```python
>>> a = torch.tensor([2, 3])
>>> b = torch.tensor([2, 1])
>>> a.dot(b)
tensor(7)
```



### eig()







### equal()

判断两个张量是否完全相等。

```python
>>> one1 = torch.ones(2,3)
>>> one2 = torch.ones(2,3)
>>> one1.equal(one2)
True
```



### exp()

对张量的所有元素应用自然指数函数。

```python
>>> a = torch.arange(10.).view(2,5)
>>> a
tensor([[0., 1., 2., 3., 4.],
        [5., 6., 7., 8., 9.]])
>>> a.exp()
tensor([[1.0000e+00, 2.7183e+00, 7.3891e+00, 2.0086e+01, 5.4598e+01],
        [1.4841e+02, 4.0343e+02, 1.0966e+03, 2.9810e+03, 8.1031e+03]])
```



### floor()

对张量的所有元素应用向下取整函数。

```python
>>> a = torch.randn(4)
>>> a
tensor([-0.8166,  1.5308, -0.2530, -0.2091])
>>> a.floor()
tensor([-1.,  1., -1., -1.])
```



### fmax(), fmin()



### frac()



### gcd(), lcm()

张量逐元素计算最大公约数/最小公倍数。

```python
>>> a = torch.tensor([5, 10, 15])
>>> b = torch.tensor([3, 4, 5])
>>> torch.gcd(a, b)
tensor([1, 2, 5])
>>> torch.lcm(a, b)
tensor([15, 20, 15])
>>> 
>>> c = torch.tensor([3])
>>> torch.gcd(a, c)              # 扩张的逐元素计算
tensor([1, 1, 3])
>>> torch.lcm(a, c)
tensor([[10, 10, 15]])
```



### gt(), ge(), eq(), le(), lt(), ne()

张量逐元素比较。符号 `>, >=, ==, <=, <, !=` 重载了这些方法。

```python
>>> a = torch.randint(4, (3, 3))
>>> b = torch.randint(4, (3, 3))
>>> a 
tensor([[2, 3, 0],
        [2, 1, 2],
        [3, 3, 2]])
>>> b
tensor([[2, 2, 2],
        [0, 1, 0],
        [2, 1, 3]])
>>> a > b
tensor([[False,  True, False],   # 相同形状的布尔类型张量
        [ True, False,  True],
        [ True,  True, False]])
>>> a >= b
tensor([[ True,  True, False],
        [ True,  True,  True],
        [ True,  True, False]])
>>> a == b
tensor([[ True, False, False],
        [False,  True, False],
        [False, False, False]])
>>> a <= b
tensor([[ True, False,  True],
        [False,  True, False],
        [False, False,  True]])
>>> a < b
tensor([[False, False,  True],
        [False, False, False],
        [False, False,  True]])
>>> a != b
tensor([[False,  True,  True],
        [ True, False,  True],
        [ True,  True,  True]])
>>> a[a < b] = -1                # 可作为张量索引
>>> a
tensor([[ 2,  3, -1],
        [ 2,  1,  2],
        [ 3,  3, -1]])
>>> 
>>> c = torch.randint(4, (1, 3))
>>> c
tensor([[2, 1, 3]])
>>> b == c                       # 扩张的逐元素比较
tensor([[ True, False, False],
        [False,  True, False],
        [ True,  True,  True]])
```



### hypot()

张量逐元素计算给定两直角边长的斜边长度。

```python
>>> a = torch.tensor([4.0])
>>> b = torch.tensor([3.0, 4.0, 5.0])
>>> a.hypot(b)                   # 扩张的逐元素计算
tensor([5.0000, 5.6569, 6.4031])
```



### inner()



### lerp()



### log(), log10(), log2()

对张量的所有元素应用对数函数。

```python
>>> a
tensor([[0., 1., 2., 3., 4.],
        [5., 6., 7., 8., 9.]])
>>> a.log()
tensor([[  -inf, 0.0000, 0.6931, 1.0986, 1.3863],
        [1.6094, 1.7918, 1.9459, 2.0794, 2.1972]])
>>> a.log2()
tensor([[  -inf, 0.0000, 1.0000, 1.5850, 2.0000],
        [2.3219, 2.5850, 2.8074, 3.0000, 3.1699]])
>>> a.log10()
tensor([[  -inf, 0.0000, 0.3010, 0.4771, 0.6021],
        [0.6990, 0.7782, 0.8451, 0.9031, 0.9542]])
```



### logaddexp(), logaddexp2(), logsumexp()





### logit()





### lu()



### matmul()

张量乘法。符号 `@` 重载了此方法。

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

# 矩阵序列×向量序列: 逐元素的矩阵乘法不适用,会被识别为矩阵序列×矩阵
# 须将向量序列扩展为矩阵序列
```



### max(), min(), mean(), std()

返回张量元素的统计量。

```python
>>> a = torch.arange(10.).view(2, 5)
>>> a
tensor([[0., 1., 2., 3., 4.],
        [5., 6., 7., 8., 9.]])

>>> a.max()
tensor(9.)
>>> a.max(0)
torch.return_types.max(
values=tensor([5., 6., 7., 8., 9.]),  # 沿轴0的最大值
indices=tensor([1, 1, 1, 1, 1]))      # 沿轴0的索引
>>> a.max(1)
torch.return_types.max(
values=tensor([4., 9.]),
indices=tensor([4, 4]))

>>> a.mean()
tensor(4.5000)
>>> a.mean(0)
tensor([2.5000, 3.5000, 4.5000, 5.5000, 6.5000])
>>> a.mean(1)
tensor([2., 7.])

>>> a.std(unbiased=False)
tensor(2.8723)
>>> a.std(0, unbiased=False)
tensor([2.5000, 2.5000, 2.5000, 2.5000, 2.5000])
>>> a.std(1, unbiased=False)
tensor([1.4142, 1.4142])

# 统计学的标准差
>>> a.std()
tensor(3.0277)
>>> a.std(0)
tensor([3.5355, 3.5355, 3.5355, 3.5355, 3.5355])
>>> a.std(1)
tensor([1.5811, 1.5811])
```



### mm()

矩阵乘法。

```python
>>> m1 = torch.randn(1, 3)
>>> m2 = torch.randn(3, 1)
>>> m1.mm(m2)
tensor([[0.9749]])
```



### mul(), div(), fmod(), pow()

张量逐元素乘法/除法（）/除法取余/乘方。符号 `*, /, %, **` 重载了这些方法。

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

div



### multinomial()



### neg()

对张量的所有元素取相反数。

```python
>>> a
tensor([ 0.1498,  1.7276,  0.8081, -0.0058])
>>> a.neg()
tensor([-0.1498, -1.7276, -0.8081,  0.0058])
```





### outer()



### quantile()





### real(), imag()



### reciprocal()

对张量的所有元素求倒数。

```python
>>> a = torch.arange(1, 6)
>>> a
tensor([ 1,  2,  3,  4,  5])
>>> a.reciprocal()
tensor([1.0000, 0.5000, 0.3333, 0.2500, 0.2000])
```



### round()



### sigmoid()

Sigmoid 激活函数。见 `torch.nn.Sigmoid`。

```python
>>> a = torch.randn(4)
>>> a
tensor([ 1.3938, -0.2393, -0.6540, -1.2838])
>>> a.sigmoid()
tensor([0.8012, 0.4405, 0.3421, 0.2169])
```



### sign()

对张量的所有元素应用符号函数。

```python
>>> a = torch.randn(4)
>>> a
tensor([ 0.8365, -0.8114,  0.6972, -0.8606])
>>> a.sign()
tensor([ 1., -1.,  1., -1.])
```



### sin(), cos(), tan(), arcsin(), arccos(), arctan(), sinh(), cosh(), tanh(), arcsinh(), arccosh(), arctanh()

对张量的所有元素应用三角函数和双曲函数。

```python
>>> a = torch.arange(10.).view(2, 5)
>>> a
tensor([[0., 1., 2., 3., 4.],
        [5., 6., 7., 8., 9.]])
>>> a.sin()
tensor([[ 0.0000,  0.8415,  0.9093,  0.1411, -0.7568],
        [-0.9589, -0.2794,  0.6570,  0.9894,  0.4121]])
```



### sort()





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

```python
>>> a = torch.randn(4)
>>> a
tensor([ 0.8848, -1.7775,  0.4125, -1.0188])
>>> a.square()
tensor([0.7829, 3.1595, 0.1702, 1.0380])
```



### sum(), prod()

对张量的所有元素求和/求积。

```python
>>> a = torch.arange(1, 6)
>>> a
tensor([ 1,  2,  3,  4,  5])
>>> a.sum()
tensor(15)
>>> a.prod()
tensor(120)
```



### tanh()

tanh 激活函数。

```python
>>> a = torch.randn(4)
>>> a
tensor([-1.2798,  2.2348,  0.2324, -1.9393])
>>> a.tanh()
tensor([-0.8564,  0.9774,  0.2283, -0.9595])
```



### topk()



### trace()

返回二维张量的对角线元素的和。

```python
>>> a = torch.arange(1., 10.).view(3, 3)
>>> a
tensor([[ 1.,  2.,  3.],
        [ 4.,  5.,  6.],
        [ 7.,  8.,  9.]])
>>> torch.trace(a)
tensor(15.)
```



### trunk()



### vdot()

计算两个向量（一维张量）的点积。`vdot()` 和 `dot()` 的区别在于，计算复向量的点积时，前者会对第一个参数取共轭，而后者不会。

```python
>>> a = torch.tensor([2, 3])
>>> b = torch.tensor([2, 1])
>>> a.vdot(b)
tensor(7)
>>> 
>>> z1 = torch.tensor((1 +2j, 3 - 1j))
>>> z2 = torch.tensor((2 +1j, 4 - 0j))
>>> z1.vdot(z2)
tensor([16.+1.j])
>>> z2.vdot(z1)
tensor([16.-1.j])
```







## 逻辑运算

### bitwise_and(), bitwise_or(), bitwise_xor(), bitwise_not()





### logical_and(), logical_or(), logical_xor(), logical_not()



## 其它运算

### numel()

返回张量的元素数量。

```python
>>> a = torch.randn(1, 2, 3, 4, 5)
>>> a.numel()
120
```





## 随机数

### manual_seed()

设置产生随机数的种子。

```python
>>> torch.manual_seed(1)
```



### rand()

返回指定形状的随机张量，其中每个元素服从 $$(0, 1)$$ 区间的均匀分布。

```python
>>> torch.rand(2, 3)
tensor([[ 0.8237,  0.5781,  0.6879],
        [ 0.3816,  0.7249,  0.0998]])
```



### randint()

返回指定形状的随机张量，其中每个元素等可能地取到 $$[{\rm low}, {\rm high})$$ 区间内的各个整数。

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





# torch.view

view 相当于 numpy 中的 resize 功能，即改变张量的形状。

```python
>>> a = torch.arange(12)
>>> b = a.view(2, 6)
>>> c = b.view(1, -1)   # -1位置的参数会根据元素总数和其它维度的长度计算
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





# torch.special

> 下列所有函数都是 `torch.Tensor` 方法，即张量可以调用下列函数的同名方法，相当于将张量自身作为函数的第一个张量参数。



## erf()

计算张量的误差函数。误差函数定义如下：
$$
{\rm erf}(x)=\frac{2}{\sqrt{\pi}}\int_0^xe^{-t^2}{\rm d}t
$$

```python
>>> a = torch.arange(0, 3, 0.5)
>>> a
tensor([0.0000, 0.5000, 1.0000, 1.5000, 2.0000, 2.5000])
>>> a.erf()
tensor([0.0000, 0.5205, 0.8427, 0.9661, 0.9953, 0.9996])
```



## erfc()

计算张量的互补误差函数。互补误差函数定义如下：
$$
{\rm erfc}(x)=1-\frac{2}{\sqrt{\pi}}\int_0^xe^{-t^2}{\rm d}t
$$

```python
>>> a = torch.arange(0, 3, 0.5)
>>> a
tensor([0.0000, 0.5000, 1.0000, 1.5000, 2.0000, 2.5000])
>>> a.erfc()
tensor([1.0000e+00, 4.7950e-01, 1.5730e-01, 3.3895e-02, 4.6777e-03, 4.0695e-04])
```



## exp2()

对张量的所有元素应用以 2 为底的指数函数。

```python
>>> a = torch.arange(10.).view(2,5)
>>> a
tensor([[0., 1., 2., 3., 4.],
        [5., 6., 7., 8., 9.]])
>>> a.exp2()
tensor([[  1.,   2.,   4.,   8.,  16.],
        [ 32.,  64., 128., 256., 512.]])
```



## logit()

对张量的所有元素应用 logit 函数。logit 函数定义如下：
$$
{\rm logit}(p)=\log (\frac{p}{1-p}),\quad p\in(0,1)
$$

```python
>>> a = torch.rand(4)
>>> a
tensor([0.4785, 0.7844, 0.9248, 0.1627])
>>> a.logit()
tensor([-0.0861,  1.2912,  2.5101, -1.6382])
```

