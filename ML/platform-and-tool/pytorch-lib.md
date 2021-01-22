[toc]

# torch

## abs

对张量的所有元素应用绝对值函数。亦为`torch.Tensor`方法。

```python
>>> torch.abs(torch.tensor([-1, -2, 3]))
tensor([1,  2,  3])
>>> torch.tensor([-1, -2, 3]).abs()
tensor([1, 2, 3])
```



## add, sub

张量加法/减法。亦为`torch.Tensor`方法。`+, -`符号重载了此方法。

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



## arange

生成包含指定等差数列的一维张量。

```python
>>> torch.arange(5)
tensor([ 0,  1,  2,  3,  4])
>>> torch.arange(1, 4)
tensor([ 1,  2,  3])
>>> torch.arange(1, 2.5, 0.5)
tensor([ 1.0000,  1.5000,  2.0000])
```



## bmm

批量矩阵乘法。

```python
>>> mat1 = torch.randn(10, 3, 4)
>>> mat2 = torch.randn(10, 4, 5)
>>> res = torch.bmm(mat1, mat2)    # 相同索引的矩阵对应相乘
>>> res.size()
torch.Size([10, 3, 5])
```



## cat

拼接张量。

```python
>>> x = torch.randn(2, 3)
>>> x
tensor([[ 0.6580, -1.0969, -0.4614],
        [-0.1034, -0.5790,  0.1497]])
>>> torch.cat((x, x, x), 0)
tensor([[ 0.6580, -1.0969, -0.4614],
        [-0.1034, -0.5790,  0.1497],
        [ 0.6580, -1.0969, -0.4614],
        [-0.1034, -0.5790,  0.1497],
        [ 0.6580, -1.0969, -0.4614],
        [-0.1034, -0.5790,  0.1497]])
>>> torch.cat((x, x, x), 1)
tensor([[ 0.6580, -1.0969, -0.4614,  0.6580, -1.0969, -0.4614,  0.6580,
         -1.0969, -0.4614],
        [-0.1034, -0.5790,  0.1497, -0.1034, -0.5790,  0.1497, -0.1034,
         -0.5790,  0.1497]])
```



## clamp

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

`torch.clamp(x, min=0)`即为 ReLU 激活函数。



## equal

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



## exp

对张量的所有元素应用指数函数。亦为`torch.Tensor`方法。

```python
>>> t
tensor([[0., 1., 2., 3., 4.],
        [5., 6., 7., 8., 9.]])
>>> torch.exp(t)
tensor([[1.0000e+00, 2.7183e+00, 7.3891e+00, 2.0086e+01, 5.4598e+01],
        [1.4841e+02, 4.0343e+02, 1.0966e+03, 2.9810e+03, 8.1031e+03]])
```



## flatten

将张量展开为向量。

```python
>>> t = torch.tensor([[[1, 2],
                       [3, 4]],
                      [[5, 6],
                       [7, 8]]])
>>> torch.flatten(t)
tensor([1, 2, 3, 4, 5, 6, 7, 8])
>>> torch.flatten(t, start_dim=1)
tensor([[1, 2, 3, 4],
        [5, 6, 7, 8]])
```



## log, log10, log2

对张量的所有元素应用对数函数。亦为`torch.Tensor`方法。

```python
>>> t
tensor([[0., 1., 2., 3., 4.],
        [5., 6., 7., 8., 9.]])
>>> torch.log(t)
tensor([[  -inf, 0.0000, 0.6931, 1.0986, 1.3863],
        [1.6094, 1.7918, 1.9459, 2.0794, 2.1972]])
>>> torch.log2(t)
tensor([[  -inf, 0.0000, 1.0000, 1.5850, 2.0000],
        [2.3219, 2.5850, 2.8074, 3.0000, 3.1699]])
>>> torch.log10(t)
tensor([[  -inf, 0.0000, 0.3010, 0.4771, 0.6021],
        [0.6990, 0.7782, 0.8451, 0.9031, 0.9542]])
```



## matmul

张量乘法。亦为`torch.Tensor`方法。`@`符号重载了此方法。

```python
>>> # 向量x向量: 内积
>>> v1 = torch.tensor([1, 2, 3])
>>> torch.matmul(v1, v1)
tensor(14)
>>> # 矩阵x向量, 向量x矩阵: 矩阵乘法
>>> m1 = torch.arange(1, 10).view(3, 3)
>>> m1
tensor([[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]])
>>> torch.matmul(m1, v1)                 # 向量会自动补全维度
tensor([14, 32, 50])                     # 3x3 x 3(x1) = 3(x1)
>>> torch.matmul(v1, m1)
tensor([30, 36, 42])                     # (1x)3 x 3x3 = (1x)3
>>> # 矩阵序列x向量: 扩张的矩阵乘法
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
>>> # 矩阵序列x矩阵: 扩张的矩阵乘法
>>> m2 = torch.ones(3, 3, dtype=torch.int64)
>>> torch.matmul(bm1, m2)
tensor([[[ 6,  6,  6],                    # [2x]3x3 x 3x3 = [2x]3x3
         [15, 15, 15],
         [24, 24, 24]],

        [[ 6,  6,  6],
         [15, 15, 15],
         [24, 24, 24]]])
>>> # 矩阵序列x矩阵序列: 逐元素的矩阵乘法
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
>>> # 矩阵序列x向量序列: 逐元素的矩阵乘法 不适用,会被识别为 矩阵序列x矩阵
    # 请将向量序列扩展为矩阵序列
```





## max, min, mean, std

返回张量所有元素统计量。亦为`torch.Tensor`方法。

```python
>>> t = torch.arange(10.).view(2, -1)
>>> t
tensor([[0., 1., 2., 3., 4.],
        [5., 6., 7., 8., 9.]])
>>> torch.max(t)
tensor(9.)
>>> torch.min(t)
tensor(0.)
>>> torch.mean(t)
tensor(4.5000)
>>> torch.std(t)
tensor(3.0277)
```



## mm

矩阵乘法。亦为`torch.Tensor`方法。

```python
>>> mat1 = torch.randn(1, 3)
>>> mat2 = torch.randn(3, 1)
>>> torch.mm(mat1, mat2)
tensor([[0.0717]])
```



## mul, div

张量逐元素乘法/除法。亦为`torch.Tensor`方法。`*, /`符号重载了此方法。

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





## ones

生成指定形状的全1张量。

```python
>>> torch.ones(2, 3)
tensor([[ 1.,  1.,  1.],
        [ 1.,  1.,  1.]])
```





## randn

生成指定形状的随机张量，其中每个元素服从标准正态分布。

```python
>>> torch.randn(2, 3)
tensor([[ 1.5954,  2.8929, -1.0923],
        [ 1.1719, -0.4709, -0.1996]])
```



## sigmoid

Sigmoid 激活函数。亦为`torch.Tensor`方法。见`torch.nn.Sigmoid`。

```python
>>> input = torch.randn(2)
>>> input
tensor([1.7808, -0.9893])
>>> torch.sigmoid(input)
tensor([0.8558, 0.2710])
```





## sin, cos, tan, arcsin, arccos, arctan, sinh, cosh, tanh, arcsinh, arccosh, arctanh

对张量的所有元素应用三角函数和双曲函数。

```python
>>> t
tensor([[0., 1., 2., 3., 4.],
        [5., 6., 7., 8., 9.]])
>>> torch.sin(t)
tensor([[ 0.0000,  0.8415,  0.9093,  0.1411, -0.7568],
        [-0.9589, -0.2794,  0.6570,  0.9894,  0.4121]])
>>> t.sin()
tensor([[ 0.0000,  0.8415,  0.9093,  0.1411, -0.7568],
        [-0.9589, -0.2794,  0.6570,  0.9894,  0.4121]])
```





## tanh

tanh 激活函数。亦为`torch.Tensor`方法。

```python
>>> input = torch.randn(2)
>>> input
tensor([-1.5400,  0.3318])
>>> torch.tanh(input)
tensor([-0.9121,  0.3202])
```



## Tensor

### detach

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



### expand

将张量在某些维度上以复制的方式扩展。注意内存共享问题。

```python
>>> x = torch.tensor([[1], [2], [3]])
>>> x.size()
torch.Size([3, 1])
>>> x.expand(-1, 4)       # -1 表示此维度保持不变
tensor([[ 1,  1,  1,  1],
        [ 2,  2,  2,  2],
        [ 3,  3,  3,  3]])
>>> x1 = x.expand(-1, 4)  # 共享内存
>>> x1[0][0] = 0
>>> x1
tensor([[0, 0, 0, 0],     # 共享内存
        [2, 2, 2, 2],
        [3, 3, 3, 3]])
>>> x
tensor([[0],
        [2],
        [3]])
```



### item

对于只有一个元素的张量，返回该元素的值。

```python
>>> t = torch.tensor([[[1]]])
>>> t.shape
torch.Size([1, 1, 1])
>>> t.item()
1
```



### new_full, new_ones, new_zeros

`new_full()`返回一个指定形状和所有元素值的张量，并且该张量与调用对象有同样的`torch.dtype`和`torch.device`。

```python
>>> tensor = torch.randn((2,), dtype=torch.float64)
>>> tensor.new_full((2, 3), 3.141592)
tensor([[ 3.1416,  3.1416,  3.1416],
        [ 3.1416,  3.1416,  3.1416]], dtype=torch.float64)
```

`new_ones(), new_zeros()`返回一个指定形状的全1/全0张量，并且该张量与调用对象有同样的`torch.dtype`和`torch.device`。

```python
>>> tensor = torch.randn((2,), dtype=torch.float64)
>>> tensor.new_ones((2,3))
tensor([[1., 1., 1.],
        [1., 1., 1.]], dtype=torch.float64)
>>> tensor.new_zeros((2,3))
tensor([[0., 0., 0.],
        [0., 0., 0.]], dtype=torch.float64)
```



### permute

返回将调用对象的所有维度重新排序得到的张量。

```python
>>> tensor = torch.arange(24).view(2,3,4)
>>> tensor
tensor([[[ 0,  1,  2,  3],
         [ 4,  5,  6,  7],
         [ 8,  9, 10, 11]],

        [[12, 13, 14, 15],
         [16, 17, 18, 19],
         [20, 21, 22, 23]]])
>>> tensor.permute(2, 0, 1)
>>> tensor.permute(2, 0, 1)
tensor([[[ 0,  4,  8],
         [12, 16, 20]],

        [[ 1,  5,  9],
         [13, 17, 21]],

        [[ 2,  6, 10],
         [14, 18, 22]],

        [[ 3,  7, 11],
         [15, 19, 23]]])
```



### repeat

将张量在某些维度上重复。

```python
>>> x = torch.arange(24).view(2, 3, 4)
>>> x.repeat(1, 1, 1).shape     # x自身
torch.Size([2, 3, 4])
>>> x.repeat(2, 1, 1).shape     # 在维度0上重复2次
torch.Size([4, 3, 4])
>>> x.repeat(2, 1, 1)
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
>>> x1 = x.repeat(2, 1, 1)
>>> x1[0][0][0] = 1
>>> x1
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



### squeeze

返回一个张量，其在输入张量的基础上删除所有规模为1的维度。返回张量与输入张量共享内存。

```python
>>> input = torch.randn(1,2,1,3,1,4)
>>> input.shape
torch.Size([1, 2, 1, 3, 1, 4])
>>> input.squeeze().shape
torch.Size([2, 3, 4])
```



### T

返回将调用对象的所有维度反转后的张量。

```python
>>> tensor = torch.randn(3, 4, 5)
>>> tensor.T.shape
torch.Size([5, 4, 3])
```



### to

返回调用对象更改`torch.dtype`和`torch.device`后的张量。

```python
>>> tensor = torch.randn(2, 2)  # Initially dtype=float32, device=cpu
>>> tensor.to(torch.float64)
tensor([[-0.5044,  0.0005],
        [ 0.3310, -0.0584]], dtype=torch.float64)
>>> cuda0 = torch.device('cuda:0')
>>> tensor.to(cuda0)
tensor([[-0.5044,  0.0005],
        [ 0.3310, -0.0584]], device='cuda:0')
>>> tensor.to(cuda0, dtype=torch.float64)
tensor([[-0.5044,  0.0005],
        [ 0.3310, -0.0584]], dtype=torch.float64, device='cuda:0')
```



### unsqueeze

返回一个张量，其在输入张量的基础上在指定位置增加一个规模为1的维度。返回张量与输入张量共享内存。

```python
>>> input = torch.randn(2,3,4)
>>> input.shape
torch.Size([2, 3, 4])
>>> input.unsqueeze(0).shape
torch.Size([1, 2, 3, 4])
>>> input.unsqueeze(3).shape
torch.Size([2, 3, 4, 1])
```





## topk

返回一维张量的最大的k个数。对于二维张量，返回每行的最大的k个数。

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



## transpose

交换张量的指定两个维度。亦为`torch.Tensor`方法。

```python
>>> t = torch.arange(10.).view(2, -1)
>>> t
tensor([[0., 1., 2., 3., 4.],
        [5., 6., 7., 8., 9.]])
>>> torch.transpose(t, 0, 1)
tensor([[0., 5.],
        [1., 6.],
        [2., 7.],
        [3., 8.],
        [4., 9.]])
```







## zeros

生成指定形状的全0张量。

```python
>>> torch.zeros(2, 3)
tensor([[ 0.,  0.,  0.],
        [ 0.,  0.,  0.]])
```





# torch.nn



## Conv1d

一维卷积层。

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



## Conv2d

二维卷积层。

```python
>>> m1 = nn.Conv2d(1, 32, 3, 1)                 # 卷积核大小为(3,3),步长为1
												# 将1个通道映射到32个卷积特征/通道
>>> m2 = nn.Conv2d(1, 32, (3,5), 1)             # 卷积核大小为(3,5)
>>> m3 = nn.Conv2d(1, 32, 3, 3)                 # 步长为3
>>> m4 = nn.Conv2d(1, 32, 3, 3, padding=(1,1))  # 上下,左右各用1,1行零填充
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



## CrossEntropyLoss

交叉熵损失函数。见`torch.nn.NLLLoss`。

```python
>>> loss = nn.CrossEntropyLoss()
>>> a1 = torch.tensor([[0.1, 0.8, 0.1]])
>>> a2 = torch.tensor([1])
>>> b = loss(a1, a2)
>>> b
tensor(0.6897)
>>> a2 = torch.tensor([0])
>>> b = loss(a1, a2)
>>> b
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



## Dropout

以给定概率将张量中的每个数置零，剩余的数乘以$$1/(1-p)$$。每次使用Dropout层的结果是随机的。

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



## Dropout2d

以给定概率将张量$$(N,C,H,W)$$的每个通道置零，剩余的通道乘以$$1/(1-p)$$。每次使用Dropout层的结果是随机的。

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



## Embedding

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



## GRU

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



## L1Loss

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



## Linear

全连接层。

```python
>>> m = nn.Linear(20, 4)
>>> input = torch.randn(128, 20)
>>> m(input).size()
torch.Size([128, 4])
```



## LSTM

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



## MaxPool1d

一维最大汇聚层。见torch.nn.functional.max_pool1d。

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



## MaxPool2d

二维最大汇聚层。见torch.nn.functional.max_pool2d。

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



## MSELoss

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



## NLLLoss

见`torch.nn.CrossEntropyLoss`。

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



## ReLU

ReLU 激活函数层。见`torch.nn.functional.relu`。

```python
>>> m = nn.ReLU()
>>> input = torch.randn(2)
>>> output = m(input)
>>> input
tensor([ 1.2175, -0.7772])
>>> output
tensor([1.2175, 0.0000])
```



## Sigmoid

Logistic 激活函数层。见`torch.sigmoid`。

```python
>>> m = nn.Sigmoid()
>>> input = torch.randn(2)
>>> output = m(input)
>>> input
tensor([ 1.7808, -0.9893])
>>> output
tensor([0.8558, 0.2710])
```



## Softmax, LogSoftmax

Softmax层。torch.nn.LogSoftmax相当于在Softmax的基础上为每个输出值求（自然）对数。

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





# torch.nn.functional



## max_pool1d

一维最大汇聚函数。见`torch.nn.MaxPool1d`。

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



## max_pool2d

二维最大汇聚函数。见`torch.nn.MaxPool2d`。

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



## one_hot

将向量转换为one-hot表示。

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



## relu

ReLU 激活函数。见`torch.nn.ReLU`。

```python
>>> input = torch.randn(2)
>>> input
tensor([1.2175, -0.7772])
>>> F.relu(input)
tensor([1.2175, 0.0000])
```



## sigmoid (deprecated)

Sigmoid 激活函数。见`torch.nn.Sigmoid, torch.sigmoid`。

```python
>>> input = torch.randn(2)
>>> input
tensor([1.7808, -0.9893])
>>> F.sigmoid(input)
tensor([0.8558, 0.2710])
```



## softmax

softmax回归。

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



## tanh

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

`torch.device`对象表示`torch.Tensor`被分配在哪个设备上。

`torch.device`包含了设备类型（`cpu`或`cuda`）和可选的设备序号。如果设备序号没有指定，对象总是会代表设备类型的当前设备。例如，构造`torch.Tensor`时选择设备`cuda`，即为选择设备`cuda:X`，其中X为`torch.cuda.current_device()`的结果。

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

view相当于numpy中的resize功能，即改变张量的形状。

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





# torch.cuda

```python
>>> torch.cuda.is_available()
True
>>> torch.cuda.device_count()
1
>>> torch.cuda.current_device()
0
>>> torch.cuda.device(0)
<torch.cuda.device at 0x7efce0b03be0>
>>> torch.cuda.get_device_name(0)
'GeForce GTX 3080'

```





# torch.distributed

> 参考[Distributed communication package - torch.distributed](https://pytorch.org/docs/stable/distributed.html#)



## 初始化

### is_available

distributed包是否可用。

```python
>>> torch.distributed.is_available()
True
```



### init_process_group

初始化默认的分布式进程组。

```python
torch.distributed.init_process_group(backend, init_method=None, timeout=datetime.timedelta(0, 1800), world_size=-1, rank=-1, store=None, group_name='')
# backend      使用的后端
# init_method  指明如何初始化进程组的URL.如果init_method和store都没有指定,则默认为'env://'
# world_size   参与任务的进程数
# rank         当前进程的rank
# store        对于所有worker可见的键值存储,用于交换连接/地址信息
# timeout      进程组执行操作的超时时间,默认为30min
```



### get_backend, get_rank, get_world_size

返回指定进程组的后端、rank和world_size。





# torch.optim

```python
optimizer = optim.SGD(model.parameters(), lr=0.01)
              # 梯度下降法   需要学习的参数  学习率
```





# torch.utils.data


