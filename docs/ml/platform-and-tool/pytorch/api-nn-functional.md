# torch.nn.functional

## 汇聚函数（池化函数）

### max_pool1d()

一维最大汇聚函数。用法参见 `torch.nn.MaxPool1d`。

```python
>>> input = torch.randint(10, (1, 10)).to(torch.float32)
>>> input
tensor([[6., 3., 5., 9., 7., 1., 8., 2., 5., 7.]])
>>> F.max_pool1d(input, 3, stride=1)
tensor([[6., 9., 9., 9., 8., 8., 8., 7.]])
>>> F.max_pool1d(input, 3, stride=2)
tensor([[6., 9., 8., 8.]])
>>> F.max_pool1d(input, 3, stride=2, ceil_mode=True)
tensor([[6., 9., 8., 8., 7.]])
```

### max_pool2d()

二维最大汇聚函数。用法参见 `torch.nn.MaxPool2d`。

```python
>>> input = torch.randint(10, (1, 6, 6)).to(torch.float32)
>>> input
tensor([[[3., 5., 9., 4., 5., 6.],
         [0., 0., 5., 1., 9., 3.],
         [4., 5., 0., 9., 2., 2.],
         [8., 0., 3., 0., 0., 4.],
         [7., 7., 3., 7., 0., 0.],
         [2., 5., 0., 0., 8., 3.]]])
>>> F.max_pool2d(input, 3, stride=1)
tensor([[[9., 9., 9., 9.],
         [8., 9., 9., 9.],
         [8., 9., 9., 9.],
         [8., 7., 8., 8.]]])
>>> F.max_pool2d(input, 3, stride=2)
tensor([[[9., 9.],
         [8., 9.]]])
>>> F.max_pool2d(input, 3, stride=2, ceil_mode=True)
tensor([[[9., 9., 9.],
         [8., 9., 4.],
         [7., 8., 8.]]])
```

### max_pool3d()

三维最大汇聚函数。用法参见 `torch.nn.MaxPool3d`。

### avg_pool1d()

一维平均汇聚函数。用法参见 `torch.nn.AvgPool1d`。

```python
>>> input = torch.randint(10, (1, 1, 10)).to(torch.float32)
>>> input
tensor([[[6., 3., 5., 9., 7., 1., 8., 2., 5., 7.]]])
>>> F.avg_pool1d(input, 3, stride=1)
tensor([[[4.6667, 5.6667, 7.0000, 5.6667, 5.3333, 3.6667, 5.0000, 4.6667]]])
>>> F.avg_pool1d(input, 3, stride=2)
tensor([[[4.6667, 7.0000, 5.3333, 5.0000]]])
>>> F.avg_pool1d(input, 3, stride=2, ceil_mode=True)
tensor([[[4.6667, 7.0000, 5.3333, 5.0000, 6.0000]]])
```

### avg_pool2d()

二维平均汇聚函数。用法参见 `torch.nn.AvgPool2d`。

```python
>>> input = torch.randint(10, (1, 1, 6, 6)).to(torch.float32)
>>> input
tensor([[[[3., 5., 9., 4., 5., 6.],
          [0., 0., 5., 1., 9., 3.],
          [4., 5., 0., 9., 2., 2.],
          [8., 0., 3., 0., 0., 4.],
          [7., 7., 3., 7., 0., 0.],
          [2., 5., 0., 0., 8., 3.]]]])
>>> F.avg_pool2d(input, 3, stride=1)
tensor([[[[3.4444, 4.2222, 4.8889, 4.5556],
          [2.7778, 2.5556, 3.2222, 3.3333],
          [4.1111, 3.7778, 2.6667, 2.6667],
          [3.8889, 2.7778, 2.3333, 2.4444]]]])
>>> F.avg_pool2d(input, 3, stride=2)
tensor([[[[3.4444, 4.8889],
          [4.1111, 2.6667]]]])
>>> F.avg_pool2d(input, 3, stride=2, ceil_mode=True)
tensor([[[[3.4444, 4.8889, 4.5000],
          [4.1111, 2.6667, 1.3333],
          [4.0000, 3.0000, 2.7500]]]])
```

### avg_pool3d()

三维平均汇聚函数。用法参见 `torch.nn.AvgPool3d`。

## 激活函数

### elu()

ELU 激活函数。
$$
{\rm ELU}(x)=\max(0,x)+\min(0,\alpha(e^x-1))
$$

```python
torch.nn.functional.elu(input, alpha=1.0, inplace=False)
```

```python
>>> input = torch.randn(4)
>>> input
tensor([ 0.4309, -2.3080,  1.2376,  1.2595])
>>> F.elu(input)
tensor([ 0.4309, -0.9005,  1.2376,  1.2595])
>>> F.elu_(input)          # 原位操作
tensor([ 0.4309, -0.9005,  1.2376,  1.2595])
```

### leaky_relu()

Leaky ReLU 激活函数。
$$
{\rm LeakyReLU}(x)=\max(0,x)+{\rm negative\_slope*\min(0,x)}
$$

```python
>>> input = torch.randn(4)
>>> input
tensor([-0.6722,  0.3839,  0.7086, -0.9332])
>>> F.leaky_relu(input)
tensor([-0.0067,  0.3839,  0.7086, -0.0093])
>>> F.leaky_relu_(input)
tensor([-0.0067,  0.3839,  0.7086, -0.0093])
```

### relu()

ReLU 激活函数。见 `torch.nn.ReLU`。
$$
{\rm ReLU}(x)=\max(0,x)
$$

```python
>>> input = torch.randn(4)
>>> input
tensor([-0.5151,  0.0423, -0.8955,  0.0784])
>>> F.relu(input)
tensor([0.0000, 0.0423, 0.0000, 0.0784])
>>> F.relu_(input)        # 原位操作
tensor([0.0000, 0.0423, 0.0000, 0.0784])
```

### sigmoid()

Sigmoid 激活函数（实际上是 Logistic 激活函数）。见 `torch.nn.Sigmoid`、`torch.sigmoid`、`torch.special.expit`。
$$
f(x)=\frac{1}{1+e^{-x}}
$$

```python
>>> input = torch.randn(4)
>>> input
tensor([-0.0796, -0.5545,  1.6273, -1.3333])
>>> F.sigmoid(input)
tensor([0.4801, 0.3648, 0.8358, 0.2086])
```

> `nn.functional.sigmoid` is deprecated. Use `torch.sigmoid` instead.

### softmax()

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

### tanh()

tanh 激活函数。

```python
>>> input = torch.randn(4)
>>> input
tensor([ 0.7553,  1.6975, -0.0451,  0.3348])
>>> F.tanh(input)
tensor([ 0.6383,  0.9351, -0.0451,  0.3228])
```

> `nn.functional.tanh` is deprecated. Use `torch.tanh` instead.

## 稀疏函数

### one_hot()

将向量转换为 one-hot 表示。

```python
>>> F.one_hot(torch.arange(0, 5))
tensor([[1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1]])
>>> 
>>> F.one_hot(torch.arange(0, 5) % 3, num_classes=5)
tensor([[1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0]])
```

## 距离函数

### cosine_similarity()

计算两个张量沿指定维度的余弦相似度。
$$
{\rm cos\_similarity}=\frac{x_1\cdot x_2}{\max(\|x_1\|_2\cdot \|x_2\|_2, \epsilon)}
$$

```python
>>> input1 = torch.randn(3, 4, 5)
>>> input2 = torch.randn(3, 4, 5)
>>> F.cosine_similarity(input1, input2).shape
torch.Size([3, 5])
```

## 损失函数

### binary_cross_entropy()

### cross_entropy()

交叉熵损失函数。见 `torch.nn.CrossEntropyLoss`。

### kl_div()

### mse_loss()

### nll_loss()
