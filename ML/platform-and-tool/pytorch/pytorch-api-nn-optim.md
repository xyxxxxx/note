[toc]

# torch.nn

## 容器

### Module





#### named_parameters



#### parameters



## 线性层

### Linear

全连接层。

此模块支持 TensorFloat32。

```python
class torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
# in_features    输入特征数
# out_features   输出特征数
# bias           是否使用偏置
```

+ 输入形状：$$(N,*,H_{\rm in})$$，其中 $$N$$ 表示批次规模，$$*$$ 表示任意个额外的维度，$$H_{\rm in}={\rm in\_features}$$。
+ 输出形状：$$(N,*,H_{\rm out})$$，其中 $$H_{\rm out}={\rm out\_features}$$。
+ 参数：
  + `weight`：可学习的权重张量，形状为 `[out_features, in_features]`，初始值服从 $$(-\sqrt{k},\sqrt{k})$$ 区间上的均匀分布，其中 $$k=1/{\rm in\_features}$$。
  + `bias`：可学习的偏置张量，形状为 `[out_features,]`，初始值服从 $$(-\sqrt{k},\sqrt{k})$$ 区间上的均匀分布，其中 $$k=1/{\rm in\_features}$$。



```python
>>> linear1 = nn.Linear(10, 4)
>>> input = torch.randn(32, 10)
>>> output = linear1(input)
>>> output.size()
torch.Size([32, 4])
>>> 
>>> linear1.weight
Parameter containing:
tensor([[-0.0285,  0.0458, -0.0013,  0.2764,  0.0984, -0.1178, -0.1910, -0.0530,
         -0.1364, -0.1013],
        [ 0.0151,  0.1885,  0.1719, -0.3091,  0.1960,  0.0883,  0.3000,  0.2087,
         -0.2881, -0.3007],
        [-0.1525,  0.2777, -0.0527,  0.1353, -0.1470,  0.3103, -0.1338,  0.2371,
          0.0037, -0.1666],
        [ 0.1625, -0.1679,  0.0930, -0.0913, -0.0347, -0.3040, -0.1508,  0.1716,
         -0.0769,  0.3150]], requires_grad=True)
>>> linear1.bias
Parameter containing:
tensor([ 0.2535, -0.0148, -0.2111,  0.1926], requires_grad=True)
```



## 卷积层

### Conv1d

一维卷积层。

此模块支持 TensorFloat32。

```python
class torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
# in_channels     输入通道数
# out_channels    输出通道数
# kernel_size     卷积核大小
# stride          卷积步长
# padding         输入的两端填充的个数
# padding_mode    填充模式.若为`zeros`,则填充零;……
# dilation        卷积核元素的间隔
# groups          控制输入通道和输出通道之间的连接.例如若为`1`,则所有的输入通道连接所有的输出通道;若为`2`,则输入通道和
#                 输出通道各均分为2组,每个输入通道只会连接同组的输出通道;若为`in_channels`,则每个输入通道单独生成几个
#                 输出通道.此参数必须是`in_channels`和`out_channels`的公约数
# bias            若为`True`,为输出加上一个可以学习的偏置
```

参照 [Conv2d](#Conv2d)。

```python
>>> conv1 = nn.Conv1d(1, 32, 3, 1)                   # 卷积核长度为3,步长为1
>>> conv2 = nn.Conv1d(1, 32, 3, 3)                   # 步长为3
>>> conv3 = nn.Conv1d(1, 32, 3, 3, padding=(1, 1))   # 左右各用1个零填充
>>> input = torch.rand((100, 1, 28))
>>> conv1(input).shape
torch.Size([100, 32, 26])
>>> conv2(input).shape
torch.Size([100, 32, 9])
>>> conv3(input).shape
torch.Size([100, 32, 10])
```



### Conv2d

二维卷积层。

此模块支持 TensorFloat32。

```python
class torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
# in_channels     输入通道数
# out_channels    输出通道数
# kernel_size     卷积核大小,可以是单个整数(同时表示高和宽)或两个整数组成的元组(分别表示高和宽),下同
# stride          卷积步长
# padding         输入的四边填充的行/列数,可以是单个整数(同时表示上下填充的行数和左右填充的列数)或两个整数组成的元组(分别表示
#                 上下填充的行数和左右填充的列数)
# padding_mode    填充模式.若为`zeros`,则填充零;……
# dilation        卷积核元素的间隔
# groups          控制输入通道和输出通道之间的连接.例如若为`1`,则所有的输入通道连接所有的输出通道;若为`2`,则输入通道和
#                 输出通道各均分为2组,每个输入通道只会连接同组的输出通道;若为`in_channels`,则每个输入通道单独生成几个
#                 输出通道.此参数必须是`in_channels`和`out_channels`的公约数
# bias            若为`True`,为输出加上一个可以学习的偏置
```

> `kernel_size` 等参数的具体意义请参见 [An Introduction to different Types of Convolutions in Deep Learning](https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d)。

+ 输入形状：$$(N,C_{\rm in},H_{\rm in}, W_{\rm in})$$，其中 $$N$$ 表示批次规模，$$C$$ 表示通道数，$$H$$ 表示高，$$W$$ 表示宽，下同。
+ 输出形状：$$(N,C_{\rm out},H_{\rm out}, W_{\rm out})$$。
+ 参数：
  + `weight`：可学习的权重张量，形状为 `[out_channels, in_channels // groups, kernel_size[0], kernel_size[1]]`，初始值服从 $$(-\sqrt{k},\sqrt{k})$$ 区间上的均匀分布，其中 $$k=\cdots$$。
  + `bias`：可学习的偏置张量，形状为 `[out_channels,]`，初始值服从 $$(-\sqrt{k},\sqrt{k})$$ 区间上的均匀分布，其中 $$k=\cdots$$。

```python
>>> conv1 = nn.Conv2d(1, 32, 3, 1)                 # 卷积核大小为(3,3),步长为1
                                                   # 将1个通道(卷积特征)映射到32个通道(卷积特征)
>>> conv2 = nn.Conv2d(1, 32, (3, 5), 1)            # 卷积核大小为(3,5)
>>> conv3 = nn.Conv2d(1, 32, 3, 3)                 # 步长为3
>>> conv4 = nn.Conv2d(1, 32, 3, 3, padding=(1,1))  # 上下/左右各填充1行/1列零
>>> input = torch.rand((100, 1, 28, 28))
>>> conv1(input).shape
torch.Size([100, 32, 26, 26])
>>> conv2(input).shape
torch.Size([100, 32, 26, 24])
>>> conv3(input).shape
torch.Size([100, 32, 9, 9])
>>> conv4(input).shape
torch.Size([100, 32, 10, 10])
```



### Conv3d

三维卷积层。



## 汇聚层（池化层）

### MaxPool1d

一维最大汇聚层。见 `torch.nn.functional.max_pool1d`。

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

二维最大汇聚层。见 `torch.nn.functional.max_pool2d`。

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





# torch.optim

`torch.optim` 包实现了多种优化算法。最常用的优化方法已经得到支持，并且接口足够泛用，使得更加复杂的方法在未来也能够容易地集成进去。



## Adam

实现 Adam 算法。

```python
torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
# params        要优化的参数的可迭代对象,或定义了参数组的字典
# lr            学习率
# betas         用于计算梯度的移动平均和其平方的系数
# eps           添加到分母的项,用于提升数值稳定性
# weight_decay  权重衰退(L2惩罚)
# amsgrad       是否使用此算法的AMSGrad变体
```



## Optimizer

所有优化器的基类。



### add_param_group()

向优化器的 `param_groups` 添加一个参数组。

```python

```



### load_state_dict()

加载优化器状态字典。



### param_groups

返回优化器的所有参数组。

```python
>>> w = torch.tensor([1.], requires_grad=True)
>>> b = torch.tensor([1.], requires_grad=True)
>>> x = torch.tensor([2.])
>>> y = torch.tensor([4.])
>>> z = w @ x + b
>>> l = (y - z)**2
>>> l.backward()
>>> w.grad
tensor([-4.])
>>> b.grad
tensor([-2.])
>>> optimizer = torch.optim.SGD([
        {'params': w},
        {'params': b, 'lr':1e-3},
    ], lr=1e-2)
>>> optimizer.step()
>>> w
tensor([1.0400], requires_grad=True)
>>> b
tensor([1.0020], requires_grad=True)
>>> optimizer.param_groups      # 两组参数
[{'params': [tensor([1.0400], requires_grad=True)], 'lr': 0.01, 'momentum': 0, 'dampening': 0, 'weight_decay': 0, 'nesterov': False}, {'params': [tensor([1.0020], requires_grad=True)], 'lr': 0.001, 'momentum': 0, 'dampening': 0, 'weight_decay': 0, 'nesterov': False}]
```



### state_dict()

返回优化器的状态为一个字典，其中包含两项：

+ `state`：包含当前优化状态的字典
+ `param_groups`：包含所有参数组的字典



### step()

执行单步优化。



### zero_grad()

将所有参数的梯度置零。



## SGD

实现随机梯度下降算法。

```python
class torch.optim.SGD(params, lr=<required parameter>, momentum=0, dampening=0, weight_decay=0, nesterov=False)
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



## lr_scheduler

学习率规划器。



### _LRScheduler

所有学习率规划器的基类。



#### get_last_lr()

返回规划器计算的最后一个学习率。



#### load_state_dict()

加载规划器状态字典。



#### print_lr()

打印规划器的当前学习率。



#### state_dict()

返回规划器的状态为一个字典。



#### step()

更新学习率，具体操作取决于规划器的实现以及当前回合数。



### CosineAnnealingLR

使用余弦退火算法设定学习率，……



### CosineAnnealingWarmRestarts



### CyclicLR

根据循环学习率策略设定学习率，此策略下学习率在两个边界之间以固定频率变化。

循环学习率策略在每个批次结束时都要改变学习率，因此 `step()` 应在每个批次训练完毕后调用。

```python
class torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr, max_lr, step_size_up=2000, step_size_down=None, mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle', cycle_momentum=True, base_momentum=0.8, max_momentum=0.9, last_epoch=-1, verbose=False)
# optimzer    包装的优化器
# base_lr     初始学习率,同时是学习率的下界
# max_lr      学习率的上界
# step_size_up    循环上升期的训练批次数
# step_size_down  循环下降期的训练批次数
# mode        `'triangular'`,`'triangular2'`或`'exp_range'`,对应的三种模式见论文
#             https://arxiv.org/pdf/1506.01186.pdf
# gamma       `'exp_range'`模式下的学习率上下界的衰减乘数
# scale_fn    自定义缩放策略,由接收单个参数的匿名函数定义.若指定了此参数,则`mode`参数将被忽略(到底缩放的是上界还是下界?)
# scale_mode  若为`'cycle'`,则`scale_fn`接收的参数视为回合数;若为`'iterations'`,则`scale_fn`接收的参数视为批次数
# cycle_momentum  若为`True`,则动量与学习率反相循环
# base_momentum   动量的下界
# max_momentum    动量的上界
# last_epoch  最后一个回合的索引.若为`-1`,则此参数没有作用
# verbose     若为`True`,则每次更新学习率时向标准输出打印一条消息
```



### ExponentialLR

```python
class torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=-1, verbose=False)
# optimzer    包装的优化器
# gamma       学习率衰减乘数
# last_epoch  最后一个回合的索引.若为`-1`,则此参数没有作用
# verbose     若为`True`,则每次更新学习率时向标准输出打印一条消息
```

每回合学习率衰减为原来的 `gamma` 倍。



### LambdaLR

每回合学习率设定为原来的自定义函数返回值的倍数。

```python
class torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1, verbose=False)
# optimzer    包装的优化器
# lr_lambda   接收一个整数参数(回合数)并返回一个乘数的自定义函数,或为每组参数分别指定的自定义函数列表
# last_epoch  最后一个回合的索引.若为`-1`,则此参数没有作用
# verbose     若为`True`,则每次更新学习率时向标准输出打印一条消息
```

```python
>>> lambda1 = lambda epoch: epoch // 30
>>> lambda2 = lambda epoch: 0.95 ** epoch
>>> scheduler = LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])  # 此优化器有两组参数,两个函数分别对应一组
>>> for epoch in range(100):
    train(...)
    validate(...)
    scheduler.step()
```



### MultiplicativeLR

每回合学习率设定为原来的自定义函数返回值的倍数。

```python
class torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda, last_epoch=-1, verbose=False)
# optimzer    包装的优化器
# lr_lambda   接收一个整数参数(回合数)并返回一个乘数的自定义函数,或为每组参数分别指定的自定义函数列表
# last_epoch  最后一个回合的索引.若为`-1`,则此参数没有作用
# verbose     若为`True`,则每次更新学习率时向标准输出打印一条消息
```

```python
>>> lambda1 = lambda epoch: epoch // 30
>>> lambda2 = lambda epoch: 0.95 ** epoch
>>> scheduler = MultiplicativeLR(optimizer, lr_lambda=[lambda1, lambda2])  # 此优化器有两组参数,两个函数分别对应一组
>>> for epoch in range(100):
    train(...)
    validate(...)
    scheduler.step()
```



### OneCycleLR





### StepLR

```python
class torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1, verbose=False)
# optimzer    包装的优化器
# step_size   学习率衰减周期
# gamma       学习率衰减乘数
# last_epoch  最后一个回合的索引.若为`-1`,则此参数没有作用
# verbose     若为`True`,则每次更新学习率时向标准输出打印一条消息
```

每 `step_size` 回合学习率衰减为原来的 `gamma` 倍。注意此衰减可以与其它规划器引起的学习率变化同时发生。

```python
# Assuming optimizer uses lr = 0.05 for all groups
# lr = 0.05     if epoch < 30
# lr = 0.005    if 30 <= epoch < 60
# lr = 0.0005   if 60 <= epoch < 90
# ...
>>> scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
>>> for epoch in range(100):
    train(...)
    validate(...)
    scheduler.step()
```



### MultiStepLR

```python
class torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1, verbose=False)
# optimzer    包装的优化器
# milestones  回合索引列表,必须是递增的
# gamma       学习率衰减乘数
# last_epoch  最后一个回合的索引.若为`-1`,则此参数没有作用
# verbose     若为`True`,则每次更新学习率时向标准输出打印一条消息
```

每当回合数到达里程碑之一时学习率衰减为原来的 `gamma` 倍。注意此衰减可以与其它规划器引起的学习率变化同时发生。

```python
# Assuming optimizer uses lr = 0.05 for all groups
# lr = 0.05     if epoch < 30
# lr = 0.005    if 30 <= epoch < 80
# lr = 0.0005   if epoch >= 80
>>> scheduler = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
>>> for epoch in range(100):
    train(...)
    validate(...)
    scheduler.step()
```



### ReduceLROnPlateau

当指标不再改善时降低学习率。每当学习停滞时，降低学习率为原来的二到十分之一一般都能够改善模型。此规划器读取一个指标的值，并在若干个回合内没有看到改善时降低学习率。

```python
class torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)
# optimzer    包装的优化器
# mode        若为`min`,则在监视的量不再减小时降低学习率;若为`max`,则在监视的量不再增加时降低学习率
# factor      学习率衰减的乘数
# patience    等待的没有改善的回合数.例如此参数为2,则会忽略前2次指标没有改善,而在第3次指标仍没有改善时降低学习率
# threshold   可以被视作指标改善的阈值
# threshold_mode  若为`rel`,则动态阈值为`best*(1+threshold)`(对于`mode=max`)或`best*(1-threshold)`(对于`mode=min`)
#                 若为`abs`,则动态阈值为`best+threshold`(对于`mode=max`)或`best-threshold`(对于`mode=min`)
# cooldown    降低学习率之后再次恢复工作的冷却回合数
# min_lr      学习率的下限,或为每组参数分别指定的列表
# eps         应用于学习率的最小衰减,若新旧学习率之间的差值小于此参数,则忽略此次更新
# verbose     若为`True`,则每次更新学习率时向标准输出打印一条消息
```





