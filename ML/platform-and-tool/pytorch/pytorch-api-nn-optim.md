[toc]

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



### step()

执行单步优化。



### zero_grad()







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



