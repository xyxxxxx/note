# 示例

## [Learning PyTorch with Examples](https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#)

首先使用numpy实现一个简单的二层FNN：

```python
import numpy as np

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random input and output data
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

# Randomly initialize weights
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

learning_rate = 1e-6
for t in range(500):
    # Forward pass: compute predicted y
    h = x.dot(w1)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)

    # Compute and print loss
    loss = np.square(y_pred - y).sum()
    if t % 10 == 0:
      print(t, loss)

    # 计算loss对w1,w2的梯度,需要手动输入公式
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)

    # Update weights
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
```

现在用torch实现上述FNN，并且使用自动梯度计算autograd：

```python
import torch

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# 创建随机的输入和输出向量
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# 创建随机的权重向量
# 参数 requires_grad=True 表示希望在backward pass过程中计算对于这些张量的梯度
w1 = torch.randn(D_in, H, requires_grad=True)
w2 = torch.randn(H, D_out, requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    # Forward pass: 使用x(输入向量)和网络结构计算(张量操作)出y(输出向量)
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    # 计算损失
    # loss 是 (1,) 形状的张量,故使用 loss.item() 获取其中的标量值
    loss = (y_pred - y).pow(2).sum()
    if t % 100 == 99:
        print(t, loss.item())

    # 使用autograd计算backward pass过程,这将计算loss对于所有参数为
    # requires_grad=True 的向量的梯度.调用结束后梯度计算结果将保存在
    # w1.grad 和 w2.grad 中.
    loss.backward()

    # 手动更新梯度.
    # Wrap in torch.no_grad() because weights have requires_grad=True, 
    # but we don't need to track this in autograd.
    # An alternative way is to operate on weight.data and weight.grad.data.
    # Recall that tensor.data gives a tensor that shares the storage with
    # tensor, but doesn't track history.
    # 使用 torch.optim.SGD 可以达到同样的效果.
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # 手动在更新权重之后将梯度置零
        w1.grad.zero_()
        w2.grad.zero_()
```

`torch.nn`提供了更高级的抽象（相当于Tensorflow的keras），帮助我们更便捷地搭建网络。使用`nn`再次实现上述FNN：

```python
import torch

N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# 使用 nn 包定义模型,nn.Sequential包含了一个层的序列,按照顺序依次执行
# 各种类型的层和nn.Sequential都是Module对象
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)
# 使用 nn 包定义损失函数,这里使用MSE
loss_fn = torch.nn.MSELoss(reduction='sum')

# 使用 optim 包定义一个优化器来为我们的模型更新权重参数,这里使用Adam
# 第一个参数为优化器需要去更新的参数,第二个参数为学习率
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(500):
    # Forward pass: 将x输入模型得到y的预测结果
    # Module类重载了__call__方法因而可以像函数一样调用
    y_pred = model(x)

    # 输入y的预测值和实际值,由损失函数返回损失值
    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    # 在backward pass之前,使用优化器归零(损失)对于需要更新的参数的梯度(也就是模型的权重参数)
    # 这在因为在调用 loss.backward() 时缓存区中的梯度会累积(而不是覆盖)
    optimizer.zero_grad()

    # Backward pass: 计算损失对于模型中所有可学习参数的梯度.因此每个Module中的参数都有
    # requires_grad=True
    loss.backward()

    # 调用优化器的step函数来更新一次参数
    optimizer.step()
```



## CNN

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义神经网络模型类
class CNNnet(nn.Module):
    def __init__(self): # 构造函数必需,对象属性为各网络层
      super(Net, self).__init__()

      # 二维卷积层,输入1通道(灰度图片),输出32卷积特征/通道
      # 卷积核为3x3,步长为1
      self.conv1 = nn.Conv2d(1, 32, 3, 1)
      # 二维卷积层,输入32卷积特征/通道,输出64卷积特征/通道
      self.conv2 = nn.Conv2d(32, 64, 3, 1)

      # 丢弃层
      self.dropout1 = nn.Dropout2d(0.25)
      self.dropout2 = nn.Dropout2d(0.5)

      # 全连接层
      self.fc1 = nn.Linear(9216, 128)
      self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
      print(x.shape)  # [100, 1, 28, 28]
    
      x = self.conv1(x)
      print(x.shape)  # [100, 32, 26, 26]
      # ReLU激活函数
      x = F.relu(x)
      print(x.shape)

      x = self.conv2(x)  # [100, 64, 24, 24]
      print(x.shape)
      x = F.relu(x)
      print(x.shape)

      # 最大汇聚
      x = F.max_pool2d(x, 2)  # [100, 64, 12, 12]
      print(x.shape)
      # Pass data through dropout1
      x = self.dropout1(x)
      print(x.shape)
      # Flatten x with start_dim=1
      x = torch.flatten(x, 1) # [100, 9216]
      print(x.shape)
      x = self.fc1(x)
      print(x.shape)          # [100, 128]
      x = F.relu(x)
      print(x.shape)
      x = self.dropout2(x)
      print(x.shape)          # [100, 10]
      x = self.fc2(x)
      print(x.shape)

      # softmax回归
      output = F.softmax(x, dim=1)
      print(output.shape)
      return output  

my_nn = CNNnet()
# 100张28x28灰度图片
random_data = torch.rand((100, 1, 28, 28))
# [N, C, H, W]
# N = batch_size = 100
# C = channel = 1
# H = height = 28
# W = width = 28
result = my_nn(random_data)
print(result)
```



## RNN





# 库函数

## torch



### clamp

将输入张量中的所有数应用下限和上限。

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



### mm

矩阵乘法。

```python
>>> mat1 = torch.randn(1, 3)
>>> mat2 = torch.randn(3, 1)
>>> torch.mm(mat1, mat2)
tensor([[0.0717]])
```



## torch.nn



### Conv1d

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



### Conv2d

二维卷积层。

```python
>>> m1 = nn.Conv2d(1, 32, 3, 1)                 # 卷积核大小为(3,3),步长为1
												# 将1个通道映射到32个卷积特征/通道
>>> m2 = nn.Conv2d(1, 32, (3,5), 1)             # 卷积核大小为(3,5)
>>> m3 = nn.Conv2d(1, 32, 3, 3)                 # 步长为3
>>> m4 = nn.Conv2d(1, 32, 3, 3, padding=(1,1))  # 上下,左右各用1,1行零填充
>>> input = torch.rand((100, 1, 28, 28))
>>> output1, output2, output3, output4= m1(input), m2(input), m3(input), m4(input)
>>> output1.shape
torch.Size([100, 32, 26, 26])
>>> output2.shape
torch.Size([100, 32, 26, 24])
>>> output3.shape
torch.Size([100, 32, 9, 9])
>>> output4.shape
torch.Size([100, 32, 10, 10])
```



### Dropout

以给定概率将张量中的所有数置零，剩余的数乘以$$1/(1-p)$$。

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

以给定概率将张量$$(N,C,H,W)$$的整个通道置零，剩余的通道乘以$$1/(1-p)$$。

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



### Embedding

嵌入层。

```python
>>> embedding = nn.Embedding(10, 3)  # 词汇表规模 = 10, 嵌入维数 = 3, 共30个参数
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



### Linear

全连接层。

```python
>>> m = nn.Linear(20, 4)
>>> input = torch.randn(128, 20)
>>> output = m(input)
>>> print(output.size())
torch.Size([128, 4])
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
torch.Size([20, 64, 10])                   # 输出最上层的所有隐状态
>>> hn.shape
torch.Size([2, 64, 10])                    # 每一(层,方向)的最终隐状态


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



### MaxPool1d

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



### MaxPool2d

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



### ReLU

ReLU 激活函数层。见torch.nn.functional.relu。

```python
>>> m = nn.ReLU()
>>> input = torch.randn(2)
>>> output = m(input)
>>> input
tensor([ 1.2175, -0.7772,  1.3282, -0.1987,  0.3403, -1.3309, -0.3600, -1.5150])
>>> output
tensor([1.2175, 0.0000, 1.3282, 0.0000, 0.3403, 0.0000, 0.0000, 0.0000])
```



## torch.nn.functional



### max_pool1d

一维最大汇聚函数。见torch.nn.MaxPool1d。

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



### max_pool2d

二维最大汇聚函数。见torch.nn.MaxPool2d。

```python
>>> input = torch.randn(1, 1, 4, 4)
>>> output1 = F.max_pool2d(input, 2, 1)
>>> output2 = F.max_pool2d(input, 2)
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



### one_hot

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



### relu

ReLU 激活函数。见torch.nn.ReLU。

```python
>>> input = torch.randn(8)
>>> output = F.relu(input)
>>> input
tensor([ 1.2175, -0.7772,  1.3282, -0.1987,  0.3403, -1.3309, -0.3600, -1.5150])
>>> output
tensor([1.2175, 0.0000, 1.3282, 0.0000, 0.3403, 0.0000, 0.0000, 0.0000])
```



### softmax

softmax回归。

```python
>>> input = torch.tensor([[0.1,0.2,0.3,1],[0.2,0.3,0.4,2]])
>>> output0 = F.softmax(input, dim=0) # for every column
>>> output1 = F.softmax(input, dim=1) # for every row
>>> input
tensor([[0.1000, 0.2000, 0.3000, 1.0000],
        [0.2000, 0.3000, 0.4000, 2.0000]])
>>> output0
tensor([[0.4750, 0.4750, 0.4750, 0.2689],
        [0.5250, 0.5250, 0.5250, 0.7311]])
>>> output1
tensor([[0.1728, 0.1910, 0.2111, 0.4251],
        [0.1067, 0.1179, 0.1303, 0.6452]])
```

