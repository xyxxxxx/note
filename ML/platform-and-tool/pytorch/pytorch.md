[toc]

# [Learning PyTorch with Examples](https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#)

首先使用 numpy 实现一个简单的二层 FNN：

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



现在用 torch 实现上述 FNN，并且使用自动梯度计算 autograd：

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



`torch.nn` 提供了更高级的抽象（相当于 TensorFlow 的 keras），帮助我们更便捷地搭建网络。使用 `nn` 再次实现上述 FNN：

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
    if t % 10 == 0:
        print(t, loss.item())

    # 在backward pass之前,使用优化器归零(损失对于)需要更新的参数的梯度(也就是模型的权重参数)
    # 这在因为在调用 loss.backward() 时缓存区中的梯度会累积(而不是覆盖)
    optimizer.zero_grad()

    # Backward pass: 计算损失对于模型中所有可学习参数的梯度.因为每个Module中的参数都有
    # requires_grad=True
    loss.backward()

    # 调用优化器的step函数来更新一次参数
    optimizer.step()
```

（接下来请参照[自定义层](#自定义层)部分）





# 自定义层

不带参数的自定义层仅对输入张量进行一系列的操作再返回，实现简单。

```python
import torch
from torch import nn


class Norm(nn.Module):       # 必须继承nn.Module
    def __init__(self):
        super(Norm, self).__init__()
    def forward(self, x):    # 仅需实现forward方法
        return (x - x.mean())/x.std()

norm = Norm()    
x = torch.arange(11.)
y = norm(x)
print(y)
# tensor([-1.5076, -1.2060, -0.9045, -0.6030, -0.3015,  0.0000,  0.3015,  0.6030,
#         0.9045,  1.2060,  1.5076])
```



带参数的自定义层则需要在构造函数中声明作为模型参数的张量或者含有参数的预定义层。

```python
# 亦为Learning PyTorch with Examples的示例
import torch
import torch.nn as nn


class TwoLayerNet(nn.Module):
    def __init__(self, D_in, H, D_out):
        """构造函数中初始化2个全连接模块或张量参数
        """
        super(TwoLayerNet, self).__init__()
        
        # 初始化层
        # self.linear1 = nn.Linear(D_in, H)
        # self.linear2 = nn.Linear(H, D_out)

        # 初始化参数,需要nn.Parameter声明
        self.weight1 = nn.Parameter(torch.randn(1000, 100) / 25)
        self.bias1 = nn.Parameter(torch.randn(1, 100) / 25)
        self.weight2 = nn.Parameter(torch.randn(100, 10) / 25)
        self.bias2 = nn.Parameter(torch.randn(1, 10) / 25)

    def forward(self, x):
        """使用构造函数中声明的模块和参数
        """
        # 使用层
        # h_relu = self.linear1(x).clamp(min=0)
        # y_pred = self.linear2(h_relu)

        # 使用参数
        h = torch.mm(x, self.weight1) + self.bias1
        h_relu = h.clamp(min=0)
        y_pred = torch.mm(h_relu, self.weight2) + self.bias2

        return y_pred


# batch size; 输入维度; 隐藏层维度; 输出维度
N, D_in, H, D_out = 64, 1000, 100, 10

# 随机生成输入输出
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# 实例化
model = TwoLayerNet(D_in, H, D_out)
for name, parameters in model.named_parameters():  # 查看模型参数
    print(name, ':', parameters)

# 定义损失函数和优化器
# 调用model.parameters()将包含模型的成员变量(模块和参数)中的所有可学习参数
criterion = nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
for t in range(500):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x)

    # Compute and print loss
    loss = criterion(y_pred, y)
    if t % 50 == 49:
        print(t, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```





# MNIST

准备数据

```python
import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5), (0.5))])

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
```

展示数据示例

```python
import matplotlib.pyplot as plt
import numpy as np

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
```

建立模型

```python
import torch.nn as nn
import torch.nn.functional as F

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
        x = self.conv3(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.dense1(x))
        output = F.softmax(self.dense2(x), dim=1)

        return output

net = Net()
```

设定损失和优化器

```python
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
```

训练模型

```python
for epoch in range(3):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 1):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 0:    # print every 2000 mini-batches
            print('epoch %d/3, batch %5d with loss: %.3f' %
                  (epoch + 1, i, running_loss / 2000))
            running_loss = 0.0

print('Training Finished')
```

测试模型

```python
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {:.4f}'.format(correct / total))
```



## 使用GPU训练

如果你有一块具有 CUDA 功能的 GPU，就可以利用它加速模型计算。首先检查 PyTorch 是否可以使用 GPU：

```python
print(torch.cuda.is_available())
# True
```

创建一个设备对象：

```python
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
```

将模型和输入模型的张量都移动到 GPU 中：

```python
model.to(device)
data = data.to(device)
```

`model.to(device)` 将模型中的所有参数移动到 GPU 中；`tensor.to(device)` 则是返回 `tensor` 在 GPU 中的一个新副本，因此需要覆写原张量 `tensor = tensor.to(device)`。注意模型和数据需要在同一设备（CPU 或 GPU）中，否则会产生一个运行时错误。

上面的 MNIST 例子使用 CPU 训练，将其改造为使用 GPU 训练，需要增加如下代码：

```python
# device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# model
net.to(device)

# tensor
inputs = inputs.to(device)
labels = labels.to(device)
images = images.to(device)
```





# CNN

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义神经网络模型类
class CNNnet(nn.Module):
    def __init__(self): # 构造函数必需,对象属性为各网络层
      super(CNNnet, self).__init__()

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





# 基于注意力的机器翻译

```python
# NLP From Scratch: Translation with a Sequence to Sequence Network and Attention

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loading data files
# The data for this project is a set of many thousands of English to
# French translation pairs.

SOS_token = 0
EOS_token = 1


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# The files are all in Unicode, to simplify we will turn Unicode
# characters to ASCII, make everything lowercase, and trim most
# punctuation.
# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


# Lowercase, trim, and remove non-letter characters


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split(ceng'\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

# filter sentences longer than 10 words
def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)             # reduce number of pairs to 10k+


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


# The full process for preparing the data is:
# -  Read text file and split into lines, split lines into pairs
# -  Normalize text, filter by length and content
# -  Make word lists from sentences in pairs
def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
print(random.choice(pairs))


# seq2seq model
# Encoder
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        # embedding dimension = size of hidden state
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        # input one index of word
        # output embedding tensor with shape [1, 1, 256]
        output = self.embedding(input).view(1, 1, -1)
        # only one GRU cell
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# Simple Decoder, not used here
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, input_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size,
                             input_size)  # map hidden state to index of word
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# Decoder with attention mechanism
class AttnDecoderRNN(nn.Module):
    def __init__(self,
                 hidden_size,
                 output_size,
                 dropout_p=0.1,
                 max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)  # [1, 1, 256]
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(  # ??? # [1, 10]  
                torch.cat((embedded[0], hidden[0]), 1)),
            dim=1)
        attn_applied = torch.bmm(
            attn_weights.unsqueeze(0),  # [1, 1, 256] [1, 1, 10]
            encoder_outputs.unsqueeze(0))  # [1, 10 ,256]
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)  # [1, 1, 256]

        output = F.relu(output)
        output, hidden = self.gru(output,
                                  hidden)  # for one GRU cell, output==hidden

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# Preparing Training Data
def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)


# Training the Model
#
# To train we run the input sentence through the encoder, and keep track
# of every output and the latest hidden state. Then the decoder is given
# the ``<SOS>`` token as its first input, and the last hidden state of the
# encoder as its first hidden state.
#
# "Teacher forcing" is the concept of using the real target outputs as
# each next input, instead of using the decoder's guess as the next input.
# Using teacher forcing causes it to converge faster but `when the trained
# network is exploited, it may exhibit
# instability <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.378.4095&rep=rep1&type=pdf>`__.

use_teacher_forcing = True  # use teacher forcing or not


def train(
        input_tensor,  # train one pair of input&target tensor
        target_tensor,
        encoder,
        decoder,
        encoder_optimizer,
        decoder_optimizer,
        criterion,
        max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()  # zero tensor of shape [1, 1, 256]

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    # example:
    # tensor([[ 350],
    #         [  72],
    #         [1031],
    #         [ 349],
    #         [ 131],
    #         [2248],
    #         [  14],
    #         [   1]])
    # input_length: 8
    # tensor([[  23],
    #         [ 230],
    #         [ 444],
    #         [  94],
    #         [1079],
    #         [   3],
    #         [   1]])
    # target_length: 7

    encoder_outputs = torch.zeros(max_length,
                                  encoder.hidden_size,
                                  device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                 encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])

            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach(
            )  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


# print time elapsed and estimates time remaining given the current time and progress %.
import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


# The whole training process looks like this:
#
# -  Start a timer
# -  Initialize optimizers and criterion
# -  Create set of training pairs
# -  Start empty losses array for plotting
#
# Then we call ``train`` many times and occasionally print the progress (%
# of examples, time so far, estimated time) and average loss.
#
def trainIters(encoder,
               decoder,
               n_iters,
               print_every=1000,
               plot_every=100,
               learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [
        tensorsFromPair(random.choice(pairs)) for i in range(n_iters)
    ]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder, decoder,
                     encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' %
                  (timeSince(start, iter / n_iters), iter,
                   iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)


# Plotting losses
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


# Evaluation is mostly the same as training, but there are no targets so
# we simply feed the decoder's predictions back to itself for each step.
# Every time it predicts a word we add it to the output string, and if it
# predicts the EOS token we stop there.
def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length,
                                      encoder.hidden_size,
                                      device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


# evaluate random sentences from the training set
def evaluateRandomly(encoder, decoder, n=10):
    print(n, 'examples of evaluation:')
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')
    print('end')    


# Training and Evaluating
hidden_size = 256
encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words,
                               dropout_p=0.1).to(device)

trainIters(encoder1, attn_decoder1, 50000, print_every=1000)

evaluateRandomly(encoder1, attn_decoder1)


# Visualizing Attention
# You could simply run ``plt.matshow(attentions)`` to see attention output
# displayed as a matrix, with the columns being input steps and rows being
# output steps:
output_words, attentions = evaluate(encoder1, attn_decoder1,
                                    "je suis trop froid .")
plt.matshow(attentions.numpy())
plt.show()


# For a better viewing experience we will do the extra work of adding axes
# and labels:
def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') + ['<EOS>'],
                       rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def evaluateAndShowAttention(input_sentence):
    output_words, attentions = evaluate(encoder1, attn_decoder1,
                                        input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)


evaluateAndShowAttention("elle a cinq ans de moins que moi .")

evaluateAndShowAttention("elle est trop petit .")

evaluateAndShowAttention("je ne crains pas de mourir .")

evaluateAndShowAttention("c est un jeune directeur plein de talent .")

```





# `torch.autograd` 的简单入门

> 推荐阅读：
>
> [Automatic Differentiation with `torch.autograd`](https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html#)
>
> 参考：
>
> [`torch.autograd`](./pytorch-api-0.md#torchautograd)
>
> Notebook ml/pytorch/autograd.ipynb
>

`torch.autograd` 是 PyTorch 的自动微分引擎，用于驱动神经网络训练。



## 用法

先来看一个单步训练的例子：我们从 `torchvision` 中加载一个预处理 resnet18 模型，创建一个随机的张量代表一张 3 通道、宽 64、高 64 的图片，其相应的标签也用随机数进行初始化：

```python
import torch, torchvision
model = torchvision.models.resnet18(pretrained=True)
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)
```

然后进行前向计算：

```python
prediction = model(data)   # forward pass
```

下一步计算损失和进行反向传播。反向传播通过我们对损失张量调用 `.backward()` 启动，autograd 会计算所有模型参数的梯度并保存在每个参数的 `.grad` 属性中。

```python
loss = (prediction - labels).sum()
loss.backward()            # backward pass
```

接着，加载一个优化器，并在当中注册模型的所有参数：

```python
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
```

最后，调用 `.step()` 初始化梯度下降，优化器会根据每一个参数的 `.grad` 属性中存储的梯度调整各参数。

```python
optim.step()
```



## 张量

autograd 使用张量的下列属性：

* `data`：存储的数据信息
* `requires_grad`：设置为 `True` 表示张量需要计算梯度
* `grad`：张量的梯度值，记得每次迭代时归零，否则会累加
* `grad_fn`：表示得到张量的运算，叶节点通常为 `None`
* `is_leaf`：指示张量是否为叶节点



## 自动微分

来看一个简单的例子。我们创建两个张量 `a` 和 `b`，使用选项 `requires_grad=True`：

```python
import torch

a = torch.tensor(1., requires_grad=True)
b = torch.tensor(2., requires_grad=True)
```

> 也可以初始化之后再设置张量的 `requires_grad` 属性：
>
> ```python
> a = torch.tensor(1.)
> a.requires_grad = True
> ```

创建另一个张量 $c=a^2+b$ ：

```python
c = a**2 + b
```

> `requires_grad = True` 具有传递性，只要 `a` 和 `b` 中有一个 `requires_grad = True`，那么 `c` 也有 `requires_grad = True`。这表示 `requires_grad = True` 张量的所有运算都需要追踪，其计算路径构成下面的计算图。

假定 `a` 和 `b` 是神经网络的参数，`c` 是误差。训练过程中，我们想要误差对各参数的梯度，即
$$
\frac{\partial c}{\partial a}=2a，\ 
\frac{\partial c}{\partial b}=1
$$
当我们对 `c` 调用 `.backward()` 时，autograd 计算这些梯度并将其保存在各张量的 `.grad` 属性中。

```python
c.backward()
```

梯度现在被放置在 `a.grad` 和 `b.grad` 中：

```python
print(c)
print(a.data, a.requires_grad, a.grad, a.grad_fn, a.is_leaf)
print(b.data, b.requires_grad, b.grad, b.grad_fn, b.is_leaf)

# output:
# tensor(3., grad_fn=<AddBackward0>)
# tensor(1.) True tensor(2.) None True
# tensor(2.) True tensor(1.) None True
```



再来看一个例子，此时 $c$ 是一个向量：

```python
import torch

a = torch.tensor([1., 3.], requires_grad=True)
b = torch.tensor([2., 4.], requires_grad=True)

c = a**2 + b
```

如果 $c$ 对 $a,b$ 直接求梯度，将会得到一个矩阵，但我们想要得到与 $a,b$ 形状相同的梯度向量。我们可以通过求 $c$ 和某常数向量的内积将其转换为标量，例如和全 1 向量的内积相当于求和所有元素，通过 `backward()` 的 `gradient` 参数传入：

```python
external_grad = torch.tensor([1., 1.])
c.backward(gradient=external_grad)

print(c)
print(a.data, a.requires_grad, a.grad, a.grad_fn, a.is_leaf)
print(b.data, b.requires_grad, b.grad, b.grad_fn, b.is_leaf)

# tensor([ 3., 13.], grad_fn=<AddBackward0>)
# tensor([1., 3.]) True tensor([2., 6.]) None True
# tensor([2., 4.]) True tensor([1., 1.]) None True
```

改变 `gradient` 再看结果：

```python
external_grad = torch.tensor([1., 2.])
c.backward(gradient=external_grad)

print(c)
print(a.data, a.requires_grad, a.grad, a.grad_fn, a.is_leaf)
print(b.data, b.requires_grad, b.grad, b.grad_fn, b.is_leaf)

# tensor([ 3., 13.], grad_fn=<AddBackward0>)
# tensor([1., 3.]) True tensor([ 2., 12.]) None True
# tensor([2., 4.]) True tensor([1., 2.]) None True
```

改变 `gradient` 即为多个损失项赋予不同的权重。



## `Function`

> 参考：pytorch-lib-torch.autograd.Function

对（`requires_grad=True` 的）张量的每一次运算都会创建一个新的 `Function` 对象，用于执行计算、记录过程。一个最简单的例子：

```python
import torch

a = torch.tensor(1., requires_grad=True)
b = torch.tensor(2., requires_grad=True)
c = a**2 + b

print(c)
# tensor(3., grad_fn=<AddBackward0>)
```

这里的张量加法就是一个 `Function` 对象。

我们在构建网络的时候，通常使用 `nn.Module` 对象（例如 `nn.Conv2d`，`nn.ReLU` 等）作为基本单元。而实际上这些 Module 通常包裹了 `Function` 对象，作为实际运算（前向和反向计算）的部分。例如 `nn.ReLU` 实际使用 `torch.nn.functional.relu`（`F.relu`）：

```python
class ReLU(Module):
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super(ReLU, self).__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.relu(input, inplace=self.inplace)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str
```

我们可以自定义 `Function` 对象，以 `torch.autograd.Function` 为基类，实现 `forward()`（前向计算）和 `backward()`（反向计算）方法。来看下面的例子：

```python
class Exp(Function):                    # 此层计算e^x

    @staticmethod
    def forward(ctx, i):                # 模型前向
        result = i.exp()
        ctx.save_for_backward(result)   # 保存所需内容，以备backward时使用，所需的结果会被保存在saved_tensors元组中；此处仅能保存tensor类型变量，若其余类型变量（Int等），可直接赋予ctx作为成员变量，也可以达到保存效果
        return result

    @staticmethod
    def backward(ctx, grad_output):     # 模型梯度反传
        result, = ctx.saved_tensors     # 取出forward中保存的result
        return grad_output * result     # 计算梯度并返回

x = torch.tensor([1.], requires_grad=True)  # 需要设置tensor的requires_grad属性为True，才会进行梯度反传
ret = Exp.apply(x)                          # 使用apply方法调用自定义autograd function
print(ret)                                  # tensor([2.7183], grad_fn=<ExpBackward>)
ret.backward()                              # 反传梯度
print(x.grad)                               # tensor([2.7183])
```

下面的例子展示了如何保存 `tensor` 之外的变量：

```python
class GradCoeff(Function):       
       
    @staticmethod
    def forward(ctx, x, coeff):                 # 模型前向
        ctx.coeff = coeff                       # 将coeff存为ctx的成员变量
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):             # 模型梯度反传
        return ctx.coeff * grad_output, None    # backward的输出个数，应与forward的输入个数相同，此处coeff不需要梯度，因此返回None

# 尝试使用
x = torch.tensor([2.], requires_grad=True)
ret = GradCoeff.apply(x, -0.1)                  # 前向需要同时提供x及coeff，设置coeff为-0.1
ret = ret ** 2                          
print(ret)                                      # tensor([4.], grad_fn=<PowBackward0>)
ret.backward()  
print(x.grad)                                   # tensor([-0.4000])，梯度已乘以相应系数
```

再来看一个更复杂的例子，一个线性层的 `Function` 实现：

```python
# Inherit from Function
class LinearFunction(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias
```



## 计算图

> 参考：
>
> [PyTorch 的 Autograd](https://zhuanlan.zhihu.com/p/69294347)
>
> [PyTorch 源码解读之 torch.autograd](https://zhuanlan.zhihu.com/p/321449610)

前一节我们描述了单个 `Function` 对象的前向和反向计算，而实际的模型是由多个函数复合而成，可以抽象为由 Function 对象组成的有向无环图（DAG）。本节将介绍图级别的前向和反向计算过程。

在前向计算的过程中，autograd 会维护一个计算图（有向无环图），每次（`requires_grad=True` 的）张量运算都会向其中添加一个 `Function` 对象，运算结果的 `grad_fn` 属性即指向该对象。来看下面的例子：

```python
import torch

input = torch.ones([2, 2], requires_grad=False)
w1 = torch.tensor(2.0, requires_grad=True)
w2 = torch.tensor(3.0, requires_grad=True)
w3 = torch.tensor(4.0, requires_grad=True)

l1 = input * w1
l2 = l1 + w2
l3 = l1 * w3
l4 = l2 * l3
loss = l4.mean()
```

对于上述前向计算过程，构造的计算图为：

<img src="https://pic3.zhimg.com/80/v2-1781041624f4c9fb31df04d11dd6a84a_720w.jpg" style="zoom：50%;"/>

```python
def print_tensor(t):
    print(t.data, t.grad, t.grad_fn, t.is_leaf)

print_tensor(w1)
print_tensor(w2)
print_tensor(w3)
print_tensor(l1)
print_tensor(l2)
print_tensor(l3)
print_tensor(l4)
print_tensor(loss)

# tensor(2.) None None True
# tensor(3.) None None True
# tensor(4.) None None True
# tensor([[2., 2.],
#         [2., 2.]]) None <MulBackward0 object at 0x7fd7f99aa668> False
# tensor([[5., 5.],
#         [5., 5.]]) None <AddBackward0 object at 0x7fd7f918add8> False
# tensor([[8., 8.],
#         [8., 8.]]) None <MulBackward0 object at 0x7fd7f99aa668> False
# tensor([[40., 40.],
#         [40., 40.]]) None <MulBackward0 object at 0x7fd7f918add8> False
# tensor(40.) None <MeanBackward0 object at 0x7fd7f99aa668> False
```

可以看到，变量 `l1` 的 `grad_fn` 指向乘法运算符 `<MulBackward0>` 对象，用于在反向传播中指导梯度计算；叶节点的 `grad_fn` 为 None，因为它们由创建而非运算得到。

计算图中的叶节点是输入张量（模型参数），根节点是输出张量（损失）。在反向传播过程中，autograd 会从根节点溯源，利用链式法则计算所有叶节点的梯度。

张量的 `is_leaf` 属性表示该张量是否为叶节点。反向计算过程中只有 `is_leaf=True` 的张量的梯度会被保留。

反向计算过程为：

![](https://pic4.zhimg.com/80/v2-18add4601e35e4b26fb73a50245e8de7_720w.jpg)

```python
loss.backward()

print_tensor(w1)
print_tensor(w2)
print_tensor(w3)
print_tensor(l1)
print_tensor(l2)
print_tensor(l3)
print_tensor(l4)
print_tensor(loss)

# tensor(2.) tensor(28.) None True
# tensor(3.) tensor(8.) None True
# tensor(4.) tensor(10.) None True
# tensor([[2., 2.],
#         [2., 2.]]) None <MulBackward0 object at 0x7fd7f9160b38> False
# tensor([[5., 5.],
#         [5., 5.]]) None <AddBackward0 object at 0x7fd7f9160c50> False
# tensor([[8., 8.],
#         [8., 8.]]) None <MulBackward0 object at 0x7fd7f9160b38> False
# tensor([[40., 40.],
#         [40., 40.]]) None <MulBackward0 object at 0x7fd7f9160c50> False
# tensor(40.) None <MeanBackward0 object at 0x7fd7f9160b38> False
```

可以看到，只有 `is_leaf=True` 的张量的 `grad` 不为 None。因为用户一般不会使用中间变量的梯度，为了节约内存/显存，这些梯度在使用之后就被释放了。



注意 PyTorch 的计算图是动态的：每次反向计算结束，即调用 `.backward()` 返回后，计算图就在内存中被释放了；在下次前向计算过程中 autograd 会再创建一个新的计算图并为其填充数据。而 TensorFlow 使用的静态计算图是预先设计好的。

```python
# PyTorch使用动态计算图
a = torch.tensor([3.0, 1.0], requires_grad=True)
b = a * a
loss = b.mean()

loss.backward() # 正常
loss.backward() # RuntimeError

a = torch.tensor([3.0, 1.0], requires_grad=True)
b = a * a
loss = b.mean()
loss.backward() # 正常
```

理论上，静态图在效率上比动态图要高。因为首先，静态图只需要构建一次，之后可以重复使用；其次，静态图由于是固定的，因此可以做进一步的优化，比如可以将用户原本定义的 Conv 层和 ReLU 层合并成 ConvReLU 层，提高效率。

但是，深度学习框架的速度不仅仅取决于图的类型，还有很多其它的因素，比如底层代码质量，所使用的底层 BLAS 库等都有关。从实际测试结果来说，至少在主流模型的训练时间上，PyTorch 有着至少不逊于静态图框架 Caffe，TensorFlow 的表现。具体对比数据可以参考[这里](https://github.com/ilkarman/DeepLearningFrameworks)。

如今动态图和静态图之间的界限已经开始慢慢模糊。PyTorch 模型转成 Caffe 模型越来越方便，而 TensorFlow 也加入了一些动态图机制。



## 从DAG中移除

autograd 追踪所有 `requires_grad` 属性为 `True` 的张量。对于那些不需要计算梯度的张量，设定该属性为 `False` 以将其移除出 DAG。

对于一个张量运算，只要有一个输入张量有 `requires_grad=True`，那么输出张量就会有 `requires_grad=True`。

```python
x = torch.rand(5, 5)
y = torch.rand(5, 5)
z = torch.rand((5, 5), requires_grad=True)

a = x + y
print(f"Does `a` require gradients?: {a.requires_grad}")
b = x + z
print(f"Does `b` require gradients?: {b.requires_grad}")

# Does `a` require gradients?: False
# Does `b` require gradients?: True
```

在神经网络中，不计算梯度的参数通常称为**冻结参数（frozen parameters）**。如果你预先知道模型中的部分参数不需要计算梯度，那么可以冻结这些参数，这将降低 autograd 的计算量从而提升性能。



另一个从 DAG 中移除参数的常见例子是精调预训练模型。在精调过程中，我们冻结模型的大部分而只修改其中几层。来看下面这个例子，我们加载了预训练 resnet18 模型，并冻结所有参数：

```python
from torch import nn, optim

model = torchvision.models.resnet18(pretrained=True)

# Freeze all the parameters in the network
for param in model.parameters():
    param.requires_grad = False
```

比如我们想要在一个新的数据集上精调该模型，resnet 模型的分类器是最后一个线性层 `model.fc`，我们可以简单地将其替换为一个新的线性层，以用作我们的分类器（默认是解冻状态）：

```python
model.fc = nn.Linear(512, 10)
```

现在模型的所有参数，除了 `model.fc` 以外，都是冻结的，需要计算梯度的参数只有 `model.fc` 的权重和偏置：

```python
# Optimize only the classifier
optimizer = optim.SGD(model.fc.parameters(), lr=1e-2, momentum=0.9)
# Same, for other parameters have requires_grad = False
optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
```





# 使用优化器

> 参考：
>
> [torch.optim](https://pytorch.org/docs/stable/optim.html#)

## 如何使用优化器

要使用 `torch.optim` 包，你需要构造一个优化器实例，其保存了当前状态并会根据计算的梯度更新模型参数。



### 构造优化器

为了构造 `Optimizer` 实例，你需要传入一个包含要优化的模型参数（应为 `Variable` 实例）的可迭代对象，然后指定优化器相关的选项，例如学习率、权重衰减等等。

示例：

```python
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.Adam([var1, var2], lr=0.0001)
```

> 如果你需要将模型移动到 GPU（通过调用 `.cuda()` 或 `.to()`），请在为此模型构造优化器之前完成这一操作。`.cuda()` 或 `.to()` 调用之后的模型的参数将会是一组不同的对象。
>
> 总的来说，你需要保证在优化器构建和使用的整个过程中，被优化的模型参数存活在一致的位置。



`Optimizer` 也支持为部分参数单独指定选项，这时需要传入一个字典的可迭代对象，其中每个字典定义一个单独的参数组。字典应包含一个 `params` 键，对应包含要优化的部分模型参数的可迭代对象；其它键应匹配 `Optimizer` 实例接受的关键字参数，并用作这一组参数的选项。

例如为每一层单独设定学习率：

```python
optim.SGD([{
    'params': model.base.parameters()
}, {
    'params': model.classifier.parameters(),
    'lr': 1e-3            # 为这一组参数重载学习率
}],
          lr=1e-2,        # 关键字参数作为全局的默认选项
          momentum=0.9)
```



### 执行一步优化

所有的优化器都实现了 `step()` 方法，用于更新模型参数。可以有两种使用方法：

+ `optimizer.step()`：大多数优化器支持的简化方法。一旦使用例如 `backward()` 计算出梯度后，就可以调用此方法。例如：

  ```python
  for input, target in dataset:
      optimizer.zero_grad()
      output = model(input)
      loss = loss_fn(output, target)
      loss.backward()
      optimizer.step()
  ```

+ `optimizer.step(closure)`：一些优化算法例如共轭梯度和 LBFGS，需要多次……，因此你必须传入一个能够反复计算模型的闭包。该闭包应清除梯度，计算损失，最后返回损失。例如：

  ```python
  for input, target in dataset:
      def closure():
          optimizer.zero_grad()
          output = model(input)
          loss = loss_fn(output, target)
          loss.backward()
          return loss
      optimizer.step(closure)
  ```

  

## 如何调整学习率

`torch.optim.lr_scheduler` 提供了数种基于回合数调整学习率的方法。`torch.optim.lr_scheduler.ReduceLROnPlateau` 允许基于一些验证方法动态降低学习率。

学习率的调整应在每个回合的结束时应用，例如：

```python
model = [Parameter(torch.randn(2, 2, requires_grad=True))]
optimizer = SGD(model, 0.1)
scheduler = ExponentialLR(optimizer, gamma=0.9)

for epoch in range(20):
    for input, target in dataset:
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
    scheduler.step()
```



大多数学习率规划器都可以连续调用（也被称为连锁规划器），依次作用于学习率，例如：

```python
model = [Parameter(torch.randn(2, 2, requires_grad=True))]
optimizer = SGD(model, 0.1)
scheduler1 = ExponentialLR(optimizer, gamma=0.9)
scheduler2 = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)

for epoch in range(20):
    for input, target in dataset:
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
    scheduler1.step()
    scheduler2.step()
```



## 随机权重平均







# 保存和加载模型

保存和加载模型主要用到下面三个函数：

* `torch.save`：保存序列化对象到磁盘。此函数使用 Python 的 pickle 包序列化。模型、张量、词典等所有类型的对象都可以使用此函数。
* `torch.load`：使用 pickle 包的 unpickle 功能反序列化对象文件到内存。此函数方便设备加载各种数据。
* `torch.nn.Module.load_state_dict`：使用一个反序列化的 `state_dict` 加载模型的参数词典



## `state_dict`是什么

在 PyTorch 中，一个 `torch.nn.Module` 模型的可学习的参数（即权重和偏置）包含在模型的*参数*（`model.parameters()`）中。`state_dict` 就是将每一层映射到其参数张量的一个 Python 词典对象。注意只有带可学习参数的层（卷积层、线性层等）和注册的缓冲区在 `state_dict` 中有词条。优化器对象 `torch.optim` 也有一个 `state_dict`，包含优化器的状态和超参数信息。



来看一个简单的模型的 `state_dict`：

```python
# Define model
class TheModelClass(nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize model
model = TheModelClass()

# Initialize optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])
```

输出：

```python
Model's state_dict:
conv1.weight     torch.Size([6, 3, 5, 5])
conv1.bias   torch.Size([6])
conv2.weight     torch.Size([16, 6, 5, 5])
conv2.bias   torch.Size([16])
fc1.weight   torch.Size([120, 400])
fc1.bias     torch.Size([120])
fc2.weight   torch.Size([84, 120])
fc2.bias     torch.Size([84])
fc3.weight   torch.Size([10, 84])
fc3.bias     torch.Size([10])

Optimizer's state_dict:
state    {}
param_groups     [{'lr': 0.001, 'momentum': 0.9, 'dampening': 0, 'weight_decay': 0, 'nesterov': False, 'params': [4675713712, 4675713784, 4675714000, 4675714072, 4675714216, 4675714288, 4675714432, 4675714504, 4675714648, 4675714720]}]
```



## 保存和加载模型

> 使用notebook: ml/pytorch/save_and_load_model.ipynb

**保存/加载 `state_dict`（推荐）**

```python
# save
torch.save(model.state_dict(), PATH)

# load
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()
```

若保存一个模型是为了以后使用，那么只需要保存已训练的模型的参数。

PyTorch 习惯使用 `.pt` 或 `.pth` 扩展名保存模型。

记得在使用前调用 `model.eval()` 以设定模型（丢弃层，批归一化层等）为评价模式。



**保存/加载整个模型**

```python
# save
torch.save(model, PATH)

# load
model = torch.load(PATH)
model.eval()
```

这种保存方式将会使用 Python 的 pickle 模块保存整个模型。这种方法的坏处是序列化数据绑定了保存模型时特定的类和实际的目录结构，原因是 pickle 并不保存模型类 `TheModelClass` 本身，而只保存一个包含该类的文件的路径，供加载时使用。因此在用到其它项目或项目重构后你的代码会失效。



## 保存和加载检查点

> 使用notebook: ml/pytorch/save_and_load_checkpoint.ipynb

```python
# save
torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            ...
            }, PATH)

# load
model = TheModelClass(*args, **kwargs)
optimizer = TheOptimizerClass(*args, **kwargs)

checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()
# - or -
model.train()
```

当保存一个检查点，不管是为了以后的使用还是继续训练，你都必须保存除了模型的 `state_dict` 之外的更多内容。优化器的 `state_dict` 十分重要，因为它包含了模型训练过程中不断更新的缓冲区和参数。其它可能需要保存的项包括最后训练的 epoch，最新记录的训练损失等。因此检查点的存档大小经常是模型的两到三倍。

为了保存多个成分，将它们组织到一个词典里再使用 `torch.save()` 序列化这个词典。PyTorch 习惯使用 `.tar` 扩展名保存检查点。

加载这些项时，首先初始化模型和优化器，再使用 `torch.load()` 加载词典。这里你只需要简单地查询词典就能获取所有保存项。



## 保存多个模型到一个文件

```python
# save
torch.save({
            'modelA_state_dict': modelA.state_dict(),
            'modelB_state_dict': modelB.state_dict(),
            'optimizerA_state_dict': optimizerA.state_dict(),
            'optimizerB_state_dict': optimizerB.state_dict(),
            ...
            }, PATH)

# load
modelA = TheModelAClass(*args, **kwargs)
modelB = TheModelBClass(*args, **kwargs)
optimizerA = TheOptimizerAClass(*args, **kwargs)
optimizerB = TheOptimizerBClass(*args, **kwargs)

checkpoint = torch.load(PATH)
modelA.load_state_dict(checkpoint['modelA_state_dict'])
modelB.load_state_dict(checkpoint['modelB_state_dict'])
optimizerA.load_state_dict(checkpoint['optimizerA_state_dict'])
optimizerB.load_state_dict(checkpoint['optimizerB_state_dict'])

modelA.eval()
modelB.eval()
# - or -
modelA.train()
modelB.train()

```

当保存由多个 `torch.nn.Modules` 组成的模型，例如编码器解码器模型，使用的方法和保存检查点是一样的。换言之，保存一个由每个模型的 `state_dict` 和相应的优化器组成的词典。正如之前提到的，你还可以添加其它可以帮助你继续训练的项。

与保存检查点相同，PyTorch 习惯使用 `.tar` 扩展名。



## 通过使用预训练参数热启动模型

```python
# save
torch.save(modelA.state_dict(), PATH)

# load
modelB = TheModelBClass(*args, **kwargs)
modelB.load_state_dict(torch.load(PATH), strict=False)
```

充分利用预训练参数可以有效帮助热启动训练过程，使得训练更快地收敛。

可能你加载的词典不能完全对应模型的 `state_dict`，例如缺失或多出了一些键，这时可以设置 `strict` 参数为 `False` 来忽略不匹配的键。

如果你想更精细地控制模型每一层加载的参数，只需要修改加载的词典的键和值，使其与模型的层（字段）名匹配。



## 保存和加载跨设备模型

**保存在 GPU，加载在 CPU**

```python
# save
torch.save(model.state_dict(), PATH)  # model is trained on GPU

# load
device = torch.device('cpu')
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location=device))
```



**保存在 GPU，加载在 GPU**

```python
# save
torch.save(model.state_dict(), PATH)  # model is trained on GPU

# load
device = torch.device("cuda")
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.to(device)
# Make sure to call input = input.to(device) on any input tensors that you feed to the model
```

注意 `my_tensor.to(device)` 返回的是 `my_tensor` 在 GPU 中的一个新副本，它不会覆写 `my_tensor`，因此记得手动覆写张量 `my_tensor = my_tensor.to(device)`。



**保存在 CPU，加载在 GPU**

```python
# save
torch.save(model.state_dict(), PATH)  # model is trained on CPU

# load
device = torch.device("cuda")
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location="cuda:0"))  # Choose whatever GPU device number you want
model.to(device)
# Make sure to call input = input.to(device) on any input tensors that you feed to the model
```

设定参数 `map_location` 为 `cuda:device_id` 将模型加载到指定的 GPU 设备中。之后还要再调用 `model.to(torch.device("cuda"))` 使模型中的所有参数张量转换为 CUDA 张量。





# 使用 GPU

```python

```









# 使用 TensorBoard

## 设置 TensorBoard



## 记录图片到 TensorBoard



## 记录模型结构到 TensorBoard













# 分布式训练

> 分布式训练的基本概念参见[tensorflow-分布式训练-基本概念](./tensorflow.md#基本概念)



## 设置







## 数据并行训练

PyTorch 提供了几种数据并行训练的选项。对于逐渐从简单到复杂、从原型到生产的各种应用，常见的升级路径为：

1. 使用**单卡训练**：如果数据和模型可以容纳在单个 GPU 中，并且训练速度不成问题。
2. 使用**单机多卡数据并行**：如果机器上有多个 GPU，并且你想要通过最少的代码修改来加速训练。
3. 使用**单机多卡分布式数据并行**：如果你想要进一步加速训练，并且愿意多写一点代码来进行设置。
4. 使用**多机分布式数据并行和启动脚本**：如果应用需要在多机之间伸缩。
5. 使用 torchelastic 以启动分布式训练：如果训练可能出错或者资源会在训练过程中动态地增减。



### `torch.nn.DataParallel`

`DataParallel` 能够以最少的代码修改来启用单机多卡数据并行——它只需要在应用中增加一行代码。尽管如此，我们通常会使用 `DistributedDataParallel` 而不是 `DataParallel`，因为 `DistributedDataParallel` 能提供更好的性能表现，而 `DataParallel` 的具体实现实际上损失了很多性能（详见 [`torch.nn.parallel.DistributedDataParallel`](#`torch.nn.parallel.DistributedDataParallel`) 部分）。



#### 示例

在 PyTorch 中使用 GPU 非常简单，只需要把模型放到 GPU 中：

```python
device = torch.device("cuda:0")
model.to(device)
```

再复制所有的张量到 GPU 中：

```python
mytensor = my_tensor.to(device)
```

注意 `my_tensor.to(device)` 返回的是 `my_tensor` 在 GPU 中的一个新副本，因此你需要将其赋给一个新的张量并使用该张量。

PyTorch 默认只使用一个 GPU，你可以使用 `DataParallel` 来让模型并行运行在多个 GPU 上：

```Python
model = nn.DataParallel(model)
```

下面是一个详细的例子：

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Parameters and DataLoaders
input_size = 5
output_size = 2

batch_size = 30
data_size = 100

class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),
                         batch_size=batch_size, shuffle=True)

# Model
class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("\tIn Model: input size", input.size(),
              "output size", output.size())
        return output

model = Model(input_size, output_size)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

model.to(device)

for data in rand_loader:
    input = data.to(device)
    output = model(input)
    print("Outside: input size", input.size(),
          "output_size", output.size())
```

如果机器没有 GPU 或只有一个 GPU，那么 `In Model` 和 `Outside` 的输入是相同的：

```
	In Model: input size torch.Size([30, 5]) output size torch.Size([30, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
	In Model: input size torch.Size([30, 5]) output size torch.Size([30, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
	In Model: input size torch.Size([30, 5]) output size torch.Size([30, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
	In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
Outside: input size torch.Size([10, 5]) output_size torch.Size([10, 2])
```

如果有两个 GPU，那么每个 GPU 各有一个模型副本，各处理 `input` 的二分之一：

```
# on 2 GPUs
Let's use 2 GPUs!
    In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
    In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
    In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
    In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
    In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
    In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
    In Model: input size torch.Size([5, 5]) output size torch.Size([5, 2])
    In Model: input size torch.Size([5, 5]) output size torch.Size([5, 2])
Outside: input size torch.Size([10, 5]) output_size torch.Size([10, 2])
```

类似地，如果有 8 个 GPU：

```
# on 8 GPUs
Let's use 8 GPUs!
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
    In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])  # 10个样本仅分配到5个GPU上
    In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
    In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
    In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
    In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
Outside: input size torch.Size([10, 5]) output_size torch.Size([10, 2])
```



### `torch.nn.parallel.DistributedDataParallel`

`DistributedDataParallel` 是一种广泛采用的数据并行训练范式。使用 DDP 时，模型被复制到各个进程中，而每个模型副本会传入不同组的输入数据样本（可能是对同一数据集的切分）。DDP 负责梯度通信，以保证各模型副本同步和梯度计算叠加。

`DistributedDataParallel` 实现了模块级别的数据并行，并且可以跨多台机器运行。使用 DDP 的应用应产生多个进程并在每个进程中创建一个 DDP 实例，DDP 使用 `torch.distributed` 包中的集体通信方法来同步梯度和缓冲区。更具体地说，DDP 为每个由 `model.parameters()` 给出的模型参数注册一个 autograd 钩子，这些钩子会在相应的梯度在反向传递过程中被计算时激活，随后 DDP 接收到该信号并引发跨进程的梯度同步。 



比较 `DataParallel` 和 `DistributedDataParallel`：为什么你会考虑使用 `DistributedDataParallel` 而非 `DataParallel`：

* 首先，`DataParallel` 是单进程多线程，并且只能单机运行，而 `DistributedDataParallel` 是多进程，可以单机或多机运行。`DataParallel` 通常比 `DistributedDataParallel` 慢，即便是单机运行，因为线程间的 GIL 争夺、每次迭代（前向计算）都要广播模型，以及切分输入和汇总输入带来的额外花销。
* 如果你的模型太大以至于不能在单个 GPU 上训练，那么就必须用模型并行来将其切分到多个 GPU 中。`DistributedDataParallel` 兼容模型并行而 `DataParallel` 不能。当 DDP 结合模型并行时，每个 DDP 进程都会使用模型并行，而所有的进程共同使用数据并行。
* 如果你的模型需要跨多台机器或者不适用于数据并行范式，请参考 [RPC API](#一般分布式训练) 以获取更通用的分布式训练支持。



#### 示例1

让我们来看一个简单的 `torch.nn.parallel.DistributedDataParallel` 示例：

```python
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP


def example(rank, world_size):
    # setup distributed environment
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    # create default process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    # create local model
    model = nn.Linear(10, 10).to(rank)           # require 2 GPUs
    # construct DDP model
    ddp_model = DDP(model, device_ids=[rank])
    # define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    # forward pass
    outputs = ddp_model(torch.randn(20, 10).to(rank))
    labels = torch.randn(20, 10).to(rank)
    # backward pass
    loss_fn(outputs, labels).backward()
    # update parameters
    optimizer.step()

def main():
    world_size = 2
    mp.spawn(example, args=(world_size, ), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
```

其中，模型是一个线性层，将其用 DDP 包装后，对 DDP 模型进行一次前馈计算、反向计算和更新参数。在这之后，模型的参数会被更新，并且所有进程的模型都完全相同。



#### 示例2

首先设置进程组。

```python
import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
```

然后创建一个玩具模型，用 DDP 包装，并输入一些随机数据。请注意，DDP 构造函数会广播 rank0 进程的模型状态到所有其它进程，因此不必担心不同的进程有不同的模型参数初始值。

```python
class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    # create model and move it to GPU with id rank
    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(rank)
    loss_fn(outputs, labels).backward()
    optimizer.step()

    cleanup()


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)
```

可以看到，DDP 包装了底层分布式通信的细节并提供了一个简洁的 API。梯度同步通信发生在反向传递过程中，并且与反向计算部分重叠。当 `backward()` 返回时，`param.grad` 已经包含了同步的梯度张量。



### [注意事项](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html#skewed-processing-speeds)

对于基本使用，DDP 只需要多几行代码来创建进程组；但当 DDP 应用到更高级的用例中，则还需要注意一些问题。



**不一致的处理速度**

在 DDP 中，（DDP 的）构造函数、前向传递和反向传递是分布式同步点。不同的进程应当启动相同数量的同步，以相同的顺序到达这些同步点，以及在大致相同的时间到达这些同步点，否则快的进程会先到而等待落后的进程。因此用户应负责进程之间的负载均衡。

有时由于网络延迟、资源争夺、无法预测的负载峰值等原因，不一致的处理速度也难以避免。但为了防止这些情形下超时，在调用 `init_process_group()` 时请确保传入了一个足够大的 `timeout` 值。



**保存和加载检查点**

使用 `torch.save` 和 `torch.load` 在检查点保存和恢复模型是非常常见的操作。使用 DDP 时的一种优化方法是，保存模型仅在一个进程中进行，而加载模型则加载到所有进程，这样可以减少写的花销，但注意加载不要在保存结束之前开始。

当加载模型时，你需要提供一个合适的 `map_location` 参数以防止进程进入其它进程的设备。当 `map_location` 参数缺失时，`torch.load` 会首先将模型加载到 CPU，再将每一个参数复制到它被保存的地方，这将导致同一机器上的所有进程会使用相同的一组设备。对于更高级的错误恢复和弹性支持，请参考 TorchElastic。

```python
def demo_checkpoint(rank, world_size):
    print(f"Running DDP checkpoint example on rank {rank}.")
    setup(rank, world_size)

    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    CHECKPOINT_PATH = tempfile.gettempdir() + "/model.checkpoint"
    if rank == 0:
        # All processes should see same parameters as they all start from same
        # random parameters and gradients are synchronized in backward passes.
        # Therefore, saving it in one process is sufficient.
        torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)

    # Use a barrier() to make sure that process 1 loads the model after process
    # 0 saves it.
    dist.barrier()
    # configure map_location properly
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    ddp_model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=map_location))

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(rank)
    loss_fn = nn.MSELoss()
    loss_fn(outputs, labels).backward()
    optimizer.step()

    # Not necessary to use a dist.barrier() to guard the file deletion below
    # as the AllReduce ops in the backward pass of DDP already served as
    # a synchronization.

    if rank == 0:
        os.remove(CHECKPOINT_PATH)

    cleanup()
```



**结合 DDP 和模型并行**

DDP 兼容多 GPU 模型。当用巨量数据训练大型模型时，DDP 包装的多 GPU 模型十分有用。

```python
class ToyMpModel(nn.Module):
    def __init__(self, dev0, dev1):
        super(ToyMpModel, self).__init__()
        self.dev0 = dev0   # first GPU
        self.dev1 = dev1   # second GPU
        self.net1 = torch.nn.Linear(10, 10).to(dev0)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 5).to(dev1)

    def forward(self, x):
        x = x.to(self.dev0)
        x = self.relu(self.net1(x))
        x = x.to(self.dev1)
        return self.net2(x)
```

当 DDP 传入一个多 GPU 模型时，不能设置 `device_ids` 和 `output_device`，输入和输出数据会被放在合适的设备中。

```python
def demo_model_parallel(rank, world_size):
    print(f"Running DDP with model parallel example on rank {rank}.")
    setup(rank, world_size)

    # setup mp_model and devices for this process
    dev0 = rank * 2
    dev1 = rank * 2 + 1
    mp_model = ToyMpModel(dev0, dev1)
    ddp_mp_model = DDP(mp_model)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_mp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    # outputs will be on dev1
    outputs = ddp_mp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(dev1)
    loss_fn(outputs, labels).backward()
    optimizer.step()

    cleanup()


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    if n_gpus < 8:
        print(f"Requires at least 8 GPUs to run, but got {n_gpus}.")
    else:
        run_demo(demo_basic, 8)
        run_demo(demo_checkpoint, 8)
        run_demo(demo_model_parallel, 4)
```



### [内部设计](https://pytorch.org/docs/stable/notes/ddp.html#internal-design)

* 前提：DDP 依赖 c10d `ProcessGroup` 用于进程间通信，因此应用在构建 DDP 之前必须先创建 `ProcessGroup` 实例
* 构造：DDP 构造函数引用本地模块，并广播 rank0 进程的 `state_dict()` 到组内的所有进程以确保所有模型副本都从同样的状态开始。随后每个 DDP 进程创建一个本地 `Reducer`，其在之后的反向计算过程中负责梯度同步。为了提高通信效率，`Reducer` 组织参数梯度为桶结构，每次 reduce 一个桶。……



### TorchElastic

随着应用的复杂度和规模的增长，故障恢复变成了一个非常迫切的需求。在使用 DDP 时，我们有时会不可避免地遇见诸如 OOM 这样的错误，但 DDP 自己无法从这些错误中恢复，这是因为 DDP 要求所有进程以一种紧密同步的方式工作并且不同进程中启动的 `AllReduce` 通信必须匹配。如果一个进程抛出了 OOM 异常，这将很可能导致不同步（不匹配的 `AllReduce` 操作），进而导致训练崩溃或挂起。如果你预期故障会在训练过程中发生或者资源会动态地增减，那么请使用 TorchElastic 启动分布式数据并行训练。





## 一般分布式训练

许多训练范式不能由数据并行的形式所包容，例如参数服务器范式、分布式流水线范式、有多个观察者或代理的强化学习应用等。`torch.distributed.rpc` 的目标就是支持一般的分布式训练场景。

`torch.distributed.rpc` 包有以下四大支柱：

+ RPC 支持在远程工作器上运行给定的函数
+ RRef 帮助管理远程对象的生命周期
+ Distributed Autograd 扩展 autograd 引擎到跨多个机器
+ Distributed Optimizer 自动联系所有参与的工作器以使用分布式 autograd 引擎计算得到的梯度更新参数

