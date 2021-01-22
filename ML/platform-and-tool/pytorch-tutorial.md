[toc]

# [Learning PyTorch with Examples](https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#)

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
    if t % 10 == 0:
        print(t, loss.item())

    # 在backward pass之前,使用优化器归零(损失)对于需要更新的参数的梯度(也就是模型的权重参数)
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
from torch import nn


class TwoLayerNet(nn.Module):
    def __init__(self, D_in, H, D_out):
        """构造函数中初始化2个全连接模块或张量参数
        """
        super(TwoLayerNet, self).__init__()
        
        # 初始化模块
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
        # 使用模块
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
    for i, data in enumerate(trainloader, 0):
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
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('epoch %d/3, batch %5d with loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
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
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {:.4f}'.format(correct / total))

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





# 保存和加载模型

保存和加载模型主要用到下面三个函数：

+ `torch.save`：保存序列化对象到磁盘。此函数使用Python的pickle包序列化。模型、张量、词典等所有类型的对象都可以使用此函数。
+ `torch.load`：使用pickle包的unpickle功能反序列化对象文件到内存。此函数方便设备加载各种数据。
+ `torch.nn.Module.load_state_dict`: 使用一个反序列化的`state_dict`加载模型的参数词典



## `state_dict`是什么

在PyTorch中，一个`torch.nn.Module`模型的可学习的参数（即权重和偏置）包含在模型的*参数*（`model.parameters()`）中。`state_dict`就是将每一层映射到其参数张量的一个Python词典对象。注意只有带可学习参数的层（卷积层、线性层等）和注册的缓冲区在`state_dict`中有词条。优化器对象`torch.optim`也有一个`state_dict`，包含优化器的状态和超参数信息。



来看一个简单的模型的`state_dict`：

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

> 使用notebook: ml/pytorch/SaveandLoadModel.ipynb

**保存/加载`state_dict`（推荐）**

```python
# save
torch.save(model.state_dict(), PATH)

# load
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()
```

若保存一个模型是为了以后使用，那么只需要保存已训练的模型的参数。

PyTorch习惯使用`.pt`或`.pth`扩展名保存模型。

记得在使用前调用`model.eval()`以设定模型（丢弃层，批归一化层等）为评价模式。



**保存/加载整个模型**

```python
# save
torch.save(model, PATH)

# load
model = torch.load(PATH)
model.eval()
```

这种保存方式将会使用Python的pickle模块保存整个模型。这种方法的坏处是序列化数据绑定了保存模型时特定的类和实际的目录结构，原因是pickle并不保存模型类`TheModelClass`本身，而只保存一个包含该类的文件的路径，供加载时使用。因此在用到其它项目或项目重构后你的代码会失效。



## 保存和加载检查点

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

当保存一个检查点，不管是为了以后的使用还是继续训练，你都必须保存除了模型的`state_dict`之外的更多内容。优化器的`state_dict`十分重要，因为它包含了模型训练过程中不断更新的缓冲区和参数。其它可能需要保存的项包括最后训练的epoch，最新记录的训练损失等。因此检查点的存档大小经常是模型的两到三倍。

为了保存多个成分，将它们组织到一个词典里再使用`torch.save()`序列化这个词典。PyTorch习惯使用`.tar`扩展名保存检查点。

加载这些项时，首先初始化模型和优化器，再使用`torch.load()`加载词典。这里你只需要简单地查询词典就能获取所有保存项。



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

当保存由多个`torch.nn.Modules`组成的模型，例如编码器解码器模型，使用的方法和保存检查点是一样的。换言之，保存一个由每个模型的`state_dict`和相应的优化器组成的词典。正如之前提到的，你还可以添加其它可以帮助你继续训练的项。

与保存检查点相同，PyTorch习惯使用`.tar`扩展名。



## 通过使用预训练参数热启动模型

```python
# save
torch.save(modelA.state_dict(), PATH)

# load
modelB = TheModelBClass(*args, **kwargs)
modelB.load_state_dict(torch.load(PATH), strict=False)
```

充分利用预训练参数可以有效帮助热启动训练过程，使得训练更快地收敛。

可能你加载的词典不能完全对应模型的`state_dict`，例如缺失或多出了一些键，这时可以设置`strict`参数为`False`来忽略不匹配的键。

如果你想更精细地控制模型每一层加载的参数，只需要修改加载的词典的键和值，使其与模型的层（字段）名匹配。



## 保存和加载跨设备模型

**保存在GPU，加载在CPU**

```python
# save
torch.save(model.state_dict(), PATH)  # model is trained on GPU

# load
device = torch.device('cpu')
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location=device))
```



**保存在GPU，加载在GPU**

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

注意`my_tensor.to(device)`返回的是`my_tensor`在GPU中的一个新副本，它不会覆写`my_tensor`，因此记得手动覆写张量`my_tensor = my_tensor.to(device)`。



**保存在CPU，加载在GPU**

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

设定参数`map_location`为`cuda:device_id`将模型加载到指定的GPU设备中。之后还要再调用`model.to(torch.device("cuda"))`使模型中的所有参数张量转换为CUDA张量。

