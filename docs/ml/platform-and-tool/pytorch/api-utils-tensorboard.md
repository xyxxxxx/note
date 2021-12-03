
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
