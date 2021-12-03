[toc]

# datasets

所有的数据集都是 `torch.utils.data.Dataset` 的子类，即实现了 `__getitem__` 和 `__len__` 方法，因此它们可以被传入到一个 `torch.utils.data.DataLoader` 对象。`DataLoader` 可以使用 `torch.multiprocessing` 并行加载多个样本。例如：

```python
imagenet_data = torchvision.datasets.MNIST('path/to/imagenet_root/')
data_loader = torch.utils.data.DataLoader(imagenet_data,
                                          batch_size=4,
                                          shuffle=True,
                                          num_workers=args.nThreads)
```



## CIFAR10

CIFAR10 数据集。

```python
class torchvision.datasets.CIFAR10(root: str, train: bool = True, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False)
# root         同`torchvision.datasets.MNIST`
# ...
```



## FashionMNIST

Fashion-MNIST 数据集。

```python
class torchvision.datasets.FashionMNIST(root: str, train: bool = True, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False)
# root         同`torchvision.datasets.MNIST`
# ...
```



## ImageNet

ImageNet 2012 分类数据集。

```python
class torchvision.datasets.ImageNet(root: str, split: str = 'train', download: Optional[str] = None, **kwargs: Any)
# root         数据集的根目录的路径
# split        数据集划分,支持`train`或`val`
# transform    一个函数或变换,其接收一个PIL图像并返回变换后的版本
```

> 使用此类需要安装 `scipy`



## MNIST

MNIST 数据集。

```python
class torchvision.datasets.MNIST(root: str, train: bool = True, transform: Union[Callable, NoneType] = None, target_transform: Union[Callable, NoneType] = None, download: bool = False) → None
# root         数据集的根目录,其中`MNIST/processed/training.pt`和`MNIST/processed/test.pt`存在
# train        若为`True`,从`MNIST/processed/training.pt`创建数据集(训练集),否则从`MNIST/processed/test.pt`
#              创建数据集(测试集)
# transform    一个函数或变换,其接收一个PIL图像并返回变换后的版本
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



## ImageFolder

一个通用的数据加载器，其加载路径下的图片应组织如下：

```
root/dog/xxx.png
root/dog/xxy.png
root/dog/[...]/xxz.png

root/cat/123.png
root/cat/nsdf3.png
root/cat/[...]/asd932_.png
```









# models

此模块包含下列用于图像分类的模型架构的定义：

- [AlexNet](https://arxiv.org/abs/1404.5997)
- [VGG](https://arxiv.org/abs/1409.1556)
- [ResNet](https://arxiv.org/abs/1512.03385)
- [SqueezeNet](https://arxiv.org/abs/1602.07360)
- [DenseNet](https://arxiv.org/abs/1608.06993)
- [Inception](https://arxiv.org/abs/1512.00567) v3
- [GoogLeNet](https://arxiv.org/abs/1409.4842)
- [ShuffleNet](https://arxiv.org/abs/1807.11164) v2
- [MobileNetV2](https://arxiv.org/abs/1801.04381)
- [MobileNetV3](https://arxiv.org/abs/1905.02244)
- [ResNeXt](https://arxiv.org/abs/1611.05431)
- [Wide ResNet](https://pytorch.org/vision/stable/models.html#wide-resnet)
- [MNASNet](https://arxiv.org/abs/1807.11626)
- [EfficientNet](https://arxiv.org/abs/1905.11946)
- [RegNet](https://arxiv.org/abs/2003.13678)

你可以通过调用模型的构造函数来构造一个有着随机初始权重的模型：

```python
import torchvision.models as models
resnet18 = models.resnet18()
alexnet = models.alexnet()
vgg16 = models.vgg16()
squeezenet = models.squeezenet1_0()
densenet = models.densenet161()
inception = models.inception_v3()
googlenet = models.googlenet()
shufflenet = models.shufflenet_v2_x1_0()
mobilenet_v2 = models.mobilenet_v2()
...
```

也可以传入 `pretrained=True` 来构造一个预训练的模型：

```python
import torchvision.models as models
resnet18 = models.resnet18(pretrained=True)
alexnet = models.alexnet(pretrained=True)
squeezenet = models.squeezenet1_0(pretrained=True)
vgg16 = models.vgg16(pretrained=True)
densenet = models.densenet161(pretrained=True)
inception = models.inception_v3(pretrained=True)
googlenet = models.googlenet(pretrained=True)
shufflenet = models.shufflenet_v2_x1_0(pretrained=True)
mobilenet_v2 = models.mobilenet_v2(pretrained=True)
...
```

使用预训练的模型将下载其权重到缓存目录下，该目录可以通过环境变量 `TORCH_MODEL_ZOO` 设定。

一些模型使用的模块在训练和测试中有着不同的行为，例如批归一化等。使用 `model.train()` 和 `model.eval()` 以在这两种模式中切换。

所有预训练的模型需要输入图像以同样的方式进行归一化。输入图像必须是小批量的形状为 `3*H*W` 的三通道 RGB 图像，其中 H 和 W 不小于 224；必须被加载到 $[0,1]$ 区间范围内再按照 `mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]` 进行归一化。你可以使用下面的变换进行归一化：

```python
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
```









## ResNet





# transforms

`transforms` 模块包含了常用的图像变换，这些变换可以通过 `Compose()` 链接起来。大部分变换类都有一个等价的函数：函数变换给予了细粒度的对于变换的控制。这会在你需要构建一个更加复杂的变换流水线时十分有用。

大部分变换同时接受 PIL 图像和张量图像，个别变换仅接受 PIL 图像或张量图像。

接受张量图像的变换同样接受批量的张量图像，前者是形状为 `(C, H, W)` 的张量，后者是形状为 `(B, C, H, W)` 的张量。

张量图像的值的范围隐式地由张量数据类型决定。浮点数据类型的张量图像应有 `[0, 1)` 区间内的值，整数数据类型的张量图像应有 `[0, MAX_DTYPE]` 区间内的值，其中 `MAX_DTYPE` 为该整数类型的最大值。

随机化的变换会对同一个批次的所有图像应用相同的变换，但对不同的批次产生不同的变换。若要在多次调用中复现变换，请使用函数变换。

各个变换的预览效果请参见[此示例](https://pytorch.org/vision/stable/auto_examples/plot_transforms.html)。



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



## 变换的组合

### Compose()

将数个变换组合在一起。

```python
transform = transforms.Compose([
    transforms.CenterCrop(10),
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float),
])
```



## 适用于 PIL 图像和 torch 张量的变换

### CenterCrop()

裁剪图像的中心。如果图像是张量，则应有形状 `[..., H, W]`。



### FiveCrop()

裁剪图像的四角和中心。如果图像是张量，则应有形状 `[..., H, W]`。



### Grayscale()

将图像转换为灰度图。如果图像是张量，则应有形状 `[..., 3, H, W]`。



### Pad()

在给定图像的四周填充指定值。如果图像是张量，则应有形状 `[..., H, W]`。



### RandomApply()

以给定概率随机地应用一个变换列表。



### RandomHorizontalFlip()

以给定概率随机地水平翻转给定图像。如果图像是张量，则应有形状 `[..., H, W]`。



### RandomResizedCrop()

裁剪图像的随机部分并修改为指定大小。如果图像是张量，则应有形状 `[..., H, W]`。



### RandomRotation()

旋转图像随机角度。如果图像是张量，则应有形状 `[..., H, W]`。



### RandomVerticalFlip()

以给定概率随机地垂直翻转给定图像。如果图像是张量，则应有形状 `[..., H, W]`。



### Resize()

将图像修改为指定大小（进行下采样）。





## 仅适用于 PIL 图像的变换



## 仅适用于 torch 张量的变换

### ConvertImageDtype()

将张量图像转换为给定的数据类型并相应地缩放值。



### Normalize()

以给定的均值和标准差归一化张量图像。给定均值 `(mean[1], ..., mean[n])` 和标准差 `(std[1], ..., std[n])` 的情况下，此变换会分别归一化张量图像的每个通道，即 `output[channel] = (input[channel] - mean[channel]) / std[channel]`。

> 此变换不是原位操作，即不会修改原输入张量。

```python
class torchvision.transforms.Normalize(mean, std, inplace=False)
# mean       每个通道的均值的序列
# std        每个通道的标准差的序列
# inplace    若为`True`,则此变换变为原位操作
```



#### forward()

归一化张量图像。



## 转换变换

### ToPILImage()

将张量（形状为 `(C, H, W)`）或 NumPy 数组（形状为 `(H, W, C)`）转换为 PIL 图像。



### ToTensor()

将 PIL 图像或 NumPy 数组（形状为 `(H, W, C)`）转换为张量（形状为 `(C, H, W)`）。



# utils

## make_grid()

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



## save_image()

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

