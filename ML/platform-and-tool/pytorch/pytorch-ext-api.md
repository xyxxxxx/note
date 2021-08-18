[toc]

# torchvision

## datasets

所有的数据集都是 `torch.utils.data.Dataset` 的子类，即实现了 `__getitem__` 和 `__len__` 方法，因此它们可以被传入到一个 `torch.utils.data.DataLoader` 对象。`DataLoader` 可以使用 `torch.multiprocessing` 并行加载多个样本。例如：

```python
imagenet_data = torchvision.datasets.MNIST('path/to/imagenet_root/')
data_loader = torch.utils.data.DataLoader(imagenet_data,
                                          batch_size=4,
                                          shuffle=True,
                                          num_workers=args.nThreads)
```



### CIFAR10

CIFAR10 数据集。

```python
torchvision.datasets.CIFAR10(root: str, train: bool = True, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False)
# root         同`torchvision.datasets.MNIST`
# ...
```



### FashionMNIST

Fashion-MNIST 数据集。

```python
torchvision.datasets.FashionMNIST(root: str, train: bool = True, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False)
# root         同`torchvision.datasets.MNIST`
# ...
```



### MNIST

MNIST 数据集。

```python
torchvision.datasets.MNIST(root: str, train: bool = True, transform: Union[Callable, NoneType] = None, target_transform: Union[Callable, NoneType] = None, download: bool = False) → None
# root         数据集的根目录的路径
# train        若为`True`,使用训练集,否则使用测试集
# transform    对图片数据应用的变换函数
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



## transforms





## utils

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



### make_grid()

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



### save_image()

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

