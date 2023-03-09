# torch.nn

## 容器

### Module

所有神经网络模块（module）的基类。你的自定义模型应继承此类。

模块可以包含其它模块，使得它们可以嵌套成为树形结构。你可以将子模块赋为常规属性：

```python
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)     # 子模块赋为常规属性
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))
```

以此种方式赋值的子模块将被注册，当你对模块调用 `to()` 等方法时这些子模块的参数也会被同样地转换。

#### add_module()

为当前模块添加一个子模块。添加的子模块可以作为属性访问（使用给定的名称）。

```python
add_module(name, module)
# name     子模块的名称
# module   被添加的子模块
```

#### apply()

递归地为模块及其子模块应用指定函数。典型的应用包括初始化模型的参数。

```python
>>> @torch.no_grad()
def init_weights(m):
    print(m)
    if type(m) == nn.Linear:
        m.weight.fill_(1.0)
        print(m.weight)
... 
>>> net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
>>> net.apply(init_weights)
Linear(in_features=2, out_features=2, bias=True)
Parameter containing:
tensor([[ 1.,  1.],
        [ 1.,  1.]])
Linear(in_features=2, out_features=2, bias=True)
Parameter containing:
tensor([[ 1.,  1.],
        [ 1.,  1.]])
Sequential(
  (0): Linear(in_features=2, out_features=2, bias=True)
  (1): Linear(in_features=2, out_features=2, bias=True)
)
Sequential(
  (0): Linear(in_features=2, out_features=2, bias=True)
  (1): Linear(in_features=2, out_features=2, bias=True)
)
```

#### buffers()

返回所有缓冲区的一个迭代器。

#### children()

返回所有直接子模块的一个迭代器。

#### cpu()

移动所有的模型参数和缓冲区到 CPU。

#### cuda()

移动所有的模型参数和缓冲区到 GPU。

如果你需要将模型移动到 GPU（通过调用 `.cuda()` 或 `.to()`），请在为此模型构造优化器之前完成这一操作。`.cuda()` 或 `.to()` 调用之后的模型的参数将会是一组不同的对象。

#### eval()

设置模块为测试模式。

仅对某些模块有效，请参阅特定模块的文档以了解其在训练/测试模式下的行为细节。

等价于 `train(False)`。

#### forward()

定义每次调用时执行的前向计算。

#### get_parameter()

```python
get_parameter(target)
```

返回由 `target` 给出的参数，若其不存在则引发异常。

#### get_submodule()

```python
get_submodule(target)
```

返回由 `target` 给出的子模块，若其不存在则引发异常。

例如，现有模块 `a`，其有一个嵌套的子模块 `net_b`，`net_b` 又有两个子模块 `net_c` 和 `linear`，`net_c` 又有子模块 `conv`。为了检查模块 `a` 是否有子模块 `linear`，应调用 `get_submodule("net_b.linear")`；为了检查模块 `a` 是否有子模块 `conv`，应调用 `get_submodule("net_b.net_c.linear")`。

#### load_state_dict()

```python
load_state_dict(state_dict, strict=True)
# state_dict    包含模块参数和持久缓冲区的字典
# strict        若为`True`,则`state_dict`的键必须正好匹配当前模块的`state_dict()`方法返回的模块的键
```

从 `state_dict` 复制参数和缓冲区到当前模块及其子模块。

#### modules()

返回所有模块（当前模块及其子模块）的一个迭代器。

```python
>>> net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
>>> for idx, m in enumerate(net.modules()):
    print(idx, '->', m)
0 -> Sequential(                                          # 0 -> 当前模块
  (0): Linear(in_features=2, out_features=2, bias=True)
  (1): Linear(in_features=2, out_features=2, bias=True)
)
1 -> Linear(in_features=2, out_features=2, bias=True)     # 1 -> 子模块1
2 -> Linear(in_features=2, out_features=2, bias=True)     # 2 -> 子模块2
```

#### named_buffers()

返回所有缓冲区的一个迭代器，产出缓冲区及其名称。

#### named_children()

返回所有直接子模块的一个迭代器，产出模块及其名称。

#### named_parameters()

返回所有参数的一个迭代器，产出模块及其名称。

#### parameters()

返回所有参数的一个迭代器，通常用于传给优化器。

#### register_buffer()

为模块添加一个缓冲区。

通常用于注册一个不应被看作为模型参数的缓冲区。例如，BatchNorm 的 `running_mean` 不是一个参数，但却是模块状态的一部分。默认情况下，缓冲区是持久的，并和参数一起保存。

缓冲区可以作为属性访问（使用给定的名称）。

```python
register_buffer(name, tensor, persistent=True)
# name        缓冲区的名称
# tensor      要注册的缓冲区
# persistent  若为`True`,则缓冲区是持久的,并会成为模块的`state_dict`的一部分
```

#### register_forward_hook()

为模块注册一个前向钩子。

此钩子会在每次 `forward()` 返回后被调用；其应该有如下签名：

```python
hook(module, input, output) -> None or modified output
```

其中 `input` 仅包含传给模块（`forward()`）的位置参数，传给模块的关键字参数不会传给钩子。钩子可以修改 `output`，也可以原位修改 `input`，但是这并不会影响到前向计算，因为钩子在 `forward()` 返回后才被调用。

#### register_pre_hook()

为模块注册一个前向前钩子。

此钩子会在每次 `forward()` 调用前被调用；其应该有如下签名：

```python
hook(module, input) -> None or modified input
```

其中 `input` 仅包含传给模块（`forward()`）的位置参数，传给模块的关键字参数不会传给钩子。钩子可以修改 `input`，返回修改后的值。

#### register_full_backward_hook()

为模块注册一个反向钩子。

此钩子会在每次 `backward()` 返回后被调用；其应该有如下签名：

```python
hook(module, grad_input, grad_output) -> tuple(Tensor) or None
```

其中 `grad_input` 和 `grad_output` 是分别包含了对于输入和输出的梯度的元组。此钩子不应修改其参数，但可以可选地返回一个新的对于输入的梯度以替代 `grad_input` 用于接下来的计算。`grad_input` 仅对应于传给模块的位置参数，传给模块的关键字参数将被忽略；`grad_input` 和 `grad_output` 中对应于非张量的元素为 `None`。

#### register_parameter()

为模块添加一个参数。

该参数可以作为属性访问（使用给定的名称）。

```python
register_parameter(name, param)
# name        参数的名称
# param       要添加的参数
```

#### requires_grad_()

设置 autograd 是否应该记录模块参数参与的运算，通过设置参数的 `requires_grad` 属性。

此方法有助于冻结部分模块以精调，或单独训练模型的各个部分（例如 GAN 训练）。

#### state_dict()

返回包含了模块完整状态的字典。

参数和持久缓冲区都包含在内；字典的键对应于参数和缓冲区的名称。

#### to()

移动参数和缓冲区到指定设备，或转换其数据类型。此方法**原位**修改模块。

```python
to(device=None, dtype=None, non_blocking=False)
to(dtype, non_blocking=False)
to(tensor, non_blocking=False)
to(memory_format=torch.channels_last)
# device          要移动到的设备
# dtype           要转换为的浮点或复数数据类型
# tensor          张量实例,其设备和数据类型作为此模块所有参数和缓冲区要移动和转换的目标
# memory_format   要转换为的内存格式
```

其函数签名类似于 `torch.Tensor.to()`，但 `dtype` 仅接受浮点或复数类型，并且只有浮点或复数参数和缓冲区会被转型；整数参数和缓冲区的类型保持不变。

```python
>>> linear = nn.Linear(2, 2)
>>> linear.weight
Parameter containing:
tensor([[ 0.1913, -0.3420],
        [-0.5113, -0.2325]])
>>> linear.to(torch.double)
Linear(in_features=2, out_features=2, bias=True)
>>> linear.weight
Parameter containing:
tensor([[ 0.1913, -0.3420],
        [-0.5113, -0.2325]], dtype=torch.float64)
>>> gpu1 = torch.device("cuda:1")
>>> linear.to(gpu1, dtype=torch.half, non_blocking=True)
Linear(in_features=2, out_features=2, bias=True)
>>> linear.weight
Parameter containing:
tensor([[ 0.1914, -0.3420],
        [-0.5112, -0.2324]], dtype=torch.float16, device='cuda:1')
>>> cpu = torch.device("cpu")
>>> linear.to(cpu)
Linear(in_features=2, out_features=2, bias=True)
>>> linear.weight
Parameter containing:
tensor([[ 0.1914, -0.3420],
        [-0.5112, -0.2324]], dtype=torch.float16)

>>> linear = nn.Linear(2, 2, bias=None).to(torch.cdouble)
>>> linear.weight
Parameter containing:
tensor([[ 0.3741+0.j,  0.2382+0.j],
        [ 0.5593+0.j, -0.4443+0.j]], dtype=torch.complex128)
>>> linear(torch.ones(3, 2, dtype=torch.cdouble))
tensor([[0.6122+0.j, 0.1150+0.j],
        [0.6122+0.j, 0.1150+0.j],
        [0.6122+0.j, 0.1150+0.j]], dtype=torch.complex128)
```

#### train()

设置模块为训练模式。

仅对某些模块有效，请参阅特定模块的文档以了解其在训练/测试模式下的行为细节。

#### training

若为 `True`，表示模块处于训练模式；若为 `False`，表示模块处于测试模式。

#### type()

将所有参数和缓冲区转换为指定数据类型。

#### zero_grad()

设置所有模型参数的梯度为 0。见 `torch.optim.Optimizer.zero_grad()`。

### Sequential

模块的顺序容器。模块将以它们被传入到初始化函数的顺序被添加，并以同样的顺序组成流水线。

整个容器可以被当作一个模块进行调用，对其执行的变换会被应用于其保存的每个模块（每个模块都被注册为 `Sequential` 实例的子模块）。

```python
# Using Sequential to create a small model. When `model` is run,
# input will first be passed to `Conv2d(1,20,5)`. The output of
# `Conv2d(1,20,5)` will be used as the input to the first
# `ReLU`; the output of the first `ReLU` will become the input
# for `Conv2d(20,64,5)`. Finally, the output of
# `Conv2d(20,64,5)` will be used as input to the second `ReLU`
model = nn.Sequential(
          nn.Conv2d(1,20,5),
          nn.ReLU(),
          nn.Conv2d(20,64,5),
          nn.ReLU()
        )
```

## 线性层

### Linear

全连接层。对输入数据应用线性变换 $y=xA^{\rm T}+b$。

此模块支持 TensorFloat32。

```python
class torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
# in_features    输入特征数
# out_features   输出特征数
# bias           是否使用偏置
```

* 输入形状： $(N,*,H_{\rm in})$，其中 $N$ 表示批次规模， $*$ 表示任意个额外的维度， $H_{\rm in}={\rm in\_features}$。
* 输出形状： $(N,*,H_{\rm out})$，其中 $H_{\rm out}={\rm out\_features}$。
* 参数：
    * `weight`：可学习的权重张量，形状为 `[out_features, in_features]`，初始值服从 $(-\sqrt{k},\sqrt{k})$ 区间上的均匀分布，其中 $k=1/{\rm in\_features}$。
    * `bias`：可学习的偏置张量，形状为 `[out_features,]`，初始值服从 $(-\sqrt{k},\sqrt{k})$ 区间上的均匀分布，其中 $k=1/{\rm in\_features}$。

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

### LazyLinear

和 `torch.nn.Linear` 模块相同，除了 `in_features` 参数通过推断得到。

此模块中，`weight` 和 `bias` 都属于 `torch.nn.UninitializedParameter` 类。它们将在第一次调用 `forward()` 后被初始化，然后模块会变成一个常规的 `torch.nn.Linear` 模块。`in_features` 参数从 `input.shape[-1]` 推断得到。

## 卷积层

### Conv1d

一维卷积层。

此模块支持 TensorFloat32。

```python
class torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, 
dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
# in_channels     输入通道数
# out_channels    输出通道数
# kernel_size     卷积核大小
# stride          卷积步长
# padding         输入的两端填充的个数
# padding_mode    填充模式.若为`zeros`,则填充零;……
# dilation        卷积核元素的间隔
# groups          控制输入通道和输出通道之间的连接.例如若为`1`,则所有的输入通道连接所有的输出通道;
#                 若为`2`,则输入通道和输出通道各均分为2组,每个输入通道只会连接同组的输出通道;
#                 若为`in_channels`,则每个输入通道单独生成几个输出通道.此参数必须是
#                 `in_channels`和`out_channels`的公约数
# bias            若为`True`,为输出加上一个可以学习的偏置
```

* 输入形状： $(N,C_{\rm in},L_{\rm in})$，其中 $N$ 表示批次规模， $C$ 表示通道数， $L$ 表示长，下同。
* 输出形状： $(N,C_{\rm out},L_{\rm out})$。
* 参数：
    * `weight`：可学习的权重张量，形状为 `[out_channels, in_channels // groups, kernel_size]`，初始值服从 $(-\sqrt{k},\sqrt{k})$ 区间上的均匀分布，其中 $k=\cdots$。
    * `bias`：可学习的偏置张量，形状为 `[out_channels,]`，初始值服从 $(-\sqrt{k},\sqrt{k})$ 区间上的均匀分布，其中 $k=\cdots$。

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
class torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, 
dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
# in_channels     输入通道数
# out_channels    输出通道数
# kernel_size     卷积核大小,可以是单个整数(同时表示高和宽)或两个整数组成的元组(分别表示高和宽),下同
# stride          卷积步长
# padding         输入的四边填充的行/列数,可以是单个整数(同时表示上下填充的行数和左右填充的列数)或
#                 两个整数组成的元组(分别表示上下填充的行数和左右填充的列数)
# padding_mode    填充模式.若为`zeros`,则填充零;……
# dilation        卷积核元素的间隔
# groups          控制输入通道和输出通道之间的连接.例如若为`1`,则所有的输入通道连接所有的输出通道;
#                 若为`2`,则输入通道和输出通道各均分为2组,每个输入通道只会连接同组的输出通道;
#                 若为`in_channels`,则每个输入通道单独生成几个输出通道.此参数必须是
#                 `in_channels`和`out_channels`的公约数
# bias            若为`True`,为输出加上一个可以学习的偏置
```

> `kernel_size` 等参数的具体意义请参见 [An Introduction to different Types of Convolutions in Deep Learning](https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d)。

* 输入形状： $(N,C_{\rm in},H_{\rm in}, W_{\rm in})$，其中 $N$ 表示批次规模， $C$ 表示通道数， $H$ 表示高， $W$ 表示宽，下同。
* 输出形状： $(N,C_{\rm out},H_{\rm out}, W_{\rm out})$。
* 参数：
    * `weight`：可学习的权重张量，形状为 `[out_channels, in_channels // groups, kernel_size[0], kernel_size[1]]`，初始值服从 $(-\sqrt{k},\sqrt{k})$ 区间上的均匀分布，其中 $k=\cdots$。
    * `bias`：可学习的偏置张量，形状为 `[out_channels,]`，初始值服从 $(-\sqrt{k},\sqrt{k})$ 区间上的均匀分布，其中 $k=\cdots$。

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

此模块支持 TensorFloat32。

```python
class torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, 
dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
# in_channels     输入通道数
# out_channels    输出通道数
# kernel_size     卷积核大小,可以是单个整数(同时表示深,高,宽)或三个整数组成的元组(分别表示深,高,宽),下同
# stride          卷积步长
# padding         输入的六面填充的层数,可以是单个整数(同时表示三个方向填充的层数)或三个整数组成的元组(分别表示
#                 深,高,宽三个方向填充的层数)
# padding_mode    填充模式.若为`zeros`,则填充零;……
# dilation        卷积核元素的间隔
# groups          控制输入通道和输出通道之间的连接.例如若为`1`,则所有的输入通道连接所有的输出通道;
#                 若为`2`,则输入通道和输出通道各均分为2组,每个输入通道只会连接同组的输出通道;
#                 若为`in_channels`,则每个输入通道单独生成几个输出通道.此参数必须是
#                 `in_channels`和`out_channels`的公约数
# bias            若为`True`,为输出加上一个可以学习的偏置
```

* 输入形状： $(N,C_{\rm in},D_{\rm in}, H_{\rm in}, W_{\rm in})$，其中 $N$ 表示批次规模， $C$ 表示通道数， $D$ 表示深， $H$ 表示高， $W$ 表示宽，下同。
* 输出形状： $(N,C_{\rm out},D_{\rm out},H_{\rm out}, W_{\rm out})$。
* 参数：
    * `weight`：可学习的权重张量，形状为 `[out_channels, in_channels // groups, kernel_size[0], kernel_size[1], kernel_size[2]]`，初始值服从 $(-\sqrt{k},\sqrt{k})$ 区间上的均匀分布，其中 $k=\cdots$。
    * `bias`：可学习的偏置张量，形状为 `[out_channels,]`，初始值服从 $(-\sqrt{k},\sqrt{k})$ 区间上的均匀分布，其中 $k=\cdots$。

## 汇聚层（池化层）

### MaxPool1d

一维最大汇聚层。见 `torch.nn.functional.max_pool1d()`。

```python
class torch.nn.MaxPool1d(kernel_size, stride=None, padding=0, dilation=1, 
return_indices=False, ceil_mode=False)
# kernel_size     滑动窗口的大小
# stride          滑动窗口的步长,默认为`kernel_size`
# padding         输入的两端填充的负无穷的个数
# dilation        滑动窗口元素的间隔
# return_indices  若为`True`,将最大值连同索引一起返回,用于之后调用`MaxUnpool1d`
# ceil_mode       若为`True`,则保证输入张量的每个元素都会被一个滑动窗口覆盖
```

+ 输入形状： $(N,C,L_{\rm in})$，其中 $N$ 表示批次规模， $C$ 表示通道数， $L$ 表示长，下同。
+ 输出形状： $(N,C,L_{\rm out})$。

```python
>>> input = torch.randint(10, (1, 10)).to(torch.float32)
>>> mp1 = nn.MaxPool1d(3, stride=1)
>>> mp2 = nn.MaxPool1d(3, stride=2)
>>> mp3 = nn.MaxPool1d(3, stride=2, ceil_mode=True)
>>> input
tensor([[6., 3., 5., 9., 7., 1., 8., 2., 5., 7.]])
>>> mp1(input)
tensor([[6., 9., 9., 9., 8., 8., 8., 7.]])
>>> mp2(input)
tensor([[6., 9., 8., 8.]])
>>> mp3(input)
tensor([[6., 9., 8., 8., 7.]])
```

### MaxPool2d

二维最大汇聚层。见 `torch.nn.functional.max_pool2d()`。

```python
class torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, 
return_indices=False, ceil_mode=False)
# kernel_size     滑动窗口的大小,可以是单个整数(同时表示高和宽)或两个整数组成的元组(分别表示高和宽),下同
# stride          滑动窗口的步长,默认为`kernel_size`
# padding         输入的四边填充的负无穷的行/列数,可以是单个整数(同时表示上下填充的行数和
#                 左右填充的列数)或两个整数组成的元组(分别表示上下填充的行数和左右填充的列数)
# dilation        滑动窗口元素的间隔
# return_indices  若为`True`,将最大值连同索引一起返回,用于之后调用`MaxUnpool2d`
# ceil_mode       若为`True`,则保证输入张量的每个元素都会被一个滑动窗口覆盖
```

+ 输入形状： $(N,C,H_{\rm in}, W_{\rm in})$，其中 $N$ 表示批次规模， $C$ 表示通道数， $H$ 表示高， $W$ 表示宽，下同。
+ 输出形状： $(N,C,H_{\rm out}, W_{\rm out})$。

```python
>>> input = torch.randint(10, (1, 6, 6)).to(torch.float32)
>>> mp1 = nn.MaxPool2d(3, stride=1)
>>> mp2 = nn.MaxPool2d(3, stride=2)
>>> mp3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
>>> input
tensor([[[3., 5., 9., 4., 5., 6.],
         [0., 0., 5., 1., 9., 3.],
         [4., 5., 0., 9., 2., 2.],
         [8., 0., 3., 0., 0., 4.],
         [7., 7., 3., 7., 0., 0.],
         [2., 5., 0., 0., 8., 3.]]])
>>> mp1(input)
tensor([[[9., 9., 9., 9.],
         [8., 9., 9., 9.],
         [8., 9., 9., 9.],
         [8., 7., 8., 8.]]])
>>> mp2(input)
tensor([[[9., 9.],
         [8., 9.]]])
>>> mp3(input)
tensor([[[9., 9., 9.],
         [8., 9., 4.],
         [7., 8., 8.]]])
```

### MaxPool3d

三维最大汇聚层。见 `torch.nn.functional.max_pool3d()`。

```python
class torch.nn.MaxPool3d(kernel_size, stride=None, padding=0, dilation=1, 
return_indices=False, ceil_mode=False)
# kernel_size     滑动窗口的大小,可以是单个整数(同时表示深,高,宽)或三个整数组成的元组(分别表示深,高,宽),下同
# stride          滑动窗口的步长,默认为`kernel_size`
# padding         输入的六面填充的负无穷的层数,可以是单个整数(同时表示三个方向填充的层数)或
#                 三个整数组成的元组(分别表示深,高,宽三个方向填充的层数)
# dilation        滑动窗口元素的间隔
# return_indices  若为`True`,将最大值连同索引一起返回,用于之后调用`MaxUnpool3d`
# ceil_mode       若为`True`,则保证输入张量的每个元素都会被一个滑动窗口覆盖
```

+ 输入形状： $(N,C,D_{\rm in},H_{\rm in}, W_{\rm in})$，其中 $N$ 表示批次规模， $C$ 表示通道数， $D$ 表示深， $H$ 表示高， $W$ 表示宽，下同。
+ 输出形状： $(N,C,D_{\rm out},H_{\rm out}, W_{\rm out})$。

### AvgPool1d

一维平均汇聚层。见 `torch.nn.functional.avg_pool1d()`。

```python
class torch.nn.AvgPool1d(kernel_size, stride=None, padding=0, ceil_mode=False, 
count_include_pad=True)
# kernel_size     滑动窗口的大小
# stride          滑动窗口的步长,默认为`kernel_size`
# padding         输入的两端填充的零的个数
# ceil_mode       若为`True`,则保证输入张量的每个元素都会被一个滑动窗口覆盖
# count_include_pad  若为`True`,则平均计算将包括填充的零
```

+ 输入形状： $(N,C,L_{\rm in})$，其中 $N$ 表示批次规模， $C$ 表示通道数， $L$ 表示长，下同。
+ 输出形状： $(N,C,L_{\rm out})$。

```python
>>> input = torch.randint(10, (1, 1, 10)).to(torch.float32)
>>> ap1 = nn.AvgPool1d(3, stride=1)
>>> ap2 = nn.AvgPool1d(3, stride=2)
>>> ap3 = nn.AvgPool1d(3, stride=2, ceil_mode=True)
>>> input
tensor([[[6., 3., 5., 9., 7., 1., 8., 2., 5., 7.]]])
>>> ap1(input)
tensor([[[4.6667, 5.6667, 7.0000, 5.6667, 5.3333, 3.6667, 5.0000, 4.6667]]])
>>> ap2(input)
tensor([[[4.6667, 7.0000, 5.3333, 5.0000]]])
>>> ap3(input)
tensor([[[4.6667, 7.0000, 5.3333, 5.0000, 6.0000]]])
```

### AvgPool2d

二维平均汇聚层。见 `torch.nn.functional.avg_pool2d()`。

```python
class torch.nn.AvgPool2d(kernel_size, stride=None, padding=0, ceil_mode=False, 
count_include_pad=True, divisor_override=None)
# kernel_size     滑动窗口的大小,可以是单个整数(同时表示高和宽)或两个整数组成的元组(分别表示高和宽),下同
# stride          滑动窗口的步长,默认为`kernel_size`
# padding         输入的四边填充的零的行/列数,可以是单个整数(同时表示上下填充的行数和
#                 左右填充的列数)或两个整数组成的元组(分别表示上下填充的行数和左右填充的列数)
# ceil_mode       若为`True`,则保证输入张量的每个元素都会被一个滑动窗口覆盖
# count_include_pad  若为`True`,则平均计算将包括填充的零
# divisor_override   若指定了此参数,则将被用作平均计算的分母,替代滑动窗口的元素数量
```

+ 输入形状： $(N,C,H_{\rm in}, W_{\rm in})$，其中 $N$ 表示批次规模， $C$ 表示通道数， $H$ 表示高， $W$ 表示宽，下同。
+ 输出形状： $(N,C,H_{\rm out}, W_{\rm out})$。

```python
>>> input = torch.randint(10, (1, 1, 6, 6)).to(torch.float32)
>>> ap1 = nn.AvgPool2d(3, stride=1)
>>> ap2 = nn.AvgPool2d(3, stride=2)
>>> ap3 = nn.AvgPool2d(3, stride=2, ceil_mode=True)
>>> input
tensor([[[[3., 5., 9., 4., 5., 6.],
          [0., 0., 5., 1., 9., 3.],
          [4., 5., 0., 9., 2., 2.],
          [8., 0., 3., 0., 0., 4.],
          [7., 7., 3., 7., 0., 0.],
          [2., 5., 0., 0., 8., 3.]]]])
>>> ap1(input)
tensor([[[[3.4444, 4.2222, 4.8889, 4.5556],
          [2.7778, 2.5556, 3.2222, 3.3333],
          [4.1111, 3.7778, 2.6667, 2.6667],
          [3.8889, 2.7778, 2.3333, 2.4444]]]])
>>> ap2(input)
tensor([[[[3.4444, 4.8889],
          [4.1111, 2.6667]]]])
>>> ap3(input)
tensor([[[[3.4444, 4.8889, 4.5000],
          [4.1111, 2.6667, 1.3333],
          [4.0000, 3.0000, 2.7500]]]])
```

### AvgPool3d

三维平均汇聚层。见 `torch.nn.functional.avg_pool3d()`。

```python
class torch.nn.AvgPool3d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)
# kernel_size     滑动窗口的大小,可以是单个整数(同时表示深,高,宽)或三个整数组成的元组(分别表示深,高,宽),下同
# stride          滑动窗口的步长,默认为`kernel_size`
# padding         输入的六面填充的零的层数,可以是单个整数(同时表示三个方向填充的层数)或
#                 三个整数组成的元组(分别表示深,高,宽三个方向填充的层数)
# ceil_mode       若为`True`,则保证输入张量的每个元素都会被一个滑动窗口覆盖
# count_include_pad  若为`True`,则平均计算将包括填充的零
# divisor_override   若指定了此参数,则将被用作平均计算的分母,替代滑动窗口的元素数量
```

+ 输入形状： $(N,C,D_{\rm in},H_{\rm in}, W_{\rm in})$，其中 $N$ 表示批次规模， $C$ 表示通道数， $D$ 表示深， $H$ 表示高， $W$ 表示宽，下同。
+ 输出形状： $(N,C,D_{\rm out},H_{\rm out}, W_{\rm out})$。

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

> 参考：[理解 PyTorch 中 LSTM 的输入输出参数含义](https://www.cnblogs.com/marsggbo/p/12123755.html)

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

## 归一化层

### BatchNorm1d

### BatchNorm2d

### BatchNorm3d

## Transformer 层

## 嵌入层

### Embedding

嵌入层。

此模块保存固定词汇表规模和维数的嵌入，输入索引列表，输出相应的嵌入。

```python
class torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, 
max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, 
_weight=None, device=None, dtype=None)
# num_embeddings      词汇表规模
# embedding_dim       嵌入维数
# padding_idx         指定索引的嵌入向量默认为全0,并且在训练过程中不会更新
# max_norm            若嵌入向量的范数大于此参数,则重新规范化到范数等于此参数
# norm_type           lp范数的p值
# scale_grad_by_freq  若为`True`,则梯度乘以小批次中词频的倒数
# sparse              若为`True`,则对于`weight`矩阵的梯度将会是一个稀疏张量
```

```python
>>> embedding = nn.Embedding(10, 3)   # 词汇表规模 = 10, 嵌入维数 = 3, 共30个参数
                                      # 注意10表示词汇表规模,输入为0-9之间的整数而非10维向量
>>> input = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
>>> embedding(input)
tensor([[[-0.0251, -1.6902,  0.7172],
         [-0.6431,  0.0748,  0.6969],
         [ 1.4970,  1.3448, -0.9685],
         [-0.3677, -2.7265, -0.1685]],

        [[ 1.4970,  1.3448, -0.9685],
         [ 0.4362, -0.4004,  0.9400],
         [-0.6431,  0.0748,  0.6969],
         [ 0.9124, -2.3616,  1.1151]]])
>>> 
>>> padding_idx = 0
>>> embedding = nn.Embedding(3, 3, padding_idx=padding_idx)
>>> embedding.weight
Parameter containing:
tensor([[ 0.0000,  0.0000,  0.0000],    # 默认为全0
        [-0.7895, -0.7089, -0.0364],
        [ 0.6778,  0.5803,  0.2678]], requires_grad=True)
>>> with torch.no_grad():
...     embedding.weight[padding_idx] = torch.ones(3)   # 手动修改
>>> embedding.weight
Parameter containing:
tensor([[ 1.0000,  1.0000,  1.0000],
        [-0.7895, -0.7089, -0.0364],
        [ 0.6778,  0.5803,  0.2678]], requires_grad=True)
```

!!! note 注意
    当 `max_norm` 不为 `None` 时，嵌入层的前向方法会原位修改 `weight` 张量的值（如果嵌入向量的范数超限）。由于需要计算梯度的张量不能被原位修改，如果在调用嵌入层的前向方法之前要对 `weight` 张量执行可微运算就需要克隆 `weight` 张量，例如：
    ```python
    n, d, m = 3, 5, 7
    embedding = nn.Embedding(n, d, max_norm=1.)
    W = torch.randn((m, d), requires_grad=True)
    idx = torch.tensor([1, 2])
    a = embedding.weight.clone() @ W.t()  # weight must be cloned for this to be differentiable
    b = embedding(idx) @ W.t()            # modifies weight in-place
    out = (a.unsqueeze(0) + b.unsqueeze(1))
    loss = out.sigmoid().prod()
    loss.backward()
    ```

## 丢弃层

### Dropout

一维丢弃层。

在训练模式下，以给定概率 $p$ 将张量的每个元素随机置零，剩余的元素乘以 $1/(1-p)$。每次调用丢弃层的结果是独立的。

在测试模式下，直接返回输入张量。

```python
class torch.nn.Dropout(p=0.5, inplace=False)
# p        元素置零的概率
# inplace  若为`True`,则原位执行此操作
```

```python
>>> dropout = nn.Dropout(0.5)
>>> input = torch.randn(4, 4)
>>> input
tensor([[-1.1218,  0.1338, -0.0065, -1.6416],
        [ 0.8897, -1.6002, -0.6922,  0.0689],
        [-1.3392, -0.5207, -0.2739, -0.9653],
        [ 0.6608,  0.9212,  0.0579,  0.9670]])
>>> output = dropout(input)
>>> output
tensor([[-2.2436,  0.0000, -0.0000, -3.2832],
        [ 1.7794, -3.2004, -1.3844,  0.0000],
        [-0.0000, -1.0414, -0.0000, -1.9306],
        [ 0.0000,  0.0000,  0.0000,  1.9340]])
>>> output = dropout(output)
>>> output
tensor([[-0.0000,  0.0000, -0.0000, -6.5664],
        [ 0.0000, -6.4008, -2.7688,  0.0000],
        [-0.0000, -0.0000, -0.0000, -0.0000],
        [ 0.0000,  0.0000,  0.0000,  3.8680]])
>>> 
>>> dropout.eval()
>>> dropout(input)
tensor([[-1.1218,  0.1338, -0.0065, -1.6416],
        [ 0.8897, -1.6002, -0.6922,  0.0689],
        [-1.3392, -0.5207, -0.2739, -0.9653],
        [ 0.6608,  0.9212,  0.0579,  0.9670]])
```

### Dropout2d

二维丢弃层。

在训练模式下，以给定概率 $p$ 将张量 $(N,C,H,W)$ 的每个通道随机置零，剩余的通道乘以 $1/(1-p)$。通常用于 `nn.Conv2d` 模块的输出。每次调用丢弃层的结果是独立的。

在测试模式下，直接返回输入张量。

```python
class torch.nn.Dropout2d(p=0.5, inplace=False)
# p        通道置零的概率
# inplace  若为`True`,则原位执行此操作
```

```python
>>> dropout = nn.Dropout2d(0.5)
>>> input = torch.randn(1, 8, 2, 2)
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
>>> output = dropout(input)
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
>>> 
>>> dropout.eval()
Dropout2d(p=0.5, inplace=False)
>>> dropout(input)
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
```

## 激活函数

### ELU

ELU 激活函数层。见 `torch.nn.functional.elu`。

$$
{\rm ELU}(x)=\max(0,x)+\min(0,\alpha(\exp(x)-1))
$$

```python
class torch.nn.ELU(alpha=1.0, inplace=False)
# alpha    α值
# inplace  若为`True`,则进行原位操作
```

![](https://pytorch.org/docs/stable/_images/ELU.png)

```python
>>> elu = nn.ELU()
>>> input = torch.randn(4)
>>> input
tensor([-1.0358, -0.9567, -0.9125,  0.7638])
>>> elu(input)
tensor([-0.6451, -0.6159, -0.5985,  0.7638])
```

### LeakyReLU

Leaky ReLU 激活函数层。见 `torch.nn.functional.leaky_relu`。

$$
{\rm LeakyReLU}(x)=\max(0,x)+{\rm negative\_slope*\min(0,x)}
$$

![](https://pytorch.org/docs/stable/_images/LeakyReLU.png)

```python
>>> lrelu = nn.LeakyReLU()
>>> input = torch.randn(4)
>>> input
tensor([-1.4089, -1.1398,  1.3921, -0.5492])
>>> lrelu(input)
tensor([-0.0141, -0.0114,  1.3921, -0.0055])
```

### ReLU

ReLU 激活函数层。见 `torch.nn.functional.relu`。

$$
{\rm ReLU}(x)=\max(0,x)
$$

![](https://pytorch.org/docs/stable/_images/ReLU.png)

```python
>>> relu = nn.ReLU()
>>> input = torch.randn(4)
>>> input
tensor([-0.5151,  0.0423, -0.8955,  0.0784])
>>> relu(input)
tensor([0.0000, 0.0423, 0.0000, 0.0784])
```

### Sigmoid

Logistic 激活函数层。见 `torch.sigmoid`、`torch.special.expit`。

$$
\sigma(x)=\frac{1}{1+\exp(-x)}
$$

![](https://pytorch.org/docs/stable/_images/Sigmoid.png)

```python
>>> logistic = nn.Sigmoid()
>>> input = torch.randn(4)
>>> input
tensor([-0.0796, -0.5545,  1.6273, -1.3333])
>>> logistic(input)
tensor([0.4801, 0.3648, 0.8358, 0.2086])
```

### Softmax, LogSoftmax

Softmax 层。`torch.nn.LogSoftmax` 相当于在 Softmax 层的基础上再对所有元素求（自然）对数。

$$
{\rm Softmax}(x_i)=\frac{\exp(x_i)}{\sum_j\exp(x_j)}\\
{\rm LogSoftmax}(x_i)=\ln \frac{\exp(x_i)}{\sum_j\exp(x_j)}
$$

```python
>>> sm = nn.Softmax(dim=0)
>>> lsm = nn.LogSoftmax(dim=0)
>>> input = torch.arange(4.0)
>>> input
tensor([0., 1., 2., 3.])
>>> sm(input)
tensor([0.0321, 0.0871, 0.2369, 0.6439])
>>> lsm(input)            # logsoftmax() = softmax() + log()
tensor([-3.4402, -2.4402, -1.4402, -0.4402])
>>> sm(input).log()
tensor([-3.4402, -2.4402, -1.4402, -0.4402])
```

### Tanh

tanh 激活函数层。见 `torch.tanh`。

$$
\tanh(x)=\frac{\exp(x)-\exp(-x)}{\exp(x)+\exp(-x)}
$$

![](https://pytorch.org/docs/stable/_images/Tanh.png)

```python
>>> tanh = nn.Tanh()
>>> input = torch.randn(4)
>>> input
tensor([ 1.1921, -1.0885,  0.2970,  0.3345])
>>> tanh(input)
tensor([ 0.8312, -0.7963,  0.2886,  0.3225])
```

## 损失函数

### BCELoss

二元交叉熵损失函数层。
$$
l_n=-w_n(t_n\log y_n+(1-t_n)\log (1-y_n))\\
l=\sum_n l_n\ {\rm 或}\ l=\frac{1}{N}\sum_n l_n
$$
其中 $N$ 为批次规模， $w_n$ 为  `weight` 参数指定的权重。

> 若 $y_n$ 取 $0$ 或 $1$，则 $l_n$ 表达式中的对数项之一就会在数学上无意义。PyTorch 选择设 $\log(0)=-\infty$，但损失表达式中存在无穷项会产生一些问题：
>
> 1. 若 $t_n=0$ 或 $1-t_n=0$，则会出现 0 乘以无穷。
> 2. 梯度计算链中也会存在无穷项，因为 $\frac{\partial l_n}{\partial y_n}=-w_n(\frac{t_n}{y_n}-\frac{1-t_n}{1-y_n})$。
>
> PyTorch 的解决方法是为对数项应用最小值 -100，这样损失值总是有限值，并且反向计算也是线性的。

```python
class torch.nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='mean')
# weight         为每个类别手动指定的权重,应为长度为C的一维张量.默认为全1张量
# size_average   deprecated
# reduce         deprecated
# reduction      指定对输出应用的归约方法.若为`'none'`,则不归约;若为`'sum'`,则对输出的所有元素求和;
#                若为`'mean'`,则对输出的所有元素求平均
```

```python
>>> y = torch.rand(3)
>>> y
tensor([0.5620, 0.1098, 0.8769])
>>> t = torch.tensor([1., 0, 1])
>>> loss = nn.BCELoss()
>>> loss(y, t)
tensor(0.2746)
>>> t = torch.tensor([0., 0, 1])
>>> loss(y, t)
tensor(0.3577)
>>> t = torch.tensor([1., 0, 0])
>>> loss(y, t)
tensor(0.9293)

>>> y = tensor([1.])
>>> t = tensor([.5])
>>> loss(y, t)
tensor(50.)
```

### CrossEntropyLoss

交叉熵损失函数层。相当于将 `LogSoftmax` 和 `NLLLoss` 组合为一个模块。

通常用于多分类问题（$C$ 个类别）；输入张量应当包含的是生的、未归一化的每个类别的分数，形状为 $(batch\_size,C)$ ；目标张量应当是批次规模长度的一维张量，其中每个值是 $[0, C-1]$ 范围内的整数索引，代表正确的类别。

```python
class torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')
# weight         为每个类别手动指定的权重,应为长度为C的一维张量.默认为全1张量
# size_average   deprecated
# ignore_index   忽略所有类别为指定索引的样本
# reduce         deprecated
# reduction      指定对输出应用的归约方法.若为`'none'`,则不归约;若为`'sum'`,则对输出的所有元素求和;
#                若为`'mean'`,则对输出的所有元素求平均
```

* 输入形状： $(N,C)$，其中 $N$ 表示批次规模，$C$ 表示类别数；或 $(N,C,d_1,d_2,\cdots,d_k)$，其中 $d_i$ 表示额外的维度。
* 目标形状： $(N)$，其中 $N$ 表示批次规模，每一个值是 $[0, C-1]$ 范围内的整数索引；或 $(N,d_1,d_2,\cdots,d_k)$，其中 $d_i$ 表示额外的维度。
* 输出形状：标量；若 `reduction` 为 `'none'`，则与目标形状相同。

```python
>>> y = torch.tensor([[0.2, 5.0, 0.8]])    # 输出分数
>>> t = torch.tensor([0])                  # 标签
>>> loss = nn.CrossEntropyLoss()
>>> loss(y, t)
tensor(4.8230)
>>> t = torch.tensor([1])
>>> loss(y, t)
tensor(0.0230)
>>> t = torch.tensor([2])
>>> loss(y, t)
tensor(4.2230)

# CrossEntropyLoss() = softmax() + log() + NLLLoss() = logsoftmax() + NLLLoss()
>>> y = torch.tensor([[ 0.4377, -0.3976, -1.3221],
                      [ 1.8402, -0.1696,  0.4744],
                      [-3.4641, -0.2303,  0.3552]])
>>> t = torch.tensor([0, 0, 2])
>>> loss = nn.CrossEntropyLoss()
>>> loss(y, t)
tensor(0.4197)

>>> y = torch.tensor([[ 0.4377, -0.3976, -1.3221],
                      [ 1.8402, -0.1696,  0.4744],
                      [-3.4641, -0.2303,  0.3552]])
>>> y = y.softmax(dim=1)
>>> y = y.log()
>>> t = torch.tensor([0, 0, 2])
>>> loss = nn.NLLLoss()
>>> loss(y, t)
tensor(0.4197)

>>> y = torch.tensor([[ 0.4377, -0.3976, -1.3221],
                      [ 1.8402, -0.1696,  0.4744],
                      [-3.4641, -0.2303,  0.3552]])
>>> logsoftmax = nn.LogSoftmax(dim=1)
>>> y = logsoftmax(y)
>>> t = torch.tensor([0, 0, 2])
>>> loss = nn.NLLLoss()
>>> loss(y, t)
tensor(0.4197)
```

### KLDivLoss

### MSELoss

均方差损失函数层。

$$
l_n=(y_n-t_n)^2\\
l=\sum_n l_n\ {\rm 或}\ l=\frac{1}{N}\sum_n l_n
$$

其中 $N$ 为批次规模。

```python
class torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
# size_average    deprecated
# reduce          deprecated
# reduction       指定对输出应用的归约方法.若为`'none'`,则不归约;若为`'sum'`,则对输出的所有元素求和;
#                 若为`'mean'`,则对输出的所有元素求平均
```

```python
>>> y = torch.randint(4, (4,)).to(torch.float32)
>>> t = torch.randint(4, (4,)).to(torch.float32)
>>> y
tensor([3., 2., 1., 3.])
>>> t
tensor([3., 1., 3., 1.])
>>> 
>>> loss = nn.MSELoss()
>>> loss(y, t)
tensor(2.2500)
>>> loss = nn.MSELoss(reduction='sum')
>>> loss(y, t)
tensor(9.)
```

### NLLLoss

负对数似然损失层。

$$
l_n = -w_{y_n}x_{n,y_n}\\
l=\sum_n l_n\ {\rm 或}\ l=\frac{1}{N}\sum_n l_n
$$

其中 $x$ 为输入，$y$ 为目标，$w$ 为 `weight` 参数指定的权重，$N$ 为批次规模。

通常用于多分类问题（$C$ 个类别）；输入张量应当包含的是每个类别的概率的（自然）对数，形状为 $(batch\_size,C)$ ；目标张量应当是批次规模长度的一维张量，其中每个值是 $[0, C-1]$ 范围内的整数索引，代表正确的类别。

```python
class torch.nn.NLLLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')
# weight         为每个类别手动指定的权重,应为长度为C的一维张量.默认为全1张量
# size_average   deprecated
# ignore_index   忽略所有类别为指定索引的样本
# reduce         deprecated
# reduction      指定对输出应用的归约方法.若为`'none'`,则不归约;若为`'sum'`,则对输出的所有元素求和;
#                若为`'mean'`,则对输出的所有元素求平均
```

* 输入形状：$(N,C)$，其中 $N$ 表示批次规模，$C$ 表示类别数；或 $(N,C,d_1,d_2,\cdots,d_k)$，其中 $d_i$ 表示额外的维度。
* 目标形状：$(N)$，其中 $N$ 表示批次规模，每一个值是 $[0, C-1]$ 范围内的整数索引；或 $(N,d_1,d_2,\cdots,d_k)$，其中 $d_i$ 表示额外的维度。
* 输出形状：标量；若 `reduction` 为 `'none'`，则与目标形状相同。

```python
>>> y = torch.tensor([[ 0.4377, -0.3976, -1.3221],
                      [ 1.8402, -0.1696,  0.4744],
                      [-3.4641, -0.2303,  0.3552]])
>>> lsm = nn.LogSoftmax(dim=1)
>>> y = lsm(y)
>>> y
tensor([[-0.4736, -1.3089, -2.2334],
        [-0.3287, -2.3385, -1.6945],
        [-4.2759, -1.0421, -0.4566]])
>>> t = torch.tensor([0, 0, 2])
>>> loss = nn.NLLLoss()
>>> loss(y, t)
tensor(0.4197)         # 0.4197 = (0.4736 + 0.3287 + 0.4566) / 3
>>> loss = nn.NLLLoss(ignore_index=2)
>>> loss(y, t)
tensor(0.4012)         # 0.4012 = (0.4736 + 0.3287) / 2
```

### L1Loss

平均绝对误差损失函数层。

$$
l_n=|y_n-t_n|\\
l=\sum_n l_n\ {\rm 或}\ l=\frac{1}{N}\sum_n l_n
$$

其中 $N$ 为批次规模。

支持实数值和复数值输入。

```python
class torch.nn.L1Loss(size_average=None, reduce=None, reduction='mean')
# size_average    deprecated
# reduce          deprecated
# reduction       指定对输出应用的归约方法.若为`'none'`,则不归约;若为`'sum'`,则对输出的所有元素求和;
#                 若为`'mean'`,则对输出的所有元素求平均
```

```python
>>> y = torch.randint(4, (4,)).to(torch.float32)
>>> t = torch.randint(4, (4,)).to(torch.float32)
>>> y
tensor([3., 2., 1., 3.])
>>> t
tensor([3., 1., 3., 1.])
>>> 
>>> loss = nn.L1Loss()
>>> loss(y, t)
tensor(1.2500)
>>> loss = nn.L1Loss(reduction='sum')
>>> loss(y, t)
tensor(5.)
```

## 数据并行模组

### DataParallel

### parallel.DistributedDataParallel

在模块级别实现基于 `torch.distributed` 包的分布式数据并行。

此容器通过沿批次维度分割输入数据并分配到各指定设备来并行化指定模块的运行。模块被复制到每一台机器和每一个设备上，每一个模型副本处理输入数据的一部分。在反向传递的过程中，来自每一个模型副本的梯度会被平均。

创建此类的对象需要 `torch.distributed` 已经初始化，通过调用 `torch.distributed.init_process_group()`。

要在一台有 N 个 GPU 的主机上使用 `DistributedDataParallel`，你需要 spawn N 个进程，并保证每个进程独占地使用一个 GPU。这可以通过为每个进程设定环境变量 `CUDA_VISIBLE_DEVICES` 或调用 `torch.cuda.set_device(i)` 来实现。在每一个进程中，你需要参照下面的方法构造模块：

```python
torch.distributed.init_process_group(
    backend='nccl', world_size=N, init_method='...'
)
model = DistributedDataParallel(model, device_ids=[i], output_device=i)
```

> 当使用 GPU 时，`nccl` 后端是目前最快和最推荐使用的后端。该后端同时适用于单节点和多节点分布式训练。

> 一个模型训练在 M 个节点上并且批次规模为 N，相比其训练在单个节点上并且批次规模为 MN，在损失在批次的各样本间求和（而非求平均）的情况下，前者的梯度将是后者的 M 分之一。当你想要得到一个与本地训练在数学上等价的分布式训练过程时，你需要将这一点考虑在内。但在大部分情况下，你可以将 `DistributedDataParallel` 包装的模型、 `DateParallel` 包装的模型和单个 GPU 上的普通模型同等对待。

> 模型参数不会在进程间广播；`DistributedDataParallel` 模块对梯度执行 All-Reduce 操作，并假定所有进程中的参数被优化器以同样的方式修改。缓冲区（例如 BatchNorm 数据）在每一次迭代中从 rank 0 进程的模型副本广播到所有其他模型副本。

```python
class torch.nn.parallel.DistributedDataParallel(module, device_ids=None, output_device=None, dim=0, broadcast_buffers=True, process_group=None, bucket_cap_mb=25, find_unused_parameters=False, check_reduction=False, gradient_as_bucket_view=False)
# module             要并行化的模块
# device_ids         1)对于单设备模块,`device_ids`只能包含一个设备的id,其代表此进程的模块所放置的CUDA设备
#                    2)对于多设备模块,`device_ids`必须为`None`
#                    在这两种情况下,当`device_ids`为`None`时,前向计算的输入数据和实际的模块都必须放置在
#                    正确的设备上
# output_device      单CUDA设备模块的输出放置的设备位置.对于多设备模块和CPU模块,此参数必须为`None`,并且模块本身
#                    决定了输出的位置
# broadcast_buffers
# process_group      用于进行分布式数据All-Reduce的进程组.若为`None`,则使用默认进程组,即由
#                    `torch.distributed.init_process_group()`创建的进程组
```

## 实用功能

### Flatten

展开张量的若干个连续维度，用于 `Sequential` 顺序模型。

```python
>>> input = torch.randn(2, 3, 4, 5)
>>> flatten = nn.Flatten()
>>> flatten(input).shape
torch.Size([2, 60])
>>> flatten = nn.Flatten(start_dim=1, end_dim=2)   # 展开第1到第2个维度
>>> flatten(input).shape
torch.Size([2, 12, 5])
```
