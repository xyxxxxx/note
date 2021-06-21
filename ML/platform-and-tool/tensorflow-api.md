[toc]

# tf

## broadcast_to()

将张量广播到兼容的形状。

```python
>>> a = tf.constant([1, 2, 3])
>>> tf.broadcast_to(a, [3, 3])
<tf.Tensor: shape=(3, 3), dtype=int32, numpy=
array([[1, 2, 3],
       [1, 2, 3],
       [1, 2, 3]], dtype=int32)>
```



## cast()

将张量转型为新类型。

```python
>>> the_f64_tensor = tf.constant([2.2, 3.3, 4.4], dtype=tf.float64)
>>> the_f16_tensor = tf.cast(the_f64_tensor, dtype=tf.float16)
>>> the_u8_tensor = tf.cast(the_f16_tensor, dtype=tf.uint8)
>>> the_u8_tensor
<tf.Tensor: shape=(3,), dtype=uint8, numpy=array([2, 3, 4], dtype=uint8)>
```



## concat()

沿指定维度拼接张量。见 `tf.tile()`，`tf.stack()`，`tf.repeat()`。

```python
>>> a1 = tf.constant([[1, 2, 3], [4, 5, 6]])
>>> a2 = tf.constant([[7, 8, 9], [10, 11, 12]])
>>> tf.concat([a1, a2], 0)
<tf.Tensor: shape=(4, 3), dtype=int32, numpy=
array([[ 1,  2,  3],
       [ 4,  5,  6],
       [ 7,  8,  9],
       [10, 11, 12]], dtype=int32)>
>>> tf.concat([a1, a2], 1)
<tf.Tensor: shape=(2, 6), dtype=int32, numpy=
array([[ 1,  2,  3,  7,  8,  9],
       [ 4,  5,  6, 10, 11, 12]], dtype=int32)>
```



## constant()

用类似张量的对象（python 数组，numpy 数组等）创建一个常数张量。

```python
>>> tf.constant([1, 2, 3, 4, 5, 6])
<tf.Tensor: shape=(6,), dtype=int32, numpy=array([1, 2, 3, 4, 5, 6], dtype=int32)>
  
>>> tf.constant(np.arange(1,7))
<tf.Tensor: shape=(6,), dtype=int64, numpy=array([1, 2, 3, 4, 5, 6])>
```



## device()

创建一个上下文，指定其中创建/执行的操作使用的设备。

```python
>>> with tf.device('CPU:0'):
...   a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
...   b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
...   c = tf.matmul(a, b)
>>> a.device
/job:localhost/replica:0/task:0/device:CPU:0
>>> b.device
/job:localhost/replica:0/task:0/device:CPU:0
>>> c.device
/job:localhost/replica:0/task:0/device:CPU:0
```



## expand_dims()

返回一个张量，其在输入张量的基础上在指定位置增加一个维度。

```python
>>> a = tf.reshape(tf.range(10.), [2,5])
>>> tf.expand_dims(a, 0).shape
TensorShape([1, 2, 5])
>>> tf.expand_dims(a, 1).shape
TensorShape([2, 1, 5])
```



## GradientTape

`tf.GradientTape()` 是一个自动求导的记录器。以下示例计算 $$y=x^2$$ 在 $$x=3$$ 位置的导数：

```python
import tensorflow as tf

x = tf.Variable(initial_value=3.)   # 初值为3.0的变量
with tf.GradientTape() as tape:     # 在 tf.GradientTape() 的上下文内，所有计算步骤都会被记录以用于求导
    y = tf.square(x)
y_grad = tape.gradient(y, x)        # 计算y关于x的导数
print(y, y_grad)                    # tf.Tensor(6.0, shape=(), dtype=float32)
```

以下示例计算 $$\mathcal{L}=||X\pmb w+b-\pmb y||^2$$ 在 $$\pmb w=[1,2]^{\rm T},b=1$$ 位置的对 $$\pmb w,b$$ 的导数：

```python
X = tf.constant([[1., 2.], [3., 4.]])
y = tf.constant([[1.], [2.]])
w = tf.Variable(initial_value=[[1.], [2.]])
b = tf.Variable(initial_value=1.)
with tf.GradientTape() as tape:
    L = tf.reduce_sum(tf.square(tf.matmul(X, w) + b - y))
    # tf.square 将输入张量的每个元素平方
    # tf.reduce_sum 对输入张量的所有元素求和,输出一个标量
w_grad, b_grad = tape.gradient(L, [w, b])        # 计算L(w, b)关于w, b的偏导数

print(L, w_grad, b_grad)
# tf.Tensor(125.0, shape=(), dtype=float32) tf.Tensor(
# [[ 70.]
#  [100.]], shape=(2, 1), dtype=float32) tf.Tensor(30.0, shape=(), dtype=float32)
```

可以看到计算结果
$$
\mathcal{L}=125，\\frac{\partial\mathcal{L}}{\partial\pmb w}=\begin{bmatrix}70\\100\end{bmatrix}，\\frac{\partial\mathcal{L}}{\partial b}=30
$$



## matmul()

张量乘法。`@` 符号重载了此方法。

```python
# 矩阵×矩阵: 矩阵乘法
>>> m1 = tf.reshape(tf.range(1., 10), [3, 3])
>>> m1 
<tf.Tensor: shape=(3, 3), dtype=float32, numpy=
array([[1., 2., 3.],
       [4., 5., 6.],
       [7., 8., 9.]], dtype=float32)>
>>> m1 @ m1
<tf.Tensor: shape=(3, 3), dtype=float32, numpy=
array([[ 30.,  36.,  42.],
       [ 66.,  81.,  96.],
       [102., 126., 150.]], dtype=float32)>

# 矩阵序列×矩阵序列: 逐元素的矩阵乘法
>>> bm1 = tf.repeat(tf.reshape(m1, [1, 3, 3]), repeats=2, axis=0)
>>> bm1
<tf.Tensor: shape=(2, 3, 3), dtype=float32, numpy=
array([[[1., 2., 3.],
        [4., 5., 6.],
        [7., 8., 9.]],

       [[1., 2., 3.],
        [4., 5., 6.],
        [7., 8., 9.]]], dtype=float32)>
>>> bm2 = tf.concat([tf.ones([1, 3, 3]), tf.ones([1, 3, 3]) * 2], 0)
>>> bm2
<tf.Tensor: shape=(2, 3, 3), dtype=float32, numpy=
array([[[1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.]],

       [[2., 2., 2.],
        [2., 2., 2.],
        [2., 2., 2.]]], dtype=float32)>
>>> bm1 @ bm2
<tf.Tensor: shape=(2, 3, 3), dtype=float32, numpy=
array([[[ 6.,  6.,  6.],
        [15., 15., 15.],
        [24., 24., 24.]],

       [[12., 12., 12.],
        [30., 30., 30.],
        [48., 48., 48.]]], dtype=float32)>
```



## ones()

生成指定形状的全 1 张量。

```python
>>> tf.ones([2,3])
<tf.Tensor: shape=(2, 3), dtype=float32, numpy=
array([[1., 1., 1.],
       [1., 1., 1.]], dtype=float32)>
>>> tf.ones([2,3], tf.int32)
<tf.Tensor: shape=(2, 3), dtype=int32, numpy=
array([[1, 1, 1],
       [1, 1, 1]], dtype=int32)>
```



## one_hot()

返回张量的独热编码。

```python
tf.one_hot(indices, depth, on_value=None, off_value=None, axis=None, dtype=None, name=None)
# indices     索引张量
# depth       独热维度的规模
# on_value    索引位置填入的值,默认为1
# off_value   非索引位置填入的值,默认为0
```

```python
>>> tf.one_hot([0, 1, 2], 3)
<tf.Tensor: shape=(3, 3), dtype=float32, numpy=
array([[1., 0., 0.],                       # indices[0] = 1.0
       [0., 1., 0.],                       # indices[1] = 1.0
       [0., 0., 1.]], dtype=float32)>      # indices[2] = 1.0
>>> tf.one_hot([0, 1, 2], 2)
<tf.Tensor: shape=(3, 2), dtype=float32, numpy=
array([[1., 0.],
       [0., 1.],
       [0., 0.]], dtype=float32)>
>>> tf.one_hot([0, 1, 2], 4)
<tf.Tensor: shape=(3, 4), dtype=float32, numpy=
array([[1., 0., 0., 0.],
       [0., 1., 0., 0.],
       [0., 0., 1., 0.]], dtype=float32)>
```





## RaggedTensor

某些维度上长度可变的张量类型。

```python
>>> ragged_list = [
...     [0, 1, 2, 3],
...     [4, 5],
...     [6, 7, 8],
...     [9]]
>>> ragged_tensor = tf.ragged.constant(ragged_list)
>>> ragged_tensor
<tf.RaggedTensor [[0, 1, 2, 3], [4, 5], [6, 7, 8], [9]]>
>>> ragged_tensor.shape
TensorShape([4, None])
```



## range()

根据给定的初值，末值和步长创建一维张量。与 python 的 `range()` 用法相同。

```python
>>> tf.range(10)
<tf.Tensor: shape=(10,), dtype=int32, numpy=array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int32)>
>>> tf.range(1, 10, 2)
<tf.Tensor: shape=(5,), dtype=int32, numpy=array([1, 3, 5, 7, 9], dtype=int32)>
```



## rank()

返回张量的维数。

```python
>>> a = tf.zeros([2,3])
>>> tf.rank(a)
<tf.Tensor: shape=(), dtype=int32, numpy=2>
#                                   维数=2
```



## repeat()

以重复输入列表元素的方式构建张量。

```python
>>> tf.repeat([[1, 2], [3, 4]], repeats=[2, 3], axis=0)   # 沿轴0,分别重复2次和3次
<tf.Tensor: shape=(5, 2), dtype=int32, numpy=
array([[1, 2],
       [1, 2],
       [3, 4],
       [3, 4],
       [3, 4]], dtype=int32)>
>>> tf.repeat([[1, 2], [3, 4]], repeats=[2, 3], axis=1)   # 沿轴1,分别重复2次和3次
<tf.Tensor: shape=(2, 5), dtype=int32, numpy=
array([[1, 1, 2, 2, 2],
       [3, 3, 4, 4, 4]], dtype=int32)>
>>> tf.repeat([[1, 2], [3, 4]], repeats=2)                # 展开为列表,每个元素重复2次
<tf.Tensor: shape=(8,), dtype=int32, numpy=array([1, 1, 2, 2, 3, 3, 4, 4], dtype=int32)>
```



## reshape()

改变张量的形状。

```python
>>> a = tf.range(10)
>>> a
<tf.Tensor: shape=(10,), dtype=int32, numpy=array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int32)>
>>> tf.reshape(a, [2,5])
<tf.Tensor: shape=(2, 5), dtype=int32, numpy=
array([[0, 1, 2, 3, 4],
       [5, 6, 7, 8, 9]], dtype=int32)>
>>> tf.reshape(a, [2,-1])                  # -1表示自动补全该位置的值
<tf.Tensor: shape=(2, 5), dtype=int32, numpy=
array([[0, 1, 2, 3, 4],
       [5, 6, 7, 8, 9]], dtype=int32)>
```



## reverse()

沿指定维度反转张量。

```python
>>> a = tf.reshape(tf.range(10), [2,5])
>>> a
<tf.Tensor: shape=(2, 5), dtype=int32, numpy=
array([[0, 1, 2, 3, 4],
       [5, 6, 7, 8, 9]], dtype=int32)>
>>> tf.reverse(a, [0])                   # 沿轴0反转
<tf.Tensor: shape=(2, 5), dtype=int32, numpy=
array([[5, 6, 7, 8, 9],
       [0, 1, 2, 3, 4]], dtype=int32)>
>>> tf.reverse(a, [1])                   # 沿轴1反转
<tf.Tensor: shape=(2, 5), dtype=int32, numpy=
array([[4, 3, 2, 1, 0],
       [9, 8, 7, 6, 5]], dtype=int32)>
```



## shape()

返回张量的形状。

```python
>>> a = tf.zeros([2,3])
>>> tf.shape(a)
<tf.Tensor: shape=(2,), dtype=int32, numpy=array([2, 3], dtype=int32)>
#                                             形状=[2,3]
```



## size()

返回张量的元素总数。

```python
>>> a = tf.zeros([2,3])
>>> tf.size(a)
<tf.Tensor: shape=(), dtype=int32, numpy=6>
#                                 元素总数=6
```



## squeeze()

返回一个张量，其在输入张量的基础上删除所有规模为 1 的维度。

```python
>>> a = tf.reshape(tf.range(10.), [1,2,1,5,1])
>>> a.shape
TensorShape([1, 2, 1, 5, 1])
>>> tf.squeeze(a).shape
TensorShape([2, 5])
```



## stack()

将张量的列表沿指定维度堆叠起来。

```python
>>> a1 = tf.constant([1, 4])   # [2,]
>>> a2 = tf.constant([2, 5])
>>> a3 = tf.constant([3, 6])
>>> tf.stack([a1, a2, a3], axis=0)             # [3,2]
<tf.Tensor: shape=(3, 2), dtype=int32, numpy=
array([[1, 4],
       [2, 5],
       [3, 6]], dtype=int32)>
>>> tf.stack([a1, a2, a3], axis=1)             # [2,3]
<tf.Tensor: shape=(2, 3), dtype=int32, numpy=
array([[1, 2, 3],
       [4, 5, 6]], dtype=int32)>
```



## Tensor

张量类型。





### device

张量位于的设备的名称。



### dtype

张量的元素的数据类型。

```python
>>> a = tf.zeros([2,3])
>>> a.dtype
tf.float32
```



### graph

包含张量的图。



### name

张量的字符串名称。



### op

产生此张量的运算。



### shape, ndim

张量的形状、维数。

```python
>>> a = tf.zeros([2,3])
>>> a.shape
TensorShape([2, 3])
>>> a.ndim
2
```



### index & slice op

```python
# 一维张量
rank_1_tensor = tf.constant([0, 1, 1, 2, 3, 5, 8, 13, 21, 34])
print("First:", rank_1_tensor[0].numpy())
print("Second:", rank_1_tensor[1].numpy())
print("Last:", rank_1_tensor[-1].numpy())
print("Everything:", rank_1_tensor[:].numpy())
print("Before 4:", rank_1_tensor[:4].numpy())
print("From 4 to the end:", rank_1_tensor[4:].numpy())
print("From 2, before 7:", rank_1_tensor[2:7].numpy())
print("Every other item:", rank_1_tensor[::2].numpy())
print("Reversed:", rank_1_tensor[::-1].numpy())

# 二维张量
rank_2_tensor = tf.constant([[1, 2],
                             [3, 4],
                             [5, 6]], dtype=tf.float16)
print(rank_2_tensor[1, 1].numpy())
print("Second row:", rank_2_tensor[1, :].numpy())
print("Second column:", rank_2_tensor[:, 1].numpy())
print("Last row:", rank_2_tensor[-1, :].numpy())
print("First item in last column:", rank_2_tensor[0, -1].numpy())
print("Skip the first row:")
print(rank_2_tensor[1:, :].numpy(), "\n")

# 三维张量
rank_3_tensor = tf.constant([
  [[0, 1, 2, 3, 4],
   [5, 6, 7, 8, 9]],
  [[10, 11, 12, 13, 14],
   [15, 16, 17, 18, 19]],
  [[20, 21, 22, 23, 24],
   [25, 26, 27, 28, 29]],])
print(rank_3_tensor[:, :, 4])
# tf.Tensor(
# [[ 4  9]
#  [14 19]
#  [24 29]], shape=(3, 2), dtype=int32)
```



## tile()

将张量在各维度上重复指定次数。

```python
>>> a = tf.constant([[1,2,3],[4,5,6]])
>>> a
<tf.Tensor: shape=(2, 3), dtype=int32, numpy=
array([[1, 2, 3],
       [4, 5, 6]], dtype=int32)>
>>> tf.tile(a, [1,2])
<tf.Tensor: shape=(2, 6), dtype=int32, numpy=
array([[1, 2, 3, 1, 2, 3],
       [4, 5, 6, 4, 5, 6]], dtype=int32)>
>>> tf.tile(a, [2,1])
<tf.Tensor: shape=(4, 3), dtype=int32, numpy=
array([[1, 2, 3],
       [4, 5, 6],
       [1, 2, 3],
       [4, 5, 6]], dtype=int32)>
```



## Variable

```python

```



## zeros()

生成指定形状的全 0 张量。

```python
>>> tf.zeros([2,3])
<tf.Tensor: shape=(2, 3), dtype=float32, numpy=
array([[0., 0., 0.],
       [0., 0., 0.]], dtype=float32)>
>>> tf.zeros([2,3], tf.int32)
<tf.Tensor: shape=(2, 3), dtype=int32, numpy=
array([[0, 0, 0],
       [0, 0, 0]], dtype=int32)>
```





# tf.config

## get_soft_device_placement()

返回软设备放置是否启用。

```python
>>> tf.config.set_soft_device_placement(True)
>>> tf.config.get_soft_device_placement()
True
>>> tf.config.set_soft_device_placement(False)
>>> tf.config.get_soft_device_placement()
False
```



## get_visible_devices()

返回运行时当前可见的 `PhysicalDevice` 对象的列表。

```python
physical_devices = tf.config.list_physical_devices('GPU')
try:
  # Disable all GPUS
  tf.config.set_visible_devices([], 'GPU')
  visible_devices = tf.config.get_visible_devices()
  for device in visible_devices:
    assert device.device_type != 'GPU'
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass
```



## list_logical_devices()

返回运行时创建的逻辑设备列表。

调用 `list_logical_devices()` 会引发运行时初始化所有可见的 `PhysicalDevice`，因而不能继续配置。若不想要初始化运行时，请调用 `list_physical_devices()`。

```python
>>> cpus = tf.config.list_physical_devices('CPU')
>>> cpus
[PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU')]
>>> tf.config.set_logical_device_configuration(              # 先设定逻辑设备配置
...   cpus[0],
...   [tf.config.LogicalDeviceConfiguration(),
...    tf.config.LogicalDeviceConfiguration()])
>>> logical_cpus = tf.config.list_logical_devices('CPU')     # 再调用list_logical_devices()以初始化
>>> logical_cpus
[LogicalDevice(name='/device:CPU:0', device_type='CPU'), LogicalDevice(name='/device:CPU:1', device_type='CPU')]

>>> gpus = tf.config.list_physical_devices('GPU')
>>> gpus
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
>>> logical_gpus = tf.config.list_logical_devices('GPU')     # 初始化之后
>>> logical_gpus
[LogicalDevice(name='/device:GPU:0', device_type='GPU')]
>>> tf.config.set_logical_device_configuration(              # 就不能再设定逻辑设备配置
...   gpus[0],
...   [tf.config.LogicalDeviceConfiguration(100),
...    tf.config.LogicalDeviceConfiguration(100)])
RuntimeError: Virtual devices cannot be modified after being initialized
```



## list_physical_devices()

返回运行时可见的物理设备列表。

物理设备指当前主机现有的硬件设备，包括所有已发现的 CPU 和 GPU 设备。此 API 用于在初始化运行时之前查询硬件资源，进而帮助调用更多的配置 API。

```python
>>> tf.config.list_physical_devices()
[PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU'),
 PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU'),
 PhysicalDevice(name='/physical_device:GPU:1', device_type='GPU')]

>>> tf.config.list_physical_devices('GPU')
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU'),
 PhysicalDevice(name='/physical_device:GPU:1', device_type='GPU')]
```



## LogicalDevice

初始化运行时得到的逻辑设备的抽象。

```python
tf.config.LogicalDevice(name, device_type)
```

一个 `LogicalDevice` 对象对应一个 `PhysicalDevice` 对象或者集群上的远程设备。张量或操作可以通过调用 `tf.device()` 并指定 `LogicalDevice`，而被放置在指定的逻辑设备上。



## LogicalDeviceConfiguration

逻辑设备的配置类。

```python
tf.config.LogicalDeviceConfiguration(memory_limit=None, experimental_priority=None)
# memory_limit   为逻辑设备分配的显存
```

此类用于在初始化运行时过程中，指定配置参数将 `PhysicalDevice` 初始化为 `LogicalDevice`。



## PhysicalDevice

本地物理设备的抽象。

```python
tf.config.PhysicalDevice(name, device_type)
```

TensorFlow 可以利用各种设备进行计算，例如 CPU 或者（多个）GPU。在初始化本地设备之前，用户可以自定义设备的一些属性，例如可见性或者内存配置。



## set_logical_device_configuration()

为一个 `PhysicalDevice` 对象设定逻辑设备配置。

一旦初始化运行时，一个可见的 `PhysicalDevice` 对象就默认创建一个 `LogicalDevice` 对象与之关联。指定 `LogicalDeviceConfiguration` 对象列表会在一个 `PhysicalDevice` 对象上创建多个 `LogicalDevice` 对象。

```python
# 将CPU分为2个逻辑设备
>>> cpus = tf.config.list_physical_devices('CPU')
>>> cpus
[PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU')]
>>> tf.config.set_logical_device_configuration(
...   cpus[0],
...   [tf.config.LogicalDeviceConfiguration(),
...    tf.config.LogicalDeviceConfiguration()])
>>> logical_cpus = tf.config.list_logical_devices('CPU')
>>> logical_cpus
[LogicalDevice(name='/device:CPU:0', device_type='CPU'), LogicalDevice(name='/device:CPU:1', device_type='CPU')]
```

```python
# 将GPU分为2个逻辑设备,每个分配100M显存
>>> gpus = tf.config.list_physical_devices('GPU')
>>> gpus 
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
>>> tf.config.set_logical_device_configuration(
...   gpus[0],
...   [tf.config.LogicalDeviceConfiguration(memory_limit=100),
...    tf.config.LogicalDeviceConfiguration(memory_limit=100)])
>>> logical_gpus = tf.config.list_logical_devices('GPU')
>>> logical_gpus
[LogicalDevice(name='/device:GPU:0', device_type='GPU'), LogicalDevice(name='/device:GPU:1', device_type='GPU')]
```



## experimental.set_memory_growth()

设定一个 `PhysicalDevice` 对象是否启用内存增长。

```python
>>> gpus = tf.config.list_physical_devices('GPU')
>>> tf.config.experimental.set_memory_growth(gpus[0], True)  # 启用内存增长
```



## set_soft_device_placement()

设定是否启用软设备放置。若启用，则当指定的设备不存在时自动选择可用的设备。

```python
>>> tf.config.set_soft_device_placement(True)
>>> tf.config.get_soft_device_placement()
True
>>> tf.config.set_soft_device_placement(False)
>>> tf.config.get_soft_device_placement()
False
```



## set_visible_devices()

指定运行时可见的 `PhysicalDevice` 对象的列表。TensorFlow 只会将张量和操作分配到可见的物理设备，因为 `LogicalDevice` 只能创建在可见的 `PhysicalDevice` 上。默认情况下，所有已发现的 CPU 和 GPU 设备都是可见的。

```python
>>> tf.config.list_physical_devices()                                                 # 所有物理设备
[PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU'),
 PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU'),
 PhysicalDevice(name='/physical_device:GPU:1', device_type='GPU')]
>>> tf.config.get_visible_devices()                                                   # 可见物理设备
[PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU'),
 PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU'),
 PhysicalDevice(name='/physical_device:GPU:1', device_type='GPU')]
>>> tf.config.set_visible_devices([], 'GPU')                                          # 设定GPU设备全部不可见
>>> tf.config.get_visible_devices()                                                   # 可见物理设备
[PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU')]
>>> tf.config.set_visible_devices(tf.config.list_physical_devices('GPU')[0], 'GPU')   # 设定GPU:0可见
>>> tf.config.get_visible_devices()                                                   # 可见物理设备
[PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU'),
 PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
>>> tf.config.list_logical_devices()                                                  # 所有虚拟设备
[LogicalDevice(name='/device:CPU:0', device_type='CPU'),
 LogicalDevice(name='/device:GPU:0', device_type='GPU')]
```





# tf.dataset

## Dataset

数据集类型，用于构建高效的输入流水线。`Dataset` 实例的使用通常遵循以下模式：

1. 根据输入数据创建一个数据集
2. 应用数据集变换以预处理数据
3. 迭代数据集并处理样本

迭代以流的方式进行，因此不会将整个数据集全部放到内存中。



### apply()

为数据集应用一个变换函数，返回一个新的数据集。该变换函数通常是 `Dataset` 实例的变换方法的组合。

```python
>>> ds = Dataset.range(100)
>>> def dataset_fn(ds):
  return ds.filter(lambda x: x < 5)
... 
>>> ds = ds.apply(dataset_fn)
>>> list(ds.as_numpy_iterator())
[0, 1, 2, 3, 4]
```



### as_numpy_iterator()

返回一个迭代器，其中数据集的所有元素都被转换为 NumPy 数组。

```python
>>> ds = Dataset.range(5)
```



### batch()

将数据集的连续元素组合为批，返回一个新的数据集。

```python
>>> list(Dataset.range(10).batch(3).as_numpy_iterator())
[array([0, 1, 2]), array([3, 4, 5]), array([6, 7, 8]), array([9])]
>>> list(Dataset.range(10).batch(3, drop_remainder=True).as_numpy_iterator())   # 丢弃达不到指定规模的最后一个批次
[array([0, 1, 2]), array([3, 4, 5]), array([6, 7, 8])]
```



### cache()

返回一个新的数据集，该数据集的元素被缓存到内存中。



### cardinality()

返回数据集的元素个数。如果数据集的元素个数是无限或不能确定，则返回 `tf.data.INFINITE_CARDINALITY`（`-1`） 或 `tf.data.UNKNOWN_CARDINALITY`（`-2`）。

```python
>>> Dataset.range(100).cardinality()
<tf.Tensor: shape=(), dtype=int64, numpy=100>
>>>
>>> Dataset.range(100).repeat().cardinality()
<tf.Tensor: shape=(), dtype=int64, numpy=-1>
>>> Dataset.range(100).repeat().cardinality() == tf.data.INFINITE_CARDINALITY
<tf.Tensor: shape=(), dtype=bool, numpy=True>
>>>
>>> Dataset.range(100).filter(lambda x: True).cardinality()
<tf.Tensor: shape=(), dtype=int64, numpy=-2>
>>> Dataset.range(100).filter(lambda x: True).cardinality() == tf.data.UNKNOWN_CARDINALITY
<tf.Tensor: shape=(), dtype=bool, numpy=True>
```







### filter()







### from_generator()

创建由



### from_tensor_slices()

创建由指定张量的元素构成的数据集。

```python
>>> ds = Dataset.from_tensor_slices([1, 2, 3])                  # 张量
>>> list(ds.as_numpy_iterator())
[1, 2, 3]                         # 张量的元素
>>> ds = Dataset.from_tensor_slices([[1, 2, 3], [4, 5, 6]])
>>> list(ds.as_numpy_iterator())
[array([1, 2, 3], dtype=int32), array([4, 5, 6], dtype=int32)]
>>> 
>>> ds = Dataset.from_tensor_slices(([1, 2, 3], [3, 4], [5, 6]))    # 张量构成的元组
>>> list(ds.as_numpy_iterator())
[(1, 3, 5), (2, 4, 6)]                  # 元组,元素来自各张量
>>> ds = Dataset.from_tensor_slices(([1, 2, 3], ['a', 'b', 'a']))   # 应用:绑定数据和标签
>>> list(ds.as_numpy_iterator())
[(1, b'a'), (2, b'b'), (3, b'a')]
>>> ds = Dataset.from_tensor_slices({"a": [1, 2], "b": [3, 4], "c": [5, 6]})  # 张量构成的字典
>>> list(ds.as_numpy_iterator())
[{'a': 1, 'b': 3, 'c': 5}, {'a': 2, 'b': 4, 'c': 6}]    # 字典,元素来自各张量
```



### from_tensors()

创建由单个张量元素构成的数据集。

```python
>>> ds = Dataset.from_tensors([1, 2, 3])
>>> list(ds.as_numpy_iterator())
[array([1, 2, 3], dtype=int32)]
>>> 
>>> ds = Dataset.from_tensors([[1, 2, 3], [4, 5, 6]])
>>> list(ds.as_numpy_iterator())
[array([[1, 2, 3],
       [4, 5, 6]], dtype=int32)]
```



### list_files()



### map()



### range()

创建由等差数列构成的数据集。

```python
>>> Dataset.range(5)
<RangeDataset shapes: (), types: tf.int64>
>>> list(Dataset.range(5).as_numpy_iterator())
[0, 1, 2, 3, 4]
>>> Dataset.range(1, 5, 2, output_type=tf.float32)
<RangeDataset shapes: (), types: tf.float32>
>>> list(Dataset.range(1, 5, 2, output_type=tf.float32).as_numpy_iterator())
[1.0, 3.0]
```



### reduce()

将数据集归约为一个元素。

```python
reduce(initial_state, reduce_func)
# initial_state    初始状态
# reduce_func      归约函数,将`(old_state, element)`映射到`new_state`.此函数会被不断地调用直到数据集被耗尽
```

```python
>>> Dataset.range(5).reduce(np.int64(0), lambda x, y: x + y)
<tf.Tensor: shape=(), dtype=int64, numpy=10>
>>> Dataset.range(5).reduce(np.int64(0), lambda x, _: x + 1)
<tf.Tensor: shape=(), dtype=int64, numpy=5>
```



### repeat()

重复数据集多次，返回一个新的数据集。

```python
>>> list(Dataset.range(5).repeat(3).as_numpy_iterator())
[0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4]

>>> pprint(list(Dataset.range(100).shuffle(100).repeat(2).batch(10).as_numpy_iterator()))
[array([63, 46, 66, 70, 96, 98,  7, 52, 50, 37]),
 array([45, 60, 69, 61, 84, 41, 22, 27, 14, 57]),
 array([ 9, 81,  8, 75,  4, 87, 18,  1, 51, 76]),
 array([65, 59, 90, 23, 39, 74, 26,  3, 20, 78]),
 array([91, 68, 85, 24, 53, 55, 16, 83, 94, 86]),
 array([92, 54, 48, 93, 38, 13, 67, 71, 82, 56]),
 array([40, 62,  5, 33, 15, 99, 32, 17,  6, 25]),
 array([80, 43, 28, 77, 21, 11, 72, 88, 44, 89]),
 array([31, 19, 12, 47, 36, 95, 29, 34,  0, 10]),
 array([97,  2, 42, 30, 64, 49, 79, 58, 35, 73]),
 array([19, 13, 77, 51, 56, 46, 67, 48, 27, 89]),
 array([26, 72, 54, 14,  2, 18, 25, 44, 63, 82]),
 array([24, 10, 23, 68, 64, 39, 28, 52, 53, 38]),
 array([69,  1, 15, 92, 31, 49, 60, 33, 81, 37]),
 array([65, 40, 20, 50,  5, 90, 34, 97, 84,  7]),
 array([99,  0, 75, 59, 98,  3, 85, 83, 61,  8]),
 array([73, 35, 36, 76, 55, 96, 91, 21, 94,  6]),
 array([70, 47, 16, 86, 11, 57, 95, 45, 74, 43]),
 array([87, 62, 29, 17, 12, 22, 66, 93, 88, 42]),
 array([79,  4, 80, 78, 32, 30, 71,  9, 58, 41])]
```



### shard()

从数据集中等间距地抽取样本以构成新的数据集并返回。此方法在进行分布式训练时十分有用。

```python
shard(num_shards, index)
# num_shards, index   每`num_shards`个样本抽取第`index`个
```

```python
>>> ds = Dataset.range(10)
>>> ds0 = ds.shard(num_shards=3, index=0)
>>> list(ds0.as_numpy_iterator())
[0, 3, 6, 9]
>>> ds1 = ds.shard(num_shards=3, index=1)
>>> list(ds1.as_numpy_iterator())
[1, 4, 7]
>>> ds2 = ds.shard(num_shards=3, index=2)
>>> list(ds2.as_numpy_iterator())
[2, 5, 8]
```

```python
# 分布式训练准备数据集
d = tf.data.TFRecordDataset(input_file)
d = d.shard(num_workers, worker_index)
d = d.repeat(num_epochs)
d = d.shuffle(shuffle_buffer_size)
d = d.map(parser_fn, num_parallel_calls=num_map_threads)
```



### shuffle()

随机打乱数据集中的元素，返回一个新的数据集。

```python
shuffle(buffer_size, seed=None, reshuffle_each_iteration=True)
# buffer_size                缓冲区大小.例如数据集包含1000个元素而`buffer_size`设为100,那么前100个元素首先进入缓冲区,
#                            从中随机抽取一个,然后第101个元素进入缓冲区,再随机抽取一个,...
# seed                       随机数种子
# reshuffle_each_iteration   若为`True`,则数据集每迭代完成一次都会重新打乱
```

```python
>>> list(Dataset.range(5).shuffle(5).as_numpy_iterator())    # 随机打乱
[1, 0, 2, 4, 3]
>>> 
>>> list(Dataset.range(5).shuffle(2).as_numpy_iterator())    # 缓冲区设为2
[0, 2, 1, 3, 4]   # 首先从0,1中抽取到0,再从1,2中抽取到2,再从1,3中抽取到1,...
>>> 
>>> ds = Dataset.range(5).shuffle(5)
>>> list(ds.as_numpy_iterator())                 # 每次迭代的顺序不同
[1, 0, 3, 4, 2]
>>> list(ds.as_numpy_iterator())
[2, 0, 1, 4, 3]
>>> list(ds.repeat(3).as_numpy_iterator())
[0, 1, 3, 2, 4, 2, 0, 4, 3, 1, 2, 4, 3, 0, 1]    # 即使调用`repeat()`,每次迭代的顺序也不同
>>> 
>>> ds = Dataset.range(5).shuffle(5, reshuffle_each_iteration=False)
>>> list(ds.as_numpy_iterator())
[1, 4, 2, 3, 0]
>>> list(ds.as_numpy_iterator())                 # 每次迭代的顺序相同
[1, 4, 2, 3, 0]
```



### skip()

去除数据集的前几个元素以构成新的数据集并返回。

```python
>>> list(Dataset.range(5).skip(2).as_numpy_iterator())
[2, 3, 4]
```



### take()

用数据集的前几个元素构成新的数据集并返回。

```python
>>> list(Dataset.range(5).take(2).as_numpy_iterator())
[0, 1]
```



### unbatch()

将数据集的元素（批）拆分为多个元素，返回一个新的数据集。

```python
>>> elements = [[1, 2, 3], [1, 2], [1, 2, 3, 4]]
>>> ds = Dataset.from_generator(lambda: elements, tf.int64)
>>> list(ds.as_numpy_iterator())
[array([1, 2, 3]), array([1, 2]), array([1, 2, 3, 4])]
>>> list(ds.unbatch().as_numpy_iterator())
[1, 2, 3, 1, 2, 1, 2, 3, 4]
```



### window()

将数据集中的相邻元素组合为窗口（窗口也是一个小规模的数据集），由这些窗口构成新的数据集并返回。

```python
window(size, shift=None, stride=1, drop_remainder=False)
# size    窗口的元素数量
# shift   窗口的移动距离
# stride  取样的间距
# drop_remainder   丢弃最后一个规模不足`size`的窗口
```

```python
```



### zip()

组合多个数据集的对应元素以构成新的数据集并返回。类似于 Python 的内置函数 `zip()`。

```python
>>> ds1 = Dataset.range(1, 4)
>>> ds2 = Dataset.range(4, 7)
>>> ds = Dataset.zip((ds1, ds2))
>>> list(ds.as_numpy_iterator())
[(1, 4), (2, 5), (3, 6)]
>>> 
>>> ds3 = Dataset.range(7, 13).batch(2)
>>> ds = Dataset.zip((ds1, ds2, ds3))    # 数据集的元素类型不要求相同
>>> list(ds.as_numpy_iterator())
[(1, 4, array([7, 8])), (2, 5, array([ 9, 10])), (3, 6, array([11, 12]))]
>>> 
>>> ds4 = Dataset.range(13, 15)
>>> ds = Dataset.zip((ds1, ds2, ds4))    # 数据集的元素数量不要求相同,受限于数量最少的数据集
>>> list(ds.as_numpy_iterator())
[(1, 4, 13), (2, 5, 14)]
```









## TextLineDataset

```python
dataset = tf.data.TextLineDataset(["file1.txt", "file2.txt"])
```









# tf.distribute

> 此模块设计复杂，难以使用，越来越多的用户开始使用 [Horovod](./hovorod.md)。

分布式训练。



## CentralStorageStrategy

中央存储策略。



## CrossDeviceOps

跨设备归约和广播算法的基类，用于实现不同的跨设备通信方法并传入到 `MirroredStrategy`。

`ReductionToOneDevice`, `NcclAllReduce` 和 `HierarchicalCopyAllReduce` 是 `CrossDeviceOps` 的子类，实现了具体的归约算法。



## DistributedDataset





## DistributedIterator





## DistributedValues





## get_replica_context()

返回当前的 `ReplicaContext` 实例。

```python
>>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
>>> print(tf.distribute.get_replica_context())    # 在默认分布式策略下,返回默认模型副本上下文实例
<tensorflow.python.distribute.distribute_lib._DefaultReplicaContext object at 0x7f1b8057e190>
>>> with strategy.scope():                        # 在`MirroredStrategy`下,返回`None`
  print(tf.distribute.get_replica_context())
None
>>> def f():
  return tf.distribute.get_replica_context()
>>> strategy.run(f)
PerReplica:{                                      # 在`strategy.run()`下,返回镜像模型副本上下文实例,为此函数的通常用法
  0: <tensorflow.python.distribute.mirrored_run._MirroredReplicaContext object at 0x7f1b805c0610>,
  1: <tensorflow.python.distribute.mirrored_run._MirroredReplicaContext object at 0x7f1b805c0390>
}
```



## get_strategy()

返回当前的 `Strategy` 实例。

```python
>>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
>>> print(tf.distribute.get_strategy())           # 默认分布式策略实例
<tensorflow.python.distribute.distribute_lib._DefaultDistributionStrategy object at 0x7fb547e94bd0>
>>> with strategy.scope():
  print(tf.distribute.get_strategy())             # `MirroredStrategy`实例
<tensorflow.python.distribute.mirrored_strategy.MirroredStrategy object at 0x7fb54795bf50>
```



## has_strategy()

返回当前是否为非默认的 `Strategy` 实例。

```python
>>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
>>> print(tf.distribute.has_strategy())
False
>>> with strategy.scope():
  print(tf.distribute.has_strategy())
True
```



## HierarchicalCopyAllReduce

hierarchical copy all-reduce 算法的实现。



## in_cross_replica_context()

返回当前是否为跨模型副本的上下文。

```python
>>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
>>> print(tf.distribute.in_cross_replica_context())
False
>>> with strategy.scope():
  print(tf.distribute.in_cross_replica_context())
True
>>> def f():
  return tf.distribute.in_cross_replica_context()
>>> strategy.run(f)
False
```



## MirroredStrategy

单机多卡同步训练。此策略下模型的参数是 `MirroredVariable` 类型的变量，在所有的模型副本中保持同步。

```python
tf.distribute.MirroredStrategy(devices=None, cross_device_ops=None)
# devices             设备列表.若为`None`或空列表,则使用所有可用的GPU;若没有发现GPU,则使用可用的CPU
#                     注意TensorFlow将一台机器上的多核CPU视作单个设备,并且隐式使用线程并行
# cross_device_ops    `CrossDeviceOps`的子类的实例,默认使用`NcclAllReduce()`
```

```python
>>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
>>> with strategy.scope():
  x = tf.Variable(1.)       # 在`MirroredStrategy`下创建的变量是一个`MirroredVariable`
>>> x
MirroredVariable:{
  0: <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=1.0>,
  1: <tf.Variable 'Variable/replica_1:0' shape=() dtype=float32, numpy=1.0>
}
```



### cluster_resolver









## experimental.MultiWorkerMirroredStrategy

多机多卡同步训练。此策略下模型的参数是 `MirroredVariable` 类型的变量，在所有的模型副本中保持同步。

```python
```





## NcclAllReduce

Nvidia NCCL all-reduce 算法的实现。默认使用的 all-reduce 算法。



## OneDeviceStrategy

在单个设备上运行。在此策略下创建的变量和通过 `strategy.run()` 调用的函数都会被放置在指定设备上。此策略通常用于测试代码对于 `tf.distribute.Strategy` API 的使用。

```python
tf.distribute.OneDeviceStrategy(device)
```

```python
>>> strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
>>> with strategy.scope():
  v = tf.Variable(1.0)
  print(v.device)
/job:localhost/replica:0/task:0/device:GPU:0
>>> def step_fn(x):
  return x * 2
>>> result = 0
>>> for i in range(10):
  result += strategy.run(step_fn, args=(i,))
>>> print(result)
90
```



## ReduceOp

值的归约方法，`ReduceOp.SUM` 表示求和，`ReduceOp.MEAN` 表示求平均值。



## ReductionToOneDevice



## ReplicaContext

此类具有在模型副本上下文中调用的一系列 API，其实例通常由 `get_replica_context()` 得到，用于在 `strategy.run()` 传入的函数中调用以获取模型副本的信息。

```python
>>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
>>> def f():
  replica_context = tf.distribute.get_replica_context()
  return replica_context.replica_id_in_sync_group
>>> strategy.run(f)
PerReplica:{
  0: <tf.Tensor: shape=(), dtype=int32, numpy=0>,
  1: <tf.Tensor: shape=(), dtype=int32, numpy=1>
}
```



### all_gather



### all_reduce



### merge_call



### num_replicas_in_sync

返回进行梯度汇总的模型副本的数量。



### replica_id_in_sync_group

返回模型副本的索引。



### strategy

返回当前的 `Strategy` 实例。





## Server





## Strategy

在一组设备上的分布式计算策略。





### gather()



### num_replicas_in_sync

返回进行梯度汇总的模型副本的数量。



### reduce()



### run()

```python
run(fn, args=(), kwargs=None, options=None)
```

在每个模型副本上调用 `fn`，使用给定的参数。









### scope()

返回一个上下文管理器，







# tf.image

图像操作。

## adjust_brightness()



## adjust_contrast()



## adjust_gamma()



## adjust_hue()





# tf.linalg

## det()

返回一个或多个方阵的行列式。

```python
>>> a = tf.constant([[1., 2], [3, 4]])
>>> a
<tf.Tensor: shape=(2, 2), dtype=float32, numpy=
array([[1., 2.],
       [3., 4.]], dtype=float32)>
>>> tf.linalg.det(a)
<tf.Tensor: shape=(), dtype=float32, numpy=-2.0>
```



## diag()

返回一批对角矩阵，对角值由输入的一批向量给定。

```python
diagonal = np.array([[1, 2, 3, 4],            # Input shape: (2, 4)
                     [5, 6, 7, 8]])
tf.matrix_diag(diagonal) ==> [[[1, 0, 0, 0],  # Output shape: (2, 4, 4)
                               [0, 2, 0, 0],
                               [0, 0, 3, 0],
                               [0, 0, 0, 4]],
                              [[5, 0, 0, 0],
                               [0, 6, 0, 0],
                               [0, 0, 7, 0],
                               [0, 0, 0, 8]]]
```



## eigh()

返回张量的一个特征分解 $$A=Q\Lambda Q^{-1}$$。



## svd()

返回张量的一个奇异值分解 $$A=U\Sigma V^*$$。





# tf.math

## abs()

张量逐元素应用绝对值函数。

```python
>>> tf.abs(tf.constant([-1, -2, 3]))
<tf.Tensor: shape=(3,), dtype=int32, numpy=array([1, 2, 3], dtype=int32)>
```



## add(), subtract()

张量加法/减法。`+,-` 符号重载了这些方法。

```python
>>> a = tf.reshape(tf.range(12), [3, 4])
>>> a
<tf.Tensor: shape=(3, 4), dtype=int32, numpy=
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]], dtype=int32)>
>>> a + 1                    # 张量+标量: 扩张的张量加法
<tf.Tensor: shape=(3, 4), dtype=int32, numpy=
array([[ 1,  2,  3,  4],
       [ 5,  6,  7,  8],
       [ 9, 10, 11, 12]], dtype=int32)>
>>> a + tf.constant([1])     # 同前
<tf.Tensor: shape=(3, 4), dtype=int32, numpy=
array([[ 1,  2,  3,  4],
       [ 5,  6,  7,  8],
       [ 9, 10, 11, 12]], dtype=int32)>
>>> a + tf.range(4)          # 张量+子张量: 扩张的张量加法
<tf.Tensor: shape=(3, 4), dtype=int32, numpy=
array([[ 0,  2,  4,  6],
       [ 4,  6,  8, 10],
       [ 8, 10, 12, 14]], dtype=int32)>
>>> a + a                    # 张量+张量: 张量加法
<tf.Tensor: shape=(3, 4), dtype=int32, numpy=
array([[ 0,  2,  4,  6],
       [ 8, 10, 12, 14],
       [16, 18, 20, 22]], dtype=int32)>
```



## argmax(), argmin()

返回张量沿指定维度的最大值的索引。

```python
>>> a = tf.random.normal([4])
>>> a
<tf.Tensor: shape=(4,), dtype=float32, numpy=array([ 0.80442244,  0.01440545, -0.9266029 ,  0.23776768], dtype=float32)>
>>> tf.argmax(a)
<tf.Tensor: shape=(), dtype=int64, numpy=0>
>>> tf.argmin(a)
<tf.Tensor: shape=(), dtype=int64, numpy=2>

>>> a = tf.random.normal([4, 4])
>>> a
<tf.Tensor: shape=(4, 4), dtype=float32, numpy=
array([[ 1.2651453 , -0.9885311 , -1.9029404 ,  1.0343136 ],
       [ 0.4773587 ,  1.2282255 ,  0.66903603, -1.9187453 ],
       [ 0.94859433, -0.50704604,  1.6308597 ,  0.517232  ],
       [ 0.5004154 ,  0.38485277,  0.9955068 , -1.865893  ]],
      dtype=float32)>
>>> tf.argmax(a, 1)
<tf.Tensor: shape=(4,), dtype=int64, numpy=array([0, 1, 2, 2])>
>>> tf.argmin(a, 1)
<tf.Tensor: shape=(4,), dtype=int64, numpy=array([2, 3, 1, 3])>
```



## ceil()

张量逐元素应用向上取整。

```python
>>> tf.math.ceil([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
<tf.Tensor: shape=(7,), dtype=float32, numpy=array([-1., -1., -0.,  1.,  2.,  2.,  2.], dtype=float32)>
```



## equal()

逐元素判断两个张量是否相等。`==` 符号重载了此方法。

```python
>>> one1 = tf.ones([2,3])
>>> one2 = tf.ones([2,3])
>>> one1 == one2
<tf.Tensor: shape=(2, 3), dtype=bool, numpy=
array([[ True,  True,  True],
       [ True,  True,  True]])>
>>> tf.equal(one1, one2)
<tf.Tensor: shape=(2, 3), dtype=bool, numpy=
array([[ True,  True,  True],
       [ True,  True,  True]])>
```



## exp()

张量逐元素应用自然指数函数。

```python
>>> a = tf.reshape(tf.range(10.), [2,5])
>>> a
<tf.Tensor: shape=(2, 5), dtype=float32, numpy=
array([[0., 1., 2., 3., 4.],
       [5., 6., 7., 8., 9.]], dtype=float32)>
>>> tf.exp(a)
<tf.Tensor: shape=(2, 5), dtype=float32, numpy=
array([[1.0000000e+00, 2.7182817e+00, 7.3890562e+00, 2.0085537e+01,
        5.4598152e+01],
       [1.4841316e+02, 4.0342880e+02, 1.0966332e+03, 2.9809580e+03,
        8.1030840e+03]], dtype=float32)>
```



## floor()

张量逐元素应用向下取整。

```python
>>> tf.math.floor([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
<tf.Tensor: shape=(7,), dtype=float32, numpy=array([-2., -2., -1.,  0.,  1.,  1.,  2.], dtype=float32)>
```



## greater(), greater_equal(), less(), less_equal()

逐元素比较两个张量的大小。`>,>=,<,<=` 符号重载了这些方法。

```python
>>> a = tf.constant([5, 4, 6])
>>> b = tf.constant([5, 2, 5])
>>> a > b
<tf.Tensor: shape=(3,), dtype=bool, numpy=array([False,  True,  True])>
>>> tf.greater(a, b)
<tf.Tensor: shape=(3,), dtype=bool, numpy=array([False,  True,  True])>
>>> a >= b
<tf.Tensor: shape=(3,), dtype=bool, numpy=array([ True,  True,  True])>
>>> tf.greater_equal(a, b)
<tf.Tensor: shape=(3,), dtype=bool, numpy=array([ True,  True,  True])>
```



## log()

张量逐元素应用自然对数函数。注意 TensorFlow 没有 `log2()` 和 `log10()` 函数。

```python
>>> a = tf.reshape(tf.range(10.), [2,5])
>>> a
<tf.Tensor: shape=(2, 5), dtype=float32, numpy=
array([[0., 1., 2., 3., 4.],
       [5., 6., 7., 8., 9.]], dtype=float32)>
>>> tf.math.log(a)
<tf.Tensor: shape=(2, 5), dtype=float32, numpy=
array([[     -inf, 0.       , 0.6931472, 1.0986123, 1.3862944],
       [1.609438 , 1.7917595, 1.9459102, 2.0794415, 2.1972246]],
      dtype=float32)>
```



## maximum(), minimum()

逐元素取两个张量的较大值、较小值。

```python
>>> a = tf.constant([0., 0., 0., 0.])
>>> b = tf.constant([-2., 0., 2., 5.])
>>> tf.math.maximum(a, b)
<tf.Tensor: shape=(4,), dtype=float32, numpy=array([0., 0., 2., 5.], dtype=float32)>
>>> tf.math.minimum(a, b)
<tf.Tensor: shape=(4,), dtype=float32, numpy=array([-2.,  0.,  0.,  0.], dtype=float32)>
```



## multiply(), divide()

张量逐元素乘法/除法。`*,/` 符号重载了此方法。

```python
>>> a = tf.reshape(tf.range(12), [3,4])
>>> a
<tf.Tensor: shape=(3, 4), dtype=int32, numpy=
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]], dtype=int32)>
>>> a * 100                                      # 张量 * 标量: 张量的数乘
<tf.Tensor: shape=(3, 4), dtype=int32, numpy=
array([[   0,  100,  200,  300],
       [ 400,  500,  600,  700],
       [ 800,  900, 1000, 1100]], dtype=int32)>
>>> a * tf.range(4)                              # 张量 * 子张量: 张量的扩张逐元素乘法
<tf.Tensor: shape=(3, 4), dtype=int32, numpy=
array([[ 0,  1,  4,  9],
       [ 0,  5, 12, 21],
       [ 0,  9, 20, 33]], dtype=int32)>
>>> a * a                                        # 张量 * 张量: 张量的逐元素乘法
<tf.Tensor: shape=(3, 4), dtype=int32, numpy=
array([[  0,   1,   4,   9],
       [ 16,  25,  36,  49],
       [ 64,  81, 100, 121]], dtype=int32)>

>>> a = tf.reshape(tf.range(1,4),[3,1])
>>> b = tf.range(1,5)
>>> a * b                                        # 一维张量 * 一维张量: 向量外积
<tf.Tensor: shape=(3, 4), dtype=int32, numpy=
array([[ 1,  2,  3,  4],
       [ 2,  4,  6,  8],
       [ 3,  6,  9, 12]], dtype=int32)>
>>> a * b
<tf.Tensor: shape=(3, 4), dtype=int32, numpy=
array([[ 1,  2,  3,  4],
       [ 2,  4,  6,  8],
       [ 3,  6,  9, 12]], dtype=int32)>
```



## pow()

张量逐元素幂乘。`**` 符号重载了此方法。

```python
>>> a = tf.constant([[2, 2], [3, 3]])
>>> a ** 2
<tf.Tensor: shape=(2, 2), dtype=int32, numpy=
array([[4, 4],
       [9, 9]], dtype=int32)>
>>> a ** tf.range(2)
<tf.Tensor: shape=(2, 2), dtype=int32, numpy=
array([[1, 2],
       [1, 3]], dtype=int32)>
>>> b = tf.constant([[8, 16], [2, 3]])
>>> a ** b
<tf.Tensor: shape=(2, 2), dtype=int32, numpy=
array([[  256, 65536],
       [    9,    27]], dtype=int32)>
```



## reduce_max(), reduce_min(), reduce_mean(), reduce_std()

计算张量沿指定维度的最大值、最小值、平均值和标准差。

```python
>>> a = tf.reshape(tf.range(10.), [2,5])
>>> a
<tf.Tensor: shape=(2, 5), dtype=float32, numpy=
array([[0., 1., 2., 3., 4.],
       [5., 6., 7., 8., 9.]], dtype=float32)>

>>> tf.reduce_max(a)
<tf.Tensor: shape=(), dtype=float32, numpy=9.0>
>>> tf.reduce_max(a, 0)
<tf.Tensor: shape=(5,), dtype=float32, numpy=array([5., 6., 7., 8., 9.], dtype=float32)>
>>> tf.reduce_max(a, 1)
<tf.Tensor: shape=(2,), dtype=float32, numpy=array([4., 9.], dtype=float32)>
  
>>> tf.reduce_min(a)
<tf.Tensor: shape=(), dtype=float32, numpy=0.0>
>>> tf.reduce_min(a, 0)
<tf.Tensor: shape=(5,), dtype=float32, numpy=array([0., 1., 2., 3., 4.], dtype=float32)>
>>> tf.reduce_min(a, 1)
<tf.Tensor: shape=(2,), dtype=float32, numpy=array([0., 5.], dtype=float32)>
  
>>> tf.reduce_mean(a)
<tf.Tensor: shape=(), dtype=float32, numpy=4.5>
>>> tf.reduce_mean(a, 0)
<tf.Tensor: shape=(5,), dtype=float32, numpy=array([2.5, 3.5, 4.5, 5.5, 6.5], dtype=float32)>
>>> tf.reduce_mean(a, 1)
<tf.Tensor: shape=(2,), dtype=float32, numpy=array([2., 7.], dtype=float32)>

>>> tf.math.reduce_std(a)
<tf.Tensor: shape=(), dtype=float32, numpy=2.8722813>  
>>> tf.math.reduce_std(a, 0)
<tf.Tensor: shape=(5,), dtype=float32, numpy=array([2.5, 2.5, 2.5, 2.5, 2.5], dtype=float32)>
>>> tf.math.reduce_std(a, 1)
<tf.Tensor: shape=(2,), dtype=float32, numpy=array([1.4142135, 1.4142135], dtype=float32)>
```



## reduce_sum()

计算张量沿指定维度的元素和。

```python
>>> a = tf.constant([[1, 1, 1], [1, 1, 1]])
>>> tf.reduce_sum(a)                    # 求和
<tf.Tensor: shape=(), dtype=int32, numpy=6>
>>> tf.reduce_sum(a, 0)                 # 沿轴0
<tf.Tensor: shape=(3,), dtype=int32, numpy=array([2, 2, 2], dtype=int32)>
>>> tf.reduce_sum(a, 0, keepdims=True)  # 沿轴0并保持张量形状
<tf.Tensor: shape=(1, 3), dtype=int32, numpy=array([[2, 2, 2]], dtype=int32)>
>>> tf.reduce_sum(a, 1)                 # 沿轴1
<tf.Tensor: shape=(2,), dtype=int32, numpy=array([3, 3], dtype=int32)>
>>> tf.reduce_sum(a, 1, keepdims=True)  # 沿轴1并保持张量形状
<tf.Tensor: shape=(2, 1), dtype=int32, numpy=
array([[3],
       [3]], dtype=int32)>
>>> tf.reduce_sum(a, [0,1])             # 同时沿轴0和轴1,即对所有元素求和
<tf.Tensor: shape=(), dtype=int32, numpy=6>
```



## round()

张量逐元素应用舍入函数，0.5 会向偶数取整。

```python
>>> a = tf.constant([0.9, 2.5, 2.3, 1.5, -4.5])
>>> tf.round(a)
<tf.Tensor: shape=(5,), dtype=float32, numpy=array([ 1.,  2.,  2.,  2., -4.], dtype=float32)>
```



## sigmoid()

Sigmoid 激活函数。

```python
>>> input = tf.random.normal([2])
>>> input
<tf.Tensor: shape=(2,), dtype=float32, numpy=array([1.3574934 , 0.30114314], dtype=float32)>
>>> tf.sigmoid(input)
<tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.795352  , 0.57472193], dtype=float32)>
```



## sign()

张量逐元素应用符号函数。

```python
>>> tf.math.sign([0., 2., -3.])
<tf.Tensor: shape=(3,), dtype=float32, numpy=array([ 0.,  1., -1.], dtype=float32)>
```



## sin(), cos(), tan(), arcsin(), arccos(), arctan(), sinh(), cosh(), tanh(), arcsinh(), arccosh(), arctanh()

张量逐元素应用三角函数和双曲函数。

```python
>>> a = tf.reshape(tf.range(10.), [2,5])
>>> a
<tf.Tensor: shape=(2, 5), dtype=float32, numpy=
array([[0., 1., 2., 3., 4.],
       [5., 6., 7., 8., 9.]], dtype=float32)>
>>> tf.sin(a)
<tf.Tensor: shape=(2, 5), dtype=float32, numpy=
array([[ 0.        ,  0.84147096,  0.9092974 ,  0.14112   , -0.7568025 ],
       [-0.9589243 , -0.2794155 ,  0.6569866 ,  0.98935825,  0.4121185 ]],
      dtype=float32)>
```



## sqrt()

张量逐元素开平方。

```python
>>> a = tf.constant([4.0, 9.0])
>>> tf.sqrt(a)
<tf.Tensor: shape=(2,), dtype=float32, numpy=array([2., 3.], dtype=float32)>
```



## square()

张量逐元素平方。相当于 `**2`。

```python
>>> a = tf.constant([4.0, 9.0])
>>> tf.square(a)
<tf.Tensor: shape=(2,), dtype=float32, numpy=array([16., 81.], dtype=float32)>
```





# tf.random

## normal()

生成指定形状的随机张量，其中每个元素服从正态分布。

```python
>>> tf.random.set_seed(5)
>>> tf.random.normal([2, 3])         # 标准正态分布
<tf.Tensor: shape=(2, 3), dtype=float32, numpy=
array([[-0.18030666, -0.95028627, -0.03964049],
       [-0.7425406 ,  1.3231523 , -0.61854804]], dtype=float32)>

>>> tf.random.set_seed(5)
>>> tf.random.normal([2, 3], 80, 10)
<tf.Tensor: shape=(2, 3), dtype=float32, numpy=
array([[78.19693, 70.49714, 79.60359],
       [72.57459, 93.23152, 73.81452]], dtype=float32)>
```



## set_seed()

设置全局随机种子。

涉及随机数的操作从全局种子和操作种子推导其自身的种子。全局种子和操作种子的关系如下：

1. 若都没有设定，则操作随机选取一个种子。
2. 若只设定了全局种子，则接下来的若干操作选取的种子都是确定的。注意不同版本的 TensorFlow 可能会得到不同的结果。
3. 若只设定了操作种子，则使用默认的全局种子。
4. 若都设定，则两个种子共同确定操作的种子。

```python
# 都没有设定: 每次调用的结果都是随机的
>>> tf.random.uniform([1])
<tf.Tensor: shape=(1,), dtype=float32, numpy=array([0.3124857], dtype=float32)>
>>> tf.random.uniform([1])
<tf.Tensor: shape=(1,), dtype=float32, numpy=array([0.72942686], dtype=float32)>

# now close the program and run it again

>>> tf.random.uniform([1])
<tf.Tensor: shape=(1,), dtype=float32, numpy=array([0.21320045], dtype=float32)>
>>> tf.random.uniform([1])
<tf.Tensor: shape=(1,), dtype=float32, numpy=array([0.06874764], dtype=float32)>
```

```python
# 设定全局种子: 每次调用set_seed()之后的调用结果都是确定的
>>> tf.random.set_seed(1)
>>> tf.random.uniform([1])
<tf.Tensor: shape=(1,), dtype=float32, numpy=array([0.16513085], dtype=float32)>
>>> tf.random.uniform([1])
<tf.Tensor: shape=(1,), dtype=float32, numpy=array([0.51010704], dtype=float32)>

>>> tf.random.set_seed(1)
>>> tf.random.uniform([1])
<tf.Tensor: shape=(1,), dtype=float32, numpy=array([0.16513085], dtype=float32)>
>>> tf.random.uniform([1])
<tf.Tensor: shape=(1,), dtype=float32, numpy=array([0.51010704], dtype=float32)>
```

```python
# 设定操作种子: 每次启动程序之后的调用结果都是确定的
>>> tf.random.uniform([1], seed=1)
<tf.Tensor: shape=(1,), dtype=float32, numpy=array([0.2390374], dtype=float32)>
>>> tf.random.uniform([1], seed=1)
<tf.Tensor: shape=(1,), dtype=float32, numpy=array([0.22267115], dtype=float32)>

# now close the program and run it again

>>> tf.random.uniform([1], seed=1)
<tf.Tensor: shape=(1,), dtype=float32, numpy=array([0.2390374], dtype=float32)>
>>> tf.random.uniform([1], seed=1)
<tf.Tensor: shape=(1,), dtype=float32, numpy=array([0.22267115], dtype=float32)>
```

```python
# 设定全局种子和操作种子: 完全确定调用结果
>>> tf.random.set_seed(1)
>>> tf.random.uniform([1], seed=1)
<tf.Tensor: shape=(1,), dtype=float32, numpy=array([0.05554414], dtype=float32)>
  
>>> tf.random.set_seed(1)
>>> tf.random.uniform([1], seed=3)
<tf.Tensor: shape=(1,), dtype=float32, numpy=array([0.7787856], dtype=float32)>
>>> tf.random.uniform([1], seed=2)
<tf.Tensor: shape=(1,), dtype=float32, numpy=array([0.40639675], dtype=float32)>
>>> tf.random.uniform([1], seed=1)
<tf.Tensor: shape=(1,), dtype=float32, numpy=array([0.05554414], dtype=float32)>
```



## poisson()

生成指定形状的随机张量，其中每个元素服从泊松分布。

```python
>>> tf.random.poisson([10], [1, 2])              # 第0,1列分别服从λ=1,2的泊松分布
<tf.Tensor: shape=(10, 2), dtype=float32, numpy=
array([[1., 0.],
       [0., 2.],
       [1., 4.],
       [0., 1.],
       [0., 1.],
       [0., 1.],
       [0., 4.],
       [1., 0.],
       [1., 2.],
       [1., 2.]], dtype=float32)>
```



## uniform()

生成指定形状的随机张量，其中每个元素服从均匀分布。

```python
>>> tf.random.set_seed(5)
>>> tf.random.uniform([2,3])
<tf.Tensor: shape=(2, 3), dtype=float32, numpy=
array([[0.6263931 , 0.5298432 , 0.7584572 ],
       [0.5084884 , 0.34415376, 0.31959772]], dtype=float32)>

>>> tf.random.set_seed(5)
>>> tf.random.uniform([2,3], 0, 10)
<tf.Tensor: shape=(2, 3), dtype=float32, numpy=
array([[6.263931 , 5.2984324, 7.584572 ],
       [5.084884 , 3.4415376, 3.1959772]], dtype=float32)>
```





# tf.signal

信号处理操作。



## fft()

快速傅立叶变换。







# tf.sparse

## add



## concat



## mask



## SparseTensor

稀疏张量类型。

```python
>>> sparse_tensor = tf.sparse.SparseTensor(indices=[[0, 0], [1, 2]],
...                                        values=[1, 2],
...                                        dense_shape=[3, 4])
>>> tf.sparse.to_dense(sparse_tensor)
<tf.Tensor: shape=(3, 4), dtype=int32, numpy=
array([[1, 0, 0, 0],
       [0, 0, 2, 0],
       [0, 0, 0, 0]], dtype=int32)>
```





# tf.strings

字符串张量操作。

## string tensor

```python
scalar_of_string = tf.constant("Gray wolf")
print(scalar_of_string)
# tf.Tensor(b'Gray wolf', shape=(), dtype=string)
tensor_of_strings = tf.constant(["Gray wolf",
                                 "Quick brown fox",
                                 "Lazy dog"])
print(tensor_of_strings)
# tf.Tensor([b'Gray wolf' b'Quick brown fox' b'Lazy dog'], shape=(3,), dtype=string)

unicode_string = tf.constant("🥳👍") # unicode string
print(unicode_string)
# <tf.Tensor: shape=(), dtype=string, numpy=b'\xf0\x9f\xa5\xb3\xf0\x9f\x91\x8d'>

print(tf.strings.split(scalar_of_string, sep=" "))  # split
# tf.Tensor([b'Gray' b'wolf'], shape=(2,), dtype=string)
print(tf.strings.split(tensor_of_strings))          # split vector of strings
# <tf.RaggedTensor [[b'Gray', b'wolf'], [b'Quick', b'brown', b'fox'], [b'Lazy', b'dog']]>

# ascii char
print(tf.strings.bytes_split(scalar_of_string))     # split to bytes
# tf.Tensor([b'G' b'r' b'a' b'y' b' ' b'w' b'o' b'l' b'f'], shape=(9,), dtype=string)
print(tf.io.decode_raw(scalar_of_string, tf.uint8)) # cast to ascii
# tf.Tensor([ 71 114  97 121  32 119 111 108 102], shape=(9,), dtype=uint8)

# unicode char
print(tf.strings.unicode_split(unicode_string, "UTF-8"))
print(tf.strings.unicode_decode(unicode_string, "UTF-8"))
# tf.Tensor([b'\xf0\x9f\xa5\xb3' b'\xf0\x9f\x91\x8d'], shape=(2,), dtype=string)
# tf.Tensor([129395 128077], shape=(2,), dtype=int32)
```



## format()



## join()



## ngrams()



## regex_full_match()



## regex_replace()





## split()



## strip()





# tfds

`tfds` 模块定义了一系列 TensorFlow 可以直接使用的数据集的集合。每个数据集都定义为一个 `tfds.core.DatasetBuilder` 实例，该实例封装了下载数据集、构建输入流水线的逻辑，也包含了数据集的文档。



## as_dataframe()

将数据集转换为 Pandas dataframe。

```python
tfds.as_dataframe(
    ds: tf.data.Dataset,
    ds_info: Optional[tfds.core.DatasetInfo] = None
) -> StyledDataFrame
# ds        要转换为Pandas dataframe的`Dataset`实例,其中样本不应分批
# ds_info   `DatasetInfo`实例,用于帮助改善格式
```



## as_numpy()

将数据集转换为 NumPy 数组。

```python
tfds.as_numpy(
    dataset: Tree[TensorflowElem]
) -> Tree[NumpyElem]
```

```python
ds = tfds.load(name="mnist", split="train")
ds_numpy = tfds.as_numpy(ds)  # Convert `tf.data.Dataset` to Python generator
for ex in ds_numpy:
  # `{'image': np.array(shape=(28, 28, 1)), 'labels': np.array(shape=())}`
  print(ex)
```



## build()

通过数据集名称获取一个 `DatasetBuilder` 实例。

```python
tfds.builder(name: str, *, try_gcs: bool = False, **builder_kwargs) -> tfds.core.DatasetBuilder
# name                数据集名称,需要是`DatasetBuilder`中注册的名称.可以是'dataset_name'或'dataset_name/config_name'
#                     (对于有`Builderconfig`的数据集)
# try_gcs
# **builder_kwargs    传递给`DatasetBuilder`的关键字参数字典
```



```python
```





## core.BuilderConfig



## core.DatasetBuilder





## core.DatasetInfo







## load()

通过数据集名称加载一个 `Dataset` 实例。

```python
tfds.load(
    name: str,
    *,
    split: Optional[Tree[splits_lib.Split]] = None,
    data_dir: Optional[str] = None,
    batch_size: tfds.typing.Dim = None,
    shuffle_files: bool = False,
    download: bool = True,
    as_supervised: bool = False,
    decoders: Optional[TreeDict[decode.Decoder]] = None,
    read_config: Optional[tfds.ReadConfig] = None,
    with_info: bool = False,
    builder_kwargs: Optional[Dict[str, Any]] = None,
    download_and_prepare_kwargs: Optional[Dict[str, Any]] = None,
    as_dataset_kwargs: Optional[Dict[str, Any]] = None,
    try_gcs: bool = False
)
# name                数据集名称,需要是`DatasetBuilder`中注册的名称.可以是'dataset_name'或
#                     'dataset_name/config_name'(对于有`Builderconfig`的数据集)
# split               加载的数据集部分,例如'train','test',['train','test'],'train[80%:]',....若为`None`,
#                     则返回部分名称到`Dataset`实例的字典
# data_dir            读/写数据的目录
# batch_size          批次规模,设定后会为样本增加一个批次维度
# shuffle_files       若为`True`,打乱输入文件
# download            若为`True`,在调用`DatasetBuilder.as_dataset()`之前调用
#                     `DatasetBuilder.download_and_prepare()`,如果数据已经在`data_dir`下,则不执行任何操作;
#                     若为`False`,则数据应当存在于`data_dir`下.
# as_supervised
# decoders
# read_config
# with_info           若为`True`,返回元组(`Dataset`, `DatasetInfo`)
# builder_kwargs      传递给`DatasetBuilder`的关键字参数字典
# download_and_prepare_kwargs     传递给`DatasetBuilder.download_and_prepare()`的关键字参数字典
# as_dataset_kwargs               传递给`DatasetBuilder.as_dataset()`的关键字参数字典
# try_gcs
```

```python
```



