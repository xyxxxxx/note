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

沿指定维度拼接张量。见`tf.tile()`, `tf.stack()`, `tf.repeat()`。

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

用类似张量的对象（python数组，numpy数组等）创建一个常数张量。

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



## matmul()

张量乘法。`@`符号重载了此方法。

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

生成指定形状的全1张量。

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

根据给定的初值，末值和步长创建一维张量。与python的`range()`用法相同。

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

返回一个张量，其在输入张量的基础上删除所有规模为1的维度。

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

### dtype

张量的数据类型。

```python
>>> a = tf.zeros([2,3])
>>> a.dtype
tf.float32
```



### shape, dim

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



## zeros()

生成指定形状的全0张量。

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

返回运行时当前可见的`PhysicalDevice`对象的列表。

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

调用`list_logical_devices()`会引发运行时初始化所有可见的`PhysicalDevice`，因而不能继续配置。若不想要初始化运行时，请调用`list_physical_devices()`。

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

物理设备指当前主机现有的硬件设备，包括所有已发现的CPU和GPU设备。此API用于在初始化运行时之前查询硬件资源，进而帮助调用更多的配置API。

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

一个`LogicalDevice`对象对应一个`PhysicalDevice`对象或者集群上的远程设备。张量或操作可以通过调用`tf.device()`并指定`LogicalDevice`，而被放置在指定的逻辑设备上。



## LogicalDeviceConfiguration

逻辑设备的配置类。

```python
tf.config.LogicalDeviceConfiguration(memory_limit=None, experimental_priority=None)
# memory_limit   为逻辑设备分配的显存
```

此类用于在初始化运行时过程中，指定配置参数将`PhysicalDevice`初始化为`LogicalDevice`。



## PhysicalDevice

本地物理设备的抽象。

```python
tf.config.PhysicalDevice(name, device_type)
```

tensorflow可以利用各种设备进行计算，例如CPU或者（多个）GPU。在初始化本地设备之前，用户可以自定义设备的一些属性，例如可见性或者内存配置。



## set_logical_device_configuration()

为一个`PhysicalDevice`对象设定逻辑设备配置。

一旦初始化运行时，一个可见的`PhysicalDevice`对象就默认创建一个`LogicalDevice`对象与之关联。指定`LogicalDeviceConfiguration`对象列表则会在一个`PhysicalDevice`对象上创建多个`LogicalDevice`对象。

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

设定一个`PhysicalDevice`对象是否启用内存增长。

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

指定运行时可见的`PhysicalDevice`对象的列表。tensorflow只会将张量和操作分配到可见的物理设备，因为`LogicalDevice`只能创建在可见的`PhysicalDevice`上。默认情况下，所有已发现的CPU和GPU设备都是可见的。

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





# Variable

```python

```



## GradientTape

`tf.GradientTape()` 是一个自动求导的记录器。以下示例计算$$y=x^2$$在$$x=3$$位置的导数：

```python
import tensorflow as tf

x = tf.Variable(initial_value=3.)   # 初值为3.0的变量
with tf.GradientTape() as tape:     # 在 tf.GradientTape() 的上下文内，所有计算步骤都会被记录以用于求导
    y = tf.square(x)
y_grad = tape.gradient(y, x)        # 计算y关于x的导数
print(y, y_grad)                    # tf.Tensor(6.0, shape=(), dtype=float32)
```

以下示例计算$$\mathcal{L}=||X\pmb w+b-\pmb y||^2$$在$$\pmb w=[1,2]^{\rm T},b=1$$位置的对$$\pmb w,b$$的导数：

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
\mathcal{L}=125,\ \frac{\partial \mathcal{L}}{\partial \pmb w}=\begin{bmatrix}70\\100\end{bmatrix},\ \frac{\partial \mathcal{L}}{\partial b}=30
$$






# tf.keras

在 TensorFlow 中，推荐使用 Keras（ `tf.keras` ）构建模型。Keras 是一个广为流行的高级神经网络 API，简单、快速而不失灵活性，现已得到 TensorFlow 的官方内置和全面支持。

Keras 提供了定义和训练任何类型的神经网络模型的便捷方法，具有以下特性：

+ 允许代码在CPU或GPU上运行并且无缝切换
+ 提供用户友好的 API 以使得能够快速建模
+ 提供对于 CNN (for CV)，RNN (for time series) 的内置支持
+ 支持任意类型的网络结构

keras 有两个重要的概念： **模型（model）** 和 **层（layer）** 。层将各种计算流程和变量进行了封装（例如基本的全连接层，CNN 的卷积层、池化层等），而模型则将各种层进行组织和连接，并封装成一个整体，描述了如何将输入数据通过各种层以及运算而得到输出。



## callbacks

### EarlyStopping

当监视的参数不再改善时提前停止训练。

```python
tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=0, verbose=0,
    mode='auto', baseline=None, restore_best_weights=False
)
# monitor               监视的指标
# min_delta             可以视为改善的最小绝对变化量,换言之,小于该值的指标绝对变化量视为没有改善
# patience              若最近patience次epoch的指标都没有改善(即最后patience次的指标都比倒数第patience+1次差),则停止训练
# mode                  若为'min',则指标减小视为改善;若为'max',则指标增加视为改善;若为'auto',则方向根据指标的名称自动推断
# baseline              监视的指标的基线值,若指标没有超过基线值则停止训练
# restore_best_weights  若为True,训练结束时会恢复监视指标取最好值的epoch的权重;若为False,训练结束时会保留最后一个epoch的权重
```



### LambdaCallback

创建简单的自定义回调。

```python
tf.keras.callbacks.LambdaCallback(
    on_epoch_begin=None, on_epoch_end=None, on_batch_begin=None, on_batch_end=None,
    on_train_begin=None, on_train_end=None, **kwargs
)
# on_epoch_begin  在每个epoch开始时调用
# ...
```

```python
# 自定义batch回调示例
print_batch_callback = callbacks.LambdaCallback(
    on_batch_end=lambda batch,logs: print(batch, logs))   # 需要两个位置参数: batch, logs
                                                          # 分别代表当前batch的序号和指标
# 训练输出
  20/1500 [..............................] - ETA: 18s - loss: 2.2107 - accuracy: 0.206319 {'loss': 2.0734667778015137, 'accuracy': 0.3140625059604645}
20 {'loss': 2.042363405227661, 'accuracy': 0.331845223903656}
21 {'loss': 2.0119757652282715, 'accuracy': 0.3480113744735718}
22 {'loss': 1.9936082363128662, 'accuracy': 0.3586956560611725}
  24/1500 [..............................] - ETA: 18s - loss: 2.1762 - accuracy: 0.230623 {'loss': 1.967929720878601, 'accuracy': 0.3697916567325592}
24 {'loss': 1.934678316116333, 'accuracy': 0.3824999928474426}
25 {'loss': 1.9046880006790161, 'accuracy': 0.39423078298568726}
26 {'loss': 1.8743277788162231, 'accuracy': 0.40509259700775146}
27 {'loss': 1.838417887687683, 'accuracy': 0.4151785671710968}  
  
  
# 自定义epoch回调示例
print_epoch_callback = callbacks.LambdaCallback(
    on_epoch_end=lambda epoch,logs: print(epoch, logs))   # 需要两个位置参数: epoch, logs
                                                          # 分别代表当前epoch的序号和指标
# 训练输出
Epoch 1/10
1500/1500 [==============================] - 20s 13ms/step - loss: 0.3875 - accuracy: 0.8781 - val_loss: 0.0871 - val_accuracy: 0.9728
0 {'loss': 0.16673415899276733, 'accuracy': 0.9478958249092102, 'val_loss': 0.0870571881532669, 'val_accuracy': 0.9728333353996277}
Epoch 2/10
1500/1500 [==============================] - 19s 13ms/step - loss: 0.0564 - accuracy: 0.9816 - val_loss: 0.0502 - val_accuracy: 0.9848
1 {'loss': 0.05158458277583122, 'accuracy': 0.9834166765213013, 'val_loss': 0.0502360574901104, 'val_accuracy': 0.9848333597183228}
```





### LearningRateScheduler



### ModelCheckpoint



### TensorBoard

为TensorBoard可视化记录日志。

```python
tf.keras.callbacks.TensorBoard(
    log_dir='logs', histogram_freq=0, write_graph=True,
    write_images=False, update_freq='epoch', profile_batch=2,
    embeddings_freq=0, embeddings_metadata=None, **kwargs
)
# log_dir      待TensorBoard解析的日志文件的保存路径
# update_freq  若为'batch',则在每个batch结束时记录损失和指标;若为'epoch',则在每个epoch结束时记录损失和指标;
#              若为整数,则每update_freq个batch记录一次损失和指标.注意过于频繁地写日志会减慢你的训练.
```









## layer

层是进行数据处理的模块，它输入一个张量，然后输出一个张量。尽管有一些层是无状态的，更多的层都有其权重张量，通过梯度下降法学习。`tf.keras.layers` 下内置了深度学习中大量常用的的预定义层，同时也允许我们自定义层。

### Dense

全连接层（densely connected layer, fully connected layer, `tf.keras.layers.Dense` ）是 Keras 中最基础和常用的层之一，对输入矩阵 $$A$$ 进行 $$f(A\pmb w+b)$$ 的线性变换 + 激活函数操作。如果不指定激活函数，即是纯粹的线性变换 $$A\pmb w+b$$。具体而言，给定输入张量 `input = [batch_size, input_dim]` ，该层对输入张量首先进行 `tf.matmul(input, kernel) + bias` 的线性变换（ `kernel` 和 `bias` 是层中可训练的变量），然后对线性变换后张量的每个元素通过激活函数 `activation` ，从而输出形状为 `[batch_size, units]` 的二维张量。

[![../../_images/dense.png](https://tf.wiki/_images/dense.png)](https://tf.wiki/_images/dense.png)

其包含的主要参数如下：

+ `units` ：神经元的个数，也是输出张量的维度
+ `activation` ：激活函数，默认为无激活函数。常用的激活函数包括 `tf.nn.relu` 、 `tf.nn.tanh` 和 `tf.nn.sigmoid` 
+ `use_bias` ：是否加入偏置向量 `bias` ，默认为 `True` 
+ `kernel_initializer` 、 `bias_initializer` ：权重矩阵 `kernel` 和偏置向量 `bias` 两个变量的初始化器。默认为 `tf.glorot_uniform_initializer`。设置为 `tf.zeros_initializer` 表示将两个变量均初始化为全 0

该层包含权重矩阵 `kernel = [input_dim, units]` 和偏置向量 `bias = [units]`两个可训练变量，对应于 $$f(A\pmb w+b)$$ 中的 $$\pmb w$$ 和 $$b$$。



### Conv2D

卷积层。

其包含的主要参数如下：

+ `filters`：输出特征映射的个数
+ `kernel_size`：整数或整数1×2向量，（分别）表示二维卷积核的高和宽
+ `strides`：整数或整数1×2向量，（分别）表示卷积的纵向和横向步长
+ `padding`：`"valid"`表示对于不够卷积核大小的部分丢弃，`"same"`表示对于不够卷积核大小的部分补0，默认为`"valid"`
+ `activation`：激活函数，默认为无激活函数
+ `use_bias`：是否使用偏置，默认为使用

示例：

```python
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
# 输入32x32RGB图片,输出32个特征映射,使用3x3卷积核,每个输出特征映射使用1个偏置
# 参数数量为3x32x(3x3)+32=896
model.add(keras.layers.MaxPooling2D((2, 2)))
# 对每个2x2区块执行最大汇聚
model.add(keras.layers.Conv2D(64, (3, 3), (2, 2), activation='relu'))
# 卷积的步长设为2
model.add(keras.layers.MaxPooling2D((2, 2)))
# 7%2=1,因此丢弃一行一列的数据
model.summary()

# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# conv2d (Conv2D)              (None, 30, 30, 32)        896       
# _________________________________________________________________
# max_pooling2d (MaxPooling2D) (None, 15, 15, 32)        0         
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 7, 7, 64)          18496     
# _________________________________________________________________
# max_pooling2d_1 (MaxPooling2 (None, 3, 3, 64)          0         
# =================================================================
# Total params: 19,392
# Trainable params: 19,392
# Non-trainable params: 0
```



### MaxPooling2D

汇聚层（池化层）。

其包含的主要参数如下：

+ `pool_size`：最大汇聚的区域规模，默认为`(2,2)`
+ `strides`：最大汇聚的步长，默认为`None`
+ `padding`：`"valid"`表示对于不够区域大小的部分丢弃，`"same"`表示对于不够区域大小的部分补0，默认为`"valid"`



### Embedding

> 参考[单词嵌入向量](https://www.tensorflow.org/tutorials/text/word_embeddings)

嵌入层可以被理解为整数（单词索引）到密集向量的映射。嵌入层输入形如`(samples, sequence_length)`的二维整数张量，因此所有的整数序列都应填充或裁剪到相同的长度；输出形如`(samples, sequence_ length, embedding_dimensionality)`的三维浮点张量，再输入给 RNN 层处理。

嵌入层在刚初始化时所有的权重参数都是随机的，就如同其它的层一样。在训练过程中这些参数会根据反向传播算法逐渐更新，嵌入空间会逐渐显现出更多结构（这些结构适应于当前的具体问题）。

其包含的主要参数如下：

+ `input_dim`：字典的规模
+ `output_dim`：嵌入向量的规模
+ `mask_zero`：是否将输入中的0看作填充值而忽略之，默认为`False`
+ `input_length`：输入序列的长度（如果该长度固定），默认为`None`；如果此嵌入层后接`Flatten`层，再接`Dense`层，则必须制定此参数

示例见SimpleRNN，LSTM。



### SimpleRNN

SRN层是最简单的循环神经网络层。

其包含的主要参数如下：

+ `units`：输出向量的维度
+ `activation`：激活函数，默认为`tanh`
+ `return_sequences`：`False`表示最后输出一个向量，即序列到类别模式；`True`表示每个时间步长输出一个向量，即序列到序列模式。

实践中一般使用 LSTM 和 GRU 而非 SRN。

示例：

```python
model = keras.Sequential()
model.add(keras.layers.Embedding(10000, 32))
model.add(keras.layers.SimpleRNN(32))
# 输入timestepsx32的二阶张量,输出32维向量,即序列到类别模式
model.summary()
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# embedding (Embedding)        (None, None, 32)          320000    
# _________________________________________________________________
# simple_rnn (SimpleRNN)       (None, 32)                2080      
# =================================================================
# Total params: 322,080
# Trainable params: 322,080
# Non-trainable params: 0
# _________________________________________________________________
```

```python
model = keras.Sequential()
model.add(keras.layers.Embedding(10000, 32))
model.add(keras.layers.SimpleRNN(32, return_sequences=True))
# 输入timestepsx32的二阶张量,输出timestepsx32的二阶张量,即序列到序列模式
model.summary()
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# embedding (Embedding)        (None, None, 32)          320000    
# _________________________________________________________________
# simple_rnn (SimpleRNN)       (None, None, 32)          2080      
# =================================================================
# Total params: 322,080
# Trainable params: 322,080
# Non-trainable params: 0
# _________________________________________________________________
```

```python
model = keras.Sequential()
model.add(keras.layers.Embedding(10000, 32))
model.add(keras.layers.SimpleRNN(32, return_sequences=True))
# SRN的堆叠
model.add(keras.layers.SimpleRNN(32, return_sequences=True))
model.add(keras.layers.SimpleRNN(32, return_sequences=True))
model.add(keras.layers.SimpleRNN(32))
model.summary()
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# embedding (Embedding)        (None, None, 32)          320000    
# _________________________________________________________________
# simple_rnn (SimpleRNN)       (None, None, 32)          2080      
# _________________________________________________________________
# simple_rnn_1 (SimpleRNN)     (None, None, 32)          2080      
# _________________________________________________________________
# simple_rnn_2 (SimpleRNN)     (None, None, 32)          2080      
# _________________________________________________________________
# simple_rnn_3 (SimpleRNN)     (None, 32)                2080      
# =================================================================
# Total params: 328,320
# Trainable params: 328,320
# Non-trainable params: 0
# _________________________________________________________________
```





### LSTM

LSTM 层。

其包含的主要参数如下：

+ `units`：输出空间的规模



示例：

```python
model = keras.Sequential()
model.add(keras.layers.Embedding(10000, 16))
# 将规模为10000的词典嵌入到16维向量
# 输入256维向量,输出256x16的二阶张量
model.add(keras.layers.LSTM(64))
# 输入256x16的二阶张量,输出64维向量(隐状态),即序列到类别模式
model.add(keras.layers.Dense(16, activation='relu'))
# 全连接层,ReLU激活函数,分类器
model.add(keras.layers.Dense(1, activation='sigmoid'))  
# 全连接层,Logistic激活函数
model.summary()
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# embedding (Embedding)        (None, None, 16)          160000    
# _________________________________________________________________
# lstm (LSTM)                  (None, 64)                20736     
# _________________________________________________________________
# dense (Dense)                (None, 16)                1040      
# _________________________________________________________________
# dense_1 (Dense)              (None, 1)                 17        
# =================================================================
# Total params: 181,793
# Trainable params: 181,793
# Non-trainable params: 0
# _________________________________________________________________
```

```python
model = keras.Sequential()
model.add(keras.layers.LSTM(64, 
                            dropout=0.2,
                            recurrent_dropout=0.2,
                            return_sequences=True,
                            # 堆叠了rnn层,必须在每个时间步长输出
                            input_shape=(None, df.shape[-1])))
                            # 尽管这里序列的长度是确定的(120),但也不必传入
model.add(keras.layers.LSTM(64,
                            activation='relu',
                            dropout=0.2,
                            recurrent_dropout=0.2))
# 亦可以使用 layers.GRU
model.add(keras.layers.Dense(1))
```



### GRU

GRU 层。



### Bidirectional

双向 RNN 层在某些特定的任务上比一般的 RNN 层表现得更好，经常应用于 NLP。

RNN 的输入序列存在顺序，打乱或反序都会彻底改变 RNN 从序列中提取的特征。双向 RNN 包含两个一般的 RNN，分别从一个方向处理输入序列。在许多 NLP 任务中，反向处理输入序列能够达到与正向处理相当的结果，并且提取出不同但同样有效的特征，此种情况下双向 RNN 将捕获到更多的有用的模式，但也会更快地过拟合。

![Screenshot from 2020-09-28 19-00-29.png](https://i.loli.net/2020/09/28/i2V36gZhtIv7BJA.png)

`keras`中，`Bidirectional`实现为创建一个参数指定的 RNN 层，再创建一个相同的 RNN 层处理反序的输入序列。

示例：

```python
model = keras.Sequential()
model.add(keras.layers.Embedding(max_features, 32))
model.add(keras.layers.Bidirectional(keras.layers.LSTM(32)))
model.add(keras.layers.Dense(1, activation='sigmoid'))
```





### Dropout

示例：

```python
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))
```





## model

![](https://i.loli.net/2020/09/27/hvxUc9eyiqJkGVu.png)

### compile

其包含的主要参数如下：

+ `optimizer`：优化方法，可以选择`'rmsprop','adam'`等，默认为`'rmsprop'`
+ `loss`：损失函数，可以选择`'binary_crossentropy','categorical_crossentropy','sparse_categorical_crossentropy','mse','mae'`等
+ `metrics`：评价指标，可以选择`'accuracy','mse','mae','precision','recall','auc'`等中的多个



### fit

其包含的主要参数如下：

+ `x`：输入数据
+ `y`：输出标签
+ `batch_size`：每个`batch`的规模
+ `epoch`：训练集的循环迭代次数
+ `verbose`：控制台输出信息，0=无输出，1=输出进度条，2=对每个epoch输出一行记录，默认为1
+ `callbacks`：回调
+ `validation_split`：指定从训练集划分验证集的比例
+ `validation_data`：指定验证集



### evaluate

其包含的主要参数如下：

+ `x`：输入数据
+ `y`：输出标签
+ `verbose`：控制台输出信息，0=无输出，1=输出进度条，默认为1



### summary

`summary()`方法显示模型的基本结构。



### Sequential

`Sequential`返回一个`keras.Model`对象。`Sequential`模型将各层线性组合，适用于FNN，CNN，RNN，其中每一层都有**一个输入张量和一个输出张量** 。

以下`Sequential`模型，

```python
model = keras.Sequential(
    [
        layers.Dense(2, activation="relu", name="layer1"),
        layers.Dense(3, activation="relu", name="layer2"),
        layers.Dense(4, name="layer3"),
    ]
)
# Call model on a test input
x = tf.ones((3, 3))
y = model(x)
```

等效于以下功能

```python
# Create 3 layers
layer1 = layers.Dense(2, activation="relu", name="layer1")
layer2 = layers.Dense(3, activation="relu", name="layer2")
layer3 = layers.Dense(4, name="layer3")

# Call layers on a test input
x = tf.ones((3, 3))
y = layer3(layer2(layer1(x)))
```

`Sequential`模型也可以用`add()`方法创建

```python
model = keras.Sequential()
model.add(layers.Dense(2, activation="relu"))
model.add(layers.Dense(3, activation="relu"))
model.add(layers.Dense(4))
```



CNN模型示例：

```python
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
# FNN, CNN 需要指定输入张量的shape
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dense(10))
```



### 自定义模型

Keras 模型以类的形式呈现，我们可以通过继承 `tf.keras.Model` 这个 Python 类来定义自己的模型。在继承类中，我们需要重写 `__init__()` （构造函数，初始化）和 `call(input)` （模型调用）两个方法，同时也可以根据需要增加自定义的方法。

```python
class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()     # Python 2 下使用 super(MyModel, self).__init__()
        # 此处添加初始化代码（包含 call 方法中会用到的层），例如
        # layer1 = tf.keras.layers.BuiltInLayer(...)
        # layer2 = MyCustomLayer(...)

    def call(self, input):
        # 此处添加模型调用的代码（处理输入并返回输出），例如
        # x = layer1(input)
        # output = layer2(x)
        return output

    # 还可以添加自定义的方法
```

以下为自定义线性模型示例：

```python
class Linear(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(
            units=1,
            activation=None,
            kernel_initializer=tf.zeros_initializer(),
            bias_initializer=tf.zeros_initializer()
        )

    def call(self, input):
        output = self.dense(input)
        return output
```



## optimizers







## regularizers

```python
model = models.Sequential()
model.add(layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001),
                                              # l2 regularization, coefficient = 0.001
activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, kernel_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001),                                    # l1 & l2
activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
```





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

返回张量的一个特征分解$$A=Q\Lambda Q^{-1}$$。



## svd()

返回张量的一个奇异值分解$$A=U\Sigma V^*$$。





# tf.math

## abs()

张量逐元素应用绝对值函数。

```python
>>> tf.abs(tf.constant([-1, -2, 3]))
<tf.Tensor: shape=(3,), dtype=int32, numpy=array([1, 2, 3], dtype=int32)>
```



## add(), subtract()

张量加法/减法。`+, -`符号重载了这些方法。

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

逐元素判断两个张量是否相等。`==`符号重载了此方法。

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

逐元素比较两个张量的大小。`>, >=, <, <=`符号重载了这些方法。

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

张量逐元素应用自然对数函数。注意tensorflow没有`log2()`和`log10()`函数。

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

张量逐元素乘法/除法。`*, /`符号重载了此方法。

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

张量逐元素幂乘。`**`符号重载了此方法。

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

张量逐元素应用舍入函数，0.5会向偶数取整。

```python
>>> a = tf.constant([0.9, 2.5, 2.3, 1.5, -4.5])
>>> tf.round(a)
<tf.Tensor: shape=(5,), dtype=float32, numpy=array([ 1.,  2.,  2.,  2., -4.], dtype=float32)>
```



## sigmoid()

Sigmoid激活函数。

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

张量逐元素平方。相当于`** 2`。

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
>>> tf.random.normal([2,3])
<tf.Tensor: shape=(2, 3), dtype=float32, numpy=
array([[-0.18030666, -0.95028627, -0.03964049],
       [-0.7425406 ,  1.3231523 , -0.61854804]], dtype=float32)>

>>> tf.random.set_seed(5)
>>> tf.random.normal([2,3], 80, 10)
<tf.Tensor: shape=(2, 3), dtype=float32, numpy=
array([[78.19693, 70.49714, 79.60359],
       [72.57459, 93.23152, 73.81452]], dtype=float32)>
```



## set_seed()

设置全局随机种子。

涉及随机数的操作从全局种子和操作种子推导其自身的种子。全局种子和操作种子的关系如下：

1. 若都没有设定，则操作随机选取一个种子。
2. 若只设定了全局种子，则接下来的若干操作选取的种子都是确定的。注意不同版本的tensorflow可能会得到不同的结果。
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





# tf.sparse

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


