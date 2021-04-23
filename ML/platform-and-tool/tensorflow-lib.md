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


