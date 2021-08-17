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



## newaxis

用于在张量的起始或末尾位置增加一个维度。

```python
>>> a = tf.reshape(tf.range(10.), [2,5])
>>> a[tf.newaxis, ...].shape      # `...`代表其余所有原有维度
TensorShape([1, 2, 5])
>>> a[..., tf.newaxis].shape
TensorShape([2, 5, 1])

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

根据给定的初值，末值和步长创建一维张量。与 Python 的 `range()` 用法相同。

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
>>> tf.reshape(a, [2, 5])
<tf.Tensor: shape=(2, 5), dtype=int32, numpy=
array([[0, 1, 2, 3, 4],
       [5, 6, 7, 8, 9]], dtype=int32)>
>>> tf.reshape(a, [2, -1])                  # -1表示自动补全该位置的值
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



## split()

划分张量为多个部分。

```python
>>> a = tf.reshape(tf.range(36), [6, 6])
>>> a0, a1, a2 = tf.split(a, 3, axis=0)           # 沿轴0 3等分
>>> a0
<tf.Tensor: shape=(2, 6), dtype=int32, numpy=
array([[ 0,  1,  2,  3,  4,  5],
       [ 6,  7,  8,  9, 10, 11]], dtype=int32)>
>>> a0, a1 = tf.split(a, 2, axis=1)               # 沿轴1 2等分
>>> a0
<tf.Tensor: shape=(6, 3), dtype=int32, numpy=
array([[ 0,  1,  2],
       [ 6,  7,  8],
       [12, 13, 14],
       [18, 19, 20],
       [24, 25, 26],
       [30, 31, 32]], dtype=int32)>
>>> a0, a1, a2, a3 = tf.split(a, 4, axis=1)       # 不能4等分,出错
tensorflow.python.framework.errors_impl.InvalidArgumentError: Number of ways to split should evenly divide the split dimension, but got split_dim 1 (size = 6) and num_split 4 [Op:Split] name: split
>>> a0, a1, a2 = tf.split(a, [1, 2, 3], axis=1)   # 沿轴1划分
>>> a0
<tf.Tensor: shape=(6, 1), dtype=int32, numpy=
array([[ 0],
       [ 6],
       [12],
       [18],
       [24],
       [30]], dtype=int32)>
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



## strided_slice()

提取张量的切片。

```python
tf.strided_slice(
    input_, begin, end, strides=None, begin_mask=0, end_mask=0, ellipsis_mask=0,
    new_axis_mask=0, shrink_axis_mask=0, var=None, name=None
)
# input_         被切片的张量
# begin          各维度的起始索引(包含)
# end            各维度的结束索引(不包含)
# strides        各维度的步长
# begin_mask     若设置了此参数的第i位,则`begin[i]`将被忽略,即第i个维度从最初的索引开始
# end_mask       若设置了此参数的第i位,则`end[i]`将被忽略,即第i个维度到最终的索引结束
# ellipsis_mask  若设置了此参数的第i位,则第i个维度及其之后的未指定的维度(贪婪匹配)将被完全保留.仅允许设置1位
# new_axis_mask  若设置了此参数的第i位,则增加一个维度作为第i个维度
# shrink_axis_mask  若设置了此参数的第i位,则移除第i个维度
```

```python
>>> a = tf.reshape(tf.range(25.), [5, 5])

>>> tf.strided_slice(a, [1, 1], [4, 4], [2, 2])    # 切片
# or
>>> a[1:4:2, 1:4:2]                                # 实际更常用NumPy风格的切片语法
<tf.Tensor: shape=(2, 2), dtype=float32, numpy=
array([[ 6.,  8.],
       [16., 18.]], dtype=float32)>

>>> tf.strided_slice(a, [0, 0], [0, 0], [2, 2], begin_mask=0b11, end_mask=0b11)
# or
>>> a[::2, ::2]
<tf.Tensor: shape=(3, 3), dtype=float32, numpy=
array([[ 0.,  2.,  4.],
       [10., 12., 14.],
       [20., 22., 24.]], dtype=float32)>

>>> tf.strided_slice(a, [0, 0], [0, 0], [1, 2], begin_mask=0b11, end_mask=0b11)
# or
>>> a[:, ::2]
<tf.Tensor: shape=(5, 3), dtype=float32, numpy=
array([[ 0.,  2.,  4.],
       [ 5.,  7.,  9.],
       [10., 12., 14.],
       [15., 17., 19.],
       [20., 22., 24.]], dtype=float32)>

>>> tf.strided_slice(a, [0, 0], [0, 0], strides=[1, 1], begin_mask=0b11, end_mask=0b11, new_axis_mask=0b10)
# or
>>> a[:, tf.newaxis, :]                            # 增加维度
<tf.Tensor: shape=(5, 1, 5), dtype=float32, numpy=
array([[[ 0.,  1.,  2.,  3.,  4.]],
       [[ 5.,  6.,  7.,  8.,  9.]],
       [[10., 11., 12., 13., 14.]],
       [[15., 16., 17., 18., 19.]],
       [[20., 21., 22., 23., 24.]]], dtype=float32)>

>>> tf.strided_slice(a, [0, 0], [0, 0], strides=[1, 1], begin_mask=0b11, end_mask=0b11, shrink_axis_mask=0b10)
# or
>>> a[:, 0]                                        # 移除维度
<tf.Tensor: shape=(5,), dtype=float32, numpy=array([ 0.,  5., 10., 15., 20.], dtype=float32)>
```

```python
>>> a = tf.reshape(tf.range(256.), [4, 4, 4, 4])
>>> a[::2, ..., ::2].shape                         # `...`贪婪匹配该位置及其之后的未指明的维度,并保留这些维度
TensorShape([2, 4, 4, 2])
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

调用 `list_logical_devices()` 会引发运行时初始化所有可见的 `PhysicalDevice`（一个可见 `PhysicalDevice` 对象默认创建一个 `LogicalDevice` 对象），因而不能继续配置。若不想要初始化运行时，请调用 `list_physical_devices()`。

即使不调用 `list_logical_devices()`，执行任何运算或使用任何 CPU 或 GPU 同样会初始化运行时。

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

一旦初始化运行时，一个可见的 `PhysicalDevice` 对象就默认创建一个 `LogicalDevice` 对象与之关联。在运行时初始化之前指定 `LogicalDeviceConfiguration` 对象列表会在一个 `PhysicalDevice` 对象上创建多个 `LogicalDevice` 对象。

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
    gpus[0],
    [tf.config.LogicalDeviceConfiguration(memory_limit=100),
     tf.config.LogicalDeviceConfiguration(memory_limit=100)])
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
>>> tf.config.list_logical_devices()                                                  # 所有逻辑设备
[LogicalDevice(name='/device:CPU:0', device_type='CPU'),
 LogicalDevice(name='/device:GPU:0', device_type='GPU')]
```





# tf.data

## Dataset

数据集类型，用于构建描述性的、高效的输入流水线。`Dataset` 实例的使用通常遵循以下模式：

1. 根据输入数据创建一个源数据集
2. 应用数据集变换以预处理数据
3. 迭代数据集并处理数据

迭代以流的方式进行，因此不会将整个数据集全部放到内存中。



> 以下方法均不是原位操作，即返回一个新的数据集，而不改变原数据集。

### apply()

为数据集应用一个变换函数。该变换函数通常是 `Dataset` 变换方法的组合。

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

返回一个迭代器，其将数据集的所有元素都转换为 NumPy 数组。常用于检查数据集的内容。

```python
>>> ds = Dataset.range(5)
>>> list(ds.as_numpy_iterator())
[0, 1, 2, 3, 4]
```



### batch()

将数据集的连续元素组合为批。

```python
>>> list(Dataset.range(10).batch(3).as_numpy_iterator())
[array([0, 1, 2]), array([3, 4, 5]), array([6, 7, 8]), array([9])]
>>> list(Dataset.range(10).batch(3, drop_remainder=True).as_numpy_iterator())   # 丢弃达不到指定规模的最后一个批次
[array([0, 1, 2]), array([3, 4, 5]), array([6, 7, 8])]
```



### cache()

缓存数据集的元素到内存中。

返回的数据集在第一次迭代后，其元素将被缓存到内存或指定文件中；接下来的迭代将使用缓存的数据。



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



### concatenate()

将数据集与给定数据集进行拼接。

```python
>>> ds1 = Dataset.range(1, 4)
>>> ds2 = Dataset.range(4, 8)
>>> ds = ds1.concatenate(ds2)
>>> list(ds.as_numpy_iterator())
[1, 2, 3, 4, 5, 6, 7]
```



### enumerate()

对数据集的元素进行计数。

```python
>>> ds = Dataset.from_tensor_slices([2, 4, 6]).enumerate(start=1)
>>> list(ds.as_numpy_iterator())
[(1, 2), (2, 4), (3, 6)]
```



### filter()

使用给定函数过滤数据集的元素。

```python
>>> ds = Dataset.range(10)
>>> ds = ds.filter(lambda x: x % 2 == 0)
>>> list(ds.as_numpy_iterator())
[0, 2, 4, 6, 8]
```



### from_generator()

创建由生成器生成的元素构成的数据集。



### from_tensor_slices()

创建由指定张量的元素构成的数据集。

```python
>>> ds = Dataset.from_tensor_slices([1, 2, 3])                   # 张量
>>> list(ds.as_numpy_iterator())
[1, 2, 3]                               # 张量的元素
>>> ds = Dataset.from_tensor_slices([[1, 2, 3], [4, 5, 6]])
>>> list(ds.as_numpy_iterator())
[array([1, 2, 3], dtype=int32), array([4, 5, 6], dtype=int32)]
>>> 
>>> ds = Dataset.from_tensor_slices(([1, 2], [3, 4], [5, 6]))    # 张量构成的元组
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

为数据集的每个元素应用一个函数。

```python
>>> ds = Dataset.from_tensor_slices([1, 2, 3])
>>> ds = ds.map(lambda x: x**2)
>>> list(ds.as_numpy_iterator())
[1, 4, 9]
```



### options()

返回数据集和其输入的选项。



### prefetch()

预取数据集的之后几个元素。

大部分数据集输入流水线都应该在最后调用 `prefetch()`，以使得在处理当前元素时后几个元素就已经准备好，从而改善时延和吞吐量，代价是使用额外的内存来保存这些预取的元素。

```python
>>> Dataset.range(5).prefetch(2)
<PrefetchDataset shapes: (), types: tf.int64>
>>> list(Dataset.range(5).prefetch(2).as_numpy_iterator())
[0, 1, 2, 3, 4]
```



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

重复数据集多次。

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

从数据集中等间距地抽取样本以构成子集。此方法在进行分布式训练时十分有用。

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
# 准备分布式训练数据集

# 每个worker分到数据集的不固定子集并打乱(推荐)
>>> ds = Dataset.range(10).shuffle(100, seed=1).shard(num_shards=3, index=0).batch(2)
>>> list(ds.as_numpy_iterator())
[array([0, 1]), array([3, 4])]
>>> ds = Dataset.range(10).shuffle(100, seed=1).shard(num_shards=3, index=1).batch(2)
>>> list(ds.as_numpy_iterator())
[array([9, 2]), array([8])]
>>> ds = Dataset.range(10).shuffle(100, seed=1).shard(num_shards=3, index=2).batch(2)
>>> list(ds.as_numpy_iterator())
[array([5, 7]), array([6])]

# 每个worker分到数据集的固定子集并打乱
>>> ds = Dataset.range(10).shard(num_shards=3, index=0).shuffle(100).batch(2)
>>> list(ds.as_numpy_iterator())
[array([3, 0]), array([6, 9])]

# 每个worker分到数据集的固定子集并跨epoch打乱(不推荐)
>>> ds = Dataset.range(10).shard(num_shards=3, index=0).repeat(3).shuffle(100).batch(2)
>>> list(ds.as_numpy_iterator())       # 将3个epoch的输入数据放在一起打乱
[array([0, 0]), array([3, 3]), array([6, 6]), array([9, 6]), array([3, 0]), array([9, 9])]
```



### shuffle()

随机打乱数据集中的元素。

```python
shuffle(buffer_size, seed=None, reshuffle_each_iteration=True)
# buffer_size                缓冲区大小.例如数据集包含1000个元素而`buffer_size`设为100,那么前100个元素首先进入缓冲区,
#                            从中随机抽取一个,然后第101个元素进入缓冲区,再随机抽取一个,...若要完全打乱数据集,则
#                            `buffer_size`应不小于数据集的规模
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

去除数据集的前几个元素。

```python
>>> list(Dataset.range(5).skip(2).as_numpy_iterator())
[2, 3, 4]
```



### take()

取数据集的前几个元素。

```python
>>> list(Dataset.range(5).take(2).as_numpy_iterator())
[0, 1]
```



### unbatch()

将数据集的元素（批）拆分为多个元素。

```python
>>> elements = [[1, 2, 3], [1, 2], [1, 2, 3, 4]]
>>> ds = Dataset.from_generator(lambda: elements, tf.int64)
>>> list(ds.as_numpy_iterator())
[array([1, 2, 3]), array([1, 2]), array([1, 2, 3, 4])]
>>> list(ds.unbatch().as_numpy_iterator())
[1, 2, 3, 1, 2, 1, 2, 3, 4]
```



### window()

将数据集中的相邻元素组合为窗口（窗口也是一个小规模的数据集），由这些窗口构成新的数据集。

```python
window(size, shift=None, stride=1, drop_remainder=False)
# size             窗口的元素数量
# shift            窗口的移动距离,默认为`size`
# stride           取样的间距
# drop_remainder   丢弃最后一个规模不足`size`的窗口
```

```python
>>> ds = Dataset.range(10).window(3)
>>> for window in ds:
  print(list(window.as_numpy_iterator()))
... 
[0, 1, 2]
[3, 4, 5]
[6, 7, 8]
[9]
>>> 
>>> ds = Dataset.range(10).window(3, drop_remainder=True)
>>> for window in ds:                                    
  print(list(window.as_numpy_iterator()))
... 
[0, 1, 2]
[3, 4, 5]
[6, 7, 8]
>>> 
>>> ds = Dataset.range(10).window(3, 2, 1, drop_remainder=True)
>>> for window in ds:                                          
  print(list(window.as_numpy_iterator()))
... 
[0, 1, 2]
[2, 3, 4]
[4, 5, 6]
[6, 7, 8]
>>> 
>>> ds = Dataset.range(10).window(3, 1, 2, drop_remainder=True)
>>> for window in ds:                                          
  print(list(window.as_numpy_iterator()))
... 
[0, 2, 4]
[1, 3, 5]
[2, 4, 6]
[3, 5, 7]
[4, 6, 8]
[5, 7, 9]
```



### zip()

组合多个数据集的对应元素。类似于 Python 的内置函数 `zip()`。

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

创建由一个或多个文本文件的各行构成的数据集。`TextLineDataset` 是 `Dataset` 的子类。

```
# file1.txt
the cow
jumped over
the moon
```

```
# file2.txt
jack and jill
went up
the hill
```

```python
>>> ds = tf.data.TextLineDataset(["file1.txt", "file2.txt"])
>>> for element in dataset.as_numpy_iterator():
  print(element)
b'the cow'
b'jumped over'
b'the moon'
b'jack and jill'
b'went up'
b'the hill'
```





# tf.distribute

> 此模块设计复杂，难以使用，越来越多的用户开始使用 [Horovod](./hovorod.md)。

分布式训练。



## CentralStorageStrategy

中央存储策略。



## cluster_resolver.ClusterResolver

所有集群解析器实现的基类。集群解析器是 TensorFlow 与各种集群管理系统（K8s、GCE、AWS 等）进行交流的手段，也为 TensorFlow 建立分布式训练提供必要的信息。

通过集群解析器和集群管理系统的交流，我们能够自动发现并解析各 TensorFlow 工作器的 IP 地址，进而能够自动从底层机器的故障中恢复，或者对 TensorFlow 工作器进行伸缩。



### cluster_spec()

获取集群的当前状态并返回一个 `tf.train.ClusterSpec` 实例。



### num_accelerators()

返回每个工作器可用的加速器核心（GPU 和 TPU）数量。



### task_id

返回 `ClusterResolver` 实例指明的 task ID。

一般在 TensorFlow 分布式环境中，每个 task 有一个相应的 task ID，即在该 task 类型中的索引。这在用户需要根据 task 索引运行特定代码时十分有用，例如：

```python
cluster_spec = tf.train.ClusterSpec({
    "ps": ["localhost:2222", "localhost:2223"],
    "worker": ["localhost:2224", "localhost:2225", "localhost:2226"]
})

# SimpleClusterResolver is used here for illustration; other cluster
# resolvers may be used for other source of task type/id.
cluster_resolver = SimpleClusterResolver(cluster_spec, task_type="worker", task_id=0)

if cluster_resolver.task_type == 'worker' and cluster_resolver.task_id == 0:
    # Perform something that's only applicable on 'worker' type, id 0. This
    # block will run on this particular instance since we've specified this
    # task to be a 'worker', id 0 in above cluster resolver.
else:
    # Perform something that's only applicable on other ids. This block will
    # not run on this particular instance.
```

若 task ID 在当前分布式环境中不适用，则返回 `None`。



### task_type

返回 `ClusterResolver` 实例指明的 task 类型。

一般在 TensorFlow 分布式环境中，每个 task 有一个相应的 task 类型。这在用户需要根据 task 类型运行特定代码时十分有用，例如：

```python
cluster_spec = tf.train.ClusterSpec({
    "ps": ["localhost:2222", "localhost:2223"],
    "worker": ["localhost:2224", "localhost:2225", "localhost:2226"]
})

# SimpleClusterResolver is used here for illustration; other cluster
# resolvers may be used for other source of task type/id.
cluster_resolver = SimpleClusterResolver(cluster_spec, task_type="worker", task_id=1)

if cluster_resolver.task_type == 'worker':
    # Perform something that's only applicable on workers. This block
    # will run on this particular instance since we've specified this task to
    # be a worker in above cluster resolver.
elif cluster_resolver.task_type == 'ps':
    # Perform something that's only applicable on parameter servers. This
    # block will not run on this particular instance.
```

TensorFlow 中有效的 task 类型包括：

+ `'worker'`：常规的用于训练/测试的工作器
+ `'chief'`：被分配了更多任务的工作器
+ `'ps'`：参数服务器
+ `'evaluator'`：测试检查点保存的模型

若 task 类型在当前分布式环境中不适用，则返回 `None`。



## cluster_resolver.SimpleClusterResolver

集群解析器的简单实现。

```python
tf.distribute.cluster_resolver.SimpleClusterResolver(cluster_spec, master='', task_type=None, task_id=None, environment='', num_accelerators=None, rpc_layer=None)
```



## cluster_resolver.TFConfigClusterResolver

读取 `TF_CONFIG` 环境变量的集群解析器。

```python
tf.distribute.cluster_resolver.TFConfigClusterResolver(task_type=None, task_id=None, rpc_layer=None, environment=None)
```



## coordinator.ClusterCoordinator

用于创建容错资源并分派需要执行的函数给远程 TensorFlow 服务器。

目前此类仅支持与分布式策略 `ParameterServerStrategy` 一起使用。

**处理 task 故障**

此类有内置的对于工作器故障的容错机制，即当部分工作器由于任何原因变得对于协调器不可用时，训练过程由剩余的工作器继续完成。……



### create_per_worker_dataset()

通过在工作器的设备上调用 `dataset_fn()` 创建工作器的数据集。



```python
strategy = tf.distribute.experimental.ParameterServerStrategy(
    cluster_resolver=...)
coordinator = tf.distribute.experimental.coordinator.ClusterCoordinator(
    strategy=strategy)

@tf.function
def worker_fn(iterator):
  return next(iterator)

def per_worker_dataset_fn():
  return strategy.distribute_datasets_from_function(
      lambda x: tf.data.Dataset.from_tensor_slices([3] * 3))

per_worker_dataset = coordinator.create_per_worker_dataset(
    per_worker_dataset_fn)
per_worker_iter = iter(per_worker_dataset)
remote_value = coordinator.schedule(worker_fn, args=(per_worker_iter,))
assert remote_value.fetch() == 3
```



### done()

返回是否所有分派的函数都已经执行完毕。

如果先前分派的函数引发了错误，此方法将引发这些错误中的任意一个。

当此方法返回或引发错误时，可以保证没有任何函数仍在执行。



### fetch()

阻塞直到获取 `RemoteValue` 实例的结果。

```python
strategy = ...
coordinator = tf.distribute.experimental.coordinator.ClusterCoordinator(
    strategy)

def dataset_fn():
  return tf.data.Dataset.from_tensor_slices([1, 1, 1])

with strategy.scope():
  v = tf.Variable(initial_value=0)

@tf.function
def worker_fn(iterator):
  def replica_fn(x):
    v.assign_add(x)
    return v.read_value()
  return strategy.run(replica_fn, args=(next(iterator),))

distributed_dataset = coordinator.create_per_worker_dataset(dataset_fn)
distributed_iterator = iter(distributed_dataset)
result = coordinator.schedule(worker_fn, args=(distributed_iterator,))
assert coordinator.fetch(result) == 1
```



### join()

阻塞直到所有分派的函数执行完毕。

如果先前分派的函数引发了错误，此方法将引发这些错误中的任意一个，并清除已经收集的错误。

当此方法返回或引发错误时，可以保证没有任何函数仍在执行。



### schedule()

分派函数到某个工作器以异步执行。

```python
schedule(fn, args=None, kwargs=None)
# fn       要异步执行的函数
# args     `fn`的位置参数
# kwargs   `fn`的关键字参数
```

此方法是非阻塞的，其将函数 `fn` 排进执行队列并立即返回一个 `RemoteValue` 实例。可以对该实例调用 `fetch()` 方法以等待函数执行结束并从远程工作器获取返回值，或者调用 `join()` 方法以等待所有分派的函数执行结束。

此方法保证 `fn` 将在工作器上执行至少一次；如果在执行的过程中相应的工作器出现故障，则会分派到另一工作器上重新执行。

如果先前分派的函数引发了错误，此方法将引发这些错误中的任意一个，并清除已经收集的错误。这时，部分先前分派的函数可能并没有开始执行（或执行完毕），用户可以通过对返回的 `RemoteValue` 实例调用 `fetch()` 方法来检查函数的当前执行状态。

当此方法引发错误时，可以保证没有任何函数仍在执行。

目前尚不支持为函数指定工作器，或设定函数执行的优先级。





## CrossDeviceOps

跨设备归约和广播算法的基类。`ReductionToOneDevice`, `NcclAllReduce` 和 `HierarchicalCopyAllReduce` 是 `CrossDeviceOps` 的子类，实现了具体的归约算法。

此类的主要目标是被传入到 `MirroredStrategy` 以在不同的跨设备通信实现中进行选择。



## DistributedDataset

表示分布式数据集。

当使用 `tf.distribute` API 进行分布式训练时，通常也需要分布输入数据，这时我们选择 `DistributedDataset` 实例，而不是非分布式情况下的 `tf.data.Dataset` 实例。

有两个 API 用于创建 `DistributedDataset` 实例：`Strategy.experimental_distribute_dataset(dataset)` 和 `Strategy.distribute_datasets_from_function(dataset_fn)`。如果你现有一个 `tf.data.Dataset` 实例，并且适用常规的分批和自动 sharding（即 `tf.data.experimental.AutoShardPolicy` 选项）时，使用前一个 API；如果你不是在使用一个 `tf.data.Dataset` 实例，或者你想要自定义分批和 sharding，那么你可以将这些逻辑包装到 `dataset_fn` 函数中并使用后一个 API。

`DistributedDataset` 实例的主要用法是迭代以产生分布式输入数据，是一个 `DistributedValues` 实例。

```python
>>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
>>> dataset = tf.data.Dataset.from_tensor_slices([5., 6., 7., 8.]).batch(2)
>>> dist_dataset = strategy.experimental_distribute_dataset(dataset)
>>> for x in dist_dataset:
    print(x)
PerReplica:{
  0: tf.Tensor([5.], shape=(1,), dtype=float32),
  1: tf.Tensor([6.], shape=(1,), dtype=float32)
}
PerReplica:{
  0: tf.Tensor([7.], shape=(1,), dtype=float32),
  1: tf.Tensor([8.], shape=(1,), dtype=float32)
}
>>> dataset_iterator = iter(dist_dataset)   # 创建迭代器
>>> next(dataset_iterator)
PerReplica:{
  0: <tf.Tensor: shape=(1,), dtype=float32, numpy=array([5.], dtype=float32)>,
  1: <tf.Tensor: shape=(1,), dtype=float32, numpy=array([6.], dtype=float32)>
}
```

   

## DistributedIterator





## DistributedValues

表示分布式值的基类。

`DistributedValues` 的实例在迭代 `DistributedDataset` 实例、调用 `Strategy.run()` 或在分布式策略内创建变量时被创建；不应直接实例化此基类。`DistributedValues` 实例对每一个模型副本包含一个值，这些值可以是自动同步、手动同步或从不同步，取决于子类的具体实现。

```python
# 由`DistributedDataset`实例创建
>>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
>>> dataset = tf.data.Dataset.from_tensor_slices([5., 6., 7., 8.]).batch(2)
>>> dataset_iterator = iter(strategy.experimental_distribute_dataset(dataset))
>>> next(dataset_iterator)
PerReplica:{
  0: <tf.Tensor: shape=(1,), dtype=float32, numpy=array([5.], dtype=float32)>,
  1: <tf.Tensor: shape=(1,), dtype=float32, numpy=array([6.], dtype=float32)>
}
```

```python
# 由`Strategy.run()`返回
>>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
>>> @tf.function
def f():
    ctx = tf.distribute.get_replica_context()
    return ctx.replica_id_in_sync_group
>>> strategy.run(f)
PerReplica:{
  0: <tf.Tensor: shape=(), dtype=int32, numpy=0>,
  1: <tf.Tensor: shape=(), dtype=int32, numpy=1>
}
```

```python
# 作为`Strategy.run()`的输入
>>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
>>> dataset = tf.data.Dataset.from_tensor_slices([5., 6., 7., 8.]).batch(2)
>>> dataset_iterator = iter(strategy.experimental_distribute_dataset(dataset))
>>> distributed_values = next(dataset_iterator)
>>> @tf.function
def f(x):
    return x * 2.0
>>> strategy.run(f, args=(distributed_values,))
PerReplica:{
  0: <tf.Tensor: shape=(1,), dtype=float32, numpy=array([10.], dtype=float32)>,
  1: <tf.Tensor: shape=(1,), dtype=float32, numpy=array([12.], dtype=float32)>
}
```

```python
# 归约分布式值
>>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
>>> dataset = tf.data.Dataset.from_tensor_slices([5., 6., 7., 8.]).batch(2)
>>> dataset_iterator = iter(strategy.experimental_distribute_dataset(dataset))
>>> distributed_values = next(dataset_iterator)
>>> strategy.reduce(tf.distribute.ReduceOp.SUM,
                    distributed_values,
                    axis=0)
<tf.Tensor: shape=(), dtype=float32, numpy=11.0>
```



## get_replica_context()

返回当前的 `ReplicaContext` 实例。

```python
>>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
>>> tf.distribute.get_replica_context()           # 在默认分布式策略下,返回默认模型副本上下文实例
<tensorflow.python.distribute.distribute_lib._DefaultReplicaContext object at 0x7f1b8057e190>
>>> with strategy.scope():                        # 在`MirroredStrategy`下,返回`None`
    print(tf.distribute.get_replica_context())
None
>>> def f():
    return tf.distribute.get_replica_context()
>>> strategy.run(f)
PerReplica:{                                      # `strategy.run()`返回镜像模型副本上下文实例.是为此函数的通常用法
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



## InputContext

此类是一个上下文类，其实例包含了关于模型副本和输入流水线的信息，用以传入到用户的输入函数。



### get_per_replica_batch_size()

```python
get_per_replica_batch_size(global_batch_size)
```

返回每个模型副本的批次规模。



### input_pipeline_id

输入流水线的 ID。



### num_input_pipelines

输入流水线的数量。



### num_replicas_in_sync

同步的模型副本的数量。





## MirroredStrategy

单机多卡同步训练。此策略下模型的参数是 `MirroredVariable` 类型的变量，在所有的模型副本中通过 all-reduce 模式保持同步。

```python
tf.distribute.MirroredStrategy(devices=None, cross_device_ops=None)
# devices             设备列表.若为`None`或空列表,则使用所有可用的GPU;若没有发现GPU,则使用可用的CPU
#                     注意TensorFlow将一台机器上的多核CPU视作单个设备,并且在内部使用线程并行
# cross_device_ops    `CrossDeviceOps`的子类的实例,默认使用`NcclAllReduce()`.通常在NCCL不可用或者有可用的
#                     能够充分利用特殊硬件的特殊实现时自定义此参数
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

在单个设备上运行。在此策略下创建的变量和通过 `strategy.run()` 调用的函数都会被放置在指定设备上。此策略通常用于在使用其它策略实际分布训练到多个设备/机器之前，测试代码对于 `tf.distribute.Strategy` API 的使用。

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



## partitioners.FixedShardsPartitioner

将数据（张量）分割为固定份数。

```python
>>> partitioner = FixedShardsPartitioner(num_shards=2)
>>> partitioner(tf.TensorShape([10, 3]), tf.float32, axis=0)
[2, 1]
>>> partitioner(tf.TensorShape([10, 3]), tf.float32, axis=1)
[1, 2]
```



## partitioners.MinSizePartitioner

在保证每一份的最小规模的前提下，将数据（张量）分割为尽量多的份数。

```python
>>> partitioner = MinSizePartitioner(min_shard_bytes=24, max_shards=4)   # 每一份最小为24字节
>>> partitioner(tf.TensorShape([10, 3]), tf.float32, axis=0)
[4, 1]              # 最多分为4份
>>> partitioner = MinSizePartitioner(min_shard_bytes=24, max_shards=8)
>>> partitioner(tf.TensorShape([10, 3]), tf.float32, axis=0)
[5, 1]              # 分为5份,每份最小为24字节
```



## partitioners.Partitioner

所有分割器的基类。



### \__call__

分割指定的张量形状并返回分割结果。

```python
__call__(shape, dtype, axis=0)
# shape     要分割的张量的形状,是`tf.TensorShape`实例
# dtype     要分割的张量的数据类型
# axis      沿此轴进行分割
```



## ReduceOp

表示一组值的归约方法。`ReduceOp.SUM` 表示求和，`ReduceOp.MEAN` 表示求平均值。



## ReductionToOneDevice

一种 `CrossDeviceOps` 实现，其复制所有值到一个设备上进行归约，再广播归约结果到目标 rank。

```python
tf.distribute.ReductionToOneDevice(reduce_to_device=None, accumulation_fn=None)
# reduce_to_device   进行归约的中间设备
# accumulation_fn
```







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

进程内的 TensorFlow 服务器，用于分布式训练。

服务器从属于一个集群（通过 `tf.train.ClusterSpec` 指定），并对应指定名称的 job 中的一个特定 task。服务器可以与同一集群中的所有其它服务器进行通信。

```python
tf.distribute.Server(server_or_cluster_def, job_name=None, task_index=None, protocol=None, config=None, start=True)
# server_or_cluster_def    `tf.train.ServerDef`或`tf.train.ClusterDef`协议缓冲区,或`tf.train.ClusterSpec`对象,
#                          用于描述要创建的服务器或其从属的集群
# job_name                 服务器从属的job的名称.默认为`server_or_cluster_def`中的相应值(如果指定了该值)
# task_index               服务器对应的task的索引.默认为`server_or_cluster_def`中的相应值(如果指定了该值);
#                          若job仅有一个task,则默认为0
# protocol                 服务器使用的协议,可以是`'grpc'`或`'grpc+verbs'`.默认为`server_or_cluster_def`中的
#                          相应值(如果指定了该值);其余情况下默认为`'grpc'`
# config
# start                    若为`True`,则在创建服务器之后立即启动
```



### create_local_server()

在本地主机创建一个新的单进程集群。

此方法是一个便利的包装器，用于创建一个服务器，其 `tf.train.ServerDef` 指定了一个单进程集群，集群在名为 `'local'` 的 job 下包含单个 task。

```python
@staticmethod
create_local_server(config=None, start=True)
# config                   
# start                    若为`True`,则在创建服务器之后立即启动它
```



### join()

阻塞直到服务器关闭。



### start()

启动服务器。





## Strategy

在一组设备上进行分布式计算的策略。



### distribute_datasets_from_function()

接收一个输入函数并返回一个分布式数据集（`tf.distribute.DistributedDataset` 实例）。用户传入的输入函数应接收一个 `tf.distribute.InputContext` 实例，返回一个 `tf.data.Dataset` 实例，并进行由用户自定义的分批和分割操作，`tf.distribute` 不会对返回的数据集再进行任何修改。相对于 `experimental_distribute_dataset()`，此方法不仅更加灵活（允许用户自定义分批和分割操作），而且在用于分布式训练时显示出了更好的伸缩性和性能。

```python
>>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
>>> def dataset_fn(input_context):                # 定义输入函数
    global_batch_size = 4
    batch_size = input_context.get_per_replica_batch_size(global_batch_size)
    dataset = tf.data.Dataset.range(8).batch(global_batch_size)
    # dataset = dataset.shard(                      # 手动分割数据集,适用于`MultiWorkerMirroredStrategy`
    #     input_context.num_input_pipelines, input_context.input_pipeline_id)
    dataset = dataset.unbatch().batch(batch_size) # 手动再分批
    dataset = dataset.prefetch(2)                 # 手动添加预取;每个设备预取2个批次(而不是全局总共预取2个批次)
    return dataset
>>> dist_dataset = strategy.distribute_datasets_from_function(dataset_fn)
>>> for x in dist_dataset:
    print(x)
PerReplica:{
  0: tf.Tensor([0 1], shape=(2,), dtype=int64),
  1: tf.Tensor([2 3], shape=(2,), dtype=int64)
}
PerReplica:{
  0: tf.Tensor([4 5], shape=(2,), dtype=int64),
  1: tf.Tensor([6 7], shape=(2,), dtype=int64)
}
```



### experimental_distribute_dataset()

将数据集（`tf.data.Dataset` 实例）转换为分布式数据集（`tf.distribute.DistributedDataset` 实例），`tf.data.Dataset` 实例的批次规模就是分布式训练的全局批次规模。如果你没有特定的想要去分割数据集的方法，则推荐使用此方法。

```python
>>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
>>> dataset = tf.data.Dataset.from_tensor_slices([5., 6., 7., 8.]).batch(2)
>>> dist_dataset = strategy.experimental_distribute_dataset(dataset)
>>> @tf.function
def f(x):
  return x * 2.0
>>> for x in dist_dataset:
    print(strategy.run(f, args=(x,)))
PerReplica:{
  0: tf.Tensor([10.], shape=(1,), dtype=float32),
  1: tf.Tensor([12.], shape=(1,), dtype=float32)
}
PerReplica:{
  0: tf.Tensor([14.], shape=(1,), dtype=float32),
  1: tf.Tensor([16.], shape=(1,), dtype=float32)
}
```



此方法在底层进行了三个关键操作：分批、分割和预取。

**分批**

对输入数据集进行重新分批，新的批次规模等于全局批次规模除以同步的模型副本数量。例如：

+ 输入：`tf.data.Dataset.range(10).batch(4, drop_remainder=False)`

  原分批：`[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]`

  模型副本数量为 2 时分批：副本 1：`[0, 1], [4, 5], [8]`；副本 2：`[2, 3], [6, 7], [9]`

+ 输入：`tf.data.Dataset.range(8).batch(4)`

  原分批：`[0, 1, 2, 3], [4, 5, 6, 7]`

  模型副本数量为 3 时分批：副本 1：`[0, 1], [4, 5]`；副本 2：`[2, 3], [6, 7]`；副本 3：`[], []`
  
+ 输入：`tf.data.Dataset.range(10).batch(5)`
  
  原分批：`[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]`
  
  模型副本数量为 3 时分批：副本 1：`[0, 1], [5, 6]`；副本 2：`[2, 3], [7, 8]`；副本 3：`[4], [9]`

> 上面的例子仅用于展示一个全局批次是如何划分到多个模型副本中的，实际使用时不应对划分结果有任何的假定，因为划分结果可能会随着具体实现而发生变化。

重新分批操作的空间复杂度与模型副本的数量呈线性关系，因此当模型副本数量较多时输入流水线可能会引发 OOM 错误。



**分割**

对输入数据集进行自动分割（在 `MultiWorkerMirroredStrategy` 下），每个模型副本被（不重复不遗漏地）分配原数据集的一个子集，具体到每个 step 中，每个模型副本被（不重复不遗漏地）分配全局批次的一个子集并处理。

自动分割有三种策略可供选择，通过以下方式进行设定：

```python
>>> dataset = tf.data.Dataset.range(16).batch(4)
>>> options = tf.data.Options()                                                      # 设定为`DATA`
>>> options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
>>> dataset = dataset.with_options(options)
```

+ `DATA`：将数据集的样本自动分割到所有模型副本。每个模型副本会读取整个数据集，保留分割给它的那一份，丢弃所有其它的份。此策略通常用于输入文件数量小于模型副本数量的情形，例如将 1 个输入文件分布到 2 个模型副本中：
  + 全局批次规模为 4
  + 文件：`[0, 1, 2, 3, 4, 5, 6, 7]`
  + 副本 1：`[0, 1], [4, 5]`；副本 2：`[2, 3], [6, 7]`
+ `FILE`：将输入文件分割到所有模型副本。每个模型副本会读取分配给它的输入文件，而不会去读取其它文件。此策略通常用于输入文件数量远大于模型副本数量的情形（并且数据均匀分布在各文件中），例如将 2 个输入文件分布到 2 个模型副本中：
  + 全局批次规模为 4
  + 文件 1：`[0, 1, 2, 3]`；文件 2： `[4, 5, 6, 7]`
  + 副本 1：`[0, 1], [2, 3]`；副本 2：`[4, 5], [6, 7]`
+ `AUTO`：默认选项。首先尝试 `FILE` 策略，如果没有检测到基于文件的数据集则尝试失败；然后尝试 `DATA` 策略。
+ `OFF`：关闭自动分割，每个模型副本会处理所有样本：
  + 文件：`[0, 1, 2, 3, 4, 5, 6, 7]`
  + 副本 1：`[0, 1], [2, 3], [4, 5], [6, 7]`；副本 2：`[0, 1], [2, 3], [4, 5], [6, 7]`



**预取**

默认对输入数据集增加一个 `prefetch()` 变换，参数 `buffer_size` 取模型副本的数量。



### experimental_distribute_values_from_function()





### gather()



```python
>>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
INFO:tensorflow:Using MirroredStrategy with devices ('/job:localhost/replica:0/task:0/device:GPU:0', '/job:localhost/replica:0/task:0/device:GPU:1')
>>> distributed_values = strategy.experimental_distribute_values_from_function(lambda _: tf.identity(tf.constant([[1], [2]])))

>>> with strategy.scope():
    distributed_values = tf.Variable([[1], [2]])

strategy.gather(distributed_values, axis=0)


INFO:tensorflow:Gather to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
<tf.Tensor: shape=(4, 1), dtype=int32, numpy=
array([[1],
       [2],
       [1],
       [2]], dtype=int32)>
```





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

```python
strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
# Variable created inside scope:
with strategy.scope():
    mirrored_variable = tf.Variable(1.)
mirrored_variable


# Variable created outside scope:
regular_variable = tf.Variable(1.)
regular_variable

```







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





# tf.train

## Checkpoint



## CheckpointManager





## ClusterDef

协议消息类型。



## ClusterSpec

`ClusterSpec` 实例表示进行 TensorFlow 分布式计算的集群的规格。集群由一组 job 构成，而每个 job 又包含若干 task。

为了创建有 2 个 job 和 5 个 task 的一个集群，我们传入从 job 名称到网络地址列表的映射：

```python
cluster_spec = tf.train.ClusterSpec({"worker": ["worker0.example.com:2222",
                                                "worker1.example.com:2222",
                                                "worker2.example.com:2222"],
                                     "ps": ["ps0.example.com:2222",
                                            "ps1.example.com:2222"]})
```



### as_cluster_def()

返回基于此集群的 `tf.train.ClusterDef` 协议缓冲区。



### as_dict()

返回从 job 名称到其包含的 task 的字典。



### job_tasks()

返回指定 job 中的从 task ID 到网络地址的映射。



### num_tasks()

返回指定 job 中的 task 数量。



### task_address()

返回指定 job 中指定索引的 task 的网络地址。



### task_indices()

返回指定 job 中的有效 task 索引列表。





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

加载已命名的数据集为一个 `tf.data.Dataset` 实例。

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
# split               加载的数据集部分,例如'train','test',['train','test'],'train[80%:]',etc.若为`None`,
#                     则返回部分名称到`Dataset`实例的字典
# data_dir            读/写数据的目录
# batch_size          批次规模,设定后会为样本增加一个批次维度
# shuffle_files       若为`True`,打乱输入文件
# download            若为`True`,在调用`DatasetBuilder.as_dataset()`之前调用
#                     `DatasetBuilder.download_and_prepare()`,如果数据已经在`data_dir`下,则不执行任何操作;
#                     若为`False`,则数据应当存在于`data_dir`下.
# as_supervised       若为`True`,则`Dataset`实例返回的每个样本是一个二元组`(input, label)`;若为`False`,
#                     则`Dataset`实例返回的每个样本是一个包含所有特征的字典,例如`{'input': ..., 'label': ...}`
# decoders
# read_config
# with_info           若为`True`,则返回元组`(Dataset, DatasetInfo)`,后者包含了数据集的信息
# builder_kwargs      传递给`DatasetBuilder`的关键字参数字典
# download_and_prepare_kwargs     传递给`DatasetBuilder.download_and_prepare()`的关键字参数字典
# as_dataset_kwargs               传递给`DatasetBuilder.as_dataset()`的关键字参数字典
# try_gcs
```

```python
>>> datasets, info = tfds.load(name='mnist', with_info=True, as_supervised=True)

```

