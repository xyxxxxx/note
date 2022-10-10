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

用类似张量的对象（python 数组、numpy 数组等）创建一个常数张量。

```python
>>> tf.constant([1, 2, 3, 4, 5, 6])
<tf.Tensor: shape=(6,), dtype=int32, numpy=array([1, 2, 3, 4, 5, 6], dtype=int32)>
  
>>> tf.constant(np.arange(1,7))
<tf.Tensor: shape=(6,), dtype=int64, numpy=array([1, 2, 3, 4, 5, 6])>
```

## convert_to_tensor()

将指定值转换为张量。

```python
>>> tf.convert_to_tensor([[1, 2], [3, 4]])
<tf.Tensor: shape=(2, 2), dtype=int32, numpy=
array([[1, 2],
       [3, 4]], dtype=int32)>

>>> tf.convert_to_tensor(tf.constant([[1, 2], [3, 4]]))
<tf.Tensor: shape=(2, 2), dtype=int32, numpy=
array([[1, 2],
       [3, 4]], dtype=int32)>

>>> tf.convert_to_tensor(np.array([[1, 2], [3, 4]], dtype=np.int32))
<tf.Tensor: shape=(2, 2), dtype=int32, numpy=
array([[1, 2],
       [3, 4]], dtype=int32)>
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

`tf.GradientTape()` 是一个自动求导的记录器。以下示例计算 $y=x^2$ 在 $x=3$ 位置的导数：

```python
import tensorflow as tf

x = tf.Variable(initial_value=3.)   # 初值为3.0的变量
with tf.GradientTape() as tape:     # 在 tf.GradientTape() 的上下文内，所有计算步骤都会被记录以用于求导
    y = tf.square(x)
y_grad = tape.gradient(y, x)        # 计算y关于x的导数
print(y, y_grad)                    # tf.Tensor(6.0, shape=(), dtype=float32)
```

以下示例计算 $\mathcal{L}=||X\pmb w+b-\pmb y||^2$ 在 $\pmb w=[1,2]^{\rm T},b=1$ 位置的对 $\pmb w,b$ 的导数：

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
\mathcal{L}=125，\frac{\partial\mathcal{L}}{\partial\pmb w}=\begin{bmatrix}70\\100\end{bmatrix}，\frac{\partial\mathcal{L}}{\partial b}=30
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
