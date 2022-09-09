[toc]

# NumPy

[NumPy](https://numpy.org/) 是使用 Python 进行科学计算的基本包。

!!! abstract "参考"
    * [NumPy user guide](https://numpy.org/doc/stable/user/index.html)
    * [NumPy Reference](https://numpy.org/doc/stable/reference/index.html)

## 快速入门

### 数组对象

```python
>>> import numpy as np
>>> a = np.array([[6, 5], [11, 7], [4, 8]])
>>> a
array([[ 6,  5],
       [11,  7],
       [ 4,  8]])
>>> type(a)
<class 'numpy.ndarray'>
>>> a[0][0]
6
>>> a.ndim         # 数组维数
2
>>> a.shape        # 数组形状
(3, 2)
>>> a.size         # 数组元素总数
6
>>> a.dtype        # 数组元素数据类型
dtype('int64')
>>> a.itemsize     # 数组元素大小(字节)
8
```

### 创建

见 [np.array()](#array)、[np.zeros()](#zeros)、[np.ones()](#ones)、[np.empty()](#empty)、[np.arange()](#arange)、[np.linspace()](#linspace) 等函数。

### 基本运算

```python
>>> a = np.arange(6).reshape(2, 3)
>>> b = np.arange(6, 12).reshape(2, 3)
>>> c = np.ones(3)
>>> a
array([[0, 1, 2],
       [3, 4, 5]])
>>> b
array([[ 6,  7,  8],
       [ 9, 10, 11]])
>>> c
array([1., 1., 1.])

>>> a + b                  # 逐元素加法
array([[ 6,  8, 10],
       [12, 14, 16]])
>>> a + 1                  # 扩张的逐元素加法
array([[1, 2, 3],
       [4, 5, 6]])
>>> a + c                  # 扩张的逐元素加法
array([[1., 2., 3.],
       [4., 5., 6.]])

>>> np.transpose(a)        # 转置
array([[0, 3],
       [1, 4],
       [2, 5]])
>>> a.T                    # 转置
array([[0, 3],
       [1, 4],
       [2, 5]])

>>> d = np.array([[1, 0], [1, 1]])
>>> e = np.array([1, 2])
>>> d
array([[1, 0],
       [1, 1]])

>>> d * d                  # 逐元素乘法
array([[1, 0],
       [1, 1]])
>>> d * e                  # 扩张的逐元素乘法
array([[1, 0],
       [1, 2]])
>>> d ** 2                 # 逐元素乘方
array([[1, 0],
       [1, 1]])
>>> d ** e                 # 扩张的逐元素乘方
array([[1, 0],
       [1, 1]])

>>> d @ d                  # 矩阵乘法
array([[1, 0],
       [2, 1]])
>>> np.dot(d, d)           # 矩阵乘法
array([[1, 0],             # 矩阵乘法可以扩展到高维张量之间,具体运算规则与矩阵乘法类似
       [2, 1]])
>>> d.dot(d)               # 矩阵乘法
array([[1, 0],
       [2, 1]])

>>> d > 0                  # 逐元素比较
array([[ True, False],
       [ True,  True]])
```

### 切片

```python
>>> a = np.arange(10) ** 2            # 一维数组切片
>>> a
array([ 0,  1,  4,  9, 16, 25, 36, 49, 64, 81])
>>> a[2:5]
array([ 4,  9, 16])
>>> a[:6:2]
array([ 0,  4, 16])
>>> a[::-1]
array([81, 64, 49, 36, 25, 16,  9,  4,  1,  0])

>>> b = np.arange(24).reshape(4, 6)   # 二维数组切片
>>> b
array([[ 0,  1,  2,  3,  4,  5],
       [ 6,  7,  8,  9, 10, 11],
       [12, 13, 14, 15, 16, 17],
       [18, 19, 20, 21, 22, 23]])
>>> b[2][3]
15
>>> b[1:3, 2]
array([ 8, 14])
>>> b[:, 1]
array([ 1,  7, 13, 19])
>>> b[-1]
array([18, 19, 20, 21, 22, 23])

>>> c = np.arange(24).reshape(2, 3, 4)
>>> c
array([[[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]],

       [[12, 13, 14, 15],
        [16, 17, 18, 19],
        [20, 21, 22, 23]]])
>>> c[1, ...]               # 等价于 c[1, :, :]
array([[12, 13, 14, 15],
       [16, 17, 18, 19],
       [20, 21, 22, 23]])
>>> c[..., 2]               # 等价于 c[:, :, 2]
array([[ 2,  6, 10],
       [14, 18, 22]])

>>> for i in c:             # 沿轴0迭代
  print(i)
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]
[[12 13 14 15]
 [16 17 18 19]
 [20 21 22 23]]

>>> for i in c.flat:        # 迭代所有元素
  print(i)
0
1
...
23
```

### 操纵形状

```python
>>> a = np.arange(12).reshape(3, 4)   # 改变数组形状
>>> a
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
>>> a.ravel()             # 展开
array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
>>> a.reshape(2, 6)
array([[ 0,  1,  2,  3,  4,  5],
       [ 6,  7,  8,  9, 10, 11]])
>>> a.T                   # 转置
array([[ 0,  4,  8],
       [ 1,  5,  9],
       [ 2,  6, 10],
       [ 3,  7, 11]])
>>> a.shape               # 不改变原数组形状
(3, 4)
```

```python
>>> a = np.arange(0, 4).reshape(2, 2)  # 堆叠数组
>>> b = np.arange(4, 8).reshape(2, 2)
>>> np.vstack((a, b))     # 垂直堆叠
array([[0, 1],
       [2, 3],
       [4, 5],
       [6, 7]])
>>> np.hstack((a, b))     # 水平堆叠
array([[0, 1, 4, 5],
       [2, 3, 6, 7]])
```

```python
>>> a = np.arange(24).reshape(4, 6)    # 拆分数组
>>> a
array([[ 0,  1,  2,  3,  4,  5],
       [ 6,  7,  8,  9, 10, 11],
       [12, 13, 14, 15, 16, 17],
       [18, 19, 20, 21, 22, 23]])
>>> np.hsplit(a, 3)
[array([[ 0,  1],
       [ 6,  7],
       [12, 13],
       [18, 19]]), array([[ 2,  3],
       [ 8,  9],
       [14, 15],
       [20, 21]]), array([[ 4,  5],
       [10, 11],
       [16, 17],
       [22, 23]])]
>>> np.vsplit(a, 2)
[array([[ 0,  1,  2,  3,  4,  5],
       [ 6,  7,  8,  9, 10, 11]]), array([[12, 13, 14, 15, 16, 17],
       [18, 19, 20, 21, 22, 23]])]
```

### 复制和视图

```python
>>> a = np.arange(12)

>>> b = a           # 同一对象
>>> b is a
True

>>> c = a.view()    # 浅拷贝(视图)
# or
# >>> c = a[:]
>>> c is a
False
>>> c.base is a
True
>>> c = c.reshape(2, 6)
>>> c.shape
(2, 6)              # 视图形状改变
>>> a.shape
(12,)               # 原数组形状不变
>>> c[0, 4] = 0     # 视图数据改变
>>> a               # 原数组数据改变
array([ 0,  1,  2,  3,  0,  5,  6,  7,  8,  9, 10, 11])

>>> d = a.copy()    # 深拷贝(副本)
>>> d is a
False
>>> d.base is a
False
>>> d[4] = 0
>>> d = a.copy()
>>> d[8] = 0        # 副本数据改变
>>> a               # 原数组数据不变
array([ 0,  1,  2,  3,  0,  5,  6,  7,  8,  9, 10, 11])
```

## API

### numpy

#### add(), sub()

逐元素相加/相减。符号 `+, -` 重载了这些方法。

#### all(), any()

测试是否数组的所有/任意元素为 `True`。亦为 `numpy.ndarray` 方法。

```python
>>> np.all([[True,False],[True,True]])
False
>>> np.any([[True,False],[True,True]])
True
```

#### amax(), amin()

返回数组沿指定轴的最大/最小值。亦为 `numpy.ndarray` 方法（方法名为 `max()` 和 `min()`）。

```python
>>> a = np.arange(4).reshape((2,2))
>>> a
array([[0, 1],
       [2, 3]])
>>> np.amax(a)           # 展开的数组的最大值 
3
>>> np.amax(a, axis=0)   # 沿轴0的最大值
array([2, 3])
>>> np.amax(a, axis=1)   # 沿轴1的最大值
array([1, 3])
```

#### append()

将数组展开为向量，并在末尾添加给定元素。

```python
>>> a = np.arange(6).reshape(2,3)
>>> np.append(a, 10)
array([ 0,  1,  2,  3,  4,  5, 10])
```

#### arange()

根据给定的初值、末值和步长创建向量。与 Python 的 `range()` 用法相同。

```python
>>> np.arange(10)
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> np.arange(0, 10)
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> np.arange(0, 10, 0.5)
array([0. , 0.5, 1. , 1.5, 2. , 2.5, 3. , 3.5, 4. , 4.5, 5. , 5.5, 6. ,
       6.5, 7. , 7.5, 8. , 8.5, 9. , 9.5])
```

#### argmax(), argmin()

返回沿指定轴的最大值的索引。

```python
>>> a = np.array([[0, 3, 5], [2, 1, 4]])
>>> np.argmax(a)            # 默认展开
2
>>> np.argmin(a)
0
>>> np.argmax(a, axis=0)    # 沿轴0
array([1, 0, 0])
>>> np.argmin(a, axis=0)
array([0, 1, 1])
>>> np.argmax(a, axis=1)    # 沿轴1
array([2, 2])
>>> np.argmin(a, axis=1)
array([0, 1])
```

#### argsort()

返回沿指定轴的排序索引，默认沿轴 -1。

```python
>>> a = np.array([[0, 3, 5], [2, 1, 4]])
>>> np.argsort(a)
array([[0, 1, 2],
       [1, 0, 2]])
>>> np.argsort(a, axis=0)
array([[0, 1, 1],
       [1, 0, 0]])
```

#### array()

创建数组。

```python
>>> vector = np.array([1.2, 2.4, 3.5, 4.7, 6.1, 7.2, 8.3, 9.5])    # 一维数组
>>> vector
array([1.2, 2.4, 3.5, 4.7, 6.1, 7.2, 8.3, 9.5])

>>> matrix = np.array([[6, 5], [11, 7], [4, 8]])                   # 二维数组
>>> matrix
array([[ 6,  5],
       [11,  7],
       [ 4,  8]])

>>> cmatrix = np.array([[6, 5], [11, 7], [4, 8]], dtype=complex)   # 指定数据类型
>>> cmatrix
array([[ 6.+0.j,  5.+0.j],
       [11.+0.j,  7.+0.j],
       [ 4.+0.j,  8.+0.j]])

>>> matrix = np.array(((6, 5), (11, 7), (4, 8)))         # 可以是Python列表或元组
>>> matrix
array([[ 6,  5],
       [11,  7],
       [ 4,  8]])
```

#### array_equal()

#### asarray()

将输入数据转换为数组。

```python
>>> a = np.asarray([1, 2])
>>> a
array([1, 2])
>>> np.asarray(a) is a
True
>>> np.asarray(a, dtype=np.int8) is a
False
```

#### c_

将切片对象沿轴 1 拼接。

```python
>>> np.c_[np.array([1, 2, 3]), np.array([4, 5, 6])]
array([[1, 4],
       [2, 5],
       [3, 6]])
```

#### ceil(), floor()

逐元素向上/向下取整。

```python
>>> a = np.random.randn(4)
>>> a
array([ 0.5495602 , -1.44564046, -0.82303694,  0.49965467])
>>> np.ceil(a)
array([ 1., -1., -0.,  1.])
>>> np.floor(a)
array([ 0., -2., -1.,  0.])
```

#### choose()

使用数组列表和索引列表构建一个数组。可以视作 `where()` 方法扩展到多个数组的版本。亦为 `numpy.ndarray` 方法。

```python
>>> a = np.arange(12).reshape(3, 4)
>>> a
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
>>> np.choose([1, 2, 1, 0], a)  # 4个元素从左到右分别从第 1,2,1,0 行选取
array([4, 9, 6, 3])
```

#### concatenate()

沿指定轴拼接数组。

```python
>>> a = np.array([[1, 2], [3, 4]])
>>> b = np.array([[5, 6]])
>>> np.concatenate((a, b), axis=0)
array([[1, 2],
       [3, 4],
       [5, 6]])
>>> np.concatenate((a, b.T), axis=1)
array([[1, 2, 5],
       [3, 4, 6]])
>>> np.concatenate((a, b), axis=None)
array([1, 2, 3, 4, 5, 6])
```

#### copy()

返回数组的深拷贝（副本）。亦为 `numpy.ndarray` 方法。

#### corrcoef()

计算两个向量的相关系数。

```python
>>> a = np.array([1., 2, 3])
>>> b = np.array([3., 2, 1])
>>> np.corrcoef(a,b)
array([[ 1., -1.],
       [-1.,  1.]])

>>> c = np.array([[10., 20, 30],[30, 20, 10],[20, 20, 20]])
>>> np.corrcoef(c)
array([[ 1., -1., nan],
       [-1.,  1., nan],
       [nan, nan, nan]])
```

#### cov()

计算两个向量的协方差。

```python
>>> a = np.array([1., 2, 3])
>>> b = np.array([3., 2, 1])
>>> np.cov(a,b)
array([[ 1., -1.],
       [-1.,  1.]])

>>> c = np.array([[10., 20, 30],[30, 20, 10],[20, 20, 20]])
>>> np.cov(c)
array([[ 100., -100.,    0.],
       [-100.,  100.,    0.],
       [   0.,    0.,    0.]])
```

#### cumsum(), cumprod()

沿指定轴对数组元素累积求和/求积。。亦为 `numpy.ndarray` 方法。

```python
>>> a = np.arange(6).reshape(2, 3)
>>> a
array([[0, 1, 2],
       [3, 4, 5]])
>>> a.cumsum()
array([ 0,  1,  3,  6, 10, 15])
>>> a.cumsum(axis=1)
array([[ 0,  1,  3],
       [ 3,  7, 12]])
>>> a.cumprod()
array([0, 0, 0, 0, 0, 0])
>>> a.cumprod(axis=1)
array([[ 0,  0,  0],
       [ 3, 12, 60]])
```

#### datetime, timedeltas

创建日期时间类型和时间差类型。

```python
>>> np.datetime64('2005-02-25')
numpy.datetime64('2005-02-25')
>>> np.datetime64('2005-02-25T03:30')
numpy.datetime64('2005-02-25T03:30')
>>> np.datetime64('2005-02-25T03:30') + np.timedelta64(10, 'm')
numpy.datetime64('2005-02-25T03:40')

>>> np.arange('2005-02-25T00:00', '2005-02-26T00:00', np.timedelta64(10, 'm'), dtype='datetime64[s]')
array(['2005-02-25T00:00:00', '2005-02-25T00:10:00',
       '2005-02-25T00:20:00', '2005-02-25T00:30:00',
       ...
       '2005-02-25T23:40:00', '2005-02-25T23:50:00'],
      dtype='datetime64[s]')

```

#### delete()

删除数组指定轴和指定索引上的元素。

```python
>>> a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
>>> a
array([[ 1,  2,  3,  4],
       [ 5,  6,  7,  8],
       [ 9, 10, 11, 12]])
>>> np.delete(a, 1, axis=0)
array([[ 1,  2,  3,  4],
       [ 9, 10, 11, 12]])
>>> np.delete(a, 1, axis=1)
array([[ 1,  3,  4],
       [ 5,  7,  8],
       [ 9, 11, 12]])
```

#### diag()

返回对角矩阵的对角线向量，或以向量为对角线元素的对角矩阵。

```python
>>> a
array([1, 2, 3, 4])
>>> np.diag(a)
array([[1, 0, 0, 0],
       [0, 2, 0, 0],
       [0, 0, 3, 0],
       [0, 0, 0, 4]])
>>> np.diag(np.diag(a))
array([1, 2, 3, 4])
```

#### empty()

返回一个指定形状和类型的未初始化值的数组。

```python
>>> np.empty((2, 2))
array([[2.20373199e-316, 0.00000000e+000],
       [2.48020954e-321, 2.24706625e-310]])
>>> np.empty((2, 2), dtype=int)
array([[      44604032,              0],
       [           502, 45481127201069]])
```

#### exp()

对数组的每个元素计算自然指数。

```python
>>> a = np.arange(3)
>>> a
array([0, 1, 2])
>>> np.exp(a)
array([1.        , 2.71828183, 7.3890561 ])
```

#### expand_dims()

增加数组的一个维度。

```python
>>> a = np.random.randn(3, 4)
>>> np.expand_dims(a, 0).shape  # 等价于 a.reshape(1, 3, 4)
(1, 3, 4)
>>> np.expand_dims(a, 1).shape
(3, 1, 4)
>>> np.expand_dims(a, 2).shape
(3, 4, 1)
```

#### eye()

返回一个对角线元素为 1、其余元素为 0 的二维数组。

```python
>>> np.eye(2, dtype=int)
array([[1, 0],
       [0, 1]])
>>> np.eye(3, k=1)
array([[0.,  1.,  0.],
       [0.,  0.,  1.],
       [0.,  0.,  0.]])
```

#### fromfunction()

构建一个数组，通过执行一个所有坐标的函数。

```python
>>> np.fromfunction(lambda i, j: i == j, (3, 3), dtype=int)
                  # 计算元素值的函数      数组形状
array([[ True, False, False],
       [False,  True, False],
       [False, False,  True]])
>>> np.fromfunction(lambda i, j: i + j, (3, 3), dtype=int)
array([[0, 1, 2],
       [1, 2, 3],
       [2, 3, 4]])
```

#### hsplit(), vsplit(), dsplit()

水平/垂直/沿深度拆分数组。

```python
>>> a = np.arange(24).reshape(4, 6)
>>> a
array([[ 0,  1,  2,  3,  4,  5],
       [ 6,  7,  8,  9, 10, 11],
       [12, 13, 14, 15, 16, 17],
       [18, 19, 20, 21, 22, 23]])
>>> np.vsplit(a, 2)                 # 垂直拆分
[array([[ 0,  1,  2,  3,  4,  5],
       [ 6,  7,  8,  9, 10, 11]]), array([[12, 13, 14, 15, 16, 17],
       [18, 19, 20, 21, 22, 23]])]
>>> np.hsplit(a, 3)                 # 水平拆分
[array([[ 0,  1],
       [ 6,  7],
       [12, 13],
       [18, 19]]), array([[ 2,  3],
       [ 8,  9],
       [14, 15],
       [20, 21]]), array([[ 4,  5],
       [10, 11],
       [16, 17],
       [22, 23]])]
```

#### hstack(), vstack(), dstack()

水平/垂直/沿深度堆叠数组。

```python
>>> a = np.arange(0, 4).reshape(2, 2)
>>> b = np.arange(4, 8).reshape(2, 2)
>>> np.vstack((a, b))     # 垂直堆叠
array([[0, 1],
       [2, 3],
       [4, 5],
       [6, 7]])
>>> np.hstack((a, b))     # 水平堆叠
array([[0, 1, 4, 5],
       [2, 3, 6, 7]])
```

#### identity()

返回一个指定形状的单位数组（单位矩阵）。

```python
>>> np.identity(2)
array([[1., 0.],
       [0., 1.]])
>>> np.identity(2, dtype=int)
array([[1, 0],
       [0, 1]])
```

#### inner(), dot(), cross(), outer()

计算两个数组的点积（内积）/叉积/外积。

```python
>>> a = np.array([1, 2, 3])
>>> b = np.array([0, 1, 0])
>>> np.inner(a, b)    # 内积
2
>>> np.dot(a, b)      # 点积
2
>>> np.cross(a, b)    # 叉积
array([-3,  0,  1])
>>> np.outer(a, b)    # 外积
array([[0, 1, 0],
       [0, 2, 0],
       [0, 3, 0]])
```

#### insert()

在数组的指定轴的指定索引位置插入元素。

```python
>>> a = np.zeros((4,4))
>>> np.insert(a, 0, 1, axis=1)  # 在所有 axis=1, index=0 的位置插入1
array([[1., 0., 0., 0., 0.],
       [1., 0., 0., 0., 0.],
       [1., 0., 0., 0., 0.],
       [1., 0., 0., 0., 0.]])
>>> np.insert(a, 0, 1, axis=0)  # 在所有 axis=0, index=0 的位置插入1
array([[1., 1., 1., 1.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.]])
```

#### linspace()

返回指定区间上的指定规模的等差序列。一般用于绘制坐标图。

```python
>>> np.linspace(0, 5, 11)
array([0. , 0.5, 1. , 1.5, 2. , 2.5, 3. , 3.5, 4. , 4.5, 5. ])
```

#### logspace()

返回指定区间上的指定规模的对数坐标下的等差序列。一般用于绘制对数坐标图。

```python
>>> np.logspace(0, 3, 7)
array([   1.        ,    3.16227766,   10.        ,   31.6227766 ,
        100.        ,  316.22776602, 1000.        ])
```

#### max(), min()

`amax()` 和 `amin()` 的别名。

#### maximum(), minimum()

逐元素取较大/较小值。

```python
>>> np.maximum([2, 3, 4], [1, 5, 2])
array([2, 5, 4])
>>> np.maximum(np.eye(2), [0.5, 2])    # 扩张
array([[ 1. ,  2. ],
       [ 0.5,  2. ]])
```

#### mgrid(), ogrid()

返回一个包含密集的/开放的网格数据的数组。

```python
>>> np.mgrid[0:2, 0:3]
array([[[0, 0, 0],      # 纵坐标
        [1, 1, 1]],

       [[0, 1, 2],      # 横坐标
        [0, 1, 2]]])
>>> np.ogrid[0:2, 0:3]
[array([[0],            # 纵坐标
        [1]]), 
 array([[0, 1, 2]])]    # 横坐标
```

#### newaxis

`None` 的别名，用于为数组增加新的维度。

```python
>>> a = np.arange(4)
>>> a[:]
array([0, 1, 2, 3])
>>> a[:, np.newaxis]
array([[0],
       [1],
       [2],
       [3]])
>>> a[:, np.newaxis, np.newaxis].shape
(4, 1, 1)
```

#### nonezero

返回所有非零元素的索引。

```python
>>> a = np.array([[3, 0, 0], [0, 4, 0], [5, 6, 0]])
>>> a
array([[3, 0, 0],
       [0, 4, 0],
       [5, 6, 0]])
>>> np.nonzero(a)
(array([0, 1, 2, 2]), array([0, 1, 0, 1]))
```

#### ones()

返回一个指定形状和类型的全 1 数组。

```python
>>> np.ones((2, 2))
array([[1., 1.],
       [1., 1.]])
>>> np.ones((2, 2), dtype=int)
array([[1, 1],
       [1, 1]])
>>> np.ones((2, 2), dtype=np.int8)
array([[1, 1],
       [1, 1]], dtype=int8)
```

#### pi

圆周率 $\pi$。

```python
>>> np.sin(np.pi/2)
1.0
```

#### put

以指定值替换数组的指定元素。

```python
>>> a = np.arange(5)
>>> np.put(a, [0, 2], [-44, -55])
>>> a
array([-44,   1, -55,   3,   4])
```

#### putmask

根据条件改变数组元素的值。

```python
>>> a = np.arange(6).reshape(2, 3)
>>> np.putmask(a, a > 2, a ** 2)
>>> a
array([[ 0,  1,  2],
       [ 9, 16, 25]])
```

#### r_

将切片对象沿轴 0 拼接。

```python
>>> np.r_[np.array([1, 2, 3]), 0, 0, np.array([4, 5, 6])]
array([1, 2, 3, ..., 4, 5, 6])
```

#### ravel()

展开数组。亦为 `numpy.ndarray` 方法。

```python
>>> a = np.arange(12).reshape(3, 4)
>>> a.ravel()
array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
```

#### repeat()

以重复输入数组元素的方式构建数组。

```python
>>> a = np.arange(12).reshape(3, 4)
>>> np.repeat(a, 2, 0)           # 沿轴0, 各重复2次
array([[ 0,  1,  2,  3],
       [ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11],
       [ 8,  9, 10, 11]])
>>> np.repeat(a, 2, 1)           # 沿轴1, 各重复2次
array([[ 0,  0,  1,  1,  2,  2,  3,  3],
       [ 4,  4,  5,  5,  6,  6,  7,  7],
       [ 8,  8,  9,  9, 10, 10, 11, 11]])
```

#### reshape()

改变数组的形状而不改变元素总数。亦为 `numpy.ndarray` 方法。

```python
>>> np.arange(6).reshape(2, 3)
array([[0, 1, 2],
       [3, 4, 5]])
```

#### resize()

改变数组的形状，缺少的元素用重复的元素填充。亦为 `numpy.ndarray` 方法（原位操作，缺少的元素用 0 填充）。

```python
>>> a = np.arange(6)
>>> np.resize(a, (2, 3))
array([[0, 1, 2],
       [3, 4, 5]])
>>> np.resize(a, (2, 2))     # 舍弃2个元素
array([[0, 1],
       [2, 3]])
>>> np.resize(a, (2, 4))     # 填充2个重复元素
array([[0, 1, 2, 3],
       [4, 5, 0, 1]])
>>> a.resize((2, 4))         # 原位操作
>>> a
array([[0, 1, 2, 3],
       [4, 5, 0, 0]])        # 缺少的元素用 0 填充
```

#### sin(), cos(), tan(), arcsin(), arccos(), arctan(), sinh(), cosh(), tanh()

对数组的每个元素计算三角函数和双曲函数。

#### sort()

对数组沿指定轴进行排序，默认沿轴 -1。亦为 `numpy.ndarray` 方法（原位操作）。

```python
>>> a = np.random.permutation(np.arange(24)).reshape(4, 6)
>>> a
array([[19, 15,  9,  8,  4,  6],
       [12, 11, 18, 13, 20, 16],
       [22,  1, 21,  3,  5, 23],
       [ 0, 14,  2, 10, 17,  7]])
>>> np.sort(a)
array([[ 4,  6,  8,  9, 15, 19],
       [11, 12, 13, 16, 18, 20],
       [ 1,  3,  5, 21, 22, 23],
       [ 0,  2,  7, 10, 14, 17]])
>>> np.sort(a, axis=0)
array([[ 0,  1,  2,  3,  4,  6],
       [12, 11,  9,  8,  5,  7],
       [19, 14, 18, 10, 17, 16],
       [22, 15, 21, 13, 20, 23]])
```

#### sqrt()

对数组的每个元素计算平方根。

```python
>>> a = np.arange(3)
>>> a
array([0, 1, 2])
>>> np.sqrt(a)
array([0.        , 1.        , 1.41421356])
```

#### squeeze()

移除一个长度为 1 的轴。

```python
>>> a = np.array([[[0], [1], [2]]])
>>> a.shape
(1, 3, 1)
>>> np.squeeze(a).shape
(3,)
>>> np.squeeze(a, axis=0).shape
(3, 1)
```

#### stack()

将数组序列沿指定轴堆叠。

```python
>>> a1 = np.arange(3)
>>> np.stack((a1,a1,a1),axis=0)
array([[0, 1, 2],
       [0, 1, 2],
       [0, 1, 2]])

>>> arrays = [np.random.randn(3, 4) for _ in range(10)]  # a list of array
>>> np.stack(arrays, axis=0).shape
(10, 3, 4)
>>> np.stack(arrays, axis=1).shape
(3, 10, 4)
>>> np.stack(arrays, axis=2).shape
(3, 4, 10)
```

#### sum(), prod()

沿指定轴对数组元素求和/求积。亦为 `numpy.ndarray` 方法。

```python
>>> a = np.arange(6).reshape(2, 3)
>>> a
array([[0, 1, 2],
       [3, 4, 5]])
>>> a.sum()
15
>>> a.sum(axis=1)
array([ 3, 12])
>>> a.prod()
0
>>> a.prod(axis=1)
array([ 0, 60])
```

#### swapaxes()

交换数组的指定两个维度。

```python
>>> a = np.arange(10).reshape(2, 5)
>>> a
array([[0, 1, 2, 3, 4],
       [5, 6, 7, 8, 9]])
>>> np.swapaxes(a, 0, 1)
array([[0, 5],
       [1, 6],
       [2, 7],
       [3, 8],
       [4, 9]])
```

#### take()

（沿指定轴）拿出一些元素。

```python
>>> a = np.arange(12).reshape(3, 4)
>>> np.take(a, [5, 7])          # 相当于 np.take(a.ravel(), [5, 7])
array([5, 7])
>>> a
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
>>> np.take(a, [1, 3], axis=1)
array([[ 1,  3],
       [ 5,  7],
       [ 9, 11]])
```

#### tranpose()

返回将数组的所有维度重新排序得到的数组，默认为反转所有维度。

```python
>>> a = np.arange(24).reshape(2, 3, 4)
>>> a
array([[[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]],

       [[12, 13, 14, 15],
        [16, 17, 18, 19],
        [20, 21, 22, 23]]])
>>> np.transpose(a).shape
(4, 3, 2)
>>> np.transpose(a)
array([[[ 0, 12],
        [ 4, 16],
        [ 8, 20]],

       [[ 1, 13],
        [ 5, 17],
        [ 9, 21]],

       [[ 2, 14],
        [ 6, 18],
        [10, 22]],

       [[ 3, 15],
        [ 7, 19],
        [11, 23]]])
>>> np.transpose(a, (1, 2, 0)).shape
(3, 4, 2)
```

#### where()

根据条件从两个数组中逐元素取值。

```python
>>> a = np.arange(10)
>>> a
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> np.where(a < 5, a, 10 * a)
array([ 0,  1,  2,  3,  4, 50, 60, 70, 80, 90])

>>> np.where([[True, False], [True, True]],
...          [[1, 2], [3, 4]],
...          [[9, 8], [7, 6]])
array([[1, 8],
       [3, 4]])
```

#### zeros()

返回一个指定形状和类型的全 0 数组。

```python
>>> np.zeros((2, 2))
array([[0., 0.],
       [0., 0.]])
>>> np.zeros((2, 2), dtype=int)
array([[0, 0],
       [0, 0]])
>>> np.zeros((2, 2), dtype=np.int8)
array([[0, 0],
       [0, 0]], dtype=int8)
```

### numpy.ndarray

#### astype()

复制数组并转为指定类型。

```python
>>> a = np.array([1, 2, 2.5])
>>> a.astype(int)
array([1, 2, 2])
>>> a
array([1. , 2. , 2.5])
```

#### base

如果数组引用另一个数组的内存（即浅拷贝或视图），则返回后者。

```python
>>> a = np.arange(4)
>>> b = a.view()
>>> c = a.copy()
>>> a.base is None
True
>>> b.base is a
True
>>> c.base is a
False
```

#### fill()

以指定标量值填充数组。

```python
>>> a = np.empty(2)
>>> a.fill(1)
>>> a
array([1, 1])
```

#### flat

返回数组元素的迭代器。

#### shape

返回数组的形状。

```python
>>> np.zeros((2, 3, 4)).shape
(2, 3, 4)
```

#### size

返回数组的元素数量。

```python
>>> np.zeros((2, 3, 4)).size
24
```

#### T

返回数组的转置。

#### view()

返回数组的视图。

```python
>>> a = np.arange(4)
>>> b = a.view(dtype=np.float64)
>>> a
array([0, 1, 2, 3])
>>> b
array([0.0e+000, 4.9e-324, 9.9e-324, 1.5e-323])
```

### numpy.random

#### choice()

随机抽取向量中的元素。

```python
>>> a
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> np.random.choice(a)         # 随机抽取1次
6
>>> np.random.choice(a, 5)      # 随机抽取5次
array([7, 4, 4, 8, 5])
>>> np.random.choice(a, 5, replace=False)    # 无重复地随机抽取5次
array([8, 1, 5, 9, 0])          # 相当于 np.random.permutation(a)[:5]
>>> np.random.choice(10, 5, replace=False)   # 10 视作向量 np.arange(10)
array([8, 1, 5, 9, 0])
```

#### randint()

根据给定的初值，末值和长度给出整数向量。

```python
>>> np.random.randint(0, 10, 20)
array([1, 1, 9, 2, 3, 1, 1, 4, 4, 5, 6, 4, 3, 7, 2, 1, 9, 7, 5, 5])
```

#### randn()

根据给定的形状给出随机数数组，其中每个元素服从标准正态分布。

```python
>>> np.random.randn(3,3)
array([[ 0.1283324 , -0.22434515,  1.47125353],
       [ 0.42664038, -2.32914921,  1.09965505],
       [ 0.07222941, -1.72843008,  0.4844523 ]])
```

#### permutation()

随机返回向量中所有元素的全排列中的一种。

```python
>>> a
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> np.random.permutation(a)
array([3, 2, 8, 7, 0, 9, 6, 4, 1, 5])
>>> np.random.permutation(10)           # 10 视作向量 np.arange(10)
array([3, 2, 8, 7, 0, 9, 6, 4, 1, 5])
```

#### random(), rand()

根据给定的长度给出随机数向量，其中每个元素服从 (0, 1) 区间的均匀分布。

```python
>>> np.random.random(10)
array([0.50612532, 0.9510928 , 0.13222277, 0.14056485, 0.60235894,
       0.07380603, 0.14724741, 0.06506975, 0.61176615, 0.21780161])

# randn() * std + mean represents any normal distribution
>>> np.random.randn(4,4)* 5 + 100
array([[ 95.57991558, 106.66569132,  98.76637248, 100.29172391],
       [103.12722419,  91.42214194,  96.97500601, 102.07524186],
       [104.1694757 , 101.39605899, 105.41504323, 100.6351175 ],
       [103.24279736,  96.43976482,  98.31900979, 100.49625635]])
```

#### seed()

指定随机数种子，一次有效。

```python
>>> np.random.seed(0)
>>> np.random.random(10)
array([0.5488135 , 0.71518937, 0.60276338, 0.54488318, 0.4236548 ,
       0.64589411, 0.43758721, 0.891773  , 0.96366276, 0.38344152])
```

#### shuffle()

随机打乱数组（沿轴 0 的）各元素的位置。

```python
>>> a = np.arange(12)
>>> np.random.shuffle(a)
>>> a
array([ 5,  0, 11,  4,  1,  2,  3,  9,  7,  6,  8, 10])

>>> a = np.arange(12).reshape(3, 4)
>>> np.random.shuffle(a)
>>> a
array([[ 8,  9, 10, 11],
       [ 4,  5,  6,  7],
       [ 0,  1,  2,  3]])
```

### numpy.linalg

#### eig()

返回数组的一个特征分解 $A=Q\Lambda Q^{-1}$。

```python
>>> from numpy import linalg as LA
>>> a
array([[ 4.,  6.,  0.],
       [-3., -5.,  0.],
       [-3., -6.,  1.]])
>>> LA.eig(a)
(array([ 1., -2.,  1.]),                          # Λ
 array([[ 0.        ,  0.57735027, -0.89442719],  # Q
       [ 0.        , -0.57735027,  0.4472136 ],
       [ 1.        , -0.57735027,  0.        ]]))
```

#### svd()

返回数组的一个奇异值分解 $A=U\Sigma V^*$。

```python
>>> from numpy import linalg as LA
>>> a
array([[1., 0., 0., 0., 2.],
       [0., 0., 3., 0., 0.],
       [0., 0., 0., 0., 0.],
       [0., 4., 0., 0., 0.]])
>>> LA.svd(a, full_matrices=True)  # 返回 U 为 mxm 矩阵, V 为 nxn 矩阵
(array([[ 0.,  0.,  1.,  0.],
       [ 0.,  1.,  0.,  0.],
       [ 0.,  0.,  0., -1.],
       [ 1.,  0.,  0.,  0.]]),
 array([4.        , 3.        , 2.23606798, 0.        ]), 
 array([[-0.        ,  1.        ,  0.        , -0.        ,  0.        ],
       [-0.        ,  0.        ,  1.        , -0.        ,  0.        ],
       [ 0.4472136 ,  0.        ,  0.        ,  0.        ,  0.89442719],
       [ 0.        ,  0.        ,  0.        ,  1.        ,  0.        ],
       [-0.89442719,  0.        ,  0.        ,  0.        ,  0.4472136 ]]))
```

### 集合函数

#### intersect1d()

找到两个数组的交集，返回一个由同时属于两个数组的所有唯一元素组成的排序的数组。

```python
>>> np.intersect1d([1, 3, 4, 3], [3, 1, 2, 1])
array([1, 3])

>>> from functools import reduce  # 对多于两个数组调用
>>> reduce(np.intersect1d, ([1, 3, 4, 3], [3, 1, 2, 1], [6, 3, 4, 2]))
array([3])
```

#### isin()

对于数组中的每一个元素，判断其是否出现在另一个数组中。

```python
>>> element = np.array([[0, 2], [4, 6]])
>>> test_elements = [1, 2, 4, 8]
>>> mask = np.isin(element, test_elements)
>>> mask
array([[False,  True],
       [ True, False]])
>>> element[mask]
array([2, 4])

>>> mask = np.isin(element, test_elements, invert=True)  # 相当于 `isnotin()`
>>> mask
array([[ True, False],
       [False,  True]])
>>> element[mask]
array([0, 6])
```

#### setdiff1d()

找到两个数组的差集，返回一个由属于第一个数组但不属于第二个数组的所有唯一元素组成的排序的数组。

```python
>>> np.setdiff1d([1, 2, 3, 2, 4, 1], [3, 4, 5, 6])
array([1, 2])
```

#### setxor1d()

找到两个数组的差集，返回一个由属于两个数组中的任意一个（但不同时属于两个数组）的所有唯一元素组成的排序的数组。

```python
>>> np.setxor1d([1, 2, 3, 2, 4], [2, 3, 5, 7, 5])
array([1, 4, 5, 7])
```

#### union1d()

找到两个数组的并集，返回一个由属于两个数组中的任意一个的所有唯一元素组成的排序的数组。

```python
>>> np.union1d([-1, 0, 1], [-2, 0, 2])
array([-2, -1,  0,  1,  2])

>>> from functools import reduce  # 对多于两个数组调用
>>> reduce(np.union1d, ([1, 3, 4, 3], [3, 1, 2, 1], [6, 3, 4, 2]))
array([1, 2, 3, 4, 6])
```

#### unique()

找到数组的所有唯一元素，返回一个由所有唯一元素组成的排序的数组。

```python
>>> np.unique([3, 3, 1, 1, 2, 2])            # 排序数值
array([1, 2, 3])
>>> np.unique(np.array([[3, 1], [1, 2]]))
array([1, 2, 3])

>>> a = np.array(['a', 'b', 'b', 'c', 'a'])  # 排序字符串
>>> u, indices = np.unique(a, return_index=True)
>>> u                       # 返回各元素在原数组中第一次出现的索引
array(['a', 'b', 'c'], dtype='<U1')
>>> indices
array([0, 1, 3])

>>> a = np.array([1, 2, 6, 4, 2, 3, 2])
>>> u, indices = np.unique(a, return_inverse=True)
>>> u                       # 返回可用于重建原数组的索引
array([1, 2, 3, 4, 6])
>>> indices
array([0, 1, 4, 3, 1, 2, 1])

>>> a = np.array([1, 2, 6, 4, 2, 3, 2])
>>> u, counts = np.unique(a, return_counts=True)
>>> u                      # 返回各元素在原数组中出现的次数
array([1, 2, 3, 4, 6])
>>> counts
array([1, 3, 1, 1, 1])
```

### 统计函数

#### histogram()

计算数组中所有元素构成的直方图。

!!! note "注意"
    所有组都是半开区间，除了最后一组为闭区间。例如，如果设置参数 `bins=[0, 1, 2, 3]`，则第一组为 `[0, 1)`，第二组为 `[1, 2)`，而第三组（最后一组）为 `[2, 3]`。

```python
>>> np.histogram([0, 1, 1, 2, 3], bins=3)  # 等组距的组数为3, 即将区间 [0,3] 等分为3组
(array([1, 2, 2]), array([0., 1., 2., 3.]))

>>> np.histogram([0, 1, 2, 3], bins=[0, 1, 2, 3])  # 组边界为 [0,1,2,3] 的3组
(array([1, 1, 2]), array([0, 1, 2, 3]))            # 2 和 3 落在最后一组 [2,3] 中

>>> np.histogram([-1, 0, 1, 2, 3, 4], bins=[0, 1, 2, 3])
(array([1, 1, 2]), array([0, 1, 2, 3]))            # 落在所有组外的元素不会计入

>>> np.histogram([0, 1, 2, 3], bins=[0, 1, 2, 3], density=True)  # 计算概率密度图
(array([0.25, 0.25, 0.5 ]), array([0, 1, 2, 3]))                 # 面积之和为 1
```

#### mean(), std()

返回数组中所有元素或沿指定轴的平均值/标准差。亦为 `numpy.ndarray` 方法。

```python
>>> a = np.random.randn(3, 3)
>>> a
array([[ 0.94980824,  1.04584228,  0.83793739],
       [-0.34161272, -0.5179232 ,  0.06009127],
       [ 0.67584914,  0.37504747,  0.43473474]])
>>> a.mean()                 # 展开的数组的平均值
0.3910860682553663
>>> a.mean(axis=0)           # 沿轴0的平均值; 列平均值
array([0.42801489, 0.30098885, 0.44425447])
>>> a.mean(axis=1)           # 沿轴1的平均值; 行平均值
array([ 0.9445293 , -0.26648155,  0.49521045])
>>> a.std()
0.5266780199541472
>>> a.std(axis=0)
array([0.55558281, 0.64054879, 0.31762569])
>>> a.std(axis=1)
array([0.08495886, 0.24187973, 0.13003434])
```

#### median(), percentile(), quantile()

返回数组中所有元素或沿指定轴的中位数/指定分位数。

```python
>>> a = np.array([[10, 7, 4], [3, 2, 1]])
>>> a
array([[10,  7,  4],
       [ 3,  2,  1]])
>>> np.median(a)            # 所有元素的中位数
3.5
>>> np.percentile(a, 50)    #         50% 分位数
3.5
>>> np.quantile(a, 0.5)     #         0.5 分位数
3.5
>>> np.percentile(a, 75)    # 6.25 = (7 - 4) / 0.20 * 0.15 + 4
6.25
>>> np.quantile(a, 0.75)
6.25
>>> np.median(a, axis=0)    # 沿轴0的中位数
array([6.5, 4.5, 2.5])
>>> np.median(a, axis=1)    # 沿轴1的中位数
array([7., 2.])
>>> np.median(a, axis=1, keepdims=True)
array([[7.],
       [2.]])
```
