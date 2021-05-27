[toc]

[NumPy]() 是 Python 中进行科学计算的基本包。

> tutorial参见[Quickstart tutorial](https://numpy.org/doc/stable/user/quickstart.html)



# 教程

> 官方教程参见[NumPy user guide](https://numpy.org/doc/stable/user/index.html)

## 数组对象

```python
>>> a = np.array([[6, 5], [11, 7], [4, 8]])
>>> a
array([[ 6,  5],
       [11,  7],
       [ 4,  8]])
>>> a[0][0]
6
>>> a.ndim         #　数组维数
2
>>> a.shape　　　　　# 数组形状
(3, 2)
```





## 基本操作

### 运算

```python
>>> a = np.arange(6).reshape(2,3)
>>> b = np.arange(6,12).reshape(2,3)
>>> c = np.ones(3)
>>> d = np.array([[1, 0], [1, 1]])
>>> a
array([[0, 1, 2],
       [3, 4, 5]])
>>> b
array([[ 6,  7,  8],
       [ 9, 10, 11]])
>>> c
array([1., 1., 1.])
>>> d
array([[1, 0],
       [1, 1]])

>>> a + b                  # 逐元素加法
array([[ 6,  8, 10],
       [12, 14, 16]])
>>> a + 1                  # broadcasting
array([[1, 2, 3],
       [4, 5, 6]])
>>> a + c                  # broadcasting
array([[1., 2., 3.],
       [4., 5., 6.]])
>>> np.dot(d,d)            # 矩阵乘法(1)
array([[1, 0],             # 矩阵乘法可以扩展到高维张量之间,具体运算规则与矩阵乘法类似
       [2, 1]])
>>> d.dot(d)               # 矩阵乘法(2)
array([[1, 0],
       [2, 1]])
>>> d * d                  # 逐元素乘法
array([[1, 0],
       [1, 1]])
>>> np.transpose(a)        # 转置(1)
array([[0, 3],
       [1, 4],
       [2, 5]])
>>> a.T                    # 转置(2)
array([[0, 3],
       [1, 4],
       [2, 5]])

```



### 复制







# API

> 参见[NumPy Reference](https://numpy.org/doc/stable/reference/index.html)



## numpy

### amax(), amin()

返回数组沿指定轴的最大/最小值。

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



### append()

将数组展开为向量，并在末尾添加给定元素。

```python
>>> a = np.arange(6).reshape(2,3)
>>> np.append(a, 10)
array([ 0,  1,  2,  3,  4,  5, 10])
```



### arange()

根据给定的初值，末值和步长创建向量。与 python 的 `range()` 用法相同。

```python
>>> np.arange(10)
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> np.arange(0,10)
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> np.arange(0,10,0.5)
array([0. , 0.5, 1. , 1.5, 2. , 2.5, 3. , 3.5, 4. , 4.5, 5. , 5.5, 6. ,
       6.5, 7. , 7.5, 8. , 8.5, 9. , 9.5])
```



### argsort()

返回向量的排序索引。

```python
>>> a
array([6, 5, 5, 8, 7])
>>> np.argsort(a)
array([1, 2, 0, 4, 3])  # 从小到大依次为索引为1, 2, 0, 4, 3的元素
```



### array()

创建数组。

```python
>>> vector = np.array([1.2, 2.4, 3.5, 4.7, 6.1, 7.2, 8.3, 9.5])
>>> vector
array([1.2, 2.4, 3.5, 4.7, 6.1, 7.2, 8.3, 9.5])

>>> matrix = np.array([[6, 5], [11, 7], [4, 8]])     # 二维数组
>>> matrix
array([[ 6,  5],
       [11,  7],
       [ 4,  8]])
```



### array_equal()





### concatenate()

沿指定轴拼接数组。

```python
>>> a1 = np.array([[1, 2], [3, 4]])
>>> a2 = np.array([[5, 6]])
>>> np.concatenate((a1, a2), axis=0)
array([[1, 2],
       [3, 4],
       [5, 6]])
>>> np.concatenate((a1, a2.T), axis=1)
array([[1, 2, 5],
       [3, 4, 6]])
>>> np.concatenate((a1, a2), axis=None)
array([1, 2, 3, 4, 5, 6])
```



### corrcoef()

计算两个向量的相关系数。

```python
>>> a = np.array([1.,2,3])
>>> b = np.array([3.,2,1])
>>> np.corrcoef(a,b)
array([[ 1., -1.],
       [-1.,  1.]])

>>> c = np.array([[10.,20,30],[30,20,10],[20,20,20]])
>>> np.corrcoef(c)
array([[ 1., -1., nan],
       [-1.,  1., nan],
       [nan, nan, nan]])
```



### cov()

计算两个向量的协方差。

```python
>>> a = np.array([1.,2,3])
>>> b = np.array([3.,2,1])
>>> np.cov(a,b)
array([[ 1., -1.],
       [-1.,  1.]])

>>> c = np.array([[10.,20,30],[30,20,10],[20,20,20]])
>>> np.cov(c)
array([[ 100., -100.,    0.],
       [-100.,  100.,    0.],
       [   0.,    0.,    0.]])
```



### datetime, timedeltas

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
#      ...
       '2005-02-25T23:40:00', '2005-02-25T23:50:00'],
      dtype='datetime64[s]')

```



### delete()

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



### diag()

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



### expand_dims()

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



### insert()

在数组的指定轴的指定索引位置插入元素。

```python
>>> a = np.zeros((4,4))
>>> np.insert(a, 0, 1, axis=1)  # 在所有axis=1, index=0位置插入1
array([[1., 0., 0., 0., 0.],
       [1., 0., 0., 0., 0.],
       [1., 0., 0., 0., 0.],
       [1., 0., 0., 0., 0.]])
>>> np.insert(a, 0, 1, axis=0)  # 在所有axis=0, index=0位置插入1
array([[1., 1., 1., 1.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.]])
```



### linspace()

在给定区间上生成一个给定规模的等差序列。一般用于绘制折线图。

```python
>>> np.linspace(0, 5, 51)
array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. , 1.1, 1.2,
       1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2. , 2.1, 2.2, 2.3, 2.4, 2.5,
       2.6, 2.7, 2.8, 2.9, 3. , 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8,
       3.9, 4. , 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5. ])
```



### max(), min()

返回数组中所有元素的最大/最小值。

```python
>>> a = np.arange(4).reshape((2,2))
>>> a
array([[0, 1],
       [2, 3]])
>>> np.max(a)
3
>>> np.min(a)
0
```



### mean(), std()

返回数组中所有元素或沿指定轴的平均值/标准差。

```python
>>> a = np.random.randn(3,3)
>>> a
array([[ 0.94980824,  1.04584228,  0.83793739],
       [-0.34161272, -0.5179232 ,  0.06009127],
       [ 0.67584914,  0.37504747,  0.43473474]])
>>> np.mean(a)                 # 展开的数组的平均值
0.3910860682553663
>>> np.mean(a, axis=0)         # 沿轴0的平均值;列平均值
array([0.42801489, 0.30098885, 0.44425447])
>>> np.mean(a, axis=1)         # 沿轴1的平均值;行平均值
array([ 0.9445293 , -0.26648155,  0.49521045])
>>> np.std(a)
0.5266780199541472
>>> np.std(a, axis=0)
array([0.55558281, 0.64054879, 0.31762569])
>>> np.std(a, axis=1)
array([0.08495886, 0.24187973, 0.13003434])
```



### ones(), zeros()

创建全 1/0 数组。

```python
>>> np.ones(6)
array([1., 1., 1., 1., 1., 1.])
>>> np.zeros((2,6))
array([[0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0.]])
>>> np.zeros(6, dtype=int)
array([0, 0, 0, 0, 0, 0])
```



### pi

圆周率 $$\pi$$。

```python
>>> np.sin(np.pi/2)
1.0
```



### repeat()

以重复输入数组元素的方式构建数组。

```python
>>> a = np.arange(12).reshape(3,4)
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



### reshape()

改变数组的形状。

```python
>>> np.arange(6).reshape(2,3)
array([[0, 1, 2],
       [3, 4, 5]])
```



### sin(), cos(), tan(), arcsin(), arccos(), arctan(), sinh(), cosh(), tanh()

对数组的每个元素计算三角函数和双曲函数。



### stack()

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



### swapaxes()

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



### tranpose()

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



## numpy.random

### choice()

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
>>> np.random.choice(10, 5, replace=False)   # 10视作向量 np.arange(10)
array([8, 1, 5, 9, 0])
```



### randint()

根据给定的初值，末值和长度给出整数向量。

```python
>>> np.random.randint(0, 10, 20)
array([1, 1, 9, 2, 3, 1, 1, 4, 4, 5, 6, 4, 3, 7, 2, 1, 9, 7, 5, 5])
```



### randn()

根据给定的形状给出随机数数组，其中每个元素服从标准正态分布。

```python
>>> np.random.randn(3,3)
array([[ 0.1283324 , -0.22434515,  1.47125353],
       [ 0.42664038, -2.32914921,  1.09965505],
       [ 0.07222941, -1.72843008,  0.4844523 ]])
```



### permutation()

随机返回向量中所有元素的全排列中的一种。

```python
>>> a
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> np.random.permutation(a)
array([3, 2, 8, 7, 0, 9, 6, 4, 1, 5])
>>> np.random.permutation(10)           # 10视作向量 np.arange(10)
array([3, 2, 8, 7, 0, 9, 6, 4, 1, 5])
```



### random()/rand()

根据给定的长度给出随机数向量，其中每个元素服从（0，1）区间的均匀分布。

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



### seed()

指定随机数种子，一次有效。

```python
>>> np.random.seed(0)
>>> np.random.random(10)
array([0.5488135 , 0.71518937, 0.60276338, 0.54488318, 0.4236548 ,
       0.64589411, 0.43758721, 0.891773  , 0.96366276, 0.38344152])
```



### shuffle()

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



## numpy.linalg

### eig()

返回数组的一个特征分解 $$A=Q\Lambda Q^{-1}$$。

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



### svd()

返回数组的一个奇异值分解 $$A=U\Sigma V^*$$。

```python
>>> from numpy import linalg as LA
>>> a
array([[1., 0., 0., 0., 2.],
       [0., 0., 3., 0., 0.],
       [0., 0., 0., 0., 0.],
       [0., 4., 0., 0., 0.]])
>>> LA.svd(a, full_matrices=True) # 返回U为mxm矩阵,V为nxn矩阵
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

