[toc]

NumPy包用于向量和矩阵运算。

> tutorial参见[Quickstart tutorial](https://numpy.org/doc/stable/user/quickstart.html)





# 数组对象

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





# 基本操作

## 运算

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

>>> a+b                  # 逐元素加法
array([[ 6,  8, 10],
       [12, 14, 16]])
>>> a+1                  # broadcasting
array([[1, 2, 3],
       [4, 5, 6]])
>>> a+c                  # broadcasting
array([[1., 2., 3.],
       [4., 5., 6.]])
>>> np.dot(d,d)          # 矩阵乘法(1)
array([[1, 0],           # 矩阵乘法可以扩展到高维张量之间,具体运算规则与矩阵乘法类似
       [2, 1]])
>>> d.dot(d)             # 矩阵乘法(2)
array([[1, 0],
       [2, 1]])
>>> d*d                  # 逐元素乘法
array([[1, 0],
       [1, 1]])
>>> np.transpose(a)      # 转置(1)
array([[0, 3],
       [1, 4],
       [2, 5]])
>>> a.T                  # 转置(2)
array([[0, 3],
       [1, 4],
       [2, 5]])

```



## 复制







# 库函数

## numpy

### append

将数组展开为向量，并在末尾添加给定元素。

```python
>>> a = np.arange(6).reshape(2,3)
>>> np.append(a,10)
array([ 0,  1,  2,  3,  4,  5, 10])
```



### arange

根据给定的初值，末值和步长创建向量。

```python
>>> np.arange(10)
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> np.arange(0,10)
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> np.arange(0,10,0.5)
array([0. , 0.5, 1. , 1.5, 2. , 2.5, 3. , 3.5, 4. , 4.5, 5. , 5.5, 6. ,
       6.5, 7. , 7.5, 8. , 8.5, 9. , 9.5])
```



### array

创建数组。

```python
>>> vector = np.array([1.2, 2.4, 3.5, 4.7, 6.1, 7.2, 8.3, 9.5])
>>> vector
array([1.2, 2.4, 3.5, 4.7, 6.1, 7.2, 8.3, 9.5])

>>> matrix = np.array([[6, 5], [11, 7], [4, 8]])
>>> matrix
array([[ 6,  5],
       [11,  7],
       [ 4,  8]])
```



### concatenate

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





### delete

删除数组指定轴和指定索引上的元素。

```python
>>> arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
>>> arr
array([[ 1,  2,  3,  4],
       [ 5,  6,  7,  8],
       [ 9, 10, 11, 12]])
>>> np.delete(arr, 1, axis=0)
array([[ 1,  2,  3,  4],
       [ 9, 10, 11, 12]])
>>> np.delete(arr, 1, axis=1)
array([[ 1,  3,  4],
       [ 5,  7,  8],
       [ 9, 11, 12]])
```



### expand_dims

增加数组的一个维度。

```python
>>> a = np.random.randn(3,4)
>>> np.expand_dims(a, 0).shape  # 等价于 a.reshape(1, 3, 4)
(1, 3, 4)
>>> np.expand_dims(a, 1).shape
(3, 1, 4)
>>> np.expand_dims(a, 2).shape
(3, 4, 1)
```



### linspace

在给定区间上生成一个给定规模的等差序列。一般用于绘制折线图。

```python
>>> np.linspace(0, 5, 51)
array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. , 1.1, 1.2,
       1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2. , 2.1, 2.2, 2.3, 2.4, 2.5,
       2.6, 2.7, 2.8, 2.9, 3. , 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8,
       3.9, 4. , 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5. ])
```



### max, min

返回数组的最大/最小值。

```python
>>> a = np.random.randn(3,3)
>>> a
array([[ 0.77738867,  0.17869281,  0.44872468],
       [ 1.08848871,  0.43864816, -1.60004663],
       [ 0.60062474,  1.94394622, -0.73857706]])
>>> np.max(a)
1.943946216560545
>>> np.min(a)
-1.600046631389522
```



### ones， zeros

创建全1/0数组。

```python
>>> np.ones(6)
array([1., 1., 1., 1., 1., 1.])
>>> np.zeros((2,6))
array([[0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0.]])
```



### pi

圆周率$$\pi$$。

```python
>>> np.sin(np.pi/2)
1.0
```



### repeat

以给定重复次数和给定轴重复数组得到更大数组。

```python
>>> a = np.arange(12).reshape(3,4)
>>> np.repeat(a, 2, 0)
array([[ 0,  1,  2,  3],
       [ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11],
       [ 8,  9, 10, 11]])
>>> np.repeat(a, 2, 1)
array([[ 0,  0,  1,  1,  2,  2,  3,  3],
       [ 4,  4,  5,  5,  6,  6,  7,  7],
       [ 8,  8,  9,  9, 10, 10, 11, 11]])
```





### reshape

改变数组的形状。

```python
>>> np.arange(6).reshape(2,3)
array([[0, 1, 2],
       [3, 4, 5]])
```



### sin, cos, tan, arcsin, arccos, arctan

对数组的每个元素计算三角函数。



### sinh, cosh, tanh

对数组的每个元素计算双曲函数。



### stack

将数组序列沿指定轴拼接。

```python
>>> arrays = [np.random.randn(3, 4) for _ in range(10)] # a list of array
>>> np.stack(arrays, axis=0).shape
(10, 3, 4)
>>> np.stack(arrays, axis=1).shape
(3, 10, 4)
>>> np.stack(arrays, axis=2).shape
(3, 4, 10)
```





## numpy.random

### choice

随机抽取数组（沿轴0的）一个元素。

```python
>>> b
array([ 5,  0, 11,  4,  1,  2,  3,  9,  7,  6,  8, 10])
>>> np.random.choice(b)
6
```



### randint

根据给定的初值，末值和长度给出整数向量。

```python
>>> np.random.randint(0, 10, 20)
array([1, 1, 9, 2, 3, 1, 1, 4, 4, 5, 6, 4, 3, 7, 2, 1, 9, 7, 5, 5])
```



### randn

根据给定的形状给出随机数数组，其中每个元素服从标准正态分布。

```python
>>> np.random.randn(3,3)
array([[ 0.1283324 , -0.22434515,  1.47125353],
       [ 0.42664038, -2.32914921,  1.09965505],
       [ 0.07222941, -1.72843008,  0.4844523 ]])
```



### random

根据给定的长度给出随机数向量，其中每个元素服从(0,1)区间的均匀分布。

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



### shuffle

随机打乱数组（沿轴0的）各元素的位置。

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





