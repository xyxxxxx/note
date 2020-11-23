[toc]

NumPy包用于向量和矩阵运算。

> tutorial参见[Quickstart tutorial](https://numpy.org/doc/stable/user/quickstart.html)





# 数组性质

```python
>>> a = np.array([[6, 5], [11, 7], [4, 8]])
>>> a
array([[ 6,  5],
       [11,  7],
       [ 4,  8]])
>>> a.ndim         #　数组维数
2
>>> a.shape　　　　　# 数组形状
(3, 2)
```





# 基本运算

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





# 库函数

## numpy

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



### ones， zeros

创建全1/0数组。

```python
>>> np.ones(6)
array([1., 1., 1., 1., 1., 1.])
>>> np.zeros((2,6))
array([[0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0.]])
```



### reshape

改变数组的形状。

```python
>>> np.arange(10).reshape(5,2)
array([[0, 1],
       [2, 3],
       [4, 5],
       [6, 7],
       [8, 9]])
```



## numpy.random



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
```







```python


# randn() * std + mean represents any normal distribution
random_floatn = pd.DataFrame(np.random.randn(4,4)* 4 + 3)
```





## 拼接矩阵

```python
a = np.arange(6).reshape(2,3)
b = np.arange(6,12).reshape(2,3)

print(np.append(a,10)) # append returns vector
# [ 0  1  2  3  4  5 10]
print(np.append(a,b))
# [ 0  1  2  3  4  5  6  7  8  9 10 11]

print(np.concatenate((a,b)))
# [[ 0  1  2]
# [ 3  4  5]
# [ 6  7  8]
# [ 9 10 11]]
```



