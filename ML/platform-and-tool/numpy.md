NumPy包用于向量和矩阵运算。

> tutorial参见[Quickstart tutorial](https://numpy.org/doc/stable/user/quickstart.html)

## 创建矩阵

```python
import numpy as np

vector = np.array([1.2, 2.4, 3.5, 4.7, 6.1, 7.2, 8.3, 9.5])
print(vector)

matrix = np.array([[6, 5], [11, 7], [4, 8]])
print(matrix)
#[[ 6  5]
# [11  7]
# [ 4  8]]
print(matrix.ndim)
# 2

sequence_matrix = np.arange(5, 11).reshape(2, 3)
print(sequence_matrix)
#[[ 5  6  7], 
# [ 8  9 10]]

one = np.ones(6).reshape(2,3)
zero = np.zeros((300,20))
```



## 基本运算

```python
a = np.arange(6).reshape(2,3)
b = np.arange(6,12).reshape(2,3)
c = np.ones(3)
d = np.array([[1, 0], [1, 1]])

r = a+b               # element-wise addition
r = np.maximum(z, 0.) # element-wise ReLU

r = a+1               # broadcasting
r = a+2*c             # broadcasting can be applied when one tensor has shape 
                      # (n1, n2, ...) and the other has (m1, m2, ..., n1, n2, ...)
    
r = np.dot(d,d)       # matrix multiplication
r = d*d               # element-wise multiplication
                      # 矩阵乘法可以扩展到高维张量之间,具体运算规则与矩阵乘法类似

r = np.transpose(a)   # transpose

```



## random

```python
random_integer = np.random.randint(low=50, high=101, size=(6))
print(random_integer)
#[59 77 94 60 97 92]

random_float01 = np.random.random([6])
print(random_float) 
#[0.19546204 0.18011937 0.41153588 0.45157418 0.16954296 0.63709503]

# NumPy uses broadcasting to virtually expand the smaller operand to dimensions compatible for linear algebra
random_float23 = random_float + 2.0
print(random_float1)
#[2.19546204 2.18011937 2.41153588 2.45157418 2.16954296 2.63709503]

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



