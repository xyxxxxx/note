[toc]



# 序列类型——list, tuple, range

## 通用序列操作

大多数序列类型，包括可变序列和不可变序列都支持下表中的操作，其中 *s* 和 *t* 是相同类型的序列，*n*，*i*，*j* 和 *k* 是整数而 *x* 是任意对象：

| 运算                   | 结果                                                         |
| :--------------------- | :----------------------------------------------------------- |
| `x in s`               | 如果 *s* 中的某项等于 *x* 则结果为 `True`，否则为 `False`    |
| `x not in s`           | 如果 *s* 中的某项等于 *x* 则结果为 `False`，否则为 `True`    |
| `s + t`                | *s* 与 *t* 相拼接                                            |
| `s * n` 或 `n * s`     | 相当于 *s* 与自身进行 *n* 次拼接                             |
| `s[i]`                 | *s* 的第 *i* 项，起始为 0                                    |
| `s[i:j]`               | *s* 从 *i* 到 *j* 的切片                                     |
| `s[i:j:k]`             | *s* 从 *i* 到 *j* 步长为 *k* 的切片                          |
| `len(s)`               | *s* 的长度                                                   |
| `min(s)`               | *s* 的最小项                                                 |
| `max(s)`               | *s* 的最大项                                                 |
| `s.index(x[, i[, j]])` | *x* 在 *s* 中首次出现项的索引号（索引号在 *i* 或其后且在 *j* 之前） |
| `s.count(x)`           | *x* 在 *s* 中出现的总次数                                    |
| `s > t`                | 比较序列 *s* 和 *t* 元素的字典顺序                           |



### 方法

#### count()

返回指定元素在序列中出现的次数。

```python
>>> a = [1, 2, 7, 4, 3]
>>> a.count(2)
1
>>> a.count(5)
0
```



#### index()

返回序列中的第一个指定元素的从零开始的索引，可选的第二和第三个参数用于指定特定的搜索序列。

```python
>>> a = [1, 2, 7, 4, 3, 2]
>>> a.index(2)
1
>>> a.index(2, 2)
5
>>> a.index(5)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: 5 is not in list
```



## 列表[]

列表（list）是可变序列，通常用于存放同类项目的集合。

```python
>>> s = [1,2,3,4]      # 创建列表
>>> s
[1, 2, 3, 4]
>>> s = []             # 创建空列表
>>> s
[]
>>> s = [1,2] + [3,4] * 2  # 拼接和重复列表
>>> s
[1, 2, 3, 4, 3, 4]
>>> s = [1,2,[3,4]]    # 嵌套列表
>>> s
[1, 2, [3, 4]]

>>> s = [1,2,3,4]
>>> len(s)             # 列表长度
4
>>> max(s)             # 最大元素
4
>>> min(s)             # 最小元素
1
>>> 1 in s             # 判断元素存在
True
>>> 2 in s
False

>>> s[0]               # 按索引访问/循秩访问
1
>>> s[1]
2
>>> s[-1]
4
>>> s[-2]
3

>>> s.append(5)        # 增删元素
>>> s
[1, 2, 3, 4, 5]
>>> s.insert(1,1.5)
>>> s
[1, 1.5, 2, 3, 4, 5]
>>> s.pop()
5
>>> s.pop(1)
1.5
>>> s
[1, 2, 3, 4]

>>> s[1] = 2.1         # 修改元素
>>> s
[1, 2.1, 3, 4]
>>> s[1:3] = [1.9,2.9]
>>> s
[1, 1.9, 2.9, 4]
>>> s[1:3] = []
>>> s
[1, 4]

>>> s > [1, 2]         # 比较列表
True
```

构造器 `list(iterable)` 将构造一个列表，其中的项与 `iterable` 中的项具有相同的的值与顺序。`iterable` 可以是任何可迭代对象。例如：

```python
>>> list(range(10))
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
>>> list('abc')
['a', 'b', 'c']
```

```python
# 列表切片
>>> L = list(range(100))
>>> L[:10]
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
>>> L[-10:]
[90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
>>> L[10:20]
[10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
>>> L[:10:2]
[0, 2, 4, 6, 8]
>>> L[::5]
[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
>>> L[:]
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ..., 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
```



### 引用列表

当

多个列表会引用同一个列表对象，例如：



地址图



另一个会令 Python 初学者感到困惑的例子：

```python
>>> a = 0
>>> b = [a]
>>> b
[0]
>>> a = 1
>>> b
[0]

>>> a = [0]
>>> b = [a]
>>> b
[[0]]
>>> a.append(1)
>>> b
[[0, 1]]
```

当 `a` 是一个数值或者字符串类型时，将其添加到列表 `b` 仅会添加其值；但当 `a` 是一个列表时，将其添加到列表 `b` 会添加该列表对象，之后该对象的任何改变都会反映在列表 `b` 中。再例如：

```python
>>> lists = [[1]] * 3
>>> lists
[[1], [1], [1]]
>>> lists[0].append(2)      # 修改第一个列表元素
>>> lists
[[1, 2], [1, 2], [1, 2]]    # 所有列表元素都被修改
```

这里重复了 3 次的列表 `[1]` 实际上引用的是同一列表，因此修改任意一个列表，所有的列表都会被同样地修改。要想创建多维列表，推荐使用列表推导式。再例如：

```python
>>> a = [1, 2, 3]
>>> a.append(a)
>>> a
[1, 2, 3, [...]]
>>> a[-1]
[1, 2, 3, [...]]

>>> a = [1, 2, 3]
>>> a.append(a[:])
>>> a
[1, 2, 3, [1, 2, 3]]
```

列表添加自身将造成自身的循环引用，即成为一个递归对象，添加自身的副本不会产生这个问题。



有时我们想要拷贝列表，创建独立的副本，而不是引用同一列表。下面演示了三种拷贝列表的方法：

```python
>>> import copy
>>> s = [1, 2, [3, 4]]
>>> t = s                 # 引用同一列表
>>> u = list(s)           # 返回s的浅拷贝
                          # 相当于创建新列表,各元素分别为1,2和列表[3,4](引用同一列表)
>>> v = copy.deepcopy(s)  # 返回s的深拷贝
>>> t[0] = 0
>>> t[2].append(5)
>>> s
[0, 2, [3, 4, 5]]
>>> u
[1, 2, [3, 4, 5]]
>>> v
[1, 2, [3, 4]]
```



### 方法

#### append()

将指定元素添加到列表的末尾。

```python
>>> a = [1, 2, 7, 4, 3]
>>> a.append(5)
>>> a
[1, 2, 7, 4, 3, 5]
```



#### clear()

移除列表中的所有元素。

```python
>>> a = [1, 2, 7, 4, 3]
>>> a.clear()
>>> a
[]
```



#### copy()

返回列表的一个浅拷贝。

```python
>>> a = [1, 2, 7, [4, 3]]
>>> b = a.copy()
>>> a[0] = 0
>>> a[3].append(0)
>>> a
[0, 2, 7, [4, 3, 0]]
>>> b
[1, 2, 7, [4, 3, 0]]
```



#### extend()

使用可迭代对象中的所有元素来扩展列表。

```python
>>> a = [1, 2, 7, 4, 3]
>>> a.extend(range(5))
>>> a
[1, 2, 7, 4, 3, 0, 1, 2, 3, 4]
```



#### insert()

在列表的指定位置插入一个元素。

```python
>>> a = [1, 2, 7, 4, 3]
>>> a.insert(2, 9)
>>> a
[1, 2, 9, 7, 4, 3]
```



#### pop()

删除列表指定位置的元素并返回它，默认为最后一个元素。

```python
>>> a = [1, 2, 7, 4, 3]
>>> a.pop()
3
>>> a.pop(1)
2
>>> a
[1, 7, 4]
```



#### remove()

移除列表中的第一个指定元素。

```python
>>> a = [1, 2, 7, 4, 3]
>>> a.remove(1)
>>> a
[2, 7, 4, 3]
>>> a.remove(1)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: list.remove(x): x not in list
```



#### reverse()

翻转列表中的元素。

```python
>>> a = [1, 2, 7, 4, 3]
>>> a.reverse()
>>> a
[3, 4, 7, 2, 1]
```



#### sort()

对列表中的元素进行排序。可选的接受一个参数的函数将作为排序标准，各元素按照传入该函数的函数值进行排序。

```python
>>> a = [1, -2, 7, -4, 3]
>>> a.sort()            # 原位排序,不返回
>>> a
[-4, -2, 1, 3, 7]
>>> sorted(a)           # 返回排序结果,不改变原列表
[-4, -2, 1, 3, 7]
>>> 
```



### 迭代

```python
>>> for v in ['tic', 'tac', 'toe']:                   # 迭代所有元素
...     print(v)
... 
tic
tac
toe

>>> for i, v in enumerate(['tic', 'tac', 'toe']):     # 迭代所有元素及索引
...     print(i, v)
... 
0 tic
1 tac
2 toe

>>> for i, v in enumerate(['tic', 'tac', 'toe'], 1):  # 指定索引的初值
...     print(i, v)
... 
1 tic
2 tac
3 toe
```

在迭代过程中添加、删除元素可能会引发 `RuntimeError` 或者重复、遗漏一些元素，正确的方法是创建新列表：

```python
>>> d = [1, 0, 3, 0]
>>> d = [x for x in d if x > 0]        # 创建新列表
>>> d
[1, 3]
```

> 永远不要修改正在迭代的对象。



### 实现栈

列表使用 `append()` 和 `pop()` 方法。



### 实现队列

列表也可以实现队列，但在列表的开头插入或弹出元素很慢。建议使用 `collections.deque`，它被设计成可以快速地从两端添加或弹出元素。



### 列表推导式

```python
# 一般形式: [<expression> for <variable_name> in <sequence> if <condition>]

>>> list(range(1, 11))
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
>>> [x * x for x in range(1, 11)]                              # for循环
[1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
>>> [x * x for x in range(1, 11) if x % 2 == 0]                # for循环 + 条件
[4, 16, 36, 64, 100]
>>> [(x, y) for x in [1, 2, 3] for y in [3, 1, 4] if x != y]   # 双重for循环 + 条件
[(1, 3), (1, 4), (2, 3), (2, 1), (2, 4), (3, 1), (3, 4)]
>>> [x * x if x % 2 == 0 else -x * x for x in range(1, 11)]    # 三元表达式 + for循环
[-1, 4, -9, 16, -25, 36, -49, 64, -81, 100]

>>> matrix = [
...     [1, 2, 3, 4],
...     [5, 6, 7, 8],
...     [9, 10, 11, 12],
... ]
>>> [[row[i] for row in matrix] for i in range(4)]
[[1, 5, 9], [2, 6, 10], [3, 7, 11], [4, 8, 12]]                 # 嵌套的列表推导式
```





## 元组()

元组（tuple）是不可变序列，通常用于储存异构数据的多项集。

```python
>>> t = 12345, 54321, 'hello!'   # 创建元组
>>> t
(12345, 54321, 'hello!')
>>> t = tuple([12345, 54321, 'hello!'])   # 将可迭代对象转换为元组
>>> t
(12345, 54321, 'hello!')
>>> u = t, (1, 2, 3, 4, 5)       # 嵌套元组
>>> u
((12345, 54321, 'hello!'), (1, 2, 3, 4, 5))
>>> t[0] = 88888                 # 不允许给元组的元素赋值
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: 'tuple' object does not support item assignment
>>> x, y, z = t                  # 元组解包

>>> empty = ()                   # 创建空元组
>>> empty
()
>>> empty = tuple()
>>> empty
()
>>> singleton = 'hello',         # 创建单元组 
>>> singleton
('hello',)     
```

比较列表和元组：

* 元组是不可变对象，其序列通常包含不同种类的元素，并且通过解包或者索引来访问
* 列表是可变对象，列表中的元素一般是同种类型，并且通过迭代访问

元组之间可以进行比较，采用逐元素比较法。



### 迭代



### 可迭代拆包



## range

range 类型表示不可变的数字序列，通常用于在 for 循环中指定循环的次数。

```python
range(5)         # [0,1,2,3,4]
range(5,10)      # [5,6,7,8,9]
range(0,10,2)    # [0,2,4,6,8]
range(0,10,3)    # [0,3,6,9]
range(0,-10,-3)  # [0,-3,-6,-9]
```

```python
>>> for i in range(5):
...   print(i)
...
0
1
2
3
4
>>> list(range(5))
[0, 1, 2, 3, 4]
```

`range()` 函数返回的对象尽管表现得像一个列表，但实际上是一个迭代器。



# 字典dict-{:}

dict 查找和插入的速度快，但占用内存大，即用空间换时间。

dict 的 key 必须是**不可变对象**（字符串，整数，etc）。

```python
# 创建字典
>>> d = {'Michael': 95, 'Bob': 75, 'Tracy': 85}
# or
>>> d = dict(Michael=95, Bob=75, Tracy=85)
# or
>>> d = dict([('Michael', 95), ('Bob', 75), ('Tracy', 85)])
# or
>>> names = ['Michael', 'Bob', 'Tracy']
>>> scores = [95, 75, 85]
>>> d = dict(zip(columns, values))

# 访问/增删/修改条目
>>> d['Michael']       # 循键访问
95
>>> d['Alice']         # 访问不存在的键将引发KeyError
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
KeyError: 'Alice'
>>> d['Adam'] = 67     # 添加新键值对
>>> d.pop('Bob')       # 弹出键和相应的值
75
>>> d.popitem()        # 弹出一个键值对,在3.7版本之后会按照先进后出的顺序弹出
('Adam', 67)
>>> d['Michael'] = 90  # 修改键的值
>>> d['Michael']
90

# 常用操作
>>> len(d)             # 字典规模
4
>>> 'Thomas' in d      # 判定键是否存在
False
>>> list(d)            # 返回所有键组成的列表
['Michael', 'Bob', 'Tracy', 'Adam']
>>> sorted(d)          # 排序所有键
['Adam', 'Bob', 'Michael', 'Tracy']

# 字典视图对象
>>> d.keys()                           # 所有键组成的视图
dict_keys(['Michael', 'Bob', 'Tracy'])
>>> d.items()                          # 所有键值对组成的视图
dict_items([('Michael', 95), ('Tracy', 85), ('Adam', 67)])
>>> list(zip(d.values(), d.keys()))    # (v,k)列表
[(95, 'Michael'), (85, 'Tracy'), (67, 'Adam')]
```



## 方法

```python
>>> d = {'a':1, 'b':2, 'c':3}

# get() 判定键是否存在,若存在则返回相应的值,不存在则返回指定值
>>> d.get('b', -1)
2
>>> d.get('d', 'not found')
'not found'

# update() 使用其它字典来更新字典
>>> d.update({'c':4, 'd':8})
>>> d
{'a': 1, 'b': 2, 'c': 4, 'd': 8}
```



## 迭代

```python
>>> d = {'a': 1, 'b': 2, 'c': 3}
>>> for key in d:    # 迭代key
...     print(key)
... 
a
b
c
>>> for value in d.values():
...     print(value)  # 迭代value
... 
1
2
3
>>> for k, v in d.items():
...     print(k, v)   # 迭代key,value
... 
a 1
b 2
c 3
```

在迭代过程中添加、删除条目或修改键（修改值则不会）可能会引发 `RuntimeError` 或者重复、遗漏一些条目，正确的方法是迭代字典的副本，或者创建新字典用于修改：

```python
>>> d = {'a': 1, 'b': 0, 'c': 3, 'd': 0}
>>> for k, v in d.copy().items():                   # 迭代原字典的浅拷贝
...     if v == 0:
...         del d[k]
... 
>>> d
{'a': 1, 'c': 3}
```

```python
>>> d = {'a': 1, 'b': 0, 'c': 3, 'd': 0}
>>> d = {k: v for k, v in d.items() if v!=0}        # 创建新字典
>>> d
{'a': 1, 'c': 3}
```



## 字典推导式

```python
# 一般形式: {<expression>: <expression> for <variable_name> in <sequence> if <condition>}

>>> {x: x**2 for x in (2, 4, 6)}
{2: 4, 4: 16, 6: 36}
>>> 
```

构造字典的子集，最简单的方式就是使用字典推导式：

```python
prices = {
    'ACME': 45.23,
    'AAPL': 612.78,
    'IBM': 205.55,
    'HPQ': 37.20,
    'FB': 10.75
}
# Make a dictionary of all prices over 200
p1 = {key: value for key, value in prices.items() if value > 200}
# Make a dictionary of tech stocks
tech_names = {'AAPL', 'IBM', 'HPQ', 'MSFT'}
p2 = {key: value for key, value in prices.items() if key in tech_names}
```





# 集合set-{}

集合中元素无序，且重复元素只计算 1 个。集合的基本用法包括成员检测和消除重复元素。

```python
>>> s = {1, 2, 3}
# or
>>> s = set([1,2,3])	# 用list创建set

>>> s.add(4)		    # 添加元素
>>> s.remove(4)		    # 删除元素

>>> s1 = set([1,2,3])
>>> s2 = set([1,2,4])
>>> s1 & s2			    # 交集
{1, 2}
>>> s1 | s2			    # 并集
{1, 2, 3, 4}
>>> s1 - s2             # 差集
{3}
>>> s1 ^ s2             # 对称差集
{3, 4}
```





# 高级特性

## 切片

```python


# tuple也可以使用切片操作，只是返回的仍为tuple

# 字符串切片
>>> word = 'Python'
>>> word[:2]
'Py'
>>> word[4:]
'on'
>>> word[4:10]  # 切片中的越界索引会被自动处理
'on'
>>> word[::2]
'Pto'
```



## map()

```python
r = map(str, [1, 2, 3, 4, 5, 6, 7, 8, 9])	# map()将一元函数应用在序列上，iterator
list(r)	# list()返回iterator所有元素
```



## filter()

```python
def is_odd(n):
    return n % 2 == 1

list(filter(is_odd, [1, 2, 4, 5, 6, 9, 10, 15]))	#filter()将判别函数作用在序列上，删去返回False的元素,iterator
```



## sorted()

`sorted()` 函数传入原序列，返回一个新的排好序的序列。

```python
sorted([36, 5, -12, 9, -21])	            # [-21, -12, 5, 9, 36]
sorted([36, 5, -12, 9, -21], reverse=True)	# 逆序
sorted([36, 5, -12, 9, -21], key=abs)    	# 按abs()进行排序

sorted(['bob', 'about', 'Zoo', 'Credit'])
#['Credit', 'Zoo', 'about', 'bob'] 默认按ASCII排序，大写字母在小写字母之前
sorted(['bob', 'about', 'Zoo', 'Credit'], key=str.lower)	#全转换为小写

#二维数组排序
L = [('Bob', 75), ('Adam', 92), ('Bart', 66), ('Lisa', 88)]
def by_name(t):
    return t[0]
def by_score(t):
    return t[1]
L1 = sorted(L, key=by_name)
L2 = sorted(L, key=by_score, reverse=True)
```





# `collections` 容器数据类型

`collections` 模块实现了一些具有特定功能的容器，以提供 Python 标准内建容器 `dict`，`list`，`set` 和 `tuple` 的替代选择。



## ChainMap

`ChainMap` 将多个字典或者其他映射组合在一起，创建一个单独的可更新的视图。如果没有参数传入，就默认提供一个空字典，这样新的 `ChainMap` 至少有一个映射。

底层映射被存储在一个列表中。这个列表是公开的，可以通过 `maps` 属性存取和更新。

搜索会查询底层映射，直到一个键被找到；但是写、更新和删除只操作第一个映射。

一个 `ChainMap` 通过引用合并底层映射。所以如果一个底层映射更新了，这些更改会反映到 `ChainMap`。

```python
>>> from collections import ChainMap
>>> d1 = {'a': 1, 'b': 2, 'c':3}
>>> d2 = {'c': 4, 'd': 8}
>>> cm = ChainMap(d1, d2)
>>> cm['a']        # d1中找到该键
1
>>> cm['c']        # d1中找到该键
3
>>> cm['d']        # d2中找到该键
8
>>> cm['c'] = 5    # 写/更新仅操作d1
>>> d1
{'a': 1, 'b': 2, 'c': 5}            # d1的键值对被修改
>>> d2
{'c': 4, 'd': 8}
>>> cm['d'] = 10   # 写/更新仅操作d1
>>> d1
{'a': 1, 'b': 2, 'c': 5, 'd': 10}   # d1增加了键值对
>>> d2
{'c': 4, 'd': 8}
>>> cm.maps        # 底层映射列表
[{'a': 1, 'b': 2, 'c': 5, 'd': 10}, {'c': 4, 'd': 8}]
```

```python
# 应用: 让用户指定的命令行参数优先于环境变量,再优先于默认值
import argparse
import os

defaults = {'color': 'red', 'user': 'guest'}

parser = argparse.ArgumentParser()
parser.add_argument('-u', '--user')
parser.add_argument('-c', '--color')
namespace = parser.parse_args()
command_line_args = {k: v for k, v in vars(namespace).items() if v is not None}

combined = ChainMap(command_line_args, os.environ, defaults)
print(combined['color'])
print(combined['user'])
```



## Counter

`Counter` 是 `dict` 的子类，用于计数可哈希对象。元素存储为键，它们的计数存储为值。计数可以是任何整数值，包括 0 和负数。

其余方法同 `dict`。

```python
>>> from collections import Counter
>>> cnt = Counter()
>>> for word in ['a', 'b', 'c', 'b', 'c', 'c']:
...      cnt[word] += 1
... 
>>> cnt
Counter({'c': 3, 'b': 2, 'a': 1})
>>> cnt['b']
2
>>> cnt['d']               # 不存在的元素计数为0
0
>>> cnt.update(['a', 'b', 'c'])                        # 用可迭代对象增加计数
>>> cnt.update({'a': -1, 'b': -1, 'c': -1})            # 用映射对象增加计数
>>> cnt1 = Counter({'c': 1, 'd': 8})
>>> cnt + cnt1             # 计数值相加
Counter({'d': 8, 'c': 4, 'b': 2, 'a': 1})
>>> cnt - cnt1             # 计数值相减
Counter({'b': 2, 'c': 2, 'a': 1})   # 被减的`Counter`中不存在的元素不会被减去
>>> cnt & cnt1             # 计数值求交集(取较小值)
Counter({'c': 1})
>>> cnt | cnt1             # 计数值求并集(取较大值)
Counter({'d': 8, 'c': 3, 'b': 2, 'a': 1})
>>> cnt = cnt + cnt1
>>> cnt.most_common(2)     # 出现次数最高的2个元素,在需要排序时使用
[('d', 8), ('c', 4)]
>>> list(cnt.elements())   # 返回一个迭代对象,每个元素会重复计数值的次数(负数和0计数的元素不会出现)
['a', 'b', 'b', 'c', 'c', 'c', 'c', 'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd']
>>> sum(cnt.values())      # 对所有计数值求和
15
```

```python
>>> from collections import Counter
>>> cnt = Counter(a=2, b=-4)
>>> -cnt                   # 清除所有非负的计数
Counter({'b': 4})
>>> +cnt                   # 清除所有非正的计数
Counter({'a': 2})
```



## defaultdict

`defaultdict` 是 `dict` 的子类，其实例包含一个名为 `default_factory` 的属性，由构造时的第一个参数传入。当 `default_factory` 不为 `None` 时，它会被不带参数地调用来为新增加的键提供一个默认值。

其余方法同 `dict`。

```python
>>> from collections import defaultdict
>>> dd = defaultdict(list)
>>> dd['a'].append(1)           # 键'a'本不存在于字典里,但在第一次调用时初始化为默认值`list()`
>>> dd
defaultdict(<class 'list'>, {'a': [1]})
>>> dd['b']
[]
>>> dd
defaultdict(<class 'list'>, {'a': [1], 'b': []})
```

```python
# 应用: 为字典中不存在的键提供默认值
>>> def constant_factory(value):
...     return lambda: value
... 
>>> d = defaultdict(constant_factory('<missing>'))
>>> d.update(name='John', action='ran')
>>> '%(name)s %(action)s to %(object)s' % d      # 键'object'不存在,因此返回默认值
'John ran to <missing>'
```



## deque

返回双向队列对象，从左到右初始化，从可迭代对象创建（如果可迭代对象没有指定，则新队列为空）。

`deque` 支持线程安全、内存高效的添加（append）和弹出（pop），从两端都可以，两个方向的大概开销都是 $$O(1)$$ 复杂度。

虽然列表对象也支持类似操作，但是 `deque` 优化了定长操作和 `pop(0)` 和 `insert(0,v)` 的开销。列表操作引起 $$O(n)$$ 复杂度的内存移动，因为改变了底层数据表达的大小和位置。

如果 `maxlen` 没有指定或者是 `None`，`deque` 可以增长到任意长度；否则 `deque` 就限定到指定最大长度，一旦限定长度的 `deque` 被占满，当新项加入时，同样数量的项就从另一端弹出。

deque 支持以下方法：

### append(), appendleft()

添加元素到右端/左端。

```python
>>> from collections import deque
>>> dq = deque('lmn')
>>> dq
deque(['l', 'm', 'n'])
>>> dq.append('o')          # 添加到右端
>>> dq.appendleft('k')      # 添加到左端
>>> dq
deque(['k', 'l', 'm', 'n', 'o'])
```



### clear()

移除所有元素，使其长度为 0。

```python
>>> dq
deque(['l', 'm', 'n'])
>>> dq.clear()
>>> dq
deque([])
```



### copy()

创建一份浅拷贝。

```python
>>> dq = deque(['l', 'm', ['n']])
>>> dq1 = dq.copy()
>>> dq[2].append('o')
>>> dq1
deque(['l', 'm', ['n', 'o']])
```



### count()

计算 `deque` 中等于参数的元素个数。

```python
>>> dq
deque(['l', 'm', 'n', 'l', 'm', 'l'])
>>> dq.count('l')
3
```



### extend(), extendleft()

将可迭代对象的元素依次添加到右端/左端。

```python
>>> dq
deque(['l', 'm', 'n'])
>>> dq.extend('opq')        # 将可迭代对象的元素依次添加到右端
>>> dq.extendleft('ijk')    # 将可迭代对象的元素依次添加到左端,注意添加的元素的顺序将颠倒
>>> dq
deque(['k', 'j', 'i', 'l', 'm', 'n', 'o', 'p', 'q'])
```



### index()

```python
index(x[, start[, stop]])
```

返回 `x` 在 `deque` 中的第一个匹配的位置（在索引 `start` 之后，索引 `stop` 之前），如果未找到匹配项则引发 `ValueError`。

```python
>>> dq
deque(['l', 'm', 'n', 'l', 'm', 'l'])
>>> dq.index('m')
1
```



### insert()

在指定位置插入元素。如果插入会导致一个限长 `deque` 超出最大长度 `maxlen`，则引发 `IndexError`。

```python
>>> dq
deque(['l', 'm', 'n'])
>>> dq
deque(['l', 'a', 'm', 'n'])
```



### maxlen

`deque` 的最大长度，如果没有限定的话就是 `None`。`deque` 对象的唯一只读属性。

```python
>>> dq = deque(['l', 'm', 'n'])
>>> dq.maxlen
>>> dq = deque(['l', 'm', 'n'], maxlen=3)
>>> dq.maxlen
3
```



### pop(), popleft()

弹出右端/左端的元素，如果没有元素则引发 `IndexError`。

```python
>>> dq
deque(['l', 'm', 'n'])
>>> dq.pop()                # 弹出右端的元素
'n'
>>> dq.popleft()            # 弹出左端的元素
'l'
```



### remove()

移除找到的第一个等于参数的元素，如果没有相应元素则引发 `ValueError`。

```python
>>> dq
deque(['l', 'm', 'n', 'l', 'm', 'l'])
>>> dq.remove('m')
1
>>> dq
deque(['l', 'n', 'l', 'm', 'l'])
```



### reverse()

将 `deque` 逆序排列。

```python
>>> dq
deque(['l', 'm', 'n'])
>>> dq.reverse()
>>> dq
deque(['n', 'm', 'l'])
```



### rotate()

```python
rotate(n=1)
```

向右循环移动 `n` 步；如果 `n` 是负数，就向左循环。

```python
>>> dq
deque(['l', 'm', 'n'])
>>> dq.rotate()
>>> dq
deque(['n', 'l', 'm'])
>>> dq.rotate(-1)
>>> dq
deque(['l', 'm', 'n'])
```



### 应用

```python
# 移动平均
def moving_average(iterable, n=3):
    # moving_average([40, 30, 50, 46, 39, 44]) --> 40.0 42.0 45.0 43.0
    it = iter(iterable)
    d = deque(itertools.islice(it, n-1))
    d.appendleft(0)
    s = sum(d)
    for elem in it:
        s += elem - d.popleft()
        d.append(elem)
        yield s / n
```

```python
# 轮询调度器
def roundrobin(*iterables):
    "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
    iterators = deque(map(iter, iterables))
    while iterators:
        try:
            while True:
                yield next(iterators[0])
                iterators.rotate(-1)
        except StopIteration:
            # Remove an exhausted iterator.
            iterators.popleft()
```





# 自定义容器数据类型

`collections.abc` 模块定义了一些抽象基类，可用于判断一个具体的类是否具有某个特定的接口，或者开发支持特定容器 API 的类。

```python
>>> import collections.abc
>>> class SizedContainer():
    def __init__(self):
        pass
    def __len__(self):
        pass
... 
>>> sc = SizedContainer()
>>> isinstance(sc, collections.abc.Sized)    # 具有`__len__`接口
True
```

```python
>>> import collections.abc
>>> class SizedContainer(collections.abc.Sized):
    def __len__(self):                       # 实现抽象方法`__len__`
        return 0
... 
>>> sc = SizedContainer()
>>> len(sc)
0
```

```python
class ListBasedSet(collections.abc.Set):
    ''' Alternate set implementation favoring space over speed
        and not requiring the set elements to be hashable. '''
    def __init__(self, iterable):
        self.elements = lst = []
        for value in iterable:
            if value not in lst:
                lst.append(value)

    def __iter__(self):                      # 实现抽象方法
        return iter(self.elements)

    def __contains__(self, value):
        return value in self.elements

    def __len__(self):
        return len(self.elements)
      
s1 = ListBasedSet('abcdef')
s2 = ListBasedSet('defghi')
overlap = s1 & s2                            # 抽象基类作为混入类补充`__and__`等方法
print(list(iter(s1)))
print(list(iter(s2)))
print(list(iter(overlap)))
```

```
['a', 'b', 'c', 'd', 'e', 'f']
['d', 'e', 'f', 'g', 'h', 'i']
['d', 'e', 'f']
```



`collections.abc` 模块提供了以下抽象基类：

| 抽象基类          | 继承自                           | 抽象方法                                                     | 混入类方法                                                   |
| :---------------- | :------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| `Container`       |                                  | `__contains__`                                               |                                                              |
| `Hashable`        |                                  | `__hash__`                                                   |                                                              |
| `Iterable`        |                                  | `__iter__`                                                   |                                                              |
| `Iterator`        | `Iterable`                       | `__next__`                                                   | `__iter__`                                                   |
| `Reversible`      | `Iterable`                       | `__reversed__`                                               |                                                              |
| `Generator`       | `Iterator`                       | `send`, `throw`                                              | `close`, `__iter__`, `__next__`                              |
| `Sized`           |                                  | `__len__`                                                    |                                                              |
| `Callable`        |                                  | `__call__`                                                   |                                                              |
| `Collection`      | `Sized`, `Iterable`, `Container` | `__contains__`, `__iter__`, `__len__`                        |                                                              |
| `Sequence`        | `Reversible`, `Collection`       | `__getitem__`, `__len__`                                     | `__contains__`, `__iter__`, `__reversed__`, `index`, and `count` |
| `MutableSequence` | `Sequence`                       | `__getitem__`, `__setitem__`, `__delitem__`, `__len__`, `insert` | 继承自 `Sequence` 的方法，以及 `append`, `reverse`, `extend`, `pop`, `remove`，和 `__iadd__` |
| `ByteString`      | `Sequence`                       | `__getitem__`, `__len__`                                     | 继承自 `Sequence` 的方法                                     |
| `Set`             | `Collection`                     | `__contains__`, `__iter__`, `__len__`                        | `__le__`, `__lt__`, `__eq__`, `__ne__`, `__gt__`, `__ge__`, `__and__`, `__or__`, `__sub__`, `__xor__`, and `isdisjoint` |
| `MutableSet`      | `Set`                            | `__contains__`, `__iter__`, `__len__`, `add`, `discard`      | 继承自 `Set` 的方法以及 `clear`, `pop`, `remove`, `__ior__`, `__iand__`, `__ixor__`，和 `__isub__` |
| `Mapping`         | `Collection`                     | `__getitem__`, `__iter__`, `__len__`                         | `__contains__`, `keys`, `items`, `values`, `get`, `__eq__`, and `__ne__` |
| `MutableMapping`  | `Mapping`                        | `__getitem__`, `__setitem__`, `__delitem__`, `__iter__`, `__len__` | 继承自 `Mapping` 的方法以及 `pop`, `popitem`, `clear`, `update`，和 `setdefault` |
| `MappingView`     | `Sized`                          |                                                              | `__len__`                                                    |
| `ItemsView`       | `MappingView`, `Set`             |                                                              | `__contains__`, `__iter__`                                   |
| `KeysView`        | `MappingView`, `Set`             |                                                              | `__contains__`, `__iter__`                                   |
| `ValuesView`      | `MappingView`, `Collection`      |                                                              | `__contains__`, `__iter__`                                   |
| `Awaitable`       |                                  | `__await__`                                                  |                                                              |
| `Coroutine`       | `Awaitable`                      | `send`, `throw`                                              | `close`                                                      |
| `AsyncIterable`   |                                  | `__aiter__`                                                  |                                                              |
| `AsyncIterator`   | `AsyncIterable`                  | `__anext__`                                                  | `__aiter__`                                                  |
| `AsyncGenerator`  | `AsyncIterator`                  | `asend`, `athrow`                                            | `aclose`, `__aiter__`, `__anext__`                           |

