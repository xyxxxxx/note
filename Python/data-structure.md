[toc]

# 列表list-[]

列表是可变序列，通常用于存放同类项目的集合。

```python
>>> s = [1,2,3,4]      # 创建列表
>>> s
[1, 2, 3, 4]
>>> s = [1,2] + [3,4]  # 拼接列表
>>> s
[1, 2, 3, 4]
>>> s = [1,2,[3,4]]    # 嵌套列表
>>> s
[1, 2, [3, 4]]
>>> len(s)             # 列表长度
4

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
```

```python
# 列表变量的关系
>>> import copy
>>> s = [1,2,[3,4]]
>>> t = s                 # t, s引用同一列表
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



## 方法

### `append()`

将指定元素添加到列表的末尾。

```python
>>> a = [1, 2, 7, 4, 3]
>>> a.append(5)
>>> a
[1, 2, 7, 4, 3, 5]
```



### `clear()`

移除列表中的所有元素。

```python
>>> a = [1, 2, 7, 4, 3]
>>> a.clear()
>>> a
[]
```



### `count()`

返回指定元素在列表中出现的次数。

```python
>>> a = [1, 2, 7, 4, 3]
>>> a.count(2)
1
>>> a.count(5)
0
```



### `copy()`

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



### `extend()`

使用可迭代对象中的所有元素来扩展列表。

```python
>>> a = [1, 2, 7, 4, 3]
>>> a.extend(range(5))
>>> a
[1, 2, 7, 4, 3, 0, 1, 2, 3, 4]
```



### `index()`

返回列表中的第一个指定元素的从零开始的索引，可选的第二和第三个参数用于指定特定的搜索序列。

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



### `insert()`

在列表的指定位置插入一个元素。

```python
>>> a = [1, 2, 7, 4, 3]
>>> a.insert(2, 9)
>>> a
[1, 2, 9, 7, 4, 3]
```



### `pop()`

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



### `remove()`

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



### `reverse()`

翻转列表中的元素。

```python
>>> a = [1, 2, 7, 4, 3]
>>> a.reverse()
>>> a
[3, 4, 7, 2, 1]
```



### `sort()`

对列表中的元素进行排序。

```python
>>> a = [1, 2, 7, 4, 3]
>>> a.sort()
>>> a
[1, 2, 3, 4, 7]
```



## 迭代

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



## 实现栈

列表使用`append()`和`pop()`方法。



## 实现队列

列表也可以实现队列，但在列表的开头插入或弹出元素很慢。建议使用`collections.deque`，它被设计成可以快速地从两端添加或弹出元素。



## 列表推导式

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





# 元组tuple-()

元组是不可变序列，通常用于储存异构数据的多项集。

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

+ 元组是不可变对象，其序列通常包含不同种类的元素，并且通过解包或者索引来访问
+ 列表是可变对象，列表中的元素一般是同种类型，并且通过迭代访问

元组之间可以进行比较，采用逐元素比较法。



# range



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

`range()`函数返回的对象尽管表现得像一个列表，但实际上是一个迭代器。



# 字典dict-{:}

dict查找和插入的速度快，但占用内存大，即用空间换时间。

dict的key必须是**不可变对象**（字符串，整数，etc）。

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

# 访问/增删/修改元素
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
>>> for k,v in d.items():
...     print(k, v)   # 迭代key,value
... 
a 1
b 2
c 3    
```



## 字典推导式

```python
>>> {x: x**2 for x in (2, 4, 6)}
{2: 4, 4: 16, 6: 36}
```





# 集合set-{}

集合中元素无序，且重复元素只计算1个。集合的基本用法包括成员检测和消除重复元素。

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
# list切片
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

`sorted()`函数传入原序列，返回一个新的排好序的序列。

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





# 容器数据类型

## Counter

`Counter`是 `dict`的子类，用于计数可哈希对象。它是一个集合，元素像字典键一样存储，它们的计数存储为值。计数可以是任何整数值，包括0和负数。

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
>>> cnt['d']
0
>>> cnt1 = Counter({'c': 1, 'd': 8})
>>> cnt + cnt1             # 计数值相加
Counter({'d': 8, 'c': 4, 'b': 2, 'a': 1})
>>> cnt & cnt1             # 计数值求交集
Counter({'c': 1})
>>> cnt | cnt1             # 计数值求并集
Counter({'d': 8, 'c': 3, 'b': 2, 'a': 1})
>>> cnt = cnt + cnt1
>>> cnt.most_common(2)     # 出现次数最高的2个元素
[('d', 8), ('c', 4)]
>>> list(cnt.elements())   # 返回一个迭代对象, 每个元素会重复计数值的次数
['a', 'b', 'b', 'c', 'c', 'c', 'c', 'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd']
>>> sum(cnt.values())      # 对所有计数值求和
15
```

迭代方法同`dict`。



## defaultdict

`defaultdict`是 `dict`的子类，可以将（键-值对组成的）序列转换为（键-列表组成的）字典：

```python
from collections import defaultdict

s = [('yellow', 1), ('blue', 2), ('yellow', 3), ('blue', 4), ('red', 1)]
d = defaultdict(list)
for k, v in s:
    d[k].append(v)
d
# defaultdict(<class 'list'>, {'yellow': [1, 3], 'blue': [2, 4], 'red': [1]})    
```

迭代方法同`dict`。



## deque

返回双向队列对象，从`iterable`数据创建（如果iterable没有指定，新队列为空）。

deque支持线程安全、内存高效的添加(append)和弹出(pop)，从两端都可以，两个方向的大概开销都是$$O(1)$$复杂度。

虽然 list 对象也支持类似操作，不过这里优化了定长操作和 `pop(0)` 和 `insert(0, v)` 的开销。它们引起 O(n) 内存移动的操作，改变底层数据表达的大小和位置。

如果`maxlen`没有指定或者是 `None` ，deque 可以增长到任意长度。否则，deque就限定到指定最大长度。一旦限定长度的deque满了，当新项加入时，同样数量的项就从另一端弹出。

deque支持以下方法：

```python
append(x)       # 添加 x 到右端
appendleft(x)   # 添加 x 到左端
clear()         # 移除所有元素，使其长度为0
copy()          # 创建一份浅拷贝
count(x)        # 计算 deque 中元素等于 x 的个数
extend(iterable)# 扩展deque的右侧，通过添加iterable参数中的元素
extendleft(iterable) 
# 扩展deque的左侧，通过添加iterable参数中的元素。注意，左添加时，在结果中iterable参数中的顺序将被反过来添加
index(x[, start[, stop]]) 
# 返回 x 在 deque 中的位置（在索引 start 之后，索引 stop 之前）。 返回第一个匹配项，如果未找到则引发 ValueError
insert(i, x)    # 在位置 i 插入 x
# 如果插入会导致一个限长 deque 超出长度 maxlen 的话，就引发一个IndexError
pop()
# 移去并且返回最右侧元素。如果没有元素的话，就引发一个IndexError
popleft()
# 移去并且返回最左侧元素。如果没有元素的话，就引发一个IndexError
remove(value)
# 移除找到的第一个 value。如果没有的话就引发 ValueError
reverse()       # 将deque逆序排列
rotate(n=1)     # 向右循环移动n步。如果n是负数，就向左循环

# deque对象的唯一只读属性
maxlen          # deque的最大尺寸，如果没有限定的话就是 None
```



## 自定义容器类

```python

```





