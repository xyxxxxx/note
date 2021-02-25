[toc]

# 列表list []

```python
>>> s = [1,2,3,4]      # 定义list
>>> s
[1, 2, 3, 4]
>>> s = [1,2] + [3,4]  # 拼接list
>>> s
[1, 2, 3, 4]

>>> len(s)
4

>>> s[0]               # 按索引访问
1
>>> s[1]
2
>>> s[-1]
4
>>> s[-2]
3

>>> s[:2]
[1, 2]
>>> s[1:4]
[2, 3, 4]
>>> s[-2:]
[3, 4]

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
>>> t = s             # t, s引用同一列表
>>> u = list(s)       # 返回s的浅拷贝
                      # 相当于创建新列表,各元素分别为1,2和列表[3,4],引用同一列表
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

```python
>>> a = [1, 2, 7, 4, 3]

# append(x) 在列表的末尾添加一个元素
>>> a.append(5)
>>> a
[1, 2, 7, 4, 3, 5]

# extend(iterable) 使用可迭代对象中的所有元素来扩展列表
>>> a.extend(range(5))
>>> a
[1, 2, 7, 4, 3, 5, 0, 1, 2, 3, 4]

# insert(i, x) 在指定的索引位置插入一个元素
>>> a.insert(2, 9)
>>> a
[1, 2, 9, 7, 4, 3, 5, 0, 1, 2, 3, 4]

# remove(x) 移除列表中第一个值为x的元素
>>> a.remove(1)
>>> a
[2, 9, 7, 4, 3, 5, 0, 1, 2, 3, 4]

# pop([i]) 删除列表中指定位置的元素并返回它,默认为最后一个元素
>>> a.pop()
4
>>> a.pop(9)
3
>>> a
[2, 9, 7, 4, 3, 5, 0, 1, 2]

# count(x) 返回元素x在列表中出现的次数
>>> a.count(2)
2

# index(x[,start[,end]]) 返回列表中第一个值为x的元素的从零开始的索引, start和end用于指定特定的搜索序列
>>> a.index(2)
0
>>> a.index(2,1)
8

# sort(key=None, reverse=False) 对列表中的元素进行排序(自定义排序参见sorted())
>>> a.sort()
>>> a
[0, 1, 2, 2, 3, 4, 5, 7, 9]

# reverse() 翻转列表中的元素
>>> a.reverse()
>>> a
[9, 7, 5, 4, 3, 2, 2, 1, 0]

# clear() 移除列表中的所有元素

# copy() 返回列表的一个浅拷贝

```



## 迭代

```python
>>> for v in ['tic', 'tac', 'toe']:
...     print(v)
... 
tic
tac
toe

>>> for i, v in enumerate(['tic', 'tac', 'toe']):
...     print(i, v)
... 
0 tic
1 tac
2 toe
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
>>> [x * x for x in range(1, 11)]
[1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
>>> [x * x for x in range(1, 11) if x % 2 == 0]
[4, 16, 36, 64, 100]
>>> [(x, y) for x in [1, 2, 3] for y in [3, 1, 4] if x != y]
[(1, 3), (1, 4), (2, 3), (2, 1), (2, 4), (3, 1), (3, 4)]
>>> [x * x if x % 2 == 0 else -x * x for x in range(1, 11)]
[-1, 4, -9, 16, -25, 36, -49, 64, -81, 100]

>>> list(range(1, 11))
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
>>> [x * x for x in range(1, 11)]
[1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
>>> [x * x for x in range(1, 11) if x % 2 == 0]      # 附加条件
[4, 16, 36, 64, 100]
>>> [(x, y) for x in [1, 2, 3] for y in [3, 1, 4] if x != y]   # 双重for循环+条件
[(1, 3), (1, 4), (2, 3), (2, 1), (2, 4), (3, 1), (3, 4)]
>>> [x * x if x % 2 == 0 else -x * x for x in range(1, 11)]    # for循环+if-else语句
[-1, 4, -9, 16, -25, 36, -49, 64, -81, 100]
```

嵌套的列表推导式

```python
>>> matrix = [
...     [1, 2, 3, 4],
...     [5, 6, 7, 8],
...     [9, 10, 11, 12],
... ]
>>> [[row[i] for row in matrix] for i in range(4)]
[[1, 5, 9], [2, 6, 10], [3, 7, 11], [4, 8, 12]]
```





# 元组tuple ()

元组由几个被逗号隔开的值组成，例如

```python
>>> t = 12345, 54321, 'hello!'
>>> t
(12345, 54321, 'hello!')
>>> u = t, (1, 2, 3, 4, 5)
>>> u
((12345, 54321, 'hello!'), (1, 2, 3, 4, 5))
>>> t[0] = 88888
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: 'tuple' object does not support item assignment
>>> x, y, z = t   # 元组解包
```

元组总是被圆括号包围。不允许给元组的元素赋值，而只能创建新元组。

比较列表和元组：

+ 元组是不可变对象，其序列通常包含不同种类的元素，并且通过解包或者索引来访问
+ 列表是可变对象，列表中的元素一般是同种类型，并且通过迭代访问

构造包含0个或1个元素的元组有特殊的语法：

```python
>>> empty = ()
>>> empty
()
>>> singleton = 'hello',  # 注意逗号,用于创建单元素元组 
>>> singleton
('hello',)           
```

元组可以进行比较，采用逐元素比较法。





# 字典dict {:}

dict查找和插入的速度快，但占用内存大，即用空间换时间

dict的key必须是**不可变对象**（字符串，整数，etc）

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

>>> d['Michael']     # 按key访问
95
>>> d['Adam'] = 67   # 添加新key 
>>> d.pop('Bob')     # 弹出key
75
>>> d['Michael'] = 90
>>> d['Michael']
90

>>> 'Thomas' in d  # 判定key存在
False
>>> d.get('Thomas', -1)  # 判定key存在，存在则返回索引位置，不存在返回-1
-1
>>> list(d)        # 返回所有key组成的列表
['Michael', 'Bob', 'Tracy', 'Adam']
>>> sorted(d)      # 排序所有key
['Adam', 'Bob', 'Michael', 'Tracy']

>>> d.items()                          # d.items()	k, v二维数组
dict_items([('Michael', 95), ('Tracy', 85), ('Adam', 67)])
>>> list(zip(d.values(), d.keys()))    # (v,k)列表
[(95, 'Michael'), (85, 'Tracy'), (67, 'Adam')]
```



## 方法

```python
>>> d = {'a':1, 'b':2, 'c':3}

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
...     print(k, v)   # 迭代key, value
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





# 集合set {}

集合中元素无序，且重复元素只计算1个。集合的基本用法包括成员检测和消除重复元素。

```python
s = set([1,2,3])	# 用list定义set,亦可使用
                    # s = {1, 2, 3}

s.add(4)		    # 添加元素
s.remove(4)		    # 删除元素

s1 = set([1,2,3])
s2 = set([1,2,4])
s1 & s2			    # 交集
s1 | s2			    # 并集
s1 - s2             # 差集
s1 ^ s2             # 对称差集
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

# str切片
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



## map() & reduce()

```python
r = map(str, [1, 2, 3, 4, 5, 6, 7, 8, 9])	# map()将一元函数作用在序列上，iterator
list(r)	# list()返回iterator所有元素


from functools import reduce
def fn(x, y):
     return x * 10 + y
reduce(fn, [1, 3, 5, 7, 9])	#reduce()将多元函数依次作用在序列上
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





