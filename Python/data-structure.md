# 列表list []

```python
s = [1,2,3,4]	    #定义list
s = [1,2] + [3,4]	#拼接list

len(s)	# 4

s[0]	# 1
s[1]	# 2
s[-1]	# 4
s[-2]	# 3
s[-3]	# 2

s[:2]	# [1,2]
s[1:4]	# [2,3,4]
s[-2:]	# [3,4]

len(s)          # 4
s.append(5)	    # [1,2,3,4,5]
s.insert(1,1.5)	# [1,1.5,2,3,4,5]
s.pop()			# [1,1.5,2,3,4]
s.pop(1)		# [1,2,3,4]

s[1] = 2.1		     # [1,2.1,3,4]
s[1:3] = [1.9,2.9]   # [1,1.9,2.9,4]
s[1:3] = []          # [1,4]
```



## 方法

```python
append(x)         #在列表的末尾添加一个元素
extend(iterable)  #使用可迭代对象中的所有元素来扩展列表
insert(i, x)      #在指定的索引位置插入一个元素
remove(x)         #移除列表中第一个值为x的元素
pop([i])          #删除列表中指定位置的元素并返回它,默认为最后一个元素
clear()           #移除列表中的所有元素
index(x[,start[,end]]) #返回列表中第一个值为x的元素的从零开始的索引,start和end用于指定特定的搜索序列
count(x)          #返回元素x在列表中出现的次数
sort(key=None, reverse=False)  #对列表中的元素进行排序(自定义排序参见sorted())
reverse()         #翻转列表中的元素
copy()            #返回列表的一个浅拷贝
```



## 遍历

```python
for i, v in enumerate(['tic', 'tac', 'toe']):
    print(i, v)
    
# output
# 0 tic
# 1 tac
# 2 toe
```



## 实现栈

列表使用`append()`和`pop()`方法



## 实现队列

列表也可以实现队列，但在列表的开头插入或弹出元素很慢。建议使用`collections.dueue`，它被设计成可以快速地从两端添加或弹出元素。



## 列表推导式

```python
list(range(1,11))	#[1,2,...,10]
[x*x for x in range(1,11)]	                #for, [1,4,...,100]
[x*x for x in range(1, 11) if x % 2 == 0]	#for*if,[4,16,...,100]
[(x, y) for x in [1,2,3] for y in [3,1,4] if x != y]  #for*for*if,组合两个list中不相等的元素
[str(round(pi, i)) for i in range(1, 6)]    #['3.1', '3.14', '3.142', '3.1416', '3.14159']
```

嵌套的列表推导式

```python
matrix = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
]

[[row[i] for row in matrix] for i in range(4)]
# [[1, 5, 9], [2, 6, 10], [3, 7, 11], [4, 8, 12]]
```





# 元组tuple ()

元组由几个被逗号隔开的值组成，例如

```python
t = 12345, 54321, 'hello!'
t    # (12345, 54321, 'hello!')
u = t, (1, 2, 3, 4, 5)
u    # ((12345, 54321, 'hello!'), (1, 2, 3, 4, 5))
t[0] = 88888   # err
x, y, z = t    # 元组解包
```

元组总是被圆括号包围；不允许单独给元组的一个元素赋值。

比较列表和元组：

+ 元组是不可变对象，其序列通常包含不同种类的元素，并且通过解包或者索引来访问
+ 列表是可变对象，列表中的元素一般是同种类型，并且通过迭代访问

构造包含0个或1个元素的元组有特殊的语法：

```python
empty = ()
singleton = 'hello',    # 注意逗号,用于创建单元素元组
singleton               # ('hello',)
```





# 字典dict {:}

dict查找和插入的速度快，但占用内存大，即用空间换时间

dict的key必须是**不可变对象**（字符串，整数，etc）

```python
d = {'Michael': 95, 'Bob': 75, 'Tracy': 85}
# d = dict(Michael=95, Bob=75, Tracy=85)
# d = dict([('Michael', 95), ('Bob', 75), ('Tracy', 85)])

d['Michael']	#返回指定key的value
d['Adam']=67	#添加新key
'Thomas' in d	#判定key存在
list(d)         #返回所有key组成的列表
sorted(d)       #排序所有key
d.get('Thomas',-1)	#判定key存在，存在则返回索引位置，不存在返回-1
d.pop('Bob')	#删除key

d.items()		#k,v二维数组
```



## 字典推导式

```python
{x: x**2 for x in (2, 4, 6)}   # {2: 4, 4: 16, 6: 36}
```



## 遍历

```python
d = {'Michael': 95, 'Bob': 75, 'Tracy': 85}
for k, v in d.items():
    print(k, v)
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
#list切片
L = list(range(100))

L[:10]
L[-10:]
L[10:20]
L[:10:2]
L[::5]
L[:]

#tuple也可以使用切片操作，只是返回的仍为tuple

#str切片
word = 'Python'
word[:2]      # 'Py'
word[4:]      # 'on'
word[4:10]    # 'on',切片中的越界索引会被自动处理
word[::2]     # 

```



## 迭代

```python
#list迭代
L=[1,2,3]
for i in L:
    pass

#dict迭代
d = {'a': 1, 'b': 2, 'c': 3}
for key in d:	#迭代key
	print(key)
for value in d.values():
    print(value)	#迭代value
for k,v in d.items():
    print(k,v)		#迭代key和value
    
#str迭代
for ch in 'ABC':
    print(ch)
```



## 生成器generator

```python
g = (x * x for x in range(10))	#生成器需要对象
for i in g:		#循环：把g的过程走完
    print(i)	#循环一次计算一次，节省空间
    
def fib(max):	#generator型函数
    n, a, b = 0, 0, 1
    while n < max:
        yield b
        a, b = b, a + b
        n = n + 1
    return 'done'   
for n in fib(6):
     print(n)
```



## 迭代器

可以作用于for循环的对象为可迭代对象Iterable，包括list,tuple,dict,set,str,...

可以被next()调用并不断返回下一个值的对象称为迭代器Iterator，包括生成器



## map() & reduce()

```python
r=map(str, [1, 2, 3, 4, 5, 6, 7, 8, 9])	#map()将一元函数作用在序列上，iterator
list(r)	#list()返回iterator所有元素


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


