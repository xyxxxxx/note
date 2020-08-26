# 列表list []

```python
s = ['Michael', 'Sarah', 'Tracy', 'Bob', 'Jack']	#定义list
s=[1,2]+[3,4]	#拼接list

len(s)	#返回长度

s[0]	#返回列表首元素
s[1]	#返回列表第二个元素
s[-1]	#返回列表末元素
s[-2]	#返回列表倒数第二个元素
s[-3]	#返回列表倒数第三个元素

s[:2]	#返回列表前2个元素,左闭右开区间
s[1:4]	#返回列表第2至第4个元素
s[-2:]	#返回列表后2个元素

s.append('Adam')	#添加元素至末尾
s.insert(1,'Jack')	#添加元素至1索引位置
s.pop()			#删除末元素
s.pop(1)			#删除1索引位置的元素
s[1]='Sarah'		#赋值1索引位置元素

```





# 元组tuple ()

同list，但是tuple初始化后不得修改





# 字典dict {:}

dict查找和插入的速度快，但占用内存大，即用空间换时间

dict的key必须是**不可变对象**（字符串，整数，etc）

```python
d = {'Michael': 95, 'Bob': 75, 'Tracy': 85}	#定义dict，其中Michael为key，95为value

d['Michael']	#返回Michael的value
d['Adam']=67	#添加新key
'Thomas' in d	#判定key存在
d.get('Thomas',-1)	#判定key存在，存在则返回索引位置，不存在返回-1
d.pop('Bob')	#删除key

d.items()		#k,v二维数组
```





# 集合set {}

set中元素无序，且重复元素只计算1个

```python
s=set([1,2,3])	#用list定义set

s.add(4)		#添加元素
s.remove(4)		#删除元素

s1=set([1,2,3])
s2=set([1,2,4])
s1&s2			#交集
s1|s2			#并集
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
'ABCDEFG'[::2]

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



## 生成list

```python
list(range(1,11))	#[1,2,...,10]

[x*x for x in range(1,11)]	#表达式生成,for
[x * x for x in range(1, 11) if x % 2 == 0]	#for+if
[m + n for m in 'ABC' for n in 'XYZ']	#for^2
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

```python
sorted([36, 5, -12, 9, -21])	#[-21, -12, 5, 9, 36]
sorted([36, 5, -12, 9, -21],reverse=True)	#逆序
sorted([36, 5, -12, 9, -21], key=abs) 	#按abs()进行排序

sorted(['bob', 'about', 'Zoo', 'Credit'])
#['Credit', 'Zoo', 'about', 'bob'], 默认按ASCII排序，大写字母在小写字母之前
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


