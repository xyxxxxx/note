# 数据类型

## 整数

```python
bin()	0b1100	#二进制数
oct()	0o1472	#八进制数
int()	1989
hex()	0xff00	#十六进制数
```



## 浮点数

```python
1.2e9	#1.2*10^9
1.2e-5	#1.2*10^-5
```



## 字符串

```python
print('包含中文的str')	#python3的字符串以Unicode编码，支持多语言

#ord()字符→编码,chr()编码→字符
ord('A')	#65
ord('中')	#20013
chr(66)		#B
chr(25991)	#文

a='INnoVation'
a.replace('a','A')	#'INnoVAtion'
a.lower()			#'innovation'
a.upper()			#'INNOVATION'
a.capitalize()		#'Innovation'
```

### str和bytes转换



### 转义字符

| \\'  | ‘      | \\\  | \    |
| ---- | ------ | ---- | ---- |
| \\"  | “      | %%   | %    |
| \n   | 换行   |      |      |
| \t   | 制表符 |      |      |



```python
print('I\'m OK.')	#转义字符
print(r'\t\n\\')	#默认不转义，引号不可

print('''line1	#打印多行
line2
line3''')

```



## 布尔值和布尔运算

```python
3>2			#True
3>2 or 3>4	#True
not True	#False
#任意非零数值，非空字符串，非空数组都为True, 数值0为False
```



## 空值

```python
None
```



## 数据转换

```python
int()	#转换为整数类型
```



## 其他

### 赋值

```python
a=123
b='ABC'

```

### 常量

```python
PI=3.14159265359	#习惯用全大写字母表示常量
```





# 运算

## 赋值

```python
a=1		#无需声明类型，但必要时需要转换
b=int(input())

a,b,c=1,2,3	#多重赋值
a,b=b,a+b	#无需临时变量
```



## 算术运算

```python
# +-*/
10//3	#除法取整
10%3	#除法取余
10**3	#10^3
```





# 数组

## 列表list []

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



## 元组tuple ()

同list，但是tuple初始化后不得修改



## 字典dict {:}

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



## 集合set {}

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



## 高级特性

### 切片

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



### 迭代

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



### 生成list

```python
list(range(1,11))	#[1,2,...,10]

[x*x for x in range(1,11)]	#表达式生成,for
[x * x for x in range(1, 11) if x % 2 == 0]	#for+if
[m + n for m in 'ABC' for n in 'XYZ']	#for^2
```



### 生成器generator

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



### 迭代器

可以作用于for循环的对象为可迭代对象Iterable，包括list,tuple,dict,set,str,...

可以被next()调用并不断返回下一个值的对象称为迭代器Iterator，包括生成器



### map() & reduce()

```python
r=map(str, [1, 2, 3, 4, 5, 6, 7, 8, 9])	#map()将一元函数作用在序列上，iterator
list(r)	#list()返回iterator所有元素


from functools import reduce
def fn(x, y):
     return x * 10 + y
reduce(fn, [1, 3, 5, 7, 9])	#reduce()将多元函数依次作用在序列上
```



### filter()

```python
def is_odd(n):
    return n % 2 == 1

list(filter(is_odd, [1, 2, 4, 5, 6, 9, 10, 15]))	#filter()将判别函数作用在序列上，删去返回False的元素,iterator
```



### sorted()

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





# 流程控制

## 条件语句

```python
age = 3
if age >= 18:	#if语句执行下述缩进的语句
    print('adult')
elif age >= 6:
    print('teenager')
else:
    print('kid')
```



## 循环语句

### for循环

```python
sum = 0
for x in range(101):	#range(101)生成0-100的整数序列
    sum = sum + x
print(sum)

#多变量遍历
for x, y in [(1, 1), (2, 4), (3, 9)]:
     print(x, y)
```



### while循环

```python
sum = 0
n = 100
while n > 0:
    sum = sum + n
    n = n - 1
print(sum)
```

```python
break	#跳出循环
continue#跳过当次循环
```





# 函数

## built-in functions

> https://docs.python.org/3/library/functions.html#abs

```python
abs()	#绝对值
bin()	#转换为二进制
enumerate()	#对list每个元素添加索引，得到二维list
float()	#转换为float
int()	#转换为int

max()	#最大值
min()	#最小值
str()	#转换为str
sum()	#求和
```



## 定义函数

```python
#定义函数
def my_abs(x):
    if x >= 0:
        return x	#return以返回空值
    else:
        return -x

#空函数
def complex():
    pass	#pass占位以保证语法正确
```



## 参数和返回值

### 参数

```python
#多个参数
def enroll(name, gender, age=6, city='Beijing'):	#设定age,city默认值，默认值必须使用不变对象
    print('name:', name)
    print('gender:', gender)
    print('age:', age)
    print('city:', city)

enroll('Bob', 'M', 7)
enroll('Adam', 'M', city='Tianjin')    

#可变参数
def calc(numbers):	#1.将所有参数以list或tuple传入
    sum = 0
    for n in numbers:
        sum = sum + n * n
    return sum

def calc(*numbers):	#2.*表示传入tuple,多个参数/list可转换
    sum = 0
    for n in numbers:
        sum = sum + n * n
    return sum

#关键字参数
def person(name, age, **kw):	#kw为关键字参数，可以传入任意值或不传入值
        if 'city' in kw:		#关键字检查
        pass
    print('name:', name, 'age:', age, 'other:', kw)
    
person('Bob', 35, city='Beijing')
extra = {'city': 'Beijing', 'job': 'Engineer'}
person('Jack', 24, **extra)		#**表示传入dict

def person(name, age, *, city=Beijing, job):	# *后的city,job为关键字参数，可以传入这2个值或不传入
    print(name, age, city, job)

person('Jack', 24, job='Engineer')    
```

参数组合使用时，顺序必须为：必选参数、默认参数、可变参数、命名关键字参数和关键字参数

```python
#参数组合使用
def f1(a, b, c=0, *args, **kw):
    print('a =', a, 'b =', b, 'c =', c, 'args =', args, 'kw =', kw)
    
f1(1, 2, 3, 'a', 'b', x=99)
#a = 1 b = 2 c = 3 args = ('a', 'b') kw = {'x': 99} 
#args为命名关键字参数,kw为五明明关键字参数


def f2(a, b, c=0, *, d, **kw):
    print('a =', a, 'b =', b, 'c =', c, 'd =', d, 'kw =', kw)

f2(1, 2, d=99, ext=None)
#a = 1 b = 2 c = 0 d = 99 kw = {'ext': None}
```



### 返回值

```python
#多个返回值
import math

def move(x, y, step, angle=0):
    nx = x + step * math.cos(angle)
    ny = y - step * math.sin(angle)
    return nx, ny	#返回2个值

x, y = move(100, 100, 60, math.pi / 6)	#为2个变量赋值
r = move(100, 100, 60, math.pi / 6)		#返回1个tuple
```



## 递归函数

```python
def fact(n):
    if n==1:
        return 1
    return n * fact(n - 1)	#递归调用次数过多可能导致栈溢出，采用尾递归优化

def fact(n):
    return fact_iter(n, 1)
def fact_iter(num, product):
    if num == 1:
        return product
    return fact_iter(num - 1, num * product)	#仅调用自身，不会导致栈溢出
```



## 变量与函数

```python
#变量指向函数
f = abs
f(-10)	#10

#函数作为参数
def add(x,y,f):
    return f(x)+f(y)

#函数作为返回值
def lazy_sum(*args):
    def sum():
        ax = 0
        for n in args:
            ax = ax + n
        return ax
    return sum

f = lazy_sum(1, 3, 5, 7, 9)
f	#返回函数地址
f()	#返回函数值
#NOTICE:返回一个函数时，牢记该函数并未执行，返回函数中不要引用任何可能会变化的变量（循环变量等）
```



##  匿名函数lambda

```python
list(map(lambda x: x * x, [1, 2, 3, 4, 5, 6, 7, 8, 9]))
#              参数 返回值

#匿名函数赋值
f = lambda x: x*x
f(5)	#25

```



## 装饰器Decorator

```python
def log(func):		#decorator以函数为参数
    def wrapper(*args, **kw):
        print('call %s():' % func.__name__)
        return func(*args, **kw)
    return wrapper

@log		#调用装饰器, now=log(now)
def now():
    print('2015-3-25')

now()
```



## 偏函数Partial function

```python
import functools
int2 = functools.partial(int, base=2)
int2('1000000')				#64
int2('1000000', base=10)	#1000000
```

Partial function的作用相当于把一个函数的某些参数固定，返回一个新函数





# 模块

## 导入模块

```python
import math	#导入math包
x=math.sin(math.pi/2)
```

> Python标准库https://docs.python.org/zh-cn/3/library/index.html



## 定义模块

```python
#!/usr/bin/env python3			#标准注释:py3文件
# -*- coding: utf-8 -*-			#标准注释:使用UTF-8编码

' a test module '				#文档注释

__author__ = 'Michael Liao'		#作者名

import sys						#正文

def test():
    args = sys.argv
    if len(args)==1:
        print('Hello, world!')
    elif len(args)==2:
        print('Hello, %s!' % args[1])
    else:
        print('Too many arguments!')

if __name__=='__main__':
    test()
```



## 作用域

```python
#abc		public变量
#_abc		public变量，但惯例不直接引用
#__abc		private变量，不可直接引用
#__abc__	特殊变量，可以直接引用

def _private_1(name):		#内部函数
    return 'Hello, %s' % name

def _private_2(name):		
    return 'Hi, %s' % name

def greeting(name):			#外部接口
    if len(name) > 3:
        return _private_1(name)
    else:
        return _private_2(name)
```





# Object Oriented Programming

OOP把对象作为程序的基本单元，一个对象包含了数据和操作数据的函数

面向对象的三大特点：**数据封装**，**继承**，**多态**

## class & instance

```python
class Student(object):			#类名通常首字母大写,继承object类
    def __init__(self, name, score):	#构造函数,第一个参数恒为self,表示创建的实例自身,之后的参数表示实例的属性
        self.name = name
        self.score = score
    def print_score(self):				
        print('%s: %s' % (self.name, self.score))

bart=Student()				#创建对象
bart.name='Bart Simpson'	#对象属性赋值
bart.score=59
bart=Student('Bart Simpson',59)	#创建并初始化对象

print_score(bart)			#调用内部函数
bart.print_score()

```

**动态创建类**

```python
def fn(self, name='world'): #先定义函数
     print('Hello, %s.' % name)
Hello = type('Hello', (object,), dict(hello=fn)) 
#创建Hello class,依次传入3个参数:
#class的名称；
#继承的父类集合
#class的方法名称与函数绑定
```



### 定制类

```python
#__call__()定义 实例() 返回值




#__getattr__()为不存在的属性设定返回值
class Chain(object):
    def __init__(self, path=''):
        self._path = path
    def __getattr__(self, path):	#参数作为str传入
        return Chain('%s/%s' % (self._path, path))

Chain().status.user.timeline.list

#__getitem__()取类的实例
class Fib(object):
    def __getitem__(self, n):	#n为序数
        a, b = 1, 1
        for x in range(n):
            a, b = b, a + b
        return a

f=Fib()    
f[10]		#调用方法
#也可以传入slice等类型，但需要更多工作    
    
    
#__init__()定义构造函数


#__iter__()定义迭代器
class Fib(object):
    def __init__(self):
        self.a, self.b = 0, 1 # 初始化两个计数器a，b
    def __iter__(self):
        return self # 实例本身就是迭代对象，故返回自己
    def __next__(self):
        self.a, self.b = self.b, self.a + self.b # 计算下一个值
        if self.a > 100000: # 退出循环的条件
            raise StopIteration()
        return self.a # 返回下一个值

for n in Fib():	#先后进入init,iter,next,next,...
     print(n)	
    
    
#__len__()


#__member__()返回类的所有实例



#__slots__()定义类允许绑定的属性
class Student(object):
    __slots__ = ('name', 'age') #tuple

    
#__str__()定义打印实例返回值, __repr__()定义实例直接返回值
class Student(object):
    def __str__(self):
        return 'Student object (name: %s)' % self.name
	__repr__=__str__



```



### 枚举类

```python
#Enum()定义枚举类
from enum import Enum

Month = Enum('Month', ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'))		#各实例默认从1开始赋值

#自定义枚举类
from enum import Enum, unique
@unique					#检查没有重复值
class Weekday(Enum):	#继承Enum
    Sun = 0 			#Sun的value被设定为0
    Mon = 1
    Tue = 2
    Wed = 3
    Thu = 4
    Fri = 5
    Sat = 6
    
day1=Weekday.Mon
day1=Weekday(1)
print(day1)
print(day1.value)
```

### 元类metaclass



## 数据封装

```python
class Student(object):			
    def __init__(self, name, score):	
        self.__name = name		#定义为private变量,外部不可直接访问
        self.__score = score
    def get_name(self):			
        return self.__name
    def get_score(self):		#访问private变量的函数
        return self.__score
    def set_score(self, score): #修改private变量的函数
        self.__score = score 
    def print_score(self):				
        print('%s: %s' % (self.name, self.score))
```



## 继承和多态

```python
class Animal(object):
    def run(self):
        print('Animal is running...')

class Dog(Animal):
    def eat(self):
        print('Eating meat...')   

class Cat(Animal):
    def run(self):
        print('Cat is running...')        
        
dog=Dog()
dog.run()	#继承父类的函数
dog.eat()

cat=Cat()
cat.run()	#覆盖父类的函数,即多态
```



### 多重继承

```python
class Animal(object):
    pass
class Mammal(Animal):
    pass
class Bird(Animal):
    pass

class RunnableMixIn(object):
    def run(self):
        print('Running...')
class FlyableMixIn(object):
    def fly(self):
        print('Flying...')
        
class Dog(Mammal, RunnableMixIn):	#继承主线Mammal,附加RunnableMixIn
    pass        
```



## 获取对象信息

```python
#type()判断对象类型
type(123)		#int
type('123')		#str

type('abc')==type('123')	#True
type('abc')==str			#True
type('abc')==type(123)		#False

type(abs)==types.BuiltinFunctionType		#判断函数类型
type(lambda x: x)==types.LambdaType
```

```python
#isinstance()判断对象类型

#继承关系:object -> Animal -> Dog -> Husky
h = Husky()
isinstance(h,Husky)		#True,继承父类的数据类型
isinstance(h,Dog)		#True
isinstance(h,Animal)	#True

isinstance([1, 2, 3], (list, tuple))	#True,或型判断
```

```python
#操作对象属性
class Student(object):			
    def __init__(self, name, score):	
        self.__name = name		#定义为private变量,外部不可直接访问
        self.__score = score

bart=Student()        
dir(bart)					#返回对象所有属性和方法
hasattr(bart,'name')		#判断有无某属性
setattr(bart,'age',10)		#创建属性
getattr(bart,'sex',404)		#获取属性,不存在时返回404

```



## 属性

### 实例属性和类属性

```python
class Student(object):
    age=7		#类属性
    def __init__(self, name):
        self.name = name

s = Student('Bob')
s.score = 90		#py是动态语言,允许绑定任意属性
print(Student.age)	
print(s.age)		#每个实例皆可访问类属性
s.age=8
print(s.age)		#实例属性覆盖类属性
```



### 动态绑定属性

```python
class Student(object):
    pass

s = Student()
s.name = 'Michael' 		# 动态给实例绑定一个属性

def set_age(self, age): # 定义方法
	self.age = age

from types import MethodType
s.set_age = MethodType(set_age, s) # 给实例绑定方法
s.set_age(25) 					   # 调用实例方法

Student.set_age = set_age	   # 给class绑定方法

```



### 属性限制条件

```python
class Student(object):
    @property		#定义属性和getter
    def birth(self):
        return self._birth
    @birth.setter	#定义setter
    def birth(self, value):
        if value<19000000
        	raise ValueError('Invalid birthday') #报错
        self._birth = value
    @property
    def age(self):
        return 2015 - self._birth
```





# I/O

## 交互窗口I/O

### 输出

```python
print('abc')	#输出字符串
print('abc','def','gh')	#输出多个字符串
print(100)	#输出整数
print(100+100)	#输出计算结果

print('Hi, %s, you have $%d.' % ('Michael', 1000000))	#输出变量
print('Hi, {0}, you have ${1}.' .format('Michael', 1000000))
```

| 占位符 | 类型         |
| ------ | ------------ |
| %d     | 整数         |
| %f     | 浮点数       |
| %s     | 字符串       |
| %x     | 十六进制整数 |



### 输入

```python
print('Please enter your name:')
name=input()	#输入字符串
print('Hello,',name)

```



## 文件读写

### Read

```python
f = open('/Users/michael/test.txt', 'r')	#'r'读
f.read()		#返回读取内容
f.close()

with open('/Users/michael/test.txt', 'r') as f:	#常用
    print(f.read())

f = open('/Users/michael/test.jpg', 'rb')	#rb读二进制
f = open('/Users/michael/gbk.txt', 'r', encoding='gbk')	#以gbk编码读取,默认为UTF-8


```

```python
#read函数
read()		#读取文件全部内容
read(size)	#一次读取size字节的内容
readline()	#一次读取一行内容
readlines()	#一次读取所有内容并按行返回list
```



### Write

```python
f = open('/Users/michael/test.txt', 'w')	#w写,a追加
f.write('Hello, world!')
f.close()

with open('/Users/michael/test.txt', 'w') as f:	#常用
    f.write('Hello, world!')
```



## 流

### StringIO

```python
from io import StringIO
f = StringIO()		#创建
f.write('hello')	#写入
f.write(' ')
f.write('world!')
print(f.getvalue())	#获取

#亦可用读取文件的方式读取字符串流
```



### BytesIO

```python
from io import BytesIO
f = BytesIO()
f.write('中文'.encode('utf-8'))	#传入编码
print(f.getvalue())

#亦可用读取文件的方式读取字节流
```



## 操作文件和目录



## 序列化





# 进程和线程





# 错误，调试和测试

## 错误处理

### try

```python
try:
    print('try...')
    r = 10 / 0					#error,跳至except
    print('result:', r)
except ValueError as e:
    print('ValueError:', e)    
except ZeroDivisionError as e:	#捕获错误类型
    print('except:', e)
else:							#无错误
    print('no error!')    
finally:
    print('finally...')
print('END')
```

> python内置异常[file:///D:/library/python/library/exceptions.html](file:///D:/library/python/library/exceptions.html)



### 调用栈

```python
# err.py:
def foo(s):
    return 10 / int(s)
def bar(s):
    return foo(s) * 2
def main():
    bar('0')
main()

#错误信息
division by zero
堆栈跟踪:
 >  File "test.py", line 3, in foo
 >    return 10 / int(s)
 >  File "test.py", line 5, in bar
 >    return foo(s) * 2
 >  File "test.py", line 7, in main
 >    bar('0')
 >  File "test.py", line 8, in <module>
 >    main()
```

### 记录错误



### 抛出错误



## 调试

### assert

```python
def foo(s):
    n = int(s)
    assert n != 0, 'n is zero!'	#assert n!=0,else AssertionError
    return 10 / n
#可以用-O参数来关闭assert
```

### logging



## 测试

单元测试