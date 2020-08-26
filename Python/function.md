# built-in functions

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





# 定义函数

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





# 参数和返回值

## 参数

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



## 返回值

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





# 递归函数

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





# 变量与函数

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





#  匿名函数lambda

```python
list(map(lambda x: x * x, [1, 2, 3, 4, 5, 6, 7, 8, 9]))
#              参数 返回值

#匿名函数赋值
f = lambda x: x*x
f(5)	#25

```





# 装饰器Decorator

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





# 偏函数Partial function

```python
import functools
int2 = functools.partial(int, base=2)
int2('1000000')				#64
int2('1000000', base=10)	#1000000
```

Partial function的作用相当于把一个函数的某些参数固定，返回一个新函数

