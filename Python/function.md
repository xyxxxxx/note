# built-in functions

> https://docs.python.org/3/library/functions.html





# 定义函数

```python
#定义函数
def my_abs(x):
    '''my_abs returns the absolute value of number x
    '''
    if x >= 0:
        return x	#return以返回空值
    else:
        return -x

#空函数
def complex():
    pass	        #pass作为占位可以保证语法正确
```





# 参数和返回值

## 参数

### 默认参数

```python
def enroll(name, gender, age=6, city='Beijing'):	#设定age,city默认值，默认值必须使用不变对象
    print('name:', name)
    print('gender:', gender)
    print('age:', age)
    print('city:', city)

enroll('Bob', 'M', 7)
enroll('Adam', 'M', city='Tianjin')
```

```python
# 多次调用之间共享默认值
def f(a, L=[]):
    L.append(a)
    return L

print(f(1))  # [1]
print(f(2))  # [1, 2]
print(f(3))  # [1, 2, 3]
```

```python
def f(a, L=None):
    if L is None:
       	L = []
    L.append(a)
    return L

print(f(1))  # [1]
print(f(2))  # [2]
print(f(3))  # [3]
```



### 可变参数

可变参数`*args`匹配剩余的参数，以元组的方式存储在`args`：

```python
def argsFunc(a, *args):
	print(a)
	print(args)
	
argsFunc(1, 2, 3, 4)
# t = [2,3,4]
# argsFunc(1, *t)

# output
# 1
# (2, 3, 4)
```



### 关键字参数

关键字参数`**kwargs`匹配剩余的形式为`arg=value`的键值对，以`dict`的方式存储在`kwargs`：

```python
def kwargsFunc(**kwargs):
    if 'x' in kwargs:    # 检查是否存在key
    	print(kwargs)
    
kwargsFunc(x=1,y=2,z=3)
# kw = {'x':1, 'y':2, 'z':3}
# kwargsFunc(**kw)       # 这里的**表示解包map

# output
# {'x': 1, 'y': 2, 'z': 3}
```



### 特殊参数

特殊符号`/`，`*`用于确定参数是按位置传递还是按关键字传递还是两者皆可，例如：

```python
# / 前的参数仅限位置参数
def pos_only_arg(arg, /):
     print(arg)

pos_only_arg(1)       # 1
pos_only_arg(arg=1)   # err
```

```python
# * 后的参数仅限关键字参数
def kwd_only_arg(*, arg):
     print(arg)
        
kwd_only_arg(3)       # err
kwd_only_arg(arg=3)   # 3
```



参数组合使用时，顺序必须为：必选参数、默认参数、可变参数和关键字参数

```python
#参数组合使用
def f1(a, b, c=0, *args, **kw):
    print('a =', a, 'b =', b, 'c =', c, 'args =', args, 'kw =', kw)
    
f1(1, 2, 3, 'a', 'b', x=99)
#a = 1 b = 2 c = 3 args = ('a', 'b') kw = {'x': 99} 
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

```python
#0个返回值
def fib(n):
    """Print a Fibonacci series up to n."""
    a, b = 0, 1
    while a < n:
        print(a, end=' ')
        a, b = b, a+b
    print()
    #由于没有return语句，函数结束时返回None
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
#匿名函数赋值
f = lambda x: x*x
f(5)	#25

list(map(lambda x: x * x, [1, 2, 3, 4, 5, 6, 7, 8, 9]))
#              参数 返回值

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

