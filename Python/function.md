[toc]



# 定义函数

```python
# 定义函数
def my_abs(x):
    '''my_abs returns the absolute value of number x
    '''
    if x >= 0:
        return x	# return以返回空值
    else:
        return -x

# 空函数
def complex():
    pass	        # pass作为占位可以保证语法正确
```





# 参数和返回值

## 参数

### 默认参数

```python
def enroll(name, gender, age=6, city='Beijing'):	# 设定age,city默认值，默认值必须使用不变对象
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



参数组合使用时，顺序必须为：必选参数、默认参数、可变参数和关键字参数：

```python
#参数组合使用
def f1(a, b, c=0, *args, **kw):
    print('a =', a, 'b =', b, 'c =', c, 'args =', args, 'kw =', kw)
    
f1(1, 2, 3, 'a', 'b', x=99)
#a = 1 b = 2 c = 3 args = ('a', 'b') kw = {'x': 99} 
```



## 返回值

```python
# 多个返回值
import math

def move(x, y, step, angle=0):
    nx = x + step * math.cos(angle)
    ny = y - step * math.sin(angle)
    return nx, ny	  # 返回包含这2个值的tuple

x, y = move(100, 100, 60, math.pi / 6)	# 为2个变量赋值
r = move(100, 100, 60, math.pi / 6)		  # 返回1个tuple
```

```python
# 0个返回值
def fib(n):
    """Print a Fibonacci series up to n."""
    a, b = 0, 1
    while a < n:
        print(a, end=' ')
        a, b = b, a+b
    print()
    # 由于没有return语句，函数结束时返回None
```





# 递归函数

```python
def fact(n):
    if n==1:
        return 1
    return n * fact(n - 1)	# 递归调用次数过多可能导致栈溢出，采用尾递归优化

def fact(n):
    return fact_iter(n, 1)
def fact_iter(num, product):
    if num == 1:
        return product
    return fact_iter(num - 1, num * product)	# 仅调用自身，不会导致栈溢出
```





# 作为对象的函数

```python
# 变量指向函数
f = abs
f(-10)	 # 10

# 函数作为参数
def add(x,y,f):
    return f(x)+f(y)

# 函数作为返回值
def lazy_sum(*args):
    def sum():
        ax = 0
        for n in args:
            ax = ax + n
        return ax
    return sum

f = lazy_sum(1, 3, 5, 7, 9)
f	   # 返回函数地址
f()	 # 返回函数值
# NOTICE:返回一个函数时，牢记该函数并未执行，返回函数中不要引用任何可能会变化的变量（循环变量等）
```





#  匿名函数lambda

```python
# 匿名函数赋值
f = lambda x: x*x
f(5)	 # 25

list(map(lambda x: x * x, [1, 2, 3, 4, 5, 6, 7, 8, 9]))
#              参数 返回值

```





# 装饰器

装饰器(Decorator)用于增强其它函数的功能，其本质是一个函数，接受被装饰的函数作为参数，使用该参数执行某些功能后返回一个函数引用。

装饰器源于设计模式中的**装饰模式**。Python对此进行了一些语法简化，即语法糖，使得应用装饰器更加简单。



## 包装一个函数

写一个简单的函数：

```python
def add(x, y=1):
    return x + y
```

运行几次：

```python
print(add(1))
print(add(2, 3))
print(add('a', 'b'))
```

```
2
5
ab
```

现在我们想测试这个函数的性能，于是在运行前后记录时间：

```python
from time import time

before = time()
add(1)
after = time()
print("elapsed:", after - before)

before = time()
add(2, 3)
after = time()
print("elapsed:", after - before)

before = time()
add('a', 'b')
after = time()
print("elapsed:", after - before)
```

```
elapsed: 9.5367431640625e-07
elapsed: 9.5367431640625e-07
elapsed: 0.0
```

代码马上变得很繁复，每次都要复制粘贴一堆代码。为了复用代码，我们可以将它们放到函数中：

```python
from time import time

def add(x, y=1):
    before = time()
    result = x + y
    after = time()
    print("elapsed:", after - before)
    return result
  
add(1)
add(2, 3)
add('a', 'b')
```

```
elapsed: 9.5367431640625e-07
elapsed: 0.0
elapsed: 1.1920928955078125e-06
```

现在代码变得更加简单，但是依然存在一些问题：

+ 要是想再测试其它函数的性能，就还要再将这几行代码复制到这些函数中。
+ 每次启用/禁用性能测试，都需要修改函数的代码，非常麻烦。
+ 性能测试的代码放在函数中导致没有计算调用函数和函数返回的时间。

一种解决方法是写一个包含性能测试代码的新函数，传入要测试的函数并在其中调用：

```python
from time import time

def timer(func, x, y=1):
    before = time()
    result = func(x, y)
    after = time()
    print("elapsed:", after - before)
    return result

def add(x, y=1):
    return x + y

def sub(x, y=1):
    return x - y

timer(add, 1)
timer(add, 2, 3)
timer(add, 'a', 'b')
```

```
elapsed: 9.5367431640625e-07
elapsed: 0.0
elapsed: 0.0
```

但是这样还是很麻烦，因为我们得修改每一处调用的代码，例如将`add(2, 3)`修改为`timer(add, 2, 3)`。于是我们进一步改进，让`timer()`返回一个包装函数：

```python
from time import time

def timer(func):
    def wrapper(x, y=1):
        before = time()
        result = func(x, y)
        after = time()
        print("elapsed: ", after - before)
        return result
    return wrapper

def add(x, y=1):
    return x + y
add = timer(add)

def sub(x, y=1):
    return x - y
sub = timer(sub)

add(1)
add(2, 3)
add('a', 'b')
```

```
elapsed:  9.5367431640625e-07
elapsed:  0.0
elapsed:  9.5367431640625e-07
```

这里的最后一个问题是，包装的函数可能有不同的参数，于是可以用`*args, **kwargs`接受所有参数：

```python
def timer(func):
    def wrapper(*args, **kwargs):
        before = time()
        result = func(*args, **kwargs)
        after = time()
        print("elapsed: ", after - before)
        return result
    return wrapper
```

现在得到的`timer()`就是一个装饰器，它接受一个函数，并返回一个新的函数。在装饰器的内部，对原函数进行了包装。



## @ 语法糖

在上面的例子中，使用装饰器用到了以下代码：

```python
def add(x, y=1):
    return x + y
add = timer(add)   # <- notice this

def sub(x, y=1):
    return x - y
sub = timer(sub)
```

`add = timer(add)`这一语句显得比较赘余，于是Python提供了进一步简化的语法：

```python
@timer
def add(x, y=1):
    return x + y

@timer
def sub(x, y=1):
    return x - y
```

`@`写法十分简洁，这也是我们最常见的装饰器形式。



## 带参数的装饰器

装饰器也是一种函数，有时我们也希望为其传入参数，方法是在装饰器外面再定义一层函数用于接受参数：

```python
def use_logging(level):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if level == "warn":
                logging.warn("%s is running" % func.__name__)
            elif level == "info":
                logging.info("%s is running" % func.__name__)
            return func(*args, **kwargs)
        return wrapper
    return decorator

@use_logging(level="warn")    # 返回一个装饰器,其包装函数的level确定为"warn"
def foo(name='foo'):
    print("i am %s" % name)
```





