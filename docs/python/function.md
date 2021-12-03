

# 定义函数

定义一个简单的绝对值函数：

```python
#   函数名 形参列表
def my_abs(x):
    """Returns the absolute value of number x"""  # 文档字符串,参见工具-代码规范-PEP257
    if x >= 0:
        return x                                  # 函数返回
    else:
        return -x
```

使用`pass`语句定义一个空函数：

```python
def empty():
    pass	        # `pass`作为占位符
```

## 参数

### 形参和实参

**形参**指形式上的参数，即函数定义的参数列表，没有实际值，需要赋值后才有意义；**实参**指实质上的参数，即为函数的形参传入的对象。

```python
>>> def my_add(a, b):  # 形参`a`,`b`
    return a + b
...
>>> my_add(1, 2)       # 实参`1`,`2`
3
```

函数定义中可以为形参指定类型，但该类型仅起到提示代码读者的作用，解释器并不会实际检查传入的实参是否为指定的类型，例如：

```python
>>> def my_add(a: int, b: int):  # 指定形参`a`,`b`为`int`类型
    return a + b
... 
>>> my_add(1, 2)                 # 传入实参为`int`类型
3
>>> my_add('a', 'bc')            # 传入实参为`str`类型
'abc'
```

> 与 C、Java 不同，Python 中的形参无法被预定义，因此仅仅是一个名称/变量，指向给函数传入的对象（即实参）。在阅读到中文 Python 文档中普遍出现的“参数”一词时，应充分理解这一特性。

### 默认值参数和必选参数

**默认值参数**指可以不传入实参、使用默认值的参数，与之相对的称为**必选参数**。为参数指定默认值可以使用更少的参数，更加方便地调用函数，例如：

```python
>>> def enroll(name, gender, age=6, city='Beijing'):    # 设定参数age,city的默认值
    print('name:', name)
    print('gender:', gender)
    print('age:', age)
    print('city:', city)
... 
>>> enroll('Alice', 'F')
name: Alice
gender: F
age: 6
city: Beijing
>>> enroll('Bob', 'M', 7)
name: Bob
gender: M
age: 7
city: Beijing
>>> enroll('Cindy', 'F', 6, 'Shanghai')  
name: Cindy
gender: F
age: 6
city: Shanghai
```

注意：

+ 默认值参数必须在必选参数之后

默认值若为列表、字典或类实例等可变对象，则仅在函数定义时计算一次，例如：

```python
>>> i = 5
>>> def f(arg=i):     # 函数定义时计算默认值i=5
    print(arg)
... 
>>> i = 6
>>> f()
5
```

```python
>>> def f(a, L=[]):   # 函数定义时计算默认值是一个空的列表对象,此后每次调用时L都默认指向该对象
    L.append(a)
    return L
... 
>>> print(f(1))
[1]
>>> print(f(2))
[1, 2]
>>> print(f(3))
[1, 2, 3]
```

```python
>>> def f(a, L=None):  # 正确的方法,防止在各次调用中共享默认列表对象
    if L is None:
        L = []
    L.append(a)
    return L
... 
>>> print(f(1))
[1]
>>> print(f(2))
[2]
>>> print(f(3))
[3]
```

### 关键字参数和位置参数

**关键字参数**指以 `key=value` 形式传入的实参，与之相对的称为**位置参数**：

```python
>>> def enroll(name, gender, age=6, city='Beijing'):
    print('name:', name)
    print('gender:', gender)
    print('age:', age)
    print('city:', city)
... 
>>> enroll('Cindy', 'F', city='Shanghai')       # 2个位置参数+1个关键字参数
name: Cindy
gender: F
age: 6
city: Shanghai
>>> enroll('Bob', 'M', city='Beijing', age=7)   # 2个位置参数+2个关键字参数,关键字参数顺序可以打乱
name: Bob
gender: M
age: 7
city: Beijing
```

注意：

+ 关键字参数必须在位置参数之后
+ 传递的关键字参数必须匹配一个函数接受的参数
+ 关键字参数的顺序可以打乱
+ 不能对同一参数多次赋值

### 可变参数

`*args` 形式的形参接收一个元组，该元组包含函数形参列表之外的所有位置参数；或直接匹配函数形参列表之外的所有位置参数，以元组的方式保存在`args`中：

```python
>>> def enroll(name, gender, age=6, city='Beijing', *attributes):
    print('name:', name)
    print('gender:', gender)
    print('age:', age)
    print('city:', city)
    print('attributes:', attributes)
... 
>>> enroll('Cindy', 'F', 'Shanghai', 6, 'outgoing', 'energetic')  # 要想为*args传入位置参数,则形参不可使用关键字参数
name: Cindy
gender: F
age: Shanghai
city: 6
attributes: ('outgoing', 'energetic')
```

`**kwargs` 形式的形参接收一个字典，该字典包含函数形参列表之外的所有关键字参数；或直接匹配函数形参列表之外的所有关键字参数，以字典的方式保存在`kwargs`中：

```python
>>> def enroll(name, gender, age=6, city='Beijing', **comments):
    print('name:', name)
    print('gender:', gender)
    print('age:', age)
    print('city:', city)
    print('comments:', comments)
... 
>>> enroll('Cindy', 'F', city='Shanghai', age=6, **{'height': 116, 'weight': 23})
name: Cindy
gender: F
age: 6
city: Shanghai
comments: {'height': 116, 'weight': 23}
>>> enroll('Cindy', 'F', city='Shanghai', age=6, height=116, weight=23)
name: Cindy
gender: F
age: 6
city: Shanghai
comments: {'height': 116, 'weight': 23}
```

同时使用 `*args` 和 `**kwargs`：

```python
# 参数组合使用时,顺序必须为: 必选参数,默认值参数,*args,**kwargs
>>> def enroll(name, gender, age=6, city='Beijing', *attributes, **comments):
    print('name:', name)
    print('gender:', gender)
    print('age:', age)
    print('city:', city)
    print('attributes:', attributes)
    print('comments:', comments)
>>> enroll('Cindy', 'F', 6, 'Shanghai', 'outgoing', 'energetic', height=116, weight=23)
name: Cindy
gender: F
age: 6
city: Shanghai
attributes: ('outgoing', 'energetic')
comments: {'height': 116, 'weight': 23}    
```

### 特殊参数

默认情况下，参数可以按位置或关键字传递给函数。特殊符号 `/`，`*` 用于规定参数是按位置传递还是按关键字传递还是两者皆可，例如：

```python
>>> def pos_only_arg(arg, /):          # `/`之前的参数仅限位置参数
     print(arg)
... 
>>> pos_only_arg(1)
1
>>> pos_only_arg(arg=1)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: pos_only_arg() got some positional-only arguments passed as keyword arguments: 'arg'
```

```python
>>> def kwd_only_arg(*, arg):          # `*`之后的参数仅限关键字参数
     print(arg)
... 
>>> kwd_only_arg(3)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: kwd_only_arg() takes 0 positional arguments but 1 was given
>>> kwd_only_arg(arg=3)
3
```

```python
>>> def combined_example(pos_only, /, standard, *, kwd_only):   # `/`之后,`*`之前的参数可以使用两种方法传入
     print(pos_only, standard, kwd_only)
...
>>> combined_example(1, 2, 3)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: combined_example() takes 2 positional arguments but 3 were given
>>> combined_example(1, 2, kwd_only=3)
1 2 3
>>> combined_example(1, standard=2, kwd_only=3)
1 2 3
>>> combined_example(pos_only=1, standard=2, kwd_only=3)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: combined_example() got an unexpected keyword argument 'pos_only'
```

## 返回值

函数可以返回多个值，实际上返回的是包含这些值的元组：

```python
>>> import math
>>> def move(x, y, step, angle=0):
    nx = x + step * math.cos(angle)
    ny = y - step * math.sin(angle)
    return nx, ny                               # 返回元组(nx, ny)
... 
>>> move(100, 100, 60, math.pi / 6)
(151.96152422706632, 70.0)
>>> x, y = move(100, 100, 60, math.pi / 6)      # 元组解包
>>> x
151.96152422706632
>>> y
70.0
```

函数也可以不返回值，没有返回语句和 `return` 相当于 `return None`：

```python
>>> def fib(n):
    """Print a Fibonacci series up to n."""
    a, b = 0, 1
    while a < n:
        print(a, end=' ')
        a, b = b, a+b
    print()                                     # 没有return语句,函数结束时返回`None`
... 
>>> fib(100)
0 1 1 2 3 5 8 13 21 34 55 89                    # 没有返回值打印出来
>>> print(fib(100))
0 1 1 2 3 5 8 13 21 34 55 89 
None                                            # 使用print()打印得到`None`
```

## 闭包

##  匿名函数

`lambda` 关键字用于创建小巧的匿名函数。所谓匿名函数，就是一类无需定义标识符（函数名）的函数。lambda 函数可用于任何需要函数对象的地方。在语法上，匿名函数只能是单个表达式。在语义上，它只是常规函数定义的语法糖。

lambda函数的用法主要有以下几种：

+ 定义简单的函数并赋值给变量，通过此变量调用该函数：

  ```python
  >>> f = lambda a: a * a
  >>> f(-2)
  4
  ```

  ```python
  >>> def make_incrementor(n):
  ...     return lambda x: x + n   # 与嵌套函数一样,lambda函数可以引用外部变量
  ...
  >>> f = make_incrementor(42)
  >>> f(0)
  42
  >>> f(1)
  43
  ```

+ 定义简单的函数并作为实参传递给其它函数：

  ```python
  >>> pairs = [(1, 'one'), (2, 'two'), (3, 'three'), (4, 'four')]
  >>> pairs.sort(key=lambda pair: pair[1])
  >>> pairs
  [(4, 'four'), (1, 'one'), (3, 'three'), (2, 'two')]
  ```

  ```python
  >>> list(map(lambda x: x * x, [1, 2, 3, 4, 5, 6, 7, 8, 9]))
  [1, 4, 9, 16, 25, 36, 49, 64, 81]
  ```

  

# 函数对象

Python 中一切皆为对象，函数也是对象，例如：

```python
>>> def a():
...     return 3
...
>>> b = a
>>> c = a()
```

这里变量 `b` 指向了 `a()` 这个函数，而 `c` 指向了这个函数的返回值 `3`。

```python
>>> def func_add(x, y, f):
    return f(x) + f(y)
... 
>>> func_add(-1, -2, abs)
3
```

这里定义的函数的第三个形参接受一个函数对象，并调用这个函数对象。

函数对象也可以作为函数的返回值，详见[装饰器](#装饰器)。

实质上，

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
    """Add two objects."""
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

## 函数对象信息的恢复

上面定义的装饰器在实际使用时还存在一个问题：它会改变包装的函数对象的信息：

```python
>>> def add(x, y=1):
    """Add two objects."""
    return x + y
... 
>>> add.__name__                            # 原名称
'add'
>>> help(add)                               # 原docstring

Help on function add in module __main__:

add(x, y=1)
    Add two objects.
>>>
>>> add = timer(add)
>>> add.__name__                            # 包装之后的名称
'wrapper'
>>> help(add)                               # 包装之后的docstring

Help on function wrapper in module __main__:

wrapper(*args, **kwargs)
```

为了保留

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

`add = timer(add)` 这一语句显得比较赘余，于是Python提供了进一步简化的语法：

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

## 类作为装饰器

如果说 Python 里一切都是对象的话，那函数怎么表示成对象呢？其实只需要一个类实现 `__call__` 方法即可。

```python
class Timer:
    def __init__(self, func):
        self._func = func
    def __call__(self, *args, **kwargs):
        before = time()
        result = self._func(*args, **kwargs)
        after = time()
        print("elapsed: ", after - before)
        return result

@Timer
def add(x, y=10):
    """Add two numbers"""
    return x + y
```

也就是说把类的构造函数当成了一个装饰器，它接受一个函数作为参数，并返回了一个对象，而由于对象实现了 `__call__` 方法，因此返回的对象相当于返回了一个函数。因此该类的构造函数就是一个装饰器。

