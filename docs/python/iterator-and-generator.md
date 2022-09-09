# 可迭代对象、迭代器和生成器

## 迭代器

实现了下列两个方法的对象称为**迭代器（iterator）**，这两个方法共同组成了**迭代器协议**。

### \__iter__()

返回迭代器对象本身。

### \__next__()

从迭代器中返回下一项。如果已经没有项可返回，则会引发 `StopIteration` 异常。

> 参见内置函数 [`iter()`](./standard-library#iter()), [`next()`](./standard-library#next())。

下面实现了一个简单的迭代器：

```python
>>> class CustomIterator(object):
    def __init__(self):
        self._i = 0
    def __iter__(self):
        return self
    def __next__(self):
        if self._i < 5:
            self._i += 1
            return self._i - 1
        else:
            raise StopIteration()
... 
>>> it = CustomIterator()
>>> it.__next__()
0
>>> it.__next__()
1
>>> it.__next__()
2
>>> it.__next__()
3
>>> it.__next__()
4
>>> it.__next__()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 1, in __next__
StopIteration
```

## 可迭代对象

实现了以下方法的（容器）对象称为**可迭代对象（iterable）**，可迭代对象用于 [`for` 语句](./control-flow.md#for 语句)可以迭代容器中的元素。

### \__iter__()

返回一个迭代器对象，即该对象支持上文所述的迭代器协议。如果容器支持不同的迭代类型，则可以提供额外的方法来专门地请求不同迭代类型的迭代器（例如同时支持深度优先和广度优先的遍历）。

利用前面实现的迭代器实现一个简单的可迭代对象：

```python
>>> class CustomIterable(object):
    def __iter__(self):
        return CustomIterator()
... 
>>> itab = CustomIterable()
>>> for i in itab:
...   print(i)
... 
0
1
2
3
4
```

迭代器对象实现了 `__iter__()` 方法，因此也是可迭代对象，同样可以用于 `for` 语句：

```python
>>> it = CustomIterator()
>>> for i in it:
...   print(i)
... 
0
1
2
3
4
```

### 内置类型中的可迭代对象

内置的序列、集合、映射等类型的实例都是可迭代对象，例如：

```python
>>> for char in 'abc':
...   print(char)
... 
a
b
c
>>> for element in [1, 2, 3]:   # or (1, 2, 3), range(1, 4)
...   print(element)
... 
1
2
3
>>> for key in {'a': 1, 'b': 2, 'c': 3}:
...   print(key)
... 
a
b
c
>>> for element in set([1, 2, 3]):
...   print(element)
... 
1
2
3
```

查看列表的迭代器：

```python
>>> it = [1, 2, 3].__iter__()
>>> type(it)
<class 'list_iterator'>
>>> it.__next__()
1
>>> it.__next__()
2
>>> it.__next__()
3
>>> it.__next__()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
StopIteration
```

## `itertools` 创建常用迭代器

`itertools` 模块实现了一系列迭代器，意在为 Python 创建简洁、快速、高效利用内存的核心工具集。

### 总览

**无穷迭代器**

| 迭代器                  | 实参          | 结果                                  | 示例                                    |
| :---------------------- | :------------ | :------------------------------------ | :-------------------------------------- |
| [`count()`](#count())   | start, [step] | start, start+step, start+2*step, ...  | `count(10) --> 10 11 12 13 14 ...`      |
| [`cycle()`](#cycle())   | p             | p0, p1, ... plast, p0, p1, ...        | `cycle('ABCD') --> A B C D A B C D ...` |
| [`repeat()`](#repeat()) | elem [,n]     | elem, elem, elem, ... 重复无限次或n次 | `repeat(10, 3) --> 10 10 10`            |

**根据最短输入序列长度停止的迭代器**

| 迭代器                                           | 实参                        | 结果                                             | 示例                                                       |
| :----------------------------------------------- | :-------------------------- | :----------------------------------------------- | :--------------------------------------------------------- |
| [`accumulate()`](#accumulate())                  | p [,func]                   | p0, p0+p1, p0+p1+p2, ...                         | `accumulate([1,2,3,4,5]) --> 1 3 6 10 15`                  |
| [`chain()`](#chain())                            | p, q, ...                   | p0, p1, ... plast, q0, q1, ...                   | `chain('ABC', 'DEF') --> A B C D E F`                      |
| [`chain.from_iterable()`](chain.from_iterable()) | iterable                    | p0, p1, ... plast, q0, q1, ...                   | `chain.from_iterable(['ABC', 'DEF']) --> A B C D E F`      |
| [`compress()`](compress())                       | data, selectors             | (d[0] if s[0]), (d[1] if s[1]), ...              | `compress('ABCDEF', [1,0,1,0,1,1]) --> A C E F`            |
| [`dropwhile()`](dropwhile())                     | pred, seq                   | seq[n], seq[n+1], ... 从pred首次真值测试失败开始 | `dropwhile(lambda x: x<5, [1,4,6,4,1]) --> 6 4 1`          |
| [`filterfalse()`](filterfalse())                 | pred, seq                   | seq中pred(x)为假值的元素，x是seq中的元素。       | `filterfalse(lambda x: x%2, range(10)) --> 0 2 4 6 8`      |
| [`groupby()`](groupby())                         | iterable[, key]             | 根据key(v)值分组的迭代器                         |                                                            |
| [`islice()`](islice())                           | seq, [start,] stop [, step] | seq[start:stop:step]中的元素                     | `islice('ABCDEFG', 2, None) --> C D E F G`                 |
| [`starmap()`](starmap())                         | func, seq                   | func(\*seq[0]), func(\*seq[1]), ...              | `starmap(pow, [(2,5), (3,2), (10,3)]) --> 32 9 1000`       |
| [`takewhile()`](takewhile())                     | pred, seq                   | seq[0], seq[1], ..., 直到pred真值测试失败        | `takewhile(lambda x: x<5, [1,4,6,4,1]) --> 1 4`            |
| [`tee()`](tee())                                 | it, n                       | it1, it2, ... itn 将一个迭代器拆分为n个迭代器    |                                                            |
| [`zip_longest()`](zip_longest())                 | p, q, ...                   | (p[0], q[0]), (p[1], q[1]), ...                  | `zip_longest('ABCD', 'xy', fillvalue='-') --> Ax By C- D-` |

**排列组合迭代器**

| 迭代器                                                       | 实参                 | 结果                                  |
| :----------------------------------------------------------- | :------------------- | :------------------------------------ |
| [`product()`](#product())                                    | p, q, ... [repeat=1] | 笛卡尔积，相当于嵌套的for循环         |
| [`permutations()`](#permutations())                          | p[, r]               | 长度r元组，所有可能的排列，无重复元素 |
| [`combinations()`](#combinations())                          | p, r                 | 长度r元组，有序，无重复元素           |
| [`combinations_with_replacement()`](#combinations_with_replacement()) | p, r                 | 长度r元组，有序，元素可重复           |

| 示例                                       | 结果                                              |
| :----------------------------------------- | :------------------------------------------------ |
| `product('ABCD', repeat=2)`                | `AA AB AC AD BA BB BC BD CA CB CC CD DA DB DC DD` |
| `permutations('ABCD', 2)`                  | `AB AC AD BA BC BD CA CB CD DA DB DC`             |
| `combinations('ABCD', 2)`                  | `AB AC AD BC BD CD`                               |
| `combinations_with_replacement('ABCD', 2)` | `AA AB AC AD BB BC BD CC CD DD`                   |

### chain()

创建一个迭代器，它首先返回第一个可迭代对象中所有元素，接着返回下一个可迭代对象中所有元素，直到耗尽所有可迭代对象。大致相当于：

```python
def chain(*iterables):
    for it in iterables:
        for element in it:
            yield element
            
# chain('ABC', 'DEF') --> A B C D E F
```

### chain.from_iterable()

类似于 `chain()`，但是接受的参数是包含多个可迭代对象元素的可迭代对象。大致相当于：

```python
def from_iterable(iterables):
    for it in iterables:
        for element in it:
            yield element
            
# chain.from_iterable(['ABC', 'DEF']) --> A B C D E F
```

### combinations()

创建一个迭代器，它返回由输入可迭代对象中的元素组合为长度为 `r` 的所有子序列。大致相当于：

```python
def combinations(iterable, r):
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = list(range(r))
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i+1, r):
            indices[j] = indices[j-1] + 1
        yield tuple(pool[i] for i in indices)
        
# combinations('ABCD', 2) --> AB AC AD BC BD CD,   AB for tuple ('A', 'B')
# combinations(range(4), 3) --> 012 013 023 123,   012 for tuple (0, 1, 2)
```

### combinations_with_replacement()

创建一个迭代器，它返回由输入可迭代对象中的元素组合为长度为 `r` 的所有子序列，允许每个元素重复出现。大致相当于：

```python
def combinations_with_replacement(iterable, r):
    pool = tuple(iterable)
    n = len(pool)
    if not n and r:
        return
    indices = [0] * r
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != n - 1:
                break
        else:
            return
        indices[i:] = [indices[i] + 1] * (r - i)
        yield tuple(pool[i] for i in indices)
        
# combinations_with_replacement('ABC', 2) --> AA AB AC BB BC CC
```

### count()

创建一个迭代器，它从 `start` 值开始，返回一个等差数列。大致相当于：

```python
def count(start=0, step=1):
    n = start
    while True:
        yield n
        n += step
        
# count(2.5, 0.5) -> 2.5 3.0 3.5 ...
```

### cycle()

创建一个迭代器，返回 `iterable` 中所有元素并保存一个副本。当遍历完 `iterable` 中所有元素后，返回副本中的元素，无限重复。大致相当于：

```python
def cycle(iterable):
    saved = []
    for element in iterable:
        yield element
        saved.append(element)
    while saved:
        for element in saved:
              yield element
                
# cycle('ABCD') --> A B C D A B C D A B C D ...
```

### islice()

创建一个迭代器，返回从可迭代对象以切片方式选中的元素。与普通的切片不同，`islice()` 不支持将 `start,stop,step` 设为负值。大致相当于：

```python
def islice(iterable, *args):
    s = slice(*args)
    start, stop, step = s.start or 0, s.stop or sys.maxsize, s.step or 1
    it = iter(range(start, stop, step))
    try:
        nexti = next(it)
    except StopIteration:
        # Consume *iterable* up to the *start* position.
        for i, element in zip(range(start), iterable):
            pass
        return
    try:
        for i, element in enumerate(iterable):
            if i == nexti:
                yield element
                nexti = next(it)
    except StopIteration:
        # Consume to *stop*.
        for i, element in zip(range(i + 1, stop), iterable):
            pass
          
# islice('ABCDEFG', 2) --> A B                        stop = 2
# islice('ABCDEFG', 2, 4) --> C D                     start = 2, stop = 4
# islice('ABCDEFG', 2, None) --> C D E F G            start = 2, stop = None
# islice('ABCDEFG', 0, None, 2) --> A C E G           start = 0, stop = None, step = 2
```

### permutations()

创建一个迭代器，它返回由输入可迭代对象中的元素生成的长度为 `r` 的所有排列。大致相当于：

```python
def permutations(iterable, r=None):
    pool = tuple(iterable)
    n = len(pool)
    r = n if r is None else r
    if r > n:
        return
    indices = list(range(n))
    cycles = list(range(n, n-r, -1))
    yield tuple(pool[i] for i in indices[:r])
    while n:
        for i in reversed(range(r)):
            cycles[i] -= 1
            if cycles[i] == 0:
                indices[i:] = indices[i+1:] + indices[i:i+1]
                cycles[i] = n - i
            else:
                j = cycles[i]
                indices[i], indices[-j] = indices[-j], indices[i]
                yield tuple(pool[i] for i in indices[:r])
                break
        else:
            return
          
# permutations('ABCD', 2) --> AB AC AD BA BC BD CA CB CD DA DB DC,   AB for tuple ('A', 'B')
# permutations(range(3)) --> 012 021 102 120 201 210,   012 for tuple (0, 1, 2)
```

### product()

多个输入可迭代对象的笛卡尔积。大致相当于元组推导式的嵌套循环，例如 `product(A,B)` 和 `((x,y)for x in A for y in B)` 返回结果一样。

要计算可迭代对象自身的笛卡尔积，将可选参数 *repeat* 设定为要重复的次数。例如 `product(A,repeat=4)` 和 `product(A,A,A,A)` 是一样的。

```python
# product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy
# product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111
```

### repeat()

创建一个迭代器，不断重复 *object*。设定参数 *times* 将会重复如此多次，否则将无限重复。大致相当于：

```python
def repeat(object, times=None):
    if times is None:
        while True:
            yield object
    else:
        for i in range(times):
            yield object
            
# repeat(10, 3) --> 10 10 10            
```

*repeat* 最常见的用途就是在 *map* 或 *zip* 提供一个常量流：

```python
>>> list(map(pow, range(10), repeat(2)))
[0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
```

# 生成器

**生成器（generator）**提供了一种实现迭代器协议的便捷方式，因此可以视作一种特殊的迭代器。

函数体中包含 `yield` 语句的函数称为生成器函数，调用生成器函数将返回一个生成器对象，用于控制该函数的执行。当生成器的某个方法被调用时，生成器函数开始执行，直到遇到一个 `yield` 语句，在此执行被挂起，调用的生成器方法返回 `yield` 语句返回的对象。挂起时，生成器函数内部的所有局部状态都被保留下来，直到生成器的某个方法再次被调用，生成器函数才继续执行。生成器函数返回时（函数体执行完毕或遇到 `return` 语句），将引发一个 `StopIteration` 异常。在生成器函数的执行过程中，`yield` 表达式就如同是调用一个外部函数。恢复执行时 `yield` 表达式的返回值取决于生成器调用的方法。

如果生成器在生成器函数的执行被挂起（还没有返回）的状态下被销毁（因为引用计数到零或是因为被垃圾回收），则它的 `close()` 方法将被调用。

如果（容器）对象的 `__iter__()` 方法被实现为一个生成器，它将自动返回一个生成器对象，该对象提供 `__iter__()` 和 `__next__()` 方法。

## 方法

### \__next__()

开始执行生成器函数或从上次执行到的 `yield` 语句位置恢复执行，恢复执行时当前 `yield` 表达式返回 `None`，随后继续执行到下一个 `yield` 语句，返回 `yield` 语句返回的对象。如果生成器函数返回（执行完毕或遇到 `return` 语句），则引发 `StopIteration` 异常。

此方法通常是隐式地调用，例如通过 `for` 循环或内置函数 `next()`。

```python
>>> def natural_number_square(max):
    n = 0
    while n <= max:
        yield n * n
        n += 1
... 
>>> ge = natural_number_square(4)
>>> ge.__next__()
0
>>> ge.__next__()
1
>>> ge.__next__()
4
>>> ge.__next__()
9
>>> ge.__next__()
16
>>> ge.__next__()      # 函数返回
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
StopIteration
```

### send()

恢复执行并向生成器函数发送一个值，作为当前 `yield` 表达式的返回值，随后继续执行到下一个 `yield` 语句，返回 `yield` 语句返回的对象。如果生成器函数返回（执行完毕或遇到 `return` 语句），则引发 `StopIteration` 异常。当调用此方法开始执行生成器函数时，发送的值必须为 `None`，因为这时还没有可以接收值的 `yield` 表达式。

```python
>>> def natural_number_square(max):
    n = 0
    while n <= max:
        r = (yield n * n)
        print('Received {}'.format(r))
        n += 1
>>> ge = natural_number_square(4)
>>> ge.send(None)      # 首次调用时发送`None`
0
>>> ge.send('a')       # 发送的值被`yield`表达式返回
Received a
1
>>> ge.__next__()      # 调用`__next__()`方法时`yield`表达式返回`None`
Received None
4
>>> ge.send('c')
Received c
9
>>> ge.__next__()
Received None
16
>>> ge.send('e')       # 函数返回
Received e
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
StopIteration
```

### throw()

```python
generator.throw(type[, value[, traceback]])
```

恢复执行并令当前 `yield` 表达式引发 *type* 类型的异常，随后继续执行到下一个 `yield` 语句，返回 `yield` 语句返回的对象。如果生成器函数返回（执行完毕或遇到 `return` 语句），则引发 `StopIteration` 异常。如果生成器函数没有捕获引发的异常，则上抛该异常至此方法的调用者。

```python
>>> def natural_number_square(max):
    n = 0
    while n <= max:
        try:
            yield n * n
        except RuntimeError as e:
            print(e)    
        n += 1
... 
>>> ge = natural_number_square(4)
>>> ge.__next__()
0
>>> ge.__next__()
1
>>> ge.throw(RuntimeError, 'An runtime error raised')  # 引发一个`RuntimeError`并被捕获
An runtime error raised
4
>>> ge.throw(ValueError, 'A value error raised')       # 引发一个`ValueError`但没有被捕获
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 1, in natural_number_square
ValueError: A value error raised
```

### close()

恢复执行并令当前 `yield` 表达式引发一个 `GeneratorExit`。如果生成器函数随后退出、关闭或没有捕获 `GeneratorExit`，则此方法正常返回；如果生成器函数继续执行到下一个 `yield` 语句，则引发一个 `RuntimeError`；如果生成器函数引发了其它类型的异常，则上抛该异常至此方法的调用者。如果生成器已经正常退出或由于异常退出，则此方法不会执行任何操作。

```python
>>> def natural_number_square(max):
    n = 0
    while n <= max:
        try:
            yield n * n
        except RuntimeError as e:
            print(e)    
        n += 1
... 
>>> ge = natural_number_square(4)
>>> ge.__next__()
0
>>> ge.close()        # 生成器关闭
>>> ge.__next__()     # 再执行生成器函数将引发`StopIteration`异常
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
StopIteration
>>> ge.close()        # 不执行任何操作
```

## 生成器表达式

```python
>>> a = [1, 2, 3, 4]
>>> b = (2 * x for x in a)       # 返回一个generator,一般形式为
>>> b                            # (<expression> for i in s if <conditional>)
<generator object at 0x58760>
>>> for i in b:
...   print(i, end=' ')
...
2 4 6 8
```

## 生成器generator

生成器（generator）就是使用`yield`语句的函数。生成器和一般函数的区别在于执行流程不同：一般函数顺序执行，遇到`return`语句或者最后一条语句返回；生成器在每次调用`next()`的时候执行，遇到`yield`语句返回，再次执行时从上次返回的`yield`语句处继续执行。

下面的例子展示了一个生成器，其返回斐波那契数列的前若干项：

```python
def fib(max):	   # generator型函数
    n, a, b = 0, 0, 1
    while n < max:
        yield b
        a, b = b, a + b
        n = n + 1
    return 'done'

ge = fib(10)     # 每次调用generator型函数返回一个generator
print(repr(ge))  # <generator object fib at 0x7fcd124c14c0>

for n in ge:
  print(n, end=' ')    # 1 1 2 3 5 8 13 21 34 55
                       # 迭代完成后generator即失去作用,不可重用
```

## 异步生成器

## 流水线

使用生成器可以构造数据处理的流水线：

*producer* → *processing* → *consumer*

```python
def producer():
    pass
    yield item          # yields the item that is received by the `processing`

def processing(s):
    for item in s:      # Comes from the `producer`
        pass
        yield newitem   # yields a new item

def consumer(s):
    for item in s:      # Comes from the `processing`
        pass
        
a = producer()
b = processing(a)
c = consumer(b)        
```
