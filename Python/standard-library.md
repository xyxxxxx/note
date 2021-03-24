[toc]



# [内置函数](https://docs.python.org/3/library/functions.html)

## abs()

返回一个数的绝对值。 参数可以是整数、浮点数或任何实现了 \_\_abs\_\_() 的对象。 如果参数是一个复数，则返回它的模。

```python
>>> abs(-1)
1
>>> abs(-1.2)
1.2
>>> abs(complex(1, 2))
2.23606797749979
```



## all()

如果 *iterable* 的所有元素均为真值（或可迭代对象为空）则返回 `True` 。

```python
>>> all(range(5))      # include 0
False
>>> all(range(1, 5))
True
>>> all([])
True
```



## any()

如果 *iterable* 的任一元素为真值则返回 `True`。 如果可迭代对象为空，返回 `False`。

```python
>>> any(range(5))
True
>>> any([])
False
```



## bin()

将整数转换为前缀为“0b”的二进制字符串。

```python
>>> bin(3)
'0b11'
>>> bin(-10)
'-0b1010'
```



## bool()



## chr()

返回编码为输入整数的单个Unicode字符。是`ord()`的反函数。

```python
>>> chr(0x4e2d)
'中'
```



## @classmethod

将方法封装为类方法。

参见面向对象编程-方法。



## complex()

返回值为 x+iy 的复数，或者将字符串或数字转换为复数。

```python
>>> complex(1, 2)
(1+2j)                   # j表示虚数单位
>>> complex(1)
(1+0j)
>>> complex(1j, 2j)
(-2+1j)
>>> complex('1+2j')
(1+2j)
```

对于一个普通 Python 对象 `x`，`complex(x)` 会委托给 `x.__complex__(`)。 如果 `__complex__()` 未定义则将回退至 `__float__()`。 如果 `__float__(`) 未定义则将回退至 `__index__()`。



## dict()

创建一个新的字典。参见数据结构-字典。



## dir()



## divmod()

将两个数字（整数或浮点数）作为实参，执行整数除法并返回一对商和余数。对于整数，结果和 `(a // b, a % b)` 一致。对于浮点数，结果是 `(q, a % b)` ，`q` 通常是 `math.floor(a / b)`，但可能会比 1 小。

```python
>>> divmod(5, 3)
(1, 2)
>>> divmod(5.0, 1.5)
(3.0, 0.5)
```



## enumerate



## eval()



## exec()



## filter()





## float()



## getattr()



## hasattr()



## hash()

返回对象的哈希值（如果它有的话）。



## hex()

将整数转换为前缀为“0x”的小写十六进制字符串。

```python
>>> hex(255)
'0xff'
>>> hex(-42)
'-0x2a'
```



## int()



## isinstance()

如果 *object* 是 *classinfo* 的实例或（直接、间接或虚拟）子类则返回 `True`。 *classinfo* 可以是类对象的元组，在此情况下 *object* 是其中任何一个类的实例就返回 `True`。

参见面向对象编程-获取对象信息。



## issubclass()

如果 *class* 是 *classinfo* 的（直接、间接或虚拟）子类则返回 `True`。 类会被视作其自身的子类。*classinfo* 可以是类对象的元组，在此情况下 *classinfo* 中的每个条目都将被检查。



## len()

返回对象的长度（元素个数）。实参可以是序列（如 string、bytes、tuple、list 或 range 等）或集合（如 dictionary、set 或 frozen set 等）。



## map()



## max(), min()

返回可迭代对象中的最大/最小元素，或者返回两个及以上实参中的最大/最小者。

```python
>>> max(range(10))
9
>>> max(1, 2, 3)
3
>>> min(range(10))
0
>>> min(1, 2, 3)
1
```



## next()





## oct()



## open()

参见



## ord()

返回单个Unicode字符的编码的十进制整数表示。是`chr()`的反函数。

```python
>>> ord('中')
20013
>>> hex(ord('中'))
'0x4e2d'
```



## pow()

返回 *base* 的 *exp* 次幂；如果 *mod* 存在，则返回 *base* 的 *exp* 次幂对 *mod* 取余（比 `pow(base, exp) % mod` 更高效）。两参数形式 `pow(base, exp)` 等价于乘方运算符: `base**exp`。

```python
>>> pow(37, 2, mod=97)
11
```



## print()

参见



## property





## sorted()

根据可迭代对象中的项返回一个新的已排序列表。

```python

```





## @staticmethod

将方法转换为静态方法。

参见面向对象编程-方法。



## str()

返回一个`str`对象，`str`是内置字符串类型。

更多关于字符串的详细信息参见数据类型-str。



## sum()

从 *start* 开始自左向右对可迭代对象的项求和并返回总计值。可迭代对象的项通常为数字，而 *start* 值则不允许为字符串。

要拼接字符串序列，更好的方式是调用`''.join(sequence)`；要以扩展精度对浮点值求和，请使用`math.fsum()`；要拼接一系列可迭代对象，请使用`itertools.chain()`。



## super()



## tuple()

创建一个新的元组。参见数据结构-元组。



## zip()

创建一个迭代器，它返回的第 *i* 个元组包含来自每个输入可迭代对象的第 *i* 个元素。 当所输入可迭代对象中最短的一个被耗尽时，迭代器将停止迭代。

```python
def zip(*iterables):
    sentinel = object()
    iterators = [iter(it) for it in iterables]
    while iterators:
        result = []
        for it in iterators:
            elem = next(it, sentinel)
            if elem is sentinel:
                return
            result.append(elem)
        yield tuple(result)
        
# zip('ABCD', 'xy') --> Ax By        
```





# [argparse](https://docs.python.org/zh-cn/3/library/argparse.html)——命令行选项、参数和子命令解析器

如果脚本很简单或者临时使用，可以使用`sys.argv`直接读取命令行参数。`sys.argv`返回一个参数列表，其中首个元素是程序名，随后是命令行参数，所有元素都是字符串类型。例如以下脚本：

```python
# test.py

import sys

print("Input argument is %s" %(sys.argv))
```

```shell
$ python3 test.py 1 2 -a 3
Input argument is ['test.py', '1', '2', '-a', '3']
```



`argparse`模块可以让人轻松编写用户友好的命令行接口。程序定义它需要的参数，然后 `argparse` 将弄清如何从 `sys.argv` 解析出那些参数。 `argparse` 模块还会自动生成帮助和使用手册，并在用户给程序传入无效参数时报出错误信息。

教程：[Argparse 教程](https://docs.python.org/zh-cn/3/howto/argparse.html)

文档：[argparse --- 命令行选项、参数和子命令解析器](https://docs.python.org/zh-cn/3.9/library/argparse.html)

```python
# 简单的argparse实例
import argparse
parser = argparse.ArgumentParser()
# 位置参数, type表示解析类型, 默认为str
parser.add_argument("square", type=int,
                    help="display a square of a given number")
# 可选参数, 可以设置短选项, action="count"表示计数参数的出现次数
parser.add_argument("-v", "--verbosity", action="count", default=0,
                    help="increase output verbosity")
# 进行参数解析
args = parser.parse_args()
answer = args.square**2
if args.verbosity >= 2:
    print("the square of {} equals {}".format(args.square, answer))
elif args.verbosity >= 1:
    print("{}^2 == {}".format(args.square, answer))
else:
    print(answer)
```

```python
# add_argument实例

parser.add_argument('-m', '--model', nargs='*', choices=['NB', 'LR', 'SVML'], default=['NB'], help="model used to classify spam and ham")
# 限定范围多选, 有默认值

parser.add_argument('-s', '--stopwords', nargs='?', default=False, const=True, help="model used to classify spam and ham")
# default为选项未出现时取值, const为选项后没有参数时的取值
# 因此-s表示True, 没有该选项表示False
```







# [datetime](https://docs.python.org/zh-cn/3/library/datetime.html)——处理日期和时间

## timedelta

```python
import datetime
from datetime import timedelta

>>> delta = timedelta(
...     weeks=2,              # 1星期转换成7天
...     days=50,
...     hours=8,              # 1小时转换成3600秒
...     minutes=5,            # 1小时转换成60秒
...     seconds=27,
...     milliseconds=29000,   # 1毫秒转换成1000微秒
...     microseconds=10
... )
>>> delta
datetime.timedelta(64, 29156, 10)   # 日,秒,毫秒
>>> delta.total_seconds()
5558756.00001                       # 秒

>>> d1 = timedelta(minutes=5)
>>> d2 = timedelta(seconds=20)
>>> d1 + d2
datetime.timedelta(0, 320)
>>> d1 - d2
datetime.timedelta(0, 280)
>>> d1 * 2
datetime.timedelta(0, 600)
>>> d1 / d2
15.0
>>> d1 // d2
15
```



## date

```python
import datetime
from datetime import date

>>> date(
...     year=2020,
...     month=11,
...     day=27
... )
datetime.date(2020, 11, 27)
>>> date.today()
datetime.date(2020, 11, 27)
>>> date.fromtimestamp(1606468517.547344)
datetime.date(2020, 11, 27)
>>> date.today().weekday()
4                              # Friday
>>> date.today().isoweekday()
5                              # Friday
```



## datetime

```python
import datetime
from datetime import datetime, timedelta

>>> datetime(
...     year=2020,
...     month=11,
...     day=27,
...     hour=17,
...     minute=15,
...     second=17,
...     microsecond=547344
... )
datetime.datetime(2020, 11, 27, 17, 15, 17, 547344)
>>> datetime.now()             # 返回当地当前的datetime
datetime.datetime(2020, 11, 27, 17, 15, 17, 547344)
>>> datetime.utcnow()          # 返回当前的UTC datetime
datetime.datetime(2020, 11, 27, 9, 15, 17, 547344)
>>> datetime.now().timestamp()                  # 转换为timestamp
1606468517.547344
>>> datetime.fromtimestamp(1606468517.547344)   # 转换自timestamp
datetime.datetime(2020, 11, 27, 17, 15, 17, 547344)
>>> d1 = timedelta(minutes=5)
>>> datetime.now() + d1
datetime.datetime(2020, 11, 27, 17, 20, 17, 547344)
```



# functools——高阶函数和可调用对象上的操作

## partial()

返回一个新的partial对象，当被调用时其行为类似于 *func* 附带位置参数 *args* 和关键字参数 *keywords* 被调用。如果为调用提供了更多的参数，它们会被附加到 *args*。 如果提供了额外的关键字参数，它们会扩展并重载 *keywords*。 大致等价于:

```python
def partial(func, /, *args, **keywords):
    def newfunc(*fargs, **fkeywords):
        newkeywords = {**keywords, **fkeywords}
        return func(*args, *fargs, **newkeywords)
    newfunc.func = func
    newfunc.args = args
    newfunc.keywords = keywords
    return newfunc
```

常用于冻结一部分函数参数并应用：

```python
>>> import functools
>>> int2 = functools.partial(int, base=2)  # 附带关键字参数base=2调用int
>>> int2('1000000')                        # 相当于int('1000000', base=2)
64
>>> int2('1000000', base=10)               # 重载了base=2
1000000
```



## reduce()

将有两个参数的函数从左到右依次应用到可迭代对象的所有元素上，返回一个最终值。

```python
>>> from functools import reduce
>>> def fn(x, y):
...   return x * 10 + y
...
>>> reduce(fn, [1, 3, 5, 7, 9])   # reduce()将多元函数依次作用在序列上
13579
```





# itertools——为高效循环而创建迭代器的函数

## 总览

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



## chain()

创建一个迭代器，它首先返回第一个可迭代对象中所有元素，接着返回下一个可迭代对象中所有元素，直到耗尽所有可迭代对象。大致相当于：

```python
def chain(*iterables):
    for it in iterables:
        for element in it:
            yield element
            
# chain('ABC', 'DEF') --> A B C D E F
```



## chain.from_iterable()

类似于`chain()`，但是接受的参数是包含多个可迭代对象元素的可迭代对象。大致相当于：

```python
def from_iterable(iterables):
    for it in iterables:
        for element in it:
            yield element
            
# chain.from_iterable(['ABC', 'DEF']) --> A B C D E F
```



## combinations()

创建一个迭代器，它返回由输入可迭代对象中的元素组合为长度为`r`的所有子序列。大致相当于：

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



## combinations_with_replacement()

创建一个迭代器，它返回由输入可迭代对象中的元素组合为长度为`r`的所有子序列，允许每个元素重复出现。大致相当于：

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



## count()

创建一个迭代器，它从`start`值开始，返回一个等差数列。大致相当于：

```python
def count(start=0, step=1):
    n = start
    while True:
        yield n
        n += step
        
# count(2.5, 0.5) -> 2.5 3.0 3.5 ...
```



## cycle()

创建一个迭代器，返回`iterable`中所有元素并保存一个副本。当遍历完`iterable`中所有元素后，返回副本中的元素，无限重复。大致相当于：

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



## islice()

创建一个迭代器，返回从可迭代对象以切片方式选中的元素。与普通的切片不同，`islice()`不支持将 start, stop, step 设为负值。大致相当于：

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
          
# islice('ABCDEFG', 2) --> A B
# islice('ABCDEFG', 2, 4) --> C D
# islice('ABCDEFG', 2, None) --> C D E F G
# islice('ABCDEFG', 0, None, 2) --> A C E G
```



## permutations()

创建一个迭代器，它返回由输入可迭代对象中的元素生成的长度为`r`的所有排列。大致相当于：

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



## product()

多个输入可迭代对象的笛卡尔积。大致相当于元组推导式的嵌套循环，例如 `product(A, B)` 和 `((x,y) for x in A for y in B)` 返回结果一样。

要计算可迭代对象自身的笛卡尔积，将可选参数 *repeat* 设定为要重复的次数。例如 `product(A, repeat=4)` 和 `product(A, A, A, A)` 是一样的。

```python
# product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy
# product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111
```



## repeat()

创建一个迭代器，不断重复 *object* 。设定参数 *times* 将会重复如此多次 ，否则将无限重复。大致相当于：

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





# math——数学函数

该模块提供了对C标准定义的数学函数的访问。

## atan2()

计算向量 `(x, y)` 与x轴正方向的夹角，结果在 `-pi` 和 `pi` 之间。

```python
>>> math.atan2(10, 1)  # y=10,x=1
1.4711276743037347
>>> math.atan2(-1, -1) # y=-1,x=-1
-2.356194490192345
```



## ceil()

向上取整。

```python
>>> math.ceil(-0.5)
0
>>> math.ceil(0)
0
>>> math.ceil(0.5)
1
```



## comb()

组合数。

```python
>>> math.comb(6, 2)
15
```



## degrees(), radians()

角度和弧度互相转换。

```python
>>> math.degrees(math.pi)
180.0
>>> math.radians(180.0)
3.141592653589793
```



## dist()

欧几里得距离。

```python
>>> math.dist([1, 1, 1], [0, 0, 0])
1.7320508075688772
```

3.8版本新功能。



## e

自然对数底数，精确到可用精度。

```python
>>> math.e
2.718281828459045
```



## exp()

（底数为e的）指数函数。

```python
>>> math.exp(1)
2.718281828459045
```



## fabs()

绝对值。

```python
>>> math.fabs(-1)
1.0                   # 返回浮点数
```



## factorial()

阶乘。

```python
>>> math.factorial(5)
120
```



## floor()

向下取整。

```python
>>> math.floor(-0.5)
-1
>>> math.floor(0)
0
>>> math.floor(0.5)
0
```



## fmod()

余数。整数计算时推荐使用`x % y`，浮点数计算时推荐使用`fmod()`。

```python
>>> math.fmod(5.0, 1.5)
0.5
```



## fsum()

计算可迭代对象的所有元素（整数或浮点数）的和。通过跟踪多个中间部分和来避免精度损失。

```python
>>> sum([.1, .1, .1, .1, .1, .1, .1, .1, .1, .1])
0.9999999999999999
>>> math.fsum([.1, .1, .1, .1, .1, .1, .1, .1, .1, .1])
1.0
```



## gcd()

最大公约数。

```python
>>> math.gcd(20, 48)
4
>>> math.gcd(20, -48)
4
>>> math.gcd(20, 0)
20
>>> math.gcd(20, 1)
1
```

在3.9之后的版本可以传入任意个整数参数，之前的版本只能传入两个整数参数。



## hypot()

欧几里得范数，即点到原点的欧几里得距离。

```python
>>> math.hypot(1., 1, 1)
1.7320508075688772
```

在3.8之后的版本可以传入任意个实数参数，之前的版本只能传入两个实数参数。



## inf

浮点正无穷大，相当于`float('inf')`的返回值。浮点负无穷大用`-math.inf`表示。

```python
>>> math.inf
inf
>>> float('inf')
inf
```



## isclose()

若两个浮点数的值非常接近则返回`True`，否则返回`False`。

```python
# 默认的相对容差为1e-9,绝对容差为0.0
# 相对容差或绝对容差小于给定值时认为非常接近
>>> math.isclose(1e10, 1e10+1, rel_tol=1e-9, abs_tol=0.0)  # 相对容差为1e-10,小于1e-9
True
>>> math.isclose(1e8, 1e8+1)                               # 相对容差为1e-8,大于1e-9
False
>>> math.isclose(1e8, 1e8+1, abs_tol=2)                    # 绝对容差为1,小于2
True
```



## isfinite()

若参数值既不是无穷大又不是`NaN`，则返回`True`，否则返回`False`。

```python
>>> math.isfinite(0.0)
True
>>> math.isfinite(math.inf)
False
```



## isnan()

若参数值是非数字（NaN）值，则返回`True`，否则返回`False`。

```python
>>> math.isnan(0.0)
False
>>> math.isnan(math.nan)
True
```



## isqrt()

平方根向下取整。

```python
>>> math.isqrt(9)
3
>>> math.isqrt(10)
3
```

平方根向上取整可以使用`1 + isqrt(n - 1)`。



## lcm()

最大公倍数。

3.9版本新功能。



## log(), log2(), log10()

对数函数。

```python
>>> math.log(10)      # 自然对数
2.302585092994046
>>> math.log(10, 2)   # 以2为底
3.3219280948873626
>>> math.log2(10)
3.321928094887362
>>> math.log(10, 10)  # 以10为底
1.0
>>> math.log10(10)
1.0
```



## modf()

返回浮点数参数的小数和整数部分，两个结果都是浮点数并且与参数同号。

```python
>>> math.modf(0.0)
(0.0, 0.0)
>>> math.modf(1.0)
(0.0, 1.0)
>>> math.modf(1.1)
(0.10000000000000009, 1.0)
```



## nan

浮点非数字（NaN）值，相当于`float('nan')`的返回值。

```python
>>> math.nan
nan
>>> float('nan')
nan
```



## perm()

排列数。

```python
>>> math.perm(5)
120
>>> math.perm(5, 2)
20
```

3.8版本新功能。



## pi

圆周率，精确到可用精度。

```python
>>> math.pi
3.141592653589793
```



## pow()

幂运算。

```python
>>> math.pow(2, 3)
8.0
>>> math.pow(1.0, 1e10)  # 总是返回1.0
1.0
>>> math.pow(1e10, 0.0)  # 总是返回1.0
1.0
```



## prod()

计算可迭代对象的所有元素（整数或浮点数）的积。积的默认初始值为1。

```python
>>> math.prod(range(1, 6))
120
```

3.8版本新功能。



## remainder()

IEEE 754 风格的余数：对于有限 *x* 和有限非零 *y* ，返回 `x - n * y` ，其中 `n` 是与商 `x / y` 的精确值最接近的整数；如果 `x / y` 恰好位于两个连续整数之间，则 `n`  取最近的偶整数。因此余数 `r = remainder(x, y)` 总是满足 `abs(r) <= 0.5 * abs(y)` 。

```python
>>> math.remainder(10, 3)
1.0
>>> math.remainder(11, 3)
-1.0
```



## sin(), cos(), tan(),  asin(), acos(), atan(), sinh(), cosh(), tanh(), asinh(), acosh(), atanh()

三角函数和双曲函数。

```python
>>> math.sin(math.pi/4)
0.7071067811865475
```



## sqrt()

平方根。

```python
>>> math.sqrt(9)
3.0
>>> math.sqrt(10)
3.1622776601683795
```



## trunc()

将浮点数截断为整数。

```python
>>> math.trunc(1.1)
1
>>> math.trunc(-1.1)
-1
```





# multiprocessing——基于进程的并行

multiprocessing 是一个支持使用与 threading 模块类似的 API 来产生进程的包。 multiprocessing 包同时提供了本地和远程并发操作，通过使用子进程而非线程有效地绕过了全局解释器锁。 因此，multiprocessing 模块允许程序员充分利用给定机器上的多个处理器。 它在 Unix 和 Windows 上均可运行。



## Process

```python
class multiprocessing.Process(group=None, target=None, name=None, args=(), kwargs={}, *, daemon=None)
# target    由run()方法调用的可调用对象
# args      目标调用的顺序参数
# kwargs    目标调用的关键字参数
```

进程对象表示在单独进程中运行的活动。`Process`类拥有和`threading.Thread`等价的大部分方法。

**`run()`**

表示进程活动的方法。

**`start()`**

启动进程活动。

这个方法每个进程对象最多只能调用一次。它会将对象的`run()`方法安排在一个单独的进程中调用。

**`join([timeout])`**

如果可选参数 *timeout* 是 `None` （默认值），则该方法将阻塞，直到调用`join()`方法的进程终止；如果 *timeout* 是一个正数，它最多会阻塞 *timeout* 秒。不管是进程终止还是方法超时，该方法都返回 `None`。

一个进程可以被`join`多次。

进程无法`join`自身，因为这会导致死锁。尝试在启动进程之前`join`进程会产生一个错误。

**`is_alive()`**

返回进程是否处于活动状态。从`start()`方法返回到子进程终止之间，进程对象都处于活动状态。

**`name`**

进程的名称。该名称是一个字符串，仅用于识别，没有具体语义。可以为多个进程指定相同的名称。

**`daemon`**

进程的守护标志，一个布尔值。必须在`start()`被调用之前设置。

初始值继承自创建进程。

当一个进程退出时，它会尝试终止子进程中的所有守护进程。

**`pid`**

返回进程ID。

**`exitcode`**

子进程的退出代码。`None`表示进程尚未终止；负值-N表示子进程被信号N终止。

**`terminate()`**

终止进程。在Unix上由`SIGTERM`信号完成。

**`kill()`**

与`terminate()`相同，但在Unix上使用`SIGKILL`信号。

**`close`()**

关闭`Process`对象，释放与之关联的所有资源。如果底层进程仍在运行，则会引发`ValueError`。一旦`close()`成功返回，`Process`对象的大多数其他方法和属性将引发`ValueError`。



## Pipe



## Connection



## Queue



## cpu_count()

返回系统的CPU数量。



## current_process()

返回当前进程相对应的`Process`对象。



## parent_process()

返回当前进程的父进程相对应的`Process`对象。



## Lock



## RLock



## Semaphore



## Value()



## Array()



## Manager()



## Pool





# [os](https://docs.python.org/zh-cn/3/library/os.html)——多种操作系统接口

## 进程

> 参考[进程和线程](./process-and-thread.md)

```python
>>> os.system('pwd')        # 创建一个shell子进程并执行字符串代表的命令
/Users/xyx
```



## 文件和目录

```python
# test/
#     dir1/
#         file2
#     file1

>>> import os
>>> os.getcwd()             # 当前路径
'/home/test'
>>> os.chdir('dir1')        # 切换路径
>>> os.getcwd()
'/home/test/dir1'
>>> os.listdir()            # ls
['dir1', 'file1']
>>> os.mkdir('dir2')        # 创建目录
>>> os.rename('dir2', 'dir3')  # 重命名目录或文件
>>> os.rmdir('dir2')        # 删除目录
>>> os.remove('file1')      # 删除文件
```



## 环境变量

```python
>>> import os
>>> os.environ
environ({'CLUTTER_IM_MODULE': 'xim', 'LS_COLORS': 
# ...
/usr/bin/lesspipe %s', 'GTK_IM_MODULE': 'fcitx', 'LC_TIME': 'en_US.UTF-8', '_': '/usr/bin/python3'})
>>> os.environ['HOME']                           # 获取环境变量
'/home/xyx' 
>>> os.environ['MASTER_ADDR'] = 'localhost'      # 临时增加/修改环境变量
>>> os.getenv('MASTER_ADDR')                     # 获取环境变量的推荐方式
'localhost'         
```



## walk()

遍历目录。对于以`top`为根的目录树中的每个目录（包括`top`本身）都生成一个三元组`(dirpath, dirnames, filenames)`。

```python
os.walk(top, topdown=True, onerror=None, followlinks=False)

# top      目录树的根目录
# topdown  若为True,则自上而下遍历;若为False,则自下而上遍历
```

```python
# test/
#     file1
#     file2
#     format_correction.py
#     dir1/
#         file3
#         file4
#     dir2/
#         file5
#         file6

>>> import os
>>> for root, dirs, files in os.walk('.'):
...     print(root)
...     print(dirs)
...     print(files)
... 
.                                             # 代表根目录
['dir2', 'dir1']                              # 此目录下的目录
['file2', 'format_correction.py', 'file1']    # 此目录下的文件
./dir2
[]
['file5', 'file6']
./dir1
[]
['file3', 'file4']
>>>
>>> for root, _, files in os.walk('.'):
...     for f in files:
...         path = os.path.join(root, f)      # 返回所有文件的相对路径
...         print(path)
... 
./file2
./format_correction.py
./file1
./dir2/file5
./dir2/file6
./dir1/file3
./dir1/file4
>>>
>>> for root, _, files in os.walk('.', topdown=False):    # 自下而上遍历
...     for f in files:
...         path = os.path.join(root, f)
...         print(path)
... 
./dir2/file5
./dir2/file6
./dir1/file3
./dir1/file4
./file2
./format_correction.py
./file1

```





# [os.path](https://docs.python.org/zh-cn/3/library/os.path.html)——常用路径操作

```python
# test/
#     dir1/
#         file2
#     file1

>>> from os import path
>>> path.abspath('.')      # 路径的绝对路径
'/home/test'
>>> path.exists('./dir1')  # 存在路径
True
>>> path.exists('./dir2')
False
>>> path.isdir('dir1')     # 判断目录
True
>>> path.isdir('dir2')
False
>>> path.isfile('dir1')    # 判断文件
False
>>> path.isfile('file1')
True
>>> path.getsize('file1')  # 文件大小
14560
>>> path.join(path.abspath('.'), 'file1')   # 拼接路径
'/home/test/file1'
```



# re——正则表达式操作

见正则表达式。





# requests——







# subprocess——子进程管理

`subprocess`模块允许我们生成新的进程，连接它们的输入、输出、错误管道，并且获取它们的返回码。

大多数情况下，推荐使用`run()`方法调用子进程，执行操作系统命令。在更高级的使用场景，你还可以使用`Popen`接口。其实`run()`方法在底层调用的就是`Popen`接口。

## run()

```python
subprocess.run(args, *, stdin=None, input=None, stdout=None, stderr=None, shell=False, timeout=None, check=False, encoding=None, errors=None)

# args       要执行的命令.必须是一个字符串或参数列表.
# stdin,stdout,stderr  子进程的标准输入、输出和错误,其值可以是subprocess.PIPE,
#                subprocess.DEVNULL,一个已经存在的文件描述符,已经打开的文件对象或者
#                None.subprocess.PIPE表示为子进程创建新的管道,subprocess.
#                DEVNULL表示使用os.devnull.默认使用的是None,表示什么都不做.
# timeout    命令超时时间.如果命令执行时间超时,子进程将被杀死,并抛出TimeoutExpired异常.
# check      若为True,并且进程退出状态码不是0,则抛出CalledProcessError 异常.
# encoding   若指定了该参数，则stdin,stdout,stderr可以接收字符串数据,并以该编码方
#                式编码.否则只接收bytes类型的数据.
# shell      若为True,将通过操作系统的shell执行指定的命令.
```

```python
>>> import subprocess

>>> subprocess.run(['ls', '-l'])                           # 打印标准输出
total 4
-rw-rw-r-- 1 xyx xyx 2862 Dec  1 17:11 lstm.py
CompletedProcess(args=['ls', '-l'], returncode=0)

>>> subprocess.run(['ls', '-l'], stdout=subprocess.PIPE)   # 捕获标准输出
CompletedProcess(args=['ls', '-l'], returncode=0, stdout=b'total 4\n-rw-rw-r-- 1 xyx xyx 2862 Dec  1 17:11 lstm.py\n')
```

注意当`args`是一个字符串时，必须指定`shell=True`：

```python
>>> subprocess.run('ls -l')
Traceback (most recent call last):
# ...
FileNotFoundError: [Errno 2] No such file or directory: 'ls -l': 'ls -l'
            
>>> subprocess.run('ls -l', shell=True)
total 4
-rw-rw-r-- 1 xyx xyx 2862 Dec  1 17:11 lstm.py
CompletedProcess(args='ls -l', returncode=0)
```



## CompletedProcess

`run()`方法的返回类型，包含下列属性：

**`args`**

启动进程的参数，是字符串或字符串列表。

**`returncode`**

子进程的退出状态码，0表示进程正常退出。

**`stdout`**

捕获到的子进程的标准输出，是一个字节序列，或者一个字符串（如果`run()`设置了参数`encoding`,`errors`或`text=True`）。如果未有捕获，则为`None`。

如果设置了参数`stderr=subprocess.STDOUT`，标准错误会随同标准输出被捕获，并且`stderr`将为`None`。

**`stderr`**

捕获到的子进程的标准错误，是一个字节序列，或者一个字符串（如果`run()`设置了参数`encoding`,`errors`或`text=True`）。如果未有捕获，则为`None`。

**`check_returncode`()**

检查`returncode`，非零则抛出`CalledProcessError`。



## Popen





# [sys](https://docs.python.org/zh-cn/3/library/sys.html)——系统相关的参数和函数

## executable

返回当前Python解释器的可执行文件的绝对路径。

```python
>>> import sys
>>> sys.executable
'/Users/xyx/.pyenv/versions/3.8.7/bin/python'
```



## exit()

从Python中退出，实现方式是抛出一个`SystemExit`异常。

可选参数可以是表示退出状态的整数（默认为整数0），也可以是其他类型的对象。如果它是整数，则shell等将0视为“成功终止”，非零值视为“异常终止”。



## platform

本字符串是一个平台标识符，对于各种系统的值为：

| 系统           | `平台` 值  |
| :------------- | :--------- |
| AIX            | `'aix'`    |
| Linux          | `'linux'`  |
| Windows        | `'win32'`  |
| Windows/Cygwin | `'cygwin'` |
| macOS          | `'darwin'` |



## stdin, stdout, stderr

解释器用于标准输入、标准输出和标准错误的文件对象：

+ `stdin`用于所有交互式输入
+ `stdout`用于`print()`和expression语句的输出，以及输出`input()`的提示符
+ 解释器自身的提示符和错误消息发往`stderr`



## version, version_info

`version`是一个包含Python解释器版本号、编译版本号、所用编译器等信息的字符串，`version_info`是一个包含版本号五部分的元组: *major*, *minor*, *micro*, *releaselevel* 和 *serial*。

```python
>>> sys.version
'3.6.9 (default, Oct  8 2020, 12:12:24) \n[GCC 8.4.0]'
>>> sys.version_info
sys.version_info(major=3, minor=6, micro=9, releaselevel='final', serial=0)
```





# tempfile——生成临时文件和目录



## gettempdir()

返回放置临时文件的目录的名称。

Python搜索标准目录列表，以找到调用者可以在其中创建文件的目录。这个列表是：

1. `TMPDIR` ,`TEMP`或`TMP` 环境变量指向的目录。
2. 与平台相关的位置：
   + 在 Windows 上，依次为 `C:\TEMP`、`C:\TMP`、`\TEMP` 和 `\TMP`
   + 在所有其他平台上，依次为 `/tmp`、`/var/tmp` 和 `/usr/tmp`
3. 不得已时，使用当前工作目录。





# threading——基于线程的并行

## active_count()

返回当前存活的`Thread`对象的数量。



## current_thread()

返回当前调用者的控制线程的`Thread`对象。



## main_thread()

返回主`Thread`对象。



## Thread





## Lock

原始锁处于 "锁定" 或者 "非锁定" 两种状态之一。它有两个基本方法，`acquire()`和`release()`。当状态为非锁定时，`acquire()`将状态改为锁定并立即返回；当状态是锁定时，`acquire()`将阻塞至其他线程调用`release()`将其改为非锁定状态，然后`acquire()`重置其为锁定状态并返回。 `release()`只在锁定状态下调用，将状态改为非锁定并立即返回。如果尝试释放一个非锁定的锁，则会引发`RuntimeError` 异常。

原始锁在创建时为非锁定状态。当多个线程在`acquire()`阻塞，然后`release()`重置状态为未锁定时，只有一个线程能继续执行；至于哪个线程继续执行则没有定义，并且会根据实现而不同。

