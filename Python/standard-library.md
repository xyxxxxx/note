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

返回一个布尔值，`True` 或者 `False`。 *x* 使用标准的真值测试过程来转换。`bool` 类是 `int` 的子类，其他类不能继承自它。它只有 `False` 和 `True` 两个实例。



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



## delattr()

```python
delattr(object, name)
```

与 `setattr()` 对应。删除对象 *object* 的名为 *name* 的属性，*name* 必须是字符串，指定一个现有属性。如果对象允许，该函数将删除指定的属性。例如 `delattr(x, 'foobar')` 等同于 `del x.foobar`。



## dict()

创建一个新的字典。参见数据结构-字典。



## dir()

如果没有实参，则返回当前本地作用域中的对象列表：

```python
>>> a = 1
>>> b = 'abc'
>>> def f():
...   return True
... 
>>> dir()
['__annotations__', '__builtins__', '__doc__', '__loader__', '__name__', '__package__', '__spec__', 'a', 'b', 'f']
```

如果有实参，它会尝试返回该对象的有效属性（包括方法）列表：

```python
>>> dir('abc')
['__add__', '__class__', '__contains__', '__delattr__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getitem__', '__getnewargs__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__iter__', '__le__', '__len__', '__lt__', '__mod__', '__mul__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__rmod__', '__rmul__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', 'capitalize', 'casefold', 'center', 'count', 'encode', 'endswith', 'expandtabs', 'find', 'format', 'format_map', 'index', 'isalnum', 'isalpha', 'isascii', 'isdecimal', 'isdigit', 'isidentifier', 'islower', 'isnumeric', 'isprintable', 'isspace', 'istitle', 'isupper', 'join', 'ljust', 'lower', 'lstrip', 'maketrans', 'partition', 'replace', 'rfind', 'rindex', 'rjust', 'rpartition', 'rsplit', 'rstrip', 'split', 'splitlines', 'startswith', 'strip', 'swapcase', 'title', 'translate', 'upper', 'zfill']
```

如果对象有一个名为 `__dir__()` 的方法，那么该方法将被调用，并且必须返回一个属性列表。这允许实现自定义 `__getattr__()` 或 `__getattribute__()` 函数的对象能够自定义 `dir()` 来报告它们的属性。

如果对象不提供 `__dir__()`，这个函数会尝试从对象已定义的 `__dict__` 属性和类型对象收集信息。结果列表并不总是完整的，如果对象有自定义 `__getattr__()`，那结果可能不准确。

默认的 `dir()` 机制对不同类型的对象行为不同，它会试图返回最相关而不是最全的信息：

- 如果对象是模块对象，则列表包含模块的属性名称。
- 如果对象是类型或类对象，则列表包含它们的属性名称，并且递归查找所有基类的属性。
- 如果对象是实例，则列表包含对象的属性名称，它的类属性名称，并且递归查找它的类的所有基类的属性。

```python
>>> import platform
>>> dir(platform)
['_WIN32_CLIENT_RELEASES', '_WIN32_SERVER_RELEASES', '__builtins__', '__cached__', '__copyright__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__spec__', '__version__', '_comparable_version', '_component_re', '_default_architecture', '_follow_symlinks', '_ironpython26_sys_version_parser', '_ironpython_sys_version_parser', '_java_getprop', '_libc_search', '_mac_ver_xml', '_node', '_norm_version', '_platform', '_platform_cache', '_pypy_sys_version_parser', '_sys_version', '_sys_version_cache', '_sys_version_parser', '_syscmd_file', '_syscmd_uname', '_syscmd_ver', '_uname_cache', '_ver_output', '_ver_stages', 'architecture', 'collections', 'java_ver', 'libc_ver', 'mac_ver', 'machine', 'node', 'os', 'platform', 'processor', 'python_branch', 'python_build', 'python_compiler', 'python_implementation', 'python_revision', 'python_version', 'python_version_tuple', 're', 'release', 'sys', 'system', 'system_alias', 'uname', 'uname_result', 'version', 'win32_edition', 'win32_is_iot', 'win32_ver']
>>> 
>>> import multiprocessing
>>> dir(multiprocessing.Process)
['_Popen', '__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_bootstrap', '_check_closed', '_start_method', 'authkey', 'close', 'daemon', 'exitcode', 'ident', 'is_alive', 'join', 'kill', 'name', 'pid', 'run', 'sentinel', 'start', 'terminate']
```





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



## format()

```python
format(value[, format_spec])
```

将 *value* 转换为 *format_spec* 控制的格式化表示。*format_spec* 的解释取决于 *value* 实参的类型，但是大多数内置类型使用标准格式化语法。

```python
>>> format('Alice', 's')
'Alice'
>>> format(123, 'b')
'1111011'
```





## getattr()

```python
getattr(object, name[, default])
```

与 `setattr()` 对应。返回对象 *object* 的 *name* 属性的值，*name* 必须是字符串。如果该字符串是对象的属性之一，则返回该属性的值，例如 `getattr(x, 'foobar')` 等同于 `x.foobar`；如果指定的属性不存在，但提供了 *default* 值，则返回它，否则触发 `AttributeError`。



## hasattr()

```python
hasattr(object, name)
```

返回对象 *object* 是否具有名为 *name* 的属性，*name* 必须是字符串。如果字符串是对象的属性之一的名称，则返回 `True`，否则返回 `False`。（此功能是通过调用 `getattr(object, name)` 看是否有 `AttributeError` 异常来实现的。）



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



## id()

返回对象的标识值，是一个整数，并且在此对象的生命周期内保证是唯一且恒定的。两个生命周期不重叠的对象可能具有相同的 `id()` 值。



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

返回一个可控属性。

```python
class property(fget=None, fset=None, fdel=None, doc=None)
# fget        getter方法
# fset        setter方法
# fdel        deleter方法
# doc         property对象的docstring,否则将复制fget的docstring作为docstring
```

一个典型的用法是定义一个托管属性 `x`:

```python
>>> class Student(object):
    def __init__(self, name, score):
        self.name = name
        self._score = score
    def get_score(self):
        return self._score
    def set_score(self, value):
        self._score = value
    def del_score(self):
        del self._score
    score = property(get_score, set_score, del_score, "I'm the 'score' property.")
... 
>>> bart = Student('Bart Simpson', 59)
>>> bart.score                 # 调用`get_score()`
59
>>> bart.score = 60            # 调用`set_score()`
>>> del bart.score             # 调用`del_score()`
```

更常见的写法是将 `property` 作为一个装饰器：

```python
>>> class Student(object):
    def __init__(self, name, score):
        self.name = name
        self._score = score
    @property    
    def score(self):
        """Get score."""        # 作为property对象的docstring
        return self._score
... 
>>> bart = Student('Bart Simpson', 59)
>>> bart.score                  # 调用`score()`
59
>>> bart.score = 60             # 没有定义setter
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: can't set attribute
>>> bart._score = 60            # `_score`属性仍然可以自由访问和修改,尽管不应该这么做
```

`property` 对象具有 `getter`, `setter` 和 `deleter` 方法，可以用作装饰器创建该 `property` 对象的副本，并将相应的方法设为所装饰的函数：

```python
>>> class Student(object):               # 与第一个例子完全等价
    def __init__(self, name, score):
        self.name = name
        self._score = score
    @property
    def score(self):                     # getter方法
        """I'm the 'score' property."""  # 作为property对象的docstring
        return self._score
    @score.setter
    def score(self, value):              # setter方法
        self._score = value
    @score.deleter
    def score(self):                     # deleter方法
        del self._score
... 
>>> bart = Student('Bart Simpson', 59)
>>> bart.score
59
>>> bart.score = 60
>>> del bart.score
```



## setattr()

```python
setattr(object, name, value)
```

与 `getattr()` 和 `delattr()` 对应。设置对象 *object* 的 *name* 属性的值，*name* 必须是字符串，指定一个现有属性或新增属性。如果对象允许，该函数将设置指定的属性。例如 `setattr(x, 'foobar', 123)` 等同于 `x.foobar = 123`。



## sorted()

根据可迭代对象中的项返回一个新的已排序列表。

```python

```





## @staticmethod

将方法转换为静态方法。

参见面向对象编程-类-函数。



## str()

返回一个对象的 `str` 版本，`str`是内置字符串类型。

详见数据类型-字符串。



## sum()

从 *start* 开始自左向右对可迭代对象的项求和并返回总计值。可迭代对象的项通常为数字，而 *start* 值则不允许为字符串。

要拼接字符串序列，更好的方式是调用`''.join(sequence)`；要以扩展精度对浮点值求和，请使用`math.fsum()`；要拼接一系列可迭代对象，请使用`itertools.chain()`。



## super()



## tuple()

创建一个新的元组。参见数据结构-元组。



## type

```python
class type(object)
class type(name, bases, dict)
```

传入一个参数时，返回 *object* 的类型。返回值是一个 type 对象，通常与 `object.__class__` 所返回的对象相同。

```python
>>> a = 1
>>> type(a)
<class 'int'>
>>> a.__class__
<class 'int'>
```

检测对象类型推荐使用 `isinstance()`，因为它会考虑子类的情况。

传入三个参数时，返回一个新的 type 对象。 这在本质上是 `class` 语句的一种动态形式，*name* 字符串即类名并会成为 `__name__` 属性；*bases* 元组包含基类并会成为 `__bases__` 属性；如果为空则会添加所有类的终极基类 `object`； *dict* 字典包含类主体的属性和方法定义，它在成为 `__dict__` 属性之前可能会被拷贝或包装。 

```python
>>> X = type('X', (), dict(a=1, f=abs))
>>> # 相当于
>>> class X:
...     a = 1
...     f = abs
```





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





# collections——容器数据类型

参见数据结构-容器数据类型。



# collections.abc——容器的抽象基类

参见数据结构-自定义容器数据类型。



# copy——浅层和深层复制操作

Python 的赋值语句不复制对象，而是创建目标和对象的绑定关系。对于自身可变，或包含可变项的集合，有时要生成副本用于改变操作，而不必改变原始对象。本模块提供了通用的浅层复制和深层复制操作。



+ `copy()`：返回对象的浅层复制。
+ `deepcopy()`：返回对象的深层复制。

浅层与深层复制的区别仅与复合对象（即包含列表、字典或类的实例等其他对象的对象）相关：

- *浅层复制* 构造一个新的复合对象，然后（在尽可能的范围内）将原始对象中找到的对象的 *引用* 插入其中。
- *深层复制* 构造一个新的复合对象，然后递归地将在原始对象里找到的对象的 *副本* 插入其中。

```python
>>> import copy
>>> a1 = [1, 2, 3, [4, 5]]    # 复合对象
>>> a2 = copy.copy(a1)        # 浅层复制
>>> a3 = copy.deepcopy(a1)    # 深层复制
>>> hex(id(a1))
'0x107247100'
>>> hex(id(a2))               # 新的复合对象
'0x107298600'
>>> hex(id(a3))               # 新的复合对象
'0x10720bec0'
>>> hex(id(a1[3]))
'0x107299ec0'
>>> hex(id(a2[3]))            # 插入原始对象中的对象的引用
'0x107299ec0'
>>> hex(id(a3[3]))            # 插入原始对象中的对象的副本
'0x1072983c0'
```

深度复制操作通常存在两个问题，而浅层复制操作并不存在这些问题：

- 递归对象 (直接或间接包含对自身引用的复合对象) 可能会导致递归循环。
- 由于深层复制会复制所有内容，因此可能会过多复制（例如本应该在副本之间共享的数据）。

`deepcopy()` 函数用以下方式避免了这些问题：

- 保留在当前复制过程中已复制的对象的 "备忘录" （`memo`） 字典
- 允许用户定义的类重载复制操作或复制的组件集合。



制作字典的浅层复制可以使用 `dict.copy()` 方法，而制作列表的浅层复制可以通过赋值整个列表的切片完成，例如，`copied_list = original_list[:]`。



# csv——CSV 文件读写

`csv`模块实现了 csv 格式表单数据的读写。其提供了诸如“以兼容 Excel 的方式输出数据文件”或“读取 Excel 程序输出的数据文件”的功能，程序员无需知道 Excel 所采用 csv 格式的细节。此模块同样可以用于定义其他应用程序可用的 csv 格式或定义特定需求的 csv 格式。

csv 文件示例：

```
// grades.csv
Last name, First name, SSN,         Test1,   Test2,  Test3,    Test4,   Final,  Grade
Alfalfa,   Aloysius,   123-45-6789, 40.0,    90.0,   100.0,    83.0,    49.0,   D-
Alfred,    University, 123-12-1234, 41.0,    97.0,    96.0,    97.0,    48.0,   D+
Gerty,     Gramma,     567-89-0123, 41.0,    80.0,    60.0,    40.0,    44.0,   C
Android,   Electric,   087-65-4321, 42.0,    23.0,    36.0,    45.0,    47.0,   B-
Bumpkin,   Fred,       456-78-9012, 43.0,    78.0,    88.0,    77.0,    45.0,   A-
Rubble,    Betty,      234-56-7890, 44.0,    90.0,    80.0,    90.0,    46.0,   C-
Noshow,    Cecil,      345-67-8901, 45.0,    11.0,    -1.0,     4.0,    43.0,   F
Buff,      Bif,        632-79-9939, 46.0,    20.0,    30.0,    40.0,    50.0,   B+
Airpump,   Andrew,     223-45-6789, 49.0      1.0,    90.0,   100.0,    83.0,   A
Backus,    Jim,        143-12-1234, 48.0,     1.0,    97.0,    96.0,    97.0,   A+
Carnivore, Art,        565-89-0123, 44.0,     1.0,    80.0,    60.0,    40.0,   D+
Dandy,     Jim,        087-75-4321, 47.0,     1.0,    23.0,    36.0,    45.0,   C+
Elephant,  Ima,        456-71-9012, 45.0,     1.0,    78.0,    88.0,    77.0,   B-
Franklin,  Benny,      234-56-2890, 50.0,     1.0,    90.0,    80.0,    90.0,   B-
George,    Boy,        345-67-3901, 40.0,     1.0,    11.0,    -1.0,     4.0,   B
Heffalump, Harvey,     632-79-9439, 30.0,     1.0,    20.0,    30.0,    40.0,   C

```



## reader()

返回一个 reader 对象，该对象将逐行遍历 *csvfile*。

```python
>>> import csv
>>> with open('grades.csv', newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        print(row)
... 
['Last name', ' First name', ' SSN', '        Test1', ' Test2', ' Test3', ' Test4', ' Final', ' Grade']
['Alfalfa', '   Aloysius', '   123-45-6789', ' 40.0', '    90.0', '   100.0', '    83.0', '    49.0', '   D-']
['Alfred', '    University', ' 123-12-1234', ' 41.0', '    97.0', '    96.0', '    97.0', '    48.0', '   D+']
# ...
['Heffalump', ' Harvey', '     632-79-9439', ' 30.0', '     1.0', '    20.0', '    30.0', '    40.0', '   C']
```

```python
>>> import csv
>>> with open('some.csv', newline='', encoding='utf-8') as f:    # 指定解码文件的编码,默认为系统默认编码
>>>     reader = csv.reader(f)
>>>     for row in reader:
>>>         print(row)
```



## writer()

返回一个 writer 对象，该对象负责将用户的数据在给定的文件类对象上转换为带分隔符的字符串。*csvfile* 可以是具有 `write()` 方法的任何对象。

```python
>>> with open('some.csv', 'w', newline='') as f:
...   writer = csv.writer(f)
...   for i in range(5):
...     writer.writerow(['a', 'b', 'c'])
... 
7
7
7
7
7
```

```
// some.csv
a,b,c
a,b,c
a,b,c
a,b,c
a,b,c
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

一个理想化日期，它假设当今的公历在过去和未来永远有效。包含属性 `year`、`month` 和 `day`。





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

`date` 和 `time` 类的结合。



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



## time

一个独立于任何特定日期的理想化时间，它假设每一天都恰好等于 86400 秒（这里没有“闰秒”的概念）。

```python
class datetime.time(hour=0, minute=0, second=0, microsecond=0, tzinfo=None, *, fold=0)
```

所有参数都是可选的；*tzinfo* 可以是 `None`，或者是一个 `tzinfo` 子类的实例；其余的参数必须是在下面范围内的整数：

* `0 <= hour < 24`
* `0 <= minute < 60`
* `0 <= second < 60`
* `0 <= microsecond < 1000000`
* `fold in [0, 1]`

如果给出一个此范围以外的参数，则会引发 `ValueError`。 所有参数值默认为 0，只有 `tzinfo` 默认为 `None`。

```python
>>> t = datetime.time(hour=12, minute=34, second=56, microsecond=789000)
```

具有以下属性和方法：

### hour, minute, second, microsecond

```python
>>> t.hour
12
>>> t.minute
34
>>> t.second
56
>>> t.microsecond
789000
```

### replace





### tzinfo

作为 tzinfo 参数被传给 `time` 构造器的对象，如果没有传入值则为 `None`。







## timedelta

表示两个 `date`、 `time` 或 `datetime` 对象之间的时间间隔，精确到微秒。





## tzinfo

一个描述时区信息对象的抽象基类。用来给 `datetime` 和 `time` 类提供自定义的时间调整概念（例如处理时区和/或夏令时）。



## timezone

一个实现了 `tzinfo` 抽象基类的子类，用于表示相对于UTC的偏移量。



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

创建一个迭代器，返回从可迭代对象以切片方式选中的元素。与普通的切片不同，`islice()`不支持将 `start, stop, step` 设为负值。大致相当于：

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





# json——JSON 编码和解码器

> `json` 包实际上就是被添加到标准库中的 `simplejson` 包。`simplejson` 包比 Python 版本更新更加频繁，因此在条件允许的情况下使用 `simplejson` 包是更好的选择。一种好的实践如下：
>
> ```python
> try:
>     import simplejson as json
> except ImportError:
>     import json
> ```
>
> `json` 包和 `simplejson` 包具有相同的接口和类，因此下面仅 `json` 包为例进行介绍。

## 接口

### dumps()

将对象序列化为 JSON 格式的 `str`。参数的含义见 `JSONEncoder`。

```python
>>> import json
>>> json.dumps(['foo', {'bar': ('baz', None, 1.0, 2)}])
'["foo", {"bar": ["baz", null, 1.0, 2]}]'
>>> json.dumps({"c": 0, "b": 1, "a": math.nan})            # 允许NaN,inf
'{"c": 0, "b": 1, "a": NaN}'
>>> json.dumps({"c": 0, "b": 0, "a": 0}, sort_keys=True)   # 按键排序
'{"a": 0, "b": 0, "c": 0}'
>>>
>>> json.dumps({"c": 0, "b": 1, "a": {"d": 2, "e": 3}})
'{"c": 0, "b": 1, "a": {"d": 2, "e": 3}}'
>>> json.dumps({"c": 0, "b": 1, "a": {"d": 2, "e": 3}}, indent=0)           # 美化输出
'{\n"c": 0,\n"b": 1,\n"a": {\n"d": 2,\n"e": 3\n}\n}'
>>> print(json.dumps({"c": 0, "b": 1, "a": {"d": 2, "e": 3}}, indent=0))
{
"c": 0,
"b": 1,
"a": {
"d": 2,
"e": 3
}
}
>>> print(json.dumps({"c": 0, "b": 1, "a": {"d": 2, "e": 3}}, indent=2))
{
  "c": 0,
  "b": 1,
  "a": {
    "d": 2,
    "e": 3
  }
}
```



### loads()

将一个包含 JSON 的 `str`、`bytes` 或 `bytearray` 实例反序列化为Python对象。

```python
>>> import json
>>> json.loads('["foo", {"bar": ["baz", null, 1.0, 2]}]')
['foo', {'bar': ['baz', None, 1.0, 2]}]
>>> json.loads('{"c": 0, "b": 1, "a": {"d": 2, "e": 3}}')
{'c': 0, 'b': 1, 'a': {'d': 2, 'e': 3}}
```



## 编码器和解码器

### JSONEncoder

用于Python数据结构的可扩展 JSON 编码器，默认支持以下对象和类型：

| Python                              | JSON   |
| :---------------------------------- | :----- |
| dict                                | object |
| list, tuple                         | array  |
| str                                 | string |
| int, float, int 和 float 派生的枚举 | number |
| True                                | true   |
| False                               | false  |
| None                                | null   |

```python
class json.JSONEncoder(*, skipkeys=False, ensure_ascii=True, check_circular=True, allow_nan=True, sort_keys=False, indent=None, separators=None, default=None)
# skipkeys       若为`False`,则当尝试对非`str`,`int`,`float`或`None`的键进行编码时将会引发`TypeError`;
#                   否则这些条目将被直接跳过
# ensure_ascii   若为`True`,所有输入的非ASCII字符都将被转义;否则会原样输出
# check_circular 若为`True`,则列表、字典和自定义编码的对象在编码期间会被检查重复循环引用防止无限递归
#                   (无限递归将导致`OverflowError`)
# allow_nan      若为`True`,则对`NaN`,`Infinity`和`-Infinity`进行编码.此行为不符合JSON规范,但与大多数的基于Javascript
#                   的编码器和解码器一致;否则引发一个`ValueError`
# sort_keys      若为`True`,则字典的输出是按照键排序
# indent         若为一个非负整数或字符串,则JSON数组元素和对象成员会被美化输出为该值指定的缩进等级;若为零、负数或者"",
#                   则只会添加换行符;`None`(默认值)选择最紧凑的表达
```

具有以下属性和方法：

#### default()

#### encode()

返回Python数据结构的JSON字符串表达方式。

```python
>>> json.JSONEncoder().encode({"foo": ["bar", "baz"]})
'{"foo": ["bar", "baz"]}'
```



### JSONDecoder



## 异常

### JSONDecodeError

JSON 解析错误，是 `ValueError` 的子类。

```python
exception json.JSONDecodeError(msg, doc, pos)
# msg     未格式化的错误信息
# doc     正在解析的JSON文档
# pos     解析出错的文档索引位置
```

```python
>>> import json
>>> try:
    json.loads('{"c": 0, "b": 1, "a": {"d": 2, "e" 3}}')   # lack a colon between "e" and 3
except json.JSONDecodeError as e:
    print(e.msg)
    print(e.doc)
    print(e.pos)
... J
Expecting ':' delimiter
{"c": 0, "b": 1, "a": {"d": 2, "e" 3}}
35                              # ↑
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



### run()

表示进程活动的方法。



### start()

启动进程活动。

这个方法每个进程对象最多只能调用一次。它会将对象的`run()`方法安排在一个单独的进程中调用。



### join([timeout])

如果可选参数 *timeout* 是 `None` （默认值），则该方法将阻塞，直到调用`join()`方法的进程终止；如果 *timeout* 是一个正数，它最多会阻塞 *timeout* 秒。不管是进程终止还是方法超时，该方法都返回 `None`。

一个进程可以被`join`多次。

进程无法`join`自身，因为这会导致死锁。尝试在启动进程之前`join`进程会产生一个错误。



### is_alive()

返回进程是否处于活动状态。从`start()`方法返回到子进程终止之间，进程对象都处于活动状态。



### name

进程的名称。该名称是一个字符串，仅用于识别，没有具体语义。可以为多个进程指定相同的名称。



### daemon

进程的守护标志，一个布尔值。必须在`start()`被调用之前设置。

初始值继承自创建进程。

当一个进程退出时，它会尝试终止子进程中的所有守护进程。



### pid

返回进程ID。



### exitcode

子进程的退出代码。`None`表示进程尚未终止；负值-N表示子进程被信号N终止。



### terminate()

终止进程。在Unix上由`SIGTERM`信号完成。



### kill()

与`terminate()`相同，但在Unix上使用`SIGKILL`信号。



### close()

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





# operator——







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



# pickle——Python对象序列化



# platform——获取底层平台的标识数据

## machine()

返回机器类型。

```python
>>> platform.machine()
'x86_64'
```



## node()

返回计算机的网络名称。

```python
>>> platform.node()
'Yuxuans-MacBook-Pro.local'
```



## platform()

返回一个标识底层平台的字符串，其中带有尽可能多的有用信息。

```python
>>> platform.platform()
'macOS-11.2.3-x86_64-i386-64bit'
```



## python_version()

```python
>>> platform.python_version()
'3.8.7'
```



## release()

返回系统的发布版本。

```python
>>> platform.release()
'7'                           # Windows version
>>> platform.release()
'10'                          # Windows version
>>> platform.release()
'20.3.0'                      # Darwin version, refer to https://en.wikipedia.org/wiki/MacOS_Big_Sur
```



## system()

返回系统平台/OS的名称。

```python
>>> platform.system()
'Windows'                     # Windows
>>> platform.system()
'Darwin'                      # macOS
>>> platform.system()
'Linux'                       # Linux
```



## Mac OS平台

### mac_ver()

获取 Mac OS 版本信息并将其返回为元组 `(release, versioninfo, machine)`，其中 *versioninfo* 是一个元组 `(version, dev_stage, non_release_version)`。

```python
>>> platform.mac_ver()
('11.2.3', ('', '', ''), 'x86_64')    # macOS Big Sur Version 11.2.3
```





# pprint——数据美化输出

`pprint`模块提供了“美化打印”任意 Python 数据结构的功能。



## isreadable()

确定对象的格式化表示是否“可读”，或是否可以通过`eval()`重新构建对象的值。对于递归对象总是返回`False`。

```python
>>> stuff = ['spam', 'eggs', 'lumberjack', 'knights', 'ni']
>>> stuff.insert(0, stuff[:])
>>> pprint.isreadable(stuff)
True

>>> stuff = ['spam', 'eggs', 'lumberjack', 'knights', 'ni']
>>> stuff.insert(0, stuff)
>>> pprint.isreadable(stuff)
False
```



## isrecursive()

确定对象是否为递归对象。

```python
>>> stuff = ['spam', 'eggs', 'lumberjack', 'knights', 'ni']
>>> stuff.insert(0, stuff[:])
>>> pprint.isrecursive(stuff)
False

>>> stuff = ['spam', 'eggs', 'lumberjack', 'knights', 'ni']
>>> stuff.insert(0, stuff)
>>> pprint.isrecursive(stuff)
True
```



## pprint()

打印对象的格式化表示。

```python
pprint.pprint(object, stream=None, indent=1, width=80, depth=None, *, compact=False, sort_dicts=True)
# object       被打印的对象
# stream...    参见`PrettyPrinter`,将作为参数被传给`PrettyPrinter`构造函数
```

```python
>>> import pprint
>>> stuff = ['spam', 'eggs', 'lumberjack', 'knights', 'ni']
>>> stuff.insert(0, stuff[:])
>>> stuff
[['spam', 'eggs', 'lumberjack', 'knights', 'ni'], 'spam', 'eggs', 'lumberjack', 'knights', 'ni']
>>> pprint.pprint(stuff)
[['spam', 'eggs', 'lumberjack', 'knights', 'ni'],
 'spam',
 'eggs',
 'lumberjack',
 'knights',
 'ni']
>>> pprint.pprint(stuff, indent=2)
[ ['spam', 'eggs', 'lumberjack', 'knights', 'ni'],
  'spam',
  'eggs',
  'lumberjack',
  'knights',
  'ni']
>>> pprint.pprint(stuff, indent=0)
[['spam', 'eggs', 'lumberjack', 'knights', 'ni'],
'spam',
'eggs',
'lumberjack',
'knights',
'ni']
>>> pprint.pprint(stuff, width=20)
[['spam',
  'eggs',
  'lumberjack',
  'knights',
  'ni'],
 'spam',
 'eggs',
 'lumberjack',
 'knights',
 'ni']
>>> pprint.pprint(stuff, width=20, compact=True)
[['spam', 'eggs',
  'lumberjack',
  'knights', 'ni'],
 'spam', 'eggs',
 'lumberjack',
 'knights', 'ni']
>>> pprint.pprint(stuff, depth=1)
[[...], 'spam', 'eggs', 'lumberjack', 'knights', 'ni']
```



## pformat()

将对象的格式化表示作为字符串返回，其余部分与`pprint()`相同。

```python
>>> pprint.pformat(stuff)
"[['spam', 'eggs', 'lumberjack', 'knights', 'ni'],\n 'spam',\n 'eggs',\n 'lumberjack',\n 'knights',\n 'ni']"
```



## PrettyPrinter

`pprint`模块定义的实现美化打印的类。

```python
class pprint.PrettyPrinter(indent=1, width=80, depth=None, stream=None, *, compact=False, sort_dicts=True)
# indent      每个递归层次的缩进量
# width       每个输出行的最大宽度
# depth       可被打印的层级数
# stream      输出流,未指定则选择`sys.stdout`
# compact     若为True,则将在width可容纳的条件下让输出更紧凑
# sort_dicts  若为True,则字典将按键排序输出,否则按插入顺序输出
```

```python
>>> import pprint
>>> stuff = ['spam', 'eggs', 'lumberjack', 'knights', 'ni']
>>> stuff.insert(0, stuff[:])
>>> stuff
[['spam', 'eggs', 'lumberjack', 'knights', 'ni'], 'spam', 'eggs', 'lumberjack', 'knights', 'ni']
>>> pprinter = pprint.PrettyPrinter()
>>> pprinter.pprint(stuff)
[['spam', 'eggs', 'lumberjack', 'knights', 'ni'],
 'spam',
 'eggs',
 'lumberjack',
 'knights',
 'ni']
```

`PrettyPrinter`对象具有`pprint`模块的各方法。实际上`pprint`模块的各方法都是先创建`PrettyPrinter`对象再调用对象的方法。





# random——生成伪随机数

该模块实现了各种分布的伪随机数生成器。

## 重现

### seed()

初始化随机数生成器。

```python
random.seed(a=None, version=2)
# a        如果被省略或为`None`,则使用当前系统时间;如果操作系统提供随机源,则使用它们而不是系统时间
#          如果为`int`类型,则直接使用
```

```python
>>> random.seed(0)
>>> random.random()
0.8444218515250481
```



### getstate()

捕获生成器的当前内部状态的对象并返回，这个对象用于传递给`setstate()`以恢复状态。



### setstate()

```python
random.setstate(state)
```

将生成器的内部状态恢复到`getstate()`被调用时的状态。

```python
>>> random.seed(0)
>>> state = random.getstate()
>>> random.setstate(state)
>>> random.random()
0.8444218515250481
>>> random.setstate(state)
>>> random.random()
0.8444218515250481
```



## 随机整数

### randrange()

```python
random.randrange(stop)
random.randrange(start, stop[, step])
```

从`range(0, stop)`或`range(start, stop, step)`返回一个随机选择的元素。



### randint()

```python
random.randint(a, b)
```

返回随机整数 *N* 满足 `a <= N <= b`。相当于 `randrange(a, b+1)`。



## 序列用函数

### choice()

从非空序列 *seq* 返回一个随机元素。 如果 *seq* 为空，则引发 `IndexError`。



### choices()



### shuffle()

```python
random.shuffle(x[, random])
```

将序列 *x* 随机打乱位置。可选参数 *random* 是一个不带参数的函数，返回 [0.0, 1.0) 范围内的随机浮点数；默认情况下，这是函数 `random()`。

```python
>>> a = list(range(10))
>>> random.shuffle(a)              # 原位操作
>>> a
[8, 9, 1, 2, 5, 3, 7, 4, 0, 6]
```



### sample()

返回从序列中选择的唯一元素的 *k* 长度列表。 用于无重复的随机抽样。

```python
>>> random.sample(range(10), k=5)
[4, 7, 1, 9, 3]
```

要从一系列整数中选择样本，请使用`range()`对象作为参数，这种方法特别快速且节省空间：

```python
>>> random.sample(range(10000000), k=60)
[9787526, 3664860, 8467240, 2336625, 4728454, 2344545, 1590996, 4202798, 8934935, 2465603, 5203412, 1656973, 1237192, 5539790, 7921240, 9392115, 1689485, 5935633, 7284194, 5304900, 3430567, 9269809, 8002896, 7427162, 8746862, 4370335, 1044878, 9205646, 235580, 1564842, 6691148, 19173, 8280862, 5589080, 4092145, 5456023, 1056700, 3205573, 9521250, 3719574, 4003310, 2390659, 9109859, 7515682, 1530349, 1349656, 5369625, 8521829, 8208870, 1829687, 5057437, 9248729, 4883691, 2093976, 9184534, 5582627, 9064454, 3409161, 9180997, 9858578]
```





## 实值分布

### random()

返回 [0.0, 1.0) 范围内的下一个随机浮点数。

```python
>>> random.random()
0.13931343809011631
```



### uniform()

```python
random.uniform(a, b)
```

返回一个随机浮点数 *N* ，当 `a <= b` 时 `a <= N <= b` ，当 `b < a` 时 `b <= N <= a` 。

```python
>>> random.uniform(60, 80)
79.59813867742345
```



### gauss()

```python
random.gauss(mu, sigma)
```

正态分布，*mu* 是平均值，*sigma* 是标准差。

多线程注意事项：当两个线程同时调用此方法时，它们有可能将获得相同的返回值。 这可以通过三种办法来避免。 1) 让每个线程使用不同的随机数生成器实例； 2) 在所有调用外面加锁； 3) 改用速度较慢但是线程安全的 normalvariate() 函数。



### normalviriate()

```python
random.normalvariate(mu, sigma)
```

正态分布，*mu* 是平均值，*sigma* 是标准差。







# re——正则表达式操作

见正则表达式。





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
>>> sys.version
'3.8.7 (default, Mar  4 2021, 14:48:51) \n[Clang 12.0.0 (clang-1200.0.32.29)]'
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





# time——时间的访问和转换

此模块提供了各种与时间相关的函数。相关功能还可以参阅`datetime`和`calendar`模块。

下面是一些术语和惯例的解释：

- *纪元（epoch）* 是时间开始的点，其值取决于平台。对于Unix，纪元是1970年1月1日00:00:00（UTC）。要找出给定平台上的 epoch ，请查看 `time.gmtime(0)` 。

- 术语 *纪元秒数* 是指自纪元时间点以来经过的总秒数，通常不包括[闰秒](https://en.wikipedia.org/wiki/Leap_second)。 在所有符合 POSIX 标准的平台上，闰秒都不会记录在总秒数中。

- 此模块中的函数可能无法处理纪元之前或遥远未来的日期和时间。“遥远未来”的定义由对应的C语言库决定；对于32位系统，它通常是指2038年及以后。

- 函数 `strptime()` 在接收到 `%y` 格式代码时可以解析使用 2 位数表示的年份。当解析 2 位数年份时，函数会按照 POSIX 和 ISO C 标准进行年份转换：数值 69--99 被映射为 1969--1999；数值 0--68 被映射为 2000--2068。

- UTC是协调世界时（Coordinated Universal Time）的缩写。它以前也被称为格林威治标准时间（GMT）。使用UTC而不是CUT作为缩写是英语与法语（Temps Universel Coordonné）之间妥协的结果，不是什么低级错误。

- DST是夏令时（Daylight Saving Time）的缩写，在一年的某一段时间中将当地时间调整（通常）一小时。 DST的规则非常神奇（由当地法律确定），并且每年的起止时间都不同。C语言库中有一个表格，记录了各地的夏令时规则（实际上，为了灵活性，C语言库通常是从某个系统文件中读取这张表）。从这个角度而言，这张表是夏令时规则的唯一权威真理。

- 由于平台限制，各种实时函数的精度可能低于其值或参数所要求（或给定）的精度。例如，在大多数Unix系统上，时钟频率仅为每秒50或100次。

- 使用以下函数在时间表示之间进行转换：

  | 从                                                           | 到                                                           | 使用                                                         |
  | :----------------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
  | 自纪元以来的秒数                                             | UTC 的 [`struct_time`](https://docs.python.org/zh-cn/3/library/time.html#time.struct_time) | [`gmtime()`](https://docs.python.org/zh-cn/3/library/time.html#time.gmtime) |
  | 自纪元以来的秒数                                             | 本地时间的 [`struct_time`](https://docs.python.org/zh-cn/3/library/time.html#time.struct_time) | [`localtime()`](https://docs.python.org/zh-cn/3/library/time.html#time.localtime) |
  | UTC 的 [`struct_time`](https://docs.python.org/zh-cn/3/library/time.html#time.struct_time) | 自纪元以来的秒数                                             | [`calendar.timegm()`](https://docs.python.org/zh-cn/3/library/calendar.html#calendar.timegm) |
  | 本地时间的 [`struct_time`](https://docs.python.org/zh-cn/3/library/time.html#time.struct_time) | 自纪元以来的秒数                                             | [`mktime()`](https://docs.python.org/zh-cn/3/library/time.html#time.mktime) |



## ctime()

将纪元秒数转换为以下形式的字符串：`Sun Jun 20 23:21:05 1993`（本地时间）。

```python
>>> time.ctime(0)
'Thu Jan  1 08:00:00 1970'
```



## gmtime()

将纪元秒数转换为UTC的 `struct_time` 对象。若未提供 `secs` 或为 `None`，则使用 `time()` 所返回的当前时间。

```python
>>> time.gmtime(0)
time.struct_time(tm_year=1970, tm_mon=1, tm_mday=1, tm_hour=0, tm_min=0, tm_sec=0, tm_wday=3, tm_yday=1, tm_isdst=0)
```



## localtime()

将纪元秒数转换为本地时间的 `struct_time` 对象。若未提供 `secs` 或为 `None`，则使用 `time()` 所返回的当前时间。

```python
>>> time.localtime(0)
time.struct_time(tm_year=1970, tm_mon=1, tm_mday=1, tm_hour=8, tm_min=0, tm_sec=0, tm_wday=3, tm_yday=1, tm_isdst=0)
```



## mktime()

`localtime()` 的反函数，将本地时间的 `struct_time` 对象转换为纪元秒数。

```python
>>> time.mktime(time.localtime(0))
0.0
```



## monotonic(), monotonic_ns()

返回以浮点数表示的一个单调时钟的值（秒），即不能倒退的时钟。该时钟不受系统时钟更新的影响。返回值的参考点未被定义，因此只有两次调用之间的差值才是有效的。



## process_time(), process_time_ns()

返回以浮点数表示的当前进程的系统和用户 CPU 时间的总计值（秒/纳秒），不包括睡眠状态所消耗的时间。返回值的参考点未被定义，因此只有两次调用之间的差值才是有效的。



## sleep()

调用该方法的线程将被暂停执行 *secs* 秒。参数可以是浮点数，以表示更为精确的睡眠时长。由于任何捕获到的信号都会终止`sleep()`引发的该睡眠过程并开始执行信号的处理例程，因此实际的暂停时长可能小于请求的时长；此外，由于系统需要调度其他活动，实际暂停时长也可能比请求的时间长。

```python
>>> time.sleep(1)
>>>      # after 1 sec
```



## strftime()

```python
time.strftime(format[, t])
```

将 `struct_time` 对象转换为指定格式的字符串。如果未提供 *t* ，则使用由 `localtime()` 返回的当前时间。 *format* 必须是一个字符串，可以嵌入以下指令：

| 指令 | 意义                                                         |
| :--- | :----------------------------------------------------------- |
| `%a` | 本地化的缩写星期中每日的名称。                               |
| `%A` | 本地化的星期中每日的完整名称。                               |
| `%b` | 本地化的月缩写名称。                                         |
| `%B` | 本地化的月完整名称。                                         |
| `%c` | 本地化的适当日期和时间表示。                                 |
| `%d` | 十进制数 [01,31] 表示的月中日。                              |
| `%H` | 十进制数 [00,23] 表示的小时（24小时制）。                    |
| `%I` | 十进制数 [01,12] 表示的小时（12小时制）。                    |
| `%j` | 十进制数 [001,366] 表示的年中日。                            |
| `%m` | 十进制数 [01,12] 表示的月。                                  |
| `%M` | 十进制数 [00,59] 表示的分钟。                                |
| `%p` | 本地化的 AM 或 PM 。                                         |
| `%S` | 十进制数 [00,61] 表示的秒。                                  |
| `%U` | 十进制数 [00,53] 表示的一年中的周数（星期日作为一周的第一天）。 在第一个星期日之前的新年中的所有日子都被认为是在第 0 周。 |
| `%w` | 十进制数 [0(星期日),6] 表示的周中日。                        |
| `%W` | 十进制数 [00,53] 表示的一年中的周数（星期一作为一周的第一天）。 在第一个星期一之前的新年中的所有日子被认为是在第 0 周。 |
| `%x` | 本地化的适当日期表示。                                       |
| `%X` | 本地化的适当时间表示。                                       |
| `%y` | 十进制数 [00,99] 表示的没有世纪的年份。                      |
| `%Y` | 十进制数表示的带世纪的年份。                                 |
| `%z` | 时区偏移以格式 +HHMM 或 -HHMM 形式的 UTC/GMT 的正或负时差指示，其中H表示十进制小时数字，M表示小数分钟数字 [-23:59, +23:59] 。 |
| `%Z` | 时区名称（如果不存在时区，则不包含字符）。                   |
| `%%` | 字面的 `'%'` 字符。                                          |

```python
>>> time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime(0))
'Thu, 01 Jan 1970 00:00:00 +0000'
>>> time.strftime("%Y/%m/%d %H:%M:%S", time.localtime(0))
'1970/01/01 08:00:00'
```



## strptime()

根据格式解析表示时间的字符串，返回一个 `struct_time` 对象。

```python
>>> time.strptime('1970/01/01 08:00:00', "%Y/%m/%d %H:%M:%S")
time.struct_time(tm_year=1970, tm_mon=1, tm_mday=1, tm_hour=8, tm_min=0, tm_sec=0, tm_wday=3, tm_yday=1, tm_isdst=-1)
```



## struct_time

结构化的时间类型。它是一个带有 [named tuple](https://docs.python.org/zh-cn/3/glossary.html#term-named-tuple) 接口的对象：可以通过索引和属性名访问值。 存在以下值：

| 索引 | 属性        | 值                                         |
| :--- | :---------- | :----------------------------------------- |
| 0    | `tm_year`   | （例如，1993）                             |
| 1    | `tm_mon`    | range [1, 12]                              |
| 2    | `tm_mday`   | range [1, 31]                              |
| 3    | `tm_hour`   | range [0, 23]                              |
| 4    | `tm_min`    | range [0, 59]                              |
| 5    | `tm_sec`    | range [0, 61]                              |
| 6    | `tm_wday`   | range [0, 6] ，周一为 0                    |
| 7    | `tm_yday`   | range [1, 366]                             |
| 8    | `tm_isdst`  | 1表示夏令时生效，0表示不生效，-1表示不确定 |
| N/A  | `tm_zone`   | 时区名称的缩写                             |
| N/A  | `tm_gmtoff` | 以秒为单位的UTC以东偏离                    |

当一个长度不正确的元组被传递给期望 `struct_time` 的函数，或者具有错误类型的元素时，会引发 `TypeError`。



## thread_time(), thread_time_ns()

返回以浮点数表示的当前线程的系统和用户 CPU 时间的总计值（秒/纳秒），不包括睡眠状态所消耗的时间。返回值的参考点未被定义，因此只有两次调用之间的差值才是有效的。



## time(), time_ns()

返回以浮点数表示的当前纪元秒数/纳秒数值。纪元的具体日期和闰秒的处理取决于平台。

```python
>>> time.time()
1617002884.9367008
```



## 时区常量

### altzone

本地夏令时时区的偏移量，以UTC为参照的秒数，如果已定义。如果本地夏令时时区在UTC以东，则为负数。

```python
>>> time.altzone
-28800
>>> 
```



### timezone

本地（非夏令时）时区的偏移量，以UTC为参照的秒数。如果本地时区在UTC以东，则为负数。

```python
>>> time.timezone
-28800
```



### tzname

两个字符串的元组：第一个是本地非夏令时时区的名称，第二个是本地夏令时时区的名称。 如果未定义夏令时时区，则不应使用第二个字符串。 

```python
>>> time.tzname
('CST', 'CST')
```





# types——动态类型创建和内置类型名称

`types` 模块为不能直接访问的内置类型定义了名称。

| 名称                                                   | 内置类型                                         |
| ------------------------------------------------------ | ------------------------------------------------ |
| `types.FunctionType`, `types.LambdaType`               | 用户自定义函数和 `lambda` 表达式创建的函数的类型 |
| `types.GeneratorType`                                  | 生成器类型                                       |
|                                                        |                                                  |
| `types.MethodType`                                     | 用户自定义实例方法的类型                         |
| `types.BuiltinFunctionType`, `types.BuiltinMethodType` | 内置函数和内置类型方法的类型                     |
|                                                        |                                                  |

最常见的用法是进行实例和子类检测：

```python
>>> def f(): pass
... 
>>> isinstance(f, types.FunctionType)            # 自定义函数
True
>>>
>>> isinstance(len, types.BuiltinFunctionType)   # 内置函数
True
```

除此之外，还可以用来动态创建内置类型：

```python
# 动态创建自定义实例方法并绑定到实例
>>> import types
>>> class Student(object):
    def set_name(self, name):
        self.name = name
... 
>>> bart = Student()
>>> bob = Student()
>>> bart.set_name('Bart')
>>> bob.set_name('Bob')
>>> def get_name(self):
    return self.name
... 
>>> bart.get_name = types.MethodType(get_name, bart)     # 将函数`get_name`动态绑定为实例`bart`的方法
>>> bart.get_name()                                      # 注意若类设定了`__slots__`则无法绑定
'Bart'
>>> dir(bart)
['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', 'get_name', 'name', 'set_name']
>>> bob.get_name()                                       # 其它实例无法调用
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: 'Student' object has no attribute 'get_name'
>>> def set_name(self, name):
    self.name = name
    print('Succeeded.')
... 
>>> bart.set_name = types.MethodType(set_name, bart)     # 覆盖实例方法
>>> bart.set_name('Bartt')                               # 注意若实例方法设为只读则无法覆盖
Succeeded.
```





# weakref——弱引用

