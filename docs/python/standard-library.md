[toc]



# [内置函数](https://docs.python.org/3/library/functions.html)

## abs()

返回一个数的绝对值。参数可以是整数、浮点数或任何实现了 `__abs__()` 的对象。如果参数是一个复数，则返回它的模。

```python
>>> abs(-1)
1
>>> abs(-1.2)
1.2
>>> abs(complex(1, 2))
2.23606797749979
```



## all()

如果可迭代对象的所有元素均为真值则返回 `True`；如果可迭代对象为空也返回 `True`。

```python
>>> all(range(5))      # 包含0
False
>>> all(range(1, 5))
True
>>> all([])            # 可迭代对象为空
True
```



## any()

如果可迭代对象的任一元素为真值则返回 `True`；如果可迭代对象为空则返回 `False`。

```python
>>> any(range(5))
True
>>> any([])            # 可迭代对象为空
False
```



## bin()

将整数转换为前缀为 `'0b'` 的二进制字符串。如果参数不是  `int` 类型，那么它需要定义 `__index__()` 方法返回一个整数。

```python
>>> bin(3)
'0b11'
>>> bin(-10)
'-0b1010'
>>> bin(0b1100)
'0b1100'
```



## bool

```python
class bool([x])
```

返回一个布尔值，`True` 或者 `False`。参数使用标准的逻辑值检测过程进行转换，参见数据类型-逻辑值检测。`bool` 类是 `int` 类的子类，只有 `False` 和 `True` 两个实例，参见数据类型-布尔值。



## bytearray



## callable()

如果参数是可调用的则返回 `True`，否则返回 `False`。但即使返回 `True`，调用该对象仍可能失败。请注意类实例是可调用的（调用将返回一个该类的新的实例）；定义了 `__call__()` 方法的类的实例是可调用的。



## chr()

返回 Unicode 码位为输入整数的 Unicode 字符的字符串格式。是 `ord()` 的反函数。

```python
>>> chr(97)
'a'
>>> chr(0x4e2d)
'中'
```



## @classmethod

将方法封装为类方法。

参见面向对象编程-方法。



## compile()



## complex

```python
class complex([real[, imag]])
```

返回值为 *real*+i *imag* 的复数，或者将字符串或数字转换为复数。

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

对于一个普通 Python 对象 `x`，`complex(x)` 会委托给 `x.__complex__()`。如果 `__complex__()` 未定义则将回退至 `__float__()`；如果 `__float__(`）未定义则将回退至 `__index__()`。



## delattr()

```python
delattr(object, name)
```

与 `setattr()` 对应。删除对象 *object* 的名为 *name* 的属性，*name* 必须是字符串，指定一个现有属性。如果对象允许，该函数将删除指定的属性。例如 `delattr(x,'foobar')` 等同于 `del x.foobar`。



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

* 如果对象是模块对象，则列表包含模块的属性名称。
* 如果对象是类型或类对象，则列表包含它们的属性名称，并且递归查找所有基类的属性。
* 如果对象是实例，则列表包含对象的属性名称，它的类属性名称，并且递归查找它的类的所有基类的属性。

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

将两个数字（整数或浮点数）作为实参，执行整数除法并返回一对商和余数。对于整数，结果和 `(a//b,a % b)`一致。对于浮点数，结果是`(q,a % b)`，`q`通常是`math.floor(a/b)`，但可能会比 1 小。

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
>>> format('Alice', 's')   # 's': 字符串
'Alice'
>>> format(123, 'b')       # 'b': 二进制整数
'1111011'
>>> format(123, 'o')       # 'o': 八进制整数
'173'
>>> format(123, 'd')       # 'd': 十进制整数
'123'
>>> format(123, 'x')       # 'x': 十六进制整数
'7b'

```



## getattr()

```python
getattr(object, name[, default])
```

与 `setattr()` 对应。返回对象 *object* 的 *name* 属性的值，*name* 必须是字符串。如果该字符串是对象的属性之一，则返回该属性的值，例如 `getattr(x,'foobar')` 等同于 `x.foobar`；如果指定的属性不存在，但提供了 *default* 值，则返回它，否则触发 `AttributeError`。



## hasattr()

```python
hasattr(object, name)
```

返回对象 *object* 是否具有名为 *name* 的属性，*name* 必须是字符串。如果字符串是对象的属性之一的名称，则返回 `True`，否则返回 `False`。（此功能是通过调用 `getattr(object,name)` 看是否有 `AttributeError` 异常来实现的。）



## hash()

返回对象的哈希值（如果它有的话）。



## hex()

将整数转换为前缀为 `'0x'` 的小写十六进制字符串。如果参数不是  `int` 类型，那么它需要定义 `__index__()` 方法返回一个整数。

```python
>>> hex(255)
'0xff'
>>> hex(-42)
'-0x2a'
>>> hex(-0x2a)
'-0x2a'
```



## id()

返回对象的标识值，是一个整数，并且在此对象的生命周期内保证是唯一且恒定的。两个生命周期不重叠的对象可能具有相同的 `id()` 值。



## int

> `int()` 习惯上也称为内置函数，尽管它表示 `int` 类的实例化。`float()`, `complex()`, `bool()` 等同理。

```python
class int([x])
class int(x, base=10)
```

如果 *x* 是整数字面值或字符串，则返回基于 *x* 构造的整数对象；如果 *x* 是浮点数，则将 *x* 向零舍入；如果 *x* 定义了 `__int__()`，则返回 `x.__int__()`；如果 *x* 定义了 `__index__()`，则返回 `x.__index__()`；如果 *x* 定义了 `__trunc__()`，则返回 `x.__trunc__()`；如果未传入参数，则返回 `0`。

```python
>>> int(1)            # 整数字面值
1
>>> int('2')          # 字符串
2
>>> int(1.2)          # 浮点数
1
>>> int(0b1100)       # 二进制整数
12
>>> int(0o1472)       # 八进制整数
826
>>> int(0xff00)       # 十六进制整数
65280


>>> class Student:
...   def __index__(self):    # 定义了`__index__()`
...     return 1
... 
>>> s = Student()
>>> int(s)
1

>>> int()
0
```

如果有 *base* 参数，则 *x* 必须是表示进制为 *base* 的整数字面值的字符串、`bytes` 或 `bytearray` 实例，该字符串/字节串前可以有 `+` 或 `-` （中间不能有空格），前后可以有空格。允许的进制有 0、2-36，其中 2, 8, 16 进制允许字符串/字节串加上前缀 `'0b'`/`'0B'`, `'0o'`/`'0O'`, `'0x'`/`'0X'`（也可以不加）。进制为 0 表示按照字符串/字节串的前缀确定进制是 2, 8, 10 还是 16。

一个进制为 n 的整数可以使用 0 到 n-1 的数字，其中 10 到 35 用 `a` 到 `z` （或 `A` 到 `Z` ）表示。

```python
>>> int('ff00', base=16)       # 不使用前缀'0x'
65280
>>> int('0xff00', base=16)     # 使用前缀'0x'
65280
>>> int('0xff00', base=0)      # 通过前缀识别进制为16
65280
```



## isinstance()

如果 *object* 是 *classinfo* 的实例或（直接、间接或虚拟）子类则返回 `True`。*classinfo* 可以是类对象的元组，在此情况下 *object* 是其中任何一个类的实例就返回 `True`。

参见面向对象编程-获取对象信息。



## issubclass()

如果 *class* 是 *classinfo* 的（直接、间接或虚拟）子类则返回 `True`。类会被视作其自身的子类。*classinfo* 可以是类对象的元组，在此情况下 *classinfo* 中的每个条目都将被检查。



## iter()

返回一个迭代器对象。

```python
iter(object[, sentinel])
```

如果没有第二个实参，*object* 必须支持迭代器协议（提供 `__iter__()` 方法）、序列协议（提供 `__getitem__()` 方法且键为从 0 开始的整数）或映射协议（提供 `__getitem__()` 方法和 `keys()` 方法（返回键的可迭代对象）），如果它不支持这些协议，会引发 `TypeError`。如果有第二个实参 *sentinel*，那么 *object* 必须是可调用对象，这种情况下返回的迭代器，每次调用 `__next__()` 方法时都会不带实参地调用 *object*；如果返回的结果是 *sentinel* 则触发 `StopIteration`，否则返回调用结果。



```python
# 对象具有`__iter__()`方法


```



```python
# 对象具有`__getitem__()`方法
>>> class Student(object):
    def __init__(self, name, score1, score2, score3):
        self.name = name
        self.score1 = score1
        self.score2 = score2
        self.score3 = score3
    def __getitem__(self, key):        # `key`为从0开始的整数
        if isinstance(key, slice):
            print(slice)
            return [self[i] for i in slice]
        elif isinstance(key, int):
            if key < 0 :               # Handle negative indices
                key += 3
            if key < 0 or key >= 3:
                raise IndexError("The index {} is out of range".format(key))
            return self.get_data(key)  # Get the data from elsewhere
        else:
            raise TypeError("Invalid argument type")
    def get_data(self, key):
        if key == 0:
            return self.score1
        if key == 1:
            return self.score2
        if key == 2:
            return self.score3
        return
...
>>> bart = Student('Bart Simpson', 59, 60, 61)
>>> list(iter(bart))
[59, 60, 61]
```

```python
>>> class Student(object):
    def __init__(self, name, score, age):
        self._info = {'name': name, 'score': score, 'age': age}
    def keys(self):
        return self._info.keys()
    def __getitem__(self, key):        # `key`由迭代`keys()`方法返回的可迭代对象得到
        return self._info[key]
...
>>> bart = Student('Bart Simpson', 59, 10)
>>> dict(bart)
{'name': 'Bart Simpson', 'score': 59, 'age': 10}
```











## len()

返回对象的长度（元素个数）。实参可以是序列（如 string、bytes、tuple、list 或 range 等）或集合（如 dictionary、set 或 frozen set 等）。



## locals()

更新并返回表示当前本地符号表的字典。

符号表是由 Python 解释器维护的数据结构，



 在函数代码块但不是类代码块中调用 `locals()` 时将返回自由变量。

```python
```





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



## object

返回一个没有属性的新实例。`object` 是所有类的基类。

```python
>>> obj = object()
>>> dir(obj)
['__class__', '__delattr__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__']
>>> dir(object)
['__class__', '__delattr__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__']
```





## next()





## oct()

将整数转换为前缀为 `'0o'` 的八进制字符串。如果参数不是  `int` 类型，那么它需要定义 `__index__()` 方法返回一个整数。

```python
>>> oct(8)
'0o10'
>>> oct(-56)
'-0o70'
>>> oct(-0o70)
'-0o70'
```



## open()

参见



## ord()

返回字符串中的单个 Unicode 字符的 Unicode 码位的十进制整数表示。是 `chr()` 的反函数。

```python
>>> ord('a')
97
>>> ord('中')
20013
>>> hex(ord('中'))
'0x4e2d'
```



## pow()

```python
pow(base, exp[, mod])
```

返回 *base* 的 *exp* 次幂；如果 *mod* 存在，则返回 *base* 的 *exp* 次幂对 *mod* 取余（比 `pow(base,exp)% mod`更高效）。两参数形式`pow(base,exp)`等价于乘方运算符：`base**exp`。

```python
>>> pow(16, 2)
256
>>> pow(16, 2, mod=15)
1

>>> pow(0, 0)
1                     # 惯例
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

一个典型的用法是定义一个托管属性 `x`：

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

`property` 对象具有 `getter`，`setter` 和 `deleter` 方法，可以用作装饰器创建该 `property` 对象的副本，并将相应的方法设为所装饰的函数：

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

与 `getattr()` 和 `delattr()` 对应。设置对象 *object* 的 *name* 属性的值，*name* 必须是字符串，指定一个现有属性或新增属性。如果对象允许，该函数将设置指定的属性。例如 `setattr(x,'foobar',123)` 等同于 `x.foobar =123`。



## sorted()

根据可迭代对象中的项返回一个新的已排序列表。

```python

```





## @staticmethod

将方法转换为静态方法。

参见面向对象编程-类-函数。



## str()

返回一个对象的 `str` 版本，`str` 是内置字符串类型。

详见数据类型-字符串。



## sum()

从 *start* 开始自左向右对可迭代对象的项求和并返回总计值。可迭代对象的项通常为数字，而 *start* 值则不允许为字符串。

要拼接字符串序列，更好的方式是调用 `''.join(sequence)`；要以扩展精度对浮点值求和，请使用 `math.fsum()`；要拼接一系列可迭代对象，请使用 `itertools.chain()`。



## super()



## tuple()

创建一个新的元组。参见数据结构-元组。



## type

```python
class type(object)               # 表示传入一个对象;不是类的定义
class type(name, bases, dict)
```

传入一个参数时，返回 *object* 的类型。返回值是一个 `type` 对象，通常与 `object.__class__` 所返回的对象相同。

```python
>>> a = 1
>>> type(a)
<class 'int'>
>>> a.__class__
<class 'int'>
```

检测对象类型推荐使用 `isinstance()`，因为它会考虑子类的情况。

传入三个参数时，返回一个新的 `type` 对象。这在本质上是 `class` 语句的一种动态形式，*name* 字符串即类名并会成为 `__name__` 属性；*bases* 元组包含基类并会成为 `__bases__` 属性；如果为空则会添加所有类的终极基类 `object`；*dict* 字典包含类主体的属性和方法定义，它在成为 `__dict__` 属性之前可能会被拷贝或包装。 

```python
>>> X = type('X', (), dict(a=1, f=abs))   # 类本身是一个`type`实例
>>> # 相当于
>>> class X:
...     a = 1
...     f = abs
```



## zip()

创建一个迭代器，它返回的第 *i* 个元组包含来自每个输入可迭代对象的第 *i* 个元素。当所输入可迭代对象中最短的一个被耗尽时，迭代器将停止迭代。

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



## \__import__

被 `import` 语句调用以导入模块。建议使用 `importlib.import_module()` 而非此函数来导入模块。





# [argparse](https://docs.python.org/zh-cn/3/library/argparse.html)——命令行选项、参数和子命令解析器

如果脚本很简单或者临时使用，可以使用 `sys.argv` 直接读取命令行参数。`sys.argv` 返回一个参数列表，其中首个元素是程序名，随后是命令行参数，所有元素都是字符串类型。例如以下脚本：

```python
# test.py

import sys

print("Input argument is %s" %(sys.argv))
```

```shell
$ python3 test.py 1 2 -a 3
Input argument is ['test.py', '1', '2', '-a', '3']
```



`argparse` 模块可以让人轻松编写用户友好的命令行接口。程序定义它需要的参数，然后 `argparse` 将弄清如何从 `sys.argv` 解析出那些参数。`argparse` 模块还会自动生成帮助和使用手册，并在用户给程序传入无效参数时报出错误信息。

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





# base64——Base16, Base32, Base64, Base85 数据编码

`base64` 模块提供了将二进制数据编码为可打印的 ASCII 字符以及将这些编码解码回二进制数据的函数，即 RFC 3548 指定的 Base16、Base32 和 Base64 编码以及已被广泛接受的 Ascii85 和 Base85 编码的编码和解码函数。



## b64encode()

对类似字节序列的对象进行 Base64 编码，返回编码后的字节序列。

```python
>>> import base64
>>> encoded = base64.b64encode(b'data to be encoded')
>>> encoded
b'ZGF0YSB0byBiZSBlbmNvZGVk'
>>> data = base64.b64decode(encoded)
>>> data
b'data to be encoded'
```



## b64decode()

对 Base64 编码过的类似字节序列的对象进行解码，返回解码后的字节序列。







# builtins——内建对象

`builtins` 模块提供对 Python 的所有内置对象的直接访问，例如 `builtins.open` 是内置函数 `open()` 的全名。

大多数应用程序通常不会显式访问此模块，但在扩展内置对象时会很有用。

```python
>>> import builtins
>>> dir(builtins)
['ArithmeticError', 'AssertionError', 'AttributeError', 'BaseException', 'BlockingIOError', 'BrokenPipeError', 'BufferError', 'BytesWarning', 'ChildProcessError', 'ConnectionAbortedError', 'ConnectionError', 'ConnectionRefusedError', 'ConnectionResetError', 'DeprecationWarning', 'EOFError', 'Ellipsis', 'EnvironmentError', 'Exception', 'False', 'FileExistsError', 'FileNotFoundError', 'FloatingPointError', 'FutureWarning', 'GeneratorExit', 'IOError', 'ImportError', 'ImportWarning', 'IndentationError', 'IndexError', 'InterruptedError', 'IsADirectoryError', 'KeyError', 'KeyboardInterrupt', 'LookupError', 'MemoryError', 'ModuleNotFoundError', 'NameError', 'None', 'NotADirectoryError', 'NotImplemented', 'NotImplementedError', 'OSError', 'OverflowError', 'PendingDeprecationWarning', 'PermissionError', 'ProcessLookupError', 'RecursionError', 'ReferenceError', 'ResourceWarning', 'RuntimeError', 'RuntimeWarning', 'StopAsyncIteration', 'StopIteration', 'SyntaxError', 'SyntaxWarning', 'SystemError', 'SystemExit', 'TabError', 'TimeoutError', 'True', 'TypeError', 'UnboundLocalError', 'UnicodeDecodeError', 'UnicodeEncodeError', 'UnicodeError', 'UnicodeTranslateError', 'UnicodeWarning', 'UserWarning', 'ValueError', 'Warning', 'ZeroDivisionError', '__build_class__', '__debug__', '__doc__', '__import__', '__loader__', '__name__', '__package__', '__spec__', 'abs', 'all', 'any', 'ascii', 'bin', 'bool', 'breakpoint', 'bytearray', 'bytes', 'callable', 'chr', 'classmethod', 'compile', 'complex', 'copyright', 'credits', 'delattr', 'dict', 'dir', 'divmod', 'enumerate', 'eval', 'exec', 'exit', 'filter', 'float', 'format', 'frozenset', 'getattr', 'globals', 'hasattr', 'hash', 'help', 'hex', 'id', 'input', 'int', 'isinstance', 'issubclass', 'iter', 'len', 'license', 'list', 'locals', 'map', 'max', 'memoryview', 'min', 'next', 'object', 'oct', 'open', 'ord', 'pow', 'print', 'property', 'quit', 'range', 'repr', 'reversed', 'round', 'set', 'setattr', 'slice', 'sorted', 'staticmethod', 'str', 'sum', 'super', 'tuple', 'type', 'vars', 'zip']
```





# collections——容器数据类型

参见 [`collections` 容器数据类型](./container-type.md#`collections` 容器数据类型)。





# collections.abc——容器的抽象基类

参见 [自定义容器数据类型](./container-type.md#自定义容器数据类型)。





# copy——浅层和深层复制操作

Python 的赋值语句不复制对象，而是创建目标和对象的绑定关系。对于自身可变，或包含可变项的集合，有时要生成副本用于改变操作，而不必改变原始对象。此模块提供了通用的浅层复制和深层复制操作。



* `copy()`：返回对象的浅层复制。
* `deepcopy()`：返回对象的深层复制。

浅层与深层复制的区别仅与复合对象（即包含列表、字典或类的实例等其他对象的对象）相关：

* *浅层复制*构造一个新的复合对象，然后（在尽可能的范围内）将原始对象中找到的对象的*引用*插入其中。
* *深层复制*构造一个新的复合对象，然后递归地将在原始对象里找到的对象的*副本*插入其中。

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

* 递归对象（直接或间接包含对自身引用的复合对象）可能会导致递归循环。
* 由于深层复制会复制所有内容，因此可能会过多复制（例如本应该在副本之间共享的数据）。

`deepcopy()` 函数用以下方式避免了这些问题：

* 保留在当前复制过程中已复制的对象的"备忘录"（`memo`）字典
* 允许用户定义的类重载复制操作或复制的组件集合。



制作字典的浅层复制可以使用 `dict.copy()` 方法，而制作列表的浅层复制可以通过赋值整个列表的切片完成，例如，`copied_list = original_list[:]`。



# csv——CSV 文件读写

`csv` 模块实现了 csv 格式表单数据的读写。其提供了诸如“以兼容 Excel 的方式输出数据文件”或“读取 Excel 程序输出的数据文件”的功能，程序员无需知道 Excel 所采用 csv 格式的细节。此模块同样可以用于定义其他应用程序可用的 csv 格式或定义特定需求的 csv 格式。

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





# datetime——基本日期和时间类型

## timedelta

`timedelta` 对象表示两个 date 或者 time 之间的时间间隔。

```python
class datetime.timedelta(days=0, seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0)
```

所有这些参数都是可选的并且默认为 `0`，可以是整数或者浮点数，也可以是正数或者负数。

只有 *days*, *seconds* 和 *microseconds* 会存储在内部，各参数单位的换算规则如下：

- 1毫秒会转换成1000微秒。
- 1分钟会转换成60秒。
- 1小时会转换成3600秒。
- 1星期会转换成7天。

并且 *days*, *seconds* 和 *microseconds* 会经标准化处理以保证表达方式的唯一性，即：

- `0 <= microseconds < 1000000`
- `0 <= seconds < 3600*24`
- `-999999999 <= days <= 999999999`

```python
>>> from datetime import timedelta
>>> delta = timedelta(
...     weeks=2,              # 1星期转换成7天
...     days=50,
...     hours=8,              # 1小时转换成3600秒
...     minutes=5,            # 1分钟转换成60秒
...     seconds=27,
...     milliseconds=29000,   # 1毫秒转换成1000微秒,或1000毫秒转换成1秒
...     microseconds=10
... )
>>> delta
datetime.timedelta(64, 29156, 10)   # 64天,29156秒,10毫秒
>>> delta.total_seconds()
5558756.00001                       # 秒
```



下面演示了 `timedelta` 对象支持的运算：

```python
>>> t1 = timedelta(minutes=10)
>>> t2 = timedelta(seconds=10)
>>> t3 = timedelta(seconds=11)
>>> t1 + t2
datetime.timedelta(seconds=610)
>>> t1 - t2
datetime.timedelta(seconds=590)
>>> t2 * 2
datetime.timedelta(seconds=1200)
>>> t2 * 1.234                           # 乘以浮点数,结果舍入到微秒的整数倍
datetime.timedelta(seconds=12, microseconds=340000)
>>> t2 * 1.23456789
datetime.timedelta(seconds=12, microseconds=345679)
>>> t2 / 3                               # 除以整数或浮点数,结果舍入到微秒的整数倍
datetime.timedelta(seconds=3, microseconds=333333)
>>> t1 / t3                              # `timedelta`对象相除,返回一个浮点数
54.54545454545455
>>> t1 // t3                             # 带余除法取商
54
>>> t1 % t3                              # 带余除法取余
datetime.timedelta(seconds=6)
>>> -t1                                  # 相反的时间间隔
datetime.timedelta(days=-1, seconds=85800)
>>> abs(-t1)                             # 取绝对值
datetime.timedelta(seconds=600)
>>> t2 < t3                              # 比较
True
>>> str(t1)
'0:10:00'
>>> repr(t1)
'datetime.timedelta(seconds=600)'
```

除此之外，`timedelta` 对象还支持与 `date` 和 `datetime` 对象进行特定的相加和相减运算。



### days

（实例属性）天数。



### microseconds

（实例属性）微秒数。



### seconds

（实例属性）秒数。



### total_seconds()

（实例方法）返回时间间隔总共包含的秒数。



## date

`date` 对象代表一个理想化历法的日期，它假设当今的公历在过去和未来永远有效。

```python
class datetime.date(year, month, day)
```

所有参数都是必要的并且必须是在下列范围内的整数：

- `1 <= year <= 9999`
- `1 <= month <= 12`
- `1 <= day <= 给定年月对应的天数`

如果参数不在这些范围内，则抛出 `ValueError` 异常。



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



### day

（实例属性）





### month

（实例属性）



### today()

（类方法）



### year

（实例属性）



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

* `0<= hour <24`
* `0<= minute <60`
* `0<= second <60`
* `0<= microsecond <1000000`
* `fold in[0,1]`

如果给出一个此范围以外的参数，则会引发 `ValueError`。所有参数值默认为 0，只有 `tzinfo` 默认为 `None`。

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

表示两个 `date`、`time` 或 `datetime` 对象之间的时间间隔，精确到微秒。





## tzinfo

一个描述时区信息对象的抽象基类。用来给 `datetime` 和 `time` 类提供自定义的时间调整概念（例如处理时区和/或夏令时）。



## timezone

一个实现了 `tzinfo` 抽象基类的子类，用于表示相对于 UTC 的偏移量。





# enum——对枚举的支持

## 模块内容

### Enum

创建枚举常量的基类。



### IntEnum

创建 `int` 类枚举常量的基类。



### IntFlag

创建可与位运算符搭配使用，又不失去 `IntFlag` 成员资格的枚举常量的基类。



### Flag

创建可与位运算符搭配使用，又不失去 `Flag` 成员资格的枚举常量的基类。



### unique()

确保一个名称只绑定一个值的 Enum 类装饰器。



### auto

以合适的值代替 Enum 成员的实例。初始值默认从 1 开始。



## 定义枚举类

继承 `Enum` 类以定义枚举类，例如：

```python
>>> from enum import Enum
>>> class Color(Enum):        # `Color`是枚举类
    RED = 1                   # `Color.RED`是枚举类的成员,其中`RED`是名称,`1`是值
    GREEN = 2
    BLUE = 3
>>> print(Color.RED)          # 成员的打印结果
Color.RED
>>> Color.RED
<Color.RED: 1>
>>> type(Color.RED)           # 成员的类型
<enum 'Color'>
>>> isinstance(Color.RED, Color)
True
>>> print(Color.RED.name)     # 成员的名称
RED
>>> print(Color.RED.value)    # 成员的值
1
>>> list(Color)               # 迭代枚举类
[<Color.RED: 1>, <Color.GREEN: 2>, <Color.BLUE: 3>]
```

> 枚举类中定义的所有类属性将成为该枚举类的成员。
>
> 枚举类表示的是常量，因此建议成员名称使用大写字母；以单下划线开头和结尾的名称由枚举保留而不可使用。
>
> 尽管枚举类同样由 `class` 语法定义，但它并不是常规的 Python 类，详见 [How are Enums different?](https://docs.python.org/zh-cn/3/library/enum.html#how-are-enums-different)。

除了 `Color.RED`，成员还支持如下访问方式：

```python
>>> Color['RED']
<Color.RED: 1>
>>> Color(1)
<Color.RED: 1>
```

成员的值一般设定为整数、字符串等。若成员取何值并不重要，则可以使用 `auto()` 自动为成员分配值：

```python
>>> from enum import Enum, auto
>>> class Color(Enum):
    RED = auto()
    GREEN = auto()
    BLUE = auto()
... 
>>> list(Color)
[<Color.RED: 1>, <Color.GREEN: 2>, <Color.BLUE: 3>]
```

`auto()` 的行为可以由重载 `_generate_next_value_()` 方法来定义：

```python
>>> class Color(Enum):
    def _generate_next_value_(name, start, count, last_values):  # 必须定义在任何成员之前
        return name
    RED = auto()
    GREEN = auto()
    BLUE = auto()
... 
>>> list(Color)
[<Color.RED: 'RED'>, <Color.GREEN: 'GREEN'>, <Color.BLUE: 'BLUE'>]
```

成员的值可哈希，因此成员可用于字典和集合：

```python
>>> apples = {}
>>> apples[Color.RED] = 'red delicious'
```



## 重复的成员值

两个成员的名称不能相同，但值可以相同，此时后定义的成员的名称将作为先定义的成员的别名：

```python
>>> class Shape(Enum):
    SQUARE = 2
    SQUARE = 3
...
TypeError: Attempted to reuse key: 'SQUARE'
>>> 
>>> class Shape(Enum):
    SQUARE = 2
    DIAMOND = 1
    CIRCLE = 3
    ALIAS_FOR_SQUARE = 2     # 作为`SQUARE`的别名
... 
>>> Shape.SQUARE
<Shape.SQUARE: 2>
>>> Shape.ALIAS_FOR_SQUARE
<Shape.SQUARE: 2>
>>> Shape(2)
<Shape.SQUARE: 2>
```

迭代枚举类时不会给出别名；枚举类的特殊属性 `__members__` 是从名称到成员的只读有序映射，其包含别名在内的所有名称：

```python
>>> list(Shape)
[<Shape.SQUARE: 2>, <Shape.DIAMOND: 1>, <Shape.CIRCLE: 3>]
>>> list(Shape.__members__.items())
[('SQUARE', <Shape.SQUARE: 2>), ('DIAMOND', <Shape.DIAMOND: 1>), ('CIRCLE', <Shape.CIRCLE: 3>), ('ALIAS_FOR_SQUARE', <Shape.SQUARE: 2>)]
>>> [name for name, member in Shape.__members__.items() if member.name != name]  # 找出所有别名
['ALIAS_FOR_SQUARE']
```

如果想要禁用别名，则可以使用装饰器 `unique`：

```python
>>> from enum import Enum, unique
>>> @unique
class Shape(Enum):
    SQUARE = 2
    DIAMOND = 1
    CIRCLE = 3
    ALIAS_FOR_SQUARE = 2
...
ValueError: duplicate values found in <enum 'Shape'>: ALIAS_FOR_SQUARE -> SQUARE
```



## 比较

成员之间按照标识值进行比较：

```python
>>> Color.RED is Color.RED
True
>>> Color.RED is Color.GREEN
False
>>> Color.RED == Color.RED
True
>>> Color.RED == Color.GREEN
False
```

成员之间的排序比较不被支持：

```python
>>> Color.RED < Color.BLUE
...
TypeError: '<' not supported between instances of 'Color' and 'Color'
```

成员与其它类型的实例的比较将总是不相等：

```python
>>> Color.RED == 1
False
```



## 枚举类的方法

枚举类是特殊的 Python 类，同样可以具有普通方法和特殊方法，例如：

```python
>>> class Mood(Enum):
    FUNKY = 1
    HAPPY = 3
    def describe(self):       # self here is the member
        return self.name, self.value
    def __str__(self):
        return 'my custom str! {0}'.format(self.value)
    @classmethod
    def favorite_mood(cls):   # cls here is the enumeration
        return cls.HAPPY
... 
>>> Mood.favorite_mood()
<Mood.HAPPY: 3>
>>> Mood.HAPPY.describe()
('HAPPY', 3)
>>> print(Mood.FUNKY)
my custom str! 1
```



## 继承枚举类

一个新的枚举类必须继承自一个既有的枚举类，并且父类不可定义有任何成员，因此禁止下列定义：

```python
>>> class MoreColor(Color):
    PINK = 17
......
TypeError: MoreColor: cannot extend enumeration 'Color'
```

但是允许下列定义：

```python
>>> class Foo(Enum):
    def some_behavior(self):
        pass
... 
>>> class Bar(Foo):
    HAPPY = 1
    SAD = 2
... 
```



## 功能性API

`Enum` 类属于可调用对象，它提供了以下功能性 API：

```python
>>> Animal = Enum('Animal', 'ANT BEE CAT DOG')
>>> Animal
<enum 'Animal'>
>>> Animal.ANT
<Animal.ANT: 1>
>>> Animal.ANT.value
1
>>> list(Animal)
[<Animal.ANT: 1>, <Animal.BEE: 2>, <Animal.CAT: 3>, <Animal.DOG: 4>]
```

`Enum` 的第一个参数是枚举的名称；第二个参数是枚举成员名称的来源，它可以是一个用空格分隔的名称字符串、名称序列、键值对二元组的序列或者名称到值的映射，最后两种选项使得可以为枚举任意赋值，而其他选项会自动以从 1 开始递增的整数赋值（使用 `start` 形参可指定不同的起始值）。返回值是一个继承自 `Enum` 的新类，换句话说，上述对 `Animal` 的赋值就等价于:

```python
>>> class Animal(Enum):
    ANT = 1
    BEE = 2
    CAT = 3
    DOG = 4
```

默认以 `1` 而以 `0` 作为起始数值的原因在于 `0` 的布尔值为 `False`，但所有枚举成员都应被求值为 `True`。



## IntEnum

`IntEnum` 是 `Enum` 的一个变种，同时也是 `int` 的一个子类。`IntEnum` 的成员可以与整数进行比较，不同 `IntEnum` 子类的成员也可以互相比较：

```python
>>> from enum import IntEnum
>>> class Shape(IntEnum):
    CIRCLE = 1
    SQUARE = 2
... 
>>> class Request(IntEnum):
    POST = 1
    GET = 2
... 
>>> Shape == 1
False
>>> Shape.CIRCLE == 1
True
>>> Shape.CIRCLE == Request.POST
True
```

`IntEnum` 成员的值在其它方面的行为都类似于整数：

```python
>>> ['a', 'b', 'c'][Shape.CIRCLE]
'b'
```



## IntFlag

`IntFlag` 变种同样基于 `int`，与 `IntEnum` 的不同之处在于 `IntFlag` 成员可以使用按位运算符进行组合并且结果仍然为 `IntFlag` 成员：

```python
>>> from enum import IntFlag
>>> class Perm(IntFlag):
    R = 4
    W = 2
    X = 1
... 
>>> Perm.R | Perm.W
<Perm.R|W: 6>
>>> type(Perm.R | Perm.W)
<enum 'Perm'>
>>> Perm.R in (Perm.R | Perm.W)
True
>>> Perm.R + Perm.W
6
>>> Perm.R | 8               # 与整数进行组合
<Perm.8|R: 12>
>>> type(Perm.R | 8)         # 仍为`IntFlag`成员
<enum 'Perm'>
```

`IntFlag` 和 `IntEnum` 的另一个重要区别在于如果值为 0，则其布尔值为 `False`：

```python
>>> Perm.R & Perm.X
<Perm.0: 0>
>>> bool(Perm.R & Perm.X)
False
```



## Flag

`Flag` 变种与 `IntFlag` 类似，成员可以使用按位运算符进行组合，但不同之处在于成员不可与其它 `Flag` 子类的成员或整数进行组合或比较。虽然可以直接指定值，但推荐使用 `auto` 选择适当的值。

```python
>>> from enum import Flag, auto
>>> class Color(Flag):
    BLACK = 0                    # 定义值为0的flag
    RED = auto()                 # 1
    BLUE = auto()                # 2
    GREEN = auto()               # 4
    WHITE = RED | BLUE | GREEN   # 定义作为多个flag的组合的flag
... 
>>> Color.RED & Color.GREEN
<Color.0: 0>
>>> bool(Color.RED & Color.GREEN)
False
>>> Color.BLACK
<Color.BLACK: 0>
>>> bool(Color.BLACK)
False
>>> Color.WHITE
<Color.WHITE: 7>

```





# functools——高阶函数和可调用对象上的操作

## partial()

返回一个新的 partial 对象，当被调用时其行为类似于 *func* 附带位置参数 *args* 和关键字参数 *keywords* 被调用。如果为调用提供了更多的参数，它们会被附加到 *args*。如果提供了额外的关键字参数，它们会扩展并重载 *keywords*。大致等价于：

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

使用 lambda 函数 `lambda x: int(x, base=2)` 也能起到相同的作用。



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


# hmac——基于密钥的消息验证




# importlib——import的实现

## \__import__()

内置 `__import__()` 函数的实现。



## import_module()

```python
importlib.import_module(name, package=None)
```

导入一个模块。参数 *name* 指定以绝对或相对导入方式导入的模块；如果参数 *name* 使用相对导入方式，那么参数 *packages* 必须设置为相应的包名，并作为解析模块名的锚点，例如：

```python
>>> import importlib
>>> importlib.import_module('numpy')       # 返回指定的模块(或包)实例
<module 'numpy' from '/Users/xyx/.pyenv/versions/3.8.7/lib/python3.8/site-packages/numpy/__init__.py'>
>>> np = importlib.import_module('numpy')
>>> np.__version__
'1.19.5'
>>> np.arange(6)
array([0, 1, 2, 3, 4, 5])
```





# inspect——检查对象

## currentframe()

返回调用者栈帧的帧对象。



## ismodule()

若对象为模块，返回 `True`。



## isclass(), isabstract()

若对象为类/抽象基类，返回 `True`。



## isfunction(), ismethod(), isroutine()

若对象为函数/绑定的方法/函数或方法，返回 `True`。



## getdoc()

返回对象的 `docstring`。



## getfile(), getsourcefile()

返回对象被定义的模块的路径。



## getmodule()

猜测对象被定义的模块。



## getsource()

返回对象的源代码。

```python
>>> print(inspect.getsource(json.JSONDecoder))     # 类的源代码
class JSONDecoder(object):
...
    
>>> print(inspect.getsource(json.loads))           # 函数的源代码
def loads(s, *, cls=None, object_hook=None, parse_float=None,
        parse_int=None, parse_constant=None, object_pairs_hook=None, **kw):
...

>>> print(inspect.getsource(json))                 # 模块的源代码
r"""JSON (JavaScript Object Notation) <http://json.org> is a subset of
JavaScript syntax (ECMA-262 3rd edition) used as a lightweight data
interchange format.
...

```





# itertools——为高效循环而创建迭代器的函数

参见[`itertools` 创建常用迭代器](./iterator-and-generator.md#`itertools` 创建常用迭代器)。



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

将一个包含 JSON 的 `str`、`bytes` 或 `bytearray` 实例反序列化为 Python 对象。

```python
>>> import json
>>> json.loads('["foo", {"bar": ["baz", null, 1.0, 2]}]')
['foo', {'bar': ['baz', None, 1.0, 2]}]
>>> json.loads('{"c": 0, "b": 1, "a": {"d": 2, "e": 3}}')
{'c': 0, 'b': 1, 'a': {'d': 2, 'e': 3}}
```



## 编码器和解码器

### JSONEncoder

用于 Python 数据结构的可扩展 JSON 编码器，默认支持以下对象和类型：

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

返回 Python 数据结构的 JSON 字符串表达方式。

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





# keyword——检验Python关键字

`keyword` 模块用于确定某个某个字符串是否为 Python 关键字。



### iskeyword()

若传入的字符串是 Python 关键字则返回 `True`。



### kwlist

返回解释器定义的所有关键字组成的列表。



### issoftkeyword()

若传入的字符串是 Python 软关键字则返回 `True`。



### softkwlist

返回解释器定义的所有软关键字组成的列表。







# math——数学函数

`math` 模块提供了对 C 标准定义的数学函数的访问。

## atan2()

计算向量 `(x,y)` 与 x 轴正方向的夹角，结果在 `-pi` 和 `pi` 之间。

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

3.8 版本新功能。



## e

自然对数底数，精确到可用精度。

```python
>>> math.e
2.718281828459045
```



## exp()

（底数为 e 的）指数函数。

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

余数。整数计算时推荐使用 `x % y`，浮点数计算时推荐使用`fmod()`。

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

在 3.9 之后的版本可以传入任意个整数参数，之前的版本只能传入两个整数参数。



## hypot()

欧几里得范数，即点到原点的欧几里得距离。

```python
>>> math.hypot(1., 1, 1)
1.7320508075688772
```

在 3.8 之后的版本可以传入任意个实数参数，之前的版本只能传入两个实数参数。



## inf

浮点正无穷大，相当于 `float('inf')` 的返回值。浮点负无穷大用 `-math.inf` 表示。

```python
>>> math.inf
inf
>>> float('inf')
inf
```



## isclose()

若两个浮点数的值非常接近则返回 `True`，否则返回 `False`。

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

若参数值既不是无穷大又不是 `NaN`，则返回 `True`，否则返回 `False`。

```python
>>> math.isfinite(0.0)
True
>>> math.isfinite(math.inf)
False
```



## isnan()

若参数值是非数字（NaN）值，则返回 `True`，否则返回 `False`。

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

平方根向上取整可以使用 `1+ isqrt(n - 1)`。



## lcm()

最大公倍数。

3.9 版本新功能。



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

浮点非数字（NaN）值，相当于 `float('nan')` 的返回值。

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

3.8 版本新功能。



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

计算可迭代对象的所有元素（整数或浮点数）的积。积的默认初始值为 1。

```python
>>> math.prod(range(1, 6))
120
```

3.8 版本新功能。



## remainder()

IEEE 754 风格的余数：对于有限 *x* 和有限非零 *y*，返回 `x - n*y`，其中`n`是与商`x/y`的精确值最接近的整数；如果`x/y`恰好位于两个连续整数之间，则`n`取最近的偶整数。因此余数`r = remainder(x,y)`总是满足`abs(r)<=0.5*abs(y)`。

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

`multiprocessing` 是一个支持使用与 `threading` 模块类似的 API 来产生进程的包。`multiprocessing` 包同时提供了本地和远程并发操作，通过使用子进程而非线程有效地绕过了全局解释器锁。因此，`multiprocessing` 模块允许程序员充分利用给定机器上的多个处理器。它在 Unix 和 Windows 上均可运行。



## 生成进程







## 启动方法

根据不同的平台，`multiprocessing` 支持三种启动进程的方法，包括：

+ *spawn*：父进程会启动一个全新的 Python 解释器进程，子进程将只继承那些运行进程对象的 `run()` 方法所必需的资源。特别地，来自父进程的非必需文件描述符和句柄将不会被继承。 使用此方法启动进程相比使用 *fork* 或 *forkserver* 要慢上许多。

  可在 Unix 和 Windows 上使用，是 Windows 上的默认设置。

+ *fork*：父进程使用 `os.fork()` 来产生 Python 解释器分叉，子进程在开始时实际上与父进程相同，继承父进程的所有资源。请注意，安全分叉多线程进程是棘手的。

  只存在于 Unix，Unix 中的默认值。

+ *forkserver*：程序启动并选择 *forkserver* 启动方法时，将启动服务器进程。之后每当需要一个新进程时，父进程就会连接到服务器并请求它分叉一个新进程。分叉服务器进程是单线程的，因此使用 `os.fork()` 是安全的。没有不必要的资源被继承。

  可在 Unix 平台上使用，支持通过 Unix 管道传递文件描述符。

> 对于 macOS，*spawn* 启动方式是默认方式。 因为 *fork* 可能导致子进程崩溃，而被认为是不安全的，查看 [bpo-33725](https://bugs.python.org/issue33725)。

在 Unix 上通过 *spawn* 和 *forkserver* 方式启动多进程会同时启动一个 *资源追踪* 进程，负责追踪当前程序的进程产生的、并且不再被使用的命名系统资源（如命名信号量以及 `SharedMemory` 对象）。当所有进程退出后，资源追踪会负责释放这些仍被追踪的的对象。通常情况下是不会有这种对象的，但是假如一个子进程被某个信号杀死，就可能存在这一类资源的“泄露”情况。（泄露的信号量以及共享内存不会被释放，直到下一次系统重启，对于这两类资源来说，这是一个比较大的问题，因为操作系统允许的命名信号量的数量是有限的，而共享内存也会占据主内存的一片空间。）

你可以在主模块的 `if __name__ == '__main__'` 子句中调用 `set_start_method()` 以选择启动方法。

```python
import multiprocessing as mp

...

if __name__ == '__main__':
    mp.set_start_method('spawn')
    q = mp.Queue()
    p = mp.Process(target=foo, args=(q,))
    ...
```

如果你想要在同一程序中<u>使用多种启动方法</u>，可以使用 `get_context()` 来获取上下文对象，上下文对象与 `multiprocessing` 模块具有相同的 API。需要注意的是，对象在不同上下文创建的进程之间可能并不兼容，特别是使用 *fork* 上下文创建的锁不能传递给使用 *spawn* 或 *forkserver* 启动方法启动的进程。

```python
import multiprocessing as mp

...

if __name__ == '__main__':
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    p = ctx.Process(target=foo, args=(q,))
    ...
```



## 进程间交换对象

`multiprocessing` 支持进程之间的两种通信通道：

+ *队列*：先进先出的多生产者多消费者队列。队列是线程和进程安全的。

  ```python
  from multiprocessing import Process, Queue
  
  def f(q):
      q.put([42, None, 'hello'])
  
  if __name__ == '__main__':
      q = Queue()
      p = Process(target=f, args=(q,))
      p.start()
      print(q.get())              # [42, None, 'hello']
      p.join()
  ```

+ *管道*：`Pipe()` 函数返回由管道连接的两个连接对象，表示管道的两端，每个连接对象都有 `send()` 和 `recv()` 方法。注意，如果两个进程（或线程）同时尝试读取或写入管道的<u>同一端</u>，则管道中的数据可能会损坏。在不同进程中同时使用管道的<u>不同端</u>则不存在损坏的风险。

  ```python
  from multiprocessing import Process, Pipe
  
  def f(conn):
      conn.send([42, None, 'hello'])
      conn.close()
  
  if __name__ == '__main__':
      parent_conn, child_conn = Pipe()
      p = Process(target=f, args=(child_conn,))
      p.start()
      print(parent_conn.recv())   # [42, None, 'hello']
      p.join()
  ```

  

## 进程间同步

对于所有在 `threading` 中存在的同步原语，`multiprocessing` 中都有类似的等价物。例如可以使用锁来确保一次只有一个进程打印到标准输出：

```python
from multiprocessing import Process, Lock

def f(l, i):
    l.acquire()
    try:
        print('hello world', i)
        print('hello world', i)
    finally:
        l.release()

if __name__ == '__main__':
    lock = Lock()
    ps = []
    for num in range(5):
        p = Process(target=f, args=(lock, num))
        ps.append(p)
        p.start()
    for p in ps:
        p.join()
```

```
hello world 1
hello world 1
hello world 0
hello world 0
hello world 2
hello world 2
hello world 4
hello world 4
hello world 3
hello world 3
```



## 进程间共享状态

在进行并发编程时应尽量避免使用共享状态，使用多个进程时尤其如此。

但是，如果你真的需要使用一些共享数据，那么 `multiprocessing` 提供了两种方法：

+ *共享内存*：可以使用 `Value` 或 `Array` 将数据存储在共享内存映射中。

  ```python
  from multiprocessing import Process, Value, Array
  
  def f(n, a):
      n.value = 3.1415927
      for i, v in enumerate(a):
          a[i] = -v
  
  if __name__ == '__main__':
      num = Value('d', 0.0)         # 'd' 表示双精度浮点数
      arr = Array('i', range(10))   # 'i' 表示有符号整数
  
      p = Process(target=f, args=(num, arr))
      p.start()
      p.join()
  
      print(num.value)     # 3.1415927
      print(arr[:])        # [0, -1, -2, -3, -4, -5, -6, -7, -8, -9]
  ```

  这些共享对象是进程和线程安全的。

+ *服务进程*：由 `Manager()` 返回的管理器对象控制一个服务进程，该进程保存 Python 对象并允许其他进程使用代理操作它们。管理器支持下列类型：`list`、`dict`、`Namespace`、`Lock`、`RLock`、`Semaphore`、`BoundedSemaphore`、`Condition`、`Event`、`Barrier`、`Queue`、`Value` 和 `Array`。

  ```python
  from multiprocessing import Process, Manager
  
  def f(d, l):
      d[1] = '1'
      d['2'] = 2
      d[0.25] = None
      l.reverse()
  
  if __name__ == '__main__':
      with Manager() as manager:
          d = manager.dict()
          l = manager.list(range(10))
  
          p = Process(target=f, args=(d, l))
          p.start()
          p.join()
  
          print(d)         # {1: '1', '2': 2, 0.25: None}
          print(l)         # [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
  ```
  
  使用服务进程的管理器比使用共享内存对象更灵活，因为它们可以支持任意对象类型。此外单个管理器也可以通过网络由不同计算机上的进程共享。但是它们比使用共享内存慢。
  
  
  

## 使用工作进程

`Pool` 类表示一个工作进程池，它具有几种不同的将任务分配到工作进程的方法。





## 模块内容

### Array()



### Barrier

类似 `threading.Barrier` 的栅栏对象。



### connection.Connection

#### send()

将一个对象发送到连接的另一端，另一端使用 `recv()` 读取。

发送的对象必须是可以序列化的，过大的对象（接近 32MiB+，具体值取决于操作系统）有可能引发 `ValueError` 异常。



#### recv()

返回一个由另一端使用 `send()` 发送的对象。该方法会一直阻塞直到接收到对象。如果对端关闭了连接或者没有东西可接收，则抛出 `EOFError` 异常。



#### fileno()

返回由连接对象使用的描述符或者句柄。



#### close()

关闭连接对象。

当连接对象被垃圾回收时会自动调用。



#### poll()

返回连接对象中是否有可以读取的数据。

```python
poll(self, timeout=0.0)
```

如果未指定 *timeout*，此方法会马上返回；如果 *timeout* 是一个数字，则指定了最大阻塞的秒数；如果 *timeout* 是 `None`，那么将一直等待，不会超时。



#### send_bytes()

从一个字节类对象中取出字节数组并作为一条完整消息发送。

```python
send_bytes(self, buf, offset=0, size=None)
```





#### recv_bytes()

以字符串形式返回一条从连接对象另一端发送过来的字节数据。此方法在接收到数据前将一直阻塞。如果连接对象被对端关闭或者没有数据可读取，将抛出 `EOFError` 异常。

```python
recv_bytes(self, maxlength=None)
```





### cpu_count()

返回系统的 CPU 数量。



### current_process()

返回当前进程相对应的 `Process` 对象。



### get_context()

```python
multiprocessing.get_context(method=None)
```

返回一个 `Context` 对象，该对象具有和 `multiprocessing` 模块相同的API。



### get_start_method()

返回启动进程时使用的启动方法名。



### JoinableQueue



### Lock

原始锁（非递归锁）对象，类似于 `threading.Lock`。一旦一个进程或者线程拿到了锁，后续的任何其它进程或线程的其它请求都会被阻塞直到锁被释放。任何进程或线程都可以释放锁。除非另有说明，否则 `multiprocessing.Lock`  用于进程或者线程的概念和行为都和 `threading.Lock` 一致。



#### acquire()

```python
acquire(self, block=True, timeout=None)
```

获得锁。若 *block* 为 `True` 并且 *timeout* 为 `None`，则会阻塞当前进程直到锁被释放，然后将锁设置为锁定状态并返回 `True`；若 *block* 为 `True` 并且 *timeout* 为正数，则会在阻塞了最多 *timeout* 秒后锁还是没有被释放的情况下返回 `False`；若 *block* 为 `False`（此时 *timeout* 参数会被忽略），或者 *block* 为 `True` 并且 *timeout* 为 0 或负数，则会在锁被锁定的情况下返回 `False`，否则将锁设置成锁定状态并返回 `True`。

注意此函数的参数的一些行为与 `threading.Lock.acquire()` 的实现有所不同。



#### release()

释放锁。

可以在任何进程中使用，并不限于锁的拥有者。当尝试释放一个没有被持有的锁时，会抛出 `ValueError` 异常。除此之外其行为与 `threading.Lock.release()` 相同。



### Manager()



### parent_process()

返回当前进程的父进程相对应的 `Process` 对象。



### Process

```python
class multiprocessing.Process(group=None, target=None, name=None, args=(), kwargs={}, *, daemon=None)
# group     始终为`None`,仅用于兼容`threading.Thread`
# target    由`run()`方法调用的可调用对象
# name      进程名称
# args      目标调用的顺序参数
# kwargs    目标调用的关键字参数
# daemon    进程的daemon标志.若为`None`,则该标志从创建它的进程继承
```

进程对象表示在单独进程中运行的活动。`Process` 类拥有和 `threading.Thread` 等价的大部分方法。

```python
from multiprocessing import Process
import os

def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process id:', os.getppid())
    print('process id:', os.getpid())

def f(name):
    info('function f')
    print('hello', name)

if __name__ == '__main__':
    info('main line')
    p = Process(target=f, args=('bob',))
    p.start()
    p.join()
```

```
main line
module name: __main__
parent process id: 982
process id: 27363
function f
module name: __mp_main__
parent process id: 27363
process id: 27380
hello bob
```



#### run()

表示进程活动的方法。



#### start()

启动进程活动。

此方法每个进程对象最多只能调用一次。它会将对象的 `run()` 方法安排在一个单独的进程中调用。



#### join()

```python
join(self, timeout=None)
```

如果可选参数 *timeout* 是 `None`（默认值），则该方法将阻塞，直到调用 `join()` 方法的进程终止；如果 *timeout* 是一个正数，它最多会阻塞 *timeout* 秒。不管是进程终止还是方法超时，该方法都返回 `None`。

一个进程可以被 `join` 多次。

进程无法 `join` 自身，因为这会导致死锁。尝试在启动进程之前 `join` 进程会产生一个错误。



#### name

进程的名称。该名称是一个字符串，仅用于识别，没有具体语义。可以为多个进程指定相同的名称。



#### is_alive()

返回进程是否处于活动状态。从 `start()` 方法返回到子进程终止之间，进程对象都处于活动状态。



#### daemon

进程的守护标志，一个布尔值。必须在 `start()` 被调用之前设置。

初始值继承自创建进程。

当一个进程退出时，它会尝试终止子进程中的所有守护进程。



#### pid

返回进程 ID。



#### exitcode

子进程的退出代码。`None` 表示进程尚未终止；负值 *-N* 表示子进程被信号 *N* 终止。



#### authkey

进程的身份验证密钥（字节字符串）。



#### terminate()

终止进程。在 Unix 上由 `SIGTERM` 信号完成；在 Windows 上由 `TerminateProcess()` 完成。注意进程终止时不会执行退出处理程序和 finally 子句等。



#### kill()

与 `terminate()` 相同，但在 Unix 上使用 `SIGKILL` 信号。



#### close()

关闭 `Process` 对象，释放与之关联的所有资源。如果底层进程仍在运行，则会引发 `ValueError`。一旦 `close()` 成功返回，`Process` 对象的大多数其他方法和属性将引发 `ValueError`。



### Pipe

```python
multiprocessing.Pipe([duplex])
```

返回一对 `Connection` 对象 `(conn1, conn2)` ，分别表示管道的两端。

如果 *duplex* 被置为 `True`（默认值），那么该管道是双向的；如果 *duplex* 被置为 `False` ，那么该管道是单向的，即 `conn1` 只能用于接收消息，而 `conn2` 仅能用于发送消息。



### Pool



### Queue

返回一个使用一个管道和少量锁和信号量实现的共享队列实例。当一个进程将一个对象放进队列中时，一个写入线程会启动并将对象从缓冲区写入管道中。

`Queue` 实现了标准库类 `queue.Queue` 的所有方法，除了 `task_done()` 和 `join()`。一旦超时，将抛出标准库 `queue` 模块中常见的异常 `queue.Empty` 和 `queue.Full`。



#### qsize()

返回队列的大致长度。由于多线程或者多进程的环境，该数字是不可靠的。



#### empty(), full()

如果队列是空/满的，返回 `True`，反之返回 `False` 。由于多线程或多进程的环境，该状态是不可靠的。



#### put()

```python
put(self, obj, block=True, timeout=None)
```

将 *obj* 放入队列。若 *block* 为 `True` 并且 *timeout* 为 `None`，则会阻塞当前进程，直到有空的缓冲槽；若 *block* 为 `True` 并且 *timeout* 为正数，则会在阻塞了最多 *timeout* 秒后还是没有可用的缓冲槽的情况下抛出 `queue.Full` 异常；若 *block* 为 `False`，则会在有可用缓冲槽的情况下放入对象，否则抛出 `queue.Full` 异常（此时 *timeout* 参数会被忽略）。



### RLock

递归锁对象，类似于 `threading.RLock`。递归锁必须由持有线程、进程亲自释放。如果某个进程或者线程拿到了递归锁，这个进程或者线程可以再次拿到这个锁而不需要等待。但是这个进程或者线程的拿锁操作和释放锁操作的次数必须相同。

`RLock` 支持上下文管理器，因此可在 `with` 语句内使用。



#### acquire()

```python
acquire(self, block=True, timeout=None)
```

获得锁。若 *block* 为 `True` 并且 *timeout* 为 `None`，则会阻塞当前进程直到锁被释放，除非当前进程已经持有此锁，然后持有此锁，将锁的递归等级加一，并返回 `True`；若 *block* 为 `True` 并且 *timeout* 为正数，则会在阻塞了最多 *timeout* 秒后锁还是没有被释放的情况下返回 `False`；若 *block* 为 `False`（此时 *timeout* 参数会被忽略），或者 *block* 为 `True` 并且 *timeout* 为 0 或负数，则会在锁被锁定的情况下返回 `False`，否则持有此锁，将锁的递归等级加一，并返回 `True`。

注意此函数的参数的一些行为与 `threading.RLock.acquire()` 的实现有所不同。



#### release()

释放锁，即使锁的递归等级减一。如果释放后锁的递归等级降为 0，则会重置锁的状态为释放状态。

必须在拥有此锁的进程中使用，否则会抛出 `AssertionError` 异常。除了异常类型之外，其行为与 `threading.RLock.release()` 相同。



### Semaphore



### set_start_method()

```python
multiprocessing.set_start_method(method)
```

设置启动子进程的方法。*method* 可以是 `'fork'` , `'spawn'` 或者 `'forkserver'` 。



### SimpleQueue



### Value()







# operator——标准运算符替代函数







# os——多种操作系统接口

## chdir()

切换当前工作目录为指定路径。

```python
>>> os.chdir('dir1')
```



## environ

进程的环境变量，可以直接操作该映射以查询或修改环境变量。

```python
>>> os.environ
environ({'SHELL': '/bin/zsh', ...
>>> os.environ['HOME']           # 查询环境变量
'/Users/xyx'
>>> os.environ['MYENV'] = '1'    # 添加环境变量
>>> os.environ['MYENV']
'1'
>>> del os.environ['MYENV']      # 删除环境变量
>>> os.environ['MYENV']
# KeyError: 'MYENV'
```



## fork()

分叉出一个子进程，在子进程中返回 `0`，在父进程中返回子进程的进程号。



## getcwd()

返回当前工作目录的路径。

```python
>>> os.getcwd()
'/Users/xyx'
```



## getenv()

获取环境变量的值。

```python
os.getenv(key, default=None)
# key        环境变量
# default    若环境变量不存在,返回此默认值
```



## kill()

```python
os.kill(pid, sig)
```

将信号 *sig* 发送至进程 *pid*。



## listdir()

返回指定目录下各项目名称组成的列表，该列表按任意顺序排列，且不包括特殊项目 `.` 和 `..`。

```python
>>> os.listdir()
['dir1', 'dir2', 'file1', 'file2']
```



## makedirs()

递归地创建指定名称和权限的目录。与 `mkdir()` 类似，但会自动创建到达最后一级目录所需要的中间目录。

```python
>>> os.mkdir('dir1/dir2', mode=0o755)
```



## mkdir()

创建指定名称和权限的目录。若目录已存在，则引发 `FileExistsError` 异常。

```python
>>> os.mkdir('dir1', mode=0o755)
```

要递归地创建目录（一次创建多级目录），请使用 `makedirs()`。



## remove()

删除指定文件。若文件不存在，则引发 `FileNotFoundError` 异常；若路径指向目录，则引发 `IsADirectoryError` 异常。

```python
>>> os.remove('file1')
```



## rename()

```python
os.rename(src, dst, *, src_dir_fd=None, dst_dir_fd=None)
```

将文件或目录 *src* 重命名为 *dst*。若 *dst* 已存在，则下列情况下操作将会失败，并引发 `OSError` 的子类：

+ 在 Windows 上，引发 `FileExistsError` 异常
+ 在 Unix 上，若 *src* 是文件而 *dst* 是目录，将抛出 `IsADirectoryError` 异常，反之则抛出 `NotADirectoryError` 异常；若两者都是目录且 *dst* 为空，则 *dst* 将被静默替换；若 *dst* 是非空目录，则抛出 `OSError` 异常；若两者都是文件，则在用户具有权限的情况下，将对 *dst* 进行静默替换；若 *src* 和 *dst* 在不同的文件系统上，则本操作在某些 Unix 分支上可能会失败。



## rmdir()

删除指定目录。若目录不存在，则引发 `FileNotFoundError` 异常；若目录不为空，则引发 `OSError` 异常。若要删除整个目录树，请使用 `shutil.rmtree()`。

```python
>>> os.rmdir('dir1')
```



## system()

创建一个 Shell 子进程并执行指定命令（一个字符串），执行命令过程中产生的任何输出都将发送到解释器的标准输出流。

```python
>>> os.system('pwd')
/Users/xyx
```



## times()

返回当前的全局进程时间。



## wait()

等待子进程执行完毕，返回一个元组，包含其 pid 和退出状态指示——一个 16 位数字，其低字节是终止该进程的信号编号，高字节是退出状态码（信号编号为零的情况下）。



## waitid()



## waitpid()

 

## walk()

遍历目录。对于以 `top` 为根的目录树中的每个目录（包括 `top` 本身）都生成一个三元组 `(dirpath,dirnames,filenames)`。

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





# os.path——常用路径操作

## abspath()

返回路径的绝对路径。

```python
>>> path.abspath('.')
'/home/xyx'
```



## basename()

返回路径的基本名称，即将路径传入到 `split()` 函数所返回的元组的后一个元素。

```python
>>> path.basename('/Users/xyx')
'xyx'
>>> path.basename('/Users/xyx/Codes')
'Codes'
```



## dirname()

返回路径中的目录名称。

```python
>>> path.dirname('/Users/xyx')
'/Users'
>>> path.dirname('/Users/xyx/Codes')
'/Users/xyx'
```



## exists()

若路径指向一个已存在的文件或目录或已打开的文件描述符，返回 `True`。

```python
>>> path.exists('dir1')
```



## getsize()

返回路径指向的文件或目录的大小，以字节为单位。若文件或目录不存在或不可访问，则引发 `OSError` 异常。

```python
>>> path.getsize('file1')
14560
```



## isabs()

若路径是一个绝对路径，返回 `True`。在 Unix 上，绝对路径以 `/` 开始，而在 Windows 上，绝对路径可以是去掉驱动器号后以 `/` 或 `\` 开始。



## isdir()

若路径是现有的目录，返回 `True`。



## isfile()

若路径是现有的常规文件，返回 `True`。



## ismount()

若路径是挂载点，返回 `True`。



## join()

智能地拼接一个或多个路径部分。

```python
>>> path.join('/Users', 'xyx')
'/Users/xyx'
```



## split()

将路径拆分为两部分，以最后一个 `/` 为界。

```python
>>> path.split('/Users/xyx')
('/Users', 'xyx')
```



## splitdrive()

将路径拆分为两部分，其中前一部分是挂载点（对于 Windows 系统为驱动器盘符）或空字符串。



## splitext()

将路径拆分为两部分，其中后一部分是文件扩展名（以 `.` 开始并至多包含一个 `.`）或空字符串。

```python
>>> path.splitext('/path/to/foo.bar.exe')
('/path/to/foo.bar', '.exe')
```





# pickle——Python对象序列化

`pickle` 模块实现了对一个 Python 对象结构的二进制序列化和反序列化。pickle 是将 Python 对象及其所拥有的层次结构转化为一个字节流的过程，而 unpickle 是与之相反的操作，会将（来自一个二进制文件或者字节类对象的）字节流转化回一个对象层次结构。在官方文档中，前者称为**封存**，后者称为**解封**。

> 警告：pickle 模块**并不安全**。你只应该对你信任的数据进行 unpickle 操作。
> 
> 构建恶意的 pickle 数据以**在解封时执行任意代码**是可能的。绝对不要对不信任来源的数据和可能被篡改过的数据进行解封。
> 
> 请考虑使用 [hmac](#hmac) 来对数据进行签名，确保数据没有被篡改。
> 
> 在你处理不信任数据时，更安全的序列化格式如 json 可能更为适合。参见 与 json 模块的比较 。

## 模块接口

### dump()

封存对象，并将封存后的对象写入到文件对象。

### dumps()

封存对象，并将封存后的对象作为 bytes 对象返回。

### load()

从文件对象读取封存后的对象，重建其中的层次结构并返回。

### loads()

从 bytes 对象读取封存后的对象，重建其中的层次结构并返回。

## 可以封存的对象

下列类型可以被封存：

* `None`、`True`、`False`
* 整数、浮点数、复数
* 字符串、byte、bytearray
* 只包含可封存对象的集合，包括元组、列表、集合和字典
* 定义在模块最外层的函数（使用 `def` 定义，lambda 函数则不可以）
* 定义在模块最外层的内置函数
* 定义在模块最外层的类
* 某些类实例，如果这些类的 `__dict__` 属性值或 `__getstate__()` 函数的返回值可以被封存（详见封存类实例）。

尝试封存不能被封存的对象会抛出 `PicklingError` 异常，异常发生时，可能有部分字节已经被写入指定文件中。尝试封存递归层级很深的对象时，可能会超出最大递归层级限制，此时会抛出 `RecursionError` 异常，可以通过 `sys.setrecursionlimit()` 调整递归层级，不过请谨慎使用这个函数，因为可能会导致解释器崩溃。

需要注意的是，




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

返回系统平台/OS 的名称。

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

获取 Mac OS 版本信息并将其返回为元组 `(release,versioninfo,machine)`，其中 *versioninfo* 是一个元组 `(version,dev_stage,non_release_version)`。

```python
>>> platform.mac_ver()
('11.2.3', ('', '', ''), 'x86_64')    # macOS Big Sur Version 11.2.3
```





# pprint——数据美化输出

`pprint` 模块提供了“美化打印”任意 Python 数据结构的功能。



## isreadable()

确定对象的格式化表示是否“可读”，或是否可以通过 `eval()` 重新构建对象的值。对于递归对象总是返回 `False`。

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

将对象的格式化表示作为字符串返回，其余部分与 `pprint()` 相同。

```python
>>> pprint.pformat(stuff)
"[['spam', 'eggs', 'lumberjack', 'knights', 'ni'],\n 'spam',\n 'eggs',\n 'lumberjack',\n 'knights',\n 'ni']"
```



## PrettyPrinter

`pprint` 模块定义的实现美化打印的类。

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

`PrettyPrinter` 对象具有 `pprint` 模块的各方法。实际上 `pprint` 模块的各方法都是先创建 `PrettyPrinter` 对象再调用对象的方法。





# random——生成伪随机数

`random` 模块实现了各种分布的伪随机数生成器。

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

捕获生成器的当前内部状态的对象并返回，这个对象用于传递给 `setstate()` 以恢复状态。



### setstate()

将生成器的内部状态恢复到 `getstate()` 被调用时的状态。

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

从 `range(0,stop)` 或 `range(start,stop,step)` 返回一个随机选择的元素。



### randint()

```python
random.randint(a, b)
```

返回随机整数 *N* 满足 `a <= N <= b`。相当于 `randrange(a, b+1)`。



## 随机序列

### choice()

从非空序列中随机选择一个元素并返回。如果序列为空，则引发 `IndexError`。

```python
>>> a = list(range(10))
>>> random.choice(a)
9
>>> random.choice(a)
5
```



### choices()

从非空序列中（有放回地）随机选择多个元素并返回。如果序列为空，则引发 `IndexError`。如果指定了权重，则根据权重进行选择。

```python
>>> a = list(range(5))
>>> random.choices(a, k=3)
[2, 0, 2]
>>> random.choices(a, k=3)
[2, 1, 4]
>>> random.choices(a, [70.0, 22.0, 6.0, 1.5, 0.5], k=3)                    # 相对权重
[0, 0, 0]
>>> random.choices(a, cum_weights=[70.0, 92.0, 98.0, 99.5, 100.0], k=3)    # 累积权重
[1, 0, 0]
```



### shuffle()

随机打乱序列。

```python
random.shuffle(x[, random])
# x           序列
# random      一个不带参数的函数,返回[0.0,1.0)区间内的随机浮点数.默认为函数`random()`.
```

```python
>>> a = list(range(10))
>>> random.shuffle(a)              # 原位操作
>>> a
[8, 9, 1, 2, 5, 3, 7, 4, 0, 6]
```



### sample()

从非空序列中（无放回地）随机选择多个元素并返回。如果序列长度小于样本数量，则引发 `IndexError`。

```python
>>> random.sample(range(10), k=5)
[4, 7, 1, 9, 3]
```

要从一系列整数中选择样本，请使用 `range()` 对象作为参数，这种方法特别快速且节省空间：

```python
>>> random.sample(range(10000000), k=60)
[9787526, 3664860, 8467240, 2336625, 4728454, 2344545, 1590996, 4202798, 8934935, 2465603, 5203412, 1656973, 1237192, 5539790, 7921240, 9392115, 1689485, 5935633, 7284194, 5304900, 3430567, 9269809, 8002896, 7427162, 8746862, 4370335, 1044878, 9205646, 235580, 1564842, 6691148, 19173, 8280862, 5589080, 4092145, 5456023, 1056700, 3205573, 9521250, 3719574, 4003310, 2390659, 9109859, 7515682, 1530349, 1349656, 5369625, 8521829, 8208870, 1829687, 5057437, 9248729, 4883691, 2093976, 9184534, 5582627, 9064454, 3409161, 9180997, 9858578]
```



## 实值分布

### random()

返回服从 $[0.0,1.0)$ 区间内均匀分布的随机浮点数。

```python
>>> random.random()
0.13931343809011631
```



### uniform()

```python
random.uniform(a, b)
```

返回服从 [*a*, *b*] 区间（*a* <= *b*）或 [*b*, *a*] 区间（*b* <= *a*）内均匀分布的随机浮点数。

```python
>>> random.uniform(60, 80)
79.59813867742345
```



### gauss()

```python
random.gauss(mu, sigma)
```

返回服从平均值为 *mu*、标准差为 *sigma* 的正态分布的随机浮点数。

多线程注意事项：当两个线程同时调用此方法时，它们有可能将获得相同的返回值。这可以通过三种办法来避免。1）让每个线程使用不同的随机数生成器实例；2）在所有调用外面加锁；3）改用速度较慢但是线程安全的 `normalvariate()` 函数。



### normalviriate()

```python
random.normalvariate(mu, sigma)
```

返回服从平均值为 *mu*、标准差为 *sigma* 的正态分布的随机浮点数。



### expovariate()

```python
random.expovariate(lambd)
```

返回服从参数为 *lambd* 的指数分布的随机浮点数。



### gammavariate()

```python
random.gammavariate(alpha, beta)
```

返回服从参数为 *alpha* 和 *beta* 的 Gamma 分布的随机浮点数。



### betavariate()

```python
random.betavariate(alpha, beta)
```

返回服从参数为 *alpha* 和 *beta* 的 Beta 分布的随机浮点数。





# re——正则表达式操作

见正则表达式。





# shutil——高阶文件操作

`shutil` 模块提供了一系列对文件和文件集合的高阶操作，特别是一些支持文件复制和删除的函数。



## copy()

将一个文件复制到目标位置并返回目标位置。

```python
shutil.copy(src, dst, *, follow_symlinks=True)
# src            要复制的文件
# dst            目标路径
#                若`dst`不存在,则文件将复制到此路径;若`dst`是已存在的目录,则文件将使用原文件名复制到此目录中;
#                若`dst`是已存在的文件,则此文件将被覆盖
```



## copytree()

将一个目录树复制到目标位置并返回目标位置。

```python
shutil.copytree(src, dst, symlinks=False, ignore=None, copy_function=copy2, ignore_dangling_symlinks=False, dirs_exist_ok=False)
# src            要复制的目录
# dst            目标路径
```





## disk_usage()

返回给定路径所在磁盘的使用统计数据，形式为一个命名的元组，*total*、*used* 和 *free* 属性分别表示总计、已使用和未使用空间的字节数。

```python
>>> shutil.disk_usage('.')
usage(total=499963174912, used=107589382144, free=360688713728)
```



## move()

将一个文件或目录树移动到目标位置并返回目标位置。

```python
shutil.move(src, dst, copy_function=copy2)
# src            要移动的文件或目录
# dst            目标路径
#                若`dst`是已存在的目录,则`src`将被移至该目录下;...
```



## rmtree()

删除一个目录树。

```python
shutil.rmtree(path, ignore_errors=False, onerror=None)
# path           要删除的目录
# ignore_errors  若为`True`,则删除失败导致的错误将被忽略
```





# socket——底层网络接口





# subprocess——子进程管理

`subprocess` 模块允许我们生成新的进程，连接它们的输入、输出、错误管道，并且获取它们的返回码。

大多数情况下，推荐使用 `run()` 方法调用子进程，执行操作系统命令。在更高级的使用场景，你还可以使用 `Popen` 接口。其实 `run()` 方法在底层调用的就是 `Popen` 接口。

## run()

```python
subprocess.run(args, *, stdin=None, input=None, stdout=None, stderr=None, shell=False, timeout=None, check=False, encoding=None, errors=None)

# args       要执行的命令.必须是一个字符串或参数列表.
# stdin,stdout,stderr  子进程的标准输入、输出和错误,其值可以是subprocess.PIPE,
#                subprocess.DEVNULL,一个已经存在的文件描述符,已经打开的文件对象或者
#                None.subprocess.PIPE表示为子进程创建新的管道,subprocess.
#                DEVNULL表示使用os.devnull.默认使用的是None,表示什么都不做.
# timeout    命令超时时间.如果命令执行时间超时,子进程将被杀死,并引发TimeoutExpired异常.
# check      若为True,并且进程退出状态码不是0,则引发CalledProcessError 异常.
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

注意当 `args` 是一个字符串时，必须指定 `shell=True`：

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

`run()` 方法的返回类型，包含下列属性：

**`args`**

启动进程的参数，是字符串或字符串列表。

**`returncode`**

子进程的退出状态码，0 表示进程正常退出。

**`stdout`**

捕获到的子进程的标准输出，是一个字节序列，或者一个字符串（如果 `run()` 设置了参数 `encoding`，`errors` 或 `text=True`）。如果未有捕获，则为 `None`。

如果设置了参数 `stderr=subprocess.STDOUT`，标准错误会随同标准输出被捕获，并且 `stderr` 将为 `None`。

**`stderr`**

捕获到的子进程的标准错误，是一个字节序列，或者一个字符串（如果 `run()` 设置了参数 `encoding`，`errors` 或 `text=True`）。如果未有捕获，则为 `None`。

**`check_returncode`（）**

检查 `returncode`，非零则引发 `CalledProcessError`。



## Popen





# sys——系统相关的参数和函数

## argv

被传递给 Python 脚本的命令行参数列表，`argv[0]` 为脚本的名称（是否是完整的路径名取决于操作系统）。



## executable

返回当前 Python 解释器的可执行文件的绝对路径。

```python
>>> import sys
>>> sys.executable
'/Users/xyx/.pyenv/versions/3.8.7/bin/python'
```



## exit()

从 Python 中退出，实现方式是引发一个 `SystemExit` 异常。

可选参数可以是表示退出状态的整数（默认为整数 0），也可以是其他类型的对象。如果它是整数，则 shell 等将 0 视为“成功终止”，非零值视为“异常终止”。



## modules

返回当前已加载模块的名称到模块实例的字典。

```python
>>> import sys
>>> from pprint import pprint
>>> import numpy
>>> pprint(sys.modules)
{'__main__': <module '__main__' (built-in)>,
 ...
 'numpy': <module 'numpy' from '/Users/xyx/.pyenv/versions/3.8.7/lib/python3.8/site-packages/numpy/__init__.py'>,
 ...
 'numpy.version': <module 'numpy.version' from '/Users/xyx/.pyenv/versions/3.8.7/lib/python3.8/site-packages/numpy/version.py'>,
 ...
 'pprint': <module 'pprint' from '/Users/xyx/.pyenv/versions/3.8.7/lib/python3.8/pprint.py'>,
 ...
 'sys': <module 'sys' (built-in)>,
 ...
```



## path

指定模块搜索路径的字符串列表。解释器将依次搜索各路径，因此索引靠前的路径具有更高的优先级。

程序启动时将初始化本列表，列表的第一项 `path[0]` 为调用 Python 解释器的脚本所在的目录。如果脚本目录不可用（比如以交互方式调用了解释器，或脚本是从标准输入中读取的），则 `path[0]` 为空字符串，Python 将优先搜索当前目录中的模块。

程序可以根据需要任意修改本列表。

```shell
$ python                        # 以交互方式调用解释器
>>> import sys
>>> sys.path                    # sys.path[0]为空字符串
['', '/Users/xyx/.pyenv/versions/3.8.7/lib/python38.zip', '/Users/xyx/.pyenv/versions/3.8.7/lib/python3.8', '/Users/xyx/.pyenv/versions/3.8.7/lib/python3.8/lib-dynload', '/Users/xyx/.pyenv/versions/3.8.7/lib/python3.8/site-packages']
```

```python
# /Users/xyx/python/test/my_module.py
import sys
print(sys.path[0])
```

```shell
$ python my_module.py
/Users/xyx/python/test
```



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

* `stdin` 用于所有交互式输入
* `stdout` 用于 `print()` 和 expression 语句的输出，以及输出 `input()` 的提示符
* 解释器自身的提示符和错误消息发往 `stderr`



## version, version_info

`version` 是一个包含 Python 解释器版本号、编译版本号、所用编译器等信息的字符串，`version_info` 是一个包含版本号五部分的元组：*major*，*minor*，*micro*，*releaselevel* 和 *serial*。

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

Python 搜索标准目录列表，以找到调用者可以在其中创建文件的目录。这个列表是：

1.`TMPDIR`，`TEMP` 或 `TMP` 环境变量指向的目录。
2. 与平台相关的位置：
   * 在 Windows 上，依次为 `C:\TEMP`、`C:\TMP`、`\TEMP` 和 `\TMP`
   * 在所有其他平台上，依次为 `/tmp`、`/var/tmp` 和 `/usr/tmp`
3. 不得已时，使用当前工作目录。





# threading——基于线程的并行

在 CPython 中，由于存在全局解释器锁，同一时刻只有一个线程可以执行 Python 代码（虽然某些性能导向的库可能会去除此限制）。 如果你想让你的应用更好地利用多核心计算机的计算资源，推荐你使用 `multiprocessing` 或 `concurrent.futures.ProcessPoolExecutor`。 但是，如果你想要同时运行多个 I/O 密集型任务，则多线程仍然是一个合适的模型。



## active_count()

返回当前存活的 `Thread` 对象的数量。



## current_thread()

返回当前调用者的控制线程的 `Thread` 对象。



## main_thread()

返回主 `Thread` 对象。



## Thread





## Lock

原始锁处于"锁定"或者"非锁定"两种状态之一。它有两个基本方法，`acquire()` 和 `release()`。当状态为非锁定时，`acquire()` 将状态改为锁定并立即返回；当状态是锁定时，`acquire()` 将阻塞至其他线程调用 `release()` 将其改为非锁定状态，然后 `acquire()` 重置其为锁定状态并返回。`release()` 只在锁定状态下调用，将状态改为非锁定并立即返回。如果尝试释放一个非锁定的锁，则会引发 `RuntimeError` 异常。

原始锁在创建时为非锁定状态。当多个线程在 `acquire()` 阻塞，然后 `release()` 重置状态为未锁定时，只有一个线程能继续执行；至于哪个线程继续执行则没有定义，并且会根据实现而不同。





# time——时间的访问和转换

`time` 模块提供了各种与时间相关的函数。相关功能还可以参阅 `datetime` 和 `calendar` 模块。

下面是一些术语和惯例的解释：

* *纪元（epoch）*是时间开始的点，其值取决于平台。对于 Unix，纪元是 1970 年 1 月 1 日 00：00：00（UTC）。要找出给定平台上的 epoch，请查看 `time.gmtime(0)`。

* 术语*纪元秒数*是指自纪元时间点以来经过的总秒数，通常不包括[闰秒](https://en.wikipedia.org/wiki/Leap_second)。在所有符合 POSIX 标准的平台上，闰秒都不会记录在总秒数中。

* 此模块中的函数可能无法处理纪元之前或遥远未来的日期和时间。“遥远未来”的定义由对应的 C 语言库决定；对于 32 位系统，它通常是指 2038 年及以后。

* 函数 `strptime()` 在接收到 `%y` 格式代码时可以解析使用 2 位数表示的年份。当解析 2 位数年份时，函数会按照 POSIX 和 ISO C 标准进行年份转换：数值 69--99 被映射为 1969--1999；数值 0--68 被映射为 2000--2068。

* UTC 是协调世界时（Coordinated Universal Time）的缩写。它以前也被称为格林威治标准时间（GMT）。使用 UTC 而不是 CUT 作为缩写是英语与法语（Temps Universel Coordonné）之间妥协的结果，不是什么低级错误。

* DST 是夏令时（Daylight Saving Time）的缩写，在一年的某一段时间中将当地时间调整（通常）一小时。DST 的规则非常神奇（由当地法律确定），并且每年的起止时间都不同。C 语言库中有一个表格，记录了各地的夏令时规则（实际上，为了灵活性，C 语言库通常是从某个系统文件中读取这张表）。从这个角度而言，这张表是夏令时规则的唯一权威真理。

* 由于平台限制，各种实时函数的精度可能低于其值或参数所要求（或给定）的精度。例如，在大多数 Unix 系统上，时钟频率仅为每秒 50 或 100 次。

* 使用以下函数在时间表示之间进行转换：

  |从|到|使用|
  |：-----------------------------------------------------------|：-----------------------------------------------------------|：-----------------------------------------------------------|
  |自纪元以来的秒数|UTC 的[`struct_time`](https://docs.python.org/zh-cn/3/library/time.html#time.struct_time)|[`gmtime()`](https://docs.python.org/zh-cn/3/library/time.html#time.gmtime)|
  |自纪元以来的秒数|本地时间的[`struct_time`](https://docs.python.org/zh-cn/3/library/time.html#time.struct_time)|[`localtime()`](https://docs.python.org/zh-cn/3/library/time.html#time.localtime)|
  |UTC 的[`struct_time`](https://docs.python.org/zh-cn/3/library/time.html#time.struct_time)|自纪元以来的秒数|[`calendar.timegm()`](https://docs.python.org/zh-cn/3/library/calendar.html#calendar.timegm)|
  |本地时间的[`struct_time`](https://docs.python.org/zh-cn/3/library/time.html#time.struct_time)|自纪元以来的秒数|[`mktime()`](https://docs.python.org/zh-cn/3/library/time.html#time.mktime)|



## ctime()

将纪元秒数转换为以下形式的字符串：`Sun Jun 20 23:21:05 1993`（本地时间）。

```python
>>> time.ctime(0)
'Thu Jan  1 08:00:00 1970'
```



## gmtime()

将纪元秒数转换为 UTC 的 `struct_time` 对象。若未提供 `secs` 或为 `None`，则使用 `time()` 所返回的当前时间。

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

调用该方法的线程将被暂停执行 *secs* 秒。参数可以是浮点数，以表示更为精确的睡眠时长。由于任何捕获到的信号都会终止 `sleep()` 引发的该睡眠过程并开始执行信号的处理例程，因此实际的暂停时长可能小于请求的时长；此外，由于系统需要调度其他活动，实际暂停时长也可能比请求的时间长。

```python
>>> time.sleep(1)
>>>      # after 1 sec
```



## strftime()

```python
time.strftime(format[, t])
```

将 `struct_time` 对象转换为指定格式的字符串。如果未提供 *t*，则使用由 `localtime()` 返回的当前时间。*format* 必须是一个字符串，可以嵌入以下指令：

| 指令 | 意义                                                                                                                          |
| :--- | :---------------------------------------------------------------------------------------------------------------------------- |
| `%a` | 本地化的缩写星期中每日的名称。                                                                                                |
| `%A` | 本地化的星期中每日的完整名称。                                                                                                |
| `%b` | 本地化的月缩写名称。                                                                                                          |
| `%B` | 本地化的月完整名称。                                                                                                          |
| `%c` | 本地化的适当日期和时间表示。                                                                                                  |
| `%d` | 十进制数 [01,31] 表示的月中日。                                                                                               |
| `%H` | 十进制数 [00,23] 表示的小时（24小时制）。                                                                                     |
| `%I` | 十进制数 [01,12] 表示的小时（12小时制）。                                                                                     |
| `%j` | 十进制数 [001,366] 表示的年中日。                                                                                             |
| `%m` | 十进制数 [01,12] 表示的月。                                                                                                   |
| `%M` | 十进制数 [00,59] 表示的分钟。                                                                                                 |
| `%p` | 本地化的 AM 或 PM 。                                                                                                          |
| `%S` | 十进制数 [00,61] 表示的秒。                                                                                                   |
| `%U` | 十进制数 [00,53] 表示的一年中的周数（星期日作为一周的第一天）。 在第一个星期日之前的新年中的所有日子都被认为是在第 0 周。     |
| `%w` | 十进制数 [0(星期日),6] 表示的周中日。                                                                                         |
| `%W` | 十进制数 [00,53] 表示的一年中的周数（星期一作为一周的第一天）。 在第一个星期一之前的新年中的所有日子被认为是在第 0 周。       |
| `%x` | 本地化的适当日期表示。                                                                                                        |
| `%X` | 本地化的适当时间表示。                                                                                                        |
| `%y` | 十进制数 [00,99] 表示的没有世纪的年份。                                                                                       |
| `%Y` | 十进制数表示的带世纪的年份。                                                                                                  |
| `%z` | 时区偏移以格式 +HHMM 或 -HHMM 形式的 UTC/GMT 的正或负时差指示，其中H表示十进制小时数字，M表示小数分钟数字 [-23:59, +23:59] 。 |
| `%Z` | 时区名称（如果不存在时区，则不包含字符）。                                                                                    |
| `%%` | 字面的 `'%'` 字符。                                                                                                           |

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

结构化的时间类型。它是一个带有 [named tuple](https://docs.python.org/zh-cn/3/glossary.html#term-named-tuple) 接口的对象：可以通过索引和属性名访问值。存在以下值：

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

本地夏令时时区的偏移量，以 UTC 为参照的秒数，如果已定义。如果本地夏令时时区在 UTC 以东，则为负数。

```python
>>> time.altzone
-28800
>>> 
```



### timezone

本地（非夏令时）时区的偏移量，以 UTC 为参照的秒数。如果本地时区在 UTC 以东，则为负数。

```python
>>> time.timezone
-28800
```



### tzname

两个字符串的元组：第一个是本地非夏令时时区的名称，第二个是本地夏令时时区的名称。如果未定义夏令时时区，则不应使用第二个字符串。 

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





# typing——类型提示支持

> 注意：Python 运行时不强制执行函数和变量类型注解，但这些注解可用于类型检查器、IDE、静态检查器等第三方工具。



## 类型别名





## 泛型





## 模块内容

### 特殊类型原语

#### Any

不受限的特殊类型，与所有类型兼容。



#### NoReturn

标记函数没有返回值的特殊类型，例如：

```python
from typing import NoReturn

def stop() -> NoReturn:
    raise RuntimeError('no way')
```



#### Union

联合类型，`Union[X, Y]` 表示非 X 即 Y。联合类型具有以下特征：

+ 参数必须是其中某种类型

+ 联合类型的嵌套会被展开，例如：

  ```python
  Union[Union[int, str], float] == Union[int, str, float]
  ```

+ 仅有一种类型的联合类型就是该类型自身，例如：

  ```python
  Union[int] == int
  ```

+ 重复的类型会被忽略，例如：

  ```python
  Union[int, str, int] == Union[int, str]
  ```
  
+ 联合类型不能创建子类，也不能实例化
  
  
  

#### Optional

可选类型，`Optional[X]` 等价于 `Union[X, None]` 。





# urllib.request——用于打开 URL 的可扩展库

## urlretrieve()

将 URL 形式的网络对象复制为本地文件。返回值为元组 `(filename, headers)` ，其中 *filename* 是保存网络对象的本地文件名， *headers* 是由 `urlopen()` 返回的远程对象 `info()` 方法的调用结果。可能触发的异常与 `urlopen()` 相同。

```python
>>> import urllib.request
>>> url, filename = "https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg"
>>> urllib.request.urlretrieve(url, filename)
```





# weakref——弱引用

