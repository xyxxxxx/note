# 内置函数

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

## bytes

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

返回值为 *real*+*imag* j 的复数，或者将字符串或数字转换为复数。如果第一个形参是字符串，则它被解释为一个复数，并且必须没有第二个形参。第二个形参不能是字符串。每个实参都可以是任意的数值类型（包括复数）。如果省略了 *imag*，则默认值为零，构造函数会像 `int` 和 `float` 一样进行数值转换。如果两个实参都省略，则返回 `0j`。

对于一个普通 Python 对象 `x`，`complex(x)` 会委托给 `x.__complex__()`；如果 `__complex__()` 未定义则将回退至 `__float__()`；如果 `__float__()` 未定义则将回退至 `__index__()`。

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

详见[数字类型——int, float, complex]()。

## delattr()

```python
delattr(object, name)
```

与 `setattr()` 对应。删除对象 *object* 的名为 *name* 的属性，*name* 必须是字符串，指定一个现有属性。如果对象允许，该函数将删除指定的属性。例如 `delattr(x,'foobar')` 等同于 `del x.foobar`。

## dict

```python
class dict(**kwarg)
class dict(mapping, **kwarg)
class dict(iterable, **kwarg)
```

创建一个新的字典。参见数据结构-字典。

## dir()

如果没有实参，则返回当前本地作用域中的对象列表：

```python
>>> a = 1
>>> b = 'abc'
>>> def f():
...   pass
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
* 如果对象是类型或类对象，则列表包含它的属性名称，并且递归查找所有基类的属性。
* 其他情况下，列表包含对象的属性名称，它的类属性名称，并且递归查找它的类的所有基类的属性。

```python
>>> import platform
>>> dir(platform)
['_WIN32_CLIENT_RELEASES', '_WIN32_SERVER_RELEASES', '__builtins__', '__cached__', '__copyright__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__spec__', '__version__', '_comparable_version', '_component_re', '_default_architecture', '_follow_symlinks', '_ironpython26_sys_version_parser', '_ironpython_sys_version_parser', '_java_getprop', '_libc_search', '_mac_ver_xml', '_node', '_norm_version', '_platform', '_platform_cache', '_pypy_sys_version_parser', '_sys_version', '_sys_version_cache', '_sys_version_parser', '_syscmd_file', '_syscmd_uname', '_syscmd_ver', '_uname_cache', '_ver_output', '_ver_stages', 'architecture', 'collections', 'java_ver', 'libc_ver', 'mac_ver', 'machine', 'node', 'os', 'platform', 'processor', 'python_branch', 'python_build', 'python_compiler', 'python_implementation', 'python_revision', 'python_version', 'python_version_tuple', 're', 'release', 'sys', 'system', 'system_alias', 'uname', 'uname_result', 'version', 'win32_edition', 'win32_is_iot', 'win32_ver']
>>> 
>>> import multiprocessing
>>> dir(multiprocessing.Process)
['_Popen', '__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_bootstrap', '_check_closed', '_start_method', 'authkey', 'close', 'daemon', 'exitcode', 'ident', 'is_alive', 'join', 'kill', 'name', 'pid', 'run', 'sentinel', 'start', 'terminate']
```

!!! note "注意"
    因为 `dir()` 主要是为了便于在交互式时使用，所以它会试图返回人们感兴趣的名字集合，而不是试图保证结果的严格性或一致性，它具体的行为也可能在不同版本之间改变。例如，当实参是一个类时，metaclass 的属性不包含在结果列表中。

## divmod()

```python
divmod(a, b)
```

将两个数字（整数或浮点数）作为实参，执行整数除法并返回一对商和余数。对于混合操作数类型，应用双目算术运算符的类型强制转换规则。对于整数，结果和 `(a // b, a % b)`一致。对于浮点数，结果是 `(q, a % b)`，`q` 通常是 `math.floor(a/b)`，但可能会比 1 小。在任何情况下，`q * b + a % b` 和 `a` 基本相等；如果 `a % b` 非零，它的符号和 `b` 一样，并且有 `0 <= abs(a % b) < abs(b)`。

```python
>>> divmod(5, 3)
(1, 2)
>>> divmod(5.0, 1.5)
(3.0, 0.5)
```

## enumerate

```python
enumerate(iterable, start=0)
```

返回一个枚举对象。*iterable* 可以是一个序列、迭代器或其他支持迭代的对象。`enumerate()` 返回的迭代器的 `__next__()` 方法返回一个元组，里面包含一个计数值（从 *start* 开始，默认为 0）和通过迭代 *iterable* 获得的值。

```python
seasons = ['Spring', 'Summer', 'Fall', 'Winter']
list(enumerate(seasons))
[(0, 'Spring'), (1, 'Summer'), (2, 'Fall'), (3, 'Winter')]
list(enumerate(seasons, start=1))
[(1, 'Spring'), (2, 'Summer'), (3, 'Fall'), (4, 'Winter')]
```

等价于:

```python
def enumerate(sequence, start=0):
    n = start
    for elem in sequence:
        yield n, elem
        n += 1
```

## eval()

## exec()

```python
exec(object[, globals[, locals]])
```

动态执行 Python 代码。*object* 必须是字符串或代码对象：如果是字符串，那么该字符串将被解析为一系列 Python 语句并执行（除非发生语法错误）；如果是代码对象，它将被直接执行。在任何情况下，被执行的代码都应当是有效的文件输入。请注意即使在传递给 `exec()` 函数的代码的上下文中，`nonlocal`、`yield` 和 `return` 语句也不能在函数定义以外使用。该函数的返回值是 `None`。

无论哪种情况，如果省略了可选项，代码将在当前作用域内执行。如果只提供了 *globals*，则它必须是一个字典（不能是字典的子类），该字典将同时被用于全局和局部变量。如果同时提供了 *globals* 和 *locals*，它们会分别被用于全局和局部变量。如果提供了 *locals*，则它可以是任何映射对象。请记住在模块层级上，*globals* 和 *locals* 是同一个字典。如果 exec 得到两个单独对象作为 *globals* 和 *locals*，则代码将如同嵌入类定义的情况一样执行。

如果 *globals* 字典不包含 `__builtins__` 键值，则将为该键插入对内建 `builtins` 模块字典的引用。因此，在将执行的代码传递给 `exec()` 之前，可以通过将自己的 `__builtins__` 字典插入到 *globals* 中来控制可以使用哪些内置代码。

!!! note "注意"
    内置 `globals()` 和 `locals()` 函数各自返回当前的全局和本地字典，因此可以将它们传递给 `exec()` 的第二个和第三个实参。

## filter()

```python
filter(function, iterable)
```

用 *iterable* 中的那些传入函数 *function* 返回 True 的元素构建一个新的迭代器。*iterable* 可以是一个序列、迭代器或其他支持迭代的对象。如果 *function* 是 `None`，则会假设它是一个身份函数，即 *iterable* 中的所有（逻辑值检测）为 False 的元素会被移除。

换言之，`filter(function, iterable)` 相当于一个生成器表达式，当 *function* 不为 `None` 的时候为 `(item for item in iterable if function(item))`；*function* 为 `None` 的时候为 `(item for item in iterable if item)`。

请参阅 `itertools.filterfalse()` 了解只有 *function* 返回 False 时才选取 *iterable* 中元素的补充函数。

## float()

```python
class float([x])
```

返回从数字或字符串 *x* 生成的浮点数。

如果实参是字符串，则它必须是包含十进制数字的字符串，字符串前面可以有符号，之前也可以有空格。可选的符号有 `'+'` 和 `'-'`；`'+'` 对创建的值没有影响。实参也可以是 NaN（非数字）、正负无穷大的字符串。确切地说，除去首尾的空格后，输入必须遵循以下语法：

```
sign           ::=  "+" | "-"
infinity       ::=  "Infinity" | "inf"
nan            ::=  "nan"
numeric_value  ::=  floatnumber | infinity | nan
numeric_string ::=  [sign] numeric_value
```

这里 `floatnumber` 是 Python 浮点数的字符串形式，详见[浮点数字面值](https://docs.python.org/zh-cn/3.8/reference/lexical_analysis.html#floating)。字母大小写都可以，例如 `"inf"`、`"Inf"`、`"INFINITY"`、`"iNfINity"` 都可以表示正无穷大。

另一方面，如果实参是整数或浮点数，则返回具有相同值（在 Python 浮点精度范围内）的浮点数。如果实参在 Python 浮点精度范围外，则会触发 `OverflowError`。

对于一个普通 Python 对象 `x`，`float(x)` 会委托给 `x.__float__()`。如果 `__float__()` 未定义则将回退至 `__index__()`。

如果没有实参，则返回 `0.0`。

示例:

```python
>>> float('+1.23')
1.23
>>> float('   -12345\n')
-12345.0
>>> float('1e-003')
0.001
>>> float('+1E6')
1000000.0
>>> float('-Infinity')
-inf
```

详见[数字类型——int, float, complex]()。

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

## frozenset

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

返回对象的哈希值（如果它有的话）。哈希值是整数。它们在字典查找元素时用来快速比较字典的键。相同大小的数字变量有相同的哈希值（即使它们类型不同，如 1 和 1.0）。

!!! note "注意"
    如果对象实现了自己的 `__hash__()` 方法，请注意 `hash()` 会根据机器的字长来截断返回值。

## help()

启动内置的帮助系统（此函数主要在交互式中使用）。如果没有实参，解释器控制台里会启动交互式帮助系统。如果实参是一个字符串，则在模块、函数、类、方法、关键字或文档主题中搜索该字符串，并在控制台上打印帮助信息。如果实参是其他任意对象，则会生成该对象的帮助页。

该函数通过 `site` 模块加入到内置命名空间。

## hex()

```python
hex(x)
```

将整数转换为前缀为 `"0x"` 的小写十六进制字符串。如果 *x* 不是 Python `int` 对象，那么它需要定义 `__index__()` 方法返回一个整数。

```python
>>> hex(255)
'0xff'
>>> hex(-42)
'-0x2a'
>>> hex(-0x2a)
'-0x2a'
```

如果要将整数转换为大写或小写的十六进制字符串，并选择有无 `"0x"` 前缀，则可以使用如下方法：

```python
>>> '%#x' % 255, '%x' % 255, '%X' % 255
('0xff', 'ff', 'FF')
>>> format(255, '#x'), format(255, 'x'), format(255, 'X')
('0xff', 'ff', 'FF')
>>> f'{255:#x}', f'{255:x}', f'{255:X}'
('0xff', 'ff', 'FF')
```

## id()

返回对象的标识值。该值是一个整数，并且在此对象的生命周期内保证是唯一且恒定的。两个生命周期不重叠的对象可能具有相同的 `id()` 值。

## int

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
>>> int()             # 零值
0
```

如果有 *base* 参数，则 *x* 必须是表示进制为 *base* 的整数字面值的字符串、`bytes` 或 `bytearray` 实例，该字符串/字节串前可以有 `+` 或 `-` （中间不能有空格），前后可以有空格。允许的进制有 0、2-36，其中 2、8、16 进制允许字符串/字节串加上前缀 `'0b'`/`'0B'`, `'0o'`/`'0O'`, `'0x'`/`'0X'`（也可以不加）。进制为 0 表示按照字符串/字节串的前缀确定进制是 2、8、10 还是 16。

一个进制为 n 的整数可以使用 0 到 n-1 的数字，其中 10 到 35 用 `a` 到 `z` （或 `A` 到 `Z` ）表示。

```python
>>> int('ff00', base=16)       # 不使用前缀'0x'
65280
>>> int('0xff00', base=16)     # 使用前缀'0x'
65280
>>> int('0xff00', base=0)      # 通过前缀识别进制为16
65280
```

详见[数字类型——int, float, complex]()。

## isinstance()

```python
isinstance(object, classinfo)
```

如果 *object* 是 *classinfo* 的实例或（直接、间接或虚拟）子类则返回 `True`。*classinfo* 可以是类对象的元组，在此情况下 *object* 是其中任何一个类的实例就返回 `True`。

参见面向对象编程-获取对象信息。

## issubclass()

```python
issubclass(class, classinfo)
```

如果 *class* 是 *classinfo* 的（直接、间接或虚拟）子类则返回 `True`。类会被视作其自身的子类。*classinfo* 可以是类对象的元组，在此情况下 *classinfo* 中的每个条目都将被检查。

## iter()

```python
iter(object[, sentinel])
```

返回一个迭代器对象。如果没有第二个实参，*object* 必须支持迭代器协议（提供 `__iter__()` 方法）、序列协议（提供 `__getitem__()` 方法且键为从 0 开始的整数）或映射协议（提供 `__getitem__()` 方法和 `keys()` 方法（返回键的可迭代对象）），如果它不支持这些协议，会引发 `TypeError`。如果有第二个实参 *sentinel*，那么 *object* 必须是可调用对象，这种情况下返回的迭代器，每次调用 `__next__()` 方法时都会不带实参地调用 *object*；如果返回的结果是 *sentinel* 则引发 `StopIteration`，否则返回调用结果。

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
# 对象具有`__getitem__()`方法和`keys()`方法
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

## list

```python
class list([iterable])
```

详见[序列类型——list, tuple, range]()。

## locals()

更新并返回表示当前本地符号表的字典。在函数代码块但不是类代码块中调用 `locals()` 时将返回自由变量。在模块层级上，`locals()` 和 `globals()` 是同一个字典。

符号表是由 Python 解释器维护的数据结构，

```python
>>> a = 1
>>> b = 'abc'
>>> def f():
...   pass
... 
>>> locals()
{'__name__': '__main__', '__doc__': None, '__package__': None, '__loader__': <class '_frozen_importlib.BuiltinImporter'>, '__spec__': None, '__annotations__': {}, '__builtins__': <module 'builtins' (built-in)>, 'a': 1, 'b': 'abc', 'f': <function f at 0x10d327820>}
```

!!! note "注意"
     不要更改此字典的内容；更改不会影响解释器使用的局部变量或自由变量的值。

## map()

```python
map(function, iterable, ...)
```

返回一个迭代器，其将 *function* 应用于 *iterable* 的每一项，并产出其结果。如果传入了额外的 *iterable* 参数，*function* 必须接受相同个数的实参并被应用于从所有可迭代对象中并行获取的项。当有多个可迭代对象时，最短的可迭代对象耗尽则整个迭代就将结束。对于函数的输入已经是参数元组的情况，请参阅 `itertools.starmap()`。

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

```python
next(iterator[, default])
```

通过调用迭代器的 `__next__()` 方法获取下一个元素。如果迭代器耗尽，则返回给定的 *default*；如果没有默认值则引发 `StopIteration`。

## object

返回一个没有属性的新对象。`object` 是所有类的基类。

```python
>>> obj = object()
>>> dir(obj)
['__class__', '__delattr__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__']
>>> dir(object)
['__class__', '__delattr__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__']
```

!!! note "注意"
    由于 `object` 没有 `__dict__`，因此无法将任意属性赋给 `object` 的实例。

## oct()

```python
oct(x)
```

将整数转换为前缀为 `"0o"` 的八进制字符串。如果 *x* 不是 Python `int` 对象，那么它需要定义 `__index__()` 方法返回一个整数。

```python
>>> oct(8)
'0o10'
>>> oct(-56)
'-0o70'
>>> oct(-0o70)
'-0o70'
```

## open()

参见 [io](./io.md)。

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

返回 *base* 的 *exp* 次幂；如果 *mod* 存在，则返回 *base* 的 *exp* 次幂对 *mod* 取余（比 `pow(base,exp) % mod`更高效）。两参数形式 `pow(base, exp)` 等价于乘方运算符：`base**exp`。

参数必须具有数值类型。对于混合操作数类型，应用双目算术运算符的类型强制转换规则。对于 `int` 操作数，结果具有与操作数相同的类型（强制转换后），除非第二个参数为负值；在这种情况下，所有参数将被转换为浮点数并输出浮点数结果。 例如，`10**2` 返回 `100`，但 `10**-2` 返回 `0.01`。

```python
>>> pow(16, 2)
256
>>> pow(16, 2, mod=15)
1

>>> pow(0, 0)
1                     # 惯例
```

## print()

```python
print(*objects, sep=' ', end='\n', file=sys.stdout, flush=False)
```

将 *objects* 打印到 *file* 指定的文本流，以 *sep* 分隔并在末尾加上 *end*。*sep*、*end*、*file* 和 *flush* 如果存在，就必须以关键字参数的形式给出。

所有非关键字参数都会被转换为字符串，就像是执行了 `str()` 一样，并会被写入到流，以 *sep* 分隔并在末尾加上 *end*。*sep* 和 *end* 必须为字符串；它们也可以为 `None`，这意味着使用默认值。如果没有给出 *objects*，则 `print()` 将只写入 *end*。

*file* 参数必须是一个具有 `write(string)` 方法的对象；如果参数不存在或为 `None`，则将使用 `sys.stdout`。由于要打印的参数会被转换为文本字符串，因此 `print()` 不能用于二进制模式的文件对象。对于这些对象，应改用 `file.write(...)`。

输出是否被缓存通常决定于 *file*，但如果 *flush* 关键字参数为 True，流会被强制刷新。

## property

```python
class property(fget=None, fset=None, fdel=None, doc=None)
```

返回一个 property 属性。

*fget* 是用于获取属性值的函数，*fset* 是用于设置属性值的函数，*fdel* 是用于删除属性值的函数。

*doc* 如果给出，将成为该 property 属性的文档字符串。否则该 property 将拷贝 *fget* 的文档字符串（如果存在）。

一个典型的用法是定义一个托管属性：

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
...       # 托管属性
>>> bart = Student('Bart Simpson', 59)
>>> bart.score                 # 调用`get_score()`
59
>>> bart.score = 60            # 调用`set_score()`
>>> del bart.score             # 调用`del_score()`
```

更常见的写法是将 `property` 作为一个装饰器来创建只读的 property 属性：

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

property 属性对象具有 `getter`，`setter` 和 `deleter` 方法，它们可以用作装饰器来创建该 property 属性对象的副本，并将相应的方法设为所装饰的函数：

```python
>>> class Student(object):               # 与第一个例子完全等价
    def __init__(self, name, score):
        self.name = name
        self._score = score
    @property
    def score(self):                     # getter方法
        """I'm the 'score' property."""  # 作为property属性对象的docstring
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

## range

```python
class range(stop)
class range(start, stop[, step])
```

详见[序列类型——list, tuple, range]()。

## repr()

返回包含一个对象的可打印表示形式的字符串。对于许多类型，该函数尝试返回一个字符串，其在被传入 `eval()` 时所产生的对象与原对象具有相同的值；在其他情况下，表示形式会是一个括在尖括号中的字符串，其中包含对象类型的名称以及通常包括对象名称和地址的附加信息。类可以通过定义 `__repr__()` 方法来控制此函数为它的实例所返回的内容。

## reversed()

```python
reversed(seq)
```

返回一个反向的迭代器。*seq* 必须是一个具有 `__reversed__()` 方法的对象或者是支持该序列协议（具有从 0 开始的整数类型参数的 `__len__()` 方法和 `__getitem__()` 方法）。

## round()

```python
round(number[, ndigits])
```

返回 *number* 舍入到小数点后 *ndigits* 位精度的值。如果 *ndigits* 被省略或为 `None`，则返回最接近输入值的整数。

对于支持 `round()` 的内置类型，值会被舍入到最接近的 10 的负 *ndigits* 次幂的倍数；如果与两个倍数的距离相等，则向偶数舍入（因此，`round(0.5)` 和 `round(-0.5)` 均为 `0` 而 `round(1.5)` 为 `2`）。任何整数值都可作为有效的 *ndigits*（正数、零或负数）。如果 *ndigits* 被省略或为 `None` 则返回值将为整数。否则返回值与 *number* 的类型相同。

对于一般的 Python 对象 `number`，`round` 将委托给 `number.__round__`。

!!! note "注意"
    对浮点数执行 `round()` 的结果可能会令人惊讶：例如，`round(2.675, 2)` 将给出 `2.67` 而不是期望的 `2.68`。这不算是程序错误：这一结果是由于大多数十进制小数实际上都不能以浮点数精确地表示。请参阅[浮点算术：争议和限制](https://docs.python.org/zh-cn/3.8/tutorial/floatingpoint.html#tut-fp-issues)了解更多信息。

## set

## setattr()

```python
setattr(object, name, value)
```

与 `getattr()` 和 `delattr()` 对应。设置对象 *object* 的 *name* 属性的值，*name* 必须是字符串，指定一个现有属性或新增属性。如果对象允许，该函数将设置指定的属性。例如 `setattr(x, 'foobar', 123)` 等同于 `x.foobar = 123`。

## slice

## sorted()

```python
sorted(iterable, *, key=None, reverse=False)
```

根据 *iterable* 中的项返回一个新的已排序列表。

具有两个可选参数，它们都必须指定为关键字参数。*key* 指定带有单个参数的函数，用于从 *iterable* 的每个元素中提取用于比较的键（例如 `key=str.lower`）。默认值为 `None`（直接比较元素）。

*reverse* 为一个布尔值。如果设为 True，则每个列表元素将按反向顺序比较进行排序。

内置的 `sorted()` 确保是稳定的。如果一个排序确保不会改变比较结果相等的元素的相对顺序则称其为稳定的——这有利于进行多重排序（例如先按部门、再按薪级排序）。

## @staticmethod

将方法转换为静态方法。

参见面向对象编程-类-函数。

## str()

```python
class str(object='')
class str(object=b'', encoding='utf-8', errors='strict')
```

返回一个对象的 `str` 版本。`str` 是内置字符串类型。

详见[文本序列类型——str]()。

## sum()

```python
sum(iterable, /, start=0)
```

从 *start* 开始自左向右对 *iterable* 的项求和并返回总计值。*iterable* 的项通常为数字，而 *start* 值则不允许为字符串。

要拼接字符串序列，更好更快的方式是调用 `''.join(sequence)`；要以扩展精度对浮点值求和，请使用 `math.fsum()`；要拼接一系列可迭代对象，请使用 `itertools.chain()`。

## super()

## tuple()

```python
class tuple([iterable])
```

详见[序列类型——list, tuple, range]()。

## type

```python
class type(object)               # 表示传入一个对象;不是定义一个类
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

推荐使用 `isinstance()` 来检测对象的类型，因为它会考虑子类的情况。

传入三个参数时，返回一个新的 `type` 对象。这在本质上是 `class` 语句的一种动态形式，*name* 字符串即类名并会成为 `__name__` 属性；*bases* 元组包含基类并会成为 `__bases__` 属性；如果为空则会添加所有类的终极基类 `object`；*dict* 字典包含类主体的属性和方法定义，它在成为 `__dict__` 属性之前可能会被拷贝或包装。 

```python
>>> X = type('X', (), dict(a=1, f=abs))   # 类本身是一个`type`实例
>>> # 相当于
>>> class X:
...     a = 1
...     f = abs
```

## vars()

## zip()

创建一个元组的迭代器，它返回的第 *i* 个元组包含来自每个输入可迭代对象的第 *i* 个元素。当所输入可迭代对象中最短的一个被耗尽时，迭代器将停止迭代。当只有一个可迭代对象参数时，它将返回一个单元组的迭代器；不带参数时，它将返回一个空迭代器。相当于:

```python
def zip(*iterables):
    # zip('ABCD', 'xy') --> Ax By
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
```

## \__import__

此函数会被 `import` 语句调用以导入模块。建议使用 `importlib.import_module()` 而非此函数来导入模块。
