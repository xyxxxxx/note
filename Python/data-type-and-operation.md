[toc]

# 数字类型，算术运算和位运算

内置的数字类型包括整数 `int`、浮点数 `float` 和复数 `complex`。 



Python 完全支持混合运算：当一个二元算术运算符的操作数有不同数值类型时，"较窄"类型的操作数会拓宽到另一个操作数的类型，其中整数比浮点数窄，浮点数比复数窄。不同类型的数字之间的比较，同比较这些数字的精确值一样。

构造函数 `int()`、`float()` 和 `complex()` 可以用来构造特定类型的数字。



## 整数

```python
>>> width = 20
>>> type(20)
<class 'int'>
```

整数具有无限的精度：

```python
>>> a = 123456789123456789123456789123456789123456789123456789
>>> a
123456789123456789123456789123456789123456789123456789
>>> a + a
246913578246913578246913578246913578246913578246913578
```



```python
# 进制转换
bin()	0b1100	# 二进制
oct()	0o1472	# 八进制
int()	1989
hex()	0xff00	# 十六进制
```



## 浮点数

浮点数通常使用 C 语言的 `double` 来实现，程序运行所在机器上的浮点数的精度和内部表示法可通过 `sys.float_info` 查看。 

```python
>>> f = 1.2e9           # float, 1.2*10^9
>>> 
```



## 复数

复数包含实部和虚部，分别以一个浮点数表示。



## 算术运算

```python

10 / 3       # 浮点数除法
10 // 3	     # 除法取整
10 % 3       # 除法取余
10 ** 3      # 10^3
```



## 位运算





## 赋值语句

```python
>>> i, j = 0, 1     # i=0, j=1. 实质是元组的封包和解包
>>> a = b = 0       # a=0, b=0. 实质是赋值表达式`b = 0`本身返回`0`
```











# 布尔类型，布尔运算和比较运算

## 布尔类型

**布尔值**是两个常量对象 `False` 和 `True`，用来表示逻辑上的真假。在数字类型的上下文中（例如被用作算术运算符的参数时），它们的行为分别类似于整数 0 和 1。内置函数 `bool()` 可被用来将任意对象转换为布尔值（参见逻辑值检测部分）。

```python
>>> True + True   # 做算术运算时相当于整数0/1
2
>>> bool('abc')   # 将`str`实例转换为布尔值
True      
```

实际上，`bool` 类是 `int` 的子类，其他类不能继承自它，它只有 `False` 和 `True` 两个实例，分别为 `int` 实例 `0` 和 `1` 的扩展。



## 逻辑值检测

任何对象都可以进行逻辑值的检测，以作为 `if` 或 `while` 语句的条件或者布尔运算的操作数来使用。

一个对象在默认情况下被视为真值，除非该对象所属的类定义了 `__bool__()` 方法且返回 `False` 或是定义了 `__len__()` 方法且返回 0。下面基本完整地列出了会被视为假值的内置对象：

* 被定义为假值的常量：`None` 和 `False`。
* 任何数值类型的零：`0`，`0.0`，`0j`，`Decimal(0)`，`Fraction(0,1)`
* 空的序列和多项集：`''`，`()`，`[]`，`{}`，`set()`，`range(0)`

产生布尔值结果的运算和内置函数总是返回 `0` 或 `False` 作为假值，`1` 或 `True` 作为真值，除非另行说明（注意布尔运算 `or` 和 `and` 总是返回其中一个操作数）。



## 布尔运算

**布尔运算**的返回结果见下表，按优先级从低到高排列：

| 运算      | 结果                                       | 注释                                                         |
| --------- | ------------------------------------------ | ------------------------------------------------------------ |
| `x or y`  | if *x* is false, then *y*, else *x*        | 短路运算符，只有在第一个参数为 `False` 时才会对第二个参数求值 |
| `x and y` | if *x* is false, then *x*, else *y*        | 短路运算符，只有在第一个参数为 `True` 时才会对第二个参数求值 |
| `not x`   | if *x* is false, then `True`, else `False` |                                                              |

```python
# 对布尔值进行布尔运算
>>> True and False
False
>>> True or False
True
>>> not True
False
>>> 
>>> A, B, C = True, True, True
>>> A and not B or C    # 运算优先级: not > and > or,因此等价于 (A and (not B)) or C
True
```

```python
# 对其它对象进行布尔运算
>>> 2 or 3
2               # 结果不是布尔值,而是参与运算的对象之一,请回顾布尔运算的结果的表达式
>>> 2 and 3
3
>>> not 2
False           # 结果是布尔值
```



## 比较运算

**比较运算**返回一个布尔值；比较运算符共有 8 种：`>,>=,==,<=,<,!=,is[not],[not]in`，它们的优先级相同（高于布尔运算，低于算术、移位和位运算）。

比较运算可以任意串连，例如 `x < y <= z`等价于`x < y and y <= z`，前者的不同之处在于 *y* 只被求值一次（但在两种情况下当 `x < y`结果为`False` 时 *z* 都不会被求值）。



### 值比较

运算符 `<`，`>`，`==`，`>=`，`<=` 和 `!=` 用于比较两个对象的值。

由于所有类型都是 `object` 的子类型，它们都从 `object` 继承了默认的比较行为：

* 默认的**一致性比较**（`==` 和 `!=`）基于对象的标识号，具有相同标识号的对象（即为同一对象）的一致性比较结果为相等，具有不同标识号的对象的一致性比较结果为不等。
* 默认的**次序比较**（`<`，`>`，`<=` 和 `>=`）没有定义，如果尝试比较将引发 `TypeError`。

类型可以通过实现比较方法来定义实例的比较行为，详见[基本定制](./oop.md#基本定制)。



下面给出了主要内置类型的比较行为：

* 数字（内置类型 `int`，`float`，`complex` 实例或标准库类型 `fractions.Fraction`，`decimal.Decimal` 实例）可进行类型内部和跨类型的比较，除了复数不支持次序比较。在类型相关的限制以内，它们会按照数学规则正确进行比较且不会有精度损失。

  非数字值 `float('NaN')` 和 `decimal.Decimal('NaN')` 属于特例：任何数字与非数字值的次序比较均返回假值，并且非数字值不等于其自身。例如，如果 `x = float('NaN')`，则`3< x`，`x <3`和`x == x`均为假值，而`x!= x` 则为真值。此行为遵循 IEEE 754 标准。

* `None` 和 `NotImplemented` 都是单例对象。PEP 8 建议单例对象的比较应当总是通过 `is` 或 `is not` 而不是等于运算符来进行。

* 字节序列（`bytes` 或 `bytearray` 实例）可进行类型内部和跨类型的比较，它们按字典序逐个比较相应元素的数字值。

* 字符串（`str` 实例）可进行类型内部的比较，它们按字典序逐个比较相应字符的 Unicode 码位数字值（内置函数 `ord()` 的返回值）。

  注意字符串和二进制码序列不能直接比较。

* 序列（`tuple`，`list` 或 `range` 实例）可进行类型内部的比较，除了 `range` 实例不支持次序比较。它们按字典序逐个比较相应元素。

  上述实例的跨类型一致性比较结果均为不相等，跨类型次序比较将引发 `TypeError`。因此 `[1,2]==(1,2)` 为假值，因为它们的类型不同。

* 映射（`dict` 实例）可进行类型内部的一致性比较，两个映射相等当且仅当它们具有相同的键值对。进行类型内部的次序比较将引发 `TypeError`。

* 集合（`set` 或 `frozenset` 的实例）可进行类型内部和跨类型的比较，它们将比较运算符定义为子集和超集检测。这类关系没有定义完全排序（例如 `{1,2}` 和 `{2,3}` 两个集合不相等，不为彼此的子集，也不为彼此的超集），因此集合不适合作为依赖于完全排序的函数的参数（例如将集合列表作为 `min()`，`max()` 或 `sorted()` 的参数将产生未定义的结果）。

* 其它内置类型的大多数没有实现比较方法，因此它们会继承默认的比较行为。



在条件允许的情况下，用户在定制自定义类的比较行为时应遵循以下一致性规则（尽管 Python 并不强制要求）：

* 相等应当是自反射的，即相同的对象相等：

  `x is y`=>`x == y`

* 比较应当是对称的，即下列表达式应当有相同的结果：

  `x == y`和`y == x`

  `x!= y`和`y!= x`

  `x < y`和`y > x`

  `x <= y`和`y >= x`

* 比较应当是可传递的：

  `x > y and y > z`=>`x > z`

  `x > y and y >= z`=>`x > z`

* 相反的比较应当导致相反的布尔值，即下列表达式应当有相同的结果：

  `x == y`和`not x!= y`

  `x < y`和`not x >= y`（对于完全排序）



### 成员检测运算

运算符 `in` 和 `not in` 用于成员检测。如果 *x* 是 *s* 的成员则 `x in s`求值为`True`，否则为`False`。`x not in s`返回与`x in s`相反的逻辑值。所有内置序列、集合类型和字典都支持此运算，对于字典来说`in` 检测其是否有给定的键。对于 list，tuple，set，frozenset，dict 或 collections.deque 这样的容器类型，表达式 `x in y`等价于`any(x is e or x == e for e in y)`。

对于字符串和字节串类型来说，当且仅当 *x* 是 *y* 的子串时 `x in y`为`True`；一个等价的检测是`y.find(x)!=-1`。空字符串总是被视为任何其他字符串的子串，因此`""in"abc"`将返回`True`。

对于定义了 `__contains__()` 方法的用户自定义类来说，如果 `y.__contains__(x)` 返回真值则 `x in y`返回`True`，否则返回`False`。

```python
>>> a = [1, 2, 3]                  # list
>>> 1 in a
True
>>> a = 1, 2, 3                    # tuple
>>> 1 in a
True
>>> a = set([1, 2, 3])             # set
>>> 1 in a
True
>>> a = {'a': 1, 'b': 2, 'c': 3}   # dict
>>> 'a' in a
True
>>> 'ab' in 'abc'                  # string
True
```



### 标识号比较

运算符 `is` 和 `is not` 用于检测对象的标识号：当且仅当 *x* 和 *y* 是同一对象时 `x is y`为真。一个对象的标识号可使用内置函数`id()`获取。`x is not y`返回与`x is y` 相反的逻辑值。

```python
>>> a = 1             # 由于性能方面的原因,Python会缓存简单的不可变对象(int和str),因此这里赋值的两个1是同一对象
>>> b = 1
>>> a == b
True
>>> a is b            # 同一int对象
True
>>> 
>>> a = '1'
>>> b = '1'
>>> a == b
True
>>> a is b            # 同一str对象
True
>>> 
>>> a = 257
>>> b = 257
>>> a == b
True
>>> a is b            # >256的整数不再是同一int对象
False
>>> 
>>> a = [1, 2, 3]
>>> b = a
>>> c = a[:]
>>> a == b
True
>>> a is b            # a, b指向同一列表
True
>>> a == c
True
>>> a is c            # a, c指向不同列表
False
>>> 
>>> a = 1
>>> a is not None     # 根据PEP8,与单例对象(例如`None`)的比较应使用`is`而非`==`
True
```





# 文本序列（字符串）类型

字符串是不可变类型。

```python
print("abc")    # 可以使用双引号或单引号
print('abc')

print('"Yes," they said.')    # 单/双引号中的双/单引号无需转义
print("\"Yes,\" they said.")  # 单/双引号中的单/双引号需要转义

print(r'C:\some\name')        # 单引号前的r表示原始字符串方式，即不转义

print('包含中文的str')   # python3的字符串以Unicode编码，支持多语言

# 跨行输入
print('''line1
         line2
         line3''')
```



**转义字符**

| \\'  | ‘      | \\\  | \    |
| ---- | ------ | ---- | ---- |
| \\"  | “      | %%   | %    |
| \n   | 换行   |      |      |
| \t   | 制表符 |      |      |



## 字符串方法

```python
>>> a = 3 * 'IN' + 'no' + 'Vation'
>>> a
'INININnoVation'
>>> len(a)                  # 字符串长度
14
>>> a[0]                    # 字符串可视作列表，进行索引和切片操作
'I'
>>> a[:6]
'INININ'
>>> a.replace('a', 'A')     # 替换字符
'INININnoVAtion'
>>> a.lower()               # 英文字符小写
'inininnovation'
>>> a.upper()               # 英文字符大写
'INININNOVATION'
>>> a.capitalize()          # 首个英文字符大写，后面小写
'Inininnovation'
>>> a.count('IN')           # 计数子串的出现次数
3
>>> a.startswith('INININ')  # 前缀比较
True
>>> a.endswith('tion')      # 后缀比较
True
>>> a.find('IN')            # 从前往后寻找子串的位置索引
0
>>> a.rfind('IN')           # 从后往前寻找子串的位置索引
4
>>> a.split('N')            # 拆分字符串
['I', 'I', 'I', 'noVation']
>>> 'N'.join(['I', 'I', 'I', 'noVation'])   # 拼接字符串
'INININnoVation'
```



## 格式化字符串



## Unicode编解码

Python3 使用 Unicode 编码字符串，因此 Python 的字符串支持多语言：

```python
>>> print('English中文にほんご')
English中文にほんご
```

对于单个字符的编码，Python 提供了 `ord()` 和 `chr()` 函数用于转换字符和 Unicode 编码：

```python
>>> hex(ord('A'))    # 字符 to 编码
'0x41'
>>> hex(ord('中'))
'0x4e2d'
>>> chr(0x03B1)      # 编码 to 字符
'α'
>>> chr(0x51EA)
'凪'
```

Unicode 字符与转义的 Unicode 编码是等价的：

```python
>>> '\u4e2d文'
'中文'
```



Python 字符串（`str` 类型）在内存中以 Unicode 表示，如果要保存到磁盘上或者在网络上传输，就需要将其编码为字节序列（`bytes` 类型）：

```python
>>> 'ABC'.encode('ascii')    　　# ascii编码
b'ABC'                       　　# 注意'ABC'为str类型而b'ABC'为bytes类型,后者的一个字符实际上表示一个字节
>>> 'ABC'.encode('utf-8')    　　# utf-8编码
b'ABC'
>>> '中文'.encode('utf-8')
b'\xe4\xb8\xad\xe6\x96\x87'  　　# 无法用ascii字符显示的字节用\x##显示
>>> '中文'.encode('gbk')         # gbk编码
b'\xd6\xd0\xce\xc4'
>>> '中文'.encode('shift_jis')   # shift_jis编码
b'\x92\x86\x95\xb6'

>>> '中文'.encode('ascii')       # 无法编码
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
UnicodeEncodeError: 'ascii' codec can't encode characters in position 0-1: ordinal not in range(128)
>>> '忧郁'.encode('shift_jis')
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
UnicodeEncodeError: 'shift_jis' codec can't encode character '\u5fe7' in position 0: illegal multibyte sequence
```

反过来，如果我们从磁盘或者网络上读取了字节流，就需要将其解码为字符串：

```python
>>> b'ABC'.decode('ascii')
'ABC'
>>> b'\xe4\xb8\xad\xe6\x96\x87'.decode('utf-8')
'中文'

>>> b'\xe4\xb8\xad\xff'.decode('utf-8')      # 无法解码
Traceback (most recent call last):
  ...
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 3: invalid start byte
>>> b'\xe4\xb8\xad\xff'.decode('utf-8', errors='ignore')   # 忽略错误字节
'中'
>>> 'この度'.encode('shift_jis').decode('gbk')              # 使用错误的解码方法
'偙偺搙'
```

注意 `len()` 函数对于 `str` 计算的是字符数，而对于 `bytes` 计算的是字节数：

```python
>>> len('中文')
2
>>> len('中文'.encode('utf-8'))
6
```

操作字符串时，为避免乱码问题，应当始终使用 UTF-8 编码对 `str` 和 `bytes` 进行转换。例如当 Python 源代码中包含中文时，务必需要指定以 UTF-8 编码保存；当 Python 解释器读取源代码时，为了让它按 UTF-8 编码读取，我们通常在文件开头写上这两行：

```
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
```





# 容器类型

参见[容器类型](./container-type.md)，包括序列类型、二进制序列类型、集合类型、映射类型以及 `collections` 模块定义的容器类型。





# 其它内置类型

## 空类型

空类型 `NoneType` 只有一个实例（单例模式）`None`（是内置常量），该对象由不显式返回值的函数返回，并且不支持任何特殊操作。

```python
>>> r = print('Hello!')
Hello!
>>> print(r)
None

>>> type(None)
<class 'NoneType'>
>>> dir(None)
['__bool__', '__class__', '__delattr__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__']
```



## 未实现类型

未实现类型 `NotImplementedType` 只有一个实例（单例模式）`NotImplemented`（是内置常量），当比较和二元运算被应用于它们不支持的类型时，可以返回该对象。

```python
>>> class Person:
...   def __lt__(self, other):
...     return NotImplemented    # 未实现`<`
...   def __gt__(self, other):
...     return False             # 实现了`>`
... 
>>> p1 = Person()
>>> p2 = Person()
>>> p1 < p2             # 调用 `p1.__lt__(p2)` 返回 `NotImplemented`, 于是再调用 `p2.__gt__(p1)` 返回 `False`
                        # 因此最终返回 `False`
False

>>> type(NotImplemented)
<class 'NotImplementedType'>
```

`NotImplemented` 与 `NotimplementedError` 的关系参见……





# 类型转换

```python
int()	     # 转换为整数类型
float()    # 浮点
str()      # 字符串
```





