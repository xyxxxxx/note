> **关于“类”和“类型”**
>
> 即使你已经掌握了 Python 中 `class` 和 `type` 的用法，但你仍然很难讲清楚类（class）和类型（type）在语义上的差别，下面是一些想法：
>
> 1. Python 中所有的类，不管是内置的类，第三方模块包含的类，还是用户自定义的类，都是 `type` 的实例；与此同时，`type` 本身也是类，因此它也是自身的实例。
> 2. `type` 可以接受一个任意对象，返回该对象所属的类，例如 `type(1)` 返回 `<class 'int'>`。上面这个例子的常见表述包括：`1` 是 `int` 类的实例；`1` 属于 `int` 类；`1` 的类型（type）是 `int`；等等。最后一种表述又衍生出“`1` 是哪种类型？”、“`1` 是 `int` 类型”之类的表述，于是造成“`int` 类型“和”`int` 类“两种表述同时存在，例如[官方文档](https://docs.python.org/zh-cn/3/library/stdtypes.html)中"内置类型"、“数字类型”这样的表述普遍存在。
> 3. 在很多文档中，“类型”实际上代表的是“类”，在阅读时应当注意这一点。由于这两个词语义相近并且关系紧密，很多文档已经不加区分地使用它们，但基本上也不会造成阅读障碍（将“类”和“类型”都理解为“类”或“类型”）。即便如此，我们依然推荐根据 1. 给出的类（class）和类型（type）的关系来决定应当使用哪一个词。如果你仍然感到迷惑而不确定该使用哪个词，那么至少确保自己文档的上下文的用法一致。

# 数字类型，算术运算和位运算

内置的数字类型包括整数 `int`、浮点数 `float` 和复数 `complex`。数字由数字字面值或内置函数和运算符的结果创建。

Python 完全支持混合运算：当一个二元算术运算符的操作数有不同数值类型时，"较窄"类型的操作数会拓宽到另一个操作数的类型，其中整数比浮点数窄，浮点数比复数窄。不同类型的数字之间的比较结果，与这些数字的精确值（在数学上）的比较结果相同。

构造函数 `int()`、`float()` 和 `complex()` 可以用来构造特定类型的数字。

## 整数

不带修饰的整数字面值（包括十六进制、八进制和二进制数）会生成整数：

```python
>>> 20
20
>>> 0xff00       # 十六进制数,以前缀'0x'表示
65280
>>> 0o1472       # 八进制数,以前缀'0o'表示
826
>>> 0b1100       # 二进制数,以前缀'0b'表示
12
>>> type(20)
<class 'int'>      # `int`类型
```

整数具有无限的精度：

```python
>>> a = 123456789123456789123456789123456789123456789123456789
>>> a
123456789123456789123456789123456789123456789123456789
>>> a + a
246913578246913578246913578246913578246913578246913578
```

内置函数 `int()` 可以构造整数，参见 [`int()`](./standard-library#int())。

内置函数 `bin()`, `oct()`, `hex()` 用于将整数转换为前缀为 `'0b'`, `'0o'`, `'0x'` 的二、八、十六进制字符串，参见 [`bin()`](./standard-library#bin())。

内置函数 `format()` 可以将整数转换为没有前缀的二、八、十、十六进制字符串，参见 [`format()`](./standard-library#format())。

### 附加方法

#### bit_length()

返回以二进制表示一个整数所需要的位数，不包括符号位和前面的零。

```python
>>> bin(-37)
'-0b100101'
>>> (-37).bit_length()
6
```

#### to_bytes()

返回表示一个整数的字节数组。

```python
int.to_bytes(length, byteorder, *, signed=False)
# length      使用的字节数
# byteorder   字节顺序,'big'表示大端序,'little'表示小端序
# signed      是否使用补码.若为`False`且整数为负,则引发`OverflowError`
```

```python
>>> (1).to_bytes(2, byteorder='big')
b'\x00\x01'
>>> (1).to_bytes(2, byteorder='little')
b'\x01\x00'
>>> (-1).to_bytes(2, byteorder='little', signed=True)
b'\xff\xff'
```

#### from_bytes()

返回字节数组所表示的整数。

```python
classmethod int.from_bytes(bytes, byteorder, *, signed=False)
# bytes       字节串
# byteorder   字节顺序,'big'表示大端序,'little'表示小端序
# signed      是否使用补码.若为`False`且整数为负,则引发`OverflowError`
```

```python
>>> int.from_bytes(b'\x00\x01', byteorder='big')
1
>>> int.from_bytes(b'\x00\x01', byteorder='little')
256
>>> int.from_bytes(b'\xff\xff', byteorder='little', signed=False)
65535
>>> int.from_bytes(b'\xff\xff', byteorder='little', signed=True)
-1
>>> int.from_bytes([0, 1], byteorder='little')
256
```

## 浮点数

包含小数点、幂运算符 `e` 的数字字面值会生成浮点数：

```python
>>> 1.23
>>> 1.
1.0
>>> 1e3           # `e3`表示乘10的3次方
1000.0
>>> 1.2e5
120000.0
```

浮点数通常使用 C 语言的 `double` 来实现，程序运行所在机器上的浮点数的精度和内部表示法可通过 `sys.float_info` 查看。 

```python
>>> import sys
>>> sys.float_info
sys.float_info(max=1.7976931348623157e+308, max_exp=1024, max_10_exp=308, min=2.2250738585072014e-308, min_exp=-1021, min_10_exp=-307, dig=15, mant_dig=53, epsilon=2.220446049250313e-16, radix=2, rounds=1)
```

内置函数 `float()` 可以构造浮点数，参见 [`float()`](./standard-library#float())。

### 附加方法

#### as_integer_ratio()

返回一对整数，其比率正好等于原浮点数并且分母为正数。无穷大会引发 `OverflowError` 而 NaN 会引发 `ValueError`。

```python
>>> (1.2).as_integer_ratio()
(5404319552844595, 4503599627370496)
```

#### is_integer()

如果 `float` 实例可用有限位整数表示则返回 `True`，否则返回 `False`。

```python
>>> (-2.0).is_integer()
True
>>> (3.2).is_integer()
False
```

#### hex()

以十六进制字符串的形式返回一个浮点数表示。对于有限浮点数，这种表示法将总是包含前导的 `0x` 和尾随的 `p` 加指数。

```python
>>> (1.2).hex()
'0x1.3333333333333p+0'
>>> (-1.5).hex()
'-0x1.8000000000000p+0'
```

> 由于 Python 浮点数在内部存储为二进制数，因此浮点数与十进制数字符串之间的转换往往会导致微小的舍入错误，而十六进制数字符串却可以精确地表示和描述浮点数。这在进行调试和计算数值时非常有用。

#### fromhex()

返回十六进制字符串表示的浮点数。字符串可以带有前导和尾随的空格。

```python
>>> float.fromhex('-0x1.3p+0')
-1.1875
```

## 复数

在数字字面值末尾加上 `'j'` 或 `'J'` 会变为虚数，你可以将其与整数或浮点数相加来得到具有实部和虚部的复数，注意复数的实部和虚部都是浮点类型：

```python
>>> 1 + 2j
(1+2j)
>>> 1.2 + 3.4j
(1.2+3.4j)
>>> 1 + 0j
(1+0j)

>>> (1 + 2j).imag
2.0                   # 实部和虚部是浮点数,尽管该复数由整数创建
>>> (1 + 2j).real
1.0
```

内置函数 `complex()` 可以构造复数，参见 [`complex()`](./standard-library#complex())。

## 算术运算

所有数字类型都支持下列运算：

| 运算        | 结果              | 参见                                |
| ----------- | ----------------- | ----------------------------------- |
| `x + y`     | *x* 和 *y* 的和   |                                     |
| `x - y`     | *x* 和 *y* 的差   |                                     |
| `x * y`     | *x* 和 *y* 的乘积 |                                     |
| `x / y`     | *x* 和 *y* 的商   |                                     |
| `-x`        | *x* 取反          |                                     |
| `+x`        | *x* 不变          |                                     |
| `abs(x)`    | *x* 的绝对值或模  | [`abs()`](./standard-library#abs()) |
| `pow(x, y)` | *x* 的 *y* 次幂   | [`pow()`](./standard-library#pow()) |
| `x ** y`    | *x* 的 *y* 次幂   |                                     |

整数和浮点数还支持下列运算：

| 运算           | 结果              | 参见                                      |
| -------------- | ----------------- | ----------------------------------------- |
| `x // y`       | *x* 和 *y* 的商数 |                                           |
| `x % y`        | `x / y` 的余数    |                                           |
| `divmod(x, y)` | `(x // y, x % y)` | [`divmod()`](./standard-library#divmod()) |

注意 `x // y` 总是向负方向舍入，而 `x % y` 可能为正或为负：

```python
>>> divmod(10, 3)         # math.floor(10/3) = 3
(3, 1)
>>> divmod(10, -3)        # math.floor(10/-3) = -4
(-4, -2)
>>> divmod(-10, 3)
(-4, 2)
>>> divmod(-10, -3)
(3, -1)
```

复数还支持下列运算：

| 运算            | 结果       | 参见 |
| --------------- | ---------- | ---- |
| `x.conjugate()` | *x* 的共轭 |      |

## 位运算

只有整数支持位运算。位运算相当于对整数的具有无穷多个符号位的补码执行位操作。

位运算的优先级低于算术运算，高于比较运算；一元运算 `~` 具有与其它一元算术运算 (`+` 和 `-`) 相同的优先级。

位运算按优先级从低到高排列：

| 运算     | 结果                   | 注释                               |
| :------- | :--------------------- | :--------------------------------- |
| `x | y`  | *x* 和 *y* 按位 *或*   |                                    |
| `x ^ y`  | *x* 和 *y* 按位 *异或* |                                    |
| `x & y`  | *x* 和 *y* 按位 *与*   |                                    |
| `x << n` | *x* 左移 *n* 位        | 等价于乘以 `pow(2, n)`             |
| `x >> n` | *x* 右移 *n* 位        | 等价于除以 `pow(2, n)`，商向下取整 |
| `~x`     | *x* 逐位取反           |                                    |

```python
>>> 240 | 15        # 补码表示为 ..011110000 | ..01111
255                 # 因此按位或得到 ..011111111
>>> -240 | -15      # 补码表示为 ..100010000 | ..10001
-15                 # 因此按位或得到 ..10001
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

# 迭代器类型

参见[迭代器和生成器](./iterator-and-generator.md)。

# 其它内置类型

## 模块类型

参见[模块](./module-and-package.md#模块)。

```python
>>> import json
>>> json
<module 'json' from '/Users/xyx/.pyenv/versions/3.8.7/lib/python3.8/json/__init__.py'>
>>> type(json)
<class 'module'>
>>> dir(json)
['JSONDecodeError', 'JSONDecoder', 'JSONEncoder', '__all__', '__author__', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__path__', '__spec__', '__version__', '_default_decoder', '_default_encoder', 'codecs', 'decoder', 'detect_encoding', 'dump', 'dumps', 'encoder', 'load', 'loads', 'scanner']
```

## 函数类型

参见[函数](./function.md)。

```python
>>> type(abs)
<class 'builtin_function_or_method'>      # 内置函数

>>> def f():
...   pass
... 
>>> type(f)
<class 'function'>                        # 自定义函数
```

## 方法类型

参见[方法](./oop.md#方法)。

```python
>>> type(' '.join)
<class 'builtin_function_or_method'>      # 内置方法

>>> class Student:
...   def get_score(self):
...     pass
... 
>>> type(Student().get_score)
<class 'method'>                          # 自定义方法
```

## 类类型

类类型 `type` 是一个元类，所有类都是 `type` 的实例，包括 `type` 自身。

```python
>>> type(int)           # 内置类型
<class 'type'>
>>> dir(int)
['__abs__', '__add__', '__and__', '__bool__', '__ceil__', '__class__', '__delattr__', '__dir__', '__divmod__', '__doc__', '__eq__', '__float__', '__floor__', '__floordiv__', '__format__', '__ge__', '__getattribute__', '__getnewargs__', '__gt__', '__hash__', '__index__', '__init__', '__init_subclass__', '__int__', '__invert__', '__le__', '__lshift__', '__lt__', '__mod__', '__mul__', '__ne__', '__neg__', '__new__', '__or__', '__pos__', '__pow__', '__radd__', '__rand__', '__rdivmod__', '__reduce__', '__reduce_ex__', '__repr__', '__rfloordiv__', '__rlshift__', '__rmod__', '__rmul__', '__ror__', '__round__', '__rpow__', '__rrshift__', '__rshift__', '__rsub__', '__rtruediv__', '__rxor__', '__setattr__', '__sizeof__', '__str__', '__sub__', '__subclasshook__', '__truediv__', '__trunc__', '__xor__', 'as_integer_ratio', 'bit_length', 'conjugate', 'denominator', 'from_bytes', 'imag', 'numerator', 'real', 'to_bytes']

>>> type(Student)       # 自定义类
<class 'type'>

>>> type(type)          # `type`自身
<class 'type'>
```

## 空类型

空类型 `NoneType` 只有一个实例（单例模式）`None`（是内置常量），该对象表示缺少值，会由不显式返回值的函数返回，或用作默认值参数的默认值。

`None` 不支持任何特殊操作，逻辑值检测的结果为 `False`。

```python
>>> r = print('Hello!')         # 返回`None`
Hello!
>>> print(r)
None

>>> def f(data: dict = None):   # `None`作为默认值
...   if dict is None:          # 判断是否为`None`
...     dict = {}
... 

>>> type(None)
<class 'NoneType'>
>>> dir(None)
['__bool__', '__class__', '__delattr__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__']
```

## 未实现类型

未实现类型 `NotImplementedType` 只有一个实例（单例模式）`NotImplemented`（是内置常量），当比较和二元运算被应用于它们不支持的类型时，可以返回该对象。

当比较或二元运算方法返回 `NotImplemented` 时，解释器将尝试对另一种类型的反射操作（或其他一些回滚操作，取决于运算符）。如果所有尝试都返回 `NotImplemented`，则解释器将引发适当的异常。

`NotImplementedType` 不支持任何特殊操作，逻辑值检测的结果为 `True`（在未来的版本中进行逻辑值检测将引发 `TypeError`）。

```python
>>> class Person:
...   def __lt__(self, other):
...     return NotImplemented    # 未实现`<`
...   def __gt__(self, other):
...     return False             # 实现了`>`
... 
>>> p1 = Person()
>>> p2 = Person()
>>> p1 < p2             # 首先调用 `p1.__lt__(p2)` 返回 `NotImplemented`, 再调用 `p2.__gt__(p1)` 返回 `False`
                        # 因此最终返回 `False`
False

>>> type(NotImplemented)
<class 'NotImplementedType'>
>>> dir(NotImplemented)
['__class__', '__delattr__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__']
```

`NotImplemented` 与 `NotimplementedError` 的关系参见……

# 赋值语句

```python
>>> i, j = 0, 1     # i=0, j=1. 实质是元组的封包和解包
>>> a = b = 0       # a=0, b=0. 实质是赋值表达式`b = 0`本身返回`0`
```

