[toc]

# 数字和算术运算

数字是不可变类型。

## 整数

```python
width = 20   # int
i, j = 0, 1  # i=0, j=1
a = b = 0    # a=0, b=0

```

```python

```



```python
# 进制转换
bin()	0b1100	# 二进制
oct()	0o1472	# 八进制
int()	1989
hex()	0xff00	# 十六进制
```



## 浮点数

```python
f = 1.2e9           # float, 1.2*10^9
PI = 3.14159265359	# 习惯用全大写字母表示常量
```



## 算术运算

```python

10 / 3       # 浮点数除法
10 // 3	     # 除法取整
10 % 3       # 除法取余
10 ** 3      # 10^3
```





# 布尔值，布尔运算和比较运算

## 布尔值

**布尔值**是两个常量对象 `False` 和 `True`，用来表示逻辑上的真假。 在数字类型的上下文中（例如被用作算术运算符的参数时），它们的行为分别类似于整数 0 和 1。 内置函数 `bool()` 可被用来将任意对象转换为布尔值（参见逻辑值检测部分）。

```python
>>> True + True   # 做算术运算时相当于整数0/1
2
>>> bool('abc')   # 将`str`实例转换为布尔值
True      
```

实际上，`bool` 类是 `int` 的子类，其他类不能继承自它，它只有 `False` 和 `True` 两个实例。





## 逻辑值检测

任何对象都可以进行逻辑值的检测，以作为 `if` 或 `while` 语句的条件或者布尔运算的操作数来使用。

一个对象在默认情况下被视为真值，除非该对象所属的类定义了 `__bool__()` 方法且返回 `False` 或是定义了 `__len__()` 方法且返回 0。下面基本完整地列出了会被视为假值的内置对象：

- 被定义为假值的常量：`None` 和 `False`。
- 任何数值类型的零：`0`, `0.0`, `0j`, `Decimal(0)`, `Fraction(0, 1)`
- 空的序列和多项集：`''`, `()`, `[]`, `{}`, `set()`, `range(0)`

产生布尔值结果的运算和内置函数总是返回 `0` 或 `False` 作为假值，`1` 或 `True` 作为真值，除非另行说明。 （重要例外：布尔运算 `or` 和 `and` 总是返回其中一个操作数。）



## 布尔运算

**布尔运算**包括 `and, or, not`：

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
# 对其它实例进行布尔运算
>>> 2 or 3
2               # 结果不是布尔值,而是参与运算的实例之一,请回顾布尔运算的结果的表达式
>>> 2 and 3
3
>>> not 2
False           # 结果是布尔值
```



## 比较运算

**比较运算**返回一个布尔值。Python 中有 8 种比较运算符：`>, >=, ==, <=, <, !=, is [not], [not] in`，它们的优先级相同（高于布尔运算，低于算术、移位和位运算）。比较运算可以任意串连，例如 `x < y <= z` 等价于 `x < y and y <= z`，前者的不同之处在于 *y* 只被求值一次（但在两种情况下当 `x < y` 结果为 `False` 时 *z* 都不会被求值）。



### 值比较

运算符 `<`, `>`, `==`, `>=`, `<=` 和 `!=` 将比较两个对象的值：

```python
>>> 3 > 2
True
>>> 3 >= 3
True
>>> 3 == 3
True
>>> 3 != 2
True
>>> 3 > 2 > 1     # 串连比较运算
True
>>> not 1 < 2     # 运算优先级: 比较运算符 > 布尔运算符
False
>>> (not 1) < 2
True
>>> a = 1
>>> 



```

`==` 和 `!=` 对于任意对象总有定义



两个对象不要求为相同类型。



比较比较序列对象时使用字典序



### 成员检测运算

运算符 `in` 和 `not in` 用于成员检测。 如果 *x* 是 *s* 的成员则 `x in s` 求值为 `True`，否则为 `False`。 `x not in s` 返回 `x in s` 取反后的值。 所有内置序列和集合类型以及字典都支持此运算，对于字典来说 `in` 检测其是否有给定的键。 对于 list, tuple, set, frozenset, dict 或 collections.deque 这样的容器类型，表达式 `x in y` 等价于 `any(x is e or x == e for e in y)`。

对于字符串和字节串类型来说，当且仅当 *x* 是 *y* 的子串时 `x in y` 为 `True`。 一个等价的检测是 `y.find(x) != -1`。 空字符串总是被视为任何其他字符串的子串，因此 `"" in "abc"` 将返回 `True`。

对于定义了 `__contains__()` 方法的用户自定义类来说，如果 `y.__contains__(x)` 返回真值则 `x in y` 返回 `True`，否则返回 `False`。



### 标识号比较

运算符 `is` 和 `is not` 用于检测对象的标识号：当且仅当 *x* 和 *y* 是同一对象时 `x is y` 为真。 一个对象的标识号可使用 `id()` 函数来确定。 `x is not y` 会产生相反的逻辑值。

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











比较操作符`in`和`not in`校验一个值是否在/不在一个序列里

```python
i = 1
list = [1,2,3,4]
if i in list
```



操作符`is`和`is not`比较两个对象是不是同一个对象：

```python
>>> a = 1
>>> b = 1
>>> a == b    # a,b的值相等
True
>>> a is b    # 且指向同一个对象
True
>>> id(a)
10914496
>>> id(b)
10914496
>>> a = 100000000
>>> b = 100000000
>>> a == b
True
>>> a is b    # 当a,b的值较大时也不再成立
False
```

```python
>>> a = []
>>> b = []
>>> a == b
True
>>> a is b
False
```





# 字符串

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

Python3使用Unicode编码字符串，因此Python的字符串支持多语言：

```python
>>> print('English中文にほんご')
English中文にほんご
```

对于单个字符的编码，Python提供了`ord()`和`chr()`函数用于转换字符和Unicode编码：

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

Unicode字符与转义的Unicode编码是等价的：

```python
>>> '\u4e2d文'
'中文'
```



Python字符串（`str`类型）在内存中以Unicode表示，如果要保存到磁盘上或者在网络上传输，就需要将其编码为字节序列（`bytes`类型）：

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

注意`len()`函数对于`str`计算的是字符数，而对于`bytes`计算的是字节数：

```python
>>> len('中文')
2
>>> len('中文'.encode('utf-8'))
6
```

操作字符串时，为避免乱码问题，应当始终使用utf-8编码对`str`和`bytes`进行转换。例如当Python源代码中包含中文时，务必需要指定以utf-8编码保存；当Python解释器读取源代码时，为了让它按utf-8编码读取，我们通常在文件开头写上这两行：

```
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
```



# 空值

```python
None
```





# 类型转换

```python
int()	     # 转换为整数类型
float()    # 浮点
str()      # 字符串
```





