# re——正则表达式操作

`re` 模块提供了正则表达式匹配操作。

!!! info "`regex` 模块"
    第三方模块 `regex` 提供了与 `re` 模块兼容的 API 接口，同时还提供了额外的功能和更全面的 Unicode 支持。

## compile()

将正则表达式编译为一个正则表达式对象（正则对象），可以用于匹配。对于需要多次使用的正则表达式，使用`re.compile()`保存这个正则对象以便于复用，可以让程序更加高效。

```python
>>> prog = re.compile(r'^\d{3,4}\-\d{3,8}$')
>>> prog.match('0716-8834387')
<_sre.SRE_Match object at 0x1041b1ac0>
```

## findall()

对字符串从左到右扫描，找到所有不重复的正则表达式的匹配子串，返回其组成的列表。

```python
>>> re.findall(r'\d{3}', '123456789')
['123', '456', '789']
>>> re.findall(r'\d{3,5}', '123456789')
['12345', '6789']
```

## finditer()

对字符串从左到右扫描，找到所有不重复的正则表达式的匹配子串，返回一个保存了所有匹配对象的迭代器。

```python
>>> it = re.finditer(r'\d{3}', '123456789')
>>> list(it)
[<re.Match object; span=(0, 3), match='123'>, <re.Match object; span=(3, 6), match='456'>, <re.Match object; span=(6, 9), match='789'>]
```

## fullmatch()

如果整个字符串匹配正则表达式，就返回一个相应的匹配对象，否则返回`None`。 

```python
>>> re.fullmatch(r'\d{3,5}', '123456789')             # 整个字符串不匹配
>>> re.fullmatch(r'\d{3,10}', '123456789')            # 整个字符串匹配
<re.Match object; span=(0, 9), match='123456789'>
>>> 
>>> if re.fullmatch(r'\d{3,4}\-\d{3,8}', '0716-8834387'):   # 用作判断条件
    print('success')
else:
    print('failure')
...
success
```

## match()

如果字符串的一个前缀匹配正则表达式，就返回一个相应的匹配对象，否则返回`None`。 

```python
>>> re.match(r'\d{3}', '123456789')
<re.Match object; span=(0, 3), match='123'>          # 前缀匹配
>>> re.match(r'\d{3,5}', '123456789')
<re.Match object; span=(0, 5), match='12345'>        # 尽可能多地匹配
>>> re.match(r'\d{3,5}', 'a123456789')               # 前缀不匹配
>>>
>>> if re.match(r'^\d{3,4}\-\d{3,8}$', '0716-8834387'):   # 用作判断条件
    print('success')
else:
    print('failure')
... 
success
```

## Match

匹配对象。

### group()

返回一个或多个组的匹配子串。若参数为组号，则返回对应组的匹配子串；若参数为 `0`，则返回完整的匹配子串；若没有参数，则默认参数为 `0`；若有多个参数，则返回相应结果组成的元组。

```python
>>> m = re.match(r'^(\d{3,4})\-(\d{3,8})$', '0716-8834387')
>>> m.group(1)     # 第1组
'0716'
>>> m.group(2)     # 第2组
'8834387'
>>> m.group(0)     # 完整的匹配子串
'0716-8834387'
>>> m.group()      # 等价于`m.group(0)`
'0716-8834387'
>>> m.group(1, 2)       # 第1组和第2组
('0716', '8834387')
>>> m.group(0, 1, 2)    # 完整的匹配子串,第1组和第2组
('0716-8834387', '0716', '8834387')
```

### groups()

返回所有组的匹配子串组成的元组。

```python
>>> m = re.match(r"(\d+)\.(\d+)", "24.1632")
>>> m.groups()
('24', '1632')
>>> m = re.match(r"(\d+)\.?(\d+)?", "24")     # 第2组是可选的,因此不一定参与匹配
>>> m.groups()
('24', None)            # 未匹配,返回`None`
```

### re

返回参与匹配的正则对象。

### span()

返回组的匹配子串的开始和结束位置的索引组成的二元组，用法与 `start()`, `end()` 相同。

```python
>>> m = re.match(r'^(\d{3,4})\-(\d{3,8})$', '0716-8834387')
>>> m.span(2)
(5, 12)
```

### start(), end()

返回组的匹配子串的开始/结束位置的索引。若参数为 `0`，则返回完整的匹配子串的开始/结束位置的索引；若没有参数，则默认参数为 `0`。

```python
>>> m = re.match(r'^(\d{3,4})\-(\d{3,8})$', '0716-8834387')
>>> m.start(2)
5
>>> m.end(2)
12
```

### string

返回参与匹配的字符串。

## Pattern

编译后的正则表达式对象（正则对象）支持以下方法：`findall()`, `finditer()`, `fullmatch()`, `match()`, `search()`, `split()`, `sub()`。调用这些方法时不再需要传入 `pattern` 参数。

## search()

扫描整个字符串并找到匹配正则表达式的第一个位置，并返回一个相应的匹配对象；如果没有匹配则返回 `None`。 

```python
>>> re.search(r'\d{3}', '123456789')
<re.Match object; span=(0, 3), match='123'>         # 匹配的第一个位置
>>> re.search(r'\d{3,5}', '123456789')
<re.Match object; span=(0, 5), match='12345'>       # 尽可能多地匹配
>>> re.search(r'\d{3,5}', 'a123456789')
<re.Match object; span=(1, 6), match='12345'>       # 匹配的第一个位置
```

## split()

使用正则表达式将字符串划分为若干子串，返回其组成的列表。如果正则表达式中捕获到组，则组中的子串也会包含在列表里。

```python
re.split(pattern, string, maxsplit=0, flags=0)
# pattern      正则表达式
# string       要划分的字符串
# maxsplit     最多划分次数
# flags
```

```python
>>> re.split(r'\W+', 'Words, words, words.')        # 从左到右扫描,找到所有不重复的匹配子串并据此划分字符串
['Words', 'words', 'words', '']
>>> re.split(r'(\W+)', 'Words, words, words.')      # 匹配子串也包含在列表里
['Words', ', ', 'words', ', ', 'words', '.', '']
```

## sub()

对字符串从左到右扫描，找到所有不重复的正则表达式的匹配子串并进行替换；如果没有匹配的子串，则返回原字符串。

```python
re.sub(pattern, repl, string, count=0, flags=0)
# pattern      正则表达式
# repl         替换结果,可以是字符串或函数.若为函数,则会对所有不重复的匹配子串调用此函数;此函数接收一个匹配对象参数,
#              返回一个替换后的字符串
# string       要替换的字符串
# count        最多替换次数.若为`0`,则替换所有的匹配子串
# flags
```

```python
>>> re.sub(r'\d{3}', '0', '123456789')
'000'
>>> re.sub(r'\d{3,5}', '0', '123456789')
'00'
>>> 
>>> re.sub(r'node(s?)', r'Node\1', '111node222nodes333')
'111Node222Nodes333'
```
