# csv——CSV 文件读写

CSV（Comma Separated Values）格式是电子表格和数据库中最常见的输入、输出文件格式。在 RFC 4180 规范推出的很多年前，CSV 格式就已经开始被使用了。由于当时并没有合理的标准，不同应用程序读写的数据会存在细微的差别，这种差别让处理多个来源的 CSV 文件变得困难。但尽管分隔符会变化，此类文件的大致格式是相似的，所以编写一个单独的模块以高效处理此类数据，将程序员从读写数据的繁琐细节中解放出来是有可能的。

`csv` 模块实现了 CSV 格式表单数据的读写。其提供了诸如“以兼容 Excel 的方式输出数据文件”或“读取 Excel 程序输出的数据文件”的功能，程序员无需知道 Excel 所采用 CSV 格式的细节。此模块同样可以用于定义其他应用程序可用的 CSV 格式或定义特定需求的 CSV 格式。

## 模块内容

`csv` 模块定义了以下函数：

### reader()

```python
csv.reader(csvfile, dialect='excel', **fmtparams)
```

返回一个 reader 对象，该对象将逐行遍历 *csvfile*。*csvfile* 可以是任何对象，只要这个对象支持迭代器协议并在每次调用 `__next__()` 方法时都返回一个字符串——文件对象和列表对象均适用。如果 *csvfile* 是文件对象，则应以 `newline=''` 打开它。可选参数 *dialect* 用于定义特定于某种 CSV 变种的参数组，它可以是 `Dialect` 类的子类的实例，也可以是 `list_dialects()` 函数返回的字符串之一。另一个可选关键字参数 *fmtparams* 可以覆写当前变种中的单个格式参数。关于变种和格式参数的完整详细信息，请参见[变种与格式参数](#变种与格式参数)部分。

```python
>>> with open('eggs.csv', newline='') as csvfile:
...     spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
...     for row in spamreader:
...         print(', '.join(row))
Spam, Spam, Spam, Spam, Spam, Baked Beans
Spam, Lovely Spam, Wonderful Spam
```

### writer()

```python
csv.writer(csvfile, dialect='excel', **fmtparams)
```

返回一个 writer 对象，该对象负责将用户的数据转换为在给定的类文件对象上带分隔符的字符串。*csvfile* 可以是具有 `write()` 方法的任何对象。如果 *csvfile* 是文件对象，则应以 `newline=''` 打开它。可选参数 *dialect* 用于定义特定于某种 CSV 变种的参数组，它可以是 `Dialect` 类的子类的实例，也可以是 `list_dialects()` 函数返回的字符串之一。另一个可选关键字参数 *fmtparams* 可以覆写当前变种中的单个格式参数。关于变种和格式参数的完整详细信息，请参见[变种与格式参数](#变种与格式参数)部分。

为了尽量简化与实现数据库 API 的模块之间的对接，None 值会写入为空字符串。虽然这个转换是不可逆的，但它让 SQL NULL 数据值转储到 CSV 文件更容易，而无需预处理从 `cursor.fetch*` 调用返回的数据。所有非字符串数据在写入前都先使用 `str()` 转化为字符串再写入。

```python
with open('eggs.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['Spam'] * 5 + ['Baked Beans'])
    spamwriter.writerow(['Spam', 'Lovely Spam', 'Wonderful Spam'])
```

### register_dialect()

```python
csv.register_dialect(name[, dialect[, **fmtparams]])
```

将 *name* 与 *dialect* 关联起来。*name* 必须是字符串。要指定变种，可以给出 `Dialect` 的子类，或给出 *fmtparams* 关键字参数，或两者都给出（此时关键字参数会覆盖变种的参数）。

### unregister_dialect()

```python
csv.unregister_dialect(name)
```

从变种注册表中删除 *name* 关联的变种。如果 *name* 不是已注册的变种名称，则引发 `Error` 异常。

### get_dialect()

```python
csv.get_dialect(name)
```

返回 *name* 关联的变种。如果 *name* 不是已注册的变种名称，则引发 `Error` 异常。此函数返回的是不可变的 `Dialect` 对象。

### list_dialects()

返回所有已注册变种的名称。

`csv` 模块定义了以下类：

### DictReader

```python
csv.DictReader(f, fieldnames=None, restkey=None, restval=None, dialect='excel', *args, **kwds)
```

### DictWriter

```python
csv.DictWriter(f, fieldnames, restval='', extrasaction='raise', dialect='excel', *args, **kwds)
```

### Dialect

### excel

### excel_tab

### unix_dialect




## 变种与格式参数

## Reader 对象

## Writer 对象

## 使用示例

下面是一个 CSV 示例文件：

``` title="grades.csv"
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

读取此文件：

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
