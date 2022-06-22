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
