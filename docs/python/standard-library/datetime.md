# datetime——基本日期和时间类型

`datetime` 模块提供用于处理日期和时间的类。

`date`、`datetime`、`time` 和 `timezone` 类型具有以下通用特性:

* 这些类型的对象是不可变的。
* 这些类型的对象是可哈希的，这意味着它们可被作为字典的键。
* 这些类型的对象支持通过 `pickle` 模块进行高效的封存。

## timedelta

`timedelta` 对象表示两个 `date` 对象、`time` 对象或 `datetime` 对象之间的时间间隔，精确到微秒。

```python
class datetime.timedelta(days=0, seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0)
```

所有这些参数都是可选的并且默认为 `0`，可以是整数或者浮点数，也可以是正数或者负数。

只有 *days*、*seconds* 和 *microseconds* 会存储在内部，各参数单位的换算规则如下：

* 1 毫秒会转换成 1000 微秒。
* 1 分钟会转换成 60 秒。
* 1 小时会转换成 3600 秒。
* 1 星期会转换成 7 天。

并且 *days*、*seconds* 和 *microseconds* 会经标准化处理以保证表达方式的唯一性，即：

* `0 <= microseconds < 1000000`
* `0 <= seconds < 86400`
* `-999999999 <= days <= 999999999`

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

* `1 <= year <= 9999`
* `1 <= month <= 12`
* `1 <= day <= 给定年月对应的天数`

如果参数不在这些范围内，则抛出 `ValueError` 异常。

```python
>>> from datetime import date
>>> date(year=2020, month=11, day=27)
datetime.date(2020, 11, 27)
>>> date.today()
datetime.date(2020, 11, 27)
>>> date.today().weekday()
4                              # Friday
>>> date.today().isoweekday()
5                              # Friday
```

下面演示了 `date` 对象支持的运算：

```python
>>> d1 = date(2020, 11, 27)
>>> d2 = date(2019, 12, 4)
>>> t1 = timedelta(days=100)
>>> d1 + t1
datetime.date(2021, 3, 7)
>>> d2 - t1
datetime.date(2019, 8, 26)
>>> d1 - d2
datetime.timedelta(days=359)
>>> d1 > d2
True
```

### ctime()

（实例方法）返回一个表示日期的字符串。

```python
>>> d = date(2020, 11, 27)
>>> d.ctime()
'Fri Nov 27 00:00:00 2020'
```

### day

（实例属性）日。

### fromisoformat()

（类方法）返回一个对应于以 `YYYY-MM-DD` 格式给出的日期字符串的 `date` 对象。

```python
>>> date.fromisoformat('2020-11-27')
datetime.date(2020, 11, 27)
```

### fromordinal()

（类方法）返回对应于公历序号的日期，其中公元 1 年 1 月 1 日的序号为 1。

```python
>>> date.fromordinal(737756)
datetime.date(2020, 11, 27)
```

### fromtimestamp()

（类方法）返回对应于 POSIX 时间戳的日期。

```python
>>> date.fromtimestamp(1606468517.547344)
datetime.date(2020, 11, 27)
```

### isoformat()

（实例方法）返回一个以 `YYYY-MM-DD` 格式表示的日期字符串。

```python
>>> d = date(2020, 11, 27)
>>> d.isoformat()
'2020-11-27'
```

### isoweekday()

（实例方法）返回日期是星期几，星期一为 1，星期日为 7。

```python
>>> d = date(2020, 11, 27)
>>> d.isoweekday()
5
```

### month

（实例属性）月。

### replace()

（实例方法）替换日期中的部分值。

```python
>>> d = date(2020, 11, 27)
>>> d.replace(day=26)
datetime.date(2020, 11, 26)
```

### strftime()

（实例方法）返回一个由显式格式字符串所指明的代表日期的字符串。表示时、分或秒的格式代码值将为 0。详见 `datetime` 实例的 `strftime()` 方法。

```python
>>> d = date(2020, 11, 27)
>>> d.strftime("%d/%m/%y")
'27/11/20'
>>> d.strftime("%B %d %Y, %A")
'November 27 2020, Friday'
```

### today()

（类方法）返回当前的本地日期。

```python
>>> date.today()
datetime.date(2020, 11, 27)
```

### toordinal()

（实例方法）返回日期的公历序号，其中公元 1 年 1 月 1 日的序号为 1。

```python
>>> d1 = date(2020, 11, 27)
>>> d1.toordinal()
737756
>>> d2 = date(1, 2, 3)
>>> d2.toordinal()
34
```

### weekday()

（实例方法）返回日期是星期几，星期一为 0，星期日为 6。

```python
>>> d = date(2020, 11, 27)
>>> d.weekday()
4
```

### year

（实例属性）年。

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

## datetime

`datetime` 对象组合了 `date` 对象和 `time` 对象的所有信息。

```python
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

### now()


## tzinfo

一个描述时区信息对象的抽象基类。用来给 `datetime` 和 `time` 类提供自定义的时间调整概念（例如处理时区和/或夏令时）。

## timezone

一个实现了 `tzinfo` 抽象基类的子类，用于表示相对于 UTC 的偏移量。
