# datetime——基本日期和时间类型

`datetime` 模块提供用于处理日期和时间的类。

`date`、`time`、`datetime`、`timedelta` 和 `timezone` 类型具有以下通用特性:

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
datetime.timedelta(64, 29156, 10)   # 64天29156秒10毫秒
>>> delta.total_seconds()
5558756.00001                       # 秒
```

如有任何参数为浮点数导致 *microseconds* 的值为小数，那么从所有参数中余下的微秒数将被合并，它们的和会被四舍五入（奇入偶不入）到最接近的整数微秒值。如果没有任何参数为浮点数，则转换和标准化过程将是完全精确的（不会丢失信息）。

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
>>> t2 * 1.234
datetime.timedelta(seconds=12, microseconds=340000)
>>> t2 * 1.23456789                      # 乘以浮点数,结果舍入到微秒的整数倍
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
>>> t2 == t3
False
>>> str(t1)
'0:10:00'
>>> repr(t1)
'datetime.timedelta(seconds=600)'
```

除此之外，`timedelta` 对象还支持与 `date` 和 `datetime` 对象进行特定的相加和相减运算。

### days

天数。

### microseconds

微秒数。

### seconds

秒数。

### total_seconds()

返回时间间隔总共包含的秒数。

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
>>> delta = timedelta(days=100)
>>> d1 + delta
datetime.date(2021, 3, 7)
>>> d2 - delta
datetime.date(2019, 8, 26)
>>> d1 - d2
datetime.timedelta(days=359)
>>> d1 > d2
True
>>> d1 == d2
False
```

### ctime()

返回一个表示日期的字符串。

```python
>>> d = date(2020, 11, 27)
>>> d.ctime()
'Fri Nov 27 00:00:00 2020'
```

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

返回一个以 `YYYY-MM-DD` 格式表示的日期字符串。

```python
>>> d = date(2020, 11, 27)
>>> d.isoformat()
'2020-11-27'
```

### isoweekday()

返回日期是星期几，星期一为 1，星期日为 7。

```python
>>> d = date(2020, 11, 27)
>>> d.isoweekday()
5
```

### replace()

替换日期中的部分值。

```python
>>> d = date(2020, 11, 27)
>>> d.replace(day=26)
datetime.date(2020, 11, 26)
```

### strftime()

返回一个由显式格式字符串所指明的代表日期的字符串。表示时、分或秒的格式代码值将为 0。详见 `datetime` 实例的 `strftime()` 方法。

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

返回日期的公历序号，其中公元 1 年 1 月 1 日的序号为 1。

```python
>>> d1 = date(2020, 11, 27)
>>> d1.toordinal()
737756
>>> d2 = date(1, 2, 3)
>>> d2.toordinal()
34
```

### weekday()

返回日期是星期几，星期一为 0，星期日为 6。

```python
>>> d = date(2020, 11, 27)
>>> d.weekday()
4
```

### year, month, day

年，月，日。

## time

`date` 对象代表一个独立于任何特定日期的理想化时间，它假设每一天都恰好等于 86400 秒（这里没有“闰秒”的概念）。

```python
class datetime.time(hour=0, minute=0, second=0, microsecond=0, tzinfo=None, *, fold=0)
```

所有参数都是可选的；*tzinfo* 可以是 `None`，或者是一个 `tzinfo` 子类的实例；其余的参数必须是在下面范围内的整数：

* `0 <= hour < 24`
* `0 <= minute < 60`
* `0 <= second < 60`
* `0 <= microsecond < 1000000`
* `fold in [0,1]`

如果给出一个此范围以外的参数，则会引发 `ValueError`。所有参数值默认为 0，除了 *tzinfo* 默认为 `None`。

```python
>>> t = time(hour=12, minute=34, second=56, microsecond=789000)
>>> t
datetime.time(12, 34, 56, 789000)
```

下面演示了 `date` 对象支持的运算：

```python
>>> t1 = time(14, 11, 27)
>>> t2 = time(19, 12, 4)
>>> t1 < t2
True
>>> t1 == t2
False
```

### dst()

如果 `tzinfo` 为 `None`，则返回 `None`，否则返回 `self.tzinfo.dst(None)`，并且在后者不返回 `None` 或者值小于一天的 `timedelta` 对象时引发异常。

### fromisoformat()

（类方法）返回一个对应于以如下格式给出的日期字符串的 `time` 对象。

```
HH[:MM[:SS[.fff[fff]]]][+HH:MM[:SS[.ffffff]]]
```

```python
>>> time.fromisoformat('04:23:01')
datetime.time(4, 23, 1)
>>> time.fromisoformat('04:23:01.000384')
datetime.time(4, 23, 1, 384)
>>> time.fromisoformat('04:23:01+04:00')
datetime.time(4, 23, 1, tzinfo=datetime.timezone(datetime.timedelta(seconds=14400)))
```

### hour, minute, second, microsecond

时，分，秒，微秒。

```python
>>> t = time(hour=12, minute=34, second=56, microsecond=789000)
>>> t.hour
12
>>> t.minute
34
>>> t.second
56
>>> t.microsecond
789000
```

### isoformat()

```python
time.isoformat(timespec='auto')
```

返回表示为下列 ISO 8601 格式之一的时间字符串：

* `HH:MM:SS.ffffff`，如果 `microsecond` 不为 0
* `HH:MM:SS`，如果 `microsecond` 为 0
* `HH:MM:SS.ffffff+HH:MM[:SS[.ffffff]]`，如果 `utcoffset()` 不返回 `None`
* `HH:MM:SS+HH:MM[:SS[.ffffff]]`，如果 `microsecond` 为 0 并且 `utcoffset()` 不返回 None

可选参数 *timespec* 指定了包含的时间成分 (默认为 `'auto'`)，它可以是以下值之一：

* `'auto'`: 如果 `microsecond` 为 0 则与 `'seconds'` 相同，否则与 `'microseconds'` 相同。
* `'hours'`: 以两个数码的 `HH` 格式包含 `hour`。
* `'minutes'`: 以 `HH:MM` 格式包含 `hour` 和 `minute`。
* `'seconds'`: 以 `HH:MM:SS` 格式包含 `hour`, `minute` 和 `second`。
* `'milliseconds'`: 包含完整时间，但将秒值的小数部分截断至微秒，格式为 `HH:MM:SS.sss`.。
* `'microseconds'`: 以 `HH:MM:SS.ffffff` 格式包含完整时间。

```python
>>> t = time(hour=12, minute=34, second=56, microsecond=789000)
>>> t.isoformat(timespec='minutes')
'12:34'
>>> t.isoformat(timespec='microseconds')
'12:34:56.789000'
>>> t.isoformat()
'12:34:56.789000'
```

### replace

替换时间中的部分值。

```python
>>> t = time(20, 11, 27)
>>> t.replace(hour=21)
datetime.time(21, 11, 27)
```

### strftime()

返回一个由显式格式字符串所指明的代表时间的字符串。详见 `datetime` 实例的 `strftime()` 方法。

```python
>>> t = time(hour=12, minute=34, second=56, microsecond=789000)
>>> t.strftime("%H:%M:%S.%f")
'12:34:56.789000'
>>> t.strftime("%I:%M %p")
'12:34 PM'
```

### tzinfo

作为 *tzinfo* 参数被传给 `time` 构造器的对象，如果没有传入值则为 `None`。

### tzname()

如果 `tzinfo` 为 `None`，则返回 `None`，否则返回 `self.tzinfo.tzname(None)`，并且在后者不返回 `None` 或者值小于一天的 `timedelta` 对象时引发异常。

### utcoffset()

如果 `tzinfo` 为 `None`，则返回 `None`，否则返回 `self.tzinfo.utcoffset(None)`，并且在后者不返回 `None` 或者值小于一天的 `timedelta` 对象时引发异常。

## datetime

`datetime` 对象组合了 `date` 对象和 `time` 对象的所有信息。

与 `date` 对象一样，`datetime` 假设当今的公历在过去和未来永远有效；与 `time` 对象一样，`datetime` 假设每一天都恰好等于 86400 秒。

```python
class datetime.datetime(year, month, day, hour=0, minute=0, second=0, microsecond=0, tzinfo=None, *, fold=0)
```

*year*、*month* 和 *day* 参数是必须的，*tzinfo* 可以是 `None` 或一个 `tzinfo` 子类的实例，其余的参数必须是在下面范围内的整数：

* `MINYEAR <= year <= MAXYEAR`
* `1 <= month <= 12`
* `1 <= day <= 指定年月的天数`
* `0 <= hour < 24`
* `0 <= minute < 60`
* `0 <= second < 60`
* `0 <= microsecond < 1000000`
* `fold in [0, 1]`

如果参数不在这些范围内，则抛出 `ValueError` 异常。

下面演示了 `datetime` 对象支持的运算：

```python
>>> dt1 = datetime(2020, 11, 27, 15, 17, 8, 132263)
>>> dt2 = datetime(2019, 12, 4, 22, 6, 11, 159813)
>>> delta = timedelta(days=100)
>>> dt1 + delta
datetime.datetime(2021, 3, 7, 15, 17, 8, 132263)
>>> dt2 - delta
datetime.datetime(2019, 8, 26, 22, 6, 11, 159813)
>>> dt1 - dt2
datetime.timedelta(days=358, seconds=61856, microseconds=972450)
>>> dt1 > dt2
True
>>> dt1 == dt2
False
```

### astimezone()

### combine()

```python
combine(date, time, tzinfo=self.tzinfo)
```

（类方法）返回一个新的 `datetime` 对象，其日期部分等于给定的 `date` 对象的值，时间部分等于给定的 `time` 对象的值。如果提供了 *tzinfo* 参数，其值会被用于设置结果的 `tzinfo` 属性，否则将使用 *time* 参数的 `tzinfo` 属性。

如果 *date* 参数是一个 `datetime` 对象，它的时间部分和 `tzinfo` 属性会被忽略。

对于任意 `datetime` 对象 `d`，`d == datetime.combine(d.date(), d.time(), d.tzinfo)`。

### ctime()

返回一个表示日期时间的字符串:

```python
>>> datetime(2002, 12, 4, 20, 30, 40).ctime()
'Wed Dec  4 20:30:40 2002'
```

### date()

返回具有同样 year、month 和 day 值的 `date` 对象。

### dst()

如果 `tzinfo` 为 `None`，则返回 `None`，否则返回 `self.tzinfo.dst(None)`，并且在后者不返回 `None` 或者值小于一天的 `timedelta` 对象时引发异常。

### fromisoformat()

（类方法）返回一个对应于以如下格式给出的日期时间字符串的 `datetime` 对象。

```
YYYY-MM-DD[*HH[:MM[:SS[.fff[fff]]]][+HH:MM[:SS[.ffffff]]]]
```

其中 `*` 可以匹配任意的单个字符。

```python
>>> datetime.fromisoformat('2011-11-04')
datetime.datetime(2011, 11, 4, 0, 0)
>>> datetime.fromisoformat('2011-11-04T00:05:23')
datetime.datetime(2011, 11, 4, 0, 5, 23)
>>> datetime.fromisoformat('2011-11-04 00:05:23.283')
datetime.datetime(2011, 11, 4, 0, 5, 23, 283000)
>>> datetime.fromisoformat('2011-11-04 00:05:23.283+00:00')
datetime.datetime(2011, 11, 4, 0, 5, 23, 283000, tzinfo=datetime.timezone.utc)
>>> datetime.fromisoformat('2011-11-04T00:05:23+04:00')   
datetime.datetime(2011, 11, 4, 0, 5, 23,
    tzinfo=datetime.timezone(datetime.timedelta(seconds=14400)))
```

### fromtimestamp()

（类方法）返回对应于 POSIX 时间戳的 `datetime` 对象。

### fromordinal()

（类方法）返回对应于公历序号的 `datetime` 对象，其中公元 1 年 1 月 1 日的序号为 1。

### hour, minute, second, microsecond

时，分，秒，微秒。

### isoformat()

```python
isoformat(sep='T', timespec='auto')
```

返回一个以 ISO 8601 格式表示的日期时间字符串：

* `YYYY-MM-DDTHH:MM:SS.ffffff`，如果 `microsecond` 不为 0
* `YYYY-MM-DDTHH:MM:SS`，如果 `microsecond` 为 0

如果 `utcoffset()` 返回值不为 `None`，则增加一个字符串来给出 UTC 时差：

* `YYYY-MM-DDTHH:MM:SS.ffffff+HH:MM[:SS[.ffffff]]`，如果 `microsecond` 不为 0
* `YYYY-MM-DDTHH:MM:SS+HH:MM[:SS[.ffffff]]`，如果 `microsecond` 为 0

可选参数 *sep* (默认为 `'T'`) 为单个分隔字符，会被放在结果的日期和时间两部分之间。

可选参数 *timespec* 指定了包含的时间成分 (默认为 `'auto'`)，它可以是以下值之一：

* `'auto'`: 如果 `microsecond` 为 0 则与 `'seconds'` 相同，否则与 `'microseconds'` 相同。
* `'hours'`: 以两个数码的 `HH` 格式包含 `hour`。
* `'minutes'`: 以 `HH:MM` 格式包含 `hour` 和 `minute`。
* `'seconds'`: 以 `HH:MM:SS` 格式包含 `hour`, `minute` 和 `second`。
* `'milliseconds'`: 包含完整时间，但将秒值的小数部分截断至微秒，格式为 `HH:MM:SS.sss`.。
* `'microseconds'`: 以 `HH:MM:SS.ffffff` 格式包含完整时间。

```python
>>> datetime(2019, 5, 18, 15, 17, 8, 132263).isoformat()
'2019-05-18T15:17:08.132263'
>>> datetime(2019, 5, 18, 15, 17, tzinfo=timezone.utc).isoformat()
'2019-05-18T15:17:00+00:00'
```

### isoweekday()

返回日期是星期几，星期一为 1，星期日为 7。

```python
>>> dt = datetime(2020, 11, 27)
>>> dt.isoweekday()
5
```

### now()

（类方法）返回表示当前本地时间的 `datetime` 对象。

如果可选参数 *tz* 为 `None` 或未指定，则此方法类似于 `today()`，但它会在可能的情况下提供比通过 `time.time()` 时间戳所获取的时间值更高的精度（例如，在提供了 C `gettimeofday()` 函数的平台上就可以做到这一点）。

如果 *tz* 不为 `None`，它必须是 `tzinfo` 子类的一个实例，并且当前日期和时间将被转换到 *tz* 时区。

此方法可以替代 `today()` 和 `utcnow()`。

### replace()

替换日期时间中的部分值。

```python
>>> dt = datetime(2020, 11, 27, 15, 17, 8, 132263)
>>> dt.replace(day=26)
datetime.datetime(2020, 11, 26, 15, 17, 8, 132263)
```

### strftime()

返回一个由显式格式字符串所指明的代表日期时间的字符串。

以下列表显示了 1989 版 C 标准所要求的全部格式代码，它们在带有标准 C 实现的所有平台上均可使用。

|      |                                                                                                                               |                                                    |
| :--- | :---------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------- |
| 指令 | 含义                                                                                                                          | 示例                                               |
| `%a` | 当地工作日的缩写。                                                                                                            | Sun, Mon, ..., Sat                                 |
| `%A` | 本地化的星期中每日的完整名称。                                                                                                | Sunday, Monday, ..., Saturday                      |
| `%w` | 以十进制数显示的工作日，其中0表示星期日，6表示星期六。                                                                        | 0, 1, ..., 6                                       |
| `%d` | 补零后，以十进制数显示的月份中的一天。                                                                                        | 01, 02, ..., 31                                    |
| `%b` | 当地月份的缩写。                                                                                                              | Jan, Feb, ..., Dec                                 |
| `%B` | 本地化的月份全名。                                                                                                            | January, February, ..., December                   |
| `%m` | 补零后，以十进制数显示的月份。                                                                                                | 01, 02, ..., 12                                    |
| `%y` | 补零后，以十进制数表示的，不带世纪的年份。                                                                                    | 00, 01, ..., 99                                    |
| `%Y` | 十进制数表示的带世纪的年份。                                                                                                  | 0001, 0002, ..., 2013, 2014, ..., 9998, 9999       |
| `%H` | 以补零后的十进制数表示的小时（24 小时制）。                                                                                   | 00, 01, ..., 23                                    |
| `%I` | 以补零后的十进制数表示的小时（12 小时制）。                                                                                   | 01, 02, ..., 12                                    |
| `%p` | 本地化的 AM 或 PM 。                                                                                                          | AM, PM                                             |
| `%M` | 补零后，以十进制数显示的分钟。                                                                                                | 00, 01, ..., 59                                    |
| `%S` | 补零后，以十进制数显示的秒。                                                                                                  | 00, 01, ..., 59                                    |
| `%f` | 以十进制数表示的微秒，在左侧补零。                                                                                            | 000000, 000001, ..., 999999                        |
| `%z` | UTC 偏移量，格式为 `±HHMM[SS[.ffffff]]` （如果是简单型对象则为空字符串）。                                                    | (空), +0000, -0400, +1030, +063415, -030712.345216 |
| `%Z` | 时区名称（如果对象为简单型则为空字符串）。                                                                                    | (空), UTC, EST, CST                                |
| `%j` | 以补零后的十进制数表示的一年中的日序号。                                                                                      | 001, 002, ..., 366                                 |
| `%U` | 以补零后的十进制数表示的一年中的周序号（星期日作为每周的第一天）。在新的一年中第一个星期日之前的所有日子都被视为是在第 0 周。 | 00, 01, ..., 53                                    |
| `%W` | 以十进制数表示的一年中的周序号（星期一作为每周的第一天）。在新的一年中第一个星期日之前的所有日子都被视为是在第 0 周。         | 00, 01, ..., 53                                    |
| `%c` | 本地化的适当日期和时间表示。                                                                                                  | Tue Aug 16 21:30:00 1988                           |
| `%x` | 本地化的适当日期表示。                                                                                                        | 08/16/88 (None);08/16/1988                         |
| `%X` | 本地化的适当时间表示。                                                                                                        | 21:30:00                                           |
| `%%` | 字面的 `'%'` 字符。                                                                                                           | %                                                  |

```python
>>> dt = datetime.now()
>>> dt.strftime("%I:%M %p, %B %d %Y, %A")
'01:47 PM, November 27 2020, Friday'
```

### time()

返回具有同样 hour、minute、second、microsecond 和 fold 值的 `time` 对象，其 `tzinfo` 值为 `None`。

### timestamp()

返回对应于 `datetime` 实例的 POSIX 时间戳。此返回值是与 `time.time()` 返回值类似的浮点数。

### timetz()

返回具有同样 hour、minute、second、microsecond、fold 和 tzinfo 值的 `time` 对象。

### timetuple()

返回一个 `time.struct_time` 对象，即 `time.localtime()` 所返回的类型。

`dt.timetuple()` 等价于:

```python
time.struct_time((dt.year, dt.month, dt.day,
                  dt.hour, dt.minute, dt.second,
                  dt.weekday(), yday, dst))
```

其中 `yday = dt.toordinal() - date(dt.year, 1, 1).toordinal() + 1` 是日期在当前年份中的序号，起始序号 1 表示 1 月 1 日。结果的 `tm_isdst` 旗标的设定会依据 `dst()` 方法：如果 `tzinfo` 为 `None` 或 `dst()` 返回 `None`，则 `tm_isdst` 将设为 -1；否则如果 `dst()` 返回一个非零值，则 `tm_isdst` 将设为 1；在其他情况下 `tm_isdst` 将设为 0。

### today()

（类方法）返回表示当前本地时间的 `datetime` 对象，其中 `tzinfo` 为 `None`。其等价于：

```python
datetime.fromtimestamp(time.time())
```

### toordinal()

返回日期的公历序号，其中公元 1 年 1 月 1 日的序号为 1。与 `self.date().toordinal()` 相同。

### tzname()

如果 `tzinfo` 为 `None`，则返回 `None`，否则返回 `self.tzinfo.tzname(None)`，并且在后者不返回 `None` 或者值小于一天的 `timedelta` 对象时引发异常。

### tzinfo

作为 *tzinfo* 参数被传给 `datetime` 构造器的对象，如果没有传入值则为 `None`。

### utcnow()

（类方法）返回表示当前 UTC 时间的 `datetime` 对象，其中 `tzinfo` 为 `None`。

### utcoffset()

如果 `tzinfo` 为 `None`，则返回 `None`，否则返回 `self.tzinfo.utcoffset(None)`，并且在后者不返回 `None` 或者值小于一天的 `timedelta` 对象时引发异常。

### weekday()

返回日期是星期几，星期一为 0，星期日为 6。

```python
>>> dt = datetime(2020, 11, 27)
>>> dt.weekday()
4
```

### year, month, day

年，月，日。

## tzinfo

一个描述时区信息对象的抽象基类。用来给 `datetime` 和 `time` 类提供自定义的时间调整概念（例如处理时区和/或夏令时）。



## timezone

一个实现了 `tzinfo` 抽象基类的子类，用于表示相对于 UTC 的偏移量。
