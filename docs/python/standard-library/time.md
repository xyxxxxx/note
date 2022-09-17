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

返回以浮点数/整数表示的当前纪元秒数/纳秒数值。纪元的具体日期和闰秒的处理取决于平台。

```python
>>> time.time()
1617002884.9367008
>>> time.time_ns()
16170028849367008
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
