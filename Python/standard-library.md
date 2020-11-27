## [`argparse`](https://docs.python.org/zh-cn/3/library/argparse.html)——命令行选项、参数和子命令解析器

如果脚本很简单或者临时使用，可以使用`sys.argv`直接读取命令行参数。`sys.argv`返回一个参数列表，其中首个元素是程序名，随后是命令行参数，所有元素都是字符串类型。例如以下脚本：

```python
# test.py

import sys

print "Input argument is %s" %(sys.argv)
```

```shell
$ python3 test.py 1 2 -a 3
Input argument is ['test.py', '1', '2', '-a', '3']
```



`argparse`模块可以让人轻松编写用户友好的命令行接口。程序定义它需要的参数，然后 `argparse` 将弄清如何从 `sys.argv` 解析出那些参数。 `argparse` 模块还会自动生成帮助和使用手册，并在用户给程序传入无效参数时报出错误信息。

教程：[Argparse 教程](https://docs.python.org/zh-cn/3/howto/argparse.html)



## [datetime](https://docs.python.org/zh-cn/3/library/datetime.html)——处理日期和时间

### timedelta

```python
import datetime
from datetime import timedelta

>>> delta = timedelta(
...     weeks=2,              # 1星期转换成7天
...     days=50,
...     hours=8,              # 1小时转换成3600秒
...     minutes=5,            # 1小时转换成60秒
...     seconds=27,
...     milliseconds=29000,   # 1毫秒转换成1000微秒
...     microseconds=10
... )
>>> delta
datetime.timedelta(64, 29156, 10)   # 日,秒,毫秒
>>> delta.total_seconds()
5558756.00001                       # 秒

>>> d1 = timedelta(minutes=5)
>>> d2 = timedelta(seconds=20)
>>> d1 + d2
datetime.timedelta(0, 320)
>>> d1 - d2
datetime.timedelta(0, 280)
>>> d1 * 2
datetime.timedelta(0, 600)
>>> d1 / d2
15.0
>>> d1 // d2
15
```



### date

```python
import datetime
from datetime import date

>>> date(
...     year=2020,
...     month=11,
...     day=27
... )
datetime.date(2020, 11, 27)
>>> date.today()
datetime.date(2020, 11, 27)
>>> date.fromtimestamp(1606468517.547344)
datetime.date(2020, 11, 27)
>>> date.today().weekday()
4                              # Friday
>>> date.today().isoweekday()
5                              # Friday
```



### datetime

```python
import datetime
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



## [os](https://docs.python.org/zh-cn/3/library/os.html)——多种操作系统接口

```python
import os

# test/
#     dir1/
#         file2
#     file1

>>> os.getcwd()             # 当前路径
'/home/test'
>>> os.chdir('dir1')        # 切换路径
>>> os.getcwd()
'/home/test/dir1'
>>> os.listdir()            # ls
['dir1', 'file1']
>>> os.mkdir('dir2')        # 创建目录
>>> os.rename('dir2', 'dir3')  # 重命名目录或文件
>>> os.rmdir('dir2')        # 删除目录
>>> os.remove('file1')      # 删除文件
```



## [os.path](https://docs.python.org/zh-cn/3/library/os.path.html)——常用路径操作

```python
from os import path

# test/
#     dir1/
#         file2
#     file1

>>> path.abspath('.')      # 路径的绝对路径
'/home/test'
>>> path.exists('./dir1')  # 存在路径
True
>>> path.exists('./dir2')
False
>>> path.isdir('dir1')     # 判断目录
True
>>> path.isfile('dir1')    # 判断文件
False
>>> path.isfile('file1')
True
>>> path.isdir('dir2')     # 不存在的目录
False
>>> path.getsize('file1')  # 文件大小 
14560
```



## [sys](https://docs.python.org/zh-cn/3/library/sys.html)——系统相关的参数和函数

```python
import sys

>>> sys.stdin              # 标准输入 
<_io.TextIOWrapper name='<stdin>' mode='r' encoding='UTF-8'>
>>> sys.stdout             # 标准输出
<_io.TextIOWrapper name='<stdout>' mode='w' encoding='UTF-8'>
>>> sys.stderr             # 标准错误
<_io.TextIOWrapper name='<stderr>' mode='w' encoding='UTF-8'>
>>> sys.exit()             # 退出Python

```

