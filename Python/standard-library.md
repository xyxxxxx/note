[toc]

## [argparse](https://docs.python.org/zh-cn/3/library/argparse.html)——命令行选项、参数和子命令解析器

如果脚本很简单或者临时使用，可以使用`sys.argv`直接读取命令行参数。`sys.argv`返回一个参数列表，其中首个元素是程序名，随后是命令行参数，所有元素都是字符串类型。例如以下脚本：

```python
# test.py

import sys

print("Input argument is %s" %(sys.argv))
```

```shell
$ python3 test.py 1 2 -a 3
Input argument is ['test.py', '1', '2', '-a', '3']
```



`argparse`模块可以让人轻松编写用户友好的命令行接口。程序定义它需要的参数，然后 `argparse` 将弄清如何从 `sys.argv` 解析出那些参数。 `argparse` 模块还会自动生成帮助和使用手册，并在用户给程序传入无效参数时报出错误信息。

教程：[Argparse 教程](https://docs.python.org/zh-cn/3/howto/argparse.html)

文档：[argparse --- 命令行选项、参数和子命令解析器](https://docs.python.org/zh-cn/3.9/library/argparse.html)

```python
# 简单的argparse实例
import argparse
parser = argparse.ArgumentParser()
# 位置参数, type表示解析类型, 默认为str
parser.add_argument("square", type=int,
                    help="display a square of a given number")
# 可选参数, 可以设置短选项, action="count"表示计数参数的出现次数
parser.add_argument("-v", "--verbosity", action="count", default=0,
                    help="increase output verbosity")
# 进行参数解析
args = parser.parse_args()
answer = args.square**2
if args.verbosity >= 2:
    print("the square of {} equals {}".format(args.square, answer))
elif args.verbosity >= 1:
    print("{}^2 == {}".format(args.square, answer))
else:
    print(answer)
```

```python
# add_argument实例

parser.add_argument('-m', '--model', nargs='*', choices=['NB', 'LR', 'SVML'], default=['NB'], help="model used to classify spam and ham")
# 限定范围多选, 有默认值

parser.add_argument('-s', '--stopwords', nargs='?', default=False, const=True, help="model used to classify spam and ham")
# default为选项未出现时取值, const为选项后没有参数时的取值
# 因此-s表示True, 没有该选项表示False
```







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

### 文件和目录

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



### 环境变量

```python
>>> import os
>>> os.environ
environ({'CLUTTER_IM_MODULE': 'xim', 'LS_COLORS': 
# ...
/usr/bin/lesspipe %s', 'GTK_IM_MODULE': 'fcitx', 'LC_TIME': 'en_US.UTF-8', '_': '/usr/bin/python3'})
>>> os.environ['HOME']
'/home/xyx' 
>>> os.environ['MASTER_ADDR'] = 'localhost'      # 环境变量赋值
>>> os.getenv('MASTER_ADDR')
'localhost'         
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

### exit

从Python中退出，实现方式是抛出一个`SystemExit`异常。

可选参数可以是表示退出状态的整数（默认为整数0），也可以是其他类型的对象。如果它是整数，则shell等将0视为“成功终止”，非零值视为“异常终止”。



### platform

本字符串是一个平台标识符，对于各种系统的值为：

| 系统           | `平台` 值  |
| :------------- | :--------- |
| AIX            | `'aix'`    |
| Linux          | `'linux'`  |
| Windows        | `'win32'`  |
| Windows/Cygwin | `'cygwin'` |
| macOS          | `'darwin'` |



### stdin, stdout, stderr

解释器用于标准输入、标准输出和标准错误的文件对象：

+ `stdin`用于所有交互式输入
+ `stdout`用于`print()`和expression语句的输出，以及输出`input()`的提示符
+ 解释器自身的提示符和错误消息发往`stderr`



### version, version_info

`version`是一个包含Python解释器版本号、编译版本号、所用编译器等信息的字符串，`version_info`是一个包含版本号五部分的元组: *major*, *minor*, *micro*, *releaselevel* 和 *serial*。

```python
>>> sys.version
'3.6.9 (default, Oct  8 2020, 12:12:24) \n[GCC 8.4.0]'
>>> sys.version_info
sys.version_info(major=3, minor=6, micro=9, releaselevel='final', serial=0)
```



## tempfile——生成临时文件和目录



### gettempdir()

返回放置临时文件的目录的名称。

Python搜索标准目录列表，以找到调用者可以在其中创建文件的目录。这个列表是：

1. `TMPDIR` ,`TEMP`或`TMP` 环境变量指向的目录。
2. 与平台相关的位置：
   + 在 Windows 上，依次为 `C:\TEMP`、`C:\TMP`、`\TEMP` 和 `\TMP`
   + 在所有其他平台上，依次为 `/tmp`、`/var/tmp` 和 `/usr/tmp`
3. 不得已时，使用当前工作目录。