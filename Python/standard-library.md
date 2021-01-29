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

### `timedelta`

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



### `date`

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



### `datetime`

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



## multiprocessing——基于进程的并行

multiprocessing 是一个支持使用与 threading 模块类似的 API 来产生进程的包。 multiprocessing 包同时提供了本地和远程并发操作，通过使用子进程而非线程有效地绕过了全局解释器锁。 因此，multiprocessing 模块允许程序员充分利用给定机器上的多个处理器。 它在 Unix 和 Windows 上均可运行。



### `Process`

```python
class multiprocessing.Process(group=None, target=None, name=None, args=(), kwargs={}, *, daemon=None)
# target    由run()方法调用的可调用对象
# args      目标调用的顺序参数
# kwargs    目标调用的关键字参数
```

进程对象表示在单独进程中运行的活动。`Process`类拥有和`threading.Thread`等价的大部分方法。

**`run()`**

表示进程活动的方法。

**`start()`**

启动进程活动。

这个方法每个进程对象最多只能调用一次。它会将对象的`run()`方法安排在一个单独的进程中调用。

**`join([timeout])`**

如果可选参数 *timeout* 是 `None` （默认值），则该方法将阻塞，直到调用`join()`方法的进程终止；如果 *timeout* 是一个正数，它最多会阻塞 *timeout* 秒。不管是进程终止还是方法超时，该方法都返回 `None`。

一个进程可以被`join`多次。

进程无法`join`自身，因为这会导致死锁。尝试在启动进程之前`join`进程会产生一个错误。

**`is_alive()`**

返回进程是否处于活动状态。从`start()`方法返回到子进程终止之间，进程对象都处于活动状态。

**`name`**

进程的名称。该名称是一个字符串，仅用于识别，没有具体语义。可以为多个进程指定相同的名称。

**`daemon`**

进程的守护标志，一个布尔值。必须在`start()`被调用之前设置。

初始值继承自创建进程。

当一个进程退出时，它会尝试终止子进程中的所有守护进程。

**`pid`**

返回进程ID。

**`exitcode`**

子进程的退出代码。`None`表示进程尚未终止；负值-N表示子进程被信号N终止。

**`terminate()`**

终止进程。在Unix上由`SIGTERM`信号完成。

**`kill()`**

与`terminate()`相同，但在Unix上使用`SIGKILL`信号。

**`close`()**

关闭`Process`对象，释放与之关联的所有资源。如果底层进程仍在运行，则会引发`ValueError`。一旦`close()`成功返回，`Process`对象的大多数其他方法和属性将引发`ValueError`。



### `Pipe`



### `Connection`



### `Queue`



### `cpu_count()`

返回系统的CPU数量。



### `current_process`()

返回当前进程相对应的`Process`对象。



### `parent_process`()

返回当前进程的父进程相对应的`Process`对象。



### `Lock`



### `RLock`



### `Semaphore`



### `Value()`



### `Array()`



### `Manager()`



### `Pool`





## [os](https://docs.python.org/zh-cn/3/library/os.html)——多种操作系统接口

### 进程

> 参考[进程和线程](./process-and-thread.md)



### 文件和目录

```python
# test/
#     dir1/
#         file2
#     file1

>>> import os
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



## subprocess——子进程管理

`subprocess`模块允许我们生成新的进程，连接它们的输入、输出、错误管道，并且获取它们的返回码。

大多数情况下，推荐使用`run()`方法调用子进程，执行操作系统命令。在更高级的使用场景，你还可以使用`Popen`接口。其实`run()`方法在底层调用的就是`Popen`接口。

### `run()`

```python
subprocess.run(args, *, stdin=None, input=None, stdout=None, stderr=None, shell=False, timeout=None, check=False, encoding=None, errors=None)

# args       要执行的命令.必须是一个字符串或参数列表.
# stdin,stdout,stderr  子进程的标准输入、输出和错误,其值可以是subprocess.PIPE,
#                subprocess.DEVNULL,一个已经存在的文件描述符,已经打开的文件对象或者
#                None.subprocess.PIPE表示为子进程创建新的管道,subprocess.
#                DEVNULL表示使用os.devnull.默认使用的是None,表示什么都不做.
# timeout    命令超时时间.如果命令执行时间超时,子进程将被杀死,并抛出TimeoutExpired异常.
# check      若为True,并且进程退出状态码不是0,则抛出CalledProcessError 异常.
# encoding   若指定了该参数，则stdin,stdout,stderr可以接收字符串数据,并以该编码方
#                式编码.否则只接收bytes类型的数据.
# shell      若为True,将通过操作系统的shell执行指定的命令.
```

```python
>>> import subprocess

>>> subprocess.run(['ls', '-l'])                           # 打印标准输出
total 4
-rw-rw-r-- 1 xyx xyx 2862 Dec  1 17:11 lstm.py
CompletedProcess(args=['ls', '-l'], returncode=0)

>>> subprocess.run(['ls', '-l'], stdout=subprocess.PIPE)   # 捕获标准输出
CompletedProcess(args=['ls', '-l'], returncode=0, stdout=b'total 4\n-rw-rw-r-- 1 xyx xyx 2862 Dec  1 17:11 lstm.py\n')
```

注意当`args`是一个字符串时，必须指定`shell=True`：

```python
>>> subprocess.run('ls -l')
Traceback (most recent call last):
# ...
FileNotFoundError: [Errno 2] No such file or directory: 'ls -l': 'ls -l'
            
>>> subprocess.run('ls -l', shell=True)
total 4
-rw-rw-r-- 1 xyx xyx 2862 Dec  1 17:11 lstm.py
CompletedProcess(args='ls -l', returncode=0)
```





### `CompletedProcess`

`run()`方法的返回类型，包含下列属性：

**`args`**

启动进程的参数，是字符串或字符串列表。

**`returncode`**

子进程的退出状态码，0表示进程正常退出。

**`stdout`**

捕获到的子进程的标准输出，是一个字节序列，或者一个字符串（如果`run()`设置了参数`encoding`,`errors`或`text=True`）。如果未有捕获，则为`None`。

如果设置了参数`stderr=subprocess.STDOUT`，标准错误会随同标准输出被捕获，并且`stderr`将为`None`。

**`stderr`**

捕获到的子进程的标准错误，是一个字节序列，或者一个字符串（如果`run()`设置了参数`encoding`,`errors`或`text=True`）。如果未有捕获，则为`None`。

**`check_returncode`()**

检查`returncode`，非零则抛出`CalledProcessError`。



### `Popen`



## [sys](https://docs.python.org/zh-cn/3/library/sys.html)——系统相关的参数和函数

### `exit()`

从Python中退出，实现方式是抛出一个`SystemExit`异常。

可选参数可以是表示退出状态的整数（默认为整数0），也可以是其他类型的对象。如果它是整数，则shell等将0视为“成功终止”，非零值视为“异常终止”。



### `platform`

本字符串是一个平台标识符，对于各种系统的值为：

| 系统           | `平台` 值  |
| :------------- | :--------- |
| AIX            | `'aix'`    |
| Linux          | `'linux'`  |
| Windows        | `'win32'`  |
| Windows/Cygwin | `'cygwin'` |
| macOS          | `'darwin'` |



### `stdin`, `stdout`, `stderr`

解释器用于标准输入、标准输出和标准错误的文件对象：

+ `stdin`用于所有交互式输入
+ `stdout`用于`print()`和expression语句的输出，以及输出`input()`的提示符
+ 解释器自身的提示符和错误消息发往`stderr`



### `version`, `version_info`

`version`是一个包含Python解释器版本号、编译版本号、所用编译器等信息的字符串，`version_info`是一个包含版本号五部分的元组: *major*, *minor*, *micro*, *releaselevel* 和 *serial*。

```python
>>> sys.version
'3.6.9 (default, Oct  8 2020, 12:12:24) \n[GCC 8.4.0]'
>>> sys.version_info
sys.version_info(major=3, minor=6, micro=9, releaselevel='final', serial=0)
```



## tempfile——生成临时文件和目录



### `gettempdir()`

返回放置临时文件的目录的名称。

Python搜索标准目录列表，以找到调用者可以在其中创建文件的目录。这个列表是：

1. `TMPDIR` ,`TEMP`或`TMP` 环境变量指向的目录。
2. 与平台相关的位置：
   + 在 Windows 上，依次为 `C:\TEMP`、`C:\TMP`、`\TEMP` 和 `\TMP`
   + 在所有其他平台上，依次为 `/tmp`、`/var/tmp` 和 `/usr/tmp`
3. 不得已时，使用当前工作目录。





## threading——基于线程的并行

### `active_count()`

返回当前存活的`Thread`对象的数量。



### `current_thread()`

返回当前调用者的控制线程的`Thread`对象。



### `main_thread()`

返回主`Thread`对象。



### `Thread`





### `Lock`

原始锁处于 "锁定" 或者 "非锁定" 两种状态之一。它有两个基本方法，`acquire()`和`release()`。当状态为非锁定时，`acquire()`将状态改为锁定并立即返回；当状态是锁定时，`acquire()`将阻塞至其他线程调用`release()`将其改为非锁定状态，然后`acquire()`重置其为锁定状态并返回。 `release()`只在锁定状态下调用，将状态改为非锁定并立即返回。如果尝试释放一个非锁定的锁，则会引发`RuntimeError` 异常。

原始锁在创建时为非锁定状态。当多个线程在`acquire()`阻塞，然后`release()`重置状态为未锁定时，只有一个线程能继续执行；至于哪个线程继续执行则没有定义，并且会根据实现而不同。

