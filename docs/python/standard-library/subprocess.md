# subprocess——子进程管理

`subprocess` 模块允许我们生成新的进程，连接它们的输入、输出、错误管道，并且获取它们的返回码。

大多数情况下，推荐使用 `run()` 方法调用子进程，执行操作系统命令。在更高级的使用场景，你还可以使用 `Popen` 接口。其实 `run()` 方法在底层调用的就是 `Popen` 接口。

## run()

```python
subprocess.run(args, *, stdin=None, input=None, stdout=None, stderr=None, shell=False, timeout=None, check=False, encoding=None, errors=None)

# args       要执行的命令.必须是一个字符串或参数列表.
# stdin,stdout,stderr  子进程的标准输入、输出和错误,其值可以是subprocess.PIPE,
#                subprocess.DEVNULL,一个已经存在的文件描述符,已经打开的文件对象或者
#                None.subprocess.PIPE表示为子进程创建新的管道,subprocess.
#                DEVNULL表示使用os.devnull.默认使用的是None,表示什么都不做.
# timeout    命令超时时间.如果命令执行时间超时,子进程将被杀死,并引发TimeoutExpired异常.
# check      若为True,并且进程退出状态码不是0,则引发CalledProcessError 异常.
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

注意当 `args` 是一个字符串时，必须指定 `shell=True`：

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

## CompletedProcess

`run()` 方法的返回类型，包含下列属性：

**`args`**

启动进程的参数，是字符串或字符串列表。

**`returncode`**

子进程的退出状态码，0 表示进程正常退出。

**`stdout`**

捕获到的子进程的标准输出，是一个字节序列，或者一个字符串（如果 `run()` 设置了参数 `encoding`，`errors` 或 `text=True`）。如果未有捕获，则为 `None`。

如果设置了参数 `stderr=subprocess.STDOUT`，标准错误会随同标准输出被捕获，并且 `stderr` 将为 `None`。

**`stderr`**

捕获到的子进程的标准错误，是一个字节序列，或者一个字符串（如果 `run()` 设置了参数 `encoding`，`errors` 或 `text=True`）。如果未有捕获，则为 `None`。

**`check_returncode`（）**

检查 `returncode`，非零则引发 `CalledProcessError`。

## Popen
