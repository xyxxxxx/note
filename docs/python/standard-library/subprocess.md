# subprocess——子进程管理

`subprocess` 模块允许你生成新的进程，连接它们的输入、输出、错误管道，并且获取它们的返回码。

大多数情况下，推荐使用 `run()` 函数调用子进程，执行操作系统命令。在更高级的使用场景，你还可以使用 `Popen` 接口。实际上 `run()` 函数在底层调用的就是 `Popen` 接口。

## run()

```python
subprocess.run(args, *, stdin=None, input=None, stdout=None, stderr=None, capture_output=False, shell=False, cwd=None, timeout=None, check=False, encoding=None, errors=None, text=None, env=None, universal_newlines=None, **other_popen_kwargs)
# args       要执行的命令,必须是一个字符串或参数列表.若为字符串,则`shell`必须为True,
#                或者该字符串仅包含命令的名称而不指定任何参数
# stdin,stdout,stderr  子进程的标准输入、输出和错误文件句柄,其值可以是`subprocess.PIPE`,
#                `subprocess.DEVNULL`,已经存在的文件描述符(一个正整数),已经打开的文件对象或者
#                None.`subprocess.PIPE`表示为子进程创建新的管道,`subprocess.DEVNULL`表示
#                使用`os.devnull`.使用默认值None时,将不会进行重定向,子进程的文件句柄将继承自父进程
# timeout    命令超时时间.如果命令执行时间超时,子进程将被杀死,并引发TimeoutExpired异常
# check      若为True,并且进程退出状态码不是0,则引发CalledProcessError异常
# encoding   若指定了该参数,则stdin,stdout,stderr可以接收字符串数据,并以该编码方式编码.
#                否则只接收bytes类型的数据
# shell      若为True,将通过操作系统的shell执行指定的命令
```

运行 *arg* 描述的命令，等待命令完成, 然后返回一个 `CompletedProcess` 实例.

此函数的完整签名和 `Popen` 构造器大体相同——此函数接受的大部分参数都被传递给该接口（*timeout*、*input*、*check* 和 *capture_output* 除外）。

如果 *capture_output* 设为 True，stdout 和 stderr 将会被捕获。在使用时，内置的 `Popen` 对象将自动用 `stdout=PIPE` 和 `stderr=PIPE` 创建。*stdout* 和 *stderr* 参数不应与 *capture_output* 同时提供。如果你希望捕获并将两个流合并在一起，使用 `stdout=PIPE` 和 `stderr=STDOUT` 来代替 *capture_output*。

*timeout* 参数将被传递给 `Popen.communicate()`。如果发生超时，子进程将被杀死并等待。`TimeoutExpired` 异常将在子进程中断后被抛出。

*input* 参数将被传递给 `Popen.communicate()` 以及子进程的 stdin。如果使用此参数，它必须是一个字节序列。如果指定了 *encoding* 或 *errors* 或者将 *text* 设置为 True，那么它也可以是一个字符串。当使用此参数时，在创建内部 `Popen` 对象时将自动带上 `stdin=PIPE`，并且不能再手动指定 *stdin* 参数。

如果 *check* 设为 True，并且进程以非零状态码退出，一个 `CalledProcessError` 异常将被抛出。这个异常的属性将设置为参数、退出码、以及标准输出和标准错误（如果被捕获到）。

如果 *encoding* 或 *error* 被指定，或者 *text* 被设为 True，标准输入、标准输出和标准错误的文件对象将通过指定的 *encoding* 和 *errors* 以文本模式打开，否则以默认的 `io.TextIOWrapper` 打开。*universal_newline* 参数等同于 *text* 并且提供了向后兼容性。默认情况下，文件对象是以二进制模式打开的。

如果 *env* 不是 None，它必须是一个字典，为新的进程设置环境变量；它用于替换继承当前进程的环境的默认行为。它将直接被传递给 `Popen`。

```python
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
...
FileNotFoundError: [Errno 2] No such file or directory: 'ls -l': 'ls -l'
            
>>> subprocess.run('ls -l', shell=True)
total 4
-rw-rw-r-- 1 xyx xyx 2862 Dec  1 17:11 lstm.py
CompletedProcess(args='ls -l', returncode=0)
```

## CompletedProcess

`run()` 函数的返回类型，包含下列属性：

### args

用于启动进程的参数，是字符串或字符串列表。

### returncode

子进程的退出状态码，0 表示进程正常退出。

负值 `-N` 表示子进程被信号 `N` 中断 (仅 POSIX)。

### stdout

捕获到的子进程的标准输出，是一个字节序列，或者一个字符串（如果 `run()` 设置了参数 `encoding`、`errors` 或 `text=True`）。如果未有捕获，则为 None。

如果设置了参数 `stderr=subprocess.STDOUT`，标准错误会随同标准输出被捕获，并且 `stderr` 将为 None。

### stderr

捕获到的子进程的标准错误，是一个字节序列，或者一个字符串（如果 `run()` 设置了参数 `encoding`、`errors` 或 `text=True`）。如果未有捕获，则为 None。

### check_returncode()

检查 `returncode`，非零则引发 `CalledProcessError`。

## Popen

## 异常
