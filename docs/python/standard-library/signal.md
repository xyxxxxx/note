# signal——设置异步事件处理程序

`signal` 模块提供了在 Python 中使用信号处理程序的机制。

## 一般规则

`signal.signal()` 函数允许定义在接收到信号时执行的自定义处理程序。少量的默认处理程序已经设置：`SIGPIPE` 被忽略（因此管道和套接字上的写入错误可以报告为普通的 Python 异常）；如果父进程没有更改 `SIGINT`，则其会被翻译成 `KeyboardInterrupt` 异常。

一旦设置，对于特定信号的处理程序将保持安装，直到它被显式重置（Python 模拟 BSD 样式接口而不管底层实现），但 `SIGCHLD` 的处理程序除外，它遵循底层实现。

### 执行 Python 信号处理程序

Python 信号处理程序不会在低级（C）信号处理程序中执行。相反，低级信号处理程序设置一个标志，告诉虚拟机稍后执行相应的 Python 信号处理程序（例如在下一个 bytecode 指令）。这会导致：

* 捕获同步错误是没有意义的，例如 `SIGFPE` 或 `SIGSEGV`，它们是由 C 代码中的无效操作引起的。Python 将从信号处理程序返回到 C 代码，这可能会再次引发相同的信号，导致 Python 显然地挂起。从 Python 3.3 开始，你可以使用 `faulthandler` 模块来报告同步错误。
* 纯 C 中实现的长时间运行的计算（例如在大量文本上的正则表达式匹配）可以在任意时间内不间断地运行，不管接收到任何信号。计算完成后将调用 Python 信号处理程序。

### 信号与线程

Python 信号处理程序总是在主 Python 线程中执行，即使信号是在另一个线程中接收的。这意味着信号不能用作线程间通信的手段。你可以使用 `threading` 模块中的同步原函数。

此外，只允许主线程设置新的信号处理程序。

## 模块内容

### 变量

#### SIG_DFL

这是两种标准信号处理选项之一，它将执行信号的默认函数。例如，在大多数系统上，对于 `SIGQUIT` 的默认操作是转储核心并退出，而对于 `SIGCHLD` 的默认操作是简单地忽略它。

#### SIG_IGN

这是两种标准信号处理选项之一，它将简单地忽略给定的信号。

#### SIGABRT

来自 [abort(3)](https://manpages.debian.org/abort(3)) 的中止信号。

#### SIGALRM

来自 [alarm(2)](https://manpages.debian.org/bullseye/manpages-dev/alarm.2.en.html) 的计时器信号。

可用性：Unix。

#### SIGBUS

总线错误 (非法的内存访问)。

可用性：Unix。

#### SIGCHLD

子进程被停止或终结。

可用性：Unix。

#### SIGHUP

在控制终端上检测到挂起或控制进程的终止。

可用性：Unix。

#### SIGILL

非法指令。

#### SIGINT

来自键盘的中断（CTRL + C）。

默认的动作是引发 `KeyboardInterrupt`。

#### SIGKILL

终止信号。

它不能被捕获、阻塞或忽略。

可用性：Unix。

#### SIGPIPE

损坏的管道：写入到没有读取器的管道。

默认的动作是忽略此信号。

可用性：Unix。

#### SIGSEGV

段错误：无效的内存引用。

#### SIGTERM

终结信号。

### 函数

#### alarm()

```python
signal.alarm(time)
```

如果 *time* 值非零，则此函数将要求将一个 `SIGALRM` 信号在 *time* 秒之后发送到进程。任何在之前排入计划的警报都会被取消（在任何时刻都只能有一个警报被排入计划）。后续的返回值是任何之前设置的警报被传入之前的秒数。如果 *time* 值为零，则不会将任何警报排入计划，并且任何已排入计划的警报都会被取消。如果返回值为零，则目前没有任何警报被排入计划。

可用性：Unix。 更多信息请参见手册页面 [alarm(2)](https://manpages.debian.org/alarm(2))。

#### getsignal()

```python
signal.getsignal(signalnum)
```

返回当前用于信号 *signalnum* 的信号处理程序。返回值可以是一个 Python 可调用对象，或是特殊值 `signal.SIG_IGN`、`signal.SIG_DFL` 和 `None` 之一。在这里，`signal.SIG_IGN` 表示信号在之前被忽略，`signal.SIG_DFL` 表示之前在使用默认的信号处理方式，而 `None` 表示之前的信号处理程序未由 `Python` 安装。

#### strsignal()

```python
signal.strsignal(signalnum)
```

返回信号 *signalnum* 的系统描述，例如“Interrupt”、“Segmentation fault”等。如果信号无法被识别则返回 `None`。

#### valid_signals()

返回本平台上的有效信号编号集。 这可能会少于 range(1, NSIG)，如果某些信号被系统保留作为内部使用的话。

#### pause()

使进程休眠直至接收到一个信号；然后将会调用适当的处理程序。

可用性：Unix。更多信息请参见手册页面 [signal(2)](https://manpages.debian.org/signal(2))。

另请参阅 `sigwait()`、`sigwaitinfo()`、`sigtimedwait()` 和 `sigpending()`。

#### raise_signal()

向调用进程发送一个信号。

#### signal()

```python
signal.signal(signalnum, handler)
```

将信号 *signalnum* 的处理程序设为函数 *handler*。*handler* 可以是接受两个参数（见下）的 Python 可调用对象，或者是特殊值 `signal.SIG_IGN` 或 `signal.SIG_DFL` 之一。之前的信号处理程序将被返回（参见上文 `getsignal()` 的描述）。（更多信息请参阅 Unix 手册页面 [signal(2)](https://manpages.debian.org/signal(2))。）

当启用多线程时，此函数只能从主线程被调用；尝试从其他线程调用此函数将导致引发一个 `ValueError` 异常。

*handler* 将附带两个参数调用：信号编号和当前堆栈帧 (`None` 或一个帧对象；有关帧对象的描述请参阅[类型层级结构描述](https://docs.python.org/zh-cn/3.8/reference/datamodel.html#frame-objects)或者参阅 [`inspect`](./inspect.md) 模块中的属性描述)。

在 Windows 上，`signal()` 调用只能附带 `SIGABRT`、`SIGFPE`、`SIGILL`、`SIGINT`、`SIGSEGV`、`SIGTERM` 或 `SIGBREAK`，任何其他值都将引发 `ValueError`。请注意不是所有系统都定义了同样的信号名称集合；如果一个信号名称未被定义为 `SIG*` 模块层级常量则将引发 `AttributeError`。

#### sigwait()

```python
signal.sigwait(sigset)
```

挂起调用线程的执行直到信号集合 *sigset* 中指定的信号之一被传送。此函数会接受该信号（将其从等待信号列表中移除），并返回信号编号。

可用性：Unix。更多信息请参见手册页面 [sigwait(3)](https://manpages.debian.org/sigwait(3))。

另请参阅 `pause()`、`pthread_sigmask()`、`sigpending()`、`sigwaitinfo()` 和 `sigtimedwait()`。
