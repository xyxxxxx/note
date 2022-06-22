# tempfile——生成临时文件和目录

`tempfile` 模块用于创建临时文件和目录，它可以跨平台使用。

## TemporaryFile

```python
tempfile.TemporaryFile(mode='w+b', buffering=-1, encoding=None, newline=None, suffix=None, prefix=None, dir=None, *, errors=None)
```

返回一个类似文件的对象作为临时存储区域。创建该文件使用了与 `mkstemp()` 相同的安全规则。该文件在被关闭后会被立即销毁（包括垃圾回收机制关闭该对象时）。在 Unix 下，该文件在目录中的条目根本不创建，或者创建文件后立即就被删除了，但其他平台不支持此功能。您的代码不应依赖使用此功能创建的临时文件名称，因为它在文件系统中的名称可能是可见的，也可能是不可见的。

生成的对象可以用作上下文管理器。完成上下文或销毁临时文件对象后，临时文件将从文件系统中删除。

*mode* 参数默认值为 `'w+b'`，所以创建的文件不用关闭，就可以读取或写入。因为使用的是二进制模式，所以无论存的是什么数据，它在所有平台上都表现一致。

参数 *buffering*、*encoding*、*errors* 和 *newline* 的含义与 `open()` 中的相同；参数 *dir*、*prefix* 和 *suffix* 的含义和默认值与 `mkstemp()` 中的相同。

## TemporaryDirectory

```python
tempfile.TemporaryDirectory(suffix=None, prefix=None, dir=None)
```

安全地创建一个临时目录，且使用与 `mkdtemp()` 相同的规则。此函数返回的对象可用作上下文管理器。完成上下文或销毁临时目录对象后，新创建的临时目录及其所有内容将从文件系统中删除。

可以从返回对象的 `name` 属性中找到临时目录的名称。当返回的对象用作上下文管理器时，这个 `name` 会作为 `with` 语句中 `as` 子句的目标（如果有 `as` 的话）。

可以调用 `cleanup()` 方法来手动清理目录。

## mkstemp()

```python
tempfile.mkstemp(suffix=None, prefix=None, dir=None, text=False)
```

以最安全的方式创建一个临时文件。假设所在平台正确实现了 `os.open()` 的 `os.O_EXCL` 标志，则创建文件时不会有竞争的情况。该文件只能由创建者读写，如果所在平台用权限位来标记文件是否可执行，则没有人有执行权。文件描述符不会过继给子进程。

与 `TemporaryFile()` 不同，`mkstemp()` 的用户用完临时文件后需要自行将其删除。

如果 *suffix* 不是 `None` 则文件名将以该后缀结尾，是 `None` 则没有后缀。`mkstemp()` 不会在文件名和后缀之间加点，如果需要加一个点号，请将其放在 *suffix* 的开头。

如果 *prefix* 不是 `None`，则文件名将以该前缀开头，是 `None` 则使用默认前缀。默认前缀是 `gettempprefix()` 或 `gettempprefixb()` 函数的返回值（自动调用合适的函数）。

如果 *dir* 不是 `None`，则在指定的目录创建文件，是 `None` 则使用默认目录。默认目录是从一个列表中选择出来的，这个列表在不同的平台不一样，但是用户可以设置 *TMPDIR*、*TEMP* 或 *TMP* 环境变量来设置目录的位置。因此，不能保证生成的临时文件路径很规范，比如，通过 `os.popen()` 将路径传递给外部命令时仍需要加引号。

如果 *suffix*、*prefix* 和 *dir* 中的任何一个不是 `None`，就要保证它们是同一数据类型。如果它们是 bytes，则返回的名称的类型就是 bytes 而不是 str。如果确实要用默认参数，但又想要返回值是 bytes 类型，请传入 `suffix=b''`。

如果指定了 *text* 且为真值，文件会以文本模式打开，否则文件会（默认）以二进制模式打开。

`mkstemp()` 返回一个元组，元组中第一个元素是句柄，它是一个系统级句柄，指向一个打开的文件（等同于 `os.open()` 的返回值），第二元素是该文件的绝对路径。

## mkdtemp()

```python
tempfile.mkdtemp(suffix=None, prefix=None, dir=None)
```

以最安全的方式创建一个临时目录，创建该目录时不会有竞争的情况。该目录只能由创建者读取、写入和搜索。

`mkdtemp()` 用户用完临时目录后需要自行将其删除。

*prefix*、*suffix* 和 *dir* 的含义与它们在 `mkstemp()` 中的相同。

`mkdtemp()` 返回新目录的绝对路径。

## gettempdir()

返回放置临时文件的目录的名称。

Python 搜索标准目录列表，以找到调用者可以在其中创建文件的目录。这个列表是：

1.`TMPDIR`，`TEMP` 或 `TMP` 环境变量指向的目录。
2. 与平台相关的位置：
   * 在 Windows 上，依次为 `C:\TEMP`、`C:\TMP`、`\TEMP` 和 `\TMP`
   * 在所有其他平台上，依次为 `/tmp`、`/var/tmp` 和 `/usr/tmp`
3. 不得已时，使用当前工作目录。
