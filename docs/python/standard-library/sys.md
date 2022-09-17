# sys——系统相关的参数和函数

`sys` 模块提供了一些变量和函数。这些变量可能被解释器使用，也可能由解释器提供。这些函数会影响解释器。本模块总是可用的。

## argv

被传递给 Python 脚本的命令行参数列表，`argv[0]` 为脚本的名称（是否是完整的路径名取决于操作系统）。

## byteorder

本地字节顺序的指示符。在大端序操作系统上值为 `'big'`，在小端序操作系统上为 `'little'`。

## executable

当前 Python 解释器的可执行二进制文件的绝对路径。

```python
>>> sys.executable
'/Users/xyx/.pyenv/versions/3.8.7/bin/python'
```

## exit()

```python
sys.exit([arg])
```

从 Python 中退出，实现方式是引发一个 `SystemExit` 异常。

可选参数 *arg* 可以是表示退出状态的整数（默认为 0），也可以是其他类型的对象。如果它是整数，则 shell 等将 0 视为“成功终止”，非零值视为“异常终止”。大多数系统要求该值的范围是 0-127，否则会产生未定义的结果。某些系统为特定的退出代码约定了特定的含义，但通常尚不完善；Unix 程序通常用 2 表示命令行语法错误，用 1 表示所有其他类型的错误。如果传入其他类型的对象，None 相当于传入 0，任何其他对象会被打印到 `stderr`，且退出代码为 1。特别地，`sys.exit("some error message")` 可以在发生错误时快速退出程序。

## getallocatedblocks()

返回解释器当前分配的内存块数，无论它们的大小如何。本函数主要用于跟踪和调试内存泄漏。因为解释器有内部缓存，所以每一次调用的结果会有不同。可能需要调用 `_clear_type_cache()` 和 `gc.collect()` 来获得更加可预测的结果。

如果当前的 Python 构建或实现无法合理地计算此信息，允许此函数返回 0。

## hexversion

## getsizeof()

返回对象的大小（以字节为单位）。该对象可以是任何类型。所有内建对象返回的结果都是正确的，但对于第三方扩展不一定正确，因为这与具体实现有关。

只计算直接分配给对象的内存消耗，不计算它所引用的对象的内存消耗。

对象不提供计算大小的方法时，如果有给出 *default* 则返回它，否则引发一个 `TypeError`。

如果对象由垃圾回收器管理，则 `getsizeof()` 将调用对象的 `__sizeof__` 方法，并在上层添加额外的垃圾回收器。

可以参考 [recursive sizeof recipe](https://code.activestate.com/recipes/577504) 中的示例，关于递归调用 `getsizeof()` 来得到容器及其所有内容的大小。

## implementation

一个包含当前运行的 Python 解释器的实现信息的对象。所有 Python 实现中都必须存在下列属性。

*name* 是实现的标识符，如 `'cpython'`。实际的字符串由 Python 实现定义，但保证是小写字母。

*version* 是一个命名元组，格式与 `sys.version_info` 相同，它表示 Python 实现的版本。`sys.version_info` 本身表示的是当前解释器遵循的 Python 语言的版本，两者具有不同的含义。 例如，对于 PyPy 1.8，`sys.implementation.version` 可能是 `sys.version_info(1, 8, 0, 'final', 0)`，而 `sys.version_info` 则是 `sys.version_info(2, 7, 2, 'final', 0)`。对于 CPython 而言两个值是相同的，因为它是参考实现。

*hexversion* 是十六进制格式的实现版本，类似于 `sys.hexversion`。

*cache_tag* 是导入机制使用的标记，用于已缓存模块的文件名。按照惯例，它将由实现的名称和版本组成，如 `'cpython-33'`。但如果合适，Python 实现可以使用其他值。如果 `cache_tag` 被置为 None，表示模块缓存已禁用。

`sys.implementation` 可能包含特定于 Python 实现的其他属性。这些非标准属性必须以下划线开头，此处不详细阐述。无论其内容如何，`sys.implementation` 在解释器运行期间或不同实现版本之间都不会更改。（但是不同 Python 语言版本之间可能会不同。）详情请参阅 [PEP 421](https://www.python.org/dev/peps/pep-0421)。

## modules

一个将模块名称映射到已加载的模块的字典。可以操作该字典来强制重新加载模块，或是实现其他技巧。但是，替换的字典不一定会按预期工作，并且从字典中删除必要的项目可能会导致 Python 崩溃。

```python
>>> from pprint import pprint
>>> import numpy
>>> pprint(sys.modules)
{'__main__': <module '__main__' (built-in)>,
 ...
 'numpy': <module 'numpy' from '/Users/xyx/.pyenv/versions/3.8.7/lib/python3.8/site-packages/numpy/__init__.py'>,
 ...
 'numpy.version': <module 'numpy.version' from '/Users/xyx/.pyenv/versions/3.8.7/lib/python3.8/site-packages/numpy/version.py'>,
 ...
 'pprint': <module 'pprint' from '/Users/xyx/.pyenv/versions/3.8.7/lib/python3.8/pprint.py'>,
 ...
 'sys': <module 'sys' (built-in)>,
 ...
```

## path

指定模块搜索路径的字符串列表。初始化自环境变量 `PYTHONPATH`，再加上一条与安装有关的默认路径。

程序启动时将初始化本列表，列表的第一项 `path[0]` 为调用 Python 解释器的脚本所在的目录。如果脚本目录不可用（比如以交互方式调用了解释器，或脚本是从标准输入中读取的），则 `path[0]` 为空字符串，Python 将优先搜索当前目录中的模块。

解释器将依次搜索各路径，因此索引靠前的路径具有更高的优先级。程序可以根据需要任意修改本列表。

```python
# 执行 Python 脚本, 位于 /Users/xyx/python/test/my_module.py
import sys
print(sys.path[0])
# /Users/xyx/python/test
```

```shell
$ python                        # 以交互方式调用解释器
>>> import sys
>>> sys.path                    # sys.path[0] 为空字符串
['', '/Users/xyx/.pyenv/versions/3.8.7/lib/python38.zip', '/Users/xyx/.pyenv/versions/3.8.7/lib/python3.8', '/Users/xyx/.pyenv/versions/3.8.7/lib/python3.8/lib-dynload', '/Users/xyx/.pyenv/versions/3.8.7/lib/python3.8/site-packages']
```

## platform

本字符串是一个平台标识符。

对于 Unix 系统（除 Linux 和 AIX 外），该字符串是 *Python 在构建时*由 `uname -s` 返回的小写操作系统名称，并附加了由 `uname -r` 返回的系统版本的第一部分，如 `'sunos5'` 或 `'freebsd8'`。除非需要检测特定版本的系统，否则建议使用以下习惯用法：

```python
if sys.platform.startswith('freebsd'):
    # FreeBSD-specific code here...
elif sys.platform.startswith('linux'):
    # Linux-specific code here...
elif sys.platform.startswith('aix'):
    # AIX-specific code here...
```

对于其他系统，值为：

| 系统           | `平台` 值  |
| -------------- | ---------- |
| AIX            | `'aix'`    |
| Linux          | `'linux'`  |
| Windows        | `'win32'`  |
| Windows/Cygwin | `'cygwin'` |
| macOS          | `'darwin'` |

## stdin, stdout, stderr

解释器用于标准输入、标准输出和标准错误的文件对象：

* `stdin` 用于所有交互式输入（包括对 `input()` 的调用）
* `stdout` 用于 `print()` 和 expression 语句的输出，以及用于 `input()` 的提示符
* 解释器自身的提示符和错误消息发往 `stderr`

这些流都是常规文本文件，与 `open()` 函数返回的对象一致。它们的参数选择如下：

* 字符编码取决于各个平台。在非 Windows 平台上使用的是语言环境（locale）编码（可参阅 `locale.getpreferredencoding()`）。

  在 Windows 上，控制台设备使用 UTF-8 编码。非字符设备（如磁盘文件和管道）使用系统语言环境编码（即 ANSI 代码页）。非控制台字符设备（即 `isatty()` 返回的是 True，如 NUL）在启动时，会把控制台输入代码页和输出代码页的值分别用于 stdin 和 stdout/stderr。如果进程原本没有附加到控制台，则默认为系统语言环境编码。

  要重写控制台的特殊行为，可以在启动 Python 前设置 `PYTHONLEGACYWINDOWSSTDIO` 环境变量。此时，控制台代码页将用于其他字符设备。

  在所有平台上，都可以通过在 Python 启动前设置 `PYTHONIOENCODING` 环境变量来重写字符编码，或通过新的 `-X utf8` 命令行选项和 `PYTHONUTF8` 环境变量来设置。但是，对 Windows 控制台来说，上述方法仅在设置了 `PYTHONLEGACYWINDOWSSTDIO` 后才有效。

* When interactive, stdout and stderr streams are line-buffered. Otherwise, they are block-buffered like regular text files. You can override this value with the -u command-line option.

!!! note "注意"
    要从标准流写入或读取二进制数据，请使用底层的二进制 `buffer` 对象。例如，要将字节写入 `stdout`，请使用 `sys.stdout.buffer.write(b'abc')`。

    但是，如果你在写一个库（并且不限制执行库代码时的上下文），那么请注意，标准流可能会被替换为类文件对象，如 `io.StringIO`，它们是不支持 `buffer` 属性的。

## \__stdin__, \__stdout__, \__stderr__

程序开始时，这些对象存有 `stdin`、`stderr` 和 `stdout` 的初始值。它们在程序结束时使用，并且在需要向实际的标准流打印内容时很有用，无论 `sys.std*` 对象是否已经重定向。

如果实际文件已经被覆盖成一个损坏的对象了，那它也可用于将实际文件还原成能正常工作的文件对象。但是，本过程的最佳方法应该是，在原来的流被替换之前就显式地保存它，并使用这一保存的对象来还原。

## thread_info

一个包含线程实现信息的命名元组。

| 属性      | 说明                                                                                                                                       |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `name`    | 线程实现的名称：<br><ul><li>`'nt'`：Windows 线程</li><li>`'pthread'`：POSIX 线程</li><li>`'solaris'`：Solaris 线程</li></ul>               |
| `lock`    | 锁实现的名称：<br><ul><li>`'semaphore'`：锁使用信号量</li><li>`'mutex+cond'`：锁使用互斥和条件变量</li><li>`None` 如果此信息未知</li></ul> |
| `version` | 线程库的名称和版本。它是一个字符串，如果此信息未知，则为 `None` 。                                                                         |

## version

一个包含 Python 解释器版本号、编译版本号、所用编译器等信息的字符串。此字符串会在交互式解释器启动时显示。请不要从中提取版本信息，而应当使用 version_info 以及 platform 模块所提供的函数。

```shell
$ python
Python 3.8.10 (default, Sep 28 2021, 16:10:42) 
[GCC 9.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> sys.version
'3.8.10 (default, Sep 28 2021, 16:10:42) \n[GCC 9.3.0]'
```

```shell
$ python
Python 3.8.7 (default, Mar  4 2021, 14:48:51) 
[Clang 12.0.0 (clang-1200.0.32.29)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> sys.version
'3.8.7 (default, Mar  4 2021, 14:48:51) \n[Clang 12.0.0 (clang-1200.0.32.29)]'
```

## version_info

一个包含版本号五部分的元组：*major*，*minor*，*micro*，*releaselevel* 和 *serial*。除 *releaselevel* 外的所有值均为整数；*releaselevel* 的值为 `'alpha'`、`'beta'`、`'candidate'` 或 `'final'`。

这些部分也可按名称访问，因此 `sys.version_info[0]` 就等价于 `sys.version_info.major`，依此类推。

```python
>>> sys.version_info
sys.version_info(major=3, minor=8, micro=7, releaselevel='final', serial=0)
>>> sys.version_info.major
3
```
