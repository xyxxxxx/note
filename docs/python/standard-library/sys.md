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

返回

## modules

返回当前已加载模块的名称到模块实例的字典。

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

指定模块搜索路径的字符串列表。解释器将依次搜索各路径，因此索引靠前的路径具有更高的优先级。

程序启动时将初始化本列表，列表的第一项 `path[0]` 为调用 Python 解释器的脚本所在的目录。如果脚本目录不可用（比如以交互方式调用了解释器，或脚本是从标准输入中读取的），则 `path[0]` 为空字符串，Python 将优先搜索当前目录中的模块。

程序可以根据需要任意修改本列表。

```shell
$ python                        # 以交互方式调用解释器
>>> import sys
>>> sys.path                    # sys.path[0]为空字符串
['', '/Users/xyx/.pyenv/versions/3.8.7/lib/python38.zip', '/Users/xyx/.pyenv/versions/3.8.7/lib/python3.8', '/Users/xyx/.pyenv/versions/3.8.7/lib/python3.8/lib-dynload', '/Users/xyx/.pyenv/versions/3.8.7/lib/python3.8/site-packages']
```

```python
# /Users/xyx/python/test/my_module.py
import sys
print(sys.path[0])
```

```shell
$ python my_module.py
/Users/xyx/python/test
```

## platform

本字符串是一个平台标识符，对于各种系统的值为：

| 系统           | `平台` 值  |
| :------------- | :--------- |
| AIX            | `'aix'`    |
| Linux          | `'linux'`  |
| Windows        | `'win32'`  |
| Windows/Cygwin | `'cygwin'` |
| macOS          | `'darwin'` |

## stdin, stdout, stderr

解释器用于标准输入、标准输出和标准错误的文件对象：

* `stdin` 用于所有交互式输入
* `stdout` 用于 `print()` 和 expression 语句的输出，以及输出 `input()` 的提示符
* 解释器自身的提示符和错误消息发往 `stderr`

## version, version_info

`version` 是一个包含 Python 解释器版本号、编译版本号、所用编译器等信息的字符串，`version_info` 是一个包含版本号五部分的元组：*major*，*minor*，*micro*，*releaselevel* 和 *serial*。

```python
>>> sys.version
'3.6.9 (default, Oct  8 2020, 12:12:24) \n[GCC 8.4.0]'
>>> sys.version
'3.8.7 (default, Mar  4 2021, 14:48:51) \n[Clang 12.0.0 (clang-1200.0.32.29)]'
>>> sys.version_info
sys.version_info(major=3, minor=6, micro=9, releaselevel='final', serial=0)
```
