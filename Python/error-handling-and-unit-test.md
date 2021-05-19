[toc]



# 错误和异常

Python 中的错误（error）分为两类：句法错误（syntax error）和异常（exception）。



## 句法错误

句法错误又称为解析错误，表示 Python 代码中出现了语法错误。

```python
>>> if True print('Hello')
  File "<stdin>", line 1         # 文件: <stdin>, 行: 1
    if True print('Hello')       # 该行内容
            ^
SyntaxError: invalid syntax
```

解析器会复现出现句法错误的代码行，并用小箭头 `^` 指向行内检测到的第一个错误。这里在 `print()` 函数检测到错误，因为它前面缺少一个冒号 `:` 。错误信息还会输出文件名与行号，在使用脚本文件时，就可以知道去哪里查错。

```python
if True print('Hello')
```

```
$ python test.py
  File "test.py", line 1         # 文件: test.py, 行: 1 
    if True print('Hello')       # 该行内容
            ^
SyntaxError: invalid syntax
```



## 异常

即使表达式的语法是正确的，执行时仍可能触发错误。<u>执行时检测到的错误称为异常</u>，异常不一定导致严重的后果。大多数异常不会被程序处理，而是显示下列错误信息：

```python
>>> 1/0
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ZeroDivisionError: division by zero
>>> 4 + spam*3
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'spam' is not defined
>>> '2' + 2
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: Can't convert 'int' object to str implicitly
```

错误信息的最后一行说明程序遇到了什么类型的错误。异常有不同的类型，而类型名称会作为错误信息的一部分中打印出来：上述示例中的异常类型依次是：`ZeroDivisionError`， `NameError` 和 `TypeError`。作为异常类型打印的字符串是发生的内置异常的名称，对于所有内置异常都是如此，用户定义的异常也应遵循这种规范。

这一行的剩下的部分根据异常类型及其原因提供详细信息。

错误信息的开头部分以堆栈回溯的形式显示发生异常的上下文。 通常它会包含列出源代码行的堆栈回溯；但是，它将不会显示从标准输入读取的行。



### 内置异常

内置异常的类层级结构如下：

```
BaseException
 +-- SystemExit
 +-- KeyboardInterrupt
 +-- GeneratorExit
 +-- Exception
      +-- StopIteration
      +-- StopAsyncIteration
      +-- ArithmeticError
      |    +-- FloatingPointError
      |    +-- OverflowError
      |    +-- ZeroDivisionError
      +-- AssertionError
      +-- AttributeError
      +-- BufferError
      +-- EOFError
      +-- ImportError
      |    +-- ModuleNotFoundError
      +-- LookupError
      |    +-- IndexError
      |    +-- KeyError
      +-- MemoryError
      +-- NameError
      |    +-- UnboundLocalError
      +-- OSError
      |    +-- BlockingIOError
      |    +-- ChildProcessError
      |    +-- ConnectionError
      |    |    +-- BrokenPipeError
      |    |    +-- ConnectionAbortedError
      |    |    +-- ConnectionRefusedError
      |    |    +-- ConnectionResetError
      |    +-- FileExistsError
      |    +-- FileNotFoundError
      |    +-- InterruptedError
      |    +-- IsADirectoryError
      |    +-- NotADirectoryError
      |    +-- PermissionError
      |    +-- ProcessLookupError
      |    +-- TimeoutError
      +-- ReferenceError
      +-- RuntimeError
      |    +-- NotImplementedError
      |    +-- RecursionError
      +-- SyntaxError
      |    +-- IndentationError
      |         +-- TabError
      +-- SystemError
      +-- TypeError
      +-- ValueError
      |    +-- UnicodeError
      |         +-- UnicodeDecodeError
      |         +-- UnicodeEncodeError
      |         +-- UnicodeTranslateError
      +-- Warning
           +-- DeprecationWarning
           +-- PendingDeprecationWarning
           +-- RuntimeWarning
           +-- SyntaxWarning
           +-- UserWarning
           +-- FutureWarning
           +-- ImportWarning
           +-- UnicodeWarning
           +-- BytesWarning
           +-- ResourceWarning
```



下表列出了各内置异常的含义：

| 异常名称                    | 描述                                                         |
| :-------------------------- | :----------------------------------------------------------- |
| `BaseException`             | 所有内置异常的基类                                           |
| `SystemExit`                | 解释器请求退出                                               |
| `KeyboardInterrupt`         | 用户中断执行(通常是输入control C)                            |
| `GeneratorExit`             | 生成器被关闭时引发                                           |
| `Exception`                 | 所有内置的非系统退出类异常的基类，所有用户自定义异常也应当派生自此类 |
| `StopIteration`             | 迭代器没有更多的值                                           |
| `StopAsyncIteration`        |                                                              |
| `ArithmeticError`           | 所有数值计算错误的基类                                       |
| `FloatingPointError`        | 浮点计算错误                                                 |
| `OverflowError`             | 数值运算超出最大限制（溢出）                                 |
| `ZeroDivisionError`         | 除零（对于所有数据类型）                                     |
| `AssertionError`            | 断言语句失败                                                 |
| `AttributeError`            | 属性引用或赋值失败                                           |
| `BufferError`               |                                                              |
| `EOFError`                  | 没有内建输入，到达EOF标记                                    |
| `ImportError`               | 导入模块/对象失败                                            |
| `ModuleNotFoundError`       |                                                              |
| `LookupError`               | 无效数据查询的基类                                           |
| `IndexError`                | 索引超出序列范围                                             |
| `KeyError`                  | 字典中找不到指定的键                                         |
| `MemoryError`               | 内存溢出错误（对于Python 解释器不是致命的）                  |
| `NameError`                 | 找不到局部或全局名称                                         |
| `UnboundLocalError`         | 访问未初始化的本地变量                                       |
| `OSError`                   | 操作系统错误的基类                                           |
| `BlockingIOError`           |                                                              |
| `ChildProcessError`         |                                                              |
| `ConnectionError`           | 连接相关问题的基类                                           |
| `BrokenPipeError`           | 试图写入另一端已被关闭的管道，或是试图写入已关闭写入的套接字 |
| `ConnectionAbortedError`    | 连接被对方中止                                               |
| `ConnectionRefusedError`    | 连接被对方拒绝                                               |
| `ConnectionResetError`      | 连接被对方重置                                               |
| `FileExistsError`           | 创建一个已存在的文件或目录                                   |
| `FileNotFoundError`         | 指定的文件或目录不存在                                       |
| `InterruptedError`          | 系统调用被输入信号中断                                       |
| `IsADirectoryError`         | 对一个目录执行文件操作                                       |
| `NotADirectoryError`        | 对一个非目录对象执行目录操作                                 |
| `PermissionError`           | 没有操作权限                                                 |
| `ProcessLookupError`        | 指定的进程不存在                                             |
| `TimeoutError`              | 系统函数发生系统级超时                                       |
| `ReferenceError`            | 弱引用(Weak reference)试图访问已经垃圾回收了的对象           |
| `RuntimeError`              | 一般的运行时错误，通常不属于其它类型的异常会归于此类         |
| `NotImplementedError`       | 尚未实现的方法                                               |
| `RecursionError`            | 递归超过最大深度                                             |
| `SyntaxError`               | 解析器遇到语法错误                                           |
| `IndentationError`          | 与不正确的缩进相关的语法错误                                 |
| `TabError`                  | 缩进包含对制表符和空格符的不一致的使用                       |
| `SystemError`               | 一般的解释器系统错误                                         |
| `TypeError`                 | 操作或函数被应用于类型不适当的对象                           |
| `ValueError`                | 操作或函数被应用于类型正确但取值不适当的对象                 |
| `UnicodeError`              | Unicode相关的错误                                            |
| `UnicodeDecodeError`        | Unicode解码错误                                              |
| `UnicodeEncodeError`        | Unicode编码错误                                              |
| `UnicodeTranslateError`     | Unicode转换错误                                              |
| `Warning`                   | 警告的基类                                                   |
| `DeprecationWarning`        | 关于特性被废弃的警告                                         |
| `PendingDeprecationWarning` | 关于特性将被废弃的警告                                       |
| `RuntimeWarning`            | 可疑的运行时行为的警告                                       |
| `SyntaxWarning`             | 可疑的语法的警告                                             |
| `UserWarning`               | 用户代码生成的警告                                           |
| `FutureWarning`             | 关于构造将来语义会有改变的警告                               |
| `ImportWarning`             |                                                              |
| `UnicodeWarning`            |                                                              |
| `BytesWarning`              |                                                              |
| `OverflowWarning`           | 旧的关于自动提升为长整型(long)的警告                         |





## try-except-finally 结构



```python
# try-except完整语句
try:
    print('try...')
    r = 10 / 0					# error,跳至except
    print('result:', r)
except ValueError as e:
    print('ValueError:', e)    
except ZeroDivisionError as e:	# 捕获该错误类型
    print('ZeroDivisionError:', e)
except (IOError,LookupError,RuntimeError) as e:  # 捕获多个错误类型
    pass    
else:							# 无错误
    print('no error!')
finally:
    print('finally...')
print('END')
```



+ 在 `try-except-finally` 语句中使用 `return` 语句时注意，由于 `finally` 语句下的代码必定执行，即使 `try` 或 `except` 语句下有 `return` 语句，在返回之前会执行 `finally` 语句下的代码：

  ```python
  def demo():
      try:
          raise RuntimeError('')
      except RuntimeError:        # 捕捉异常
          return 1                # 返回之前执行finally语句下的代码
      else:
          return 2
      finally:
          return 3                # 返回
  
  print(demo())
  ```

  ```
  3
  ```

  

  







### 异常上抛

下列代码演示了异常层层上抛的过程。

```python
def grok():
    print('an error raised by grok()')
    raise RuntimeError('Whoa!')  # 引发异常并上抛


def spam():
    grok()                       # 继续上抛


def bar():
    try:
        spam()
    except RuntimeError as e:    # 处理错误
        print('error handled by bar()')


def foo():
    try:
        bar()
    except RuntimeError as e:    # 错误未到达此处
        print('error handled by foo()')

foo()
```

```
an error raised by grok()
error handled by bar()
```





## 调用栈

```python
# err.py:
def foo(s):
    return 10 / int(s)
def bar(s):
    return foo(s) * 2
def main():
    bar('0')
main()

# 错误信息
Traceback (most recent call last):
  File "test.py", line 8, in <module>
    main()
  File "test.py", line 7, in main
    bar('0')
  File "test.py", line 5, in bar
    return foo(s) * 2
  File "test.py", line 3, in foo
    return 10 / int(s)
ZeroDivisionError: division by zero
```



## 记录错误



## 抛出错误

```python
if name not in authorized:
	raise RuntimeError(f'{name} not authorized')

# 程序退出    
raise SystemExit(exitcode)
raise SystemExit('Informative message')
import sys
sys.exit(exitcode)
```





# 调试

## 断言assert

```python
def foo(s):
    n = int(s)
    assert n != 0, 'n is zero!'	# assert n!=0,else AssertionError
    return 10 / n
# 可以用-O参数来关闭assert
```



## 日志

> 参考：
>
> [日志 HOWTO](https://docs.python.org/zh-cn/3.6/howto/logging.html)
>
> [Python 中 logging 模块的基本用法](https://cuiqingcai.com/6080.html)

日志是对软件执行时所发生事件的一种追踪方式。软件开发人员对他们的代码添加日志调用，借此来指示某事件的发生。一个事件通过一些包含变量数据的描述信息来描述。开发者还会区分事件的重要性，也称为事件级别或严重性，各级别如下：

| 级别       | 何时使用                                                     |
| :--------- | :----------------------------------------------------------- |
| `DEBUG`    | 细节信息，仅当诊断问题时适用。                               |
| `INFO`     | 确认程序按预期运行                                           |
| `WARNING`  | 表明有已经或即将发生的意外（例如：磁盘空间不足）。程序仍按预期进行 |
| `ERROR`    | 由于严重的问题，程序的某些功能已经不能正常执行               |
| `CRITICAL` | 严重的错误，表明程序已不能继续执行                           |



### 示例

```python
import logging

logging.warning('Watch out!')  # 向控制台打印信息'Watch out!'
logging.info('I told you so')  # 不打印,因为默认的追踪等级为WARNING

# 命令行打印
# WARNING:root:Watch out!
```

```python
import logging

logging.basicConfig(filename='example.log', level=logging.DEBUG)  # 设置输出文件
logging.debug('This message should go to the log file')
logging.info('So should this')
logging.warning('And this, too')

# 日志文件写入
# DEBUG:root:This message should go to the log file
# INFO:root:So should this
# WARNING:root:And this, too
```

```python
import logging

logging.warning('%s before you %s', 'Look', 'leap!')  # 打印变量值

# 命令行打印
# WARNING:root:Look before you leap!
```

```python
import logging

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)  # 设定打印格式和追踪级别
logging.debug('This message should appear on the console')
logging.info('So should this')
logging.warning('And this, too')

# 命令行打印
# 没有打印DEBUG级别的日志
# INFO:So should this
# WARNING:And this, too
```

```python
import logging

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %I:%M:%S')  # 打印日期时间
logging.warning('is when this event was logged.')

# 命令行打印
# 2021/02/18 02:30:46 is when this event was logged.
```



### 进阶日志教程

日志库采用模块化方法，并提供几类组件：记录器、处理程序、过滤器和格式化程序。

- 记录器暴露了应用程序代码直接使用的接口。
- 处理程序将日志记录（由记录器创建）发送到适当的目标。
- 过滤器提供了更精细的附加功能，用于确定要输出的日志记录。
- 格式化程序指定最终输出中日志记录的样式。



```python
import logging

# Logger对象为应用程序代码提供了几种方法,以便应用程序可以在运行时记录消息
logger = logging.getLogger('simple_example')  # 指定Logger对象名称,多次调用相同名称时将返回同一对象
logger.setLevel(logging.DEBUG)                # 设置logger级别,默认为WARNING

# Handler对象负责将适当的日志消息(基于日志消息的严重性)分派给其指定目标
sh = logging.StreamHandler()                  # 传入一个流,默认为sys.stderr
sh.setLevel(logging.INFO)                     # 设置handler级别,默认为WARNING

# Formatter对象配置日志消息的最终顺序、结构和内容
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# handler设置formatter
sh.setFormatter(formatter)
# logger添加handler
logger.addHandler(sh)

fh = logging.FileHandler('example.log')       # 传入一个文件名
fh.setLevel(logging.DEBUG)                    # 设置handler级别,默认为WARNING

logger.addHandler(fh)

# 应用程序调用
logger.debug('debug message')
logger.info('info message')
logger.warning('warn message')
logger.error('error message')
logger.critical('critical message')

# 命令行打印
# 2021-02-18 16:26:56,280 - simple_example - INFO - info message
# 2021-02-18 16:26:56,280 - simple_example - WARNING - warn message
# 2021-02-18 16:26:56,280 - simple_example - ERROR - error message
# 2021-02-18 16:26:56,280 - simple_example - CRITICAL - critical message

# 日志文件写入
# debug message
# info message
# warn message
# error message
# critical message
```



### 格式化输出信息

```python
import logging

service_name = "Booking"
logging.error('%s service is down!', service_name)               # 使用logger的格式化，推荐
logging.error('%s service is %s!', service_name, 'down')         # 多参数格式化
logging.error('{} service is {}!'.format(service_name, 'down'))  # 使用format函数，推荐

# ERROR:root:Booking service is down!
# ERROR:root:Booking service is down!
# ERROR:root:Booking service is down!
```



# 测试

单元测试