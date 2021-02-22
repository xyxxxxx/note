[toc]



# 错误处理

## try-except

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

```python
# 异常抛出
def grok():
    pass
    raise RuntimeError('Whoa!')   # 抛出错误

def spam():
    grok()                        # 继续上抛

def bar():
    try:
       spam()
    except RuntimeError as e:     # 处理错误
        pass

def foo():
    try:
         bar()
    except RuntimeError as e:     # 错误未到达此处
        pass

foo()
```

> python内置异常https://docs.python.org/zh-cn/3/library/exceptions.html



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

## assert

```python
def foo(s):
    n = int(s)
    assert n != 0, 'n is zero!'	# assert n!=0,else AssertionError
    return 10 / n
# 可以用-O参数来关闭assert
```



## logging

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