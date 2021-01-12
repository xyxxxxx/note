# 错误，调试和测试

## 错误处理

### try-except

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



### 调用栈

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



### 记录错误



### 抛出错误

```python
if name not in authorized:
	raise RuntimeError(f'{name} not authorized')

# 程序退出    
raise SystemExit(exitcode)
raise SystemExit('Informative message')
import sys
sys.exit(exitcode)
```





## 调试

### assert

```python
def foo(s):
    n = int(s)
    assert n != 0, 'n is zero!'	# assert n!=0,else AssertionError
    return 10 / n
# 可以用-O参数来关闭assert
```



### logging

设置logger

```python

```

格式化输出信息

```python
service_name = "Booking"
logger.error('%s service is down!', service_name)        # 使用logger的格式化，推荐
logger.error('%s service is %s!', service_name, 'down')  # 多参数格式化
logger.error('{} service is {}'.format(service_name, 'down')) # 使用format函数，推荐
# 2016-10-08 21:59:19,493 ERROR   : Booking service is down!
```



## 测试

单元测试