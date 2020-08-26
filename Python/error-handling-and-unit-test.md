# 错误，调试和测试

## 错误处理

### try

```python
try:
    print('try...')
    r = 10 / 0					#error,跳至except
    print('result:', r)
except ValueError as e:
    print('ValueError:', e)    
except ZeroDivisionError as e:	#捕获错误类型
    print('except:', e)
else:							#无错误
    print('no error!')    
finally:
    print('finally...')
print('END')
```

> python内置异常[file:///D:/library/python/library/exceptions.html](file:///D:/library/python/library/exceptions.html)



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

#错误信息
division by zero
堆栈跟踪:
 >  File "test.py", line 3, in foo
 >    return 10 / int(s)
 >  File "test.py", line 5, in bar
 >    return foo(s) * 2
 >  File "test.py", line 7, in main
 >    bar('0')
 >  File "test.py", line 8, in <module>
 >    main()
```

### 记录错误



### 抛出错误



## 调试

### assert

```python
def foo(s):
    n = int(s)
    assert n != 0, 'n is zero!'	#assert n!=0,else AssertionError
    return 10 / n
#可以用-O参数来关闭assert
```

### logging



## 测试

单元测试