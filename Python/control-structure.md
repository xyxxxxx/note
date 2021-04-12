[toc]

# 条件语句

## if

```python
>>> age = 3
>>> if age >= 18:
...     print('adult')
... elif age >= 12:
...     print('teenager')
... else:
...     print('kid')
...
kid
```



## 条件表达式

```python
>>> age = 3
>>> print('adult' if age >= 18 else 'kid')
kid
```





# 循环语句

## for

```python
>>> sum = 0
>>> for i in range(101):	# range(101)生成0-100的整数序列
...   sum += i
...
>>> print(sum)
5050
```

```python
# 多变量遍历
>>> for x, y in [(1, 1), (2, 4), (3, 9)]:
...   print(x, y)
...
1 1
2 4
3 9
```

```python
# 遍历数组
>>> a = ['Mary', 'had', 'a', 'little', 'lamb']
>>> for i, v in enumerate(a):
...   print(i, v)
...
0 Mary
1 had
2 a
3 little
4 lamb
```



## while

```python
>>> sum = 0
>>> n = 100
>>> while n > 0:
...   sum += n
...   n -= 1
...
>>> print(sum)
5050
```



## break, continue, else

```python
# break跳出循环
>>> sum = 0
>>> for i in range(101):
...   if i == 10:
...     break
...   sum += i
...
>>> print(sum)
45  
```

```python
# continue跳过当次循环
>>> sum = 0
>>> for i in range(101):
...   if i == 10:
...     continue
...   sum += i
...
>>> print(sum)
5040
```

```python
# else子句在循环耗尽了可迭代对象或循环条件变为False时被执行,但在循环被break语句终止时不会执行
>>> sum = 0
>>> for i in range(101):
...   sum += i
... else:
...   sum += 1
...
>>> print(sum)
5051
```





# `pass`语句

`pass` 语句不执行任何操作。当语法上需要一个语句，但程序不实际执行任何动作时，可以使用该语句。

`pass` 语句可以用作函数、类或条件子句的占位符，让开发者聚焦更抽象的层次。

```python
while True:
  pass             # 无限循环
```

```python
class EmptyClass:
  pass             # 空类
```

```python
def f():
  pass             # 空函数
```

