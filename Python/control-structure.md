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



## 三元表达式

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



## 其它问题

循环体中定义的变量在循环结束后不会被释放，因此需要手动删除局部变量，如下例：

```python
>>> k = 0
>>> for i in range(5):
...   j = i + 1
...   k += j
...
>>> print(i)
4
>>> print(j)
5
>>> print(k)
15
```

```python
>>> k = 0
>>> for i in range(5):
...   j = i + 1
...   k += j
...
>>> del i
>>> del j
>>> print(i)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'i' is not defined
>>> print(k)
15
```

