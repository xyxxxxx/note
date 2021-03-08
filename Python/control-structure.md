[toc]

# 条件语句

## if

```python
age = 3
if age >= 18:
    print('adult')
elif age >= 12:
    print('teenager')
else:
    print('kid')
```



## 三元表达式

```python
identity = 'adult' if age >= 18 else 'kid'
```





# 循环语句

## range()

```python
range(5)         # [0,1,2,3,4]
range(5,10)      # [5,6,7,8,9]
range(0,10,2)    # [0,2,4,6,8]
range(0,10,3)    # [0,3,6,9]
range(0,-10,-3)  # [0,-3,-6,-9]
```

```python
>>> for i in range(5):
...   print(i)
...
0
1
2
3
4
>>> list(range(5))
[0, 1, 2, 3, 4]
```

`range()`函数返回的对象尽管表现得像一个列表，但实际上是一个迭代器。



## for

```python
>>> sum = 0
>>> for i in range(101):	# range(101)生成0-100的整数序列
...   sum = sum + i
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
...   sum = sum + n
...   n = n - 1
...
>>> print(sum)
5050
```



## break，continue，else

```python
break	  # 跳出循环
continue  # 跳过当次循环

else:     # else子句在循环耗尽了可迭代对象或循环条件变为False时被执行,但在循环被break语句终止时不会执行
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

