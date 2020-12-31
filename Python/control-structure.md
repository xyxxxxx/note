# 条件语句

## if

```python
age = 3
if age >= 18:
    print('adult')
elif age >= 6:
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

`range()`函数尽管返回的对象表现得像一个列表，但实际上是一个迭代器。



## for

```python
sum = 0
for x in range(101):	# range(101)生成0-100的整数序列
    sum = sum + x
print(sum) 
```

```python
# 多变量遍历
for x, y in [(1, 1), (2, 4), (3, 9)]:
     print(x, y
```

```python
# 遍历数组
a = ['Mary', 'had', 'a', 'little', 'lamb']
for i in range(len(a)):
     print(i, a[i])
```



## while

```python
sum = 0
n = 100
while n > 0:
    sum = sum + n
    n = n - 1
print(sum)
```



## break，continue，else

```python
break	  # 跳出循环
continue  # 跳过当次循环

else:     # else子句在循环耗尽了可迭代对象或循环条件变为False时被执行,但在循环被break语句终止时不会执行
```
