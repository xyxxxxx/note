# 条件语句

```python
age = 3
if age >= 18:	#if语句执行下述缩进的语句
    print('adult')
elif age >= 6:
    print('teenager')
else:
    print('kid')
```





# 循环语句

## for循环

```python
sum = 0
for x in range(101):	#range(101)生成0-100的整数序列
    sum = sum + x
print(sum)

#多变量遍历
for x, y in [(1, 1), (2, 4), (3, 9)]:
     print(x, y)
```



## while循环

```python
sum = 0
n = 100
while n > 0:
    sum = sum + n
    n = n - 1
print(sum)
```

```python
break	#跳出循环
continue#跳过当次循环
```


