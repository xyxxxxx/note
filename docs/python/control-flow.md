# 控制流

> 参考[复合语句](https://docs.python.org/zh-cn/3/reference/compound_stmts.html)

## 条件语句

### if 语句

`if` 语句用于有条件的执行：

```python
>>> age = 3
>>> if age >= 18:           # if 语句
...     print('adult')      # if 子句体
... elif age >= 12:         # elif 子句
...     print('teenager')   # elif 子句体
... else:                   # else 子句
...     print('kid')        # else 子句体
...
kid
```

`if` 语句包含任意个 `elif` 子句，以及可选的 `else` 子句。`if ... elif ... elif` 序列可以用作其它语言中 `switch` 或 `case` 语句的替代品。

`if` 语句的条件可以是任意对象（参见[逻辑值检测](./data-type-and-operation.md#逻辑值检测)），或者任意表达式，例如：

```python
>>> score = 60
>>> if score:         # 逻辑值检测
...   print(score)
... 
60
>>> if score >= 60:   # 比较运算
...   print('pass')
... 
pass
>>> if score >= 60 and score < 70:   # 布尔运算
...   print('D')
... 
D
>>> if score := 90:   # 赋值表达式 
...   print(score)
...
90
```

### 条件表达式

条件表达式（也称为三元运算符）`x if C else y` 首先对条件 *C* 求值，如果 *C* 为真，*x* 将被求值并返回其值；否则将对 *y* 求值并返回其值。

条件表达式在所有 Python 运算中具有最低的优先级。

```python
>>> age = 3
>>> print('adult' if age >= 18 else 'kid')
kid
```

## 循环语句

### while 语句

`while` 语句用于在条件保持为真的情况下重复地执行：

```python
>>> sum = 0
>>> n = 100
>>> while n > 0:           # while 语句
...   sum += n             # 循环体
...   n -= 1
...
>>> print(sum)
5050
```

与 `if` 语句相同，`while` 语句的条件也可以是任意对象或任意表达式。

### for 语句

`for` 语句用于对序列（例如字符串、元组或列表）或其他可迭代对象中的元素进行迭代：

```python
             # 可迭代对象
>>> for i in [1, 2, 3]:    # for 语句
...   print(i)             # 循环体
...
1
2
3
```

`for` 语句的底层实现相当于：

```python
_iter = obj.__iter__()        # Get iterator from iterable
while True:
    try:
        x = _iter.__next__()  # Get next item
    except StopIteration:     # No more items
        break
    # statements ...
```

解释器会调用可迭代对象的 `__iter__()` 方法创建一个迭代器，然后将迭代器返回的每一项（按标准赋值规则依次）赋值给目标（列表），并执行循环体。当所有项被耗尽时，循环将终止。

目标（列表）的变量在循环结束时不会被删除（参见[作用域](./oop.md#作用域)）；但如果可迭代对象为空，则它（们）根本不会被循环赋值。

迭代过程中会使用一个内部计数器来跟踪下一个要使用的项，每次迭代都会使计数器递增，而当计数器的值达到序列长度时循环就会终止。这意味着如果循环体从序列中删除了当前（或之前）的一项，则下一项会被跳过（因为其标号将变成已被处理的当前项的标号）；如果循环体在序列当前项的前面插入一项，则当前项会在循环的下一轮中再次被处理。这会导致棘手的程序错误，避免此问题的方法是迭代一个对象，修改另一个对象，参见[列表迭代](./container-type#列表)，[字典迭代](./container-type#字典)。

### break, continue 语句和 else 子句

`break` 语句用于跳出最近的 `for` 或 `while` 循环：

```python
>>> sum = 0
>>> for i in range(101):
...   if i == 10:
...     break
...   sum += i
...
>>> sum
45  
```

`continue` 语句用于结束当次迭代，继续执行循环的下一次迭代：

```python
>>> sum = 0
>>> for i in range(101):
...   if i == 10:
...     continue
...   sum += i
...
>>> sum
5040
```

`else` 子句在 `for` 循环耗尽了可迭代对象的元素或 `while` 循环的条件变为 `False` 时被执行，但循环被 `break` 语句终止时不会执行：

```python
>>> sum = 0
>>> for i in range(101):
...   sum += i
... else:              # else 子句被执行
...   sum += 1
...
>>> sum
5051
```

```python
>>> sum = 0
>>> for i in range(101):
...   sum += i
...   if i == 100:
...     break          # break 语句
... else:              # else 子句不被执行
...   sum += 1
... 
>>> sum
5050
```

## `pass` 语句

`pass` 语句不执行任何操作。当语法上需要一个语句，但程序不实际执行任何动作时，可以使用该语句。

`pass` 语句可以用作函数、类或条件子句的占位符，让开发者聚焦于更抽象的层次。

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
