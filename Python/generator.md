## 生成器generator

生成器（generator）就是使用`yield`语句的函数。生成器和一般函数的区别在于执行流程不同：一般函数顺序执行，遇到`return`语句或者最后一条语句返回；生成器在每次调用`next()`的时候执行，遇到`yield`语句返回，再次执行时从上次返回的`yield`语句处继续执行。

下面的例子展示了一个生成器，其返回斐波那契数列的前若干项：

```python
def fib(max):	   # generator型函数
    n, a, b = 0, 0, 1
    while n < max:
        yield b
        a, b = b, a + b
        n = n + 1
    return 'done'

ge = fib(10)     # 每次调用generator型函数返回一个generator
print(repr(ge))  # <generator object fib at 0x7fcd124c14c0>

for n in ge:
  print(n, end=' ')    # 1 1 2 3 5 8 13 21 34 55
                       # 迭代完成后generator即失去作用,不可重用
```

```python
>>> a = [1, 2, 3, 4]
>>> b = (2 * x for x in a)       # 返回一个generator,一般形式为
>>> b                            # (<expression> for i in s if <conditional>)
<generator object at 0x58760>
>>> for i in b:
...   print(i, end=' ')
...
2 4 6 8
```



## 迭代器

可以作用于for循环的对象为可迭代对象Iterable，包括`list,tuple,dict,set,str,...`

可以被`next()`调用并不断返回下一个值的对象称为迭代器Iterator，包括生成器。

所有迭代过程的底层实现如下：

```python
_iter = obj.__iter__()        # Get iterator from iterable
while True:
    try:
        x = _iter.__next__()  # Get next item
    except StopIteration:     # No more items
        break
    # statements ...
    
# eg
>>> x = [1,2,3]
>>> it = x.__iter__()
>>> it
<listiterator object at 0x590b0>
>>> it.__next__()             # 等同于 next(it)
1
>>> it.__next__()
2
>>> it.__next__()
3
>>> it.__next__()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
StopIteration
```

使用[`itertools`库中的函数](https://docs.python.org/zh-cn/3/library/itertools.html)可以快速创建迭代器。



## 流水线

使用生成器可以构造数据处理的流水线：

*producer* → *processing* → *consumer*

```python
def producer():
    pass
    yield item          # yields the item that is received by the `processing`

def processing(s):
    for item in s:      # Comes from the `producer`
        pass
        yield newitem   # yields a new item

def consumer(s):
    for item in s:      # Comes from the `processing`
        pass
        
a = producer()
b = processing(a)
c = consumer(b)        
```

