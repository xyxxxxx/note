# functools——高阶函数和可调用对象上的操作

## partial()

返回一个新的 `partial` 对象，当被调用时其行为类似于 *func* 附带位置参数 *args* 和关键字参数 *keywords* 被调用。如果为调用提供了更多的参数，它们会被附加到 *args*。如果提供了额外的关键字参数，它们会扩展并重载 *keywords*。大致等价于：

```python
def partial(func, /, *args, **keywords):
    def newfunc(*fargs, **fkeywords):
        newkeywords = {**keywords, **fkeywords}
        return func(*args, *fargs, **newkeywords)
    newfunc.func = func
    newfunc.args = args
    newfunc.keywords = keywords
    return newfunc
```

常用于冻结一部分函数参数并应用：

```python
>>> import functools
>>> int2 = functools.partial(int, base=2)  # 附带关键字参数base=2调用int
>>> int2('1000000')                        # 相当于int('1000000', base=2)
64
>>> int2('1000000', base=10)               # 重载了base=2
1000000
```

使用 lambda 函数 `lambda x: int(x, base=2)` 也能起到相同的作用。

## reduce()

将有两个参数的函数从左到右依次应用到可迭代对象的所有元素上，返回一个最终值。

```python
>>> from functools import reduce
>>> def fn(x, y):
...   return x * 10 + y
...
>>> reduce(fn, [1, 3, 5, 7, 9])   # reduce()将多元函数依次作用在序列上
13579
```
