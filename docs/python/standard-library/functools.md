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

wrapper     wrapped
g=decorator(f)

## update_wrapper()

```python
functools.update_wrapper(wrapper, wrapped, assigned=WRAPPER_ASSIGNMENTS, updated=WRAPPER_UPDATES)
```

更新一个 *wrapper* 函数以使其像是 *wrapped* 函数。可选参数是两个元组，分别指明原函数的哪些属性被直接赋给 *wrapper* 函数的对应属性，以及 *wrapper* 函数的哪些属性用原函数的对应属性进行更新。可选参数的默认值分别是模块级常量 `WRAPPER_ASSIGNMENTS`（它将赋值给 *wrapper* 函数的 `__module__`、`__name__`、`__qualname__`、`__annotations__` 和 `__doc__` 即文档字符串）以及 `WRAPPER_UPDATES`（它将更新 *wrapper* 函数的 `__dict__` 即实例字典）。

为了允许出于内省和其他目的（例如绕过 `lru_cache()` 之类的缓存装饰器）访问原始函数，此函数会自动为 *wrapper* 添加一个指向原函数的 `__wrapped__` 属性。

此函数主要设计用于装饰器函数。如果包装器函数未被更新，则被返回函数的元数据将反映包装器定义而不是原函数定义，这通常没有什么用处。

`update_wrapper()` 可以用于函数之外的可调用对象。在 *assigned* 或 *updated* 中命名的任何属性如果不存在于被包装对象则会被忽略（即该函数将不会尝试在包装器函数上设置它们）。如果包装器函数自身缺少在 *updated* 中命名的任何属性则仍引发 `AttributeError`。

## @wraps()

```python
@functools.wraps(wrapped, assigned=WRAPPER_ASSIGNMENTS, updated=WRAPPER_UPDATES)
```

这是一个便捷函数，用于在定义包装器函数时发起调用 `update_wrapper()` 作为函数装饰器。它等价于 `partial(update_wrapper, wrapped=wrapped, assigned=assigned, updated=updated)`。例如：

```python
>>> from functools import wraps
>>> def my_decorator(f):
...     @wraps(f)
...     def wrapper(*args, **kwds):
...         print('Calling decorated function')
...         return f(*args, **kwds)
...     return wrapper
...
>>> @my_decorator
... def example():
...     """Docstring"""
...     print('Called example function')
...
>>> example()
Calling decorated function
Called example function
>>> example.__name__
'example'
>>> example.__doc__
'Docstring'
```

如果不使用这个装饰器工厂函数，则 `example()` 的名称将变为 `'wrapper'`，并且其原本的文档字符串也会丢失。
