# timeit——测量小段代码的执行时间

`timeit` 模块提供了一种简单的方法来计算一小段 Python 代码的耗时。

以下示例展示了如何使用命令行接口来比较三个不同的表达式：

```python
$ python3 -m timeit '"-".join(str(n) for n in range(100))'
10000 loops, best of 5: 30.2 usec per loop
$ python3 -m timeit '"-".join([str(n) for n in range(100)])'
10000 loops, best of 5: 27.5 usec per loop
$ python3 -m timeit '"-".join(map(str, range(100)))'
10000 loops, best of 5: 23.2 usec per loop
```

这也可以通过 Python 接口实现：

```python
>>> import timeit
>>> timeit.timeit('"-".join(str(n) for n in range(100))', number=10000)
0.3018611848820001
>>> timeit.timeit('"-".join([str(n) for n in range(100)])', number=10000)
0.2727368790656328
>>> timeit.timeit('"-".join(map(str, range(100)))', number=10000)
0.23702679807320237
```

## Python 接口

### timeit()

```python
timeit.timeit(stmt='pass', setup='pass', timer=<default timer>, number=1000000, 
globals=None)
```

使用给定语句、`setup` 代码和 `timer` 函数创建一个 `Timer` 实例，并以给定的 `number` 次数运行其 `timeit()` 方法。可选的 `globals` 参数指定执行代码的命名空间。

```python
>>> timeit.timeit('"-".join(map(str, range(100)))', number=10000)
0.17110489700007747
```

### repeat()

```python
timeit.repeat(stmt='pass', setup='pass', timer=<default timer>, repeat=5, 
number=1000000, globals=None)
```

使用给定语句、`setup` 代码和 `timer` 函数创建一个 `Timer` 实例，并以给定的 `repeat` 计数和 `number` 次数运行其 `repeat()` 方法。可选的 `globals` 参数指定用于执行代码的命名空间。

```python
>>> timeit.repeat('"-".join(map(str, range(100)))', number=10000)
[0.17756178499985253, 0.15232318500056863, 0.16468117500062363, 0.1632813090000127, 0.15658389100008208]
```

### Timer
