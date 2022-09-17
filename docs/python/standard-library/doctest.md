# doctest——测试交互性的 Python 示例

`doctest` 模块搜索看起来像像交互式 Python 会话的文本片段，然后执行这些会话来验证它们正如展示的那样正确运行。有几种通常的方法来使用 doctest：

* 通过验证所有交互式示例仍然按照记录的方式工作，来检查模块的文档字符串是最新的。
* 通过验证来自一个测试文件或测试对象的交互式示例按照预期工作，来进行回归测试。
* 为一个 Python 包写教程文档，用输入输出的例子来进行说明。取决于是强调例子还是解释性的文本，这有一种“文本测试”或“可执行文档”的风格。

下面是一个简单而完整的示例模块：

```python
"""
This is the "example" module.

The example module supplies one function, factorial().  For example,

>>> factorial(5)
120
"""

def factorial(n):
    """Return the factorial of n, an exact integer >= 0.

    >>> [factorial(n) for n in range(6)]
    [1, 1, 2, 6, 24, 120]
    >>> factorial(30)
    265252859812191058636308480000000
    >>> factorial(-1)
    Traceback (most recent call last):
        ...
    ValueError: n must be >= 0

    Factorials of floats are OK, but the float must be an exact integer:
    >>> factorial(30.1)
    Traceback (most recent call last):
        ...
    ValueError: n must be exact integer
    >>> factorial(30.0)
    265252859812191058636308480000000

    It must also not be ridiculously large:
    >>> factorial(1e100)
    Traceback (most recent call last):
        ...
    OverflowError: n too large
    """

    import math
    if not n >= 0:
        raise ValueError("n must be >= 0")
    if math.floor(n) != n:
        raise ValueError("n must be exact integer")
    if n+1 == n:  # catch a value like 1e300
        raise OverflowError("n too large")
    result = 1
    factor = 2
    while factor <= n:
        result *= factor
        factor += 1
    return result


if __name__ == "__main__":
    import doctest
    doctest.testmod()
```

如果你直接在命令行中运行 `example.py`，`doctest` 将发挥它的作用：

```shell
$ python example.py
$
```

没有输出！这很正常，这意味着所有的示例都成功了。将 `-v` 传入脚本，`doctest` 会打印出它所尝试的详细日志，并在最后打印总结：

```shell
$ python example.py -v
Trying:
    factorial(5)
Expecting:
    120
ok
Trying:
    [factorial(n) for n in range(6)]
Expecting:
    [1, 1, 2, 6, 24, 120]
ok
```

以此类推，最终结束于：

```shell
Trying:
    factorial(1e100)
Expecting:
    Traceback (most recent call last):
        ...
    OverflowError: n too large
ok
2 items passed all tests:
   1 tests in __main__
   8 tests in __main__.factorial
9 tests in 2 items.
9 passed and 0 failed.
Test passed.
$
```

这就是对于高效地使用 `doctest` 你所需要知道的一切！开始上手吧。下面的部分提供了完整的细节。请注意，在标准的 Python 测试套件和库中有许多 doctest 的例子。其中特别有用的例子可以在标准测试文件 `Lib/test/test_doctest.py` 中找到。
