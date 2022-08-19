# importlib——import 的实现

`importlib` 包有两方面的目的。其一是以 Python 源代码的形式提供 `import` 语句（以及扩展到 `__import__()` 函数）的实现。这提供了一个可移植到任何 Python 解释器的 `import` 实现。相比使用 Python 以外的编程语言实现方式，这一实现更加易于理解。

其二是实现 `import` 的组件被暴露在这个包中，使得用户可以更容易创建他们自己的自定义对象 (通常被称为导入器) 来参与到导入过程中。

## \__import__()

内置 `__import__()` 函数的实现。`import` 语句是这个函数的语法糖。

!!! note "注意"
    以编程方式导入一个模块，应使用 `import_module()` 而不是这个函数。

## import_module()

```python
importlib.import_module(name, package=None)
```

导入一个模块。参数 *name* 指定以绝对或相对导入方式导入的模块；如果参数 *name* 使用相对导入方式，那么参数 *packages* 必须设置为相应的包名，并作为解析模块名的锚点，例如：

```python
>>> import importlib
>>> importlib.import_module('numpy')       # 返回指定的模块(或包)实例
<module 'numpy' from '/Users/xyx/.pyenv/versions/3.8.7/lib/python3.8/site-packages/numpy/__init__.py'>
>>> np = importlib.import_module('numpy')
>>> np.__version__
'1.19.5'
>>> np.arange(6)
array([0, 1, 2, 3, 4, 5])
```

`import_module()` 函数是一个对 `importlib.__import__()` 进行简化的包装器。这意味着该函数的所有主义都来自于 `importlib.__import__()`。这两个函数最重要的不同之处在于 `import_module()` 返回指定的包或模块（例如 `pkg.mod`），而 `__import__()` 返回最高层级的包或模块（例如 `pkg`）。

## importlib.util——导入器的工具程序代码

本模块包含了帮助构建导入器的多个对象。

### find_spec()

```python
import importlib.util
import sys

# For illustrative purposes
name = 'itertools'

if name in sys.modules:
    print("{} already in sys.modules".format(name))
elif (spec := importlib.util.find_spec(name)) is not None:
    # If you chose to perform the actual import ...
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    print("{} has been imported".format(name))
else:
    print("can't find the {} module".format(name))
```

### module_from_spec()
