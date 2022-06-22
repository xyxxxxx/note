# importlib——import 的实现

## \__import__()

内置 `__import__()` 函数的实现。

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
