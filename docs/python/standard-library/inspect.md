# inspect——检查对象

## currentframe()

返回调用者栈帧的帧对象。

## ismodule()

若对象为模块，返回 `True`。

## isclass(), isabstract()

若对象为类/抽象基类，返回 `True`。

## isfunction(), ismethod(), isroutine()

若对象为函数/绑定的方法/函数或方法，返回 `True`。

## getdoc()

返回对象的 `docstring`。

## getfile(), getsourcefile()

返回对象被定义的模块的路径。

## getmodule()

猜测对象被定义的模块。

## getsource()

返回对象的源代码。

```python
>>> print(inspect.getsource(json.JSONDecoder))     # 类的源代码
class JSONDecoder(object):
...
    
>>> print(inspect.getsource(json.loads))           # 函数的源代码
def loads(s, *, cls=None, object_hook=None, parse_float=None,
        parse_int=None, parse_constant=None, object_pairs_hook=None, **kw):
...

>>> print(inspect.getsource(json))                 # 模块的源代码
r"""JSON (JavaScript Object Notation) <http://json.org> is a subset of
JavaScript syntax (ECMA-262 3rd edition) used as a lightweight data
interchange format.
...

```
