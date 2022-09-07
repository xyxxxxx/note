# inspect——检查对象

## 获取源代码

### getdoc()

返回对象的文档字符串。

### getfile()

返回定义对象的（文本或二进制）文件的名称。

### getmodule()

猜测定义对象的模块。

### getsource()

返回对象的源代码文本。

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

### getsourcefile()

返回定义对象的 Python 源文件的名称。

### currentframe()

返回调用者栈帧的帧对象。

### ismodule()

若对象为模块，返回 `True`。

### isclass(), isabstract()

若对象为类/抽象基类，返回 `True`。

### isfunction(), ismethod(), isroutine()

若对象为函数/绑定的方法/函数或方法，返回 `True`。

## 解释器栈

下列函数返回的“帧记录”是一个命名元组 `FrameInfo(frame, filename, lineno, function, code_context, index)`。该元组包含帧对象、文件名、当前行的行号、函数名、源代码中上下文各行的列表以及当前行在该列表中的索引。

### currentframe()

返回调用者栈帧的帧对象。

### stack()

返回调用者栈的帧记录的列表，其中第一项代表调用者，最后一项代表栈的最外层调用。
