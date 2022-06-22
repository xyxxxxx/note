# json——JSON 编码和解码器

> `json` 包实际上就是被添加到标准库中的 `simplejson` 包。`simplejson` 包比 Python 版本更新更加频繁，因此在条件允许的情况下使用 `simplejson` 包是更好的选择。一种好的实践如下：
>
> ```python
> try:
>     import simplejson as json
> except ImportError:
>     import json
> ```
>
> `json` 包和 `simplejson` 包具有相同的接口和类，因此下面仅 `json` 包为例进行介绍。

## 接口

### dumps()

将对象序列化为 JSON 格式的 `str`。参数的含义见 `JSONEncoder`。

```python
>>> import json
>>> json.dumps(['foo', {'bar': ('baz', None, 1.0, 2)}])
'["foo", {"bar": ["baz", null, 1.0, 2]}]'
>>> json.dumps({"c": 0, "b": 1, "a": math.nan})            # 允许NaN,inf
'{"c": 0, "b": 1, "a": NaN}'
>>> json.dumps({"c": 0, "b": 0, "a": 0}, sort_keys=True)   # 按键排序
'{"a": 0, "b": 0, "c": 0}'
>>>
>>> json.dumps({"c": 0, "b": 1, "a": {"d": 2, "e": 3}})
'{"c": 0, "b": 1, "a": {"d": 2, "e": 3}}'
>>> json.dumps({"c": 0, "b": 1, "a": {"d": 2, "e": 3}}, indent=0)           # 美化输出
'{\n"c": 0,\n"b": 1,\n"a": {\n"d": 2,\n"e": 3\n}\n}'
>>> print(json.dumps({"c": 0, "b": 1, "a": {"d": 2, "e": 3}}, indent=0))
{
"c": 0,
"b": 1,
"a": {
"d": 2,
"e": 3
}
}
>>> print(json.dumps({"c": 0, "b": 1, "a": {"d": 2, "e": 3}}, indent=2))
{
  "c": 0,
  "b": 1,
  "a": {
    "d": 2,
    "e": 3
  }
}
```

### loads()

将一个包含 JSON 的 `str`、`bytes` 或 `bytearray` 实例反序列化为 Python 对象。

```python
>>> import json
>>> json.loads('["foo", {"bar": ["baz", null, 1.0, 2]}]')
['foo', {'bar': ['baz', None, 1.0, 2]}]
>>> json.loads('{"c": 0, "b": 1, "a": {"d": 2, "e": 3}}')
{'c': 0, 'b': 1, 'a': {'d': 2, 'e': 3}}
```

## 编码器和解码器

### JSONEncoder

用于 Python 数据结构的可扩展 JSON 编码器，默认支持以下对象和类型：

| Python                              | JSON   |
| :---------------------------------- | :----- |
| dict                                | object |
| list, tuple                         | array  |
| str                                 | string |
| int, float, int 和 float 派生的枚举 | number |
| True                                | true   |
| False                               | false  |
| None                                | null   |

```python
class json.JSONEncoder(*, skipkeys=False, ensure_ascii=True, check_circular=True, allow_nan=True, sort_keys=False, indent=None, separators=None, default=None)
# skipkeys       若为`False`,则当尝试对非`str`,`int`,`float`或`None`的键进行编码时将会引发`TypeError`;
#                   否则这些条目将被直接跳过
# ensure_ascii   若为`True`,所有输入的非ASCII字符都将被转义;否则会原样输出
# check_circular 若为`True`,则列表、字典和自定义编码的对象在编码期间会被检查重复循环引用防止无限递归
#                   (无限递归将导致`OverflowError`)
# allow_nan      若为`True`,则对`NaN`,`Infinity`和`-Infinity`进行编码.此行为不符合JSON规范,但与大多数的基于Javascript
#                   的编码器和解码器一致;否则引发一个`ValueError`
# sort_keys      若为`True`,则字典的输出是按照键排序
# indent         若为一个非负整数或字符串,则JSON数组元素和对象成员会被美化输出为该值指定的缩进等级;若为零、负数或者"",
#                   则只会添加换行符;`None`(默认值)选择最紧凑的表达
```

具有以下属性和方法：

#### default()

#### encode()

返回 Python 数据结构的 JSON 字符串表达方式。

```python
>>> json.JSONEncoder().encode({"foo": ["bar", "baz"]})
'{"foo": ["bar", "baz"]}'
```

### JSONDecoder

## 异常

### JSONDecodeError

JSON 解析错误，是 `ValueError` 的子类。

```python
exception json.JSONDecodeError(msg, doc, pos)
# msg     未格式化的错误信息
# doc     正在解析的JSON文档
# pos     解析出错的文档索引位置
```

```python
>>> import json
>>> try:
    json.loads('{"c": 0, "b": 1, "a": {"d": 2, "e" 3}}')   # lack a colon between "e" and 3
except json.JSONDecodeError as e:
    print(e.msg)
    print(e.doc)
    print(e.pos)
... J
Expecting ':' delimiter
{"c": 0, "b": 1, "a": {"d": 2, "e" 3}}
35                              # ↑
```
