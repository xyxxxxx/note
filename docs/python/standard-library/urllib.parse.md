# urllib.parse——用于解析 URL

`urllib.parse` 模块定义了一个标准接口，用于 URL 字符串按组件（协议、网络位置、路径等）分解，或将组件组合回 URL 字符串，并将“相对 URL”转换为给定“基础 URL”的绝对 URL。

## URL 解析

URL 解析功能可以将一个 URL 字符串分解为其组件，或者将 URL 组件组合成一个 URL 字符串。

### urlparse()

```python
urllib.parse.urlparse(urlstring, scheme='', allow_fragments=True)
```

将 URL 解析为六个组件，返回一个命名六元组，这对应于 URL 的一般结构：`scheme://netloc/path;parameters?query#fragment`。元组的每一项是一个字符串，可能为空。这些组件不能继续分解（例如网络位置是单个字符串），并且 % 转义不会展开。上面结构中的分隔符不是结果的一部分，除了路径组件的第一个 `/`。

```python
>>> from urllib.parse import urlparse
>>> o = urlparse('http://www.cwi.nl:80/%7Eguido/Python.html')
>>> o   
ParseResult(scheme='http', netloc='www.cwi.nl:80', path='/%7Eguido/Python.html',
            params='', query='', fragment='')
>>> o.scheme
'http'
>>> o.port
80
>>> o.geturl()
'http://www.cwi.nl:80/%7Eguido/Python.html'
```

根据 RFC 1808 中的语法规范，`urlparse()` 仅在 netloc 前面正确地附带了 `'//'` 的情况下才会识别它。否则输入会被当作是一个相对 URL 因而以路径的组成部分开头。

```python
>>> from urllib.parse import urlparse
>>> urlparse('//www.cwi.nl:80/%7Eguido/Python.html')
ParseResult(scheme='', netloc='www.cwi.nl:80', path='/%7Eguido/Python.html',
            params='', query='', fragment='')
>>> urlparse('www.cwi.nl/%7Eguido/Python.html')
ParseResult(scheme='', netloc='', path='www.cwi.nl/%7Eguido/Python.html',
            params='', query='', fragment='')
>>> urlparse('help/Python.html')
ParseResult(scheme='', netloc='', path='help/Python.html', params='',
            query='', fragment='')
```

*scheme* 参数给出了默认的协议，只有在 URL 未指定协议的情况下才会被使用。它应该是与 *urlstring* 相同的类型（文本或字节串），除此之外默认值 `''` 也总是被允许，并会在适当情况下自动转换为 `b''`。

如果 *allow_fragments* 参数为 False，则片段标识符不会被识别。它们会被解析为路径、参数或查询部分，在返回值中 `fragment` 会被设为空字符串。

返回值是一个命名元组，这意味着它的条目可以通过索引或作为命名属性来访问，这些属性是：

| 属性       | 索引 | 值                       | 值（如果不存在）   |
| :--------- | :--- | :----------------------- | :----------------- |
| `scheme`   | 0    | URL方案说明符            | *scheme* parameter |
| `netloc`   | 1    | 网络位置部分             | 空字符串           |
| `path`     | 2    | 分层路径                 | 空字符串           |
| `params`   | 3    | 最后路径元素的参数       | 空字符串           |
| `query`    | 4    | 查询组件                 | 空字符串           |
| `fragment` | 5    | 片段识别                 | 空字符串           |
| `username` |      | 用户名                   | `None`             |
| `password` |      | 密码                     | `None`             |
| `hostname` |      | 主机名（小写）           | `None`             |
| `port`     |      | 端口号为整数（如果存在） | `None`             |

如果在 URL 中指定了无效的端口，读取 `port` 属性将引发 `ValueError`。

### urlunparse()

```python
urllib.parse.urlunparse(parts)
```

根据 `urlparse()` 所返回的元组来构造一个 URL。 *parts* 参数可以是任何包含六个条目的可迭代对象。构造的结果可能是略有不同但保持等价的 URL，如果被解析的 URL 原本包含不必要的分隔符（例如带有空查询的 `?`；RFC 已声明这是等价的）。

## URL 转码

### urlencode()

```python
urllib.parse.urlencode(query, doseq=False, safe='', encoding=None, errors=None, quote_via=quote_plus)
```
