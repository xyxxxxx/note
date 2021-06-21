[toc]



# BeautifulSoup

[Beautiful Soup](http://www.crummy.com/software/BeautifulSoup/) 是一个可以从HTML或XML文件中提取数据的Python库.它能够通过你喜欢的转换器实现惯用的文档导航,查找,修改文档的方式.Beautiful Soup会帮你节省数小时甚至数天的工作时间.



参考[Beautiful Soup 4.4.0 文档](https://www.crummy.com/software/BeautifulSoup/bs4/doc.zh/)

下面一段HTML代码将作为例子被多次用到.这是 *爱丽丝梦游仙境的* 的一段内容:

```html
<html><head><title>The Dormouse's story</title></head>
<body>
<p class="title"><b>The Dormouse's story</b></p>

<p class="story">Once upon a time there were three little sisters; and their names were
<a href="http://example.com/elsie" class="sister" id="link1">Elsie</a>,
<a href="http://example.com/lacie" class="sister" id="link2">Lacie</a> and
<a href="http://example.com/tillie" class="sister" id="link3">Tillie</a>;
and they lived at the bottom of a well.</p>

<p class="story">...</p>
```

在浏览器中显示如下:

![](https://i.loli.net/2020/12/29/WZtx64Cdhf7QEAN.png)

使用BeautifulSoup解析这段代码,能够得到一个 `BeautifulSoup` 对象:

```shell
>>> from bs4 import BeautifulSoup
>>> html_doc = """
... <html><head><title>The Dormouse's story</title></head>
... <body>
... <p class="title"><b>The Dormouse's story</b></p>
... 
... <p class="story">Once upon a time there were three little sisters; and their names were
... <a href="http://example.com/elsie" class="sister" id="link1">Elsie</a>,
... <a href="http://example.com/lacie" class="sister" id="link2">Lacie</a> and
... <a href="http://example.com/tillie" class="sister" id="link3">Tillie</a>;
... and they lived at the bottom of a well.</p>... 
... <p class="story">...</p>
... """
>>> soup = BeautifulSoup(html_doc, 'html.parser')
```

将其按照标准的缩进格式的结构输出:

```html
>>> print(soup.prettify())
<html>
 <head>
  <title>
   The Dormouse's story
  </title>
 </head>
 <body>
  <p class="title">
   <b>
    The Dormouse's story
   </b>
  </p>
  <p class="story">
   Once upon a time there were three little sisters; and their names were
   <a class="sister" href="http://example.com/elsie" id="link1">
    Elsie
   </a>
   ,
   <a class="sister" href="http://example.com/lacie" id="link2">
    Lacie
   </a>
   and
   <a class="sister" href="http://example.com/tillie" id="link3">
    Tillie
   </a>
   ;
and they lived at the bottom of a well.
  </p>
  <p class="story">
   ...
  </p>
 </body>
</html>
```

几个简单的浏览结构化数据的方法:

```shell
>>> soup.title             # 查看<title></title>
<title>The Dormouse's story</title>

>>> soup.title.string
"The Dormouse's story"

>>> soup.title.parent.name # 查看上级名称
'head'

>>> soup.a                 # 查看(第一个)<a></a>
<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>

>>> soup.find_all('a')     # 查找所有<a></a>
[<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>, <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>, <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]

>>> soup.find(id='link2')  # 根据id查找
<a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>
```

返回文档的全部文字内容:

```shell
>>> soup.text              # 与soup.get_text()相同
"\nThe Dormouse's story\n\nThe Dormouse's story\nOnce upon a time there were three little sisters; and their names were\nElsie,\nLacie and\nTillie;\nand they lived at the bottom of a well.\n...\n"

>>> print(soup.text)

The Dormouse's story

The Dormouse's story
Once upon a time there were three little sisters; and their names were
Elsie,
Lacie and
Tillie;
and they lived at the bottom of a well.
...

```







# requests

requests 是一个简单而优雅的 HTTP 库。



## 使用示例



## 接口

Requests 所有的功能都可以通过以下 7 个方法访问，它们都会返回一个`Response`对象的实例。

### delete()

发送 `DELETE` 请求。



### get()

发送 `GET` 请求。



### head()

发送 `HEAD` 请求。



### options()

发送 `OPTIONS` 请求。



### post()

发送 `POST` 请求。

```shell
>>> r = requests.post('http://httpbin.org/post', data = {'key':'value'})
>>> r.text
'{\n  "args": {}, \n  "data": "", \n  "files": {}, \n  "form": {\n    "key": "value"\n  }, \n  "headers": {\n    "Accept": "*/*", \n    "Accept-Encoding": "gzip, deflate", \n    "Content-Length": "9", \n    "Content-Type": "application/x-www-form-urlencoded", \n    "Host": "httpbin.org", \n    "User-Agent": "python-requests/2.25.1", \n    "X-Amzn-Trace-Id": "Root=1-60616431-7b1c56ca0f9832ba30ed9655"\n  }, \n  "json": null, \n  "origin": "106.121.161.184", \n  "url": "http://httpbin.org/post"\n}\n'
```



### put()

发送 `PUT` 请求。



### request()

构造并发送一个请求。

```python
requests.request(method, url, **kwargs)
# method    请求方法
# url       url
# params    作为查询字符串的字典或字节
# data      随请求体发送的字典、元组列表`[(key,value)]`、字节或类似文件的对象
# json      随请求体发送的json数据
# headers   设定请求头的字典
# cookies   设定cookies的字典或CookieJar对象
# files     形如`{'name': file-like-objects}`或`{'name': file-tuple}`的字典
#           其中`file-tuple`可以是二元组`{'filename', fileobj}`,三元组`{'filename', fileobj, 'content-type'}`
#           或四元组`{'filename', fileobj, 'content-type', custom_headers}`
# auth      用于HTTP认证的元组
# timeout   超时时间(s)
# allow_redirects 若为`True`则启用重定向
# proxies   将协议映射到代理url的字典
# verify
# stream
# cert
```

```shell
>>> import requests
>>> requests.request('GET', 'https://www.example.com')
<Response [200]>
```



```python

```





## 请求和响应

### PreparedRequest



### Request



### Response

`Response`对象包含了服务器的对于HTTP请求的响应。

```shell
>>> import requests
>>> r = requests.get('https://www.example.com')
```

具有以下属性和方法：

#### close()

释放连接回连接池。

#### content

响应内容（响应体），以字节形式。

```shell
>>> r.content
b'<!doctype html>\n<html>\n<head>\n    <title>Example Domain</title>\n\n    <meta charset="utf-8" />\n    <meta http-equiv="Content-type" content="text/html; charset=utf-8" />\n    <meta name="viewport" content="width=device-width, initial-scale=1" />\n    <style type="text/css">\n    body {\n        background-color: #f0f0f2;\n        margin: 0;\n        padding: 0;\n        font-family: -apple-system, system-ui, BlinkMacSystemFont, "Segoe UI", "Open Sans", "Helvetica Neue", Helvetica, Arial, sans-serif;\n        \n    }\n    div {\n        width: 600px;\n        margin: 5em auto;\n        padding: 2em;\n        background-color: #fdfdff;\n        border-radius: 0.5em;\n        box-shadow: 2px 3px 7px 2px rgba(0,0,0,0.02);\n    }\n    a:link, a:visited {\n        color: #38488f;\n        text-decoration: none;\n    }\n    @media (max-width: 700px) {\n        div {\n            margin: 0 auto;\n            width: auto;\n        }\n    }\n    </style>    \n</head>\n\n<body>\n<div>\n    <h1>Example Domain</h1>\n    <p>This domain is for use in illustrative examples in documents. You may use this\n    domain in literature without prior coordination or asking for permission.</p>\n    <p><a href="https://www.iana.org/domains/example">More information...</a></p>\n</div>\n</body>\n</html>\n'
```

#### cookies

服务器返回的CookieJar对象。

#### elapsed

从发送请求到接收响应经过的时间。

#### headers

响应头字典。

```shell
>>> r.headers
{'Content-Encoding': 'gzip', 'Accept-Ranges': 'bytes', 'Age': '472709', 'Cache-Control': 'max-age=604800', 'Content-Type': 'text/html; charset=UTF-8', 'Date': 'Sun, 28 Mar 2021 04:58:10 GMT', 'Etag': '"3147526947+gzip"', 'Expires': 'Sun, 04 Apr 2021 04:58:10 GMT', 'Last-Modified': 'Thu, 17 Oct 2019 07:18:26 GMT', 'Server': 'ECS (sjc/4E5D)', 'Vary': 'Accept-Encoding', 'X-Cache': 'HIT', 'Content-Length': '648'}
>>> r.headers['Content-Type']
'text/html; charset=UTF-8'
```

#### history



#### is_redirect

若响应是一个完备的HTTP重定向（即可以自动处理），则为`True`。

#### iter_content()

迭代响应数据。当请求设定了`stream=True`时，这会避免将响应内容一次读进内存。

```python
iter_content(chunk_size=1, decode_unicode=False)
# chunk_size
# decode_unicode
```

#### iter_lines()

逐行迭代响应数据。当请求设定了`stream=True`时，这会避免将响应内容一次读进内存。

```python
iter_lines(chunk_size=512, decode_unicode=None, delimiter=None)
```

#### json()

返回json编码的响应内容，调用 `json.loads` 方法。若响应体不包含合法的json，则引发错误 `simplejson.errors.JSONDecodeError: Expecting value: line 1 column 1 (char 0)`。

#### next

#### ok

若 `status_code` 小于400，返回 `True`。

```shell
>>> r.ok
True
```

#### raise_for_status()

引发保存的 `HTTPError`，如果发生了一个。

```python
if not r.ok:                    # r.status >= 400, an HTTPError occurred
    r.raise_for_status()        # raise this HTTPError
```

#### raw

响应的类似文件的对象表示。需要请求设定 `stream=True`。

#### reason

HTTP状态的文本表示。

```shell
>>> r.reason
'OK'
```

#### status_code

HTTP状态码，是一个整数值。

```shell
>>> r.status_code
200
>>> r.status_code < 300
True
```

#### text

响应内容（响应体），以unicode形式。

```shell
>>> r.text
'<!doctype html>\n<html>\n<head>\n    <title>Example Domain</title>\n\n    <meta charset="utf-8" />\n    <meta http-equiv="Content-type" content="text/html; charset=utf-8" />\n    <meta name="viewport" content="width=device-width, initial-scale=1" />\n    <style type="text/css">\n    body {\n        background-color: #f0f0f2;\n        margin: 0;\n        padding: 0;\n        font-family: -apple-system, system-ui, BlinkMacSystemFont, "Segoe UI", "Open Sans", "Helvetica Neue", Helvetica, Arial, sans-serif;\n        \n    }\n    div {\n        width: 600px;\n        margin: 5em auto;\n        padding: 2em;\n        background-color: #fdfdff;\n        border-radius: 0.5em;\n        box-shadow: 2px 3px 7px 2px rgba(0,0,0,0.02);\n    }\n    a:link, a:visited {\n        color: #38488f;\n        text-decoration: none;\n    }\n    @media (max-width: 700px) {\n        div {\n            margin: 0 auto;\n            width: auto;\n        }\n    }\n    </style>    \n</head>\n\n<body>\n<div>\n    <h1>Example Domain</h1>\n    <p>This domain is for use in illustrative examples in documents. You may use this\n    domain in literature without prior coordination or asking for permission.</p>\n    <p><a href="https://www.iana.org/domains/example">More information...</a></p>\n</div>\n</body>\n</html>\n'
```

#### url

响应的最终url位置。

```shell
>>> r.url
'https://www.example.com/'
```



## 会话

### Session

提供持久cookie，连接池和设置。

```shell
>>> import requests
>>> s = requests.Session()
>>> s.get('http://httpbin.org/get')
<Response [200]>
# or
>>> with requests.Session() as s:
>>>     s.get('http://httpbin.org/get')
<Response [200]>
```

具有以下属性和方法：

#### auth

默认的认证元组或对象。

#### cert

默认的SSL客户证书。

#### cookies

一个CookieJar对象，包含了当前会话设定的所有cookies。

#### request(), delete(), get(),  head(), options(), patch(), post(), put()

发送请求。

#### headers

设定请求头的字典。

#### max-redirects

允许的最大重定向次数。

#### params

查询字符串的字典。

#### proxies

将协议映射到代理url的字典，例如`{'http': 'foo.bar:3128'}`。

#### verify

默认的SSL认证。



## 身份认证

| 类                    | 描述                   |
| --------------------- | ---------------------- |
| `auth.AuthBase`       | 所有身份认证类的基类   |
| `auth.HTTPBasicAuth`  | 请求附加的HTTP基本认证 |
| `auth.HTTPProxyAuth`  | 请求附加的HTTP代理认证 |
| `auth.HTTPDigestAuth` | 请求附加的HTTP摘要认证 |



## Cookie





## 异常

| 异常名称                    | 描述                               |
| --------------------------- | ---------------------------------- |
| `requests.RequestException` | 处理请求时发生的不明确的异常       |
| `requests.ConnectionError`  | 连接错误                           |
| `requests.HTTPError`        | HTTP错误，即状态码大于等于400      |
| `requests.URLRequired`      | 需要有效的URL                      |
| `requests.TooManyRedirects` | 过多的重定向                       |
| `requests.Timeout`          | 请求超时，是下列两个异常的父类     |
| `requests.ConnectTimeout`   | 尝试连接到远程服务器时超时         |
| `requests.ReadTimeout`      | 服务器在预定时间内没有发送任何数据 |
|                             |                                    |





# PyYAML (yaml)

PyYAML 是一个 YAML 编码和解码器，使用方法类似于标准库的 json 包。



## load()

将 YAML 文档转换为 Python 对象。接受一个 Unicode 字符串、字节字符串、二进制文件对象或文本文件对象，其中字节字符串和文件必须使用 utf-8、utf-16-be 或 utf-16-le 编码（若没有指定编码，则默认为 utf-8 编码）。

出于[安全上的原因](https://github.com/yaml/pyyaml/wiki/PyYAML-yaml.load(input)-Deprecation)，调用此函数时需要传入 `Loader` 参数，目前该参数有以下选项：

+ `BaseLoader`

  仅加载最基本的 YAML，所有的数字都会被加载为字符串。

+ `SafeLoader`

  安全地加载 YAML 语言的一个子集，是加载不被信任的数据时的推荐选项。对应于快捷函数 `yaml.safe_load()`。

+ `FullLoader`

  加载完整的 YAML 语言，是当前的默认选项。存在明显的漏洞，暂时不要加载不被信任的数据。对应于快捷函数 `yaml.full_load()`。

+ `UnsafeLoader`

  最初的 `Loader` 实现，可以轻易地被不被信任的数据输入利用。对应于快捷函数 `yaml.unsafe_load()`。

```python
>>> import yaml
>>> doc = """
# Project information
site_name: Test Docs
site_author: xyx

# Repository
repo_name: xyx/test-project/mkdocs/test
repo_url: http://gitlab.dev.tensorstack.net/xyx/test-project/tree/master/mkdocs/test-project
edit_uri: ""

# Copyright
copyright: Copyright &copy; 2016 - 2021 xxx

# Configuration
theme:
  name: material  # https://github.com/squidfunk/mkdocs-material
  custom_dir: overrides    # overrides HTML elements
  language: zh
  features:
    - navigation.sections  # keep this
    - navigation.tabs      # keep this
    - navigation.top       # keep this
  palette:
    scheme: default        # keep this
    primary: green         # primary color of theme
    accent: light green    # color of elements that can be interacted with
  favicon: assets/icon.svg # showed as tab icon
  logo: assets/logo.svg    # showed at top left of page
"""
>>> from pprint import pprint
>>> pprint(yaml.load(doc, Loader=yaml.SafeLoader))   # yaml to py dict
{'copyright': 'Copyright &copy; 2016 - 2021 xxx',
 'edit_uri': '',
 'repo_name': 'xyx/test-project/mkdocs/test',
 'repo_url': 'http://gitlab.dev.tensorstack.net/xyx/test-project/tree/master/mkdocs/test-project',
 'site_author': 'xyx',
 'site_name': 'Test Docs',
 'theme': {'custom_dir': 'overrides',
           'favicon': 'assets/icon.svg',
           'features': ['navigation.sections',
                        'navigation.tabs',
                        'navigation.top'],
           'language': 'zh',
           'logo': 'assets/logo.svg',
           'name': 'material',
           'palette': {'accent': 'light green',
                       'primary': 'green',
                       'scheme': 'default'}}}
```

```python
# Load yaml from file
>>> with open('test.yaml', 'rt') as f:
...   pprint(yaml.load(f, Loader=yaml.SafeLoader))
... 
{'copyright': 'Copyright &copy; 2016 - 2021 xxx',
 'edit_uri': '',
 'repo_name': 'xyx/test-project/mkdocs/test',
 'repo_url': 'http://gitlab.dev.tensorstack.net/xyx/test-project/tree/master/mkdocs/test-project',
 'site_author': 'xyx',
 'site_name': 'Test Docs',
 'theme': {'custom_dir': 'overrides',
           'favicon': 'assets/icon.svg',
           'features': ['navigation.sections',
                        'navigation.tabs',
                        'navigation.top'],
           'language': 'zh',
           'logo': 'assets/logo.svg',
           'name': 'material',
           'palette': {'accent': 'light green',
                       'primary': 'green',
                       'scheme': 'default'}}}
```





## dump()

将 Python 对象转换为 YAML 文档。

```python
>>> import yaml
>>> d = {'copyright': 'Copyright &copy; 2016 - 2021 xxx',
         'edit_uri': '',
         'repo_name': 'xyx/test-project/mkdocs/test',
         'repo_url': 'http://gitlab.dev.tensorstack.net/xyx/test-project/tree/master/mkdocs/test-project',
         'site_author': 'xyx',
         'site_name': 'Test Docs',
         'theme': {'custom_dir': 'overrides',
                   'favicon': 'assets/icon.svg',
                   'features': ['navigation.sections',
                                'navigation.tabs',
                                'navigation.top'],
                   'language': 'zh',
                   'logo': 'assets/logo.svg',
                   'name': 'material',
                   'palette': {'accent': 'light green',
                               'primary': 'green',
                               'scheme': 'default'}}}
>>> print(yaml.dump(d))                                # py dict to yaml
copyright: Copyright &copy; 2016 - 2021 xxx
edit_uri: ''
repo_name: xyx/test-project/mkdocs/test
repo_url: http://gitlab.dev.tensorstack.net/xyx/test-project/tree/master/mkdocs/test-project
site_author: xyx
site_name: Test Docs
theme:
  custom_dir: overrides
  favicon: assets/icon.svg
  features:
  - navigation.sections
  - navigation.tabs
  - navigation.top
  language: zh
  logo: assets/logo.svg
  name: material
  palette:
    accent: light green
    primary: green
    scheme: default

```

```python
>>> import yaml
>>> bart = Student('Bart Simpson', 59)
>>> print(yaml.dump(bart))                              # py normal object to yaml
!!python/object:__main__.Student
name: Bart Simpson
score: 59

```



