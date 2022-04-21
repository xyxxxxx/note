[toc]

# 常用库

## BeautifulSoup

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

## click

[click](https://click.palletsprojects.com/en/8.0.x/) 是一个用于创建漂亮的命令行界面的 Python 包，其采用可组合的方法，只需要很少的代码修改。

## filelock

支持 `with` 语句的文件锁。

### Timeout

若未能在 `timeout` 秒之内获得，则引发此异常。

### FileLock

### UnixFileLock

在 Unix 系统上使用 `fcntl.flock()` 以硬锁定文件。

### WindowsFileLock

## Pillow

PIL（Python Imaging Library）是 Python 的图像处理包，Pillow 是 PIL 的一个分叉，提供了扩展的文件格式的支持、高效的内部表示和强大的图像处理功能。

### Image

#### Image

图像类。此类的实例通过工厂函数 `Image.open()`、`Image.new()` 和 `Image.frombytes()` 创建得到。

##### close()

关闭文件指针。

##### convert()

返回图像转换后的副本。

```python
im_8bit = im_rgb.convert('L')   # L = R * 299/1000 + G * 587/1000 + B * 114/1000
im_1bit = im_8bit.convert('1')  # 127 -> 0, 128 -> 255 (1)
```

##### copy()

返回图像的副本。

##### create()

以给定的模式和大小创建一个新的图像。

##### crop()

返回图像的一个矩形区域。

```python
with Image.open('0.png') as im:
    im_crop = im.crop((20, 20, 100, 100))   # 元组(左,上,右,下)定义了裁剪的像素坐标
```

##### entropy()

计算并返回图像的熵。

##### filename

源文件的文件名或路径。只有由工厂函数 `open()` 创建的图像有此属性。

##### format

源文件的格式。只有由工厂函数 `open()` 创建的图像有此属性。

##### getbands()

返回包含图像各通道名称的元组。

```python
>>> im_rgb.getbands()
('R', 'G', 'B')
>>> im_8bit.getbands()
('L',)
```

##### getchannel()

返回包含图像单个通道的图像。

```python
>>> im_r = im_rgb.getchannel('R')
>>> im_r.getbands()
('L',)
```

##### getcolors()

返回图像中使用的颜色列表。

```python
>>> im = Image.effect_noise((5, 5), 32)
>>> im.getcolors()  # (num, L)
[(1, 75), (2, 101), (1, 107), (1, 108), (1, 110), (1, 111), (1, 112), (1, 114), (2, 115), (2, 117), (2, 120), (1, 127), (2, 128), (1, 149), (1, 151), (1, 153), (1, 162), (1, 163), (1, 166), (1, 182)]
```

##### getdata()

返回图像内容为包含像素值的展开的序列对象。

```python
>>> list(im_8bit.getdata())
[124, 141, 168, ..., 138]
>>> list(im_rgb.getdata())
[(255, 255, 255), (255, 255, 255), (255, 255, 255), ..., (255, 255, 255)]
```

##### getpixel()

返回图像中指定位置的像素值。

```python
>>> im_8bit.getpixel((0, 0))
124
>>> im_rgb.getpixel((0, 0))
(255, 255, 255)
```

##### height

图像的高。

##### is_animated

如果图像有超过一帧，返回 `True`，否则返回 `False`。

##### mode

图像模式。

##### n_frames

图像的帧数。

##### open()

打开并识别给定的图像文件。

```python
im = Image.open('0.png')
```

##### paste()

将另一个图像粘贴（覆盖）到图像中。

##### putpixel()

修改图像中指定位置的像素值。

##### quantize()

将图像转换为 P 模式，包含指定数量的颜色。

##### reduce()

返回图像缩小指定倍数后的副本。

```python
>>> im.size
(300, 300)
>>> im.reduce(2).size
(150, 150)
```

##### resize()

返回图像的改变大小后的副本。

```python
with Image.open('0.png') as im:
    im_resized = im.resize(((im.width // 2, im.height // 2)))
    # 宽和高各减小为原来的1/2
```

##### rotate()

返回图像的旋转后的副本。

```python
with Image.open('0.png') as im:
    im_rotated = im.rotate(angle=60, expand=True, fillcolor='white')
    # 顺时针旋转60度,扩展输出图像以容纳旋转后的整个图像,空白部分用白色填充
```

##### save()

以给定的文件名保存图像。如果没有指定格式，则格式从文件名的扩展名推断而来。

```python
with Image.open('0.png') as im:
    im.save('0-1.png')
```

##### show()

展示图像。

```python
with Image.open('0.png') as im:
    im.show()
```

##### size

图像大小，以二元组 `(width, height)` 给出。

##### split()

分割图像为单个的通道。

```python
>>> im_r, im_g, im_b = im_rgb.split()
>>> im_r.getbands()
('L',)
```

##### tobytes()

返回图像为字节对象。

##### transform()

变换图像。

##### transpose()

转置图像。

```python
im_flipped = im.transpose(method=Image.Transpose.FLIP_LEFT_RIGHT)
# FLIP_LEFT_RIGHT     左右翻转
# FLIP_TOP_BOTTOM     上下翻转
# ROTATE_90           旋转90度
# ROTATE_180          旋转180度
# ROTATE_270          旋转270度
# TRANSPOSE           转置
# TRANSVERSE          转置后旋转180度
```

##### verify()

验证文件内容。

##### width

图像的宽。

#### blend()

通过在两个输入图像之间插值以创建一个新的图像。

```python
im0 = Image.open('0.png')
im1 = Image.open('1.png')
im = Image.blend(im0, im1, alpha=0.2)  # 0.8 im0 + 0.2 im1
```

#### effect_noise()

产生以 128 为期望的高斯噪声。

```python
im = Image.effect_noise((20, 20), 32)
                                  # 噪声的标准差
```

#### eval()

将函数（应接收一个参数）应用到给定图像中的每一个像素。如果图像有多于一个通道，则相同的函数被应用到每个通道。

#### fromarray()

从具有数组接口的对象创建图像。

```python
import numpy as np

im = Image.open('0.png')
a = np.asarray(im)
# array([[  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
#           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
#           0,   0],
#        ...
#        [  0,   0,   0,   0,   0,   0,   0,   0,   0,  14, 179, 245, 236,
#         242, 254, 254, 254, 254, 245, 235,  84,   0,   0,   0,   0,   0,
#           0,   0],
#        ...
#        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
#           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
#           0,   0]], dtype=uint8)
im = Image.fromarray(a)

```

#### merge()

将一组单通道图像合并为一个多通道图像。

#### new()

以给定的模式和大小创建一个新的图像。

```python
im_8bit = Image.new('L', (200, 200), 128)
im_rgb = Image.new('RGB', (200, 200), (0, 206, 209))
```

## requests

requests 是一个简单而优雅的 HTTP 库。[使用教程](https://docs.python-requests.org/zh_CN/latest/user/quickstart.html)

### 使用示例

### 接口

requests 所有的功能都可以通过以下 7 个方法访问，它们都会返回一个 `Response` 对象的实例。

#### delete()

发送 `DELETE` 请求。

#### get()

发送 `GET` 请求。

```python
>>> payload = {'key1': 'value1', 'key2': 'value2'}
>>> r = requests.get('https://httpbin.org/get', params=payload)   # 传入字典作为请求查询参数
>>> print(r.text)
{
  "args": {
    "key1": "value1", 
    "key2": "value2"
  }, 
  "headers": {
    "Accept": "*/*", 
    "Accept-Encoding": "gzip, deflate", 
    "Host": "httpbin.org", 
    "User-Agent": "python-requests/2.25.1", 
    "X-Amzn-Trace-Id": "Root=1-6124b95d-2a6ab1a21adacae014c17dbe"
  }, 
  "origin": "64.225.113.187", 
  "url": "https://httpbin.org/get?key1=value1&key2=value2"        # URL被正确编码
}
```

#### head()

发送 `HEAD` 请求。

#### options()

发送 `OPTIONS` 请求。

#### post()

发送 `POST` 请求。

```shell
>>> r = requests.post('http://httpbin.org/post', data = {'key': 'value'})
>>> print(r.text)
{
  "args": {}, 
  "data": "", 
  "files": {}, 
  "form": {
    "key": "value"
  }, 
  "headers": {
    "Accept": "*/*", 
    "Accept-Encoding": "gzip, deflate", 
    "Cache-Control": "max-age=259200", 
    "Content-Length": "9", 
    "Content-Type": "application/x-www-form-urlencoded", 
    "Host": "httpbin.org", 
    "User-Agent": "python-requests/2.25.1", 
    "X-Amzn-Trace-Id": "Root=1-6124bace-36ee999f01d29eee1bc634a5"
  }, 
  "json": null, 
  "origin": "64.225.113.187", 
  "url": "http://httpbin.org/post"
}
```

#### put()

发送 `PUT` 请求。

```python
>>> r = requests.put('http://httpbin.org/put', data = {'key': 'value'})
>>> print(r.text)
{
  "args": {}, 
  "data": "", 
  "files": {}, 
  "form": {
    "key": "value"
  }, 
  "headers": {
    "Accept": "*/*", 
    "Accept-Encoding": "gzip, deflate", 
    "Cache-Control": "max-age=259200", 
    "Content-Length": "9", 
    "Content-Type": "application/x-www-form-urlencoded", 
    "Host": "httpbin.org", 
    "User-Agent": "python-requests/2.25.1", 
    "X-Amzn-Trace-Id": "Root=1-6124bb00-4fb3f8d0573f63313be75fc8"
  }, 
  "json": null, 
  "origin": "64.225.113.187", 
  "url": "http://httpbin.org/put"
}
```

#### request()

构造并发送一个请求。

```python
requests.request(method, url, **kwargs)
# method    请求方法
# url       URL
# params    作为请求查询参数的字典,列表或字节,用于构造查询字符串
# data      随请求体发送的字典,元组列表`[(key,value)]`,字节或类似文件的对象
# json      随请求体发送的json数据
# headers   设定请求头的字典
# cookies   设定cookies的字典或CookieJar对象
# files     形如`{'name': file-like-objects}`或`{'name': file-tuple}`的字典,用于上传
#           多部分编码的文件.其中`file-tuple`可以是二元组`('filename', fileobj)`,三元组
#           `('filename', fileobj, 'content-type')`或四元组`('filename', fileobj, 
#           'content-type', custom_headers)`
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

### 请求和响应

#### PreparedRequest

#### Request

#### Response

`Response`对象包含了服务器的对于HTTP请求的响应。

```shell
>>> import requests
>>> r = requests.get('https://www.example.com')
```

具有以下属性和方法：

##### close()

释放连接回连接池。

##### content

响应内容（响应体），以字节形式。

```shell
>>> r.content
b'<!doctype html>\n<html>\n<head>\n    <title>Example Domain</title>\n\n    <meta charset="utf-8" />\n    <meta http-equiv="Content-type" content="text/html; charset=utf-8" />\n    <meta name="viewport" content="width=device-width, initial-scale=1" />\n    <style type="text/css">\n    body {\n        background-color: #f0f0f2;\n        margin: 0;\n        padding: 0;\n        font-family: -apple-system, system-ui, BlinkMacSystemFont, "Segoe UI", "Open Sans", "Helvetica Neue", Helvetica, Arial, sans-serif;\n        \n    }\n    div {\n        width: 600px;\n        margin: 5em auto;\n        padding: 2em;\n        background-color: #fdfdff;\n        border-radius: 0.5em;\n        box-shadow: 2px 3px 7px 2px rgba(0,0,0,0.02);\n    }\n    a:link, a:visited {\n        color: #38488f;\n        text-decoration: none;\n    }\n    @media (max-width: 700px) {\n        div {\n            margin: 0 auto;\n            width: auto;\n        }\n    }\n    </style>    \n</head>\n\n<body>\n<div>\n    <h1>Example Domain</h1>\n    <p>This domain is for use in illustrative examples in documents. You may use this\n    domain in literature without prior coordination or asking for permission.</p>\n    <p><a href="https://www.iana.org/domains/example">More information...</a></p>\n</div>\n</body>\n</html>\n'
```

##### cookies

服务器返回的CookieJar对象。

##### elapsed

从发送请求到接收响应经过的时间。

##### headers

响应头字典。

```shell
>>> r.headers
{'Content-Encoding': 'gzip', 'Accept-Ranges': 'bytes', 'Age': '472709', 'Cache-Control': 'max-age=604800', 'Content-Type': 'text/html; charset=UTF-8', 'Date': 'Sun, 28 Mar 2021 04:58:10 GMT', 'Etag': '"3147526947+gzip"', 'Expires': 'Sun, 04 Apr 2021 04:58:10 GMT', 'Last-Modified': 'Thu, 17 Oct 2019 07:18:26 GMT', 'Server': 'ECS (sjc/4E5D)', 'Vary': 'Accept-Encoding', 'X-Cache': 'HIT', 'Content-Length': '648'}
>>> r.headers['Content-Type']
'text/html; charset=UTF-8'
```

##### history

##### is_redirect

若响应是一个完备的HTTP重定向（即可以自动处理），则为`True`。

##### iter_content()

迭代响应数据。当请求设定了`stream=True`时，这会避免将响应内容一次读进内存。

```python
iter_content(chunk_size=1, decode_unicode=False)
# chunk_size
# decode_unicode
```

##### iter_lines()

逐行迭代响应数据。当请求设定了`stream=True`时，这会避免将响应内容一次读进内存。

```python
iter_lines(chunk_size=512, decode_unicode=None, delimiter=None)
```

##### json()

返回json编码的响应内容，调用 `json.loads` 方法。若响应体不包含合法的json，则引发错误 `simplejson.errors.JSONDecodeError: Expecting value: line 1 column 1 (char 0)`。

##### next

##### ok

若 `status_code` 小于400，返回 `True`。

```shell
>>> r.ok
True
```

##### raise_for_status()

引发保存的 `HTTPError`，如果发生了一个。

```python
if not r.ok:                    # r.status >= 400, an HTTPError occurred
    r.raise_for_status()        # raise this HTTPError
```

##### raw

响应的类似文件的对象表示。需要请求设定 `stream=True`。

##### reason

HTTP状态的文本表示。

```shell
>>> r.reason
'OK'
```

##### status_code

HTTP状态码，是一个整数值。

```shell
>>> r.status_code
200
>>> r.status_code < 300
True
```

##### text

响应内容（响应体），以unicode形式。

```shell
>>> r.text
'<!doctype html>\n<html>\n<head>\n    <title>Example Domain</title>\n\n    <meta charset="utf-8" />\n    <meta http-equiv="Content-type" content="text/html; charset=utf-8" />\n    <meta name="viewport" content="width=device-width, initial-scale=1" />\n    <style type="text/css">\n    body {\n        background-color: #f0f0f2;\n        margin: 0;\n        padding: 0;\n        font-family: -apple-system, system-ui, BlinkMacSystemFont, "Segoe UI", "Open Sans", "Helvetica Neue", Helvetica, Arial, sans-serif;\n        \n    }\n    div {\n        width: 600px;\n        margin: 5em auto;\n        padding: 2em;\n        background-color: #fdfdff;\n        border-radius: 0.5em;\n        box-shadow: 2px 3px 7px 2px rgba(0,0,0,0.02);\n    }\n    a:link, a:visited {\n        color: #38488f;\n        text-decoration: none;\n    }\n    @media (max-width: 700px) {\n        div {\n            margin: 0 auto;\n            width: auto;\n        }\n    }\n    </style>    \n</head>\n\n<body>\n<div>\n    <h1>Example Domain</h1>\n    <p>This domain is for use in illustrative examples in documents. You may use this\n    domain in literature without prior coordination or asking for permission.</p>\n    <p><a href="https://www.iana.org/domains/example">More information...</a></p>\n</div>\n</body>\n</html>\n'
```

##### url

响应的最终url位置。

```shell
>>> r.url
'https://www.example.com/'
```

### 会话

#### Session

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

##### auth

默认的认证元组或对象。

##### cert

默认的SSL客户证书。

##### cookies

一个CookieJar对象，包含了当前会话设定的所有cookies。

##### request(), delete(), get(),  head(), options(), patch(), post(), put()

发送请求。

##### headers

设定请求头的字典。

##### max-redirects

允许的最大重定向次数。

##### params

查询字符串的字典。

##### proxies

将协议映射到代理url的字典，例如`{'http': 'foo.bar:3128'}`。

##### verify

默认的SSL认证。

### 身份认证

| 类                    | 描述                   |
| --------------------- | ---------------------- |
| `auth.AuthBase`       | 所有身份认证类的基类   |
| `auth.HTTPBasicAuth`  | 请求附加的HTTP基本认证 |
| `auth.HTTPProxyAuth`  | 请求附加的HTTP代理认证 |
| `auth.HTTPDigestAuth` | 请求附加的HTTP摘要认证 |

### Cookie

### 异常

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

## PyYAML (yaml)

PyYAML 是一个 YAML 编码和解码器，使用方法类似于标准库的 json 包。

### load()

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

### dump()

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

