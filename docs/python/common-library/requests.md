# requests

[requests](https://requests.readthedocs.io/en/latest/) 是一个简单而优雅的 HTTP 库。

## 使用示例

## 接口

requests 所有的功能都可以通过以下 7 个方法访问，它们都会返回一个 `Response` 对象的实例。

### delete()

发送 `DELETE` 请求。

### get()

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

### head()

发送 `HEAD` 请求。

### options()

发送 `OPTIONS` 请求。

### post()

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

### put()

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

### request()

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
