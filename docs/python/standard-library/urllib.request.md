# urllib.request——用于打开 URL 的可扩展库

## urlretrieve()

将 URL 形式的网络对象复制为本地文件。返回值为元组 `(filename, headers)` ，其中 *filename* 是保存网络对象的本地文件名， *headers* 是由 `urlopen()` 返回的远程对象 `info()` 方法的调用结果。可能触发的异常与 `urlopen()` 相同。

```python
>>> import urllib.request
>>> url, filename = "https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg"
>>> urllib.request.urlretrieve(url, filename)
```
