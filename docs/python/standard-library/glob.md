# glob——Unix 风格路径名模式扩展

`glob` 模块可根据 Unix 终端所用规则找出所有匹配特定模式的路径名，但会按不确定的顺序返回结果。波浪号扩展不会生效，但 `*`、`?` 以及表示为 `[]` 的字符范围将被正确地匹配。这项功能是通过配合使用 `os.scandir()` 和 `fnmatch.fnmatch()` 函数来实现的，而不是通过实际发起调用子终端。

## glob()

```python
glob(pathname, *, recursive=False)
```

返回匹配 *pathname* 的可能为空的路径名列表，其中的元素为包含路径信息的字符串。*pathname* 可以是绝对路径 (如 `/usr/src/Python-1.5/Makefile`) 或相对路径 (如 `../../Tools/*/*.gif`)，并且可包含 shell 风格的通配符。结果也将包含无效的符号链接（与在 shell 中一样）。结果是否排序取决于具体的文件系统。如果某个符合条件的文件在调用此函数期间被移除或添加，是否包括该文件的路径是没有规定的。

如果 *recursive* 为 `True`，则模式 `**` 将匹配目录中的任何文件以及零个或多个目录、子目录和符号链接。如果模式加了一个 `os.sep` 或 `os.altsep` 则将不匹配文件。

```python
>>> from glob import glob
>>> glob('*.png')
['2.png', '1.png', '0.png']
>>> glob('**/*.png', recursive=True)
['2.png', '1.png', '0.png', 'images/b.png', 'images/a.png']
```
