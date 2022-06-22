# os.path——常用路径操作

## abspath()

返回路径的绝对路径。

```python
>>> path.abspath('.')
'/home/xyx'
```

## basename()

返回路径的基本名称，即将路径传入到 `split()` 函数所返回的元组的后一个元素。

```python
>>> path.basename('/Users/xyx')
'xyx'
>>> path.basename('/Users/xyx/Codes')
'Codes'
```

## dirname()

返回路径中的目录名称。

```python
>>> path.dirname('/Users/xyx')
'/Users'
>>> path.dirname('/Users/xyx/Codes')
'/Users/xyx'
```

## exists()

若路径指向一个已存在的文件或目录或已打开的文件描述符，返回 `True`。

```python
>>> path.exists('dir1')
```

## getsize()

返回路径指向的文件或目录的大小，以字节为单位。若文件或目录不存在或不可访问，则引发 `OSError` 异常。

```python
>>> path.getsize('file1')
14560
```

## isabs()

若路径是一个绝对路径，返回 `True`。在 Unix 上，绝对路径以 `/` 开始，而在 Windows 上，绝对路径可以是去掉驱动器号后以 `/` 或 `\` 开始。

## isdir()

若路径是现有的目录，返回 `True`。

## isfile()

若路径是现有的常规文件，返回 `True`。

## ismount()

若路径是挂载点，返回 `True`。

## join()

智能地拼接一个或多个路径部分。

```python
>>> path.join('/Users', 'xyx')
'/Users/xyx'
```

## split()

将路径拆分为两部分，以最后一个 `/` 为界。

```python
>>> path.split('/Users/xyx')
('/Users', 'xyx')
```

## splitdrive()

将路径拆分为两部分，其中前一部分是挂载点（对于 Windows 系统为驱动器盘符）或空字符串。

## splitext()

将路径拆分为两部分，其中后一部分是文件扩展名（以 `.` 开始并至多包含一个 `.`）或空字符串。

```python
>>> path.splitext('/path/to/foo.bar.exe')
('/path/to/foo.bar', '.exe')
```
