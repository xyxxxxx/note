# pathlib——面向对象的文件系统路径

`pathlib` 模块提供表示文件系统路径的类，其语义适用于不同的操作系统。路径类被分为提供纯计算操作而没有 I/O 的**纯路径**，以及从纯路径继承而来但提供 I/O 操作的**具体路径**。

![pathlib-inheritance](https://docs.python.org/zh-cn/3.8/_images/pathlib-inheritance.png)

## 纯路径

### PurePath

### PurePosixPath

### PureWindowsPath

## 具体路径

### Path

#### chmod()

修改文件的模式和权限，和 `os.chmod()` 相同。

#### cwd()

（类方法）返回一个表示当前目录的路径对象，和 `os.getcwd()` 相同。

```shell
>>> Path.cwd()
PosixPath('/Users/xyx/Codes/gitbooks/note')
```

#### exists()

若路径指向一个已存在的文件或目录，返回 `True`，和 `os.path.exists()` 相同。

#### expanduser()

返回展开了 `~` 和 `~user` 的路径对象，和 `os.path.expanduser()` 相同:

```python
>>> p = PosixPath('~/Codes')
>>> p.expanduser()
PosixPath('/Users/xyx/Codes')
```

#### glob()

解析相对于此路径的通配符模式，产生所有匹配的文件:

```python
>>> sorted(Path('.').glob('*.py'))
[PosixPath('pathlib.py'), PosixPath('setup.py'), PosixPath('test_pathlib.py')]
>>> sorted(Path('.').glob('*/*.py'))
[PosixPath('docs/conf.py')]

>>> sorted(Path('.').glob('**/*.py'))
[PosixPath('build/lib/pathlib.py'),
 PosixPath('docs/conf.py'),
 PosixPath('pathlib.py'),
 PosixPath('setup.py'),
 PosixPath('test_pathlib.py')]
```

#### group()

返回拥有此文件的用户组。如果文件的 GID 无法在系统数据库中找到，将引发 `KeyError`。

#### home()

（类方法）返回一个表示用户 Home 目录的路径对象，和 `os.path.expanduser()` 传入 `'~'` 路径返回的相同。

```shell
>>> Path.home()
PosixPath('/Users/xyx')
```

#### is_dir()

若路径指向一个目录，返回 `True`。

#### is_file()

若路径指向一个正常文件，返回 `True`。

#### mkdir()

```python
Path.mkdir(mode=0o777, parents=False, exist_ok=False)
```

新建给定路径的目录。如果给出了 *mode*，它将与当前进程的 `umask` 值合并来决定文件模式和访问标志。如果路径已经存在，则引发 `FileExistsError`。

若 *parents* 为 `True`，任何找不到的父目录都会伴随着此路径被创建；它们会以默认权限被创建，而不考虑 *mode* 设置（模仿 POSIX `mkdir -p` 命令）；若为 `False`（默认），则找不到的父级目录会导致引发 `FileNotFoundError`。

若 *exist_ok* 为 `False`（默认），则在目标已存在的情况下引发 `FileExistsError`。若为 `True`，则 `FileExistsError` 异常将被忽略（和 POSIX `mkdir -p` 命令的行为相同），但是只有在最后一个路径组件不是现存的非目录文件时才生效。

#### open()

打开路径指向的文件，和内置函数 `open()` 相同。

#### owner()

返回拥有此文件的用户名。如果文件的 UID 无法在系统数据库中找到，则引发 `KeyError`。

#### read_bytes()

以字节对象的形式返回路径指向的文件的二进制内容:

```python
>>> p = Path('my_binary_file')
>>> p.write_bytes(b'Binary file contents')
20
>>> p.read_bytes()
b'Binary file contents'
```

#### read_text()

```python
Path.read_text(encoding=None, errors=None)
```

以字符串形式返回路径指向的文件的解码后文本内容。

```python
>>> p = Path('my_text_file')
>>> p.write_text('Text file contents')
18
>>> p.read_text()
'Text file contents'
```

文件先被打开然后被关闭。可选形参的含义与内置函数 `open()` 相同。

#### rename()

```python
Path.rename(target)
```

将文件或目录重命名为 *target*，并返回一个新的指向 *target* 的 Path 实例。在 Unix 上，如果 *target* 存在且为一个文件，如果用户有足够权限，则它将被静默地替换。*target* 可以是一个字符串或者另一个路径对象:

```python
>>> p = Path('foo')
>>> p.open('w').write('some text')
9
>>> target = Path('bar')
>>> p.rename(target)
PosixPath('bar')
>>> target.open().read()
'some text'
```

目标路径可以是绝对路径或相对路径。相对路径会被解释为相对于当前工作目录，而**不是**相对于 Path 对象的目录。

#### write_bytes()

```python
Path.write_bytes(data)
```

将文件以二进制模式打开，写入 *data* 并关闭:

```python
>>> p = Path('my_binary_file')
>>> p.write_bytes(b'Binary file contents')
20
>>> p.read_bytes()
b'Binary file contents'
```

同名的现存文件会被覆盖。

#### write_text()

```python
Path.write_text(data, encoding=None, errors=None)
```

将文件以文本模式打开，写入 *data* 并关闭:

```python
>>> p = Path('my_text_file')
>>> p.write_text('Text file contents')
18
>>> p.read_text()
'Text file contents'
```

同名的现存文件会被覆盖。可选形参的含义与内置函数 `open()` 相同。

## 与 `os` 模块对应的函数

下表展示了 `os` 与 `pathlib` 模块相对应的函数。

| os                                                                                                                                                         | pathlib                                                                                                                                                                                                                                                                             |
| :--------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`os.path.abspath()`](https://docs.python.org/zh-cn/3.8/library/os.path.html#os.path.abspath)                                                              | [`Path.resolve()`](https://docs.python.org/zh-cn/3.8/library/pathlib.html#pathlib.Path.resolve)                                                                                                                                                                                     |
| [`os.chmod()`](https://docs.python.org/zh-cn/3.8/library/os.html#os.chmod)                                                                                 | [`Path.chmod()`](https://docs.python.org/zh-cn/3.8/library/pathlib.html#pathlib.Path.chmod)                                                                                                                                                                                         |
| [`os.mkdir()`](https://docs.python.org/zh-cn/3.8/library/os.html#os.mkdir)                                                                                 | [`Path.mkdir()`](https://docs.python.org/zh-cn/3.8/library/pathlib.html#pathlib.Path.mkdir)                                                                                                                                                                                         |
| [`os.rename()`](https://docs.python.org/zh-cn/3.8/library/os.html#os.rename)                                                                               | [`Path.rename()`](https://docs.python.org/zh-cn/3.8/library/pathlib.html#pathlib.Path.rename)                                                                                                                                                                                       |
| [`os.replace()`](https://docs.python.org/zh-cn/3.8/library/os.html#os.replace)                                                                             | [`Path.replace()`](https://docs.python.org/zh-cn/3.8/library/pathlib.html#pathlib.Path.replace)                                                                                                                                                                                     |
| [`os.rmdir()`](https://docs.python.org/zh-cn/3.8/library/os.html#os.rmdir)                                                                                 | [`Path.rmdir()`](https://docs.python.org/zh-cn/3.8/library/pathlib.html#pathlib.Path.rmdir)                                                                                                                                                                                         |
| [`os.remove()`](https://docs.python.org/zh-cn/3.8/library/os.html#os.remove), [`os.unlink()`](https://docs.python.org/zh-cn/3.8/library/os.html#os.unlink) | [`Path.unlink()`](https://docs.python.org/zh-cn/3.8/library/pathlib.html#pathlib.Path.unlink)                                                                                                                                                                                       |
| [`os.getcwd()`](https://docs.python.org/zh-cn/3.8/library/os.html#os.getcwd)                                                                               | [`Path.cwd()`](https://docs.python.org/zh-cn/3.8/library/pathlib.html#pathlib.Path.cwd)                                                                                                                                                                                             |
| [`os.path.exists()`](https://docs.python.org/zh-cn/3.8/library/os.path.html#os.path.exists)                                                                | [`Path.exists()`](https://docs.python.org/zh-cn/3.8/library/pathlib.html#pathlib.Path.exists)                                                                                                                                                                                       |
| [`os.path.expanduser()`](https://docs.python.org/zh-cn/3.8/library/os.path.html#os.path.expanduser)                                                        | [`Path.expanduser()`](https://docs.python.org/zh-cn/3.8/library/pathlib.html#pathlib.Path.expanduser) 和 [`Path.home()`](https://docs.python.org/zh-cn/3.8/library/pathlib.html#pathlib.Path.home)                                                                                  |
| [`os.listdir()`](https://docs.python.org/zh-cn/3.8/library/os.html#os.listdir)                                                                             | [`Path.iterdir()`](https://docs.python.org/zh-cn/3.8/library/pathlib.html#pathlib.Path.iterdir)                                                                                                                                                                                     |
| [`os.path.isdir()`](https://docs.python.org/zh-cn/3.8/library/os.path.html#os.path.isdir)                                                                  | [`Path.is_dir()`](https://docs.python.org/zh-cn/3.8/library/pathlib.html#pathlib.Path.is_dir)                                                                                                                                                                                       |
| [`os.path.isfile()`](https://docs.python.org/zh-cn/3.8/library/os.path.html#os.path.isfile)                                                                | [`Path.is_file()`](https://docs.python.org/zh-cn/3.8/library/pathlib.html#pathlib.Path.is_file)                                                                                                                                                                                     |
| [`os.path.islink()`](https://docs.python.org/zh-cn/3.8/library/os.path.html#os.path.islink)                                                                | [`Path.is_symlink()`](https://docs.python.org/zh-cn/3.8/library/pathlib.html#pathlib.Path.is_symlink)                                                                                                                                                                               |
| [`os.link()`](https://docs.python.org/zh-cn/3.8/library/os.html#os.link)                                                                                   | [`Path.link_to()`](https://docs.python.org/zh-cn/3.8/library/pathlib.html#pathlib.Path.link_to)                                                                                                                                                                                     |
| [`os.symlink()`](https://docs.python.org/zh-cn/3.8/library/os.html#os.symlink)                                                                             | [`Path.symlink_to()`](https://docs.python.org/zh-cn/3.8/library/pathlib.html#pathlib.Path.symlink_to)                                                                                                                                                                               |
| [`os.stat()`](https://docs.python.org/zh-cn/3.8/library/os.html#os.stat)                                                                                   | [`Path.stat()`](https://docs.python.org/zh-cn/3.8/library/pathlib.html#pathlib.Path.stat), [`Path.owner()`](https://docs.python.org/zh-cn/3.8/library/pathlib.html#pathlib.Path.owner), [`Path.group()`](https://docs.python.org/zh-cn/3.8/library/pathlib.html#pathlib.Path.group) |
| [`os.path.isabs()`](https://docs.python.org/zh-cn/3.8/library/os.path.html#os.path.isabs)                                                                  | [`PurePath.is_absolute()`](https://docs.python.org/zh-cn/3.8/library/pathlib.html#pathlib.PurePath.is_absolute)                                                                                                                                                                     |
| [`os.path.join()`](https://docs.python.org/zh-cn/3.8/library/os.path.html#os.path.join)                                                                    | [`PurePath.joinpath()`](https://docs.python.org/zh-cn/3.8/library/pathlib.html#pathlib.PurePath.joinpath)                                                                                                                                                                           |
| [`os.path.basename()`](https://docs.python.org/zh-cn/3.8/library/os.path.html#os.path.basename)                                                            | [`PurePath.name`](https://docs.python.org/zh-cn/3.8/library/pathlib.html#pathlib.PurePath.name)                                                                                                                                                                                     |
| [`os.path.dirname()`](https://docs.python.org/zh-cn/3.8/library/os.path.html#os.path.dirname)                                                              | [`PurePath.parent`](https://docs.python.org/zh-cn/3.8/library/pathlib.html#pathlib.PurePath.parent)                                                                                                                                                                                 |
| [`os.path.samefile()`](https://docs.python.org/zh-cn/3.8/library/os.path.html#os.path.samefile)                                                            | [`Path.samefile()`](https://docs.python.org/zh-cn/3.8/library/pathlib.html#pathlib.Path.samefile)                                                                                                                                                                                   |
| [`os.path.splitext()`](https://docs.python.org/zh-cn/3.8/library/os.path.html#os.path.splitext)                                                            | [`PurePath.suffix`](https://docs.python.org/zh-cn/3.8/library/pathlib.html#pathlib.PurePath.suffix)                                                                                                                                                                                 |
