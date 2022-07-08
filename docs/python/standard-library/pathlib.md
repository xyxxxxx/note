# pathlib——面向对象的文件系统路径

`pathlib` 模块提供表示文件系统路径的类，其语义适用于不同的操作系统。路径类被分为提供纯计算操作而没有 I/O 的**纯路径**，以及从纯路径继承而来但提供 I/O 操作的**具体路径**。

![pathlib-inheritance](https://docs.python.org/zh-cn/3.8/_images/pathlib-inheritance.png)

## 纯路径

纯路径对象提供了不实际访问文件系统的路径处理操作。无论你正运行什么系统，你都可以实例化这些类，因为它们提供的操作不做任何系统调用。有三种方法来访问这些类：

### PurePath

```python
class pathlib.PurePath(*pathsegments)
```

一个通用的类，代表当前系统的路径风格（实例化为 `PurePosixPath` 或者 `PureWindowsPath`）：

```python
>>> PurePath('setup.py')      # Running on a Unix machine
PurePosixPath('setup.py')
```

每一个 *pathsegments* 的元素可以是一个代表路径片段的字符串，一个返回字符串的实现了 `os.PathLike` 接口的对象，或者另一个路径对象:

```python
>>> PurePath('foo', 'some/path', 'bar')
PurePosixPath('foo/some/path/bar')
>>> PurePath(Path('foo'), Path('bar'))
PurePosixPath('foo/bar')
```

当 *pathsegments* 为空时，假定为当前目录:

```python
>>> PurePath()
PurePosixPath('.')
```

当给出一些绝对路径时，最后一个将被当作锚（模仿 `os.path.join()` 的行为）:

```python
>>> PurePath('/etc', '/usr', 'lib64')
PurePosixPath('/usr/lib64')
>>> PureWindowsPath('c:/Windows', 'd:bar')
PureWindowsPath('d:bar')
```

但是，在 Windows 路径中，改变本地根目录并不会丢弃之前盘符的设置:

```python
>>> PureWindowsPath('c:/Windows', '/Program Files')
PureWindowsPath('c:/Program Files')
```

假斜线和单独的点都会被消除，但是双点 （`'..'`） 不会，以防改变符号链接的含义。

```python
>>> PurePath('foo//bar')
PurePosixPath('foo/bar')
>>> PurePath('foo/./bar')
PurePosixPath('foo/bar')
>>> PurePath('foo/../bar')
PurePosixPath('foo/../bar')
```

（一个很 naïve 的做法是让 `PurePosixPath('foo/../bar')` 等同于 `PurePosixPath('bar')`，但如果 `foo` 是一个指向其他目录的符号链接那么这个做法就将出错）。

### PurePosixPath

`PurePath` 的子类，路径风格不同于 Windows 文件系统:

```python
>>> PurePosixPath('/etc')
PurePosixPath('/etc')
```

*pathsegments* 参数的指定和 `PurePath` 相同。

### PureWindowsPath

`PurePath` 的子类，路径风格为 Windows 文件系统:

```python
>>> PureWindowsPath('c:/Program Files/')
PureWindowsPath('c:/Program Files')
```

*pathsegments* 参数的指定和 `PurePath` 相同。

### 运算

路径是不可变并且可哈希的。相同风格的路径可以排序与比较。这些性质尊重对应风格的大小写转换语义:

```python
>>> PurePosixPath('foo') == PurePosixPath('FOO')
False
>>> PureWindowsPath('foo') == PureWindowsPath('FOO')
True
>>> PureWindowsPath('FOO') in { PureWindowsPath('foo') }
True
>>> PureWindowsPath('C:') < PureWindowsPath('d:')
True
```

不同风格的路径比较得到不等的结果并且无法被排序:

```python
>>> PureWindowsPath('foo') == PurePosixPath('foo')
False
>>> PureWindowsPath('foo') < PurePosixPath('foo')
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: '<' not supported between instances of 'PureWindowsPath' and 'PurePosixPath'
```

斜杠 `/` 操作符有助于创建子路径，就像 `os.path.join()` 一样:

```python
>>> p = PurePath('/etc')
>>> p
PurePosixPath('/etc')
>>> p / 'init.d' / 'apache2'
PurePosixPath('/etc/init.d/apache2')
>>> q = PurePath('bin')
>>> '/usr' / q
PurePosixPath('/usr/bin')
```

文件对象可用于任何接受 `os.PathLike` 接口实现的地方。

```python
>>> import os
>>> p = PurePath('/etc')
>>> os.fspath(p)
'/etc'
```

路径的字符串表示为它自己原始的文件系统路径（以原生形式，例如在 Windows 下使用反斜杠）。你可以传递给任何需要字符串形式路径的函数。

```python
>>> p = PurePath('/etc')
>>> str(p)
'/etc'
>>> p = PureWindowsPath('c:/Program Files')
>>> str(p)
'c:\\Program Files'
```

类似地，在路径上调用 `bytes` 将原始文件系统路径作为字节对象给出，就像被 `os.fsencode()` 编码一样:

```python
>>> bytes(p)
b'/etc'
```

!!! note "注意"
    只推荐在 Unix 下调用 bytes。在 Windows， unicode 形式是文件系统路径的规范表示法。

### 方法和属性

#### drive

一个表示驱动器盘符或命名的字符串，如果存在:

```python
>>> PureWindowsPath('c:/Program Files/').drive
'c:'
>>> PureWindowsPath('/Program Files/').drive
''
>>> PurePosixPath('/etc').drive
''
```

UNC 分享也被认作是驱动器:

```python
>>> PureWindowsPath('//host/share/foo.txt').drive
'\\\\host\\share'
```

#### is_absolute()

返回路径是否为绝对路径。如果路径同时拥有驱动器符与根路径（如果风格允许）则将被认为是绝对路径。

```python
>>> PurePosixPath('/a/b').is_absolute()
True
>>> PurePosixPath('a/b').is_absolute()
False

>>> PureWindowsPath('c:/a/b').is_absolute()
True
>>> PureWindowsPath('/a/b').is_absolute()
False
>>> PureWindowsPath('c:').is_absolute()
False
>>> PureWindowsPath('//some/share').is_absolute()
True
```

#### match()

```python
PurePath.match(pattern)
```

将此路径与提供的通配符风格的模式匹配。如果匹配成功则返回 `True`，否则返回 `False`。

如果 *pattern* 是相对的，则路径可以是相对路径或绝对路径，并且匹配是从右侧完成的：

```python
>>> PurePath('a/b.py').match('*.py')
True
>>> PurePath('/a/b/c.py').match('b/*.py')
True
>>> PurePath('/a/b/c.py').match('a/*.py')
False
```

如果 *pattern* 是绝对的，则路径必须是绝对的，并且路径必须完全匹配:

```python
>>> PurePath('/a.py').match('/*.py')
True
>>> PurePath('a/b.py').match('/*.py')
False
```

#### name

表示路径最后成分的字符串，排除了驱动器与根目录，如果存在的话:

```python
>>> PurePosixPath('my/library/setup.py').name
'setup.py'
```

#### stem

去除后缀的最后一个成分:

```python
>>> PurePosixPath('my/library.tar.gz').stem
'library.tar'
>>> PurePosixPath('my/library.tar').stem
'library'
>>> PurePosixPath('my/library').stem
'library'
```

#### suffix

最后一个成分的文件扩展名，如果存在:

```python
>>> PurePosixPath('my/library/setup.py').suffix
'.py'
>>> PurePosixPath('my/library.tar.gz').suffix
'.gz'
>>> PurePosixPath('my/library').suffix
''
```

#### parts

一个元组，可以访问路径的多个成分：

```python
>>> p = PurePath('/usr/bin/python3')
>>> p.parts
('/', 'usr', 'bin', 'python3')

>>> p = PureWindowsPath('c:/Program Files/PSF')
>>> p.parts
('c:\\', 'Program Files', 'PSF')
```

#### parent

路径的逻辑父路径:

```python
>>> p = PurePosixPath('/a/b/c/d')
>>> p.parent
PurePosixPath('/a/b/c')
```

#### root

一个表示（本地或全局）根的字符串，如果存在:

```python
>>> PureWindowsPath('c:/Program Files/').root
'\\'
>>> PureWindowsPath('c:Program Files/').root
''
>>> PurePosixPath('/etc').root
'/'
```

UNC 分享一样拥有根:

```python
>>> PureWindowsPath('//host/share').root
'\\'
```

## 具体路径

具体路径是纯路径的子类。除了纯路径提供的操作之外，具体路径还提供了对路径对象进行系统调用的方法。有三种方法可以实例化具体路径：

### Path

```python
class pathlib.Path(*pathsegments)
```

`PurePath` 的子类，此类以当前系统的路径风格表示路径（实例化为 `PosixPath` 或 `WindowsPath`）:

```python
>>> Path('setup.py')
PosixPath('setup.py')
```

*pathsegments* 参数的指定和 `PurePath` 相同。

### PosixPath

```python
class pathlib.PosixPath(*pathsegments)
```

`Path` 和 `PurePosixPath` 的子类，此类表示一个非 Windows 文件系统的具体路径:

```python
>>> PosixPath('/etc')
PosixPath('/etc')
```

*pathsegments* 参数的指定和 `PurePath` 相同。

### WindowsPath

```python
class pathlib.WindowsPath(*pathsegments)
```

`Path` 和 `PureWindowsPath` 的子类，从类表示一个 Windows 文件系统的具体路径:

```python
>>> WindowsPath('c:/Program Files/')
WindowsPath('c:/Program Files')
```

*pathsegments* 参数的指定和 `PurePath` 相同。

你只能实例化与当前系统风格相同的类（允许系统调用作用于不兼容的路径风格可能在应用程序中导致缺陷或失败）:

```python
>>> import os
>>> os.name
'posix'
>>> Path('setup.py')
PosixPath('setup.py')
>>> PosixPath('setup.py')
PosixPath('setup.py')
>>> WindowsPath('setup.py')
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "pathlib.py", line 798, in __new__
    % (cls.__name__,))
NotImplementedError: cannot instantiate 'WindowsPath' on your system
```

### 方法

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

#### resolve()

```python
Path.resolve(strict=False)
```

绝对化路径，解析任何符号链接，返回一个新的路径对象:

```python
>>> p = Path()
>>> p
PosixPath('.')
>>> p.resolve()
PosixPath('/home/antoine/pathlib')

>>> p = Path('docs/../setup.py')
>>> p.resolve()
PosixPath('/home/antoine/pathlib/setup.py')
```

如果路径不存在并且 *strict* 设为 `True`，则引发 `FileNotFoundError`；如果 *strict* 为 `False`，则路径将被尽可能地解析并且任何剩余部分都会被不检查是否存在地追加。如果在解析路径上发生无限循环，则引发 `RuntimeError`。

#### rmdir()

移除此路径指向的目录。目录必须是空的。

#### touch()

```python
Path.touch(mode=0o666, exist_ok=True)
```

以给定路径创建文件。如果给出了 *mode*，它将与当前进程的 `umask` 值合并以确定文件的模式和访问标志。如果文件已经存在，则当 *exist_ok* 为 True 时函数仍会成功（并且将它的修改事件更新为当前事件），否则引发 `FileExistsError`。

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
