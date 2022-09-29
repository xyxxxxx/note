# os——多种操作系统接口

`os` 模块提供了一种使用与操作系统相关的功能的便捷式途径。

!!! note "注意"
    如果使用无效或无法访问的文件名与路径，或者其他类型正确但操作系统不接受的参数，此模块的所有函数都会抛出 `OSError` 或者它的子类。

## 进程参数

这些函数和数据项提供了操作当前进程和用户的信息。

### environ

一个表示字符串环境的映射对象。

这个映射是在第一次导入 `os` 模块时捕获的，通常作为 Python 启动过程中处理 `site.py` 的一部分。在此之后对环境所做的更改将不会反映在 `os.environ` 中，除了通过直接修改 `os.environ` 所做的更改。

如果平台支持 `putenv()` 函数，这个映射除了查询环境外还可以用于修改环境。当这个映射被修改时，`putenv()` 将被自动调用。

!!! tip "提示"
    直接调用 `putenv()` 并不会更新 `os.environ`，所以推荐直接修改 `os.environ`。

如果平台支持 `unsetenv()` 函数，你可以通过删除映射中项的方式来删除对应的环境变量。当一个元素被从 `os.environ` 删除时，以及 `pop()` 或 `clear()` 被调用时，`unsetenv()` 将被自动调用。

```python
>>> os.environ
environ({'SHELL': '/bin/zsh', ...
>>> os.environ['HOME']           # 查询环境变量
'/Users/xyx'
>>> os.environ['MYENV'] = '1'    # 添加环境变量
>>> os.environ['MYENV']
'1'
>>> del os.environ['MYENV']      # 删除环境变量
```

### fspath()

返回路径的文件系统表示。

如果传入的是 `str` 或 `bytes` 类型的字符串，则原样返回；否则 `__fspath__()` 将被调用，如果得到的是一个 `str` 或 `bytes` 类型的对象，那就返回这个值。其他所有情况会引发 `TypeError` 异常。

### getenv()

```python
os.getenv(key, default=None)
```

如果存在，返回环境变量 *key* 的值，否则返回 *default*。*key*、*default* 和返回值均为 `str` 类型。

可用性：大部分 Unix 系统，Windows。

### getgid()

返回当前进程的实际组 ID。

可用性: Unix。

### getpid()

返回当前进程 ID。

### getppid()

返回父进程 ID。当父进程已经退出，在 Unix 上返回的 ID 是初始进程(1)中的一个，在 Windows 上仍然是同一个进程 ID，该进程 ID 可能已经被另一个进程所重用。

可用性: Unix, Windows。

### getuid()

返回当前进程的真实用户 ID。

可用性: Unix。

### putenv()

```python
os.putenv(key, value)
```

将名为 *key* 的环境变量设为 *value*。该环境变量修改会影响由 `os.system()`、`popen()`、`fork()` 和 `execv()` 启动的子进程。

!!! note "注意"
    在一些平台上，包括 FreeBSD 和 Mac OS X，设置 `environ` 可能导致内存泄露。请参考关于 putenv 的系统文档。

当系统支持 `putenv()` 时，对 `os.environ` 中项的赋值会自动转换为对 `putenv()` 的调用。

!!! tip "提示"
    直接调用 `putenv()` 并不会更新 `os.environ`，所以推荐直接修改 `os.environ`。

可用性：大部分 Unix 系统，Windows。

### setgid()

设置当前进程的组 ID。

可用性: Unix。

### setuid()

设置当前进程的用户 ID。

可用性: Unix。

### uname()

返回当前操作系统的识别信息。返回值是一个有 5 个属性的对象：

* `sysname`：操作系统名
* `nodename`：机器的网络名称（由实现定义）
* `release`：操作系统发行信息
* `version`：操作系统版本
* `machine`：硬件标识符

为了向后兼容，该对象也是可迭代的，像是一个按照 `sysname`，`nodename`，`release`、`version` 和 `machine` 顺序组成的五元组。

有些系统会将 `nodename` 截断为 8 个字符或截断至前缀部分；获取主机名的一个更好方式是 `socket.gethostname()` 甚至 `socket.gethostbyaddr(socket.gethostname())`。

可用性：较新的 Unix 版本。

### unsetenv()

```python
os.unsetenv(key)
```

取消设置（删除）名为 *key* 的环境变量。该环境变量修改会影响由 `os.system()`、`popen()`、`fork()` 和 `execv()` 启动的子进程。

当系统支持 `unsetenv()` 时，对 `os.environ` 中项的删除会自动转换为对 `unsetenv()` 的调用。

!!! tip "提示"
    直接调用 `unsetenv()` 并不会更新 `os.environ`，所以推荐直接删除 `os.environ` 的项。

可用性：大部分 Unix 系统。

## 文件描述符操作

这些函数对文件描述符所引用的 I/O 流进行操作。

文件描述符是一些小的整数，对应于当前进程所打开的文件。例如，标准输入的文件描述符通常是 0，标准输出是 1，标准错误是 2。之后被进程打开的文件的文件描述符会被依次指定为 3、4、5 等。在 Unix 平台上，套接字和管道也被文件描述符所引用。

当需要时，可以使用 `fileno()` 方法来获得文件对象所对应的文件描述符。需要注意的是，直接使用文件描述符会绕过文件对象的方法，忽略如数据内部缓冲等方面。

### close()

```python
os.close(fd)
```

关闭文件描述符 *fd*。

!!! tip "提示"
    此功能被设计用于低级 I/O 操作，必须用于 `os.open()` 或 `pipe()` 返回的文件描述符。若要关闭由内建函数 `open()`、`popen()` 或 `fdopen()` 返回的“文件对象”，则应使用它的 `close()` 方法。

### closerange()

### isatty()

```python
os.isatty(fd)
```

如果文件描述符 *fd* 打开且已连接至 tty 设备（或类 tty 设备），返回 True，否则返回 False。

### open()

### pipe()

创建一个管道，返回一对分别用于读取和写入的文件描述符 `(r, w)`。新的文件描述符是不可继承的。

可用性: Unix, Windows。

### pread()

```python
os.pread(fd, n, offset)
```

从文件描述符 *fd* 所指向文件的偏移位置 *offset* 开始，读取最多 *n* 个字节，而保持文件偏移量不变。

返回包含读取字节的字节串。如果 *fd* 指向的文件已经到达 EOF，则返回空字节对象。

可用性: Unix。

### pwrite()

```python
os.pwrite(fd, str, offset)
```

将 *str* 中的字节串写入到文件描述符 *fd* 所指向文件的偏移位置 *offset* 处，而保持文件偏移量不变。

返回实际写入的字节数。

可用性: Unix。

### read()

```python
os.read(fd, n)
```

从文件描述符 *fd* 读取最多 *n* 个字节。

返回包含读取字节的字节串。如果 *fd* 指向的文件已经到达 EOF，则返回空字节对象。

!!! note "注意"
    此功能被设计用于低级 I/O 操作，必须用于 `os.open()` 或 `pipe()` 返回的文件描述符。若要读取由内建函数 `open()`、`popen()` 或 `fdopen()` 返回的“文件对象”，则应使用它的 `read()` 或 `readline()` 方法。

### write()

```python
os.write(fd, str)
```

将 *str* 中的字节串写入文件描述符 *fd*。

返回实际写入的字节数。

!!! note "注意"
    此功能被设计用于低级 I/O 操作，必须用于 `os.open()` 或 `pipe()` 返回的文件描述符。若要写入由内建函数 `open()`、`popen()` 或 `fdopen()` 返回的“文件对象”，`sys.stdout` 或 `sys.stderr`，则应使用它的 `write()` 方法。

## 文件和目录

* 指定文件描述符
* 指定基于目录描述符的相对路径
* 不跟踪符号链接

### access()

```python
access(path, mode, *, dir_fd=None, effective_ids=False, follow_symlinks=True)
```

使用实际用户 ID 或用户组ID 测试对 *path* 的访问。

本函数支持指定基于目录描述符的相对路径和不跟踪符号链接。

### F_OK, R_OK, W_OK, X_OK

作为 `access()` 的 *mode* 参数的可选值，分别测试 *path* 的存在性、可读性、可写性和可执行性。

### chdir()

```python
chdir(path)
```

切换当前工作目录为 *path*。

本函数支持指定文件描述符为参数。其中描述符必须指向打开的目录，不能是打开的文件。

本函数可以引发 `OSError` 及其子类的异常，如 `FileNotFoundError`、`PermissionError` 和 `NotADirectoryError`。

### chmod()

```python
chmod(path, mode, *, dir_fd=None, follow_symlinks=True)
```

将 *path* 的模式更改为其他由数字表示的 *mode*。*mode* 可以用以下值之一，也可以将它们按位或组合起来（以下值在 `stat` 模块中定义）：

* stat.S_ISUID
* stat.S_ISGID
* stat.S_ENFMT
* stat.S_ISVTX
* stat.S_IREAD
* stat.S_IWRITE
* stat.S_IEXEC
* stat.S_IRWXU
* stat.S_IRUSR
* stat.S_IWUSR
* stat.S_IXUSR
* stat.S_IRWXG
* stat.S_IRGRP
* stat.S_IWGRP
* stat.S_IXGRP
* stat.S_IRWXO
* stat.S_IROTH
* stat.S_IWOTH
* stat.S_IXOTH

本函数支持指定文件描述符、指定基于目录描述符的相对路径和不跟踪符号链接。

### chown()

```python
chown(path, uid, gid, *, dir_fd=None, follow_symlinks=True)
```

将 *path* 的用户和组 ID 分别修改为数字形式的 *uid* 和 *gid*。若要使其中某个 ID 保持不变，将其置为 -1。

本函数支持指定文件描述符、指定基于目录描述符的相对路径和不跟踪符号链接。

参见更高阶的函数 `shutil.chown()`，除了数字 ID 之外，它还接受名称。

### getcwd()

返回当前工作目录的路径。

```python
>>> os.getcwd()
'/Users/xyx'
```

### link()

### listdir()

返回指定目录下各项目名称组成的列表，该列表按任意顺序排列，且不包括特殊项目 `.` 和 `..`。

```python
>>> os.listdir()
['dir1', 'dir2', 'file1', 'file2']
```

### makedirs()

```python
os.makedirs(name, mode=0o777, exist_ok=False)
```

递归地创建目录。与 `mkdir()` 类似，但会自动创建到达最后一级目录所需要的中间目录。

*mode* 参数会传递给 `mkdir()`，用来创建最后一级目录，对于该参数的解释请参阅 `mkdir()` 中的描述。要设置某些新建的父目录的权限，可以在调用 `makedirs()` 之前设置 umask。现有父目录的权限不会更改。

如果 *exist_ok* 为 False 且目标目录已存在，则引发 `FileExistsError`。

### mkdir()

```python
os.mkdir(path, mode=0o777, *, dir_fd=None)
```

创建一个名为 *path* 的目录，应用以数字表示的权限模式 *mode*。

如果目录已存在，则引发 `FileExistsError` 异常。

本函数支持基于目录描述符的相对路径。

要创建临时目录，请使用 `tempfile` 模块的 `tempfile.mkdtemp()` 函数；要递归地创建目录（一次性创建多级目录），请使用 `makedirs()`。

### readlink()

```python
os.readlink(path, *, dir_fd=None)
```

返回一个字符串，代表符号链接指向的实际路径。其结果可能是绝对或相对路径；如果是相对路径，可以通过 `os.path.join(os.path.dirname(path), result)` 将其转换为绝对路径。

如果 *path* 是字符串对象（直接或通过 `PathLike` 接口间接传入），则结果也是字符串对象，并且此调用可能引发 `UnicodeDecodeError`。如果 *path* 是字节对象（直接或间接传入），则结果也是字节对象。

本函数支持基于目录描述符的相对路径。

当符号链接可能指向另一个符号链接时，请改用 `realpath()` 以正确处理递归和平台差异。

```python
>>> os.symlink('file', 'ln1')
>>> os.symlink('ln1', 'ln2')
>>> os.readlink('ln2')
'ln1'
>>> os.readlink('ln1')
'file'
>>> os.path.realpath('ln2')
'/Users/xyx/Codes/test/file'
```

可用性: Unix, Windows。

### remove()

```python
os.remove(path, *, dir_fd=None)
```

删除文件 *path*。若文件不存在，则引发 `FileNotFoundError` 异常；若路径指向目录，则引发 `IsADirectoryError` 异常。

### removedirs()

### rename()

```python
os.rename(src, dst, *, src_dir_fd=None, dst_dir_fd=None)
```

将文件或目录 *src* 重命名为 *dst*。若 *dst* 已存在，则下列情况下操作将会失败，并引发 `OSError` 的子类：

* 在 Windows 上，引发 `FileExistsError` 异常
* 在 Unix 上，若 *src* 是文件而 *dst* 是目录，将抛出 `IsADirectoryError` 异常，反之则抛出 `NotADirectoryError` 异常；若两者都是目录且 *dst* 为空，则 *dst* 将被静默替换；若 *dst* 是非空目录，则抛出 `OSError` 异常；若两者都是文件，则在用户具有权限的情况下，将对 *dst* 进行静默替换；若 *src* 和 *dst* 在不同的文件系统上，则本操作在某些 Unix 分支上可能会失败。

### rmdir()

删除指定目录。若目录不存在，则引发 `FileNotFoundError` 异常；若目录不为空，则引发 `OSError` 异常。若要删除整个目录树，请使用 `shutil.rmtree()`。

```python
>>> os.rmdir('dir1')
```

### stat()

### symlink()

```python
os.symlink(src, dst, target_is_directory=False, *, dir_fd=None)
```

创建一个指向 *src* 的名为 *dst* 的符号链接。

本函数支持基于目录描述符的相对路径。

可用性：Unix，Windows。

### walk()

遍历目录。对于以 `top` 为根的目录树中的每个目录（包括 `top` 本身）都生成一个三元组 `(dirpath,dirnames,filenames)`。

```python
os.walk(top, topdown=True, onerror=None, followlinks=False)
# top      目录树的根目录
# topdown  若为True,则自上而下遍历;若为False,则自下而上遍历
```

```python
# test/
#     file1
#     file2
#     format_correction.py
#     dir1/
#         file3
#         file4
#     dir2/
#         file5
#         file6

>>> for root, dirs, files in os.walk('.'):
...     print(root)
...     print(dirs)
...     print(files)
... 
.                                             # 代表根目录
['dir2', 'dir1']                              # 此目录下的目录
['file2', 'format_correction.py', 'file1']    # 此目录下的文件
./dir2
[]
['file5', 'file6']
./dir1
[]
['file3', 'file4']
>>>
>>> for root, _, files in os.walk('.'):
...     for f in files:
...         path = os.path.join(root, f)      # 返回所有文件的相对路径
...         print(path)
... 
./file2
./format_correction.py
./file1
./dir2/file5
./dir2/file6
./dir1/file3
./dir1/file4
>>>
>>> for root, _, files in os.walk('.', topdown=False):    # 自下而上遍历
...     for f in files:
...         path = os.path.join(root, f)
...         print(path)
... 
./dir2/file5
./dir2/file6
./dir1/file3
./dir1/file4
./file2
./format_correction.py
./file1

```

## 进程管理

### fork()

分叉出一个子进程，在子进程中返回 `0`，在父进程中返回子进程的进程号。

### kill()

```python
os.kill(pid, sig)
```

将信号 *sig* 发送至进程 *pid*。

### system()

创建一个 Shell 子进程并执行指定命令（一个字符串），执行命令过程中产生的任何输出都将发送到解释器的标准输出流。

```python
>>> os.system('pwd')
/Users/xyx
```

### times()

返回当前的全局进程时间。返回值是一个有 5 个属性的对象：

* `user`：用户时间
* `system`：系统时间
* `children_user`：所有子进程的用户时间
* `children_system`：所有子进程的系统时间
* `elapsed`：从过去的固定时间点起，经过的真实时间

在 Unix 上请参阅 [times(2)](https://manpages.debian.org/times(2)) 和 [times(3)](https://manpages.debian.org/times(3)) 手册页。

可用性：Unix，Windows。

### wait()

等待子进程执行完毕，返回一个元组，包含其 pid 和退出状态指示——一个 16 位数字，其低字节是终止该进程的信号编号，高字节是退出状态码（信号编号为零的情况下）。

可用性：Unix。

### waitpid()

```python
waitpid(pid, options)
```

本函数的细节在 Unix 和 Windows 上有所不同。

在 Unix 上：等待进程号为 *pid* 的子进程执行完毕，返回一个元组，包含其进程 ID 和退出状态指示（编码与 `wait()` 相同）。调用的语义受整数 *options* 的影响，常规操作下该值应为 `0`。

如果 *pid* 大于 0，则 `waitpid()` 会获取该指定进程的状态信息。如果 pid 为 0，则获取当前进程所在进程组中的所有子进程的状态。如果 pid 为 -1，则获取当前进程的子进程状态。如果 pid 小于 -1，则获取进程组 -pid （ pid 的绝对值）中所有进程的状态。

当系统调用返回 -1 时，将抛出带有错误码的 OSError 异常。

在 Windows 上：……
