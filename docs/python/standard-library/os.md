# os——多种操作系统接口

`os` 模块提供了一种使用与操作系统相关的功能的便捷式途径。

!!! note "注意"
    如果使用无效或无法访问的文件名与路径，或者其他类型正确但操作系统不接受的参数，此模块的所有函数都会抛出 `OSError` 或者它的子类。

## 进程参数

### environ

进程的环境变量，可以直接操作该映射以查询或修改环境变量。

```python
>>> os.environ
environ({'SHELL': '/bin/zsh', ...
>>> os.environ['HOME']           # 查询环境变量
'/Users/xyx'
>>> os.environ['MYENV'] = '1'    # 添加环境变量
>>> os.environ['MYENV']
'1'
>>> del os.environ['MYENV']      # 删除环境变量
>>> os.environ['MYENV']
# KeyError: 'MYENV'
```

### fspath()

返回路径的文件系统表示。

如果传入的是 `str` 或 `bytes` 类型的字符串，则原样返回；否则 `__fspath__()` 将被调用，如果得到的是一个 `str` 或 `bytes` 类型的对象，那就返回这个值。其他所有情况会抛出 `TypeError` 异常。

### getenv()

获取环境变量的值。

```python
os.getenv(key, default=None)
# key        环境变量
# default    若环境变量不存在,返回此默认值
```

### getpid()

返回当前进程 ID。

## 文件描述符操作

### close()

### pipe()

### pread()

### pwrite()

### read()

### write()

## 文件和目录

### access()

### chdir()

切换当前工作目录为指定路径。

```python
>>> os.chdir('dir1')
```

### chmod()

### chown()

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
