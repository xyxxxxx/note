# os——多种操作系统接口

## chdir()

切换当前工作目录为指定路径。

```python
>>> os.chdir('dir1')
```

## environ

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

## fork()

分叉出一个子进程，在子进程中返回 `0`，在父进程中返回子进程的进程号。

## getcwd()

返回当前工作目录的路径。

```python
>>> os.getcwd()
'/Users/xyx'
```

## getenv()

获取环境变量的值。

```python
os.getenv(key, default=None)
# key        环境变量
# default    若环境变量不存在,返回此默认值
```

## kill()

```python
os.kill(pid, sig)
```

将信号 *sig* 发送至进程 *pid*。

## listdir()

返回指定目录下各项目名称组成的列表，该列表按任意顺序排列，且不包括特殊项目 `.` 和 `..`。

```python
>>> os.listdir()
['dir1', 'dir2', 'file1', 'file2']
```

## makedirs()

递归地创建指定名称和权限的目录。与 `mkdir()` 类似，但会自动创建到达最后一级目录所需要的中间目录。

```python
>>> os.mkdir('dir1/dir2', mode=0o755)
```

## mkdir()

创建指定名称和权限的目录。若目录已存在，则引发 `FileExistsError` 异常。

```python
>>> os.mkdir('dir1', mode=0o755)
```

要递归地创建目录（一次创建多级目录），请使用 `makedirs()`。

## remove()

删除指定文件。若文件不存在，则引发 `FileNotFoundError` 异常；若路径指向目录，则引发 `IsADirectoryError` 异常。

```python
>>> os.remove('file1')
```

## rename()

```python
os.rename(src, dst, *, src_dir_fd=None, dst_dir_fd=None)
```

将文件或目录 *src* 重命名为 *dst*。若 *dst* 已存在，则下列情况下操作将会失败，并引发 `OSError` 的子类：

* 在 Windows 上，引发 `FileExistsError` 异常
* 在 Unix 上，若 *src* 是文件而 *dst* 是目录，将抛出 `IsADirectoryError` 异常，反之则抛出 `NotADirectoryError` 异常；若两者都是目录且 *dst* 为空，则 *dst* 将被静默替换；若 *dst* 是非空目录，则抛出 `OSError` 异常；若两者都是文件，则在用户具有权限的情况下，将对 *dst* 进行静默替换；若 *src* 和 *dst* 在不同的文件系统上，则本操作在某些 Unix 分支上可能会失败。

## rmdir()

删除指定目录。若目录不存在，则引发 `FileNotFoundError` 异常；若目录不为空，则引发 `OSError` 异常。若要删除整个目录树，请使用 `shutil.rmtree()`。

```python
>>> os.rmdir('dir1')
```

## system()

创建一个 Shell 子进程并执行指定命令（一个字符串），执行命令过程中产生的任何输出都将发送到解释器的标准输出流。

```python
>>> os.system('pwd')
/Users/xyx
```

## times()

返回当前的全局进程时间。

## wait()

等待子进程执行完毕，返回一个元组，包含其 pid 和退出状态指示——一个 16 位数字，其低字节是终止该进程的信号编号，高字节是退出状态码（信号编号为零的情况下）。

## waitid()

## waitpid()

 

## walk()

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

>>> import os
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
