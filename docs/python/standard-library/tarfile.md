# tarfile——读写 tar 归档文件

`tarfile` 模块可以用来读写 tar 归档，包括使用 gzip, bz2 和 lzma 压缩的归档。对于 `.zip` 文件，请使用 `zipfile` 模块来进行读写，或者使用 shutil 的高层级函数。

## open()

针对路径名返回 `TarFile` 对象。

```python
with tarfile.open('sample.tar') as tar:
    # read ops

with tarfile.open('sample.tar.gz') as tar:
    # read ops

with tarfile.open('sample.tar', 'w') as tar:
    # write ops
```

```python
source_file = 'dir'
with TemporaryFile() as tar_buffer:
    with tarfile.open(fileobj=tar_buffer, mode='w') as tar:
        tar.add(source_file, arcname='pvc/data/dir')
    tar_buffer.seek(0)

    command = f'tar xvf -'
    tar_cmd = subprocess.Popen(shlex.split(command),
                                stdin=subprocess.PIPE,
                                shell=False)
    tar_cmd.stdin.write(tar_buffer.read())
    tar_cmd.stdin.close()
```

## TarFile

TarFile 对象提供了一个 tar 归档的接口。tar 归档是数据块的序列；一个归档成员（被保存文件）是由一个标头块加多个数据块组成的；一个文件可以在一个 tar 归档中多次被保存；每个归档成员都由一个 TarInfo 对象来代表。

`TarFile` 对象可以在 `with` 语句中作为上下文管理器使用，当语句块结束时它将自动被关闭。请注意在发生异常事件时被打开用于写入的归档将不会被终结；只有内部使用的文件对象将被关闭。

### getmember()

根据指定名称返回成员的 `TarInfo` 对象。 如果名称在归档中找不到，则会引发 `KeyError`。

### getmembers()

以 `TarInfo` 对象列表的形式返回归档的成员，列表的顺序与归档中成员的顺序一致。

### getnames()

以名称列表的形式返回归档的成员，列表的顺序与 `getmembers()` 所返回列表的顺序一致。

### next()

当 `TarFile` 被打开用于读取时，以 `TarInfo` 对象的形式返回归档的下一个成员。如果不再有可用对象则返回 `None`。

### extract()

```python
TarFile.extract(member, path="", set_attrs=True, *, numeric_owner=False)
```

将归档中的一个成员提取到当前工作目录或 *path* 目录，将使用其完整名称。成员的文件信息会尽可能精确地被提取。*member* 可以是一个文件名或 `TarInfo` 对象。将会设置文件属性 (owner, mtime, mode) 除非 *set_attrs* 为 `False`。

如果 *numeric_owner* 为 True，则将使用来自 tarfile 的 uid 和 gid 数值来设置被提取文件的用户和用户组。在其他情况下，则会使用来自 tarfile 的名称值。

!!! warning "警告"
    绝不要未经预先检验就从不可靠的源中提取归档文件，这样有可能在 *path* 之外创建文件。例如某些成员具有以 `"/"` 开始的绝对路径文件名或带有两个点号 `".."` 的文件名。

### extractall()

```python
TarFile.extractall(path=".", members=None, *, numeric_owner=False)
```

将归档中的所有成员提取到当前工作目录或 *path* 目录。如果给定了可选的 *members*，则它必须为 `getmembers()` 所返回的列表的一个子集。字典信息例如所有者、修改时间和权限会在所有成员提取完毕后被设置，这样做是为了避免两个问题：目录的修改时间会在每当在其中创建文件时被重置。并且如果目录的权限不允许写入，提取文件到目录的操作将失败。

如果 numeric_owner 为 True，则将使用来自 tarfile 的 uid 和 gid 数值来设置被提取文件的所有者/用户组。 在其他情况下，则会使用来自 tarfile 的名称值。

### extractfile()

```python
TarFile.extractfile(member)
```

将归档中的一个成员提取为文件对象。*member* 可以是一个文件名或 `TarInfo` 对象。如果 *member* 是一个常规文件或链接，则会返回一个 `io.BufferedReader` 对象。在其他情况下将返回 `None`。

### add()

```python
TarFile.add(name, arcname=None, recursive=True, *, filter=None)
```

将文件 *name* 添加到归档，*name* 可以是任意类型的文件（或目录、fifo、符号链接等等）。如果给出 *arcname* 则它将为归档中的文件指定一个替代名称。默认情况下会递归地添加目录，这可以通过将 *recursive* 设为 `False` 来避免。递归操作会按排序顺序添加条目。如果给定了 *filter*，它应当为一个接受 `TarInfo` 对象并返回已修改 `TarInfo` 对象的函数。如果它返回 `None` 则 `TarInfo` 对象将从归档中被排除。

### addfile()

```python
TarFile.addfile(tarinfo, fileobj=None)
```

将 `TarInfo` 对象 *tarinfo* 添加到归档。如果给定了 *fileobj*，它应当是一个二进制文件，并会从中读取 `tarinfo.size` 个字节添加到归档。你可以直接创建 `TarInfo` 对象，或是使用 `gettarinfo()` 来创建。

### gettarinfo()

```python
TarFile.gettarinfo(name=None, arcname=None, fileobj=None)
```

## TarInfo

`TarInfo` 对象代表 `TarFile` 中的一个文件。除了会存储所有必要的文件属性（例如文件类型、大小、时间、权限、所有者等），它还提供了一些确定文件类型的有用方法。此对象**并不**包含文件数据本身。

`TarInfo` 对象可通过 `TarFile` 的方法 `getmember()`、`getmembers()` 和 `gettarinfo()` 返回。

### gid

最初保存该成员的用户的用户组 ID。

### gname

用户组名。

### isdir()

如果为目录则返回 `True`。

### isfile()

如果为普通文件则返回 `True`。

### islnk()

如果为硬链接则返回 `True`。

### issym()

如果为符号链接则返回 `True`。

### linkname

目标文件名的名称，该属性仅在类型为 `LNKTYPE` 和 `SYMTYPE` 的 `TarInfo` 对象中存在。

### mode

权限位。

### mtime

上次修改的时间。

### name

归档成员的名称。

### size

以字节表示的大小。

### type

文件类型。`type` 通常为以下常量之一：`REGTYPE`, `AREGTYPE`, `LNKTYPE`, `SYMTYPE`, `DIRTYPE`, `FIFOTYPE`, `CONTTYPE`, `CHRTYPE`, `BLKTYPE`, `GNUTYPE_SPARSE`。要更方便地确定一个 `TarInfo` 对象的类型，请使用 `is*()` 方法。

### uid

最初保存该成员的用户的用户 ID。

### uname

用户名。
