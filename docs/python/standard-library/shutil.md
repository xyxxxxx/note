# shutil——高阶文件操作

`shutil` 模块提供了一系列对文件和文件集合的高阶操作，特别是一些支持文件复制和删除的函数。

## copy()

将一个文件复制到目标位置并返回目标位置。

```python
shutil.copy(src, dst, *, follow_symlinks=True)
# src            要复制的文件
# dst            目标路径
#                若`dst`不存在,则文件将复制到此路径;若`dst`是已存在的目录,则文件将使用原文件名复制到此目录中;
#                若`dst`是已存在的文件,则此文件将被覆盖
```

## copytree()

将一个目录树复制到目标位置并返回目标位置。

```python
shutil.copytree(src, dst, symlinks=False, ignore=None, copy_function=copy2, 
ignore_dangling_symlinks=False, dirs_exist_ok=False)
# src            要复制的目录
# dst            目标路径
```

```python

```

## disk_usage()

返回给定路径所在磁盘的使用统计数据，形式为一个命名的元组，*total*、*used* 和 *free* 属性分别表示总计、已使用和未使用空间的字节数。

```python
>>> shutil.disk_usage('.')
usage(total=499963174912, used=107589382144, free=360688713728)
```

## move()

将一个文件或目录树移动到目标位置并返回目标位置。

```python
shutil.move(src, dst, copy_function=copy2)
# src            要移动的文件或目录
# dst            目标路径
#                若`dst`是已存在的目录,则`src`将被移至该目录下;...
```

## rmtree()

删除一个目录树。

```python
shutil.rmtree(path, ignore_errors=False, onerror=None)
# path           要删除的目录
# ignore_errors  若为`True`,则删除失败导致的错误将被忽略
```