# pprint——数据美化输出

`pprint` 模块提供了“美化打印”任意 Python 数据结构的功能。

## isreadable()

确定对象的格式化表示是否“可读”，或是否可以通过 `eval()` 重新构建对象的值。对于递归对象总是返回 `False`。

```python
>>> stuff = ['spam', 'eggs', 'lumberjack', 'knights', 'ni']
>>> stuff.insert(0, stuff[:])
>>> pprint.isreadable(stuff)
True

>>> stuff = ['spam', 'eggs', 'lumberjack', 'knights', 'ni']
>>> stuff.insert(0, stuff)
>>> pprint.isreadable(stuff)
False
```

## isrecursive()

确定对象是否为递归对象。

```python
>>> stuff = ['spam', 'eggs', 'lumberjack', 'knights', 'ni']
>>> stuff.insert(0, stuff[:])
>>> pprint.isrecursive(stuff)
False

>>> stuff = ['spam', 'eggs', 'lumberjack', 'knights', 'ni']
>>> stuff.insert(0, stuff)
>>> pprint.isrecursive(stuff)
True
```

## pprint()

打印对象的格式化表示。

```python
pprint.pprint(object, stream=None, indent=1, width=80, depth=None, *, compact=False, sort_dicts=True)
# object       被打印的对象
# stream...    参见`PrettyPrinter`,将作为参数被传给`PrettyPrinter`构造函数
```

```python
>>> import pprint
>>> stuff = ['spam', 'eggs', 'lumberjack', 'knights', 'ni']
>>> stuff.insert(0, stuff[:])
>>> stuff
[['spam', 'eggs', 'lumberjack', 'knights', 'ni'], 'spam', 'eggs', 'lumberjack', 'knights', 'ni']
>>> pprint.pprint(stuff)
[['spam', 'eggs', 'lumberjack', 'knights', 'ni'],
 'spam',
 'eggs',
 'lumberjack',
 'knights',
 'ni']
>>> pprint.pprint(stuff, indent=2)
[ ['spam', 'eggs', 'lumberjack', 'knights', 'ni'],
  'spam',
  'eggs',
  'lumberjack',
  'knights',
  'ni']
>>> pprint.pprint(stuff, indent=0)
[['spam', 'eggs', 'lumberjack', 'knights', 'ni'],
'spam',
'eggs',
'lumberjack',
'knights',
'ni']
>>> pprint.pprint(stuff, width=20)
[['spam',
  'eggs',
  'lumberjack',
  'knights',
  'ni'],
 'spam',
 'eggs',
 'lumberjack',
 'knights',
 'ni']
>>> pprint.pprint(stuff, width=20, compact=True)
[['spam', 'eggs',
  'lumberjack',
  'knights', 'ni'],
 'spam', 'eggs',
 'lumberjack',
 'knights', 'ni']
>>> pprint.pprint(stuff, depth=1)
[[...], 'spam', 'eggs', 'lumberjack', 'knights', 'ni']
```

## pformat()

将对象的格式化表示作为字符串返回，其余部分与 `pprint()` 相同。

```python
>>> pprint.pformat(stuff)
"[['spam', 'eggs', 'lumberjack', 'knights', 'ni'],\n 'spam',\n 'eggs',\n 'lumberjack',\n 'knights',\n 'ni']"
```

## PrettyPrinter

`pprint` 模块定义的实现美化打印的类。

```python
class pprint.PrettyPrinter(indent=1, width=80, depth=None, stream=None, *, compact=False, sort_dicts=True)
# indent      每个递归层次的缩进量
# width       每个输出行的最大宽度
# depth       可被打印的层级数
# stream      输出流,未指定则选择`sys.stdout`
# compact     若为True,则将在width可容纳的条件下让输出更紧凑
# sort_dicts  若为True,则字典将按键排序输出,否则按插入顺序输出
```

```python
>>> import pprint
>>> stuff = ['spam', 'eggs', 'lumberjack', 'knights', 'ni']
>>> stuff.insert(0, stuff[:])
>>> stuff
[['spam', 'eggs', 'lumberjack', 'knights', 'ni'], 'spam', 'eggs', 'lumberjack', 'knights', 'ni']
>>> pprinter = pprint.PrettyPrinter()
>>> pprinter.pprint(stuff)
[['spam', 'eggs', 'lumberjack', 'knights', 'ni'],
 'spam',
 'eggs',
 'lumberjack',
 'knights',
 'ni']
```

`PrettyPrinter` 对象具有 `pprint` 模块的各方法。实际上 `pprint` 模块的各方法都是先创建 `PrettyPrinter` 对象再调用对象的方法。
