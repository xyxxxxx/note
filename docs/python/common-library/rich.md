# rich

[Rich](https://rich.readthedocs.io/en/latest/) 是一个用于在终端中提供富文本和精美格式的 Python 库。

Rich 的 API 使为终端输出增加颜色和样式变得简单。此外，Rich 还可以绘制漂亮的表格、进度条、Markdown、语法高亮的源代码以及栈回溯信息（tracebacks）等——开箱即用。

<figure>
<img src="https://github.com/willmcgugan/rich/raw/master/imgs/features.png" alt="encoding" width="800"/>
</figure>

Rich 适用于 Linu、OSX 和 Windows。真彩色/表情符号可与新的 Windows 终端一起使用，Windows 的经典终端仅限 8 种颜色。

Rich 可以在 Jupyter Notebook 中使用，而无需额外配置。

## 富文本

### emoji

要在控制台输出中插入 emoji，只需要将其名称放置在两个冒号之间。例如：

```python
>>> console.print(":smiley: :vampire: :pile_of_poo: :thumbs_up: :raccoon:")
😃 🧛 💩 👍 🦝
```

!!! tip "提示"
    可以前往[此页面](https://squidfunk.github.io/mkdocs-material/reference/icons-emojis/)搜索 emoji。

## 日志

## 表格

## 进度展示

Rich 可以展示持续更新的关于长时间运行任务的信息。展示的信息是可以配置的，默认配置会展示任务的描述、进度条、完成百分比和预计剩余时间。

Rich 的进度展示支持多任务，每个任务都有单独的进度条以及其他进度信息。你可以使用这一功能来追踪运行在多个线程或进程中的并发的任务。

!!! note "注意"
    进度展示可以在 Jupyter Notebook 中使用，但自动刷新会被禁用。你需要显式地调用 `refresh()` 或在调用 `update()` 时设置 `refresh=True`；或者使用在每一个循环自动刷新的 `track()` 函数。

### 基本使用

对于基本使用只需要调用 `track()` 函数，其接受一个序列和一个可选的任务描述。`track()` 函数会在每一次迭代中从序列产出值并更新进度信息。下面是一个示例：

```python
import time
from rich.progress import track

for i in track(range(20), description="Processing..."):
    time.sleep(1)  # Simulate work being done
```

### 高级使用

如果你需要同时展示多个任务，或想要配置进度展示中具体的列，你可以直接操作 `Progress` 类。一旦你构建了一个 `Progress` 对象，就可以通过 `add_task()` 添加任务，通过 `update()` 更新进度。

`Progress` 对象被设计用作一个上下文管理器，其自动地开始和结束进度显示。下面是一个简单的示例：

```python
import time
from rich.progress import Progress

with Progress() as progress:
    task1 = progress.add_task("[red]Downloading...", total=1000)
    task2 = progress.add_task("[green]Processing...", total=1000)
    task3 = progress.add_task("[cyan]Cooking...", total=1000)
    while not progress.finished:
        progress.update(task1, advance=0.5)
        progress.update(task2, advance=0.3)
        progress.update(task3, advance=0.9)
        time.sleep(0.02)
```

```shell
Downloading... ━━━━━━━━━━━╺━━━━━━━━━━━━━━━━━━━━━━━━━━━━  28% 0:00:34
Processing...  ━━━━━━╸━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  17% 0:01:04
Cooking...     ━━━━━━━━━━━━━━━━━━━╸━━━━━━━━━━━━━━━━━━━━  50% 0:00:13
```

任务的 `total` 参数的值是其进度达到 100% 所需要完成的步数。在这里的语境下，“步”可以是适应于你的具体应用的任何指标，例如文件下载的字节数，已经处理的图像数，等等。

#### 更新任务

调用 `add_task()` 会返回一个 Task ID，使用这个 ID 调用 `update()`，每当你完成了一些工作，或者修改了任何信息。你可以直接设置 `completed` 参数来更新 `task.completed` 的值，也可以设置 `advance` 参数作为 `task.completed` 的增量。

`update()` 方法还收集与任务相关的关键字参数，使用此功能来提供你想要在进度展示中解析的额外信息。额外的关键字参数会保存在 `task.fields` 中，可以被 `ProgressColumn` 对象引用。

#### 列

你可以通过传入 `Progress` 构造函数的位置参数来自定义进度展示的各列。每一列必须指定为一个格式字符串或一个 `ProgressColumn` 对象。

格式字符串会使用 `Task` 实例来进行解析。例如 `"{task.description}"` 会在这一列中展示任务的描述；`"{task.completed} of {task.total}"` 会展示总步数中的多少已经完成。作为关键字参数传入 `Progress` 对象的 `update()` 方法的额外字段保存在 `task.fields` 中，你可以按照如下语法来将它们添加到格式字符串中：`"extra info: {task.fields[extra]}"`。

默认展示的列相当于如下配置：

```python
progress = Progress(
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TaskProgressColumn(),
    TimeRemainingColumn(),
)
```

要在默认的基础上增加更多的列，使用 `get_default_columns()`:

```python
progress = Progress(
    SpinnerColumn(),
    *Progress.get_default_columns(),
    TimeElapsedColumn(),
)
```

下面展示了所有可用的 `ProgressColumn` 对象：

* `BarColumn`：展示进度条.
* `TextColumn`：展示文本。
* `TimeElapsedColumn`：展示经过的时间。
* `TimeRemainingColumn`：展示预计剩余的时间。
* `MofNCompleteColumn`：以 `"{task.completed}/{task.total}"` 的形式展示完成进度（`task.completed` 和 `task.total` 最好都是整数）。
* `FileSizeColumn`：展示文件大小（假定这里的步为字节）。
* `TotalFileSizeColumn`：展示总的文件大小（假定这里的步为字节）。
* `DownloadColumn`：展示下载速度（假定这里的步为字节）。
* `TransferSpeedColumn`：展示传输速度（假定这里的步为字节）。
* `SpinnerColumn`：展示一个“转圈”的动画。
* `RenderableColumn`：Displays an arbitrary Rich renderable in the column.

你也可以扩展 `ProgressColumn` 类来实现你自己的列，使用方法与其他的列相同。

## 树

## Markdown

## 语法高亮

## 检查对象

Rich 提供了一个检查函数，其可以生成关于任何 Python 对象的报告。

```python
>>> d = {'a': 1, 'b': 2}
>>> from rich import inspect
>>> inspect(d, methods=True)
╭──────────────────────────────────── <class 'dict'> ────────────────────────────────────╮
│ dict() -> new empty dictionary                                                         │
│ dict(mapping) -> new dictionary initialized from a mapping object's                    │
│     (key, value) pairs                                                                 │
│ dict(iterable) -> new dictionary initialized as if via:                                │
│     d = {}                                                                             │
│     for k, v in iterable:                                                              │
│         d[k] = v                                                                       │
│ dict(**kwargs) -> new dictionary initialized with the name=value pairs                 │
│     in the keyword argument list.  For example:  dict(one=1, two=2)                    │
│                                                                                        │
│ ╭────────────────────────────────────────────────────────────────────────────────────╮ │
│ │ {'a': 1, 'b': 2}                                                                   │ │
│ ╰────────────────────────────────────────────────────────────────────────────────────╯ │
│                                                                                        │
│      clear = def clear(...) D.clear() -> None.  Remove all items from D.               │
│       copy = def copy(...) D.copy() -> a shallow copy of D                             │
│   fromkeys = def fromkeys(iterable, value=None, /): Create a new dictionary with keys  │
│              from iterable and values set to value.                                    │
│        get = def get(key, default=None, /): Return the value for key if key is in the  │
│              dictionary, else default.                                                 │
│      items = def items(...) D.items() -> a set-like object providing a view on D's     │
│              items                                                                     │
│       keys = def keys(...) D.keys() -> a set-like object providing a view on D's keys  │
│        pop = def pop(...)                                                              │
│              D.pop(k[,d]) -> v, remove specified key and return the corresponding      │
│              value.                                                                    │
│              If key is not found, d is returned if given, otherwise KeyError is raised │
│    popitem = def popitem(): Remove and return a (key, value) pair as a 2-tuple.        │
│ setdefault = def setdefault(key, default=None, /): Insert key with a value of default  │
│              if key is not in the dictionary.                                          │
│     update = def update(...)                                                           │
│              D.update([E, ]**F) -> None.  Update D from dict/iterable E and F.         │
│              If E is present and has a .keys() method, then does:  for k in E: D[k] =  │
│              E[k]                                                                      │
│              If E is present and lacks a .keys() method, then does:  for k, v in E:    │
│              D[k] = v                                                                  │
│              In either case, this is followed by: for k in F:  D[k] = F[k]             │
│     values = def values(...) D.values() -> an object providing a view on D's values    │
╰────────────────────────────────────────────────────────────────────────────────────────╯
```

## 栈回溯
