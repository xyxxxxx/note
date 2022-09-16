# queue——一个同步的队列类

`queue` 模块实现了多生产者、多消费者队列。它特别适用于消息必须安全地在多个线程间交换的线程编程。模块中的 `Queue` 类实现了所有需要的锁定语义（locking semantics）。

模块实现了三种类型的队列，它们的区别仅在于条目被取出的顺序。在 FIFO 队列中，先添加的任务先取出；在 LIFO 队列中，最近被添加的条目先取出（类似一个栈）；优先级队列中，条目将保持排序（使用 `heapq` 模块）并且最小值的条目先取出。

在内部，这三种类型的队列使用锁来临时阻塞竞争线程；然而，它们并未被设计用于线程的重入性处理。

此外，模块实现了一个“简单的” FIFO 队列类型 `SimpleQueue`，这个特殊实现以更少的功能换取了额外的保证。

## 模块内容

### Queue

```python
class queue.Queue(maxsize=0)
```

FIFO 队列构造函数。*maxsize* 是一个整数，用于设置可以放入队列的项数量上限。当达到这个大小的时候，插入操作将阻塞至队列中的项被消费掉。如果 *maxsize* 小于等于零，队列大小为无限大。

### LifoQueue

```python
class queue.LifoQueue(maxsize=0)
```

LIFO 队列构造函数。*maxsize* 是一个整数，用于设置可以放入队列的项数量上限。当达到这个大小的时候，插入操作将阻塞至队列中的项被消费掉。如果 *maxsize* 小于等于零，队列大小为无限大。

### PriorityQueue

```python
class queue.PriorityQueue(maxsize=0)
```

优先级队列构造函数。*maxsize* 是一个整数，用于设置可以放入队列的项数量上限。当达到这个大小的时候，插入操作将阻塞至队列中的项被消费掉。如果 *maxsize* 小于等于零，队列大小为无限大。

最小值项先被取出（最小值项是由 `sorted(list(entries))[0]` 返回的项）。项的一个典型模式是以下形式的元组：`(priority_number, data)`。

如果 *data* 元素不可比较，这些数据会被包装在一个类中，其忽略 *data* 而仅仅比较 `priority_number`：

```python
from dataclasses import dataclass, field
from typing import Any

@dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: Any=field(compare=False)
```

### SimpleQueue

无界限的 FIFO 队列构造函数。缺少任务跟踪等高级功能的简单队列。

3.7 新版功能。

### Empty

```python
exception queue.Empty
```

对于空的 `Queue` 对象调用非阻塞的 `get()`（或 `get_nowait()`）时引发的异常。

### Full

```python
exception queue.Full
```

对于满的 `Queue` 对象调用非阻塞的 `put()`（或 `put_nowait()`）时引发的异常。

## 队列对象



## SimpleQueue 对象
