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

3.7 版本添加。

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

队列对象（`Queue`、`LifoQueue` 和 `PriorityQueue`）提供下列公共方法。

### qsize()

返回队列的大致大小。注意，`qsize() > 0` 不保证后续的 `get()` 不会阻塞，`qsize() < maxsize` 也不保证 `put()` 不会阻塞。

### empty()

如果队列为空，返回 True，否则返回 False。`empty()` 返回 True 不保证后续调用的 `put()` 不会阻塞，`empty()` 返回 False 也不保证后续调用的 `get()` 不会阻塞。

### full()

如果队列为满，返回 True，否则返回 False。`full()` 返回 True 不保证后续调用的 `get()` 不会阻塞，`full()` 返回 False 也不保证后续调用的 `put()` 不会阻塞。

### put()

```python
put(item, block=True, timeout=None)
```

将 *item* 放入队列。如果可选参数 *block* 为 True 并且 *timeout* 是 None（默认），则在必要时阻塞至有空闲槽位可用；如果 *timeout* 是一个正数，将最多阻塞 *timeout* 秒，如果在这段时间没有可用的空闲槽位，则引发 `Full` 异常。如果 *block* 为 False 并且空闲槽位立即可用，则将 *item* 放入队列，否则引发 `Full` 异常（在这种情况下，*timeout* 将被忽略）。

### put_nowait()

```python
put_nowait(item)
```

等同于 `put(item, False)`。

### get()

```python
get(block=True, timeout=None)
```

从队列中移除并返回一个项目。如果可选参数 *block* 为 True 并且 *timeout* 是 None（默认），则在必要时阻塞至一个项可得到；如果 *timeout* 是一个正数，将最多阻塞 *timeout* 秒，如果在这段时间没有项可得到，则引发 `Empty` 异常。如果 *block* 为 False 并且一个项立即可得到，则返回一个项，否则引发 `Empty` 异常（在这种情况下，*timeout* 将被忽略）。

POSIX 系统 3.0 版本之前，以及所有版本的 Windows 系统中，如果 *block* 为 True 并且 *timeout* 为 None，这个操作将进入底层锁的不间断等待。这意味着没有异常能发生，特别是 `SIGINT` 将不会触发 `KeyboardInterrupt` 异常。

### get_nowait()

等同于 `get(False)`。

### task_done()

指示一个被放入队列的任务已经完成。由队列的消费者线程使用。每次调用 `get()` 获取一个任务之后，调用 `task_done()` 以告知队列该任务的处理已经完成。

如果 `join()` 当前正在阻塞，在所有项都被处理后（即对于每一个被放入队列的项都调用了 `task_done()`）它将解除阻塞。

如果调用的次数多于放入队列中的项数，将引发一个 `ValueError`。

### join()

阻塞至队列中所有的项都被获取和处理完毕。

每当项被添加到队列时，未完成任务的计数就会增加。每当消费者线程调用 `task_done()` 指示该项的所有工作已经完成，计数就会减少。当未完成任务的计数降到零时，`join()` 解除阻塞。

下面是一个等待排队的任务完成的示例：

```python
import threading, queue

q = queue.Queue()

def worker():
    while True:
        item = q.get()
        print(f'Working on {item}')
        print(f'Finished {item}')
        q.task_done()

# turn-on the worker thread
threading.Thread(target=worker, daemon=True).start()

# send thirty task requests to the worker
for item in range(30):
    q.put(item)
print('All task requests sent\n', end='')

# block until all tasks are done
q.join()
print('All work completed')
```

## SimpleQueue 对象

`SimpleQueue` 对象提供下列公共方法。

### qsize()

返回队列的大致大小。注意，`qsize() > 0` 不保证后续的 `get()` 不会阻塞。

### empty()

如果队列为空，返回 True，否则返回 False。`empty()` 返回 True 不保证后续调用的 `put()` 不会阻塞。

### put()

```python
put(item, block=True, timeout=None)
```

将 *item* 放入队列。此方法永不阻塞，始终成功（除了潜在的低级错误，例如内存分配失败）。可选参数 *block* 和 *timeout* 被忽略，它们只是为了与 `Queue.put()` 兼容而提供。

CPython 实现细节：This method has a C implementation which is reentrant. That is, a `put()` or `get()` call can be interrupted by another `put()` call in the same thread without deadlocking or corrupting internal state inside the queue. This makes it appropriate for use in destructors such as `__del__` methods or `weakref` callbacks.

### put_nowait()

```python
put_nowait(item)
```

等同于 `put(item)`，为与 `Queue.put_nowait()` 兼容而提供。

### get()

```python
get(block=True, timeout=None)
```

从队列中移除并返回一个项目。如果可选参数 *block* 为 True 并且 *timeout* 是 None（默认），则在必要时阻塞至一个项可得到；如果 *timeout* 是一个正数，将最多阻塞 *timeout* 秒，如果在这段时间没有项可得到，则引发 `Empty` 异常。如果 *block* 为 False 并且一个项立即可得到，则返回一个项，否则引发 `Empty` 异常（在这种情况下，*timeout* 将被忽略）。

### get_nowait()

等同于 `get(False)`。
