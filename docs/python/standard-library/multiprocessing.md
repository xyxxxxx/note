# multiprocessing——基于进程的并行

`multiprocessing` 是一个支持使用与 `threading` 模块类似的 API 来产生进程的包。`multiprocessing` 包同时提供了本地和远程并发操作，通过使用子进程而非线程有效地绕过了全局解释器锁。因此，`multiprocessing` 模块允许程序员充分利用给定机器上的多个处理器。它在 Unix 和 Windows 上均可运行。

## 产生进程

通过创建一个 `Process` 对象并调用它的 `start()` 方法来产生进程。`Process` 和 `threading.Thread` 的 API 相同。一个简单的多进程程序示例是:

```python
from multiprocessing import Process

def f(name):
    print('hello', name)

if __name__ == '__main__':
    p = Process(target=f, args=('bob',))
    p.start()
    p.join()
```

要显示所涉及的各个进程 ID，下面是一个扩展示例:

```python
from multiprocessing import Process
import os

def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())

def f(name):
    info('function f')
    print('hello', name)

if __name__ == '__main__':
    info('main line')
    p = Process(target=f, args=('bob',))
    p.start()
    p.join()
```

## 启动方法

取决于平台，`multiprocessing` 支持三种启动进程的方法，包括：

* *spawn*：父进程会启动一个全新的 Python 解释器进程，子进程将只继承那些运行进程对象的 `run()` 方法所必需的资源。特别地，来自父进程的非必需文件描述符和句柄将不会被继承。使用此方法启动进程相比使用 *fork* 或 *forkserver* 要慢上许多。

  在 Unix 和 Windows 上可用，是 Windows 上的默认设置。

* *fork*：父进程使用 `os.fork()` 来产生 Python 解释器分叉，子进程在开始时实际上与父进程相同，继承父进程的所有资源。请注意，安全分叉一个多线程进程是棘手的。

  仅在 Unix 上可用，是 Unix 上的默认值。

* *forkserver*：当程序启动并选择 *forkserver* 启动方法时，将启动服务器进程。之后每当需要一个新进程时，父进程就会连接到服务器并请求它分叉一个新进程。分叉服务器进程是单线程的，因此使用 `os.fork()` 是安全的。非必需的资源不会被继承。

  仅在支持通过 Unix 管道传递文件描述符的 Unix 平台上可用。

3.8 版本修改：对于 macOS，现在 *spawn* 是默认启动方式。因为 *fork* 启动方式可能导致子进程崩溃，而被认为是不安全的，查看 [bpo-33725](https://bugs.python.org/issue33725)。

在 Unix 上使用 *spawn* 或 *forkserver* 启动方式会同时启动一个*资源追踪器*进程，负责追踪当前程序的进程产生的、不再被使用的命名系统资源（例如命名信号量或 `SharedMemory` 对象）。当所有进程退出后，资源追踪器会释放任何仍被追踪的的对象。通常情况下是不会有这种对象的，但是假如一个子进程被某个信号杀死，就可能存在这一类资源的“泄露”情况。（泄露的信号量以及共享内存段不会被自动释放，直到下一次系统重启。对于这两类资源来说，这是一个比较大的问题，因为操作系统允许的命名信号量的数量是有限的，而共享内存段也会占据主内存的一些空间。）

在主模块的 `if __name__ == '__main__'` 子句中调用 `set_start_method()` 以选择启动方法。例如：

```python
import multiprocessing as mp

def foo(q):
    q.put('hello')

if __name__ == '__main__':
    mp.set_start_method('spawn')
    q = mp.Queue()
    p = mp.Process(target=foo, args=(q,))
    p.start()
    print(q.get())
    p.join()
```

`set_start_method()` 不应被多次调用。如果你想要在同一程序中<u>使用多种启动方法</u>，可以使用 `get_context()` 来获取上下文对象，上下文对象与 `multiprocessing` 模块具有相同的 API：

```python
import multiprocessing as mp

def foo(q):
    q.put('hello')

if __name__ == '__main__':
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    p = ctx.Process(target=foo, args=(q,))
    p.start()
    print(q.get())
    p.join()
```

需要注意的是，对象在不同上下文创建的进程之间可能并不兼容。特别是使用 *fork* 上下文创建的锁不能传递给使用 *spawn* 或 *forkserver* 启动方法启动的进程。

## 进程间交换对象

`multiprocessing` 支持两种进程间的通信通道：

* *队列*：先进先出的多生产者多消费者队列。队列是线程和进程安全的。

  ```python
  from multiprocessing import Process, Queue
  
  def f(q):
      q.put([42, None, 'hello'])
  
  if __name__ == '__main__':
      q = Queue()
      p = Process(target=f, args=(q,))
      p.start()
      print(q.get())              # [42, None, 'hello']
      p.join()
  ```

* *管道*：`Pipe()` 函数返回一对由管道连接的连接对象，表示管道的两端。每个连接对象都有 `send()` 和 `recv()` 方法。需要注意的是，如果两个进程（或线程）同时尝试读取或写入管道的<u>同一端</u>，管道中的数据可能会损坏。在不同进程中同时使用管道的<u>不同端</u>则不存在损坏的风险。

  ```python
  from multiprocessing import Process, Pipe
  
  def f(conn):
      conn.send([42, None, 'hello'])
      conn.close()
  
  if __name__ == '__main__':
      parent_conn, child_conn = Pipe()
      p = Process(target=f, args=(child_conn,))
      p.start()
      print(parent_conn.recv())   # [42, None, 'hello']
      p.join()
  ```

## 进程间同步

`multiprocessing` 拥有 `threading` 的所有同步原语的等价物。例如可以使用锁来确保一次只有一个进程打印到标准输出：

```python
from multiprocessing import Process, Lock

def f(l, i):
    l.acquire()
    try:
        print('hello world', i)
        print('hello world', i)
    finally:
        l.release()

if __name__ == '__main__':
    lock = Lock()
    ps = []
    for num in range(5):
        p = Process(target=f, args=(lock, num))
        ps.append(p)
        p.start()
    for p in ps:
        p.join()
```

```
hello world 1
hello world 1
hello world 0
hello world 0
hello world 2
hello world 2
hello world 4
hello world 4
hello world 3
hello world 3
```

## 进程间共享状态

在进行并发编程时，应尽量避免使用共享状态，使用多个进程时尤其如此。

但是，如果你真的需要使用一些共享数据，那么 `multiprocessing` 提供了两种方法：

* *共享内存*：可以使用 `Value` 或 `Array` 将数据存储在共享内存映射中。

  ```python
  from multiprocessing import Process, Value, Array
  
  def f(n, a):
      n.value = 3.1415927
      for i, v in enumerate(a):
          a[i] = -v
  
  if __name__ == '__main__':
      num = Value('d', 0.0)         # 'd' 表示双精度浮点数
      arr = Array('i', range(10))   # 'i' 表示有符号整数
  
      p = Process(target=f, args=(num, arr))
      p.start()
      p.join()
  
      print(num.value)     # 3.1415927
      print(arr[:])        # [0, -1, -2, -3, -4, -5, -6, -7, -8, -9]
  ```

  这些共享对象是进程和线程安全的。

* *服务进程*：由 `Manager()` 返回的管理器对象控制一个服务进程，该进程保存 Python 对象并允许其他进程使用代理操作它们。管理器支持下列类型：`list`、`dict`、`Namespace`、`Lock`、`RLock`、`Semaphore`、`BoundedSemaphore`、`Condition`、`Event`、`Barrier`、`Queue`、`Value` 和 `Array`。

  ```python
  from multiprocessing import Process, Manager
  
  def f(d, l):
      d[1] = '1'
      d['2'] = 2
      d[0.25] = None
      l.reverse()
  
  if __name__ == '__main__':
      with Manager() as manager:
          d = manager.dict()
          l = manager.list(range(10))
  
          p = Process(target=f, args=(d, l))
          p.start()
          p.join()
  
          print(d)         # {1: '1', '2': 2, 0.25: None}
          print(l)         # [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
  ```
  
  使用服务进程的管理器比使用共享内存对象更灵活，因为它们可以支持任意对象类型。此外，单个管理器也可以通过网络由不同计算机上的进程共享。但是它们比使用共享内存慢。

## 使用工作进程

`Pool` 类表示一个工作进程池，它具有几种不同的将任务分配到工作进程的方法。

```python
from multiprocessing import Pool, TimeoutError
import time
import os

def f(x):
    return x*x

if __name__ == '__main__':
    # start 4 worker processes
    with Pool(processes=4) as pool:

        # print "[0, 1, 4,..., 81]"
        print(pool.map(f, range(10)))

        # print same numbers in arbitrary order
        for i in pool.imap_unordered(f, range(10)):
            print(i)

        # evaluate "f(20)" asynchronously
        res = pool.apply_async(f, (20,))      # runs in *only* one process
        print(res.get(timeout=1))             # prints "400"

        # evaluate "os.getpid()" asynchronously
        res = pool.apply_async(os.getpid, ()) # runs in *only* one process
        print(res.get(timeout=1))             # prints the PID of that process

        # launching multiple evaluations asynchronously *may* use more processes
        multiple_results = [pool.apply_async(os.getpid, ()) for i in range(4)]
        print([res.get(timeout=1) for res in multiple_results])

        # make a single worker sleep for 10 secs
        res = pool.apply_async(time.sleep, (10,))
        try:
            print(res.get(timeout=1))
        except TimeoutError:
            print("We lacked patience and got a multiprocessing.TimeoutError")

        print("For the moment, the pool remains available for more work")

    # exiting the 'with'-block has stopped the pool
    print("Now the pool is closed and no longer available")
```

需要注意的是，进程池的方法只能由创建它的进程使用。

## 模块内容

### 函数

#### active_children()

返回当前进程的所有存活子进程的列表。

调用此方法有“等待”已经结束的进程的副作用。

#### cpu_count()

返回系统的 CPU 数量。

该数量不同于当前进程可以使用的 CPU 数量。可用的 CPU 数量可以通过 `len(os.sched_getaffinity(0))` 获得。

#### current_process()

返回当前进程对应的 `Process` 对象。

#### parent_process()

返回父进程对应的 `Process` 对象。对于主进程，`parent_process()` 会返回 `None`。

3.8 新版功能。

#### get_context()

```python
get_context(method=None)
```

返回一个上下文对象。该对象具有和 `multiprocessing` 模块相同的 API。

如果 *method* 为 `None`，则返回默认上下文对象。否则 *method* 应该是 `'fork'`、`'spawn'` 或 `'forkserver'`。如果指定的启动方法不存在，则引发 `ValueError` 异常。

#### get_start_method()

返回用于启动进程的启动方法名。

如果启动方法没有固定并且 *allow_none* 为 `False`，则启动方法将固定为默认方法并返回其名称；如果启动方法没有固定并且 *allow_none* 为 `True`，则返回 `None`。

返回值可以是 `'fork'`、`'spawn'`、`'forkserver'` 或 `None`。Unix 上默认为 `'fork'`，而 Windowns 和 macOS 上默认为 `'spawn'`。

#### set_executable()

设置在启动子进程时使用的 Python 解释器路径，默认使用 `sys.executable`。

#### set_start_method()

设置启动子进程的方法。*method* 可以是 `'fork'`、`'spawn'` 或 `'forkserver'`。

注意此函数最多只能调用一次，并且需要放置在 main 模块的 `if __name__ == '__main__'` 子句中。

### Array

### Barrier

类似 `threading.Barrier` 的栅栏对象。

### connection.Connection

连接对象允许收发可以序列化的对象或字符串。它们可以被看作是面向消息的连接套接字。

通常使用 `Pipe` 创建连接对象。

```python
>>> from multiprocessing import Pipe
>>> a, b = Pipe()
>>> a.send([1, 'hello', None])
>>> b.recv()
[1, 'hello', None]
>>> b.send_bytes(b'thank you')
>>> a.recv_bytes()
b'thank you'
>>> import array
>>> arr1 = array.array('i', range(5))
>>> arr2 = array.array('i', [0] * 10)
>>> a.send_bytes(arr1)
>>> count = b.recv_bytes_into(arr2)
>>> assert count == len(arr1) * arr1.itemsize
>>> arr2
array('i', [0, 1, 2, 3, 4, 0, 0, 0, 0, 0])
```

!!! warning "警告"
    如果一个进程在试图读写管道时被杀掉，那么管道中的数据很可能是不完整的，因为此时可能无法确定消息的边界。

#### send()

将一个对象发送到连接的另一端，另一端使用 `recv()` 读取。

发送的对象必须是可序列化的，过大的对象（接近 32MiB+，具体值取决于操作系统）可能引发 `ValueError` 异常。

#### recv()

返回一个由连接的另一端使用 `send()` 发送的对象。此方法会一直阻塞直到接收到对象。如果没有东西可接收并且另一端关闭了连接，则引发 `EOFError` 异常。

#### fileno()

返回由连接对象使用的文件描述符或者句柄。

#### close()

关闭连接。

当连接被垃圾回收时会自动调用。

#### poll()

```python
poll(timeout=0.0)
```

返回是否还有可以读取的数据。

如果未指定 *timeout*，此方法会立即返回；如果 *timeout* 是一个数字，则指定了阻塞的最大秒数；如果 *timeout* 是 `None`，那么将一直等待，不会超时。

#### send_bytes()

```python
send_bytes(buffer, offset=0, size=None)
```

从一个类字节对象中取出字节数据并作为一条完整消息发送。

如果给定了 *offset*，则会从 *buffer* 的该位置读取数据。如果给定了 *size*，则会从 *buffer* 读取这么多个字节。非常大的缓冲区（接近 32MiB+，具体值取决于操作系统）可能引发 `ValueError` 异常。

#### recv_bytes()

```python
recv_bytes(self, maxlength=None)
```

以字符串形式返回一条从连接的另一端发送过来的由字节数据构成的完整消息。此方法会一直阻塞直到接收到数据。如果没有数据可接收并且另一端关闭了连接，则引发 `EOFError` 异常。

### Event

类似 `threading.Event` 的事件对象。

### JoinableQueue

### Lock

类似 `threading.Lock` 的原始锁（非递归锁）对象。一旦一个进程或者线程获得了锁，后续的任何其他进程或线程的获取它的尝试都会阻塞直到它被释放。任何进程或线程都可以释放锁。除非另有说明，`multiprocessing.Lock` 适用于进程或者线程的概念和行为都和 `threading.Lock` 适用于线程的概念和行为一致。

`Lock` 支持上下文管理器协议，因此可以在 `with` 语句中使用。

#### acquire()

```python
acquire(block=True, timeout=None)
```

获得锁。

若 *block* 为 `True` 并且 *timeout* 为 `None`，则会阻塞当前进程直到锁被释放，然后将锁设为锁定状态并返回 `True`；若 *block* 为 `True` 并且 *timeout* 为正数，则会在阻塞了最多 *timeout* 秒后锁还是没有被释放的情况下返回 `False`；若 *block* 为 `False`（此时 *timeout* 会被忽略），或者 *block* 为 `True` 并且 *timeout* 为 0 或负数，则会在锁被锁定的情况下返回 `False`，否则将锁设为锁定状态并返回 `True`。

注意此函数的参数的一些行为与 `threading.Lock.acquire()` 的实现有所不同。

#### release()

释放锁。

可以在任何进程或线程中调用，并不限于锁的拥有者。当尝试释放一个没有被持有的锁时，会引发 `ValueError` 异常。除此之外其行为与 `threading.Lock.release()` 相同。

### Manager

### parent_process()

返回当前进程的父进程相对应的 `Process` 对象。

### Process

```python
class multiprocessing.Process(group=None, target=None, name=None, args=(), kwargs={}, *, daemon=None)
# group     始终为None,仅用于兼容`threading.Thread`
# target    由`run()`方法调用的可调用对象
# name      进程名称
# args      目标调用的位置参数
# kwargs    目标调用的关键字参数
# daemon    进程的daemon标志.若为None,则该标志从创建它的进程继承
```

进程对象表示在单独的进程中运行的活动。`Process` 类拥有 `threading.Thread` 的所有方法的等价物。

#### run()

表示进程活动的方法。

#### start()

启动进程活动。

此方法每个进程对象最多调用一次。它会将对象的 `run()` 方法安排在一个单独的进程中调用。

#### join()

```python
join(timeout=None)
```

如果可选参数 *timeout* 是 `None`（默认值），则该方法将阻塞，直到调用 `join()` 方法的进程终止；如果 *timeout* 是一个正数，它最多会阻塞 *timeout* 秒。不管是进程终止还是方法超时，此方法都返回 `None`。检查进程的 `exitcode` 以确定它是否终止。

一个进程可以被 `join` 多次。

进程无法 `join` 自身，因为这会导致死锁。尝试在启动进程之前 `join` 进程会产生一个错误。

#### name

进程的名称。该名称是一个字符串，仅用于识别，没有具体语义。可以为多个进程指定相同的名称。

#### is_alive()

返回进程是否处于活动状态。

粗略地说，从 `start()` 方法返回到子进程终止之间，进程对象都处于活动状态。

#### daemon

进程的守护标志，一个布尔值。必须在 `start()` 被调用之前设置。

初始值继承自创建进程。

当一个进程退出时，它会尝试终止子进程中的所有守护进程。

#### pid

返回进程 ID。在进程产生之前为 `None`。

#### exitcode

子进程的退出代码。`None` 表示进程尚未终止；负值 *-N* 表示子进程被信号 *N* 终止。

#### authkey

进程的身份验证密钥（字节字符串）。

#### sentinel

系统对象的数字句柄，当进程结束时将变为 ready。

如果要使用 `multiprocessing.connection.wait()` 一次等待多个事件，可以使用此值。否则调用 `join()` 更简单。

在 Windows 上，这是一个操作系统句柄，可以与 `WaitForSingleObject` 和 `WaitForMultipleObjects` 系列 API 调用一起使用。在 Unix 上，这是一个文件描述符，可以与来自 `select` 模块的原语一起使用。

#### terminate()

终止进程。在 Unix 上由 `SIGTERM` 信号完成；在 Windows 上由 `TerminateProcess()` 完成。注意进程终止时不会执行退出处理程序和 finally 子句等。

!!! note "注意"
    进程的后代进程将不会被终止——它们只会变成孤儿进程。

#### kill()

与 `terminate()` 相同，但在 Unix 上使用 `SIGKILL` 信号。

#### close()

关闭 `Process` 对象，释放与之关联的所有资源。如果底层进程仍在运行，则会引发 `ValueError`。一旦 `close()` 成功返回，`Process` 对象的大部分其他方法和属性将引发 `ValueError`。

### Pipe

```python
class multiprocessing.Pipe([duplex])
```

返回一对 `Connection` 对象 `(conn1, conn2)`，分别表示管道的两端。

如果 *duplex* 被置为 `True`（默认值），那么该管道是双向的；如果 *duplex* 被置为 `False` ，那么该管道是单向的，即 `conn1` 只能用于接收消息，而 `conn2` 仅能用于发送消息。

### Pool

### Queue

返回一个使用一个管道和少量锁和信号量实现的共享队列实例。当一个进程将一个对象放进队列中时，一个写入线程会启动并将对象从缓冲区写入管道中。

`Queue` 实现了标准库类 `queue.Queue` 的所有方法，除了 `task_done()` 和 `join()`。一旦超时，将抛出标准库 `queue` 模块中常见的异常 `queue.Empty` 和 `queue.Full`。

!!! warning "警告"
    如果一个进程在尝试使用 `Queue` 期间被 `Process.terminate()` 或 `os.kill()` 杀掉，那么队列中的数据很可能损坏。这可能导致其他进程在之后尝试使用该队列时发生异常。

#### qsize()

返回队列的大致长度。由于多线程或者多进程的环境，该数字是不可靠的。

#### empty(), full()

如果队列是空/满的，返回 `True`，反之返回 `False`。由于多线程或多进程的环境，该状态是不可靠的。

#### put()

```python
put(obj, block=True, timeout=None)
```

将 *obj* 放入队列。若 *block* 为 `True` 并且 *timeout* 为 `None`，则会阻塞当前进程，直到有空闲槽位；若 *block* 为 `True` 并且 *timeout* 为正数，则会在阻塞了最多 *timeout* 秒后还是没有可用的空闲槽位的情况下引发 `queue.Full` 异常；若 *block* 为 `False`，则会在有立即可用的空闲槽位的情况下将对象放入队列，否则引发 `queue.Full` 异常（此时 *timeout* 参数会被忽略）。

#### put_nowait()

```python
put_nowait(obj)
```

等同于 `put(obj, False)`。

#### get()

```python
get(block=True, timeout=None)
```

从队列中移除并返回对象。若 *block* 为 `True` 并且 *timeout* 为 `None`，则会阻塞当前进程，直到有可用对象；若 *block* 为 `True` 并且 *timeout* 为正数，则会在阻塞了最多 *timeout* 秒后还是没有可用对象的情况下引发 `queue.Empty` 异常；若 *block* 为 `False`，则会在有立即可用的对象的情况下返回对象，否则引发 `queue.Empty` 异常（此时 *timeout* 参数会被忽略）。

#### get_nowait()

等同于 `get(False)`。

#### close()

指示当前进程将不会再往队列中放入更多数据。一旦所有缓冲数据都被写入管道之后，后台的线程会退出。

当队列被垃圾回收时会自动调用。

### RLock

类似 `threading.RLock` 的递归锁对象。递归锁必须由持有进程或线程亲自释放。如果某个进程或线程获得了递归锁，该进程或线程可以再次获得这个锁而无需阻塞。但是这个进程或线程的获得锁操作和释放锁操作的次数必须相同。

`RLock` 支持上下文管理器协议，因此可以在 `with` 语句中使用。

#### acquire()

```python
acquire(self, block=True, timeout=None)
```

获得锁。

若 *block* 为 `True` 并且 *timeout* 为 `None`，则会阻塞当前进程直到锁被释放，除非当前进程已经持有此锁，然后持有此锁，将锁的递归等级加一，并返回 `True`；若 *block* 为 `True` 并且 *timeout* 为正数，则会在阻塞了最多 *timeout* 秒后锁还是没有被释放的情况下返回 `False`；若 *block* 为 `False`（此时 *timeout* 会被忽略），或者 *block* 为 `True` 并且 *timeout* 为 0 或负数，则会在锁被锁定的情况下返回 `False`，否则持有此锁，将锁的递归等级加一，并返回 `True`。

注意此方法的参数的一些行为与 `threading.RLock.acquire()` 的实现有所不同。

#### release()

释放锁，亦即使锁的递归等级减一。

如果释放后锁的递归等级降为 0，则会重置锁的状态为释放状态。

必须在持有该锁的进程或线程中使用，否则会引发 `AssertionError` 异常。除了异常类型之外，其行为与 `threading.RLock.release()` 相同。

### Semaphore

类似 `threading.Semaphore` 的递归锁对象。

### SimpleQueue

一个简化的 `Queue` 类型，很像带锁的 `Pipe`。

#### empty()

如果队列为空返回 `True`，否则返回 `False`。

#### get()

从队列中移除并返回一个对象。

#### put()

将对象放入队列。

### Value
