

# fork

Python的`os`模块封装了常见的系统调用，其中就包括`fork`：

```python
import os

print('Process (%s) start...' % os.getpid())
# Only works on Unix/Linux/Mac:
pid = os.fork()
if pid == 0:
    print('I am child process (%s) and my parent is %s.' % (os.getpid(), os.getppid()))
else:
    print('I (%s) just created a child process (%s).' % (os.getpid(), pid))
    
# Process (876) start...
# I (876) just created a child process (877).
# I am child process (877) and my parent is 876.
```

# 多进程

`multiprocessing`模块提供了一个`Process`类来代表一个进程对象，下面的例子演示了启动一个子进程并等待其结束：

```python
from multiprocessing import Process
import os

# 子进程要执行的代码
def run_proc(name):
    print('Run child process %s (%s)...' % (name, os.getpid()))

if __name__=='__main__':
    print('Parent process %s.' % os.getpid())
    p = Process(target=run_proc, args=('test',))  # 创建子进程
                # 传入子进程跳转的目标函数和函数参数
    print('Child process will start.')
    p.start()       # 启动子进程
    p.join()        # 等待子进程结束
    print('Child process end.')
```

```
Parent process 928.
Child process will start.
Run child process test (929)...
Child process end.
```

## 启动方法

根据不同的平台，`multiprocessing`支持三种启动进程的方法。这些启动方法有：

+ *spawn*：父进程会启动一个全新的 python 解释器进程，子进程将只继承那些运行进程对象的`run()`方法所必需的资源。使用此方法启动进程相比使用 *fork* 或 *forkserver* 要慢上许多。

  可在Unix和Windows上使用。 Windows上的默认设置。

+ *fork*：父进程使用`os.fork()`来产生 Python 解释器分叉。子进程在开始时实际上与父进程相同。

  只存在于Unix。Unix中的默认值。

+ *forkserver*

```python
import multiprocessing as mp

def foo(q):
    q.put('hello')

if __name__ == '__main__':
    mp.set_start_method('spawn')   # 选择一个启动方法
    q = mp.Queue()
    p = mp.Process(target=foo, args=(q,))
    p.start()
    print(q.get())
    p.join()
```

## IRC

操作系统提供了进程间通信机制，`multiprocessing`包装了这些底层机制，提供了`Queue`,`Pipes`两种通信通道。

### `Queue`

`Queue()`返回一个同步的队列对象，用于在多个生产者和消费者之间通信。例如：

```python
from multiprocessing import Process, Queue
import os, time, random

# 写数据进程执行的代码:
def write(q):
    print('Process to write: %s' % os.getpid())
    for value in ['A', 'B', 'C']:
        print('Put %s to queue...' % value)
        q.put(value)
        time.sleep(random.random())

# 读数据进程执行的代码:
def read(q):
    print('Process to read: %s' % os.getpid())
    while True:
        value = q.get(True)
        print('Get %s from queue.' % value)

if __name__=='__main__':
    # 父进程创建Queue，并传给各个子进程：
    q = Queue()
    pw = Process(target=write, args=(q,))
    pr = Process(target=read, args=(q,))
    # 启动子进程pw，写入:
    pw.start()
    # 启动子进程pr，读取:
    pr.start()
    # 等待pw结束:
    pw.join()
    # pr进程里是死循环，无法等待其结束，只能强行终止:
    pr.terminate()
```

```python
Process to write: 50563
Put A to queue...
Process to read: 50564
Get A from queue.
Put B to queue...
Get B from queue.
Put C to queue...
Get C from queue.
```

队列是线程和进程安全的。

### `Pipe`

`Pipe()`函数返回一个由管道连接的连接对象，默认情况下是双工（双向）。例如：

```python
from multiprocessing import Process, Pipe

def f(conn):
    conn.send([42, None, 'hello'])
    conn.close()

if __name__ == '__main__':
    parent_conn, child_conn = Pipe()
    p = Process(target=f, args=(child_conn,))
    p.start()
    print(parent_conn.recv())   # prints "[42, None, 'hello']"
    p.join()
```

返回的两个连接对象`parent_conn`和`child_conn`表示管道的两端，每个连接对象都有 `send()` 和 `recv()` 方法（相互之间的）。注意如果两个进程（或线程）同时尝试读取或写入管道的同一 端，则管道中的数据可能会损坏；当然在不同进程（或线程）中同时使用管道的不同端的情况下不会存在损坏的风险。

## 进程间同步

对于`threading`的所有同步原语，`multiprocessing`中都有同名的等价物。例如，可以使用锁来确保一次只有一个进程打印到标准输出：

```python
from multiprocessing import Process, Lock

def f(l, i):
    l.acquire()
    try:
        print('hello world', i)
    finally:
        l.release()

if __name__ == '__main__':
    lock = Lock()

    for num in range(10):
        Process(target=f, args=(lock, num)).start()
```

## 进程间共享状态

使用多个进程时应尽量避免使用共享状态。但是如果你真的需要使用一些共享数据，那么`multiprocessing`提供了两种方法。

**共享内存**

可以使用`Value`或`Array`将数据存储在共享内存映射中。例如：

```python
from multiprocessing import Process, Value, Array

def f(n, a):
    n.value = 3.1415927
    for i in range(len(a)):
        a[i] = -a[i]

if __name__ == '__main__':
    num = Value('d', 0.0)
    arr = Array('i', range(10))

    p = Process(target=f, args=(num, arr))
    p.start()
    p.join()

    print(num.value)
    print(arr[:])
```

```
3.1415927
[0, -1, -2, -3, -4, -5, -6, -7, -8, -9]
```

创建 `num` 和 `arr` 时使用的 `'d'` 和 `'i'` 参数是 `array` 模块使用的类型的类型码： `'d'` 表示双精度浮点数， `'i'` 表示有符号整数。这些共享对象是进程和线程安全的。

**服务进程**

由 `Manager()` 返回的管理器对象控制一个服务进程，该进程保存Python对象并允许其他进程使用代理操作它们。

`Manager()` 返回的管理器支持类型：`list`, `dict` , `Namespace`, `Lock`, `RLock`, `Semaphore`, `BoundedSemaphore`, `Condition`, `Event`, `Barrier`, `Queue `, `Value `和`Array`。例如：

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

        print(d)
        print(l)
```

```
{0.25: None, 1: '1', '2': 2}
[9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
```

使用服务进程的管理器比使用共享内存对象更灵活，因为它们可以支持任意对象类型；单个管理器可以通过网络由不同计算机上的进程共享。但是它们比使用共享内存更慢。

## 代理对象

## 进程池

如果要启动大量的子进程，可以用进程池的方式批量创建子进程：

```python
from multiprocessing import Pool
import os, time, random

def long_time_task(name):
    print('Run task %s (%s)...' % (name, os.getpid()))
    start = time.time()
    time.sleep(random.random() * 3)
    end = time.time()
    print('Task %s runs %0.2f seconds.' % (name, (end - start)))

if __name__=='__main__':
    print('Parent process %s.' % os.getpid())
    p = Pool(4)
    for i in range(5):
        p.apply_async(long_time_task, args=(i,))  # 提交5个任务
    print('Waiting for all subprocesses done...')
    p.close()      # 关闭任务提交
    p.join()       # 等待所有子进程结束
    print('All subprocesses done.')
```

```
Parent process 669.
Waiting for all subprocesses done...
Run task 0 (671)...
Run task 1 (672)...
Run task 2 (673)...
Run task 3 (674)...
Task 2 runs 0.14 seconds.
Run task 4 (673)...         # 任务4不能立即执行,而要等待前面的某个task完成释放进程
Task 1 runs 0.27 seconds.
Task 3 runs 0.86 seconds.
Task 0 runs 1.41 sec——基于进程的并行onds.
Task 4 runs 1.91 seconds.
All subprocesses done.
```

## 编程指导

使用`multiprocessing`时，应遵循一些指导原则和习惯用法。

+ 避免共享状态

应尽可能避免在进程间传递大量数据，越少越好。

最好坚持使用队列或者管道进行进程间通信，而不是底层的同步原语。

+ 可序列化

保证所代理的方法的参数是可以序列化的。

+ 代理的线程安全性

不要在多线程中同时使用一个代理对象，除非你用锁保护它。（但在不同进程中使用相同的代理对象则没有问题。）

+ 使用join避免僵尸进程

在 Unix 上，如果一个进程执行完成但是没有被 join，就会变成僵尸进程。一般来说僵尸进程不会很多，因为每次新启动进程（或者`active_children()`被调用）时，所有已执行完成且没有被 join 的进程都会自动被 join，而且对一个执行完的进程调用`Process.is_alive()`也会 join 这个进程。尽管如此，对自己启动的进程显式调用 join 依然是最佳实践。

+ 继承优于序列化、反序列化

当使用 *spawn* 或者 *forkserver* 的启动方式时，`multiprocessing`中的许多类型都必须是可序列化的，这样子进程才能使用它们。但是通常我们都应该避免使用管道和队列发送共享对象到另外一个进程，而是重新组织代码，让那些需要访问这些对象的子进程可以直接将它们从父进程继承过来。

+ 避免杀死进程

使用`Process.terminate()`停止一个进程很容易导致这个进程正在使用的共享资源（如锁、信号量、管道和队列）损坏或者变得不可用，无法在其他进程中继续使用。

因此最好只对那些从来不使用共享资源的进程调用`Process.terminate()`。

+ join使用队列的进程

往队列放入数据的进程会一直等待直到队列中所有项被feeder线程传给底层管道。这意味着使用队列时，需要确保在进程join之前，所有存放到队列中的项都被其他进程、线程完全消费，否则不能保证这个写过队列的进程可以正常终止。

下面是一个会导致死锁的例子:

```python
from multiprocessing import Process, Queue

def f(q):
    q.put('X' * 1000000)

if __name__ == '__main__':
    queue = Queue()
    p = Process(target=f, args=(queue,))
    p.start()
    p.join()                    # this deadlocks
    obj = queue.get()
```

交换最后两行可以修复这个问题。

+ 显式传递资源给子进程

在Unix上，使用 *fork* 方式启动的子进程可以使用父进程中全局创建的共享资源。不过最好是显式将资源对象通过参数的形式传递给子进程。

除了（部分原因）让代码兼容 Windows 以及其他的进程启动方式外，这种形式还保证了在子进程生命期这个对象是不会被父进程垃圾回收的。如果父进程中的某些对象被垃圾回收会导致资源释放，这就变得很重要。

所以对于实例：

```python
from multiprocessing import Process, Lock

def f():
    ... do something using "lock" ...

if __name__ == '__main__':
    lock = Lock()
    for i in range(10):
        Process(target=f).start()
```

应当重写为：

```python
from multiprocessing import Process, Lock

def f(l):
    ... do something using "l" ...

if __name__ == '__main__':
    lock = Lock()
    for i in range(10):
        Process(target=f, args=(lock,)).start()
```

# 子进程

`subprocess`模块可以让我们非常方便地启动一个子进程，然后控制其输入和输出。

下面的例子演示了如何在Python代码中执行shell命令，这和命令行直接运行的结果是一样的：

```python
import subprocess

print('$ echo hello')
cp = subprocess.run(['echo', 'hello'])
print('Exit code:', cp.returncode)
```

```
$ echo hello
hello
Exit code: 0
```

# 多线程

线程是操作系统直接支持的执行单元，因此高级语言通常都内置多线程的支持。并且Python的线程是真正的Posix Thread，而不是模拟出来的线程。

Python的标准库提供了两个模块：`_thread`和`threading`，`_thread`是低级模块，`threading`是高级模块，对`_thread`进行了封装。绝大多数情况下，我们只需要使用`threading`这个高级模块。

启动一个线程就是把一个函数传入并创建`Thread`实例，然后调用`start()`开始执行：

```python
import time, threading

# 新线程执行的代码:
def loop():
    print('thread %s is running...' % threading.current_thread().name)
    n = 0
    while n < 5:
        n = n + 1
        print('thread %s >>> %s' % (threading.current_thread().name, n))
        time.sleep(1)
    print('thread %s ended.' % threading.current_thread().name)

print('thread %s is running...' % threading.current_thread().name)
t = threading.Thread(target=loop, name='LoopThread')  # 创建新线程
                   # 传入新线程运行的函数和线程名
t.start()
t.join()
print('thread %s ended.' % threading.current_thread().name)
```

```
thread MainThread is running...  # 进程的默认线程称为MainThread
thread LoopThread is running...
thread LoopThread >>> 1
thread LoopThread >>> 2
thread LoopThread >>> 3
thread LoopThread >>> 4
thread LoopThread >>> 5
thread LoopThread ended.
thread MainThread ended.
```

## Lock

> 参考：
>
> [Python的GIL是什么鬼，多线程性能究竟如何](http://cenalulu.github.io/python/gil-in-python/)

当多个线程都要修改共享的变量时，可以创建一个`Lock`对象，使每次只能有一个线程进行修改。来看下面的例子：

```python
import time, threading

# 假定这是你的银行存款:
balance = 0
lock = threading.Lock()

def change_it(n):
    # 先存后取，结果应该为0:
    global balance
    balance = balance + n
    balance = balance - n

def run_thread(n):
    for i in range(2000000):
        lock.acquire()
        try:
        	change_it(n)
        finally:              # absolutely release
            lock.release()

t1 = threading.Thread(target=run_thread, args=(5,))
t2 = threading.Thread(target=run_thread, args=(8,))
t1.start()
t2.start()
t1.join()
t2.join()
print(balance)
```

当多个线程同时执行`lock.acquire()`时，只有一个线程能够获得锁然后继续执行，其它线程会阻塞直到有线程执行`lock.release()`释放锁，此时阻塞的线程之一会获得锁然后继续执行。

获得锁的线程在操作结束之后一定要释放锁，否则其它阻塞的线程将永远等待下去，成为死线程。因此我们用`try...finally`来确保锁一定会被释放。

锁的好处就是确保了某段关键代码只能由一个线程从头到尾完整地执行。坏处当然也很多，首先是阻止了多线程并发执行，包含锁的某段代码实际上只能以单线程模式执行；其次，由于可以存在多个锁，不同的线程持有不同的锁，并试图获取对方持有的锁时，可能会造成多个线程全部阻塞，即死锁。

## 全局解释器锁

如果我们执行一个死循环脚本：

```python
x = 0
while True:
    x = x*2
```

监控CPU使用率，可以看到死循环进程会占用100%的CPU。

此时再开一个terminal，也执行上面的死循环，可以看到两个死循环进程各占用100%的CPU。

在多核CPU系统中，两个Python进程可以并行执行。要想把N个核心全部跑满，就需要启动N个死循环进程。

但是如果启动两个死循环线程：

```python
import threading

def loop():
    x = 0
    while True:
        x = x*2

for i in range(2):
    t = threading.Thread(target=loop)
    t.start()
```

我们发现线程之间没有并行，进程只占用了100%的CPU。实际上，如果用C或Java来改写相同的死循环，则可以跑满两个核心，但是Python不行。

原因在于，尽管Python的线程是Posix线程，但CPython解释器执行代码时有一个**全局解释器(Global Interpreter Lock, GIL)**锁，任何Python线程执行前必须先获得GIL锁，然后每执行100条字节码，解释器就自动释放GIL锁，让其它线程有机会执行。因此多线程在Python中只能交替执行，，即使100个线程跑在100核CPU上，也只能用到1个核。

GIL是Python解释器设计的历史遗留问题，通常我们用的解释器都是官方实现的CPython。如果想要真正利用多核的计算资源，可以使用没有GIL的解释器运行多线程，或者直接改用多进程。但是，如果你想要<u>同时运行多个 I/O 密集型任务，则多线程仍然是一个合适的模型</u>。