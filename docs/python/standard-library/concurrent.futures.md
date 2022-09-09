# concurrent.futures——启动并行任务

`concurrent.futures` 模块提供异步执行可调用对象的高级接口。

异步执行可以由 `ThreadPoolExecutor` 使用线程或由 `ProcessPoolExecutor` 使用单独的进程来实现。两者实现了相同的接口，由抽象类 `Executor` 定义。

## Executor

```python
class concurrent.futures.Executor
```

提供异步执行调用方法的抽象类。应通过它的具体子类调用，而不是直接调用。

### submit()

```python
submit(fn, *args, **kwargs)
```

调度可调用对象 *fn*，以 `fn(*args **kwargs)` 的方式执行并返回 `Future` 对象代表该可调用对象的执行：

```python
with ThreadPoolExecutor(max_workers=1) as executor:
    future = executor.submit(pow, 323, 1235)
    print(future.result())
```

### map()

```python
map(func, *iterables, timeout=None, chunksize=1)
```

类似于 `map(func, *iterables)`，除了以下两点：

* *iterables* 立即返回其元素而不是延迟返回。
* *func* 是异步执行的，对 *func* 的多个调用可能是并发执行。

如果从调用 `Executor.map()` 开始经过 timeout 秒后，调用 `__next__()` 返回的结果仍不可用，那么返回的迭代器将引发 `concurrent.futures.TimeoutError`。*timeout* 可以是整数或浮点数。如果 *timeout* 没有指定或为 None，则没有超时限制。

如果 *func* 调用引发一个异常，则该异常将在从迭代器中取回其返回值时被引发。

使用 *ProcessPoolExecutor* 时，此方法会将 *iterables* 切分成若干块并将它们作为独立的任务提交到执行池中。这些块的（大致）数量可以通过设置 *chunksize* 为正整数来指定。对于非常长的可迭代对象，使用大的 *chunksize* 值相比默认值 1 能够显著提升性能。*chunksize* 对于 `ThreadPoolExecutor` 没有作用。

### shutdown()

```python
shutdown(wait=True)
```

当正在挂起的 future 对象执行完毕后，向执行器发送其应释放所有正在使用的任何资源的信号。在执行器已经关闭后调用 `submit()` 和 `map()` 将会引发 `RuntimeError`。

如果 *wait* 为 True 则此方法只有在所有挂起的 future 对象执行完毕并且释放已分配的资源后才会返回。如果 *wait* 为 False，此方法立即返回，所有挂起的 future 对象在执行完毕后会释放已分配的资源。不管 *wait* 的值是什么，整个 Python 程序将等到所有挂起的 future 对象全部执行完毕后才退出。

如果使用 `with` 语句，你可以避免显式调用这个方法，该语句将会关闭 `Executor`（就如同 *wait* 设为 True 调用此方法）：

```python
import shutil
with ThreadPoolExecutor(max_workers=4) as e:
    e.submit(shutil.copy, 'src1.txt', 'dest1.txt')
    e.submit(shutil.copy, 'src2.txt', 'dest2.txt')
    e.submit(shutil.copy, 'src3.txt', 'dest3.txt')
    e.submit(shutil.copy, 'src4.txt', 'dest4.txt')
```

## ThreadPoolExecutor

`ThreadPoolExecutor` 类是 `Executor` 的子类，它使用线程池来异步执行调用。

当可调用对象关联了一个 `Future` 又等待另一个 `Future` 的结果时就会发生死锁，例如:

```python
import time
def wait_on_b():
    time.sleep(5)
    print(b.result())  # b will never complete because it is waiting on a.
    return 5

def wait_on_a():
    time.sleep(5)
    print(a.result())  # a will never complete because it is waiting on b.
    return 6

executor = ThreadPoolExecutor(max_workers=2)
a = executor.submit(wait_on_b)
b = executor.submit(wait_on_a)
```

以及：

```python
def wait_on_future():
    f = executor.submit(pow, 5, 2)
    # This will never complete because there is only one worker thread and
    # it is executing this function.
    print(f.result())

executor = ThreadPoolExecutor(max_workers=1)
executor.submit(wait_on_future)
```

```python
class concurrent.futures.ThreadPoolExecutor(max_workers=None, thread_name_prefix='', initializer=None, initargs=())
```

使用至多 *max_workers* 个线程的线程池来异步执行调用的 `Executor` 子类。

*initializer* 是在每一个工作线程启动时调用的一个可选的可调用对象。*initargs* 是传入 *initialize* 的参数元组。如果 *initializer* 引发了一个异常，那么所有当前挂起的任务以及任何向线程池提交更多任务的尝试都将引发 `BrokenThreadPool`。

*thread_name_prefix* 参数允许用户控制由线程池创建的工作线程（`threading.Thread` 对象）的名称以方便调试。

现在 `ThreadPoolExecutor` 在启动 *max_workers* 个工作线程之前也会重用空闲的工作线程。

3.5 版本更改：如果 *max_workers* 为 None 或没有指定，将默认为机器处理器数量的 5 倍，这里假定 `ThreadPoolExecutor` 侧重于 I/O 操作而不是 CPU 运算，并且工作线程的数量应多于 `ProcessPoolExecutor` 的工作进程的数量。

3.8 版本更改：*max_workers* 的默认值更改为 `min(32, os.cpu_count() + 4)`。这个默认值会保留至少 5 个工作线程用于 I/O 密集型任务。对于那些释放了 GIL 的 CPU 密集型任务，它最多会使用 32 个 CPU 核心，以避免在多核机器上不知不觉地使用大量资源。

```python
import concurrent.futures
import urllib.request

URLS = ['http://www.foxnews.com/',
        'http://www.cnn.com/',
        'http://europe.wsj.com/',
        'http://www.bbc.co.uk/',
        'http://some-made-up-domain.com/']

# Retrieve a single page and report the URL and contents
def load_url(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as conn:
        return conn.read()

# We can use a with statement to ensure threads are cleaned up promptly
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    # Start the load operations and mark each future with its URL
    future_to_url = {executor.submit(load_url, url, 60): url for url in URLS}
    for future in concurrent.futures.as_completed(future_to_url):
        url = future_to_url[future]
        try:
            data = future.result()
        except Exception as exc:
            print('%r generated an exception: %s' % (url, exc))
        else:
            print('%r page is %d bytes' % (url, len(data)))
```

## ProcessPoolExecutor

`ProcessPoolExecutor` 类是 `Executor` 的子类，它使用进程池来异步执行调用。`ProcessPoolExecutor` 使用 `multiprocessing` 模块，这使它可以绕过 GIL，但也意味着只能处理和返回可封存的对象。

`__main__` 模块必须可以被工作进程导入。这意味着 `ProcessPoolExecutor` 不能工作在交互式解释器中。

从提交给 *ProcessPoolExecutor* 的可调用对象中调用 *Executor* 或 *Future* 的方法会导致死锁。

```python
class concurrent.futures.ProcessPoolExecutor(max_workers=None, mp_context=None, initializer=None, initargs=())
```

使用至多 *max_workers* 个进程的进程池来异步执行调用的 `Executor` 子类。如果 *max_workers* 为 None 或未给出，它将默认为机器的处理器数量，但至多为 61，即使存在更多的处理器；如果 *max_workers* 小于等于 0，将引发 `ValueError`。在 Windows 上，*max_workers* 必须小于等于 61，否则将引发 `ValueError`。*mp_context* 可以是一个多进程上下文或 None，它将被用来启动工作进程。如果 *mp_context* 为 None 或未给出，则使用默认的多进程上下文。

*initializer* 是在每一个工作进程启动时调用的一个可选的可调用对象。*initargs* 是传入 *initialize* 的参数元组。如果 *initializer* 引发了一个异常，那么所有当前挂起的任务以及任何向进程池提交更多任务的尝试都将引发 `BrokenThreadPool`。

```python
import concurrent.futures
import math

PRIMES = [
    112272535095293,
    112582705942171,
    112272535095293,
    115280095190773,
    115797848077099,
    1099726899285419]

def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False

    sqrt_n = int(math.floor(math.sqrt(n)))
    for i in range(3, sqrt_n + 1, 2):
        if n % i == 0:
            return False
    return True

def main():
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for number, prime in zip(PRIMES, executor.map(is_prime, PRIMES)):
            print('%d is prime: %s' % (number, prime))

if __name__ == '__main__':
    main()
```

## Future

`Future` 类将可调用对象封装为异步执行。`Future` 实例由 `Executor.submit()` 创建。

### cancel()

尝试取消调用。如果调用正在执行或已结束运行而不能被取消则该方法将返回 False，否则调用会被取消且该方法将返回 True。

### cancelled()

如果调用被成功取消则返回 True。

### running()

如果调用正在执行而不能被取消则返回 True 。

### done()

如果调用被成功取消或正常结束则返回 True。

### result()

```python
result(timeout=None)
```

返回由调用返回的值。如果调用还没有完成那么这个方法将等待 *timeout* 秒。如果在 *timeout* 秒内没有执行完成，将引发 `concurrent.futures.TimeoutError`。*timeout* 可以是整数或浮点数。如果 *timeout* 没有指定或为 None，那么等待时间没有限制。

如果 `Future` 对象在完成前被取消则引发 `CancelledError`。

如果调用引发了一个异常，此方法也会引发相同的异常。

### exception()

```python
exception(timeout=None)
```

返回由调用引发的异常。如果调用还没有完成那么这个方法将等待 *timeout* 秒。如果在 *timeout* 秒内没有执行完成，将引发 `concurrent.futures.TimeoutError`。*timeout* 可以是整数或浮点数。如果 *timeout* 没有指定或为 None，那么等待时间没有限制。

如果 `Future` 对象在完成前被取消则引发 `CancelledError`。

如果调用正常完成则返回 None。

### add_done_callback()

```python
add_done_callback(fn)
```

附加可调用对象 *fn* 到 `Future` 对象。当 `Future` 对象被取消或完成运行时，将会调用 *fn*，而当前 `Future` 对象将作为它的唯一参数。

附加的可调用对象以它们被添加的顺序调用，并且总是在添加它们的进程的线程中调用。如果可调用对象引发一个 `Exception` 子类，它会被记录下来并被忽略掉。如果可调用对象引发一个 `BaseException` 子类，则行为没有定义。

如果 `Future` 对象已经完成或已取消，*fn* 会被立即调用。

## wait()

## as_completed()
