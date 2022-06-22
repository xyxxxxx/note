# threading——基于线程的并行

在 CPython 中，由于存在全局解释器锁，同一时刻只有一个线程可以执行 Python 代码（虽然某些性能导向的库可能会去除此限制）。 如果你想让你的应用更好地利用多核心计算机的计算资源，推荐你使用 `multiprocessing` 或 `concurrent.futures.ProcessPoolExecutor`。 但是，如果你想要同时运行多个 I/O 密集型任务，则多线程仍然是一个合适的模型。

## active_count()

返回当前存活的 `Thread` 对象的数量。

## current_thread()

返回当前调用者的控制线程的 `Thread` 对象。

## main_thread()

返回主 `Thread` 对象。

## Thread

## Lock

原始锁处于"锁定"或者"非锁定"两种状态之一。它有两个基本方法，`acquire()` 和 `release()`。当状态为非锁定时，`acquire()` 将状态改为锁定并立即返回；当状态是锁定时，`acquire()` 将阻塞至其他线程调用 `release()` 将其改为非锁定状态，然后 `acquire()` 重置其为锁定状态并返回。`release()` 只在锁定状态下调用，将状态改为非锁定并立即返回。如果尝试释放一个非锁定的锁，则会引发 `RuntimeError` 异常。

原始锁在创建时为非锁定状态。当多个线程在 `acquire()` 阻塞，然后 `release()` 重置状态为未锁定时，只有一个线程能继续执行；至于哪个线程继续执行则没有定义，并且会根据实现而不同。
