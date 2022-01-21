# torch.multiprocessing

`torch.multiprocessing` 是对原始 `multiprocessing` 模块的一个包装。It registers custom reducers，that use shared memory to provide shared views on the same data in different processes.一旦张量被移动到共享内存（shared_memory）中，它就能被发送到其它进程而无需复制。

此 API 100% 兼容原始模块，你完全可以将 `import multiprocessing` 改为 `import torch.multiprocessing` 以将所有张量送入队列或通过其它机制分享。

由于此 API 与原始 API 的相似性，请参考 `multiprocessing` 包的文档以获取更多细节。

## 共享CUDA张量

在进程间共享 CUDA 张量使用 `spawn` 或 `forkserver` 启动方法。

不同于 CPU 张量，只要接收进程持有的是张量的副本，发送进程就需要保留原始张量。

1. 消费者应尽快释放内存

   ```python
   ## Good
   x = queue.get()
   # do somethings with x
   del x
   ```

   ```python
   ## Bad
   x = queue.get()
   # do somethings with x
   # do everything else (producer have to keep x in memory)
   ```

2. 保持生产者进程运行直到所有消费者退出。这将防止生产者释放消费者依然在使用的张量内存。

   ```python
   ## producer
   # send tensors, do something
   event.wait()
   ```

   ```python
   ## consumer
   # receive tensors and use them
   event.set()
   ```

3. 不要直接转发张量：

   ```python
   # not going to work
   x = queue.get()
   queue_2.put(x)
   ```

   ```python
   # you need to create a process-local copy
   x = queue.get()
   x_clone = x.clone()
   queue_2.put(x_clone)
   ```

   ```python
   # putting and getting from the same queue in the same process will likely end up with segfault
   queue.put(tensor)
   x = queue.get()
   ```

   

## 共享策略（CPU张量）

### `file_descriptor`

### `file_system`

## spawn()

可以通过创建 `Process` 实例以启动若干子进程，执行特定函数，然后调用 `join` 等待其完成。此方法在处理单个子进程时工作得很好，但处理多个子进程时就显露了潜在问题，亦即：以特定顺序 `join` 进程默认了它们会按照该顺序终止。如果事实上没有按照这个顺序，例如在 `join` 第一个进程时后面的进程终止，则不会被注意到。此外，在这一过程中也没有误差传播的原生工具支持。

下面的 `spawn` 函数解决了上述问题，其支持误差传播、任意顺序终止，并且当检测到错误时可以动态地终止进程。

```python
torch.multiprocessing.spawn(fn, args=(), nprocs=1, join=True, daemon=False, 
start_method='spawn')
# fn        启动进程的作为进入点的调用函数,此函数必须定义在模块的顶级作用域以能够
#           被序列化和启动,这也是`multiprocessing`规定的必要条件
# args      传递给`fn`的参数列表.注意`fn`的第一个参数应为`rank`,由`spawn()`自动传入,
#           此参数传递的参数列表对应`fn`的第二个及以后的所有参数
# nprocs    启动的进程数
# join      join所有进程并阻塞
# daemon    启动进程的守护进程标识.若为`True`,则将创建守护进程

# 若join=True,返回None;否则返回ProcessContext
```

启动 `nprocs` 个进程，以使用 `args` 参数运行 `fn` 函数。

如果进程中的任意一个以非零退出状态退出，则剩余进程将被杀掉，并且抛出一个进程退出原因的异常。
