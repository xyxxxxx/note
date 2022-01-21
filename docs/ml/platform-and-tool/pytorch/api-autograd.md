# torch.autograd

`torch.autograd` 提供了实现任意标量值函数的自动微分的类和函数。

## 核心功能

### backward()

```python
torch.autograd.backward(tensors: Union[torch.Tensor, Sequence[torch.Tensor]], grad_tensors: Union[torch.Tensor, Sequence[torch.Tensor], None] = None, retain_graph: Optional[bool] = None, create_graph: bool = False, grad_variables: Union[torch.Tensor, Sequence[torch.Tensor], None] = None) → None
# tensors       计算此张量对所有叶节点的梯度
# grad_tensors  tensors是非标量时,与此张量作内积以转换为标量.形状必须与tensors相同
# retain_graph  若为False,计算图在梯度计算完成后(backward()返回后)即被释放.注意在几
#                  乎所有情形下将其设为True都是不必要的,因为总有更好的解决方法
# create_graph  若为True,则可以计算更高阶梯度
```

计算给定张量对计算图中所有叶节点的梯度。

图使用链式法则进行微分。如果 `tensors` 不是一个标量（即拥有多于一个元素）且需要计算梯度，函数就需要指定 `grad_tensors`，`tensors` 与 `grad_tensors` 作内积从而转换为一个标量。`grad_tensors` 必须与 `tensors` 的形状相同。

此函数会在叶节点累积梯度，因此你需要在每次迭代时先将其归零。

### grad()

```python
torch.autograd.grad(outputs: Union[torch.Tensor, Sequence[torch.Tensor]], inputs: Union[torch.Tensor, Sequence[torch.Tensor]], grad_outputs: Union[torch.Tensor, Sequence[torch.Tensor], None] = None, retain_graph: Optional[bool] = None, create_graph: bool = False, only_inputs: bool = True, allow_unused: bool = False) → Tuple[torch.Tensor, ...]
# outputs       计算outputs对inputs的梯度
# inputs
# grad_outputs  类似于backward()的grad_tensors
# retain_graph  同backward()
# create_graph  同backward()
# only_inputs   若为True,则只计算对inputs的梯度;若为False,则计算对所有叶节点的梯度,
#                  并将梯度累加到.grad上
# allow_unused  若为False,则指定的inputs没有参与outputs的计算将视作一个错误
```

计算并返回 `outputs` 对 `inputs` 的梯度。

## 功能性高级 API

### functional.jacobian()

### functional.hessian()

## 局部禁用梯度计算

### no_grad()

禁用梯度计算的上下文管理器。也可用作装饰器。

在此模式下，所有运算的结果都有 `requires_grad = False`，即使参与运算的张量有 `requires_grad = True`。对于模型推断等确定不会调用 `Tensor.backward()` 的情形，禁用梯度计算可以降低 `requires_grad = True` 变量参与的运算的内存消耗。

此上下文管理器是线程局部的，它不会影响到其它线程的计算。

```python
>>> x = torch.tensor([1.], requires_grad=True)
>>> with torch.no_grad():
...   y = x * 2
>>> y.requires_grad
False
>>> @torch.no_grad()
... def doubler(x):
...     return x * 2
>>> z = doubler(x)
>>> z.requires_grad
False
```

### enable_grad()

启用梯度计算的上下文管理器。也可用作装饰器。

用于在 `no_grad()` 或 `set_grad_enabled()` 禁用梯度计算的环境下启用梯度计算。

此上下文管理器是线程局部的，它不会影响到其它线程的计算。

```python
>>> x = torch.tensor([1.], requires_grad=True)
>>> with torch.no_grad():
...   with torch.enable_grad():
...     y = x * 2
>>> y.requires_grad
True
>>> y.backward()
>>> x.grad
>>> @torch.enable_grad()
... def doubler(x):
...     return x * 2
>>> with torch.no_grad():
...     z = doubler(x)
>>> z.requires_grad
True
```

### set_grad_enabled()

启用或禁用梯度计算的上下文管理器。也可直接调用。

此上下文管理器是线程局部的，它不会影响到其它线程的计算。

```python
>>> x = torch.tensor([1], requires_grad=True)
>>> is_train = False
>>> with torch.set_grad_enabled(is_train):  # 作为上下文管理器
...   y = x * 2
>>> y.requires_grad
False
>>> 
>>> torch.set_grad_enabled(True)            # 直接调用
>>> y = x * 2
>>> y.requires_grad
True
>>> torch.set_grad_enabled(False)
>>> y = x * 2
>>> y.requires_grad
False
```

### is_grad_enabled()

若当前启用了梯度计算，返回 `True`。

### inference_mode()

启用或禁用推断模式的上下文管理器。也可用作装饰器。

推断模式是一种新的上下文，类似于 `no_grad()` 的禁用梯度计算，用于你确定运算不会使用 autograd 的情形下。此模式下运行的代码可以得到更好的性能，通过禁用视图追踪以及 version counter bumps。

此上下文管理器是线程局部的，它不会影响到其它线程的计算。

```python
>>> x = torch.tensor([1], requires_grad=True)
>>> with torch.inference_mode():
...   y = x * 2
>>> y.requires_grad
False
>>> 
>>> @torch.inference_mode()
... def doubler(x):
...     return x * 2
>>> z = doubler(x)
>>> z.requires_grad
False
```

### is_inference_mode_enabled()

若当前启用了推断模式，返回 `True`。

## 默认梯度布局

## 张量的原位操作

在 autograd 中支持原位操作是一件非常困难的事情，并且在绝大部分情况下我们都不鼓励使用原位操作。autograd 激进的缓冲区释放和重用策略已经使得 autograd 非常高效，已经很少再有原位操作能够事实上大幅降低内存使用的情形。除非你在非常严重的内存压力下操作，否则你都完全不需要使用原位操作。

### 原位正确性检查

所有的张量都记录了它们所应用过的原位操作，如果 autograd 的实现检测到计算图中的一个张量在之后进行了原位修改，那么在反向计算开始时将引发一个错误。这保证了如果你进行原位操作并且没有看到任何错误，那么你可以确定所有计算得到的梯度都是正确的。

## Function

> 参考：[Function](pytorch.md#Function)

```python
class torch.autograd.Function(*args, **kwargs)
```

用于创建自定义 `autograd.Function` 的基类。

对（`requires_grad=True` 的）张量的每一次运算都会创建一个新的 `Function` 对象，用于执行计算、记录过程。运算历史会保留在由 `Function` 对象构成的有向无环图的形式中，其中边表示数据的依赖关系（`input <- output`）。当调用 `backward()` 时，计算图按照拓扑序进行处理，依次调用每个 `Function` 对象的 `backward()` 方法，传递返回的梯度值。

要创建一个自定义 `Function`，继承此类并实现静态方法 `forward()` 和 `backward()`。在前向计算中使用此自定义操作时，调用类方法 `apply()` 而不是直接调用 `forward()`。

为保证正确性和最佳性能，确保你对 `ctx` 调用了正确的方法，并且使用 `autograd.gradcheck()` 验证了反向计算函数。

```python
class Exp(Function):

    # 执行运算
    # 必须接受一个context作为第一个参数,用于保存张量,将在backward()中取回
    @staticmethod
    def forward(ctx, i):
        result = i.exp()
        ctx.save_for_backward(result)  # 保存张量,这里为e**i
        return result
    
    # 定义该运算的微分(导数)公式
    # 必须接受一个context作为第一个参数,用于取回在forward()中保存的张量
    # 属性`ctx.needs_input_grad`是一个布尔类型的元组,表示哪些输入需要计算梯度
    # 之后的每一个参数对应于(损失)对相应输出的梯度
    # 返回的每一个变量对应于(损失)对相应输入的梯度.对于不需要计算梯度的输入,直接相应地返回None
    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_tensors    # 取回张量
        return grad_output * result    # 计算梯度并返回

# 调用`apply`方法来使用
output = Exp.apply(input)
```

## 性能分析

## 保存张量的钩子

Some operations need intermediary results to be saved during the forward pass in order to execute the backward pass. You can define how these saved tensors should be packed / unpacked using hooks. A common application is to trade compute for memory by saving those intermediary results to disk or to CPU instead of leaving them on the GPU. This is especially useful if you notice your model fits on GPU during evaluation, but not training. Also see Hooks for saved tensors.

### graph.saved_tensors_hooks()
