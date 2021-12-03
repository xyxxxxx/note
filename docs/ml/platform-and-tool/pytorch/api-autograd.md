# torch.autograd

`torch.autograd` 提供了实现自动微分的类和函数。

## backward

```python
torch.autograd.backward(tensors: Union[torch.Tensor, Sequence[torch.Tensor]], grad_tensors: Union[torch.Tensor, Sequence[torch.Tensor], None] = None, retain_graph: Optional[bool] = None, create_graph: bool = False, grad_variables: Union[torch.Tensor, Sequence[torch.Tensor], None] = None) → None
# tensors       计算此张量对所有叶节点的梯度
# grad_tensors  tensors是非标量时,与此张量作内积以转换为标量.形状必须与tensors相同
# retain_graph  若为False,计算图在梯度计算完成后(backward()返回后)即被释放.注意在几
#                  乎所有情形下将其设为True都是不必要的,因为总有更好的解决方法
# create_graph  若为True,可以计算更高阶梯度
```

计算 `tensors` 对所有计算图叶节点的梯度。

图使用链式法则进行微分。如果 `tensors` 不是一个标量（即拥有多于一个元素）且需要计算梯度，函数就需要指定 `grad_tensors`，`tensors` 与 `grad_tensors` 作内积从而转换为一个标量。`grad_tensors` 必须与 `tensors` 的形状相同。

该函数会在叶节点累计梯度，因此你需要在每次迭代时先将其归零。

## grad

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

计算 `outputs` 对 `inputs` 的梯度并返回。

## no_grad

禁用梯度计算的上下文管理器。

在此模式下，所有运算的结果都有 `requires_grad = False`，即使输入有`requires_grad = True`。当你确定不会调用`tensor.backward()`时，禁用梯度计算可以降低结果本来为`requires_grad = True` 的运算这一部分的内存消耗。

可以作为装饰器使用。

```python
>>> x = torch.tensor([1], requires_grad=True)
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

## enable_grad

启用梯度计算的上下文管理器。

用于在 `no_grad` 或 `set_grad_enabled` 禁用梯度计算的环境下启用梯度计算。

可以作为装饰器使用。

```python
>>> x = torch.tensor([1], requires_grad=True)
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

## set_grad_enabled

设置梯度计算开或关的上下文管理器。

根据其参数 `mode` 启用或禁用梯度计算。可以用作上下文管理器或函数。

```python
>>> x = torch.tensor([1], requires_grad=True)
>>> is_train = False
>>> with torch.set_grad_enabled(is_train):
...   y = x * 2
>>> y.requires_grad
False
>>> torch.set_grad_enabled(True)
>>> y = x * 2
>>> y.requires_grad
True
>>> torch.set_grad_enabled(False)
>>> y = x * 2
>>> y.requires_grad
False
```

## Function

> 参考：pytorch-tutorial-torch.autograd的简单入门-Function

记录运算历史，定义运算导数公式。

对（`requires_grad=True` 的）张量的每一次运算都会创建一个新的 `Function` 对象，用于执行计算、记录过程。运算历史会保留在由 `Function` 对象构成的有向无环图的形式中，其中边表示数据的依赖关系（`input <- output`）。当调用`backward`时，计算图按照拓扑序进行处理，依次调用每个`Function`对象的`backward()` 方法，传递返回的梯度值。

`Function` 类的一般使用方法是创建子类并定义新操作，这是扩展 `torch.autograd` 的推荐方法。

```python
class Exp(Function):

    # 执行运算
    # 第一个参数必须接受一个context,用于保存张量,在backward中取回
    @staticmethod
    def forward(ctx, i):
        result = i.exp()
        ctx.save_for_backward(result)  # 保存张量,这里为e**i
        return result
    
    # 定义该运算的导数公式
    # 第一个参数必须接受一个context,用于取回张量;属性ctx.needs_input_grad是一个
    #    布尔类型的元组,表示哪些输入需要计算梯度
    # 之后的每一个参数对应(损失)对相应输出的梯度
    # 返回的每一个变量对应(损失)对相应输入的梯度
    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_tensors    # 取回张量
        return grad_output * result    # 计算梯度并返回

# Use it by calling the apply method:
output = Exp.apply(input)
```
