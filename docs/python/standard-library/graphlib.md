# graphlib——操作类似图的结构的功能

## TopologicalSorter

```python
class graphlib.TopologicalSorter(graph=None)
```

提供以拓扑方式对可哈希节点的图进行排序的功能。

拓扑排序是指图中顶点的线性排序，使得对于每条从顶点 u 到顶点 v 的有向边 u -> v，顶点 u 都排在顶点 v 之前。例如，图的顶点可以代表要执行的任务，而边代表某一个任务必须在另一个任务之前执行的约束条件；在这个例子中，拓扑排序只是任务的有效序列。完全拓扑排序当且仅当图不包含有向环，也就是说为有向无环图时，完全拓扑排序才是可能的。

如果提供了可选的 *graph* 参数则它必须为一个表示有向无环图的字典，其中的键为节点而值为包含图中该节点的所有上级节点（即具有指向键中的值的边的节点）的可迭代对象。额外的节点可以使用 `add()` 方法添加到图中。

在通常情况下，对给定的图执行排序所需的步骤如下:

1. 通过可选的初始图创建一个 `TopologicalSorter` 的实例。
1. 添加额外的节点到图中。
1. 在图上调用 `prepare()`。
1. 当 `is_active()` 为 `True` 时，迭代 `get_ready()` 所返回的节点并加以处理。完成处理后在每个节点上调用 `done()`。

在只需要对图中的节点进行立即排序并且不涉及并行性的情况下，可以直接使用便捷方法 `static_order()`:

```python
>>> graph = {"D": {"B", "C"}, "C": {"A"}, "B": {"A"}}
>>> ts = TopologicalSorter(graph)
>>> tuple(ts.static_order())
('A', 'C', 'B', 'D')
```

### add()

```python
add(node, *predecessors)
```

将一个新节点及其上级节点添加到图中。*node* 以及 *predecessors* 中的所有元素都必须为可哈希对象。

如果附带相同的 *node* 多次调用，则依赖项的集合将为所有被传入的 *predecessors* 的并集。

可以添加不带依赖项的节点 (即不提供 *predecessors*) 或者重复提供依赖项。如果有先前未提供的节点包含在 *predecessors* 中则它将被自动添加到图中并且不带有上级节点。

### prepare()

将图标记为已完成并检查图中是否存在环。如果检测到任何环，则引发 `CycleError`。在调用此函数后，图将无法再修改，即不能再使用 `add()` 添加更多的节点。

### is_active()

### done()

### get_ready()

### static_order()

返回一个迭代器，它将按照拓扑顺序来迭代所有节点。当使用此方法时，`prepare()` 和 `done()` 不应被调用。此方法等价于：

```python
def static_order(self):
    self.prepare()
    while self.is_active():
        node_group = self.get_ready()
        yield from node_group
        self.done(*node_group)
```
