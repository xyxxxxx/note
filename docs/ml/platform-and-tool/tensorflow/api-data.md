# tf.data

## Dataset

数据集类型，用于构建描述性的、高效的输入流水线。`Dataset` 实例的使用通常遵循以下模式：

1. 根据输入数据创建一个源数据集
2. 应用数据集变换以预处理数据
3. 迭代数据集并处理数据

迭代以流的方式进行，因此不会将整个数据集全部放到内存中。

> 以下方法均不是原位操作，即返回一个新的数据集，而不改变原数据集。

### apply()

为数据集应用一个变换函数。该变换函数通常是 `Dataset` 变换方法的组合。

```python
>>> ds = Dataset.range(100)
>>> def dataset_fn(ds):
  return ds.filter(lambda x: x < 5)
... 
>>> ds = ds.apply(dataset_fn)
>>> list(ds.as_numpy_iterator())
[0, 1, 2, 3, 4]
```

### as_numpy_iterator()

返回一个迭代器，其将数据集的所有元素都转换为 NumPy 数组。常用于检查数据集的内容。

```python
>>> ds = Dataset.range(5)
>>> list(ds.as_numpy_iterator())
[0, 1, 2, 3, 4]
```

### batch()

将数据集的连续元素组合为批。

```python
>>> list(Dataset.range(10).batch(3).as_numpy_iterator())
[array([0, 1, 2]), array([3, 4, 5]), array([6, 7, 8]), array([9])]
>>> list(Dataset.range(10).batch(3, drop_remainder=True).as_numpy_iterator())   # 丢弃达不到指定规模的最后一个批次
[array([0, 1, 2]), array([3, 4, 5]), array([6, 7, 8])]
```

### cache()

缓存数据集的元素到内存中。

返回的数据集在第一次迭代后，其元素将被缓存到内存或指定文件中；接下来的迭代将使用缓存的数据。

### cardinality()

返回数据集的元素个数。如果数据集的元素个数是无限或不能确定，则返回 `tf.data.INFINITE_CARDINALITY`（`-1`） 或 `tf.data.UNKNOWN_CARDINALITY`（`-2`）。

```python
>>> Dataset.range(100).cardinality()
<tf.Tensor: shape=(), dtype=int64, numpy=100>
>>>
>>> Dataset.range(100).repeat().cardinality()
<tf.Tensor: shape=(), dtype=int64, numpy=-1>
>>> Dataset.range(100).repeat().cardinality() == tf.data.INFINITE_CARDINALITY
<tf.Tensor: shape=(), dtype=bool, numpy=True>
>>>
>>> Dataset.range(100).filter(lambda x: True).cardinality()
<tf.Tensor: shape=(), dtype=int64, numpy=-2>
>>> Dataset.range(100).filter(lambda x: True).cardinality() == tf.data.UNKNOWN_CARDINALITY
<tf.Tensor: shape=(), dtype=bool, numpy=True>
```

### concatenate()

将数据集与给定数据集进行拼接。

```python
>>> ds1 = Dataset.range(1, 4)
>>> ds2 = Dataset.range(4, 8)
>>> ds = ds1.concatenate(ds2)
>>> list(ds.as_numpy_iterator())
[1, 2, 3, 4, 5, 6, 7]
```

### enumerate()

对数据集的元素进行计数。

```python
>>> ds = Dataset.from_tensor_slices([2, 4, 6]).enumerate(start=1)
>>> list(ds.as_numpy_iterator())
[(1, 2), (2, 4), (3, 6)]
```

### filter()

使用给定函数过滤数据集的元素。

```python
>>> ds = Dataset.range(10)
>>> ds = ds.filter(lambda x: x % 2 == 0)
>>> list(ds.as_numpy_iterator())
[0, 2, 4, 6, 8]
```

### from_generator()

创建由生成器生成的元素构成的数据集。

### from_tensor_slices()

创建由指定张量的元素构成的数据集。

```python
>>> ds = Dataset.from_tensor_slices([1, 2, 3])                                # 张量
>>> list(ds.as_numpy_iterator())
[1, 2, 3]                                               # 张量的元素
>>> ds = Dataset.from_tensor_slices([[1, 2, 3], [4, 5, 6]])
>>> list(ds.as_numpy_iterator())
[array([1, 2, 3], dtype=int32), array([4, 5, 6], dtype=int32)]
>>> 
>>> ds = Dataset.from_tensor_slices(([1, 2], [3, 4], [5, 6]))                 # 张量构成的元组
>>> list(ds.as_numpy_iterator())
[(1, 3, 5), (2, 4, 6)]                                  # 元组,元素来自各张量
>>> ds = Dataset.from_tensor_slices(([1, 2, 3], ['a', 'b', 'a']))             # 应用:绑定数据和标签
>>> list(ds.as_numpy_iterator())
[(1, b'a'), (2, b'b'), (3, b'a')]

>>> ds = Dataset.from_tensor_slices({"a": [1, 2], "b": [3, 4], "c": [5, 6]})  # 张量构成的字典
>>> list(ds.as_numpy_iterator())
[{'a': 1, 'b': 3, 'c': 5}, {'a': 2, 'b': 4, 'c': 6}]    # 字典,元素来自各张量
```

### from_tensors()

创建由单个张量元素构成的数据集。

```python
>>> ds = Dataset.from_tensors([1, 2, 3])
>>> list(ds.as_numpy_iterator())
[array([1, 2, 3], dtype=int32)]
>>> 
>>> ds = Dataset.from_tensors([[1, 2, 3], [4, 5, 6]])
>>> list(ds.as_numpy_iterator())
[array([[1, 2, 3],
        [4, 5, 6]], dtype=int32)]
```

### list_files()

### map()

为数据集的每个元素应用一个函数。

```python
>>> ds = Dataset.from_tensor_slices([1, 2, 3])
>>> ds = ds.map(lambda x: x**2)
>>> list(ds.as_numpy_iterator())
[1, 4, 9]
```

### options()

返回数据集和其输入的选项。

### prefetch()

预取数据集的之后几个元素。

大部分数据集输入流水线都应该在最后调用 `prefetch()`，以使得在处理当前元素时后几个元素就已经准备好，从而改善时延和吞吐量，代价是使用额外的内存来保存这些预取的元素。

```python
>>> Dataset.range(5).prefetch(2)
<PrefetchDataset shapes: (), types: tf.int64>
>>> list(Dataset.range(5).prefetch(2).as_numpy_iterator())
[0, 1, 2, 3, 4]
```

### range()

创建由等差数列构成的数据集。

```python
>>> Dataset.range(5)
<RangeDataset shapes: (), types: tf.int64>
>>> list(Dataset.range(5).as_numpy_iterator())
[0, 1, 2, 3, 4]
>>> Dataset.range(1, 5, 2, output_type=tf.float32)
<RangeDataset shapes: (), types: tf.float32>
>>> list(Dataset.range(1, 5, 2, output_type=tf.float32).as_numpy_iterator())
[1.0, 3.0]
```

### reduce()

将数据集归约为一个元素。

```python
reduce(initial_state, reduce_func)
# initial_state    初始状态
# reduce_func      归约函数,将`(old_state, element)`映射到`new_state`.此函数会被不断地调用直到数据集被耗尽
```

```python
>>> Dataset.range(5).reduce(np.int64(0), lambda x, y: x + y)
<tf.Tensor: shape=(), dtype=int64, numpy=10>
>>> Dataset.range(5).reduce(np.int64(0), lambda x, _: x + 1)
<tf.Tensor: shape=(), dtype=int64, numpy=5>
```

### repeat()

重复数据集多次。

```python
>>> list(Dataset.range(5).repeat(3).as_numpy_iterator())
[0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4]

>>> pprint(list(Dataset.range(100).shuffle(100).repeat(2).batch(10).as_numpy_iterator()))
[array([63, 46, 66, 70, 96, 98,  7, 52, 50, 37]),
 array([45, 60, 69, 61, 84, 41, 22, 27, 14, 57]),
 array([ 9, 81,  8, 75,  4, 87, 18,  1, 51, 76]),
 array([65, 59, 90, 23, 39, 74, 26,  3, 20, 78]),
 array([91, 68, 85, 24, 53, 55, 16, 83, 94, 86]),
 array([92, 54, 48, 93, 38, 13, 67, 71, 82, 56]),
 array([40, 62,  5, 33, 15, 99, 32, 17,  6, 25]),
 array([80, 43, 28, 77, 21, 11, 72, 88, 44, 89]),
 array([31, 19, 12, 47, 36, 95, 29, 34,  0, 10]),
 array([97,  2, 42, 30, 64, 49, 79, 58, 35, 73]),
 array([19, 13, 77, 51, 56, 46, 67, 48, 27, 89]),
 array([26, 72, 54, 14,  2, 18, 25, 44, 63, 82]),
 array([24, 10, 23, 68, 64, 39, 28, 52, 53, 38]),
 array([69,  1, 15, 92, 31, 49, 60, 33, 81, 37]),
 array([65, 40, 20, 50,  5, 90, 34, 97, 84,  7]),
 array([99,  0, 75, 59, 98,  3, 85, 83, 61,  8]),
 array([73, 35, 36, 76, 55, 96, 91, 21, 94,  6]),
 array([70, 47, 16, 86, 11, 57, 95, 45, 74, 43]),
 array([87, 62, 29, 17, 12, 22, 66, 93, 88, 42]),
 array([79,  4, 80, 78, 32, 30, 71,  9, 58, 41])]
```

### shard()

从数据集中等间距地抽取样本以构成子集。此方法在进行分布式训练时十分有用。

```python
shard(num_shards, index)
# num_shards, index   每`num_shards`个样本抽取第`index`个
```

```python
>>> ds = Dataset.range(10)
>>> ds0 = ds.shard(num_shards=3, index=0)
>>> list(ds0.as_numpy_iterator())
[0, 3, 6, 9]
>>> ds1 = ds.shard(num_shards=3, index=1)
>>> list(ds1.as_numpy_iterator())
[1, 4, 7]
>>> ds2 = ds.shard(num_shards=3, index=2)
>>> list(ds2.as_numpy_iterator())
[2, 5, 8]
```

```python
# 准备分布式训练数据集

# 每个worker分到数据集的不固定子集并打乱(推荐)
>>> ds = Dataset.range(10).shuffle(100, seed=1).shard(num_shards=3, index=0).batch(2)
>>> list(ds.as_numpy_iterator())
[array([0, 1]), array([3, 4])]
>>> ds = Dataset.range(10).shuffle(100, seed=1).shard(num_shards=3, index=1).batch(2)
>>> list(ds.as_numpy_iterator())
[array([9, 2]), array([8])]
>>> ds = Dataset.range(10).shuffle(100, seed=1).shard(num_shards=3, index=2).batch(2)
>>> list(ds.as_numpy_iterator())
[array([5, 7]), array([6])]

# 每个worker分到数据集的固定子集并打乱
>>> ds = Dataset.range(10).shard(num_shards=3, index=0).shuffle(100).batch(2)
>>> list(ds.as_numpy_iterator())
[array([3, 0]), array([6, 9])]

# 每个worker分到数据集的固定子集并跨epoch打乱(不推荐)
>>> ds = Dataset.range(10).shard(num_shards=3, index=0).repeat(3).shuffle(100).batch(2)
>>> list(ds.as_numpy_iterator())       # 将3个epoch的输入数据放在一起打乱
[array([0, 0]), array([3, 3]), array([6, 6]), array([9, 6]), array([3, 0]), array([9, 9])]
```

### shuffle()

随机打乱数据集中的元素。

```python
shuffle(buffer_size, seed=None, reshuffle_each_iteration=True)
# buffer_size                缓冲区大小.例如数据集包含1000个元素而`buffer_size`设为100,那么前100个元素首先进入缓冲区,
#                            从中随机抽取一个,然后第101个元素进入缓冲区,再随机抽取一个,...若要完全打乱数据集,则
#                            `buffer_size`应不小于数据集的规模
# seed                       随机数种子
# reshuffle_each_iteration   若为`True`,则数据集每迭代完成一次都会重新打乱
```

```python
>>> list(Dataset.range(5).shuffle(5).as_numpy_iterator())    # 随机打乱
[1, 0, 2, 4, 3]
>>> 
>>> list(Dataset.range(5).shuffle(2).as_numpy_iterator())    # 缓冲区设为2
[0, 2, 1, 3, 4]   # 首先从0,1中抽取到0,再从1,2中抽取到2,再从1,3中抽取到1,...
>>> 
>>> ds = Dataset.range(5).shuffle(5)
>>> list(ds.as_numpy_iterator())                 # 每次迭代的顺序不同
[1, 0, 3, 4, 2]
>>> list(ds.as_numpy_iterator())
[2, 0, 1, 4, 3]
>>> list(ds.repeat(3).as_numpy_iterator())
[0, 1, 3, 2, 4, 2, 0, 4, 3, 1, 2, 4, 3, 0, 1]    # 即使调用`repeat()`,每次迭代的顺序也不同
>>> 
>>> ds = Dataset.range(5).shuffle(5, reshuffle_each_iteration=False)
>>> list(ds.as_numpy_iterator())
[1, 4, 2, 3, 0]
>>> list(ds.as_numpy_iterator())                 # 每次迭代的顺序相同
[1, 4, 2, 3, 0]
```

### skip()

去除数据集的前几个元素。

```python
>>> list(Dataset.range(5).skip(2).as_numpy_iterator())
[2, 3, 4]
```

### take()

取数据集的前几个元素。

```python
>>> list(Dataset.range(5).take(2).as_numpy_iterator())
[0, 1]
```

### unbatch()

将数据集的元素（批）拆分为多个元素。

```python
>>> elements = [[1, 2, 3], [1, 2], [1, 2, 3, 4]]
>>> ds = Dataset.from_generator(lambda: elements, tf.int64)
>>> list(ds.as_numpy_iterator())
[array([1, 2, 3]), array([1, 2]), array([1, 2, 3, 4])]
>>> list(ds.unbatch().as_numpy_iterator())
[1, 2, 3, 1, 2, 1, 2, 3, 4]
```

### window()

将数据集中的相邻元素组合为窗口（窗口也是一个小规模的数据集），由这些窗口构成新的数据集。

```python
window(size, shift=None, stride=1, drop_remainder=False)
# size             窗口的元素数量
# shift            窗口的移动距离,默认为`size`
# stride           取样的间距
# drop_remainder   丢弃最后一个规模不足`size`的窗口
```

```python
>>> ds = Dataset.range(10).window(3)
>>> for window in ds:
  print(list(window.as_numpy_iterator()))
... 
[0, 1, 2]
[3, 4, 5]
[6, 7, 8]
[9]
>>> 
>>> ds = Dataset.range(10).window(3, drop_remainder=True)
>>> for window in ds:                                    
  print(list(window.as_numpy_iterator()))
... 
[0, 1, 2]
[3, 4, 5]
[6, 7, 8]
>>> 
>>> ds = Dataset.range(10).window(3, 2, 1, drop_remainder=True)
>>> for window in ds:                                          
  print(list(window.as_numpy_iterator()))
... 
[0, 1, 2]
[2, 3, 4]
[4, 5, 6]
[6, 7, 8]
>>> 
>>> ds = Dataset.range(10).window(3, 1, 2, drop_remainder=True)
>>> for window in ds:                                          
  print(list(window.as_numpy_iterator()))
... 
[0, 2, 4]
[1, 3, 5]
[2, 4, 6]
[3, 5, 7]
[4, 6, 8]
[5, 7, 9]
```

### zip()

组合多个数据集的对应元素。类似于 Python 的内置函数 `zip()`。

```python
>>> ds1 = Dataset.range(1, 4)
>>> ds2 = Dataset.range(4, 7)
>>> ds = Dataset.zip((ds1, ds2))
>>> list(ds.as_numpy_iterator())
[(1, 4), (2, 5), (3, 6)]
>>> 
>>> ds3 = Dataset.range(7, 13).batch(2)
>>> ds = Dataset.zip((ds1, ds2, ds3))    # 数据集的元素类型不要求相同
>>> list(ds.as_numpy_iterator())
[(1, 4, array([7, 8])), (2, 5, array([ 9, 10])), (3, 6, array([11, 12]))]
>>> 
>>> ds4 = Dataset.range(13, 15)
>>> ds = Dataset.zip((ds1, ds2, ds4))    # 数据集的元素数量不要求相同,受限于数量最少的数据集
>>> list(ds.as_numpy_iterator())
[(1, 4, 13), (2, 5, 14)]
```

## TextLineDataset

创建由一个或多个文本文件的各行构成的数据集。`TextLineDataset` 是 `Dataset` 的子类。

```
# file1.txt
the cow
jumped over
the moon
```

```
# file2.txt
jack and jill
went up
the hill
```

```python
>>> ds = tf.data.TextLineDataset(["file1.txt", "file2.txt"])
>>> for element in dataset.as_numpy_iterator():
  print(element)
b'the cow'
b'jumped over'
b'the moon'
b'jack and jill'
b'went up'
b'the hill'
```
