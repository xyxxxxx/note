# tf.random

## normal()

生成指定形状的随机张量，其中每个元素服从正态分布。

```python
>>> tf.random.set_seed(5)
>>> tf.random.normal([2, 3])         # 标准正态分布
<tf.Tensor: shape=(2, 3), dtype=float32, numpy=
array([[-0.18030666, -0.95028627, -0.03964049],
       [-0.7425406 ,  1.3231523 , -0.61854804]], dtype=float32)>

>>> tf.random.set_seed(5)
>>> tf.random.normal([2, 3], 80, 10)
<tf.Tensor: shape=(2, 3), dtype=float32, numpy=
array([[78.19693, 70.49714, 79.60359],
       [72.57459, 93.23152, 73.81452]], dtype=float32)>
```

## set_seed()

设置全局随机种子。

涉及随机数的操作从全局种子和操作种子推导其自身的种子。全局种子和操作种子的关系如下：

1. 若都没有设定，则操作随机选取一个种子。
2. 若只设定了全局种子，则接下来的若干操作选取的种子都是确定的。注意不同版本的 TensorFlow 可能会得到不同的结果。
3. 若只设定了操作种子，则使用默认的全局种子。
4. 若都设定，则两个种子共同确定操作的种子。

```python
# 都没有设定: 每次调用的结果都是随机的
>>> tf.random.uniform([1])
<tf.Tensor: shape=(1,), dtype=float32, numpy=array([0.3124857], dtype=float32)>
>>> tf.random.uniform([1])
<tf.Tensor: shape=(1,), dtype=float32, numpy=array([0.72942686], dtype=float32)>

# now close the program and run it again

>>> tf.random.uniform([1])
<tf.Tensor: shape=(1,), dtype=float32, numpy=array([0.21320045], dtype=float32)>
>>> tf.random.uniform([1])
<tf.Tensor: shape=(1,), dtype=float32, numpy=array([0.06874764], dtype=float32)>
```

```python
# 设定全局种子: 每次调用set_seed()之后的调用结果都是确定的
>>> tf.random.set_seed(1)
>>> tf.random.uniform([1])
<tf.Tensor: shape=(1,), dtype=float32, numpy=array([0.16513085], dtype=float32)>
>>> tf.random.uniform([1])
<tf.Tensor: shape=(1,), dtype=float32, numpy=array([0.51010704], dtype=float32)>

>>> tf.random.set_seed(1)
>>> tf.random.uniform([1])
<tf.Tensor: shape=(1,), dtype=float32, numpy=array([0.16513085], dtype=float32)>
>>> tf.random.uniform([1])
<tf.Tensor: shape=(1,), dtype=float32, numpy=array([0.51010704], dtype=float32)>
```

```python
# 设定操作种子: 每次启动程序之后的调用结果都是确定的
>>> tf.random.uniform([1], seed=1)
<tf.Tensor: shape=(1,), dtype=float32, numpy=array([0.2390374], dtype=float32)>
>>> tf.random.uniform([1], seed=1)
<tf.Tensor: shape=(1,), dtype=float32, numpy=array([0.22267115], dtype=float32)>

# now close the program and run it again

>>> tf.random.uniform([1], seed=1)
<tf.Tensor: shape=(1,), dtype=float32, numpy=array([0.2390374], dtype=float32)>
>>> tf.random.uniform([1], seed=1)
<tf.Tensor: shape=(1,), dtype=float32, numpy=array([0.22267115], dtype=float32)>
```

```python
# 设定全局种子和操作种子: 完全确定调用结果
>>> tf.random.set_seed(1)
>>> tf.random.uniform([1], seed=1)
<tf.Tensor: shape=(1,), dtype=float32, numpy=array([0.05554414], dtype=float32)>
  
>>> tf.random.set_seed(1)
>>> tf.random.uniform([1], seed=3)
<tf.Tensor: shape=(1,), dtype=float32, numpy=array([0.7787856], dtype=float32)>
>>> tf.random.uniform([1], seed=2)
<tf.Tensor: shape=(1,), dtype=float32, numpy=array([0.40639675], dtype=float32)>
>>> tf.random.uniform([1], seed=1)
<tf.Tensor: shape=(1,), dtype=float32, numpy=array([0.05554414], dtype=float32)>
```

## poisson()

生成指定形状的随机张量，其中每个元素服从泊松分布。

```python
>>> tf.random.poisson([10], [1, 2])              # 第0,1列分别服从λ=1,2的泊松分布
<tf.Tensor: shape=(10, 2), dtype=float32, numpy=
array([[1., 0.],
       [0., 2.],
       [1., 4.],
       [0., 1.],
       [0., 1.],
       [0., 1.],
       [0., 4.],
       [1., 0.],
       [1., 2.],
       [1., 2.]], dtype=float32)>
```

## uniform()

生成指定形状的随机张量，其中每个元素服从均匀分布。

```python
>>> tf.random.set_seed(5)
>>> tf.random.uniform([2,3])
<tf.Tensor: shape=(2, 3), dtype=float32, numpy=
array([[0.6263931 , 0.5298432 , 0.7584572 ],
       [0.5084884 , 0.34415376, 0.31959772]], dtype=float32)>

>>> tf.random.set_seed(5)
>>> tf.random.uniform([2,3], 0, 10)
<tf.Tensor: shape=(2, 3), dtype=float32, numpy=
array([[6.263931 , 5.2984324, 7.584572 ],
       [5.084884 , 3.4415376, 3.1959772]], dtype=float32)>
```
