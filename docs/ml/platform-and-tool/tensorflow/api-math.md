# tf.math

## abs()

张量逐元素应用绝对值函数。

```python
>>> tf.abs(tf.constant([-1, -2, 3]))
<tf.Tensor: shape=(3,), dtype=int32, numpy=array([1, 2, 3], dtype=int32)>
```

## add(), subtract()

张量加法/减法。`+,-` 符号重载了这些方法。

```python
>>> a = tf.reshape(tf.range(12), [3, 4])
>>> a
<tf.Tensor: shape=(3, 4), dtype=int32, numpy=
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]], dtype=int32)>
>>> a + 1                    # 张量+标量: 扩张的张量加法
<tf.Tensor: shape=(3, 4), dtype=int32, numpy=
array([[ 1,  2,  3,  4],
       [ 5,  6,  7,  8],
       [ 9, 10, 11, 12]], dtype=int32)>
>>> a + tf.constant([1])     # 同前
<tf.Tensor: shape=(3, 4), dtype=int32, numpy=
array([[ 1,  2,  3,  4],
       [ 5,  6,  7,  8],
       [ 9, 10, 11, 12]], dtype=int32)>
>>> a + tf.range(4)          # 张量+子张量: 扩张的张量加法
<tf.Tensor: shape=(3, 4), dtype=int32, numpy=
array([[ 0,  2,  4,  6],
       [ 4,  6,  8, 10],
       [ 8, 10, 12, 14]], dtype=int32)>
>>> a + a                    # 张量+张量: 张量加法
<tf.Tensor: shape=(3, 4), dtype=int32, numpy=
array([[ 0,  2,  4,  6],
       [ 8, 10, 12, 14],
       [16, 18, 20, 22]], dtype=int32)>
```

## argmax(), argmin()

返回张量沿指定维度的最大值的索引。

```python
>>> a = tf.random.normal([4])
>>> a
<tf.Tensor: shape=(4,), dtype=float32, numpy=array([ 0.80442244,  0.01440545, -0.9266029 ,  0.23776768], dtype=float32)>
>>> tf.argmax(a)
<tf.Tensor: shape=(), dtype=int64, numpy=0>
>>> tf.argmin(a)
<tf.Tensor: shape=(), dtype=int64, numpy=2>

>>> a = tf.random.normal([4, 4])
>>> a
<tf.Tensor: shape=(4, 4), dtype=float32, numpy=
array([[ 1.2651453 , -0.9885311 , -1.9029404 ,  1.0343136 ],
       [ 0.4773587 ,  1.2282255 ,  0.66903603, -1.9187453 ],
       [ 0.94859433, -0.50704604,  1.6308597 ,  0.517232  ],
       [ 0.5004154 ,  0.38485277,  0.9955068 , -1.865893  ]],
      dtype=float32)>
>>> tf.argmax(a, 1)
<tf.Tensor: shape=(4,), dtype=int64, numpy=array([0, 1, 2, 2])>
>>> tf.argmin(a, 1)
<tf.Tensor: shape=(4,), dtype=int64, numpy=array([2, 3, 1, 3])>
```

## ceil()

张量逐元素应用向上取整。

```python
>>> tf.math.ceil([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
<tf.Tensor: shape=(7,), dtype=float32, numpy=array([-1., -1., -0.,  1.,  2.,  2.,  2.], dtype=float32)>
```

## equal()

逐元素判断两个张量是否相等。`==` 符号重载了此方法。

```python
>>> one1 = tf.ones([2,3])
>>> one2 = tf.ones([2,3])
>>> one1 == one2
<tf.Tensor: shape=(2, 3), dtype=bool, numpy=
array([[ True,  True,  True],
       [ True,  True,  True]])>
>>> tf.equal(one1, one2)
<tf.Tensor: shape=(2, 3), dtype=bool, numpy=
array([[ True,  True,  True],
       [ True,  True,  True]])>
```

## exp()

张量逐元素应用自然指数函数。

```python
>>> a = tf.reshape(tf.range(10.), [2,5])
>>> a
<tf.Tensor: shape=(2, 5), dtype=float32, numpy=
array([[0., 1., 2., 3., 4.],
       [5., 6., 7., 8., 9.]], dtype=float32)>
>>> tf.exp(a)
<tf.Tensor: shape=(2, 5), dtype=float32, numpy=
array([[1.0000000e+00, 2.7182817e+00, 7.3890562e+00, 2.0085537e+01,
        5.4598152e+01],
       [1.4841316e+02, 4.0342880e+02, 1.0966332e+03, 2.9809580e+03,
        8.1030840e+03]], dtype=float32)>
```

## floor()

张量逐元素应用向下取整。

```python
>>> tf.math.floor([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
<tf.Tensor: shape=(7,), dtype=float32, numpy=array([-2., -2., -1.,  0.,  1.,  1.,  2.], dtype=float32)>
```

## greater(), greater_equal(), less(), less_equal()

逐元素比较两个张量的大小。`>,>=,<,<=` 符号重载了这些方法。

```python
>>> a = tf.constant([5, 4, 6])
>>> b = tf.constant([5, 2, 5])
>>> a > b
<tf.Tensor: shape=(3,), dtype=bool, numpy=array([False,  True,  True])>
>>> tf.greater(a, b)
<tf.Tensor: shape=(3,), dtype=bool, numpy=array([False,  True,  True])>
>>> a >= b
<tf.Tensor: shape=(3,), dtype=bool, numpy=array([ True,  True,  True])>
>>> tf.greater_equal(a, b)
<tf.Tensor: shape=(3,), dtype=bool, numpy=array([ True,  True,  True])>
```

## log()

张量逐元素应用自然对数函数。注意 TensorFlow 没有 `log2()` 和 `log10()` 函数。

```python
>>> a = tf.reshape(tf.range(10.), [2,5])
>>> a
<tf.Tensor: shape=(2, 5), dtype=float32, numpy=
array([[0., 1., 2., 3., 4.],
       [5., 6., 7., 8., 9.]], dtype=float32)>
>>> tf.math.log(a)
<tf.Tensor: shape=(2, 5), dtype=float32, numpy=
array([[     -inf, 0.       , 0.6931472, 1.0986123, 1.3862944],
       [1.609438 , 1.7917595, 1.9459102, 2.0794415, 2.1972246]],
      dtype=float32)>
```

## maximum(), minimum()

逐元素取两个张量的较大值、较小值。

```python
>>> a = tf.constant([0., 0., 0., 0.])
>>> b = tf.constant([-2., 0., 2., 5.])
>>> tf.math.maximum(a, b)
<tf.Tensor: shape=(4,), dtype=float32, numpy=array([0., 0., 2., 5.], dtype=float32)>
>>> tf.math.minimum(a, b)
<tf.Tensor: shape=(4,), dtype=float32, numpy=array([-2.,  0.,  0.,  0.], dtype=float32)>
```

## multiply(), divide()

张量逐元素乘法/除法。`*,/` 符号重载了此方法。

```python
>>> a = tf.reshape(tf.range(12), [3,4])
>>> a
<tf.Tensor: shape=(3, 4), dtype=int32, numpy=
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]], dtype=int32)>
>>> a * 100                                      # 张量 * 标量: 张量的数乘
<tf.Tensor: shape=(3, 4), dtype=int32, numpy=
array([[   0,  100,  200,  300],
       [ 400,  500,  600,  700],
       [ 800,  900, 1000, 1100]], dtype=int32)>
>>> a * tf.range(4)                              # 张量 * 子张量: 张量的扩张逐元素乘法
<tf.Tensor: shape=(3, 4), dtype=int32, numpy=
array([[ 0,  1,  4,  9],
       [ 0,  5, 12, 21],
       [ 0,  9, 20, 33]], dtype=int32)>
>>> a * a                                        # 张量 * 张量: 张量的逐元素乘法
<tf.Tensor: shape=(3, 4), dtype=int32, numpy=
array([[  0,   1,   4,   9],
       [ 16,  25,  36,  49],
       [ 64,  81, 100, 121]], dtype=int32)>

>>> a = tf.reshape(tf.range(1,4),[3,1])
>>> b = tf.range(1,5)
>>> a * b                                        # 一维张量 * 一维张量: 向量外积
<tf.Tensor: shape=(3, 4), dtype=int32, numpy=
array([[ 1,  2,  3,  4],
       [ 2,  4,  6,  8],
       [ 3,  6,  9, 12]], dtype=int32)>
>>> a * b
<tf.Tensor: shape=(3, 4), dtype=int32, numpy=
array([[ 1,  2,  3,  4],
       [ 2,  4,  6,  8],
       [ 3,  6,  9, 12]], dtype=int32)>
```

## pow()

张量逐元素幂乘。`**` 符号重载了此方法。

```python
>>> a = tf.constant([[2, 2], [3, 3]])
>>> a ** 2
<tf.Tensor: shape=(2, 2), dtype=int32, numpy=
array([[4, 4],
       [9, 9]], dtype=int32)>
>>> a ** tf.range(2)
<tf.Tensor: shape=(2, 2), dtype=int32, numpy=
array([[1, 2],
       [1, 3]], dtype=int32)>
>>> b = tf.constant([[8, 16], [2, 3]])
>>> a ** b
<tf.Tensor: shape=(2, 2), dtype=int32, numpy=
array([[  256, 65536],
       [    9,    27]], dtype=int32)>
```

## reduce_max(), reduce_min(), reduce_mean(), reduce_std()

计算张量沿指定维度的最大值、最小值、平均值和标准差。

```python
>>> a = tf.reshape(tf.range(10.), [2,5])
>>> a
<tf.Tensor: shape=(2, 5), dtype=float32, numpy=
array([[0., 1., 2., 3., 4.],
       [5., 6., 7., 8., 9.]], dtype=float32)>

>>> tf.reduce_max(a)
<tf.Tensor: shape=(), dtype=float32, numpy=9.0>
>>> tf.reduce_max(a, 0)
<tf.Tensor: shape=(5,), dtype=float32, numpy=array([5., 6., 7., 8., 9.], dtype=float32)>
>>> tf.reduce_max(a, 1)
<tf.Tensor: shape=(2,), dtype=float32, numpy=array([4., 9.], dtype=float32)>
  
>>> tf.reduce_min(a)
<tf.Tensor: shape=(), dtype=float32, numpy=0.0>
>>> tf.reduce_min(a, 0)
<tf.Tensor: shape=(5,), dtype=float32, numpy=array([0., 1., 2., 3., 4.], dtype=float32)>
>>> tf.reduce_min(a, 1)
<tf.Tensor: shape=(2,), dtype=float32, numpy=array([0., 5.], dtype=float32)>
  
>>> tf.reduce_mean(a)
<tf.Tensor: shape=(), dtype=float32, numpy=4.5>
>>> tf.reduce_mean(a, 0)
<tf.Tensor: shape=(5,), dtype=float32, numpy=array([2.5, 3.5, 4.5, 5.5, 6.5], dtype=float32)>
>>> tf.reduce_mean(a, 1)
<tf.Tensor: shape=(2,), dtype=float32, numpy=array([2., 7.], dtype=float32)>

>>> tf.math.reduce_std(a)
<tf.Tensor: shape=(), dtype=float32, numpy=2.8722813>  
>>> tf.math.reduce_std(a, 0)
<tf.Tensor: shape=(5,), dtype=float32, numpy=array([2.5, 2.5, 2.5, 2.5, 2.5], dtype=float32)>
>>> tf.math.reduce_std(a, 1)
<tf.Tensor: shape=(2,), dtype=float32, numpy=array([1.4142135, 1.4142135], dtype=float32)>
```

## reduce_sum()

计算张量沿指定维度的元素和。

```python
>>> a = tf.constant([[1, 1, 1], [1, 1, 1]])
>>> tf.reduce_sum(a)                    # 求和
<tf.Tensor: shape=(), dtype=int32, numpy=6>
>>> tf.reduce_sum(a, 0)                 # 沿轴0
<tf.Tensor: shape=(3,), dtype=int32, numpy=array([2, 2, 2], dtype=int32)>
>>> tf.reduce_sum(a, 0, keepdims=True)  # 沿轴0并保持张量形状
<tf.Tensor: shape=(1, 3), dtype=int32, numpy=array([[2, 2, 2]], dtype=int32)>
>>> tf.reduce_sum(a, 1)                 # 沿轴1
<tf.Tensor: shape=(2,), dtype=int32, numpy=array([3, 3], dtype=int32)>
>>> tf.reduce_sum(a, 1, keepdims=True)  # 沿轴1并保持张量形状
<tf.Tensor: shape=(2, 1), dtype=int32, numpy=
array([[3],
       [3]], dtype=int32)>
>>> tf.reduce_sum(a, [0,1])             # 同时沿轴0和轴1,即对所有元素求和
<tf.Tensor: shape=(), dtype=int32, numpy=6>
```

## round()

张量逐元素应用舍入函数，0.5 会向偶数取整。

```python
>>> a = tf.constant([0.9, 2.5, 2.3, 1.5, -4.5])
>>> tf.round(a)
<tf.Tensor: shape=(5,), dtype=float32, numpy=array([ 1.,  2.,  2.,  2., -4.], dtype=float32)>
```

## sigmoid()

Sigmoid 激活函数。

```python
>>> input = tf.random.normal([2])
>>> input
<tf.Tensor: shape=(2,), dtype=float32, numpy=array([1.3574934 , 0.30114314], dtype=float32)>
>>> tf.sigmoid(input)
<tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.795352  , 0.57472193], dtype=float32)>
```

## sign()

张量逐元素应用符号函数。

```python
>>> tf.math.sign([0., 2., -3.])
<tf.Tensor: shape=(3,), dtype=float32, numpy=array([ 0.,  1., -1.], dtype=float32)>
```

## sin(), cos(), tan(), arcsin(), arccos(), arctan(), sinh(), cosh(), tanh(), arcsinh(), arccosh(), arctanh()

张量逐元素应用三角函数和双曲函数。

```python
>>> a = tf.reshape(tf.range(10.), [2,5])
>>> a
<tf.Tensor: shape=(2, 5), dtype=float32, numpy=
array([[0., 1., 2., 3., 4.],
       [5., 6., 7., 8., 9.]], dtype=float32)>
>>> tf.sin(a)
<tf.Tensor: shape=(2, 5), dtype=float32, numpy=
array([[ 0.        ,  0.84147096,  0.9092974 ,  0.14112   , -0.7568025 ],
       [-0.9589243 , -0.2794155 ,  0.6569866 ,  0.98935825,  0.4121185 ]],
      dtype=float32)>
```

## sqrt()

张量逐元素开平方。

```python
>>> a = tf.constant([4.0, 9.0])
>>> tf.sqrt(a)
<tf.Tensor: shape=(2,), dtype=float32, numpy=array([2., 3.], dtype=float32)>
```

## square()

张量逐元素平方。相当于 `**2`。

```python
>>> a = tf.constant([4.0, 9.0])
>>> tf.square(a)
<tf.Tensor: shape=(2,), dtype=float32, numpy=array([16., 81.], dtype=float32)>
```
