# tf.sparse

## add

## concat

## mask

## SparseTensor

稀疏张量类型。

```python
>>> sparse_tensor = tf.sparse.SparseTensor(indices=[[0, 0], [1, 2]],
...                                        values=[1, 2],
...                                        dense_shape=[3, 4])
>>> tf.sparse.to_dense(sparse_tensor)
<tf.Tensor: shape=(3, 4), dtype=int32, numpy=
array([[1, 0, 0, 0],
       [0, 0, 2, 0],
       [0, 0, 0, 0]], dtype=int32)>
```
