# tf.linalg

## det()

返回一个或多个方阵的行列式。

```python
>>> a = tf.constant([[1., 2], [3, 4]])
>>> a
<tf.Tensor: shape=(2, 2), dtype=float32, numpy=
array([[1., 2.],
       [3., 4.]], dtype=float32)>
>>> tf.linalg.det(a)
<tf.Tensor: shape=(), dtype=float32, numpy=-2.0>
```

## diag()

返回一批对角矩阵，对角值由输入的一批向量给定。

```python
diagonal = np.array([[1, 2, 3, 4],            # Input shape: (2, 4)
                     [5, 6, 7, 8]])
tf.matrix_diag(diagonal) ==> [[[1, 0, 0, 0],  # Output shape: (2, 4, 4)
                               [0, 2, 0, 0],
                               [0, 0, 3, 0],
                               [0, 0, 0, 4]],
                              [[5, 0, 0, 0],
                               [0, 6, 0, 0],
                               [0, 0, 7, 0],
                               [0, 0, 0, 8]]]
```

## eigh()

返回张量的一个特征分解 $A=Q\Lambda Q^{-1}$。

## svd()

返回张量的一个奇异值分解 $A=U\Sigma V^*$。
