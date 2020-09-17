## tensor

create

```python
# scalar
rank_0_tensor = tf.constant(4)
print(rank_0_tensor)
# tf.Tensor(4, shape=(), dtype=int32)

# vector
rank_1_tensor = tf.constant([2.0, 3.0, 4.0])
print(rank_1_tensor)
# tf.Tensor([2. 3. 4.], shape=(3,), dtype=float32)

# matrix
rank_2_tensor = tf.constant([[1, 2],
                             [3, 4],
                             [5, 6]], dtype=tf.float16)
print(rank_2_tensor)
# tf.Tensor(
# [[1. 2.]
#  [3. 4.]
#  [5. 6.]], shape=(3, 2), dtype=float16)

# rank 3 tensor
rank_3_tensor = tf.constant([
  [[0, 1, 2, 3, 4],
   [5, 6, 7, 8, 9]],
  [[10, 11, 12, 13, 14],
   [15, 16, 17, 18, 19]],
  [[20, 21, 22, 23, 24],
   [25, 26, 27, 28, 29]],])
print(rank_3_tensor)
# tf.Tensor(
# [[[ 0  1  2  3  4]
#   [ 5  6  7  8  9]]

#  [[10 11 12 13 14]
#   [15 16 17 18 19]]

#  [[20 21 22 23 24]
#   [25 26 27 28 29]]], shape=(3, 2, 5), dtype=int32)

# all ones tensor
ones = tf.ones([2,2])

# all zeros tensor
ones = tf.zeros([2,2])
```

convert to NumPy array

```python
np.array(rank_2_tensor)
# or
rank_2_tensor.numpy()
```

shape

```python
rank_4_tensor = tf.zeros([3, 2, 4, 5])

# show shape
print("Type of every element:", rank_4_tensor.dtype)   # <dtype: 'float32'>
print("Number of dimensions:", rank_4_tensor.ndim)     # 4
print("Shape of tensor:", rank_4_tensor.shape)         # (3,2,4,5)
print("Length of first axis:", rank_4_tensor.shape[0]) # 3
print("Length of last axis:", rank_4_tensor.shape[-1]) # 5
print("Total number of elements (3*2*4*5): ", tf.size(rank_4_tensor).numpy()) # 120

# manipulate shape
reshape = tf.reshape(rank_4_tensor,[4,5,6])
print(reshape)
# tf.Tensor(
# [[[0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0.]]

#  ...

#  [[0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0.]]], shape=(4, 5, 6), dtype=float32)

reshape = tf.reshape(rank_4_tensor,[10,-1]) # -1 fits length of last axis
print(reshape)
# tf.Tensor(
# [[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  ...
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]], shape=(10, 12), dtype=float32)

reshape = tf.reshape(rank_4_tensor,[4,5,6])
reshape = tf.reshape(reshape,[6,5,4])
```

operation

```python
a = tf.constant([[1, 2],
                 [3, 4]])
b = tf.constant([[1, 1],
                 [1, 1]])
print(a + b, "\n") # element-wise addition
# print(tf.add(a, b), "\n")
print(a * b, "\n") # element-wise multiplication
# print(tf.multiply(a, b), "\n")
print(a @ b, "\n") # matrix multiplication
# print(tf.matmul(a, b), "\n")
# tf.Tensor(
# [[2 3]
#  [4 5]], shape=(2, 2), dtype=int32) 

# tf.Tensor(
# [[1 2]
#  [3 4]], shape=(2, 2), dtype=int32) 

# tf.Tensor(
# [[3 3]
#  [7 7]], shape=(2, 2), dtype=int32) 

print(tf.reduce_max(a)) # max element
print(tf.argmax(a))     # index of max element
```

index & slice

```python
# vector
rank_1_tensor = tf.constant([0, 1, 1, 2, 3, 5, 8, 13, 21, 34])
print("First:", rank_1_tensor[0].numpy())
print("Second:", rank_1_tensor[1].numpy())
print("Last:", rank_1_tensor[-1].numpy())
print("Everything:", rank_1_tensor[:].numpy())
print("Before 4:", rank_1_tensor[:4].numpy())
print("From 4 to the end:", rank_1_tensor[4:].numpy())
print("From 2, before 7:", rank_1_tensor[2:7].numpy())
print("Every other item:", rank_1_tensor[::2].numpy())
print("Reversed:", rank_1_tensor[::-1].numpy())

# matrix
rank_2_tensor = tf.constant([[1, 2],
                             [3, 4],
                             [5, 6]], dtype=tf.float16)
print(rank_2_tensor[1, 1].numpy())
print("Second row:", rank_2_tensor[1, :].numpy())
print("Second column:", rank_2_tensor[:, 1].numpy())
print("Last row:", rank_2_tensor[-1, :].numpy())
print("First item in last column:", rank_2_tensor[0, -1].numpy())
print("Skip the first row:")
print(rank_2_tensor[1:, :].numpy(), "\n")

rank_3_tensor = tf.constant([
  [[0, 1, 2, 3, 4],
   [5, 6, 7, 8, 9]],
  [[10, 11, 12, 13, 14],
   [15, 16, 17, 18, 19]],
  [[20, 21, 22, 23, 24],
   [25, 26, 27, 28, 29]],])
print(rank_3_tensor[:, :, 4])
# tf.Tensor(
# [[ 4  9]
#  [14 19]
#  [24 29]], shape=(3, 2), dtype=int32)
```







## visualize