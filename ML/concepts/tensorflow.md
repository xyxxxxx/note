TensorFlow is an end-to-end open source platform for machine learning. TensorFlow is a rich system for managing all aspects of a machine learning system; however, this class focuses on using a particular TensorFlow API to develop and train machine learning models.

TensorFlow APIs are arranged hierarchically, with the high-level APIs built on the low-level APIs. Machine learning researchers use the low-level APIs to create and explore new machine learning algorithms. In this class, you will use a high-level API named tf.keras to define and train machine learning models and to make predictions.

The following figure shows the hierarchy of TensorFlow toolkits:

![Simplified hierarchy of TensorFlow toolkits. tf.keras API is at    the top.](https://developers.google.com/machine-learning/crash-course/images/TFHierarchyNew.svg)



## Tensor

Tensors are multi-dimensional arrays with a uniform type `dtype`, it is kind of like `arrays` of NumPy.

All tensors are immutable like Python numbers and strings: you can never update the contents of a tensor, only create a new one.



### Basics

Here is a scalar or rank-0 tensor. A scalar contains a single value, and no axes.

```python
# This will be an int32 tensor by default; see "dtypes" below.
rank_0_tensor = tf.constant(4)
print(rank_0_tensor)

# tf.Tensor(4, shape=(), dtype=int32)
```

