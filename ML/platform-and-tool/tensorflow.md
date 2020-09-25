# tensor

## create

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
zeros = tf.zeros([2,2])

# sequence
sqc = tf.range(1,5) # [1 2 3 4]


```



## random

```python
# normal distribution
normal = tf.random.normal([2,2])
# tf.random.normal(shape, mean=0.0, stddev=1.0, dtype=tf.dtypes.float32, seed=None)

# uniform distribution
uni_float = tf.random.uniform([2,2],0,10)
uni_int = tf.random.uniform([2,2],0,10,tf.dtypes.int32)
# tf.random.uniform(shape, minval=0, maxval=None, dtype=tf.dtypes.float32, seed=None)
```



## convert from & to NumPy array

```python
# np 2 tf
tf_tensor = tf.constant(np_array)

# tf 2 np
np_array = np.array(tf_tensor)
# or
np_array = tf_tensor.numpy()
```



## shape

```python
rank_4_tensor = tf.zeros([3, 2, 4, 5])
Math  				
未
定
义
的
状
态
转
移
函
数
转
移
至
开
始
运
行
结
束
错
误
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
reshape = tf.reshape(reshape,[7,-1])        # error
```



## operation

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

```python
x = tf.reshape(tf.range(1,4),[3,1])
y = tf.range(1,5)
print(tf.multiply(x,y))  # return 3*4 matrix
print(tf.multiply(y,x))  # return 3*4 matrix

print(tf.broadcast_to(x, [3, 3]))  # extend tensor
```



## index & slice

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



## type casting

```python
the_f64_tensor = tf.constant([2.2, 3.3, 4.4], dtype=tf.float64)
the_f16_tensor = tf.cast(the_f64_tensor, dtype=tf.float16)
the_u8_tensor = tf.cast(the_f16_tensor, dtype=tf.uint8)
print(the_u8_tensor)
# tf.Tensor([2 3 4], shape=(3,), dtype=uint8)
```



## ragged tensor

```python
ragged_list = [
    [0, 1, 2, 3],
    [4, 5],
    [6, 7, 8],
    [9]]
ragged_tensor = tf.ragged.constant(ragged_list)
print(ragged_tensor)
# <tf.RaggedTensor [[0, 1, 2, 3], [4, 5], [6, 7, 8], [9]]>
print(ragged_tensor.shape)
# (4, None)
```



## string tensor

```python
scalar_of_string = tf.constant("Gray wolf")
print(scalar_of_string)
# tf.Tensor(b'Gray wolf', shape=(), dtype=string)
tensor_of_strings = tf.constant(["Gray wolf",
                                 "Quick brown fox",
                                 "Lazy dog"])
print(tensor_of_strings)
# tf.Tensor([b'Gray wolf' b'Quick brown fox' b'Lazy dog'], shape=(3,), dtype=string)

unicode_string = tf.constant("🥳👍") # unicode string
print(unicode_string)
# <tf.Tensor: shape=(), dtype=string, numpy=b'\xf0\x9f\xa5\xb3\xf0\x9f\x91\x8d'>

print(tf.strings.split(scalar_of_string, sep=" "))  # split
# tf.Tensor([b'Gray' b'wolf'], shape=(2,), dtype=string)
print(tf.strings.split(tensor_of_strings))          # split vector of strings
# <tf.RaggedTensor [[b'Gray', b'wolf'], [b'Quick', b'brown', b'fox'], [b'Lazy', b'dog']]>

# ascii char
print(tf.strings.bytes_split(scalar_of_string))     # split to bytes
# tf.Tensor([b'G' b'r' b'a' b'y' b' ' b'w' b'o' b'l' b'f'], shape=(9,), dtype=string)
print(tf.io.decode_raw(scalar_of_string, tf.uint8)) # cast to ascii
# tf.Tensor([ 71 114  97 121  32 119 111 108 102], shape=(9,), dtype=uint8)

# unicode char
print(tf.strings.unicode_split(unicode_string, "UTF-8"))
print(tf.strings.unicode_decode(unicode_string, "UTF-8"))
# tf.Tensor([b'\xf0\x9f\xa5\xb3' b'\xf0\x9f\x91\x8d'], shape=(2,), dtype=string)
# tf.Tensor([129395 128077], shape=(2,), dtype=int32)
```



## sparse tensor

```python
sparse_tensor = tf.sparse.SparseTensor(indices=[[0, 0], [1, 2]],
                                       values=[1, 2],
                                       dense_shape=[3, 4])
print(tf.sparse.to_dense(sparse_tensor))
# tf.Tensor(
# [[1 0 0 0]
#  [0 0 2 0]
#  [0 0 0 0]], shape=(3, 4), dtype=int32)
```





# Variable

```python

```



## GradientTape

`tf.GradientTape()` 是一个自动求导的记录器。以下示例计算$$y=x^2$$在$$x=3$$位置的导数：

```python
import tensorflow as tf

x = tf.Variable(initial_value=3.)   # 初值为3.0的变量
with tf.GradientTape() as tape:     # 在 tf.GradientTape() 的上下文内，所有计算步骤都会被记录以用于求导
    y = tf.square(x)
y_grad = tape.gradient(y, x)        # 计算y关于x的导数
print(y, y_grad)                    # tf.Tensor(6.0, shape=(), dtype=float32)
```

以下示例计算$$\mathcal{L}=||X\pmb w+b-\pmb y||^2$$在$$\pmb w=[1,2]^{\rm T},b=1$$位置的对$$\pmb w,b$$的导数：

```python
X = tf.constant([[1., 2.], [3., 4.]])
y = tf.constant([[1.], [2.]])
w = tf.Variable(initial_value=[[1.], [2.]])
b = tf.Variable(initial_value=1.)
with tf.GradientTape() as tape:
    L = tf.reduce_sum(tf.square(tf.matmul(X, w) + b - y))
    # tf.square 将输入张量的每个元素平方
    # tf.reduce_sum 对输入张量的所有元素求和,输出一个标量
w_grad, b_grad = tape.gradient(L, [w, b])        # 计算L(w, b)关于w, b的偏导数

print(L, w_grad, b_grad)
# tf.Tensor(125.0, shape=(), dtype=float32) tf.Tensor(
# [[ 70.]
#  [100.]], shape=(2, 1), dtype=float32) tf.Tensor(30.0, shape=(), dtype=float32)
```

可以看到计算结果
$$
\mathcal{L}=125,\ \frac{\partial \mathcal{L}}{\partial \pmb w}=\begin{bmatrix}70\\100\end{bmatrix},\ \frac{\partial \mathcal{L}}{\partial b}=30
$$






# keras

在 TensorFlow 中，推荐使用 Keras（ `tf.keras` ）构建模型。Keras 是一个广为流行的高级神经网络 API，简单、快速而不失灵活性，现已得到 TensorFlow 的官方内置和全面支持。

keras 有两个重要的概念： **模型（model）** 和 **层（layer）** 。层将各种计算流程和变量进行了封装（例如基本的全连接层，CNN 的卷积层、池化层等），而模型则将各种层进行组织和连接，并封装成一个整体，描述了如何将输入数据通过各种层以及运算而得到输出。



## layer

`tf.keras.layers` 下内置了深度学习中大量常用的的预定义层，同时也允许我们自定义层。

### Dense

全连接层（fully-connected layer，`tf.keras.layers.Dense` ）是 Keras 中最基础和常用的层之一，对输入矩阵 $$A$$ 进行 $$f(A\pmb w+b)$$ 的线性变换 + 激活函数操作。如果不指定激活函数，即是纯粹的线性变换 $$A\pmb w+b$$。具体而言，给定输入张量 `input = [batch_size, input_dim]` ，该层对输入张量首先进行 `tf.matmul(input, kernel) + bias` 的线性变换（ `kernel` 和 `bias` 是层中可训练的变量），然后对线性变换后张量的每个元素通过激活函数 `activation` ，从而输出形状为 `[batch_size, units]` 的二维张量。

[![../../_images/dense.png](https://tf.wiki/_images/dense.png)](https://tf.wiki/_images/dense.png)

其包含的主要参数如下：

+ `units` ：神经元的个数，也是输出张量的维度
+ `activation` ：激活函数，默认为无激活函数。常用的激活函数包括 `tf.nn.relu` 、 `tf.nn.tanh` 和 `tf.nn.sigmoid` 
+ `use_bias` ：是否加入偏置向量 `bias` ，默认为 `True` 
+ `kernel_initializer` 、 `bias_initializer` ：权重矩阵 `kernel` 和偏置向量 `bias` 两个变量的初始化器。默认为 `tf.glorot_uniform_initializer`。设置为 `tf.zeros_initializer` 表示将两个变量均初始化为全 0

该层包含权重矩阵 `kernel = [input_dim, units]` 和偏置向量 `bias = [units]`两个可训练变量，对应于 $$f(A\pmb w+b)$$ 中的 $$\pmb w$$ 和 $$b$$。



### Conv2D

卷积层。

其包含的主要参数如下：

+ `filters`：输出特征映射的个数
+ `kernel_size`：整数或整数1×2向量，（分别）表示二维卷积核的高和宽
+ `strides`：整数或整数1×2向量，（分别）表示卷积的纵向和横向步长
+ `padding`：`"valid"`表示对于不够卷积核大小的部分丢弃，`"same"`表示对于不够卷积核大小的部分补0，默认为`"valid"`
+ `activation`：激活函数，默认为无激活函数
+ `use_bias`：是否使用偏置，默认为使用

示例：

```python
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
# 输入32x32RGB图片,输出32个特征映射,使用3x3卷积核,每个输出特征映射使用1个偏置
# 参数数量为3x32x(3x3)+32=896
model.add(keras.layers.MaxPooling2D((2, 2)))
# 对每个2x2区块执行最大汇聚
model.add(keras.layers.Conv2D(64, (3, 3), (2, 2), activation='relu'))
# 卷积的步长设为2
model.add(keras.layers.MaxPooling2D((2, 2)))
# 7%2=1,因此丢弃一行一列的数据
model.summary()

# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# conv2d (Conv2D)              (None, 30, 30, 32)        896       
# _________________________________________________________________
# max_pooling2d (MaxPooling2D) (None, 15, 15, 32)        0         
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 7, 7, 64)          18496     
# _________________________________________________________________
# max_pooling2d_1 (MaxPooling2 (None, 3, 3, 64)          0         
# =================================================================
# Total params: 19,392
# Trainable params: 19,392
# Non-trainable params: 0
```



### MaxPooling2D

汇聚层（池化层）。

其包含的主要参数如下：

+ `pool_size`：最大汇聚的区域规模，默认为`(2,2)`
+ `strides`：最大汇聚的步长，默认为`None`
+ `padding`：`"valid"`表示对于不够区域大小的部分丢弃，`"same"`表示对于不够区域大小的部分补0，默认为`"valid"`



### Embedding

> 参考[单词嵌入向量](https://www.tensorflow.org/tutorials/text/word_embeddings)

嵌入层。

其包含的主要参数如下：

+ `input_dim`：字典的规模
+ `output_dim`：嵌入向量的规模
+ `mask_zero`：是否将输入中的0看作填充值而忽略之，默认为`False`
+ `input_length`：输入序列的长度（如果该长度固定），默认为`None`；如果此嵌入层后接`Flatten`层，再接`Dense`层，则必须制定此参数

示例见LSTM。



### LSTM

LSTM层。

其包含的主要参数如下：

+ `units`：输出空间的规模



示例：

```python
model = keras.Sequential()
model.add(keras.layers.Embedding(10000, 16))
# 将规模为10000的词典嵌入到16维向量
# 输入长度为256的向量,输出规模为256x16的张量
model.add(tf.keras.layers.LSTM(64))
# LSTM层
# 输入规模为256x16的张量,输出长度为64的向量,相当于(同步或异步)序列到序列模式每4个输入向量输出一个标量值
model.add(keras.layers.Dense(16, activation='relu'))
# 全连接层,ReLU激活函数,分类器
model.add(keras.layers.Dense(1, activation='sigmoid'))  
# 全连接层,Logistic激活函数
model.summary()
```





### Bidirectional







## model

### Sequential

`Sequential`返回一个`keras.Model`对象。`Sequential`模型适用于FNN，CNN，RNN等，其中每一层都有**一个输入张量和一个输出张量** 。

以下`Sequential`模型，

```python
model = keras.Sequential(
    [
        layers.Dense(2, activation="relu", name="layer1"),
        layers.Dense(3, activation="relu", name="layer2"),
        layers.Dense(4, name="layer3"),
    ]
)
# Call model on a test input
x = tf.ones((3, 3))
y = model(x)
```

等效于以下功能

```python
# Create 3 layers
layer1 = layers.Dense(2, activation="relu", name="layer1")
layer2 = layers.Dense(3, activation="relu", name="layer2")
layer3 = layers.Dense(4, name="layer3")

# Call layers on a test input
x = tf.ones((3, 3))
y = layer3(layer2(layer1(x)))
```

`Sequential`模型也可以用`add()`方法创建

```python
model = keras.Sequential()
model.add(layers.Dense(2, activation="relu"))
model.add(layers.Dense(3, activation="relu"))
model.add(layers.Dense(4))
```

一旦创建了模型，就可以调用`summary()`方法显示其内容。



CNN模型示例

```python
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dense(10))
```



### compile





### fit





### evaluate





### 自定义模型

Keras 模型以类的形式呈现，我们可以通过继承 `tf.keras.Model` 这个 Python 类来定义自己的模型。在继承类中，我们需要重写 `__init__()` （构造函数，初始化）和 `call(input)` （模型调用）两个方法，同时也可以根据需要增加自定义的方法。

```python
class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()     # Python 2 下使用 super(MyModel, self).__init__()
        # 此处添加初始化代码（包含 call 方法中会用到的层），例如
        # layer1 = tf.keras.layers.BuiltInLayer(...)
        # layer2 = MyCustomLayer(...)

    def call(self, input):
        # 此处添加模型调用的代码（处理输入并返回输出），例如
        # x = layer1(input)
        # output = layer2(x)
        return output

    # 还可以添加自定义的方法
```

以下为自定义线性模型示例：

```python
class Linear(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(
            units=1,
            activation=None,
            kernel_initializer=tf.zeros_initializer(),
            bias_initializer=tf.zeros_initializer()
        )

    def call(self, input):
        output = self.dense(input)
        return output
```







# visualize

## example: gray image

```python
# draw image
plt.figure()
plt.imshow(train_images[0], cmap=plt.cm.binary)
plt.colorbar()
plt.grid(False)
plt.show()

# draw multiple images
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
```



## example: line chart

```python
def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [MPG]')
  plt.plot(hist['epoch'], hist['mae'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mae'],
           label = 'Val Error')
  plt.ylim([0,5])
  plt.legend()
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mse'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mse'],
           label = 'Val Error')
  plt.ylim([0,20])
  plt.legend()
  plt.show()
```

