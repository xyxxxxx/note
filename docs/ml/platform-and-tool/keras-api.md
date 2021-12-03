

在 TensorFlow 中，推荐使用 Keras（`tf.keras`）构建模型。Keras 是一个广为流行的高级神经网络 API，简单、快速而不失灵活性，现已得到 TensorFlow 的官方内置和全面支持。

Keras 提供了定义和训练任何类型的神经网络模型的便捷方法，具有以下特性：

* 允许代码在 CPU 或 GPU 上运行并且无缝切换
* 提供用户友好的 API 以使得能够快速建模
* 提供对于 CNN（for CV），RNN（for time series）的内置支持
* 支持任意类型的网络结构

keras 有两个重要的概念：**模型（model）**和**层（layer）**。层将各种计算流程和变量进行了封装（例如基本的全连接层，CNN 的卷积层、池化层等），而模型则将各种层进行组织和连接，并封装成一个整体，描述了如何将输入数据通过各种层以及运算而得到输出。

# activations

## elu

指数线性单元。
$$
{\rm elu}(x)=\begin{cases}x,& x\ge0\\\alpha(e^x-1),&x<0 \end{cases}
$$

```python
tf.keras.activations.elu(x, alpha=1.0)
```

## exponential

## linear

## relu

## softmax

## tanh

# callbacks

## Callback

用于创建新回调的抽象类。若要创建回调，继承此类并任意重载以下方法。

```python
class CustomCallback(keras.callbacks.Callback):
    def on_batch_begin(self, batch, logs=None):
        """`on_train_batch_begin`方法的别名,不建议使用"""
        pass
      
    def on_batch_end(self, batch, logs=None):
        """`on_train_batch_end`方法的别名,不建议使用"""
        pass
  
    def on_epoch_begin(self, epoch, logs=None):
        """在epoch开始时调用"""
        pass

    def on_epoch_end(self, epoch, logs=None):
        """在epoch结束时调用
        
        Args:
            logs: 字典,当前训练epoch和验证epoch(如果进行了验证)的指标.验证指标的键具有前缀`val_`.
                  *尽管官方文档(https://keras.io/guides/writing_your_own_callbacks/#a-basic-example)中
                  表述为epoch的平均指标,但实际上使用的是epoch的最后一个step的指标
        """
        pass
  
    def on_train_begin(self, logs=None):
        """在训练开始时调用"""
        pass

    def on_train_end(self, logs=None):
        """在训练结束时调用"""
        pass

    def on_test_begin(self, logs=None):
        """在测试开始时调用"""
        pass

    def on_test_end(self, logs=None):
        """在测试结束时调用"""
        pass

    def on_predict_begin(self, logs=None):
        """在预测开始时调用"""
        pass

    def on_predict_end(self, logs=None):
        """在预测结束时调用"""
        pass

    def on_train_batch_begin(self, batch, logs=None):
        """在训练过程的step开始时调用"""
        pass

    def on_train_batch_end(self, batch, logs=None):
        """在训练过程的step结束时调用"""
        pass

    def on_test_batch_begin(self, batch, logs=None):
        """在验证/测试过程的step开始时调用"""
        pass

    def on_test_batch_end(self, batch, logs=None):
        """在验证/测试过程的step结束时调用"""
        pass

    def on_predict_batch_begin(self, batch, logs=None):
        """在预测过程的step开始时调用"""
        pass

    def on_predict_batch_end(self, batch, logs=None):
        """在预测过程的step结束时调用"""
        pass
```

调用的顺序如下：

```shell
$ python mnist_custom_callback.py
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 26, 26, 32)        320       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 13, 13, 32)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 11, 11, 64)        18496     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 5, 5, 64)          0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 3, 3, 64)          36928     
_________________________________________________________________
flatten (Flatten)            (None, 576)               0         
_________________________________________________________________
dense (Dense)                (None, 64)                36928     
_________________________________________________________________
dense_1 (Dense)              (None, 10)                650       
=================================================================
Total params: 93,322
Trainable params: 93,322
Non-trainable params: 0
_________________________________________________________________
train begin ...                    # 训练开始
Epoch 1/3
epoch begin ...                    # 一个训练+验证epoch
train batch begin ...              # 一个训练batch
1/8 [==>...........................] - ETA: 12s - loss: 2.3017 - accuracy: 0.1182train batch end ...
train batch begin ...
2/8 [======>.......................] - ETA: 7s - loss: 2.2929 - accuracy: 0.1495 train batch end ...
train batch begin ...
3/8 [==========>...................] - ETA: 6s - loss: 2.2843 - accuracy: 0.1794train batch end ...
train batch begin ...
4/8 [==============>...............] - ETA: 4s - loss: 2.2756 - accuracy: 0.2042train batch end ...
train batch begin ...
5/8 [=================>............] - ETA: 3s - loss: 2.2664 - accuracy: 0.2265train batch end ...
train batch begin ...
6/8 [=====================>........] - ETA: 2s - loss: 2.2562 - accuracy: 0.2468train batch end ...
train batch begin ...
7/8 [=========================>....] - ETA: 1s - loss: 2.2452 - accuracy: 0.2644train batch end ...
train batch begin ...
8/8 [==============================] - ETA: 0s - loss: 2.2331 - accuracy: 0.2803train batch end ...
test begin ...                     # 验证开始
test batch begin ...               # 一个验证batch
test batch end ...
test batch begin ...
test batch end ...
test end ...                       # 验证结束
8/8 [==============================] - 12s 1s/step - loss: 2.2236 - accuracy: 0.2927 - val_loss: 1.8295 - val_accuracy: 0.6315
epoch end ...
Epoch 2/3
epoch begin ...
train batch begin ...
1/8 [==>...........................] - ETA: 8s - loss: 1.8353 - accuracy: 0.6113train batch end ...
train batch begin ...
2/8 [======>.......................] - ETA: 7s - loss: 1.8067 - accuracy: 0.6215train batch end ...
train batch begin ...
3/8 [==========>...................] - ETA: 6s - loss: 1.7765 - accuracy: 0.6298train batch end ...
train batch begin ...
4/8 [==============>...............] - ETA: 4s - loss: 1.7455 - accuracy: 0.6375train batch end ...
train batch begin ...
5/8 [=================>............] - ETA: 3s - loss: 1.7140 - accuracy: 0.6449train batch end ...
train batch begin ...
6/8 [=====================>........] - ETA: 2s - loss: 1.6818 - accuracy: 0.6521train batch end ...
train batch begin ...
7/8 [=========================>....] - ETA: 1s - loss: 1.6495 - accuracy: 0.6591train batch end ...
train batch begin ...
8/8 [==============================] - ETA: 0s - loss: 1.6169 - accuracy: 0.6656train batch end ...
test begin ...
test batch begin ...
test batch end ...
test batch begin ...
test batch end ...
test end ...
8/8 [==============================] - 10s 1s/step - loss: 1.5917 - accuracy: 0.6706 - val_loss: 0.7942 - val_accuracy: 0.8156
epoch end ...
Epoch 3/3
epoch begin ...
train batch begin ...
1/8 [==>...........................] - ETA: 8s - loss: 0.8264 - accuracy: 0.7970train batch end ...
train batch begin ...
2/8 [======>.......................] - ETA: 7s - loss: 0.8112 - accuracy: 0.7965train batch end ...
train batch begin ...
3/8 [==========>...................] - ETA: 5s - loss: 0.7943 - accuracy: 0.7967train batch end ...
train batch begin ...
4/8 [==============>...............] - ETA: 4s - loss: 0.7791 - accuracy: 0.7979train batch end ...
train batch begin ...
5/8 [=================>............] - ETA: 3s - loss: 0.7640 - accuracy: 0.7996train batch end ...
train batch begin ...
6/8 [=====================>........] - ETA: 2s - loss: 0.7498 - accuracy: 0.8014train batch end ...
train batch begin ...
7/8 [=========================>....] - ETA: 1s - loss: 0.7368 - accuracy: 0.8030train batch end ...
train batch begin ...
8/8 [==============================] - ETA: 0s - loss: 0.7247 - accuracy: 0.8048train batch end ...
test begin ...
test batch begin ...
test batch end ...
test batch begin ...
test batch end ...
test end ...
8/8 [==============================] - 10s 1s/step - loss: 0.7152 - accuracy: 0.8062 - val_loss: 0.4754 - val_accuracy: 0.8480
epoch end ...
train end ...                      # 训练结束
test begin ...                     # 测试开始
test batch begin ...               # 一个测试batch
test batch end ...
test batch begin ...
test batch end ...
test batch begin ...
test batch end ...
test batch begin ...
test batch end ...
test batch begin ...
test batch end ...
5/5 - 1s - loss: 0.4729 - accuracy: 0.8462
test end ...                       # 测试结束
```

## EarlyStopping

当监视的参数不再改善时提前停止训练。

```python
tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=0, verbose=0,
    mode='auto', baseline=None, restore_best_weights=False
)
# monitor               监视的指标,可以是'val_loss','val_accuracy',等等
# min_delta             可以视为改善的最小绝对变化量,换言之,小于该值的指标绝对变化量视为没有改善
# patience              若最近`patience`次epoch的指标都没有改善(即最后`patience`次的指标都比倒数第
#                       `patience+1`次差),则停止训练
# mode                  若为`'min'`,则指标减小视为改善;若为`'max'`,则指标增加视为改善;若为`'auto'`,
#                       则方向根据指标的名称自动推断
# baseline              监视的指标的基线值,若指标没有超过基线值则停止训练
# restore_best_weights  若为`True`,训练结束时会恢复监视指标取最好值的epoch的权重;若为`False`,训练结束时
#                       会保留最后一个epoch的权重
```

## LambdaCallback

创建简单的自定义回调。

```python
tf.keras.callbacks.LambdaCallback(
    on_epoch_begin=None, on_epoch_end=None, on_batch_begin=None, on_batch_end=None,
    on_train_begin=None, on_train_end=None, **kwargs
)
# on_epoch_begin    在每个epoch开始时调用的函数
# ...
```

```python
# 自定义batch回调示例
print_batch_callback = callbacks.LambdaCallback(
    on_batch_end=lambda batch, logs: print(batch, logs))  # 需要两个位置参数: `batch`, `logs`
                                                          # 分别代表当前batch的序号和指标
# 训练输出
  20/1500 [..............................] - ETA: 18s - loss: 2.2107 - accuracy: 0.206319 {'loss': 2.0734667778015137, 'accuracy': 0.3140625059604645}
20 {'loss': 2.042363405227661, 'accuracy': 0.331845223903656}
21 {'loss': 2.0119757652282715, 'accuracy': 0.3480113744735718}
22 {'loss': 1.9936082363128662, 'accuracy': 0.3586956560611725}
  24/1500 [..............................] - ETA: 18s - loss: 2.1762 - accuracy: 0.230623 {'loss': 1.967929720878601, 'accuracy': 0.3697916567325592}
24 {'loss': 1.934678316116333, 'accuracy': 0.3824999928474426}
25 {'loss': 1.9046880006790161, 'accuracy': 0.39423078298568726}
26 {'loss': 1.8743277788162231, 'accuracy': 0.40509259700775146}
27 {'loss': 1.838417887687683, 'accuracy': 0.4151785671710968}  
  

# 自定义epoch回调示例
print_epoch_callback = callbacks.LambdaCallback(
    on_epoch_end=lambda epoch, logs: print(epoch, logs))  # 需要两个位置参数: `epoch`, `logs`
                                                          # 分别代表当前epoch的序号和指标
# 训练输出
Epoch 1/10
1500/1500 [==============================] - 20s 13ms/step - loss: 0.3875 - accuracy: 0.8781 - val_loss: 0.0871 - val_accuracy: 0.9728
0 {'loss': 0.16673415899276733, 'accuracy': 0.9478958249092102, 'val_loss': 0.0870571881532669, 'val_accuracy': 0.9728333353996277}
Epoch 2/10
1500/1500 [==============================] - 19s 13ms/step - loss: 0.0564 - accuracy: 0.9816 - val_loss: 0.0502 - val_accuracy: 0.9848
1 {'loss': 0.05158458277583122, 'accuracy': 0.9834166765213013, 'val_loss': 0.0502360574901104, 'val_accuracy': 0.9848333597183228}
```

## LearningRateScheduler

## ModelCheckpoint

## TensorBoard

为 TensorBoard 可视化记录日志。

```python
tf.keras.callbacks.TensorBoard(
    log_dir='logs', histogram_freq=0, write_graph=True,
    write_images=False, update_freq='epoch', profile_batch=2,
    embeddings_freq=0, embeddings_metadata=None, **kwargs
)
# log_dir      待TensorBoard解析的日志文件的保存路径
# update_freq  若为'batch',则在每个batch结束时记录损失和指标;若为'epoch',则在每个epoch结束时记录损失和指标;
#              若为整数,则每update_freq个batch记录一次损失和指标.注意过于频繁地写日志会减慢你的训练.
```

# datasets

## cifar10

CIFAR10 数据集。

### load_data()

同 `tf.keras.datasets.mnist.load_data()`。

## cifar100

CIFAR100 数据集。

### load_data()

同 `tf.keras.datasets.mnist.load_data()`。

## mnist

MNIST 数据集。

### load_data()

```python
tf.keras.datasets.mnist.load_data(path='mnist.npz')
# path       数据集缓存在本地的路径.若该路径为相对路径,则视作相对于`~/.keras/datasets`的路径;
#            若该路径不存在,则在线下载并保存到此路径;
```

# layers

层是进行数据处理的模块，它输入一个张量，然后输出一个张量。尽管有一些层是无状态的，更多的层都有其权重参数，通过梯度下降法学习。`tf.keras.layers` 下内置了深度学习中大量常用的的预定义层，同时也允许我们自定义层。

## Dense

全连接层（densely connected layer，fully connected layer）是 Keras 中最基础和常用的层之一，对输入矩阵 $A$ 进行 $f(A\pmb w+b)$ 的线性变换 + 激活函数操作。如果不指定激活函数,即是纯粹的线性变换 $A\pmb w+b$。具体而言，给定输入张量 `input =[batch_size,input_dim]`，该层对输入张量首先进行 `tf.matmul(input,kernel)+ bias` 的线性变换（`kernel` 和 `bias` 是层中可训练的变量），然后对线性变换后张量的每个元素通过激活函数 `activation`，从而输出形状为 `[batch_size, units]` 的二维张量。

[![../../_images/dense.png](https://tf.wiki/_images/dense.png)](https://tf.wiki/_images/dense.png)

* `activation`：激活函数，默认为无激活函数。常用的激活函数包括 `tf.nn.relu`、`tf.nn.tanh` 和 `tf.nn.sigmoid` 
* `use_bias`：是否加入偏置向量 `bias`，默认为 `True` 
* `kernel_initializer`、`bias_initializer`：权重矩阵 `kernel` 和偏置向量 `bias` 两个变量的初始化器。默认为 `tf.glorot_uniform_initializer`。设置为 `tf.zeros_initializer` 表示将两个变量均初始化为全 0

该层包含权重矩阵  `kernel=[input_dim,units]` 和偏置向量 `bias=[units]`  两个可训练变量，对应于 $f(A\pmb w+b)$ 中的 $\pmb w$ 和 $b$。

```python
tf.keras.layers.Dense(units, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, **kwargs)
# units         神经元的个数,即输出空间的维数
# activation    激活函数,默认为无激活函数。常用的激活函数包括 `tf.nn.relu`、`tf.nn.tanh` 和 `tf.nn.sigmoid` 
```

```python
>>> model = models.Sequential([
    layers.Dense(16, activation='relu', input_shape=(16,)),
    layers.Dense(4, activation='relu'),
    layers.Dense(1),
])
>>> model.summary()
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 16)                272           # 16*16+16 = 272
_________________________________________________________________
dense_1 (Dense)              (None, 4)                 68            # 16*4+4 = 68
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 5             # 4*1+1 = 5
=================================================================
Total params: 345
Trainable params: 345
Non-trainable params: 0
_________________________________________________________________
```

## Conv2D

卷积层。

其包含的主要参数如下：

* `filters`：输出特征映射的个数
* `kernel_size`：整数或整数 1×2 向量，（分别）表示二维卷积核的高和宽
* `strides`：整数或整数 1×2 向量，（分别）表示卷积的纵向和横向步长
* `padding`：`"valid"` 表示对于不够卷积核大小的部分丢弃，`"same"` 表示对于不够卷积核大小的部分补 0，默认为 `"valid"`
* `activation`：激活函数，默认为无激活函数
* `use_bias`：是否使用偏置，默认为使用

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

## MaxPool2D

对二维数据（图片）进行最大汇聚（池化）操作。具有别名 `MaxPooling2D`。

```python
tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None, **kwargs)
# pool_size    窗口大小.若为2个整数组成的元组,则分别代表窗口的高和宽;若为1个整数,则同时代表窗口的高和宽.
# strides      步长.若为2个整数组成的元组,则分别代表竖向和横向的步长;若为1个整数,则同时代表竖向和横向的步长.
#              默认与`pool_size`相同.
# padding      若为`'valid'`,则不填充;若为`'same'`,则在右侧和下侧填充0以补全不完整的窗口
# data_format  若为`'channels_last'`,则输入和输出张量的形状为`(batch_size, height, width, channels)`;
#              若为`'channels_first'`,则输入和输出张量的形状为`(batch_size, channels, height, width)`.
#              默认为`'channels_last'`.
```

```python
>>> x = tf.constant([[1., 2., 3.],
                     [4., 5., 6.],
                     [7., 8., 9.]])
>>> x = tf.reshape(x, [1, 3, 3, 1])
>>> max_pool_2d = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='valid')
>>> max_pool_2d(x)
<tf.Tensor: shape=(1, 2, 2, 1), dtype=float32, numpy=
array([[[[5.], [6.]],
        [[8.], [9.]]]], dtype=float32)>

>>> x = tf.constant([[1., 2., 3., 4.],
                     [5., 6., 7., 8.],
                     [9., 10., 11., 12.]])
>>> x = tf.reshape(x, [1, 3, 4, 1])
>>> max_pool_2d = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
>>> max_pool_2d(x)
<tf.Tensor: shape=(1, 1, 2, 1), dtype=float32, numpy=
array([[[[6.], [8.]]]], dtype=float32)>

>>> x = tf.constant([[1., 2., 3., 4.],
                     [5., 6., 7., 8.],
                     [9., 10., 11., 12.]])
>>> x = tf.reshape(x, [1, 3, 4, 1])
>>> max_pool_2d = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same')
>>> max_pool_2d(x)                                              
<tf.Tensor: shape=(1, 3, 4, 1), dtype=float32, numpy=
array([[[[ 6.], [ 7.], [ 8.], [ 8.]],                       # 右侧填充一列0,下侧填充一行0
        [[10.], [11.], [12.], [12.]],
        [[10.], [11.], [12.], [12.]]]], dtype=float32)>

>>> x = tf.constant([[1., 2., 3., 4.],
                     [5., 6., 7., 8.],
                     [9., 10., 11., 12.]])
>>> x = tf.reshape(x, [1, 3, 4, 1])
>>> max_pool_2d = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')
>>> max_pool_2d(x)                                                                              
<tf.Tensor: shape=(1, 2, 2, 1), dtype=float32, numpy=
array([[[[ 6.], [ 8.]],                                     # 右侧不填充,下侧填充一行0
        [[10.], [12.]]]], dtype=float32)>
```

## Embedding

> 参考[单词嵌入向量](https://www.tensorflow.org/tutorials/text/word_embeddings)

嵌入层可以被理解为整数（单词索引）到密集向量的映射。嵌入层输入形如 `(samples,sequence_length)` 的二维整数张量，因此所有的整数序列都应填充或裁剪到相同的长度；输出形如 `(samples,sequence_ length,embedding_dimensionality)` 的三维浮点张量，再输入给 RNN 层处理。

嵌入层在刚初始化时所有的权重参数都是随机的，就如同其它的层一样。在训练过程中这些参数会根据反向传播算法逐渐更新，嵌入空间会逐渐显现出更多结构（这些结构适应于当前的具体问题）。

其包含的主要参数如下：

* `input_dim`：字典的规模
* `output_dim`：嵌入向量的规模
* `mask_zero`：是否将输入中的 0 看作填充值而忽略之，默认为 `False`
* `input_length`：输入序列的长度（如果该长度固定），默认为 `None`；如果此嵌入层后接 `Flatten` 层，再接 `Dense` 层，则必须制定此参数

示例见 SimpleRNN，LSTM。

## SimpleRNN

SRN 层是最简单的循环神经网络层。

其包含的主要参数如下：

* `units`：输出向量的维度
* `activation`：激活函数，默认为 `tanh`
* `return_sequences`：`False` 表示最后输出一个向量，即序列到类别模式；`True` 表示每个时间步长输出一个向量，即序列到序列模式。

实践中一般使用 LSTM 和 GRU 而非 SRN。

示例：

```python
model = keras.Sequential()
model.add(keras.layers.Embedding(10000, 32))
model.add(keras.layers.SimpleRNN(32))
# 输入timestepsx32的二阶张量,输出32维向量,即序列到类别模式
model.summary()
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# embedding (Embedding)        (None, None, 32)          320000    
# _________________________________________________________________
# simple_rnn (SimpleRNN)       (None, 32)                2080      
# =================================================================
# Total params: 322,080
# Trainable params: 322,080
# Non-trainable params: 0
# _________________________________________________________________
```

```python
model = keras.Sequential()
model.add(keras.layers.Embedding(10000, 32))
model.add(keras.layers.SimpleRNN(32, return_sequences=True))
# 输入timestepsx32的二阶张量,输出timestepsx32的二阶张量,即序列到序列模式
model.summary()
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# embedding (Embedding)        (None, None, 32)          320000    
# _________________________________________________________________
# simple_rnn (SimpleRNN)       (None, None, 32)          2080      
# =================================================================
# Total params: 322,080
# Trainable params: 322,080
# Non-trainable params: 0
# _________________________________________________________________
```

```python
model = keras.Sequential()
model.add(keras.layers.Embedding(10000, 32))
model.add(keras.layers.SimpleRNN(32, return_sequences=True))
# SRN的堆叠
model.add(keras.layers.SimpleRNN(32, return_sequences=True))
model.add(keras.layers.SimpleRNN(32, return_sequences=True))
model.add(keras.layers.SimpleRNN(32))
model.summary()
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# embedding (Embedding)        (None, None, 32)          320000    
# _________________________________________________________________
# simple_rnn (SimpleRNN)       (None, None, 32)          2080      
# _________________________________________________________________
# simple_rnn_1 (SimpleRNN)     (None, None, 32)          2080      
# _________________________________________________________________
# simple_rnn_2 (SimpleRNN)     (None, None, 32)          2080      
# _________________________________________________________________
# simple_rnn_3 (SimpleRNN)     (None, 32)                2080      
# =================================================================
# Total params: 328,320
# Trainable params: 328,320
# Non-trainable params: 0
# _________________________________________________________________
```

## LSTM

LSTM 层。

其包含的主要参数如下：

* `units`：输出空间的规模

示例：

```python
model = keras.Sequential()
model.add(keras.layers.Embedding(10000, 16))
# 将规模为10000的词典嵌入到16维向量
# 输入256维向量,输出256x16的二阶张量
model.add(keras.layers.LSTM(64))
# 输入256x16的二阶张量,输出64维向量(隐状态),即序列到类别模式
model.add(keras.layers.Dense(16, activation='relu'))
# 全连接层,ReLU激活函数,分类器
model.add(keras.layers.Dense(1, activation='sigmoid'))  
# 全连接层,Logistic激活函数
model.summary()
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# embedding (Embedding)        (None, None, 16)          160000    
# _________________________________________________________________
# lstm (LSTM)                  (None, 64)                20736     
# _________________________________________________________________
# dense (Dense)                (None, 16)                1040      
# _________________________________________________________________
# dense_1 (Dense)              (None, 1)                 17        
# =================================================================
# Total params: 181,793
# Trainable params: 181,793
# Non-trainable params: 0
# _________________________________________________________________
```

```python
model = keras.Sequential()
model.add(keras.layers.LSTM(64, 
                            dropout=0.2,
                            recurrent_dropout=0.2,
                            return_sequences=True,
                            # 堆叠了rnn层,必须在每个时间步长输出
                            input_shape=(None, df.shape[-1])))
                            # 尽管这里序列的长度是确定的(120),但也不必传入
model.add(keras.layers.LSTM(64,
                            activation='relu',
                            dropout=0.2,
                            recurrent_dropout=0.2))
# 亦可以使用 layers.GRU
model.add(keras.layers.Dense(1))
```

## GRU

GRU 层。

## Bidirectional

双向 RNN 层在某些特定的任务上比一般的 RNN 层表现得更好，经常应用于 NLP。

RNN 的输入序列存在顺序，打乱或反序都会彻底改变 RNN 从序列中提取的特征。双向 RNN 包含两个一般的 RNN，分别从一个方向处理输入序列。在许多 NLP 任务中，反向处理输入序列能够达到与正向处理相当的结果，并且提取出不同但同样有效的特征，此种情况下双向 RNN 将捕获到更多的有用的模式，但也会更快地过拟合。

![Screenshot from 2020-09-28 19-00-29.png](https://i.loli.net/2020/09/28/i2V36gZhtIv7BJA.png)

`keras` 中，`Bidirectional` 实现为创建一个参数指定的 RNN 层，再创建一个相同的 RNN 层处理反序的输入序列。

示例：

```python
model = keras.Sequential()
model.add(keras.layers.Embedding(max_features, 32))
model.add(keras.layers.Bidirectional(keras.layers.LSTM(32)))
model.add(keras.layers.Dense(1, activation='sigmoid'))
```

## Dropout

示例：

```python
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))
```

# losses

## BinaryCrossentropy

计算标签和预测值之间的交叉熵损失，用于二分类问题。损失函数接受的标签为 `0` 或 `1`，预测值为任意浮点数（若 `from_logits=True`，此时预测值的浮点数通过 logistic 函数映射到 $(0, 1)$ 区间内）或概率值（若 `from_logits=False`）。

> logit 函数是 logistic 函数的反函数。

```python
>>> y_true = [0, 1, 0, 0]
>>> y_pred = [-18.6, 0.51, 2.94, -12.8]
>>> bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)  # 对预测值应用logistic函数
>>> bce(y_true, y_pred)
<tf.Tensor: shape=(), dtype=float32, numpy=0.865458>
```

## CategoricalCrossentropy

计算标签和预测值之间的交叉熵损失，用于二分类或多分类问题。损失函数接受的预测值为表示各类别概率值的向量，标签为相应的 one-hot 向量。

```python
>>> y_true = [[0, 1, 0], [0, 0, 1]]
>>> y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
>>> cce = tf.keras.losses.CategoricalCrossentropy()
>>> cce(y_true, y_pred)
<tf.Tensor: shape=(), dtype=float32, numpy=1.1769392>
```

## CosineSimilarity

计算标签和预测值之间的余弦相似度。返回值介于 -1 到 1 之间，-1 表示方向相同，1 表示方向相反，0 表示正交。

```python
>>> y_true = [1., 1.]
>>> y_pred = [2., 2.]
>>> cosine_loss = tf.keras.losses.CosineSimilarity()
>>> cosine_loss(y_true, y_pred)
<tf.Tensor: shape=(), dtype=float32, numpy=-0.99999994>

>>> y_true = [1., 1.]
>>> y_pred = [2., -2.]
>>> cosine_loss = tf.keras.losses.CosineSimilarity()
>>> cosine_loss(y_true, y_pred)
<tf.Tensor: shape=(), dtype=float32, numpy=-0.0>
```

## Hinge

## KLDiverence

计算标签和预测值之间的 KL 散度。

## Loss

## MeanAbsoluteError

计算标签和预测值之间的平均绝对误差。

```python
>>> y_true = [0., 1.]
>>> y_pred = [0.1, 0.8]
>>> mae = tf.keras.losses.MeanAbsoluteError()
>>> mae(y_true, y_pred)
<tf.Tensor: shape=(), dtype=float32, numpy=0.14999999>
```

## MeanSquareError

计算标签和预测值之间的平均平方误差。

```python
>>> y_true = [0., 1.]
>>> y_pred = [0.1, 0.8]
>>> mse = tf.keras.losses.MeanSquaredError()
>>> mse(y_true, y_pred)
<tf.Tensor: shape=(), dtype=float32, numpy=0.024999999>
```

## Poisson

计算标签和预测值之间的泊松损失。

## Reduction

## SparseCategoricalCrossentropy

计算标签和预测值之间的交叉熵损失，用于二分类或多分类问题。损失函数接受的预测值为表示各类别概率值的向量，标签为类别的序号。

```python
>>> y_true = [1, 2]
>>> y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
>>> scce = tf.keras.losses.SparseCategoricalCrossentropy()
>>> scce(y_true, y_pred)
<tf.Tensor: shape=(), dtype=float32, numpy=1.1769392>
```

## SquaredHinge

# metrics

## Accuracy

准确率。

```python
>>> m = tf.keras.metrics.Accuracy()
>>> m.update_state([1, 2, 3, 4], [0, 2, 3, 4])
>>> m.result().numpy()
0.75
```

```python
model.compile(optimizer='sgd',
              loss='mse',
              metrics=[tf.keras.metrics.Accuracy()])
```

### update_state

```python
update_state(y_true, y_pred, sample_weight=None)
# y_true         真实值
# y_pred         预测值
# sample_weight  样本权重
```

## BinaryAccuracy

准确率（）。

### update_state

见 [Accuracy](#Accuracy)。

## CategoricalAccuracy

准确率（类别概率对独热标签）。

```python
>>> m = tf.keras.metrics.CategoricalAccuracy()
>>> m.update_state([[0, 0, 1], [0, 1, 0], [0, 1, 0]], [[0.1, 0.1, 0.8],
                [0.05, 0.95, 0], [0.3, 0.3, 0.4]])
>>> m.result().numpy()
0.6666667
```

### update_state

见 [Accuracy](#Accuracy)。

## FalseNegatives

## FalsePositives

## KLDivergence

KL 散度。

## Mean

平均值。

```python
>>> m = tf.keras.metrics.Mean()
>>> m.update_state([1, 3, 5, 7])
>>> m.result().numpy()
4.0
```

## Metric

指标的基类。

### reset_state

重置状态，即移除所有样本。

### result

计算并返回指标值张量。

### update_state

更新状态，即添加样本。

## Precision

精确率。

## Recall

召回率。

## Sum

求和。

```python
>>> m = tf.keras.metrics.Sum()
>>> m.update_state([1, 2, 3, 4])
>>> m.result().numpy()
10.0
```

### update_state

```python
update_state(values, sample_weight=None)
# values         样本值
# sample_weight  样本权重
```

## TrueNegatives

真阴性的数量。

```python
```

### update_state

见 [TruePositives](#TruePositives)。

## TruePositives

真阳性的数量。

```python
>>> m = tf.keras.metrics.TruePositives()
>>> m.update_state([0, 1, 1, 1], [1, 0, 1, 1])
>>> m.result().numpy()
2.0
```

### update_state

```python
update_state(y_true, y_pred, sample_weight=None)
# y_true         真实值
# y_pred         预测值
# sample_weight  样本权重
```

# Model

![](https://i.loli.net/2020/09/27/hvxUc9eyiqJkGVu.png)

### compile()

配置模型以准备训练。

```python
keras.Model.compile(optimizer='rmsprop', loss=None, metrics=None, loss_weights=None, weighted_metrics=None, run_eagerly=None, steps_per_execution=None, **kwargs)
# optimizer      优化器,是一个`keras.optimizers.Optimizer`实例
# loss           损失函数,是一个字符串,`keras.losses.Loss`实例或自定义函数.可以使用的字符串包括
#                   `keras.losses`下所有损失函数的名称;自定义函数是形如`loss = fn(y_true, y_pred)`
#                   的函数,其中`y_true`和`y_pred`的形状为`[batch_size, d0, ..., dN]`
# metrics        评价指标,是一个字符串,`keras.metrics.Metric`实例或自定义函数的列表.可以使用的字符串
#                   包括`keras.metrics`下所有指标函数的名称;自定义函数是形如`loss = fn(y_true, y_pred)`
#                   的函数,其中`y_true`和`y_pred`的形状为`[batch_size, d0, ..., dN]`
# 
```

### evaluate()

返回模型在测试模式中的损失和指标值。

```python
keras.Model.evaluate(x=None, y=None, batch_size=None, verbose=1, sample_weight=None, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False, return_dict=False)
# x             输入数据,可以是一个numpy数组,tensorflow张量,返回元组`(x, y)`的`tf.data`数据集、
#                  生成器或`keras.utils.Sequence`实例,etc
# y             目标数据,可以是一个numpy数组或tensorflow张量.若`x`是一个`tf.data`数据集、生成器或
#                  `keras.utils.Sequence`实例,则不可指定此参数
# batch_size    批次规模,即每一次梯度更新计算的样本数量,默认为32.若`x`是一个`tf.data`数据集、生成器或
#                  `keras.utils.Sequence`实例,则不可指定此参数
# verbose       若为0,控制台无输出;若为1,输出进度条
# callbacks     测试过程中使用的回调,是一个`keras.callbacks.Callback`实例的列表
# 

```

### fit()

```python
keras.Model.fit(x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, 
validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, 
sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, 
validation_batch_size=None, validation_freq=1, max_queue_size=10, workers=1, 
use_multiprocessing=False)
# x             输入数据,可以是一个numpy数组,tensorflow张量,返回元组`(x, y)`的`tf.data`数据集、
#                  生成器或`keras.utils.Sequence`实例,etc
# y             目标数据,可以是一个numpy数组或tensorflow张量.若`x`是一个`tf.data`数据集、生成器或
#                  `keras.utils.Sequence`实例,则不可指定此参数
# batch_size    批次规模,即每一次梯度更新计算的样本数量,默认为32.若`x`是一个`tf.data`数据集、生成器或
#                  `keras.utils.Sequence`实例,则不可指定此参数
# epochs        回合数,即训练过程迭代训练集的次数
# verbose       若为0,控制台无输出;若为1,输出每个epoch的进度条和结果;若为2,仅输出每个epoch的结果
# callbacks     训练过程中使用的回调,是一个`keras.callbacks.Callback`实例的列表
# validation_split   使用多少比例的训练数据作为验证数据.验证数据取自`x`和`y`的末尾样本.若`x`是一个
#                       `tf.data`数据集、生成器或`keras.utils.Sequence`实例,则不可指定此参数
# validation_data    验证集,可以是一个numpy数组或tensorflow张量的元组`(x, y)`、`tf.data`数据集、
#                       生成器或`keras.utils.Sequence`实例.此参数会重载`validation_split`
# shuffle       若为`True`,在每个epoch开始前打乱训练数据,仅会打乱batch的顺序.若`x`是一个生成器,则忽略此参数
# class_weight
# sample_weight
# initial_epoch
# steps_per_epoch    每个epoch的最大步数,即每个epoch在训练这么多批次的样本后直接结束.若`x`是一个无限重复的
#                       `tf.data`数据集,则必须指定此参数;若`x`是一个numpy数组或tensorflow张量,则不可指定
#                       此参数
# validation_steps   验证的最大步数,仅在提供了`validation_data`并且是`tf.data`数据集时有效.若为None,
#                       则验证会进行至`validation_data`数据集耗尽
# validation_batch_size   验证的批次规模,默认为`batch_size`.若验证数据是一个`tf.data`数据集、生成器或
#                  `keras.utils.Sequence`实例,则不可指定此参数
# validation_freq    进行验证的频率.若为整数,例如2,则每2个epoch进行一次验证;若为`Container`容器类型,
#                       例如[1, 2, 10],则在第1、2、10个epoch进行验证
# workers       基于进程的并行的最大进程数.仅在`x`为生成器或`keras.utils.Sequence`实例时有效
# use_multiprocessing     若为True,使用基于进程的并行.仅在`x`为生成器或`keras.utils.Sequence`实例时有效
```

### get_config()

### predict()

### save()

保存模型为 TensorFlow SavedModel 或 HDF5 文件。参见[`keras.models.save_model()`](#save_model（)。

```python
keras.Model.save(filepath, overwrite=True, include_optimizer=True, save_format=None, signatures=None, options=None, save_traces=True)
# filepath      保存的文件的路径
# overwrite     若为True,则覆盖已经存在于目标路径的文件;否则在命令行提示用户进行选择
# include_optimizer   若为True,则一并保存优化器的状态
# save_format   若为'tf',则保存为TensorFlow SavedModel文件;若为'h5',则保存为HDF5文件.tf2.x默认为'tf'.
# signatures    
# options
# save_traces
```

### summary()

打印模型的概要。

```python
>>> model = keras.Sequential([
    keras.layers.Dense(5, activation="relu", name="layer1", input_shape=(4,)),
    keras.layers.Dense(3, activation="relu", name="layer2"),
    keras.layers.Dense(1, name="layer3"),
])
>>> model.summary()
Model: "sequential"                                                 # 模型名称
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
layer1 (Dense)               (None, 5)                 25           # 层名称,类型,输出形状和参数数量
_________________________________________________________________   # None表示输入的样本数量
layer2 (Dense)               (None, 3)                 18        
_________________________________________________________________
layer3 (Dense)               (None, 1)                 4         
=================================================================
Total params: 47                                                    # 参数总数
Trainable params: 47                                                # 可训练的参数总数
Non-trainable params: 0                                             # 不可训练的参数总数
_________________________________________________________________
```

### to_json()

返回一个包含模型配置的 JSON 字符串。

```python
>>> model = keras.Sequential([
    keras.layers.Dense(5, activation="relu", name="layer1", input_shape=(4, )),
    keras.layers.Dense(3, activation="relu", name="layer2"),
    keras.layers.Dense(1, name="layer3"),
])
>>> model.compile(
    optimizer='SGD',
    loss='mse',
    metrics=['mae'])
>>> import numpy as np
>>> x = np.random.randn(100, 4)
>>> y = np.random.randn(100, 1)
>>> model.fit(x, y, batch_size=5, epochs=5)
2021-04-22 13:39:24.580192: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:116] None of the MLIR optimization passes are enabled (registered 2)
Epoch 1/5
20/20 [==============================] - 0s 477us/step - loss: 1.4276 - mae: 0.9459
Epoch 2/5
20/20 [==============================] - 0s 400us/step - loss: 1.0236 - mae: 0.8374
Epoch 3/5
20/20 [==============================] - 0s 396us/step - loss: 1.1599 - mae: 0.8905
Epoch 4/5
20/20 [==============================] - 0s 410us/step - loss: 0.9507 - mae: 0.8013
Epoch 5/5
20/20 [==============================] - 0s 397us/step - loss: 0.8971 - mae: 0.7745
<tensorflow.python.keras.callbacks.History object at 0x16d06c970>
>>> model.layers[0].weights
[<tf.Variable 'layer1/kernel:0' shape=(4, 5) dtype=float32, numpy=
array([[-0.72289854,  0.29551595,  0.38591394,  0.46075335,  0.19983876],
       [ 0.02872569,  0.579325  ,  0.20934486,  0.5554217 , -0.29025653],
       [ 0.38993195, -0.16033728,  0.7320385 ,  0.62531203,  0.12029058],
       [-0.79410976,  0.81735367,  0.31564388,  0.37734994,  0.1701165 ]],
      dtype=float32)>, <tf.Variable 'layer1/bias:0' shape=(5,) dtype=float32, numpy=
array([-0.0428277 , -0.0083819 ,  0.01140244, -0.03674569, -0.09126914],
      dtype=float32)>]
>>> config_json = model.to_json()      # 返回包含模型配置的JSON字符串
>>> model1 = keras.models.model_from_json(config_json)   # 从模型配置的JSON字符串初始化模型
>>> model1.summary()                   # 模型的网络结构被还原
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
layer1 (Dense)               (None, 5)                 25        
_________________________________________________________________
layer2 (Dense)               (None, 3)                 18        
_________________________________________________________________
layer3 (Dense)               (None, 1)                 4         
=================================================================
Total params: 47
Trainable params: 47
Non-trainable params: 0
_________________________________________________________________
>>> model1.optimizer                   # 模型的优化器为None,因为未被compile
>>> model1.layers[0].weights           # 模型的网络参数为随机初始值
[<tf.Variable 'layer1/kernel:0' shape=(4, 5) dtype=float32, numpy=
array([[ 0.46788   , -0.76049685, -0.71987045,  0.07750785, -0.64779675],
       [ 0.05875117,  0.11067849,  0.5523362 ,  0.81021357, -0.49546677],
       [-0.08038539,  0.6955117 , -0.18111247, -0.0841682 ,  0.59767616],
       [-0.7479675 , -0.26797855, -0.3999755 ,  0.35369885,  0.26355588]],
      dtype=float32)>, <tf.Variable 'layer1/bias:0' shape=(5,) dtype=float32, numpy=array([0., 0., 0., 0., 0.], dtype=float32)>]
>>> from pprint import pprint
>>> import json
>>> pprint(json.loads(config_json))    # 模型配置的JSON字符串内容
{'backend': 'tensorflow',
 'class_name': 'Sequential',
 'config': {'layers': [{'class_name': 'InputLayer',
                        'config': {'batch_input_shape': [None, 4],
                                   'dtype': 'float32',
                                   'name': 'layer1_input',
                                   'ragged': False,
                                   'sparse': False}},
                       {'class_name': 'Dense',
                        'config': {'activation': 'relu',
                                   'activity_regularizer': None,
                                   'batch_input_shape': [None, 4],
                                   'bias_constraint': None,
                                   'bias_initializer': {'class_name': 'Zeros',
                                                        'config': {}},
                                   'bias_regularizer': None,
                                   'dtype': 'float32',
                                   'kernel_constraint': None,
                                   'kernel_initializer': {'class_name': 'GlorotUniform',
                                                          'config': {'seed': None}},
                                   'kernel_regularizer': None,
                                   'name': 'layer1',
                                   'trainable': True,
                                   'units': 5,
                                   'use_bias': True}},
                       {'class_name': 'Dense',
                        'config': {'activation': 'relu',
                                   'activity_regularizer': None,
                                   'bias_constraint': None,
                                   'bias_initializer': {'class_name': 'Zeros',
                                                        'config': {}},
                                   'bias_regularizer': None,
                                   'dtype': 'float32',
                                   'kernel_constraint': None,
                                   'kernel_initializer': {'class_name': 'GlorotUniform',
                                                          'config': {'seed': None}},
                                   'kernel_regularizer': None,
                                   'name': 'layer2',
                                   'trainable': True,
                                   'units': 3,
                                   'use_bias': True}},
                       {'class_name': 'Dense',
                        'config': {'activation': 'linear',
                                   'activity_regularizer': None,
                                   'bias_constraint': None,
                                   'bias_initializer': {'class_name': 'Zeros',
                                                        'config': {}},
                                   'bias_regularizer': None,
                                   'dtype': 'float32',
                                   'kernel_constraint': None,
                                   'kernel_initializer': {'class_name': 'GlorotUniform',
                                                          'config': {'seed': None}},
                                   'kernel_regularizer': None,
                                   'name': 'layer3',
                                   'trainable': True,
                                   'units': 1,
                                   'use_bias': True}}],
            'name': 'sequential_6'},
 'keras_version': '2.4.0'}
```

### to_yaml()

返回一个包含模型配置的 yaml 字符串。参考[`keras.Model.to_json()`](#to_json（)）。

# models

### load_model()

### model_from_config()

从配置字典初始化一个 keras 模型。

```python

```

### model_from_json()

从模型配置的 JSON 字符串初始化一个 keras 模型实例。返回的模型实例仅包含网络结构，没有被 [compile](#compile（)），网络参数为随机的初始值。参见[`keras.Model.to_json()`](#to_json（)）。

### model_from_yaml()

从模型配置的 yaml 字符串初始化一个 keras 模型实例。返回的模型实例仅包含网络结构，没有被 [compile](#compile（)），网络参数为随机的初始值。参见[`keras.Model.to_yaml()`](#to_yaml（)）。

### save_model()

保存模型为 TensorFlow SavedModel 或 HDF5 文件。

```python
keras.models.save_model(model, filepath, overwrite=True, include_optimizer=True, save_format=None, signatures=None, options=None, save_traces=True)
```

# Sequential

`Sequential` 返回一个 `keras.Model` 对象。`Sequential` 模型将各层线性组合，适用于 FNN，CNN，RNN，其中每一层都有**一个输入张量和一个输出张量**。

以下 `Sequential` 模型

```python
model = keras.Sequential([
    keras.layers.Dense(5, activation="relu", name="layer1"),
    keras.layers.Dense(3, activation="relu", name="layer2"),
    keras.layers.Dense(1, name="layer3"),
])
# Call model on a test input
x = tf.ones((4, 10))
y = model(x)
```

等效于以下功能

```python
# Create 3 layers
layer1 = keras.layers.Dense(5, activation="relu", name="layer1")
layer2 = keras.layers.Dense(3, activation="relu", name="layer2")
layer3 = keras.layers.Dense(1, name="layer3")

# Call layers on a test input
x = tf.ones((4, 10))
y = layer3(layer2(layer1(x)))
```

`Sequential` 模型也可以用 `add()` 方法创建

```python
model = keras.Sequential()
model.add(keras.layers.Dense(5, activation="relu"))
model.add(keras.layers.Dense(3, activation="relu"))
model.add(keras.layers.Dense(1))
```

CNN 模型示例：

```python
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
# FNN, CNN 需要指定输入张量的shape
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dense(10))
```

### 自定义模型

Keras 模型以类的形式呈现，我们可以通过继承 `tf.keras.Model` 这个 Python 类来定义自己的模型。在继承类中，我们需要重写 `__init__()`（构造函数，初始化）和 `call(input)`（模型调用）两个方法，同时也可以根据需要增加自定义的方法。

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

# optimizers

## Adadelta

实现 Adadelta 算法的优化器。

```python
tf.keras.optimizers.Adadelta(
    learning_rate=0.001, rho=0.95, epsilon=1e-07, name='Adadelta',
    **kwargs
)
# learning_rate   学习率
# rho...          参见官方文档https://tensorflow.google.cn/api_docs/python/tf/keras/optimizers/Adadelta
```

## Adagrad

实现 Adagrad 算法的优化器。

```python
tf.keras.optimizers.Adagrad(
    learning_rate=0.001, initial_accumulator_value=0.1, epsilon=1e-07,
    name='Adagrad', **kwargs
)
# learning_rate                   学习率
# initial_accumulator_value...    参见官方文档
#                                 https://tensorflow.google.cn/api_docs/python/tf/keras/optimizers/Adagrad
```

## Adam

实现 Adam 算法的优化器。

```python
tf.keras.optimizers.Adam(
    learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
    name='Adam', **kwargs
)
# learning_rate   学习率
# beta_1...       参见官方文档https://tensorflow.google.cn/api_docs/python/tf/keras/optimizers/Adam
```

## Optimizer

Keras 优化器的基类。

```python
tf.keras.optimizers.Optimizer(
    name, gradient_aggregator=None, gradient_transformers=None, **kwargs
)
```

### apply_gradients()

### from_config()

根据设置创建一个优化器。参见 [`get_config()`](#get_config())。

### get_config()

返回优化器的设置。参见 [`from_config()`](#from_config())。

### get_weights()

返回优化器的当前参数。

### minimize()

### set_weights()

设置优化器的参数。

## SGD

梯度下降优化器。

```python
tf.keras.optimizers.SGD(
    learning_rate=0.01, momentum=0.0, nesterov=False, name='SGD', **kwargs
)
# learning_rate   学习率
# momentum        动量因子
# nesterov        使用Nesterov动量
# name            
```

# regularizers

```python
model = models.Sequential()
model.add(layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001),
                                              # l2 regularization, coefficient = 0.001
activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, kernel_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001),                                    # l1 & l2
activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
```

# utils

## array_to_img()

将 NumPy 数组转换为 PIL 图像实例。

```python
>>> from PIL import Image
>>> import tensorflow as tf
>>> import numpy as np
>>> array = np.random.random(size=(100, 100, 3))
>>> img = tf.keras.preprocessing.image.array_to_img(img)
>>> img.show()
```

## get_file()

从指定 URL 下载一个文件，如果其不在缓存中。返回到下载的文件的路径。

```python
tf.keras.utils.get_file(
    fname=None, origin=None, untar=False, md5_hash=None, file_hash=None,
    cache_subdir='datasets', hash_algorithm='auto',
    extract=False, archive_format='auto', cache_dir=None
)
```

默认情况下位于 URL `origin` 的文件被下载到缓存目录 `~/.keras`，放置在缓存子目录 `datasets`，并被命名为 `fname`。因此文件 `example.txt` 的最终位置将是 `~/.keras/datasets/example.txt`。

`tar`、`tar.gz`、`tar.bz` 和 `zip` 格式的压缩文件可以被解压。传入一个哈希值将会在下载后验证文件。

```python
path_to_downloaded_file = tf.keras.utils.get_file(
    "flower_photos",
    "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz",
    untar=True)
```

## image_dataset_from_directory()

从目录中的图像文件生成一个 `tf.data.Dataset` 实例。

```python
tf.keras.utils.image_dataset_from_directory(
    directory, labels='inferred', label_mode='int',
    class_names=None, color_mode='rgb', batch_size=32, image_size=(256,
    256), shuffle=True, seed=None, validation_split=None, subset=None,
    interpolation='bilinear', follow_links=False,
    crop_to_aspect_ratio=False, **kwargs
)
```

## img_to_array()

将 PIL 图像实例转换为 NumPy 数组。

```python
>>> from PIL import Image
>>> import tensorflow as tf
>>> import numpy as np
>>> array = np.random.random(size=(100, 100, 3))
>>> img = tf.keras.preprocessing.image.array_to_img(img)
>>> img.show()
>>> array = tf.keras.preprocessing.image.img_to_array(img)
```

