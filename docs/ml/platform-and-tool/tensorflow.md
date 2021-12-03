> TensorFlow 的官方教程没有系统性，仿佛多篇教程文章的拼凑。此文档的内容是在阅读了官方教程和 API 并实际使用之后，个人总结而成。

# Keras 建立模型

# 执行模式

eager and graph execution

# 保存和加载模型

# 使用GPU

TensorFlow 代码和 Keras 模型可以运行在单个 GPU 上而无需修改任何代码。运行在（单机或多机上的）多个 GPU 上的方法请参考[分布式训练](#分布式训练)。

## 准备

首先检查当前主机是否有可用的 GPU 设备。

```python
>>> import tensorflow as tf
>>> tf.config.list_physical_devices()
[PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU'),
 PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

> TensorFlow 支持在各种类型的设备上进行计算，包括 CPU 和 GPU。CPU 和 GPU 表示为如下的字符串标识：
>
> + `'/CPU:0'`
> + `'/GPU:0'`：TensorFlow 运行时可见的第一个 GPU 设备的简称
> + `'/job:localhost/replica:0/task:0/device:GPU:0'`：TensorFlow 运行时可见的第一个 GPU 设备的完整名称
>
> 如果一个 TensorFlow 运算同时有 CPU 和 GPU 实现，那么默认情况下该运算将被优先分配给 GPU 设备。例如 `tf.matmul` 同时有 CPU 和 GPU 实现，那么在一个有设备 `CPU:0` 和 `GPU:0` 的系统上，`GPU:0` 将被选择用于运行 `tf.matmul`，除非你显式地请求在另一个设备上运行它。

## 记录使用设备

为了弄清楚运算到底被分配到了哪个设备上，执行 `tf.debugging.set_log_device_placement(True)` 以启用设备放置记录，这时任何的运算分配都会被打印出来。

```python
>>> tf.debugging.set_log_device_placement(True)
>>> a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
>>> a.device
'/job:localhost/replica:0/task:0/device:CPU:0'
>>> b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
>>> b.device
'/job:localhost/replica:0/task:0/device:CPU:0'
>>> c = tf.matmul(a, b)
Executing op MatMul in device /job:localhost/replica:0/task:0/device:GPU:0
>>> c.device
'/job:localhost/replica:0/task:0/device:GPU:0'
```

可以看到，张量 `a`, `b` 默认分配到了 `CPU:0` 上，而运算 `MatMul` 默认分配到了 `GPU:0` 上，即优先分配给了 GPU 设备。

> TensorFlow 运行时会基于运算的实现和当前的可用设备为运算选择一个设备（这里为 `GPU:0`），并且在需要时自动在设备间复制张量（这里将张量 `a`, `b` 复制到 `GPU:0` 上再进行计算）。

## 手动指定设备

如果你想要一个运算运行在指定的设备上而不是让 TensorFlow 自动为你选择，那么你可以使用 `with tf.device()` 来创建一个设备上下文，此上下文中的所有运算都运行在指定的设备上。

```python
>>> tf.debugging.set_log_device_placement(True)
>>> with tf.device('/CPU:0'):                                 # 张量和运算都分配到指定设备`CPU:0`上
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])       # 若不指定,运算将默认分配到GPU设备上
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    c = tf.matmul(a, b)
Executing op MatMul in device /job:localhost/replica:0/task:0/device:CPU:0
```

```python
>>> tf.debugging.set_log_device_placement(True)
>>> with tf.device('/GPU:1'):                                 # 张量和运算都分配到指定设备`GPU:1`上
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])       # 若不指定,张量将默认分配到CPU设备上,而运算将
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])     # 默认分配到`GPU:0`(即序号最小的GPU设备)上
    c = tf.matmul(a, b)
Executing op MatMul in device /job:localhost/replica:0/task:0/device:GPU:1
```

若指定的设备不存在，则会引发一个异常。你可以调用 `tf.config.set_soft_device_placement(True)` 使得 TensorFlow 在设备不存在时自动选择一个存在并且支持的设备。

```python
>>> tf.debugging.set_log_device_placement(True)
>>> tf.config.set_soft_device_placement(False)                # 禁用软设备放置
>>> with tf.device('/GPU:1'):                                 # 若设备不存在,则引发异常
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    c = tf.matmul(a, b)
InvalidArgumentError: Could not satisfy device specification '/job:localhost/replica:0/task:0/device:GPU:1'. enable_soft_placement=0. Supported device types [GPU, CPU]. All available devices [/job:localhost/replica:0/task:0/device:GPU:0, /job:localhost/replica:0/task:0/device:CPU:0]. [Op:MatMul]

>>> tf.debugging.set_log_device_placement(True)
>>> tf.config.set_soft_device_placement(True)                 # 启用软设备放置
>>> with tf.device('/GPU:1'):                                 # 若设备不存在,则依然默认分配张量和运算
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    c = tf.matmul(a, b)
```

## 限制GPU使用和显存增长

默认情况下，TensorFlow 将所有可见的 GPU 的几乎所有显存映射到进程上，这是为了减少显存碎片以更加高效地利用相对宝贵的显存资源。要限制 TensorFlow 使用的 GPU ，我们使用 `tf.config.set_visible_devices()` 方法：

```python
>>> tf.config.list_physical_devices()
[PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU'),
 PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
>>> tf.config.set_visible_devices([], 'GPU')               # 不使用GPU
>>> tf.config.list_logical_devices()                       # 初始化设备
[LogicalDevice(name='/device:CPU:0', device_type='CPU')]   # 逻辑设备中没有GPU

>>> tf.config.list_physical_devices()
[PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU'),
 PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU'),
 PhysicalDevice(name='/physical_device:GPU:1', device_type='GPU')]
>>> tf.config.set_visible_devices(tf.config.list_physical_devices('GPU')[0], 'GPU')   # 仅使用GPU:0
>>> tf.config.list_logical_devices()
[LogicalDevice(name='/device:CPU:0', device_type='CPU'),                              # 逻辑设备中仅有GPU:0
 LogicalDevice(name='/device:GPU:0', device_type='GPU')]
```

在有些情况下我们想要只给进程分配部分的显存，或者仅在进程需要时增长显存的使用量，TensorFlow 各提供了一种限制的方法：

调用 ` tf.config.experimental.set_memory_growth()` 以启用内存增长，这会尝试为运行时仅分配需要的显存量：开始时分配非常少的显存，然后随着程序的运行和需要更多的显存，我们扩展分配给 TensorFlow 进程的显存区域。注意我们不会释放显存，因为这会导致显存碎片。在运行时初始化之前，调用 `tf.config.experimental.set_memory_growth()` 方法以为特定 GPU 开启显存增长：

```python
>>> gpus = tf.config.list_physical_devices('GPU')          # 为所有GPU开启显存增长的通常方法
>>> if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
>>> tf.config.list_logical_devices('GPU')
[LogicalDevice(name='/device:GPU:0', device_type='GPU')]
```

> 另一种开启显存增长的方法是设定环境变量 `TF_FORCE_GPU_ALLOW_GROWTH` 为 `True`，……

调用 `tf.config.set_logical_device_configuration()` 以配置逻辑 GPU 设备，并为分配到其上的显存设置硬性限制：

```python
>>> gpus = tf.config.list_physical_devices('GPU')
>>> tf.config.set_logical_device_configuration(            # 为GPU:0配置一个逻辑设备,并分配1GB显存
    gpus[0],                                               # 即第一个GPU仅为TensorFlow运行时分配1GB显存
    [tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
>>> tf.config.list_logical_devices('GPU')
[LogicalDevice(name='/device:GPU:0', device_type='GPU')]
```

这是本地开发时的通常做法，因为需要与其它图形应用共用 GPU。

## 使用单个 GPU 模拟多个 GPU

逻辑设备还可以用于在单个 GPU 上模拟多个 GPU，这使我们在单 GPU 系统上也可以测试分布式训练（只是不支持 NCCL 后端）。

```python
>>> gpus = tf.config.list_physical_devices('GPU')
>>> tf.config.set_logical_device_configuration(
    gpus[0],
    [tf.config.LogicalDeviceConfiguration(memory_limit=1024),
     tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
>>> tf.config.list_logical_devices('GPU')
[LogicalDevice(name='/device:GPU:0', device_type='GPU'),
 LogicalDevice(name='/device:GPU:1', device_type='GPU')]
>>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
WARNING:tensorflow:NCCL is not supported when using virtual GPUs, fallingback to reduction to one device
WARNING:tensorflow:Collective ops is not configured at program startup. Some performance features may not be enabled.
INFO:tensorflow:Using MirroredStrategy with devices ('/job:localhost/replica:0/task:0/device:GPU:0', '/job:localhost/replica:0/task:0/device:GPU:1')
>>> # 进行分布式训练
```

# 分布式训练

> TensorFlow 的分布式架构设计复杂，难以使用，越来越多的用户开始使用 [Horovod](./hovorod.md)。

## 分布式策略

## 使用分布式策略

使用分布式策略时，所有模型相关的变量的创建都应在 `strategy.scope` 内完成，这些变量将被复制到所有模型副本中，并通过 all-reduce 算法保持同步。

```python
with strategy.scope():
    # 建立/编译Keras模型应在`strategy.scope`内完成,因为模型和优化器的创建包含了参数变量的创建
    distributed_model = tf.keras.Sequential([
        layers.Conv2D(params['conv1_feature'],
                      params['conv_kernel'],
                      activation='relu',
                      input_shape=(28, 28, 1)),
        layers.MaxPooling2D((params['maxpool_size'], params['maxpool_size'])),
        layers.Conv2D(params['conv2_feature'],
                      params['conv_kernel'],
                      activation='relu'),
        layers.MaxPooling2D((params['maxpool_size'], params['maxpool_size'])),
        layers.Conv2D(params['conv3_feature'],
                      params['conv_kernel'],
                      activation='relu'),
        layers.Flatten(),
        layers.Dense(params['linear1_size'], activation='relu'),
        layers.Dense(10, activation='softmax'),
    ])
    distributed_model.compile(
        optimizer=optimizers.Adam(learning_rate=params['learning_rate']),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
```

但在 `strategy.scope` 内建立的 Keras 模型的下列高级 API：`model.compile`、`model.fit`、`model.evaluate`、`model.predict` 和 `model.save` 的调用则不必在 `strategy.scope` 内完成。

以下操作可以在 `strategy.scope` 内部或外部调用：

+ 创建数据集

## 集群配置

在多台机器上训练

