# tf.config

## get_soft_device_placement()

返回软设备放置是否启用。

```python
>>> tf.config.set_soft_device_placement(True)
>>> tf.config.get_soft_device_placement()
True
>>> tf.config.set_soft_device_placement(False)
>>> tf.config.get_soft_device_placement()
False
```

## get_visible_devices()

返回运行时当前可见的 `PhysicalDevice` 对象的列表。

```python
physical_devices = tf.config.list_physical_devices('GPU')
try:
  # Disable all GPUS
  tf.config.set_visible_devices([], 'GPU')
  visible_devices = tf.config.get_visible_devices()
  for device in visible_devices:
    assert device.device_type != 'GPU'
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass
```

## list_logical_devices()

返回运行时创建的逻辑设备列表。

调用 `list_logical_devices()` 会引发运行时初始化所有可见的 `PhysicalDevice`（一个可见 `PhysicalDevice` 对象默认创建一个 `LogicalDevice` 对象），因而不能继续配置。若不想要初始化运行时，请调用 `list_physical_devices()`。

即使不调用 `list_logical_devices()`，执行任何运算或使用任何 CPU 或 GPU 同样会初始化运行时。

```python
>>> cpus = tf.config.list_physical_devices('CPU')
>>> cpus
[PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU')]
>>> tf.config.set_logical_device_configuration(              # 先设定逻辑设备配置
...   cpus[0],
...   [tf.config.LogicalDeviceConfiguration(),
...    tf.config.LogicalDeviceConfiguration()])
>>> logical_cpus = tf.config.list_logical_devices('CPU')     # 再调用list_logical_devices()以初始化
>>> logical_cpus
[LogicalDevice(name='/device:CPU:0', device_type='CPU'), LogicalDevice(name='/device:CPU:1', device_type='CPU')]

>>> gpus = tf.config.list_physical_devices('GPU')
>>> gpus
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
>>> logical_gpus = tf.config.list_logical_devices('GPU')     # 初始化之后
>>> logical_gpus
[LogicalDevice(name='/device:GPU:0', device_type='GPU')]
>>> tf.config.set_logical_device_configuration(              # 就不能再设定逻辑设备配置
...   gpus[0],
...   [tf.config.LogicalDeviceConfiguration(100),
...    tf.config.LogicalDeviceConfiguration(100)])
RuntimeError: Virtual devices cannot be modified after being initialized
```

## list_physical_devices()

返回运行时可见的物理设备列表。

物理设备指当前主机现有的硬件设备，包括所有已发现的 CPU 和 GPU 设备。此 API 用于在初始化运行时之前查询硬件资源，进而帮助调用更多的配置 API。

```python
>>> tf.config.list_physical_devices()
[PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU'),
 PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU'),
 PhysicalDevice(name='/physical_device:GPU:1', device_type='GPU')]

>>> tf.config.list_physical_devices('GPU')
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU'),
 PhysicalDevice(name='/physical_device:GPU:1', device_type='GPU')]
```

## LogicalDevice

初始化运行时得到的逻辑设备的抽象。

```python
tf.config.LogicalDevice(name, device_type)
```

一个 `LogicalDevice` 对象对应一个 `PhysicalDevice` 对象或者集群上的远程设备。张量或操作可以通过调用 `tf.device()` 并指定 `LogicalDevice`，而被放置在指定的逻辑设备上。

## LogicalDeviceConfiguration

逻辑设备的配置类。

```python
tf.config.LogicalDeviceConfiguration(memory_limit=None, experimental_priority=None)
# memory_limit   为逻辑设备分配的显存
```

此类用于在初始化运行时过程中，指定配置参数将 `PhysicalDevice` 初始化为 `LogicalDevice`。

## PhysicalDevice

本地物理设备的抽象。

```python
tf.config.PhysicalDevice(name, device_type)
```

TensorFlow 可以利用各种设备进行计算，例如 CPU 或者（多个）GPU。在初始化本地设备之前，用户可以自定义设备的一些属性，例如可见性或者内存配置。

## set_logical_device_configuration()

为一个 `PhysicalDevice` 对象设定逻辑设备配置。

一旦初始化运行时，一个可见的 `PhysicalDevice` 对象就默认创建一个 `LogicalDevice` 对象与之关联。在运行时初始化之前指定 `LogicalDeviceConfiguration` 对象列表会在一个 `PhysicalDevice` 对象上创建多个 `LogicalDevice` 对象。

```python
# 将CPU分为2个逻辑设备
>>> cpus = tf.config.list_physical_devices('CPU')
>>> cpus
[PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU')]
>>> tf.config.set_logical_device_configuration(
...   cpus[0],
...   [tf.config.LogicalDeviceConfiguration(),
...    tf.config.LogicalDeviceConfiguration()])
>>> logical_cpus = tf.config.list_logical_devices('CPU')
>>> logical_cpus
[LogicalDevice(name='/device:CPU:0', device_type='CPU'), LogicalDevice(name='/device:CPU:1', device_type='CPU')]
```

```python
# 将GPU分为2个逻辑设备,每个分配100M显存
>>> gpus = tf.config.list_physical_devices('GPU')
>>> gpus 
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
>>> tf.config.set_logical_device_configuration(
    gpus[0],
    [tf.config.LogicalDeviceConfiguration(memory_limit=100),
     tf.config.LogicalDeviceConfiguration(memory_limit=100)])
>>> logical_gpus = tf.config.list_logical_devices('GPU')
>>> logical_gpus
[LogicalDevice(name='/device:GPU:0', device_type='GPU'), LogicalDevice(name='/device:GPU:1', device_type='GPU')]
```

## experimental.set_memory_growth()

设定一个 `PhysicalDevice` 对象是否启用内存增长。

```python
>>> gpus = tf.config.list_physical_devices('GPU')
>>> tf.config.experimental.set_memory_growth(gpus[0], True)  # 启用内存增长
```

## set_soft_device_placement()

设定是否启用软设备放置。若启用，则当指定的设备不存在时自动选择可用的设备。

```python
>>> tf.config.set_soft_device_placement(True)
>>> tf.config.get_soft_device_placement()
True
>>> tf.config.set_soft_device_placement(False)
>>> tf.config.get_soft_device_placement()
False
```

## set_visible_devices()

指定运行时可见的 `PhysicalDevice` 对象的列表。TensorFlow 只会将张量和操作分配到可见的物理设备，因为 `LogicalDevice` 只能创建在可见的 `PhysicalDevice` 上。默认情况下，所有已发现的 CPU 和 GPU 设备都是可见的。

```python
>>> tf.config.list_physical_devices()                                                 # 所有物理设备
[PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU'),
 PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU'),
 PhysicalDevice(name='/physical_device:GPU:1', device_type='GPU')]
>>> tf.config.get_visible_devices()                                                   # 可见物理设备
[PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU'),
 PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU'),
 PhysicalDevice(name='/physical_device:GPU:1', device_type='GPU')]
>>> tf.config.set_visible_devices([], 'GPU')                                          # 设定GPU设备全部不可见
>>> tf.config.get_visible_devices()                                                   # 可见物理设备
[PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU')]
>>> tf.config.set_visible_devices(tf.config.list_physical_devices('GPU')[0], 'GPU')   # 设定GPU:0可见
>>> tf.config.get_visible_devices()                                                   # 可见物理设备
[PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU'),
 PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
>>> tf.config.list_logical_devices()                                                  # 所有逻辑设备
[LogicalDevice(name='/device:CPU:0', device_type='CPU'),
 LogicalDevice(name='/device:GPU:0', device_type='GPU')]
```
