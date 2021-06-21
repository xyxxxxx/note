> TensorFlow 的官方教程没有系统性，仿佛多篇教程文章的拼凑。此文档的内容是在阅读了官方教程和 API 并实际使用之后，个人总结而成。

[toc]

# Keras 建立模型



# 执行模式

eager and graph execution



# 保存和加载模型



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

