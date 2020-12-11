TensorBoard是TensorFlow的可视化工具包，提供了机器学习实验所需的可视化功能和工具：

+ 跟踪和可视化损失及准确率等指标
+ 可视化模型图（操作和层）
+ 查看权重、偏差或其他张量随时间变化的直方图
+ 将嵌入投射到较低的维度空间
+ 显示图片、文字和音频数据
+ 剖析 TensorFlow 程序

 

# Get started

## Colab notebook

```python
# 加载 TensorBoard notebook extension
%load_ext tensorboard
```

```python
# 启动TensorBoard
%tensorboard --logdir logs
```



## 本机

```python
# 启动TensorBoard
tensorboard --logdir logs
```



## Jupyter notebook

