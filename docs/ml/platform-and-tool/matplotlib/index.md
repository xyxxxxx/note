# Matplotlib

[Matplotlib](https://matplotlib.org/) 是一个创建静态、动态和交互式可视化的综合库。

!!! abstract "参考"
    * [Getting started](https://matplotlib.org/stable/users/getting_started/)
    * [Tutorials](https://matplotlib.org/stable/tutorials/index.html)
    * [Examples](https://matplotlib.org/stable/gallery/index.html)

## 灰度图片（binary image）

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

## 折线图（line chart）

```python
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

t = np.linspace(0, 10, 100)
sin = np.sin(np.pi * t)
cos = np.cos(np.pi * t)

ax.plot(t, sin, label='s1') # 绘制折线图
ax.plot(t, cos, label='s2')
ax.set_xlim(0, 10)          # 设定坐标轴范围
ax.set_xticks(np.arange(0, 11, 1))  # 设定坐标轴刻度
ax.set_ylim(-1, 1)
ax.set_xlabel('time')       # 坐标轴名称
ax.set_ylabel('s1 and s2')
ax.legend(loc='best')       # 显示图例
ax.grid(True)               # 显示网格

plt.show()
```
