# 模型压缩

## 参考

* [【機器學習2021】神經網路壓縮 (Network Compression) (一) - 類神經網路剪枝 (Pruning) 與大樂透假說 (Lottery Ticket Hypothesis)](https://www.youtube.com/watch?v=gmsMY5kc-zw&list=PLJV_el3uVTsMhtt7_Y6sgTHGHp1Vb2P2J&index=36)
* [【機器學習2021】神經網路壓縮 (Network Compression) (二) - 從各種不同的面向來壓縮神經網路](https://www.youtube.com/watch?v=gmsMY5kc-zw&list=PLJV_el3uVTsMhtt7_Y6sgTHGHp1Vb2P2J&index=37)

## 剪枝（Pruning）

### 论文

* [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks (Frankle, 2018)](https://arxiv.org/abs/1803.03635)
* [Deconstructing Lottery Tickets: Zeros, Signs, and the Supermask (Zhou, 2019)](https://arxiv.org/abs/1905.01067)
* [Rethinking the Value of Network Pruning](https://arxiv.org/abs/1810.05270)

### 细节

大规模的神经网络通常是过参数化的（over-parameterized），其中包含大量多余的权重或神经元，去掉它们可以减小检查点文件大小，降低推理时占用的计算和存储资源。

剪枝的基本流程如下：

![](https://s2.loli.net/2023/02/23/M71pkQ43PnjRzlC.png)

直接剪掉权重会产生不规则的网络，难以实现和使用 GPU 加速，因此通常用 0 填充这些权重，但这就导致网络规模并未有效减小，只是权重变得稀疏：

![](https://s2.loli.net/2023/02/23/ebP8TmaHizKVxup.png)

实验显示即使剪掉 95+% 的权重，计算也未能得到加速，甚至会严重减速：

![](https://s2.loli.net/2023/02/23/OjIKqvrDxS95AaE.png)

剪掉神经元得到的网络仍然是规则的，容易实现和使用 GPU 加速：

![](https://s2.loli.net/2023/02/23/FMu3woNkR2piced.png)

普遍认为，大的网络要比小的网络更容易训练成功。换言之，直接训练小的网络，往往达不到训练大的网络再对其进行剪枝所得到的指标。

#### 大乐透假说

[大乐透假说（The Lottery Ticket Hypothesis）](https://arxiv.org/abs/1803.03635)认为，大的网络可以视为很多小的网络（称为子网络）的组合，训练大的网络就相当于同时训练这些子网络。任意一个子网络（在这一组初始参数下）不一定能够训练成功，但只要存在一个子网络训练成功，大的网络就训练成功（然后剪掉该子网络以外的部分）。

下面的实验支持了大乐透假说，即初始参数决定了哪一个子网络可以最成功地被训练，换一组初始参数后，同样的一个子网络的训练效果会大打折扣。

![](https://s2.loli.net/2023/02/24/z8N6GVrg45JuchY.png)

大乐透假说是否成立仍存在争议，亦有一些实验的结果与大乐透假说相矛盾。

#### 剪枝策略

[Deconstructing Lottery Tickets: Zeros, Signs, and the Supermask (Zhou, 2019)](https://arxiv.org/abs/1905.01067) 测试了多种剪枝策略，发现其中 large final 和 magnitude increase 是最佳策略，如下图所示：

![](https://s2.loli.net/2023/02/24/4s6vtu7IVAi38y5.png)

![](https://s2.loli.net/2023/02/24/ThgnSNeE2ZrJyY1.png)

## 知识蒸馏（Knowledge Distillation）

### 论文

* [Distilling the Knowledge in a Neural Network (Hinton, 2015)](https://arxiv.org/abs/1503.02531)

### 细节

剪枝是将大的网络修剪为小的网络，而知识蒸馏则是用大的网络来训练小的网络，如下图所示：

![](https://s2.loli.net/2023/02/24/pFeKBEPjfRNYo3G.png)

为什么这样做有效？一种解释是，教师网络的输出包含了一些额外的信息（相当于传授了一些经验，让学生网络可以少走弯路），比原始的 label 更适合作为学习目标（训练起来也更顺滑）。

集成学习可以通用地提升模型性能，但代价是成倍的训练量和推理量。通过知识蒸馏的方法，我们可以直接训练一个小的网络，来逼近集成模型的输出：

![](https://s2.loli.net/2023/02/24/szqfuOvbAEBdJcr.png)

#### 温度（temperature）

![](https://s2.loli.net/2023/02/24/iSdZOlQAy621Bxg.png)

如果教师网络（通过 softmax 层）输出的概率分布过于集中，和 label 几乎相同，那么知识蒸馏就失去了作用。因此我们为 softmax 层引入一个温度的概念，所有的值在输入 softmax 层之前要先除以一个温度常数 $T$。显然，当 $T>1$，更分散

## 量化（Quantization）

### 论文

* [BinaryConnect: Training Deep Neural Networks with binary weights during propagations (Courbariaux, 2015)](https://arxiv.org/abs/1511.00363)：使用二元权重 +1 和 -1；二元权重可以起到防止过拟合的效果

## 动态计算（Dynamic Computation）
 
### 论文

* [Multi-Scale Dense Networks for Resource Efficient Image Classification](https://arxiv.org/abs/1703.09844)：使用动态深度
* [Slimmable Neural Networks](https://arxiv.org/abs/1812.08928)：使用动态宽度
* [SkipNet: Learning Dynamic Routing in Convolutional Networks](https://arxiv.org/abs/1711.09485)
