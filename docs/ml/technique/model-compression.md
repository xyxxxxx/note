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

如果教师网络（通过 softmax 层）输出的概率分布过于集中，和 label 几乎相同，那么知识蒸馏就失去了作用。由此我们为 softmax 层引入一个温度的概念，所有的值在输入 softmax 层之前要先除以一个温度常数 $T$。显然，当 $T>1$ 时，输出的概率分布更分散（平滑）；当 $T<1$ 时，输出的概率分布更集中。

## 量化（Quantization）

### 论文

* [Fixed Point Quantization of Deep Convolutional Networks (Lin, 2015)](https://arxiv.org/abs/1511.06393)
* [BinaryConnect: Training Deep Neural Networks with binary weights during propagations (Courbariaux, 2015)](https://arxiv.org/abs/1511.00363)：使用二元权重 +1 和 -1；二元权重可以起到防止过拟合的效果
* [Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1 (Courbariaux, 2016)](https://arxiv.org/abs/1602.02830)

### 细节

量化是将模型中的浮点数参数转化为更小的整数或定点数，从而大幅减少模型的存储需求和计算复杂度，提高模型的运行速度和能耗效率。具体有以下方法：

* 将浮点数近似用整数或定点数表示，如对称量化
* 权重聚类（weight clustering）：$n$ 位可以表示 $2^n$ 种权重
* 直接使用二元权重（+1 和 -1）

## 动态计算（Dynamic Computation）
 
### 论文

* [Multi-Scale Dense Networks for Resource Efficient Image Classification (Huang, 2017)](https://arxiv.org/abs/1703.09844)：使用动态深度
* [Slimmable Neural Networks (Yu, 2018)](https://arxiv.org/abs/1812.08928)：使用动态宽度
* [SkipNet: Learning Dynamic Routing in Convolutional Networks (Wang, 2017)](https://arxiv.org/abs/1711.09485)：基于样本难度
 
### 细节

动态计算主要用于减小模型的计算量和内存占用，以便在移动设备和嵌入式系统等资源受限的环境下进行高效推理。

动态计算的核心思想是在推理过程中动态地计算模型的某些部分，而不是事先将整个模型编译为静态图像。这可以通过在前向传播期间仅计算必要的部分来实现，而忽略不需要的部分。具体来说，动态计算通常涉及到以下两个技术：

* 延迟计算：在推理期间，只有当某个部分被使用时，才对其进行计算。这可以通过使用条件控制流来实现，例如 if-else 和 while 循环语句。这种方法可以减少不必要的计算和内存占用，提高推理速度。
* 量化计算：在推理期间，使用低精度的数据类型来表示模型参数和中间结果，例如 8 位整数或 16 位浮点数。这可以显著减少内存占用和计算时间，并且可以通过使用特殊硬件加速器来进一步提高性能。
