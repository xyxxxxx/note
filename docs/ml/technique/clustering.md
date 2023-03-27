# 聚类

## K-Means

### 参考

* [handson-ml2/09_unsupervised_learning.ipynb](https://github.com/ageron/handson-ml2/blob/master/09_unsupervised_learning.ipynb) -- K-Means
* [k-means clustering - Wikipedia](https://en.wikipedia.org/wiki/K-means_clustering)

### 细节

算法：

1. 首先随机初始化 $k$ 个质心（centroid）：从数据集中随机选择 $k$ 个不同的样本，并将质心放置在它们的位置上。
2. 重复以下步骤，直到收敛（即直到质心不再移动）：
    1. 将每个样本分配给最近的质心。
    2. 更新质心为分配给它们的样本的平均值。

示例：

![](https://s2.loli.net/2023/03/08/9ebzSCqwKg7MdZ8.png)

!!! note "说明"
    此算法的计算复杂度通常线性于样本数量 $m$、cluster 数量 $k$ 和维数 $n$。尽管如此，这只在数据具有聚类结构时成立。如果数据不具有聚类结构，那么在最坏的情形下计算复杂度会随着样本数量增加而指数上升。在实践中，这一情形基本不会出现，因此 K-Means 通常是最快的聚类算法之一。

尽管此算法保证收敛，但它可能不收敛到最优解（而是收敛到局部最优解）。它是否收敛到最优解取决于质心的初始化。下图展示了算法可能收敛到的两个次优解：

![](https://s2.loli.net/2023/03/08/UhNfp74PZSLDio1.png)

下列方法可以改进质心的初始化以降低收敛到次优解的风险：

* 如果你正好知道这些质心的大致位置，那么可以直接提供这些位置作为初始值。
* 以不同的随机初始化多次运行算法，选取其中的最优解（即所有样本到离它最近的质心的距离平方的平均值最小）。
* 使用 K-Means++ 算法：[k-means++: The advantages of careful seeding (Arthur, 2006)](https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf) 在 K-Means 算法的基础上改进了质心的初始化方法，如下所述：

    1. 从数据集中均匀随机选择一个质心 $c_1$。
    2. 选择一个样本 $x_i$ 作为新的质心 $c_i$，概率为 $D(x_i)^2/\sum_{j=1}^m D(x_j)^2$，其中 $D(x_i)$ 是样本 $x_i$ 与已选择的最近质心之间的距离。该概率分布确保距离已选择的质心更远的样本更有可能被选为质心。
    3. 重复上一个步骤，直到选择了所有 $k$ 个质心。

    K-Means++ 算法的其余部分就是普通的 K-Means。通过这种初始化，K-Means++ 算法很少会收敛到次优解，因此可以大大减少 n_init。大多数情况下，这在很大程度上弥补了初始化过程的额外复杂性。

K-Means 还存在诸多变体，请参阅 [Variations](https://en.wikipedia.org/wiki/K-means_clustering#Variations)。

## HAC

### 参考

* [Hierarchical clustering - Wikipedia](https://en.wikipedia.org/wiki/Hierarchical_clustering)
* [Single-linkage clustering - Wikipedia](https://en.wikipedia.org/wiki/Single-linkage_clustering)

### 细节

![](https://s2.loli.net/2023/03/08/a9HnBEVuJmtKwsT.png)

## DBSCAN

### 参考

* [handson-ml2/09_unsupervised_learning.ipynb](https://github.com/ageron/handson-ml2/blob/master/09_unsupervised_learning.ipynb) -- DBSCAN
* [DBSCAN - Wikipedia](https://en.wikipedia.org/wiki/DBSCAN)

### 细节

![](https://s2.loli.net/2023/03/14/Jo8vYrsB9fwijZ6.png)

## Gaussian Mixture Model
