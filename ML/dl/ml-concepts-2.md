## Occam's Razor奥卡姆剃刀

奥卡姆剃刀原理是由14世纪逻辑学家William of Occam提出的一个解决问题的法则：“如无必要，勿增实体”。奥卡姆剃刀的思想和机器学习中的正则化思想十分类似：简单的模型泛化能力更好。如果有两个性能相近的模型，我们应该选择更简单的模型。因此，在机器学习的学习准则上，我们经常会引入参数正则化来限制模型能力，避免过拟合。



## overfitting过拟合

给定一个假设空间 $\mathcal{F}$，一个假设 $f$ 属于 $\mathcal{F}$，如果存在其他的假设 $f′$ 也属于 $\mathcal{F}$，使得在训练集上 $f$ 的损失比 $f′$ 的损失小，但在整个样本空间上 $f′$ 的损失比 $f$ 的损失小，那么就说假设 $f$ 过度拟合训练数据。

和过拟合相反的一个概念是欠拟合，即模型不能很好地拟合训练数据，在训练集上的错误率比较高。欠拟合一般是由于模型能力不足造成的。

![Screenshot from 2020-09-01 16-49-14.png](https://i.loli.net/2020/09/01/Lm9NMzw7JC1gRrT.png)



## precision and recall精确率和召回率

模型预测测试集的一个样本的结果可以分为四种情况，用**混淆矩阵（confusion matrix）**来表示：

![Screenshot from 2020-09-01 18-01-03.png](https://i.loli.net/2020/09/01/KjRt9NuwCUocJ71.png)

精确率定义为
$$
\mathcal{P}=\frac{TP}{TP+FP}
$$

召回率定义为
$$
\mathcal{R}=\frac{TP}{TP+FN}
$$
**F值（F measure）**为精确率和召回率的调和平均：
$$
\mathcal{F}=\frac{(1+\beta^2)\times \mathcal{P} \times \mathcal R}{\beta^2\times \mathcal{P}+ \mathcal{R}}
$$
其中 $β$ 用于平衡精确率和召回率的重要性，一般取值为1。 $β = 1$ 时的 $F$ 值称为 $F1$ 值，是精确率和召回率的调和平均。




## reinforcement learning, RL强化学习

强化学习是一类通过交互来学习的机器学习算法。在强化学习中，智能体根据环境的状态做出一个动作，并得到即时或延时的奖励。智能体在和环境的交互中不断学习并调整策略，以取得最大化的期望总回报。



## regression回归

见supervised learning



## regularization正则化

我们引入正则化参数来限制模型能力，使其不要过度地最小化经验风险。



## representation learning表示学习

见feature learning



## ROC curve ROC曲线

见[ROC曲线](https://zh.wikipedia.org/wiki/ROC%E6%9B%B2%E7%BA%BF)



## sample space样本空间

输入空间 $\mathcal{X}$ 和输出空间 $\mathcal{Y}$ 构成了一个样本空间 $\mathcal{X}\times \mathcal{Y}$。



## supervised learning监督学习

如果机器学习的目标是建模样本的特征 $\boldsymbol x$ 和标签 $y$ 之间的关系： $y=f(\boldsymbol x; θ)$ 或 $p(y|\boldsymbol x; θ)$，并且训练集中每个样本都有标签，那么这类机器学习称为监督学习。根据标签类型的不同，监督学习又可以分为回归问题、分类问题和结构化学习问题：

1. 回归问题中的标签 $y$ 是连续值(实数或连续整数)， $f(\boldsymbol x; θ)$ 的输出也是连续值
2. 分类问题中的标签 $y$ 是离散的类别(符号)。在分类问题中，学习到的模型也称为分类器(classifier)；分类问题根据其类别数量又可分为二分类(binary classification)和多分类(multi-class classification)问题.
3. 结构化学习……



## underfitting欠拟合

见overfitting



## unsupervised learning, UL无监督学习

无监督学习是指从不包含目标标签的训练样本中自动学习到一些有价值的信息。典型的无监督学习问题有聚类、密度估计、特征学习、降维等。



## validation set验证集

见early stop

