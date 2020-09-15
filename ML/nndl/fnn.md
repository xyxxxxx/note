# 前馈神经网络

**前馈神经网络(Feedforward Neural Network , FNN)**是最早发明的简单人工神经网络。前馈神经网络也经常称为多层感知器(Multi-Layer Perceptron ,MLP)，但多层感知器的叫法并不是十分合理，因为前馈神经网络其实是由多层的 Logistic 回归模型（连续的非线性函数）组成，而不是由多层的感知器（不连续的非线性函数）组成 [Bishop, 2007]。

在前馈神经网络中，各神经元分别属于不同的层，每一层的神经元可以接收前一层神经元的信号，并产生信号输出到下一层。第0层称为输入层，最后一层称为输出层，其他中间层称为隐藏层。整个网络中无反馈，信号从输入层向输出层单向传播，可用一个有向无环图表示。前馈神经网络如下图所示。

![Screenshot from 2020-09-15 13-19-06.png](https://i.loli.net/2020/09/15/Yi1LzNP5mfGgCaH.png)

下表给出了描述前馈神经网络的记号。

| 记号                                       | 含义                                |
| ------------------------------------------ | ----------------------------------- |
| $$L$$                                      | 神经网络的层数                      |
| $$M_i$$                                    | 第$$l$$层神经元的个数               |
| $$f_l(\cdot)$$                             | 第$$l$$层神经元的激活函数           |
| $$\pmb W^{(l)}\in \R^{M_l\times M_{l-1}}$$ | 第$$l-1$$层到第$$l$$层的权重矩阵    |
| $$\pmb b^{(l)}\in \R^{M_l}$$               | 第$$l-1$$层到第$$l$$层的偏置        |
| $$\pmb z^{(l)}\in \R^{M_l}$$               | 第$$l$$层神经元的净输入（净活性值） |
| $$\pmb a^{(l)}\in \R^{M_l}$$               | 第$$l$$层神经元的输出（活性值）     |

令$$\pmb a^{(0)}=\pmb x$$，前馈神经网络通过不断迭代以下公式进行信息传播：
$$
\pmb z^{(l)}=\pmb W^{(l)}\pmb a^{(l-1)}+\pmb b^{(l)}\\
\pmb a^{(l)}=f_l(\pmb z^{(l)})
$$
首先根据第$$l-1$$层神经元的**活性值(activation)**$$\pmb a^{(l−1)}$$计算出第$$l$$层神经元的**净活性值(net activation)**$$z^{(l)}$$，然后经过一个激活函数得到第$$l$$层神经元的活性值。因此，我们也可以把每个神经层看作一个**仿射变换(affine transformation)**和一个非线性变换。

前馈神经网络通过逐层的信息传递得到网络最后的输出$$\pmb a^{(L)}$$。整个网络可以看做一个复合函数$$\phi(\pmb x;\pmb W,\pmb b)$$：
$$
\pmb x= \pmb a^{(0)}\to \pmb z^{(1)}\to \pmb a^{(1)}\to \pmb z^{(2)}\to \cdots \to \pmb z^{(L)}\to \pmb a^{(L)}=\phi(\pmb x;\pmb W,\pmb b)
$$
其中$$\pmb W,\pmb b$$表示所有层的连接权重和偏置。



## 通用近似定理

前馈神经网络具有很强的拟合能力，常见的连续非线性函数都可以用前馈神经网络来近似。

**通用近似定理(Universal Approximation Theorem)** [Cybenko, 1989; Hornik et al., 1989] ：$$\mathcal{I}_D$$是一个$$D$$维的单位超立方体$$[0, 1]^D$$，$$C(\mathcal{I}_D)$$ 是定义在$$\mathcal{I}_D$$上的连续函数集合。对于任意给定的一个函数$$f ∈ C(\mathcal{I}_D)$$ , 存在整数$$M$$，实数$$v_m,b_m ∈ \R$$，实数向量$$w_m ∈ \R^D ,m = 1, ⋯ , M$$和非常数、有界、单调递增的连续函数$$\phi(⋅)$$，使得对于$$\forall \varepsilon>0$$，可以定义函数
$$
F(\pmb x) =\sum_{m=1}^{M} v_m \phi(\pmb w^{\rm T}_m\pmb x + b_m )
$$
作为函数$$f$$的近似实现，即
$$
|F(\pmb x)-f(\pmb x)|<\varepsilon, \forall \pmb x\in \mathcal{I}_D
$$

> 通用近似定理在实数空间$$\R^D$$的有界闭集上依然成立。

根据通用近似定理，对于具有线性输出层和至少一个使用 “挤压” 性质的激活函数的隐藏层组成的前馈神经网络，只要其隐藏层神经元的数量足够，它可以以任意的精度来近似任何一个定义在实数空间$$\R^D$$中的有界闭集函数 [Funa-hashi et al., 1993; Hornik et al., 1989] 。所谓 “挤压” 性质的函数是指像 Sigmoid 函数的有界函数，但神经网络的通用近似性质也被证明对于其他类型的激活函数，比如 ReLU ，也都是适用的。



## 应用到机器学习

根据通用近似定理，神经网络在某种程度上可以作为一个 “万能” 函数来使用，可以用来进行复杂的特征转换或逼近一个复杂的条件分布。

多层前馈神经网络也可以看成是一种特征转换方法，将输入$$\pmb x\in\R^D$$映射到输出$$\phi(\pmb x)\in \R^{D'}$$，再将输出$$\phi(\pmb x)$$作为分类器的输入进行分类。

> 根据通用近似定理，只需要一层隐藏层就可以逼近任何函数，那么多层的神经网络的前几层就可以视作特征转换过程。

特别地，如果分类器$$g(\cdot)$$是 Logistic 回归分类器或 Softmax 回归分类器，那么$$g(⋅)$$也可以看成是网络的最后一层，即神经网络直接输出不同类别的条件概率。对于二分类问题$$y\in \{0,1\}$$，Logistic 回归分类器可以看成神经网络的最后一层，只有一个神经元，并且其激活函数为 Logistic 函数. 网络的输出可以直接作为类别$$y = 1$$的条件概率，即
$$
p(y = 1|\pmb x) = a (L)
$$
其中$$a^{(L)} ∈ \R$$为第$$L$$层神经元的活性值。

对于多分类问题$$y ∈ {1, ⋯ , C}$$，如果使用 Softmax 回归分类器，相当于网络最后一层设置$$C$$个神经元，其激活函数为 Softmax 函数。网络最后一层(第$$L$$层)的输出可以作为每个类的条件概率。



## 参数学习

如果采用交叉熵损失函数，对于样本$$(\pmb x,\pmb y)$$，其损失函数为
$$
\mathcal{L}(\pmb y,\hat{\pmb y})=-\pmb y^{\rm T}\log \hat{\pmb y}
$$
其中$$\pmb y\in \{0,1\}^C$$是标签$$y$$对应的 one-hot 向量表示。

给定训练集$$\mathcal{D}=\{(\pmb x^{(n)},\pmb y^{(n)})\}_{n=1}^N$$，将每个样本$$\pmb x^{(n)}$$输入给前馈神经网络，得到输出$$\hat{\pmb y}^{(n)}$$，其在数据集$$\mathcal{D}$$上的结构化风险函数为
$$
\mathcal{R}(\pmb W,\pmb b)=\frac{1}{N}\sum_{n=1}^N\mathcal{L}(\pmb y^{(n)},\hat{\pmb y}^{(n)})+\frac{1}{2}\lambda||\pmb W||^2_F
$$
其中$$\pmb W$$和$$\pmb b$$分别表示网络中的权重矩阵和偏置向量，$$||\pmb W||^2_F$$是正则化项，一般使用Frobenius范数
$$
||\pmb W||^2_F=\sum_{l=1}^L\sum_{i=1}^{M_l}\sum_{j=1}^{M_{l-1}}(w_{ij}^{(l)})^2
$$

> 参考https://zh.wikipedia.org/wiki/%E7%9F%A9%E9%99%A3%E7%AF%84%E6%95%B8
>
> 矩阵的Frobenius范数，也称为矩阵元-2范数，定义为
> $$
> ||A||_F=\sqrt{\sum^m_{i=1}\sum^n_{j=1}|a_{ij}|^2}
> $$

有了训练集和学习准则，网络参数可以通过梯度下降法进行学习。在梯度下降法的每次迭代中，第$$l$$层的参数$$\pmb W^{(l)}$$和$$\pmb b^{(l)}$$参数的更新方式为
$$
\pmb W^{(l)} \leftarrow \pmb W^{(l)}-\alpha\frac{\partial\mathcal{R}(\pmb W,\pmb b)}{\partial \pmb W^{(l)}}\\
=\pmb W^{(l)}-\alpha(\frac{1}{N}\sum_{n=1}^N(\frac{\partial \mathcal{L}(\pmb y^{(n)},\hat{\pmb y}^{(n)})}{\partial \pmb W^{(l)}})+\lambda \pmb W^{(l)})\\
\pmb b^{(l)} \leftarrow \pmb b^{(l)}-\alpha\frac{\partial\mathcal{R}(\pmb W,\pmb b)}{\partial \pmb b^{(l)}}\\
=\pmb b^{(l)}-\alpha(\frac{1}{N}\sum_{n=1}^N\frac{\partial \mathcal{L}(\pmb y^{(n)},\hat{\pmb y}^{(n)})}{\partial \pmb b^{(l)}})\\
$$
其中$$\alpha$$为学习率。

梯度下降法需要计算损失函数对参数的偏导数，然而使用链式法则求偏导比较低效，在神经网络的训练中一般使用反向传播算法来高效计算梯度。





# 反向传播算法





# 优化问题