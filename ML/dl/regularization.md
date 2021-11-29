**正则化(regularization)**是一类通过限制模型复杂度，从而避免过拟合，提高泛化能力的方法，比如引入约束、增加先验、提前停止等。

在传统的机器学习中，提高泛化能力的方法主要是限制模型复杂度，比如采用 $\ell_1$ 和 $\ell_2$ 正则化等方式。而在训练深度神经网络时，特别是在**过度参数化(over-parameterization)**时， $\ell_1$ 和 $\ell_2$ 正则化的效果往往不如浅层机器学习模型中显著。因此训练深度学习模型时，往往还会使用其他的正则化方法，比如数据增强、 提前停止、 丢弃法、 集成法等。

> 过度参数化是指模型参数的数量远远大于训练数据的数量。



## $\ell_1$ 和 $\ell_2$ 正则化

$\ell_1$ 和 $\ell_2$ 正则化是机器学习中最常用的正则化方法，通过约束参数的 $\ell_1$ 和 $\ell_2$ 范数来减小模型在训练数据集上的过拟合现象。

通过加入 $\ell_1$ 和 $\ell_2$ 正则化，优化问题可以写为
$$
\pmb\theta^*=\arg \min_{\pmb \theta} \frac{1}{N}\sum_{i=1}^N(\mathcal{L}(y^{(i)},f(\pmb x^{(i)};\pmb \theta))+\lambda\ell_p(\pmb \theta))
$$
其中 $\mathcal{L}(\cdot)$ 为损失函数， $N$ 为训练样本数量， $f(\cdot)$ 为待学习的神经网络， $\pmb \theta$ 为其参数， $\ell_p$ 为范数函数， $p$ 通常取1或2代表 $\ell_1$ 和 $\ell_2$ 范数， $\lambda$ 为正则化系数。

带正则化的优化问题（有条件地）等价于下面带约束条件的优化问题，其中 $c>0$ 
$$
\pmb\theta^*=\arg \min_{\pmb \theta} \frac{1}{N}\sum_{i=1}^N\mathcal{L}(y^{(i)},f(\pmb x^{(i)};\pmb \theta))\\
{\rm s.t.}\quad \ell_p(\pmb\theta)\le c
$$
> 参考Convex Optimization (S. Boyd) Chap. 5 Duality

$\ell_1$ 范数在零点不可导，因此经常用下式来近似：
$$
\ell_1(\pmb\theta)=\sum_{i=1}^D\sqrt{\theta_i^2+\varepsilon}
$$
其中 $\varepsilon$ 为非常小的正常数。

下图给出了不同范数约束条件下的最优化问题示例，可以看到， $\ell_1$ 正则化项通常会使得最优解位于坐标轴上，从而使得最终的参数为稀疏性向量。

![](https://img2018.cnblogs.com/blog/71977/202001/71977-20200101161322601-524903143.png)

![](https://img2018.cnblogs.com/blog/71977/202001/71977-20200101161323095-2101121814.png)

> $\ell_1$ 正则化的稀疏特性起到了特征选择的作用。

> $\ell_1$ 和 $\ell_2$ 正则化为什么能够防止过拟合？一种思路是， $\ell_1$ 和 $\ell_2$ 正则化惩罚参数取较大值，得到的模型的参数都比较小（ $\ell_1$ 正则化使得很多参数归零），这样的模型更简单。更简单的模型更不容易发生过拟合。
>
> ![](https://i.loli.net/2020/09/03/WZLiyxq1zEue6K2.png)

一种折中的正则化方法是同时加入 $\ell_1$ 和 $\ell_2$ 正则化项，称为**弹性网络正则化(elastic net regularization)** [Zou et al., 2005]。
$$
\pmb\theta^*=\arg \min_{\pmb \theta} \frac{1}{N}\sum_{i=1}^N\mathcal{L}(y^{(i)},f(\pmb x^{(i)};\pmb \theta))+\lambda_1\ell_1(\pmb \theta)+\lambda_2\ell_2(\pmb \theta)
$$
其中 $λ_1$ 和 $λ_2$ 分别为两个正则化项的系数。



## 权重衰减

**权重衰减(weight decay)**是一种有效的正则化方法[Hanson et al., 1989]，在每次参数更新时，引入一个衰减系数
$$
\pmb\theta_t \leftarrow (1-\beta)\pmb\theta_{t-1}-\alpha\pmb g_t
$$
其中 $\pmb g_t$ 是第 $t$ 步更新时的梯度， $\alpha$ 为学习率， $\beta$ 为权重衰减系数，一般取值比较小，比如 0.0005。在标准的随机梯度下降中，权重衰减正则化和 $\ell_2$ 正则化的效果相同。因此，权重衰减在一些深度学习框架中通过 $\ell_2$ 正则化来实现。但是，在较为复杂的优化方法(比如Adam)中，权重衰减正则化和 $\ell_2$ 正则化并不等价[Loshchilov et al., 2017b]。

> 回顾梯度下降法的参数迭代
> $$
> \pmb \theta_{t+1}=\pmb \theta_t-\alpha\frac{\partial \mathcal{R}_\mathcal{D}(\pmb \theta)}{\partial \pmb \theta}\bigg|_{\pmb\theta = \pmb \theta_t} \\
> =\pmb \theta_t-\alpha(\frac{1}{N}\sum_{i=1}^{N}\frac{\partial \mathcal{L}(y^{(i)},f(\pmb x^{(i)};\pmb \theta))}{\partial \pmb \theta}\bigg|_{\pmb\theta = \pmb \theta_t}+\lambda\pmb \theta_t)
> $$



## 提前停止

**提前停止(early stop)**对于深度神经网络来说是一种简单有效的正则化方法。由于深度神经网络的拟合能力非常强，因此比较容易在训练集上过拟合。在使用梯度下降法进行优化时，我们可以使用一个和训练集独立的样本集合，称为**验证集(validation set)**，并用验证集上的错误来代替期望错误。当验证集上的错误率不再下降，就停止迭代。

然而在实际操作中，验证集上的错误率变化曲线并不一定是如图所示的平衡曲线，很可能是先升高再降低。因此，提前停止的具体停止标准需要根据实际任务进行优化 [Prechelt, 1998]。

![](https://i.loli.net/2020/09/04/QTfOXJ9toj6lNFR.png)



## 丢弃法

> 参考：
>
> Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I. and Salakhutdinov, R., 2014. Dropout: a simple way to prevent neural networks from overfitting. *The journal of machine learning research*, *15*(1), pp.1929-1958.
>
> [理解dropout](https://blog.csdn.net/stdcoutzyx/article/details/49022443)

当训练一个深度神经网络时，我们可以随机丢弃一部分神经元（同时丢弃其对应的连接边）来避免过拟合，这种方法称为**丢弃法(dropout method)** [Srivastava et al., 2014]。每次选择丢弃的神经元是随机的。最简单的方法是设置一个固定的概率 $p$，对每一个神经元都以概率 $p$ 来判定要不要保留。对于一个神经层 $\pmb y = f(W\pmb x +\pmb b)$，我们可以引入一个<u>掩蔽函数</u> ${\rm mask}(⋅)$ 使得 $\pmb y =f(W{\rm mask}(\pmb x) +\pmb b)$。掩蔽函数 ${\rm mask(⋅)}$ 的定义为
$$
{\rm mask}(\pmb x)=\begin{cases}\pmb m\odot \pmb x\quad 训练阶段\\
p\pmb x\quad\quad\ \ \ 测试阶段

\end{cases}
$$
其中 $\pmb m ∈\{0, 1\}^D$ 是**丢弃掩码(dropout mask)**，通过以概率为 $p$ 的伯努利分布随机生成。在训练时，激活神经元的平均数量为原来的 $p$ 倍。而<u>在测试时，所有的神经元都是可以激活的</u>，这会造成训练和测试时网络的输出不一致。为了缓解这个问题，在测试时需要将神经层的输入 $\pmb x$ 乘以 $p$，也相当于把不同的神经网络做了平均。保留率 $p$ 可以通过验证集来选取一个最优的值，一般来讲，对于隐藏层的神经元，保留率 $p = 0.5$ 时效果最好，这对大部分的网络和任务都比较有效。当 $p = 0.5$ 时，在训练时有一半的神经元被丢弃，只剩余一半的神经元是可以激活的，随机生成的网络结构最具多样性。对于输入层的神经元，其保留率通常设为更接近 1 的数，使得输入变化不会太大。对输入层神经元进行丢弃时，相当于给数据增加噪声，以此来提高网络的鲁棒性。

丢弃法一般是针对神经元进行随机丢弃，但是也可以扩展到对神经元之间的连接进行随机丢弃 [Wan et al., 2013] ，或每一层进行随机丢弃。下图给出了一个网络应用丢弃法后的示例。

![](https://i.loli.net/2020/11/10/G3AQmitulyNxqfK.png)

**集成学习角度的解释**

每做一次丢弃，相当于从原始的网络中采样得到一个子网络。如果一个神经网络有 $n$ 个神经元，那么总共可以采样出 $2^n$ 个子网络。每次迭代都相当于训练一个不同的子网络，这些子网络都共享原始网络的参数，那么最终的网络可以近似看作集成了指数级个不同网络的组合模型。

**贝叶斯学习角度的解释**

丢弃法也可以解释为一种贝叶斯学习的近似 [Gal et al., 2016a] 。用 $y = f(\pmb x;\pmb θ)$ 来表示要学习的神经网络，贝叶斯学习是假设参数 $\pmb θ$ 为随机向量，并且先验分布为 $q(\pmb θ)$，贝叶斯方法的预测为
$$
E_{q(\pmb \theta)}(y)=\int_q f(\pmb x;\pmb \theta)q(\pmb \theta){\rm d}\theta\\
\approx \frac{1}{M}\sum_{m=1}^M f(\pmb x,\pmb\theta_m)
$$
其中 $f(\pmb x,\pmb \theta_m)$ 为第 $m$ 次应用丢弃方法后的网络，其参数 $\pmb\theta_m$ 为对全部参数 $\pmb θ$ 的一次采样。



### 循环神经网络上的丢弃法

当在循环神经网络上应用丢弃法时，不能直接对每个时刻的隐状态进行随机丢弃，这样会损害循环网络在时间维度上的记忆能力。一种简单的方法是对非时间维度的连接（即非循环连接）进行随机丢失 [Zaremba et al., 2014] 。如下图所示，虚线边表示进行随机丢弃，不同的颜色表示不同的丢弃掩码。

![](https://i.loli.net/2020/12/03/i8dk4QPnZyVI3Hf.png)

然而根据贝叶斯学习的解释，丢弃法是一种对参数 $\pmb θ$ 的采样，每次采样的参数需要在每个时刻保持不变。因此，在对循环神经网络上使用丢弃法时，需要对参数矩阵的每个元素进行随机丢弃，并在所有时刻都使用相同的丢弃掩码。这种方法称为**变分丢弃法(variational dropout)** [Gal et al., 2016b] 。 下图给出了变分丢弃法的示例，相同颜色表示使用相同的丢弃掩码。

![](https://i.loli.net/2020/12/03/NX3hqRQnEdHDMpy.png)



## 数据增强

深度神经网络一般都需要大量的训练数据才能获得比较理想的效果。在数据量有限的情况下，可以通过**数据增强(data augmentation)**来增加数据量，提高模型鲁棒性，避免过拟合。目前，数据增强还<u>主要应用在图像数据</u>上，在文本等其他类型的数据上还没有太好的方法。

图像数据的增强主要是通过算法对图像进行转变，引入噪声等方法来增加数据的多样性。增强的方法主要有几种：

1. 旋转(rotation)：将图像按顺时针或逆时针方向随机旋转一定角度
2. 翻转(flip)：将图像沿水平或垂直方法随机翻转一定角度
3. 缩放(zoom in/out)：将图像放大或缩小一定比例
4. 平移(shift)：将图像沿水平或垂直方法平移一定步长
5. 加噪声(noise)：加入随机噪声



## 标签平滑

> 参考：
>
> Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J. and Wojna, Z., 2016. Rethinking the inception architecture for computer vision. In *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 2818-2826).
>
> Müller, R., Kornblith, S. and Hinton, G.E., 2019. When does label smoothing help?. In *Advances in Neural Information Processing Systems* (pp. 4694-4703).

在数据增强中，我们可以给样本特征加入随机噪声来避免过拟合。同样，我们也可以给样本的标签引入一定的噪声。假设训练数据集中有一些样本的标签是被错误标注的，那么最小化这些样本上的损失函数会导致过拟合。一种改善的正则化方法是**标签平滑(label smoothing)**，即在输出标签中添加噪声来避免模型过拟合 [Szegedy et al., 2016] 。

一个样本 $\pmb x$ 的标签可以用 one-hot 向量表示，即 $\pmb y=[0,\cdots,0,1,0,\cdots,0]^{\rm T}$，这种标签可以看作**硬目标(hard target)**。如果使用Softmax分类器并使用交叉熵损失函数，最小化损失函数会使得正确类和其他类的权重差异变得很大。根据Softmax 函数的性质可知，如果要使得某一类的输出概率接近于 1 ，其未归一化的得分需要远大于其他类的得分，可能会导致其权重越来越大，并导致过拟合。此外，如果样本标签是错误的，会导致更严重的过拟合现象。为了改善这种情况，我们可以引入一个噪声对标签进行平滑，即假设样本以 $ε$ 的概率为其他类。平滑后的标签为
$$
\tilde{\pmb y}=[\frac{\varepsilon}{K-1} ,\cdots,\frac{\varepsilon}{K-1},1-\varepsilon,\frac{\varepsilon}{K-1},\cdots,\frac{\varepsilon}{K-1}]^{\rm T}
$$
其中 $K$ 为标签数量，这种标签可以看作软目标(soft target)。标签平滑可以避免模型的输出过拟合到硬目标上，并且通常不会损害其分类能力。上面的标签平滑方法是给其他 $K − 1$ 个标签相同的概率 $ε/(K−1)$，没有考虑标签
之间的相关性。一种更好的做法是按照类别相关性来赋予其他标签不同的概率，比如先训练另外一个更复杂（一般为多个网络的集成）的**教师网络(teacher network)**，并使用大网络的输出作为软目标来训练学生网络(student network)。这种方法也称为**知识蒸馏(knowledge distillation)** [Hinton et al., 2015] 。

