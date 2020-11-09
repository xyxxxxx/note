**线性模型(linear model)**是机器学习中应用最广泛的模型，指通过样本特征的线性组合来进行预测的模型。给定一个$$D$$维样本$$\pmb x=[x_1,\cdots,x_D]^{\rm T}$$，其线性组合函数为
$$
f(\pmb x;\pmb w)=\pmb w^{\rm T}\pmb x+b
$$
其中$$\pmb w=[w_1,\cdots,w_D]^{\rm T}$$为$$D$$维的权重向量，$$b$$为偏置。线性回归即是典型的线性模型，直接使用**判别函数(discriminant function)**$$f(\pmb x;\pmb w)$$来预测输出目标$$y=f(\pmb x;\pmb w)$$。

在分类问题中，由于输出目标$$y$$是一些离散的标签，而$$f(\pmb x;\pmb w)$$的值域为实数，因此无法直接用$$f(\pmb x;\pmb w)$$来进行预测，需要引入一个非线性的**决策函数(decision function)**$$g(⋅)$$来预测输出目标
$$
\hat y=g(f(\pmb x;\pmb w))
$$
对于二分类问题，$$g(\cdot)$$可以是**符号函数(sign function)**，定义为
$$
g(f(\pmb x;\pmb w))={\rm sgn}(f(\pmb x;\pmb w))=\begin{cases}1,&f(\pmb x;\pmb w)>0\\
-1,&f(\pmb x;\pmb w)<0
\end{cases}
$$
当$$f(\pmb x;\pmb w)=0$$时无定义。由此得到的线性模型如下图所示

![](https://i.loli.net/2020/09/15/NSXC3Q5TjELZ9gn.png)





# 线性判别函数和决策边界

一个**线性分类模型(linear classification model)**或**线性分类器(linear classifier)**，是由一个（或多个）线性的判别函数$$f(\pmb x;\pmb w)=\pmb w^{\rm T}\pmb x+b$$和非线性的决策函数$$g(⋅)$$组成。

## 二分类

**二分类(binary classification)**问题的类别标签$$y$$只有两种取值，通常可以设为$$\{+1, −1\}$$或$$\{0, 1\}$$。在二分类问题中，常用**正例(positive sample)**和**负例(negative sample)**来分别表示属于类别 +1 和 −1 的样本。

在二分类问题中，我们只需要一个线性判别函数$$f(\pmb x;\pmb w)=\pmb w^{\rm T}\pmb x+b$$。 **特征空间**$$\mathbb{R}^D$$中所有满足$$f(\pmb x;\pmb w)=0$$的点组成一个分割**超平面(hyperplane)**，称为**决策边界(decision boundary)**或**决策平面(decision surface)**。决策边界将特征空间一分为二，划分成两个区域，每个区域对应一个类别。

> 超平面就是三维空间中的平面在$$D$$维空间的推广。$$D$$维空间中的超平面是$$D − 1$$维的。

在特征空间中，决策平面与权重向量$$\pmb w$$正交；每个样本点到决策平面的**有向距离(signed distance)**为
$$
\gamma = \frac{f(\pmb x;\pmb w)}{||\pmb w||}
$$

> 设想解析几何的立体坐标系中，$$Ax+By+Cz+D=0(A,B,C不全为0)$$表示一个平面，向量$$(A,B,C)$$与该平面正交；任意一点$$(x_0,y_0,z_0)$$到该平面的有向距离为
> $$
> D=\frac{Ax_0+By_0+Cz_0+D}{\sqrt{A^2+B^2+C^2}}
> $$
> $$Ax+By+Cz+D$$可以视作$$f(\pmb x;\pmb w)$$的一个实例。



给定$$N$$个样本的训练集$$\mathcal{D} = \{(\pmb x^{(n)}, y^{(n)})\}_{n=1}^N$$，其中$$y^{(n)} ∈ \{+1, −1\}$$，线性模型试图学习到参数 $$\pmb w^∗$$，使得对于每个样本$$ (\pmb x^{(n)},y^{(n)})$$尽量满足
$$
y^{(n)}f(\pmb x^{(n)};\pmb w^*)>0,\forall n\in [1,N]
$$

**定义** 如果存在权重向量$$\pmb w^*$$，使得上式对所有$$n$$满足，则称训练集$$\mathcal{D}$$是**线性可分**的。

为了学习参数$$\pmb w$$，我们需要定义合适的损失函数以及优化方法。对于二分类问题，最直接的损失函数为 0-1 损失函数，即
$$
\mathcal{L}(y,f(\pmb x;\pmb w))=I(yf(\pmb x;\pmb w)<0)
$$
其中$$I(\cdot)$$为指示函数。但 0-1 损失函数的数学性质不好，其关于$$\pmb w$$的导数为0，因而无法使用梯度下降法。

![Screenshot from 2020-10-27 21-52-01.png](https://i.loli.net/2020/10/27/Wo7N4GanOXms9BI.png)



## 多分类

**多分类(multi-class classification)**问题是指分类的类别数$$C$$大于 2 。多分类一般需要多个线性判别函数，但设计这些判别函数有很多种方式。

假设一个多分类问题的类别为$$\{1, 2, ⋯ , C\}$$，常用的方式有以下三种：

1. “一对其余”方式：把多分类问题转换为$$C$$个二分类问题，这种方式共需要$$C$$个判别函数，其中第$$c$$个判别函数$$f_c$$是将类别$$c$$的样本和不属于类别$$c$$的样本分开。

2. “一对一”方式：把多分类问题转换为$$C(C − 1)/2$$个 “一对一” 的二分类问题，这种方式共需要$$C(C − 1)/2$$个判别函数，其中第$$(i, j)$$个判别函数是把类别$$i$$和类别$$j$$的样本分开。

3. “argmax”方式：这是一种改进的“一对其余”方式，共需要$$C$$个判别函数
   $$
   f_c(\pmb x;\pmb w_c)=\pmb w_c^{\rm T} \pmb x+b_c,\quad c\in \{1,2,\cdots,C\}
   $$
   对于样本$$\pmb x$$，如果存在一个类别$$c$$，相对于所有的其他类别$$\tilde c(\tilde c ≠ c)$$有$$f_c(\pmb x;\pmb w_c ) >
   f_{\tilde c}(\pmb x;\pmb w_{\tilde c} )$$，那么$$\pmb x$$属于类别$$c$$。“argmax” 方式的预测函数定义为
   $$
   y=\arg \max_{c=1}^C f_c(\pmb x;\pmb w_c)
   $$

“一对其余”方式和“一对一”方式都存在一个缺陷：特征空间中会存在一些难以确定类别的区域，而“ argmax ”方式很好地解决了这个问题。下图给出了用这三种方式进行多分类的示例，其中红色直线表示判别函数$$f(⋅) = 0$$的直线，不同颜色的区域表示预测的三个类别的区域($$ω_1 , ω_2$$和$$ω_3$$)和难以确定类别的区域(‘?’)。在“argmax”方式中，相邻两类$$i$$和$$j$$的决策边界实际上是由$$f_i(\pmb x;\pmb w_i) − f_j(\pmb x;\pmb w_j) = 0$$决定, 其法向量为$$\pmb w_i −\pmb w_j$$。

![](https://i.loli.net/2020/09/17/lWAS7GHzIQEePBn.png)

> 按照“一对其余”方式的定义，图(a)应该是：$$f_1$$以上部分为$$w_1$$，$$f_1$$以下$$f_2$$以上为$$w_2$$，剩余部分的$$f_2$$以下$$f_3$$以上为$$w_3$$，剩余部分为?。图(a)实际是按照$${\rm sgn}(f_1),{\rm sgn}(f_2),{\rm sgn}(f_3)$$的组合划分（共$$\frac{(1+C)C}{2}+1$$个）区域。

**定义** 如果存在$$C$$个权重向量$$\pmb w_1^*,\pmb w_2^*,\cdots,\pmb w_C^*$$，使得第$$c$$类的所有样本都满足$$f_c(\pmb x;\pmb w_c^*) >
f_{\tilde c}(\pmb x;\pmb w_{\tilde c}^*),\forall\tilde c\neq c$$，则称训练集$$\mathcal{D}$$是**多类线性可分**的。

由以上定义可知，如果训练集多类线性可分的，那么一定存在一个“argmax”方式的线性分类器可以将它们正确分开。





# Logistic回归

**Logistic 回归(Logistic Regression , LR)**是一种常用的处理二分类问题的线性模型。在本节中，我们采用$$y ∈ \{0, 1\}$$以符合 Logistic 回归的描述习惯。

这里引入非线性函数$$g:\mathbb{R}\to (0,1)$$来预测类别标签的后验概率$$\hat p(y=1|\pmb x)$$，即
$$
\hat p(y=1|\pmb x)=g(f(\pmb x;\pmb w))
$$
其中$$g(⋅)$$通常称为**激活函数(activation function)**，其作用是把线性函数的值域从实数区间 “挤压” 到了$$(0, 1)$$之间, 可以用来表示概率。

在 Logistic 回归中，我们使用 Logistic 函数来作为激活函数。标签$$y = 1$$的后验概率为
$$
\hat p(y=1|\pmb x)=\sigma(\pmb w^{\rm T}\pmb x)\\
\triangleq\frac{1}{1+\exp(-\pmb w^{\rm T}\pmb x)}
$$
为简单起见，这里$$\pmb x=[x_1,\cdots,x_D,1]^{\rm T},\pmb w=[w_1,\cdots,w_D,b]^{\rm T}$$分别为$$D+1$$维的**増广特征向量**和**増广权重向量**。

将上式进行变换得到
$$
\pmb w^{\rm T}\pmb x=\ln \frac{\hat p(y=1|\pmb x)}{\hat p(y=0|\pmb x)}
$$
其中$$\hat p(y=1|\pmb x)/\hat p(y=0|\pmb x)$$为样本$$\pmb x$$是正反例后验概率的比值，称为**几率(odds)**，几率的对数称为**对数几率(log odds, 或logit)**。上式等号左边是线性函数，因此Logistic回归可以看作预测值为<u>标签的对数几率</u>的线性回归模型。因此Logistic回归也称为**对数几率回归(logit regression)**。

下图

![Screenshot from 2020-10-28 10-35-00.png](https://i.loli.net/2020/10/28/GybApwse1ZChBYI.png)



## 参数学习

Logistic回归采用交叉熵作为损失函数，并使用梯度下降法对参数进行优化。

给定N个训练样本$$\{\pmb x^{(n)},y^{(n)} \}^N_{n=1}$$，用Logistic回归模型对每个样本$$\pmb x^{(n)}$$进行预测，输出其标签为1的后验概率，记作$$\hat y^{(n)}$$，
$$
\hat y^{(n)}=\sigma(\pmb w^{\rm T}\pmb x^{(n)}),\ 1\le n\le N
$$
由于$$y^{(n)}\in \{0,1\}$$，样本$$(\pmb x^{(n)},y^{(n)})$$的真实条件概率可以表示为
$$
p(y^{(n)}=1|\pmb x^{(n)})=y^{(n)}\\
p(y^{(n)}=0|\pmb x^{(n)})=1-y^{(n)}
$$
使用交叉熵损失函数，其风险函数为
$$
\mathcal{R}(\pmb w)=-\frac{1}{N}\sum_{n=1}^N(y^{(n)}\log \hat y^{(n)}+(1-y^{(n)})\log (1-\hat y^{(n)}))
$$

> 这里没有引入正则化项。

关于参数$$\pmb w$$的偏导数为
$$
\frac{\partial \mathcal{R}(\pmb w)}{\partial \pmb w}=-\frac{1}{N}\sum_{n=1}^N\pmb x^{(n)}(y^{(n)}-\hat y^{(n)})
$$
采用梯度下降法，Logistic回归的训练过程为：初始化$$\pmb w_0←0$$，然后通过下式迭代更新参数：
$$
\pmb w_{t+1}←\pmb w_t+\alpha\frac{1}{N}\sum_{n=1}^N\pmb x^{(n)}(y^{(n)}-\hat y_{\pmb w_t}^{(n)})
$$
其中$$\alpha$$是学习率，$$\hat y_{\pmb w_t}^{(n)}$$是当参数为$$\pmb w_t$$时，Logistic回归模型的输出。

此外，风险函数$$\mathcal{R}(\pmb w)$$是关于参数$$\pmb w$$的连续可导的凸函数，因此Logistic回归还可以使用凸优化中的高阶优化方法（如牛顿法）进行优化。





# Softmax回归

**Softmax回归(Softmax regression)**，也称为多项(multinomial)或多分类(multi-class)的Logistic回归，是Logistic回归在多分类问题上的推广。

对于多分类问题，类别标签$$y\in \{1,2,\cdots,C\}$$可以有$$C$$个取值。给定一个样本$$\pmb x$$，Softmax回归预测的属于类别c的后验概率为
$$
\hat p(y=c|\pmb x)={\rm softmax}(\pmb w_c^{\rm T}\pmb x)\\
=\frac{\exp(\pmb w_c^{\rm T}\pmb x)}{\sum_{i=1}^C \exp(\pmb w_i^{\rm T}\pmb x)}
$$
其中$$\pmb w_c$$是第c类的权重向量。

上式用向量形式可以写为
$$
\hat{\pmb y}={\rm softmax}(W^{\rm T}\pmb x)\\
=\frac{\exp(W^{\rm T}\pmb x)}{\pmb 1_C^{\rm T}\exp(W^{\rm T}\pmb x)}
$$
其中$$W=[\pmb w_1,\cdots,\pmb w_C]$$是由C个类的权重向量组成的矩阵，$$\pmb 1_C$$是C维的全1向量，$$\hat{\pmb y}\in\mathbb{R}^C$$是所有类别的预测后验概率组成的向量。



## 参数学习

给定N个训练样本$$\{\pmb x^{(n)},y^{(n)} \}^N_{n=1}$$，Softmax回归模型使用交叉熵损失函数来学习最优的参数矩阵$$\pmb W$$。

为了方便起见，我们用C维的one-hot向量$$\pmb y$$来表示类别标签。对于类别c，其向量表示为
$$
\pmb y=[I(c=1),I(c=2),\cdots,I(C=c)]^{\rm T}
$$
其中$$I(\cdot)$$是指示函数。

使用交叉熵损失函数，其风险函数为
$$
\mathcal{R}(W)=-\frac{1}{N}\sum_{n=1}^N(\pmb y^{(n)})^{\rm T}\log \hat{\pmb y}^{(n)}
$$
其中$$\hat{\pmb y}^{(n)}={\rm softmax}(W^{\rm T}\pmb x^{(n)})$$为样本$$\pmb x^{(n)}$$在每个类别的后验概率。

> 这里没有引入正则化项。

关于$$W$$的偏导数为
$$
\frac{\partial \mathcal{R}(W)}{\partial W}=-\frac{1}{N}\sum_{n=1}^N\pmb x^{(n)}(\pmb y^{(n)}-\hat{\pmb y}^{(n)})^{\rm T}
$$
采用梯度下降法，Softmax回归的训练过程为：初始化$$W_0←0$$，然后通过下式迭代更新参数：
$$
W_{t+1}←W_t+\alpha\frac{1}{N}\sum_{n=1}^N\pmb x^{(n)}(\pmb y^{(n)}-\hat{\pmb y}_{W_t}^{(n)})^{\rm T}
$$
其中$$\alpha$$是学习率，$$\hat y_{W_t}^{(n)}$$是当参数为$$W_t$$时，Softmax回归模型的输出。





# 感知器

**感知器(perceptron)**由Frank Roseblatt于1957年提出，是一种广泛使用的线性分类器。感知器可以看作最简单的人工神经网络，只有一个神经元。

感知器是对生物神经元的简单数学模拟，包含了与生物神经元相对应的部件，如权重（突触）、偏置（阈值）和激活函数（细胞体），输出为1或-1。

感知器是一种简单的两类线性分类模型，其分类准则与最开始介绍的线性模型相同，即
$$
\hat{y}={\rm sgn}(\pmb w^{\rm T}\pmb x)
$$

![](https://i.loli.net/2020/09/15/NSXC3Q5TjELZ9gn.png)

![Screenshot from 2020-10-27 21-52-01.png](https://i.loli.net/2020/10/27/Wo7N4GanOXms9BI.png)



## 参数学习

感知器学习算法是一个经典的线性分类器的参数学习算法。

给定N个训练样本$$\{\pmb x^{(n)},y^{(n)} \}^N_{n=1}$$，其中$$y^{(n)}\in \{1,-1 \}$$，感知器学习算法试图找到一组参数$$\pmb w^*$$，使得对于每个样本$$(\pmb x^{(n)},y^{(n)})$$有
$$
y^{(n)}\pmb w^{*{\rm T}}\pmb x^{(n)}>0,\quad \forall n\in \{1,\cdots,N\}
$$
感知器的学习算法是一种错误驱动的在线学习算法[Rosenblatt, 1958]。首先初始化一个权重向量$$\pmb w_0←\pmb 0$$，然后每次分错一个样本$$(\pmb x^{(n)},y^{(n)})$$，即$$y^{(n)}\pmb w_k^{\rm T}\pmb x^{(n)}\le 0$$时，就用这个样本来更新权重
$$
\pmb w_{k+1}←\pmb w_k+y^{(n)}\pmb x^{(n)}
$$
> 在线学习算法按照顺序处理数据，每个数据点训练后直接更新权重，即更新模型，并把新模型投入实际操作中。随着更多的实时数据到达，模型会在操作中不断地更新。

###### 感知器的损失函数为

$$
\mathcal{L}(y,{\rm sgn}(\pmb w^{\rm T}\pmb x))=\max(0,-y\pmb w^{\rm T}\pmb x)
$$
关于参数$$\pmb w$$的偏导数为（采用随机梯度下降法）
$$
\frac{\partial \mathcal{L}(y,{\rm sgn}(\pmb w^{\rm T}\pmb x))}{\partial \pmb w}=\begin{cases}0,\quad\ \ y\pmb w^{\rm T}\pmb x>0,\\
-y\pmb x,\ y\pmb w^{\rm T}\pmb x<0,
\end{cases}
$$


@给定训练样本
$$
X=[\pmb x^{(1)},\cdots,\pmb x^{(10)}]=\begin{bmatrix}5 & 6 & 4 & 3 & 5 & 1 & 2 & 1 & 2 & 3\\
5 & 4 & 5 & 4 & 3 & 3 & 3 & 1 & 0 & 1\\
1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1
\end{bmatrix}\\
\pmb y=[y^{(1)},\cdots,y^{(5)},y^{(6)},\cdots,y^{(10)}]=\begin{bmatrix} 1&\cdots& 1& -1&\cdots& -1
\end{bmatrix}
$$
求感知器模型。



## 感知器的收敛性

Novikoff(1963)证明对于二分类问题，如果训练集是线性可分的，那么感知器算法可以在有限次迭代后收敛。然而，如果训练集不是线性可分的，那么这个算法则不能确保会收敛。

**定理（感知器收敛性）** 对于线性可分的训练集$$\mathcal{D}=\{\pmb x^{(n)},y^{(n)} \}^N_{n=1}$$，那么存在$$\gamma >0$$和权重向量$$\pmb w^*$$，并且$$\|\pmb w^*\|=1$$，对任意$$n$$满足
$$
y^{(n)}\pmb w^{*\rm T}\pmb x^{(n)}\ge \gamma
$$
再令$$R$$是训练集中最大的特征向量的模，即
$$
R=\max_n ||x^{(n)} ||
$$
那么二分类感知器的参数学习算法的权重更新次数不超过$$R^2/\gamma^2$$。



虽然感知器在线性可分的数据上可以保证收敛，但存在以下不足：

1. 在训练集线性可分时，感知器虽然可以找到一个超平面将两类数据分开，但并不能保证其泛化能力
2. 感知器对样本顺序比较敏感：对于不同的迭代顺序，找到的分割超平面也不同
3. 如果训练集不是线性可分的，则永远不会收敛



## 参数平均感知器

根据定理（感知器收敛性），如果训练数据是线性可分的，那么感知器可以找到一个判别函数来分割不同类的数据。但是<u>感知器并不能保证找到的判别函数是最优的</u>（比如泛化能力高），这样可能导致过拟合。

<u>感知器学习到的权重向量和训练样本的顺序相关</u>。在迭代次序上排在后面的错误样本比前面的错误样本对最终的权重向量影响更大。比如有 1000 个训练样本，在迭代 100 个样本后，感知器已经学习到一个很好的权重向量。在接下来的 899 个样本上都预测正确，也没有更新权重向量；但是在最后第 1000 个样本时预测错误，并更新了权重。这次更新可能反而使得权重向量变差。

为了提高感知器的鲁棒性和泛化能力，我们可以将在感知器学习过程中的所有（共$$K$$个）权重向量保存起来, 并赋予每个权重向量$$\pmb w_k$$一个置信系数$$c_k (1\le k\le K)$$，最终的分类结果由这$$K$$个不同权重的感知器投票决定，这个模型也称为**投票感知器(voted perceptron)**[Freund et al., 1999]。

令$$τ_k$$为第$$k$$次更新权重$$\pmb w_k$$时的迭代次数(即训练过的样本数量)，则权重$$\pmb w_k$$的置信系数$$c_k$$设置为从$$τ_k$$到$$τ_{k+1}$$之间间隔的迭代次数，即$$c_k = τ_{k+1} − τ_k$$。置信系数$$c_k$$越大，说明权重$$\pmb w_k$$在之后的训练过程中正确分类样本的数量越多，越值得信赖。这样，投票感知器的形式为
$$
\hat{y}={\rm sgn}(\sum_{k=1}^K c_k {\rm sgn}(\pmb w_k^{\rm T}\pmb x))
$$
投票感知器虽然提高了感知器的泛化能力，但是需要保存$$K$$个权重向量，在实际操作中会带来额外的开销。因此，人们经常会使用一个简化的版本，通过使用“参数平均”的策略来减少投票感知器的参数数量，也叫作**平均感知器(averaged perceptron)** [Collins, 2002]。平均感知器的形式为
$$
\hat{y}={\rm sgn}(\frac{1}{T} \sum_{k=1}^K c_k \pmb w_k^{\rm T}\pmb x)\\
={\rm sgn}(\overline{\pmb w}^{\rm T}\pmb x)
$$
其中$$\overline{\pmb w}$$为所有（共$$T$$次）迭代的平均权重向量。这个方法非常简单，只需要在每次迭代时都更新$$\overline{\pmb w}$$
$$
t\overline{\pmb w}\leftarrow (t-1)\overline{\pmb w}+\pmb w_t\\
$$
但这个方法需要在处理每一个样本时都要更新$$\overline{\pmb w}$$ ，由于$$\overline{\pmb w}$$和$$\pmb w_t$$都是稠密向量，更新操作会比较费时。为了提高迭代速度，有很多改进的方法，让这个更新只需要在错误预测发生时才进行更新。



## *扩展到多分类

原始的感知器是一种二分类模型，但也可以很容易地扩展到多分类问题，甚至是更一般的结构化学习问题 [Collins, 2002]。

之前介绍的分类模型中，分类函数都是在输入$$\pmb x$$的特征空间上。为了使得感知器可以处理更复杂的输出，我们引入一个构建在输入输出联合空间上的特征函数$$\phi(\pmb x,\pmb y)$$，将样本对$$(\pmb x,\pmb y)$$映射到一个特征向量空间。

在联合特征空间中，我们可以建立一个广义的感知器模型，
$$
\hat{\pmb y}=\arg \max\pmb w^{\rm T}\phi(\pmb x,\pmb y)
$$
广义感知器模型一般用来处理结构化学习问题。当用广义感知器模型来处理$$C$$分类问题时，$$\pmb y ∈ \{0, 1\}^C$$ 为类别的 one-hot 向量表示。在 C 分类问题中，一种常用的特征函数是$$\pmb x$$和$$\pmb y$$的外积，即
$$
\phi(\pmb x,\pmb y)={\rm vec}(\pmb x\pmb y^{\rm T})\\
=\begin{bmatrix}\vdots\\0\\(第(c-1)\times D+1行)\quad x_1\\\vdots\\(第(c-1)\times D+D行)\quad x_D\\0\\\vdots
\end{bmatrix}\in \mathbb{R}^{D\times C}
$$
其中$$\pmb x\in \mathbb{R}^D$$，$$\pmb y$$为第$$c$$维为1的 one-hot 向量。

广义感知器的学习算法是，首先初始化一个权重向量$$\pmb w_0←\pmb 0$$，然后每次预测错误一个样本$$(\pmb x^{(n)},\pmb y^{(n)})$$，即$$\hat{\pmb y}^{(n)}\neq \pmb y^{(n)}$$时，就用这个样本来更新权重
$$
\pmb w_{k+1}←\pmb w_k+(\phi(\pmb x^{(n)},\pmb y^{(n)})-\phi(\pmb x^{(n)},\hat{\pmb y}^{(n)})
$$


……





# 支持向量机

**支持向量机(Support Vector Machine , SVM)**是一个经典的二分类算法，其找到的分割超平面具有更好的鲁棒性，因此广泛使用在很多任务上，并表现出了很强优势。

给定线性可分的训练集$$\mathcal{D}=\{\pmb x^{(n)},y^{(n)} \}^N_{n=1}$$，那么存在一个超平面
$$
\pmb w^{\rm T}\pmb x+b=0
$$

> 本节不使用増广的特征向量和特征权重向量。

将两类样本分开，每个样本$$\pmb x^{(n)}$$到分割超平面的距离为
$$
\gamma^{(n)}=\frac{y^{(n)}(\pmb w^{\rm T}\pmb x^{(n)}+b)}{\|\pmb w\|}
$$
我们定义**间隔(margin)**$$\gamma$$为$$\mathcal{D}$$中所有样本到分割超平面的最短距离
$$
\gamma = \min_n \gamma^{(n)}
$$
如果间隔$$\gamma$$越大，那么分割超平面对两个数据集的划分越稳定，不容易受噪声等因素影响。SVM的目标是寻找一个超平面$$\pmb w^*$$使得$$\gamma $$最大，即
$$
\begin{align}
\max_{\pmb w^*} &\quad \gamma\\
{\rm s.t.} &\quad \frac{y^{(n)}(\pmb w^{\rm T}\pmb x^{(n)}+b)}{\|\pmb w\|}\ge \gamma,\ n=1,\cdots,N
\end{align}
$$
由于同时缩放$$\pmb w\to k\pmb w$$和$$b\to kb$$不改变样本$$\pmb x^{(n)}$$到分割超平面的距离，我们可以限制$$\|\pmb w\|\cdot\gamma=1$$，那么上式化为
$$
\begin{align}
\max_{\pmb w^*} &\quad \frac{1}{\|\pmb w\|} \\
{\rm s.t.} &\quad y^{(n)}(\pmb w^{\rm T}\pmb x^{(n)}+b)\ge 1,\ n=1,\cdots,N
\end{align}
$$
训练集中所有满足$$y^{(n)}(\pmb w^{\rm T}\pmb x^{(n)}+b)= 1$$的样本点称为**支持向量(support vector)**。

对于一个线性可分的数据集，其分割超平面有很多个，但是间隔最大的超平面是唯一的。如图给出了支持向量机的最大间隔分割超平面的示例，其中轮廓加粗的样本点为支持向量。

![Screenshot from 2020-10-28 17-01-01.png](https://i.loli.net/2020/10/28/7nxbqHRU5kPVXct.png)



## 参数学习

为了找到最大间隔分割超平面，将前面的优化问题改写为凸优化问题
$$
\begin{align}
\min_{\pmb w,b} &\quad \frac{1}{2} \|\pmb w\|^2 \\
{\rm s.t.} &\quad 1-y^{(n)}(\pmb w^{\rm T}\pmb x^{(n)}+b)\le 0,\ n=1,\cdots,N
\end{align}
$$
这里目标函数是二次的，约束函数是线性的，因此是一个凸二次规划问题。使用拉格朗日乘数法，目标函数的拉格朗日函数为
$$
Λ
$$
令$$Λ(\pmb w,b,\lambda)$$关于$$\pmb w$$和$$b$$的导数等于0，得到
$$
\pmb w=\sum_{n=1}^N\lambda_n y^{(n)}\pmb x^{(n)}\\
0=\sum_{n=1}^N\lambda_n y^{(n)}
$$
根据以上各式，得到拉格朗日对偶函数
$$
\Gamma(\pmb \lambda)=-\frac{1}{2}\sum_{n=1}^N\sum_{m=1}^N\lambda_m\lambda_ny^{(m)}y^{(n)}(\pmb x^{(m)})^{\rm T}\pmb x^{(n)}+\sum_{n=1}^N\lambda_n
$$


