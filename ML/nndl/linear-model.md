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

![Screenshot from 2020-09-15 18-54-03.png](https://i.loli.net/2020/09/15/NSXC3Q5TjELZ9gn.png)





# 线性判别函数和决策边界

一个**线性分类模型(linear classification model)**或**线性分类器(linear classifier)**，是由一个（或多个）线性的判别函数$$f(\pmb x;\pmb w)=\pmb w^{\rm T}\pmb x+b$$和非线性的决策函数$$g(⋅)$$组成。

## 二分类

**二分类(binary classification)**问题的类别标签$$y$$只有两种取值，通常可以设为$$\{+1, −1\}$$或$$\{0, 1\}$$。在二分类问题中，常用**正例(positive sample)**和**负例(negative sample)**来分别表示属于类别 +1 和 −1 的样本。

在二分类问题中，我们只需要一个线性判别函数$$f(\pmb x;\pmb w)=\pmb w^{\rm T}\pmb x+b$$。 **特征空间**$$\R^D$$中所有满足$$f(\pmb x;\pmb w)=0$$的点组成一个分割**超平面(hyperplane)**，称为**决策边界(decision boundary)**或**决策平面(decision surface)**。决策边界将特征空间一分为二，划分成两个区域，每个区域对应一个类别。

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
\mathcal{L}(y,\hat y)=I(yf(\pmb x;\pmb w)>0)
$$
其中$$I(\cdot)$$为指示函数。但 0-1 损失函数的数学性质不好，其关于$$\pmb w$$的导数为0，因而无法使用梯度下降法。



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

![Screenshot from 2020-09-17 10-21-16.png](https://i.loli.net/2020/09/17/lWAS7GHzIQEePBn.png)

> 按照“一对其余”方式的定义，图(a)应该是：$$f_1$$以上部分为$$w_1$$，$$f_1$$以下$$f_2$$以上为$$w_2$$，剩余部分的$$f_2$$以下$$f_3$$以上为$$w_3$$，剩余部分为?。图(a)实际是按照$${\rm sgn}(f_1),{\rm sgn}(f_2),{\rm sgn}(f_3)$$的组合划分（共$$\frac{(1+C)C}{2}+1$$个）区域。

**定义** 如果存在$$C$$个权重向量$$\pmb w_1^*,\pmb w_2^*,\cdots,\pmb w_C^*$$，使得第$$c$$类的所有样本都满足$$f_c(\pmb x;\pmb w_c^*) >
f_{\tilde c}(\pmb x;\pmb w_{\tilde c}^*),\forall\tilde c\neq c$$，则称训练集$$\mathcal{D}$$是**（多类）线性可分**的。

由以上定义可知，如果数据集是多类线性可分的，那么一定存在一个“argmax”方式的线性分类器可以将它们正确分开。





# Logistic回归

**Logistic 回归(Logistic Regression , LR)**是一种常用的处理二分类问题的线性模型。在本节中，我们采用$$y ∈ \{0, 1\}$$以符合 Logistic 回归的描述习惯。

这里引入非线性函数$$g:\R\to (0,1)$$来预测类别标签的后验概率$$\hat p(y=1|\pmb x)$$，即
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



# Softmax回归



# 感知器





# 支持向量机

**支持向量机(Support Vector Machine , SVM)**是一个经典的二分类算法，其找到的分割超平面具有更好的鲁棒性，因此广泛使用在很多任务上，并表现出了很强优势。

