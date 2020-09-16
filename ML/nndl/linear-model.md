**线性模型(linear model)**是机器学习中应用最广泛的模型，指通过样本特征的线性组合来进行预测的模型。给定一个$$D$$维样本$$\pmb x=[x_1,\cdots,x_D]^{\rm T}$$，其线性组合函数为
$$
f(\pmb x;\pmb w)=\pmb w^{\rm T}\pmb x+b
$$
其中$$\pmb w=[w_1,\cdots,w_D]^{\rm T}$$为$$D$$维的权重向量，$$b$$为偏置。线性回归即是典型的线性模型，直接使用**判别函数(discriminant function)**$$f(\pmb x;\pmb w)$$来预测输出目标$$y=f(\pmb x;\pmb w)$$。

在分类问题中，由于输出目标$$y$$是一些离散的标签，而$$f(\pmb x;\pmb w)$$的值域为实数，因此无法直接用$$f(\pmb x;\pmb w)$$来进行预测，需要引入一个非线性的**决策函数(decision function)**$$g(⋅)$$来预测输出目标
$$
y=g(f(\pmb x;\pmb w))
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

> 如果存在权重向量$$\pmb w^*$$，使得上式对所有$$n$$满足，则称训练集$$\mathcal{D}$$是线性可分的。

为了学习参数$$\pmb w$$，我们需要定义合适的损失函数以及优化方法。对于二分类问题，最直接的损失函数为 0-1 损失函数，即
$$
\mathcal{L}(y,\hat y)=I(yf(\pmb x;\pmb w)>0)
$$
其中$$I(\cdot)$$为指示函数。但 0-1 损失函数的数学性质不好，其关于$$\pmb w$$的导数为0，因而无法使用梯度下降法。



## 多分类

**多分类(multi-class classification)**问题是指分类的类别数$$C$$大于 2 。多分类一般需要多个线性判别函数，但设计这些判别函数有很多种方式。

