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

在二分类问题中，我们只需要一个线性判别函数$$f(\pmb x;\pmb w)=\pmb w^{\rm T}\pmb x+b$$。 特征空间$$\R^D$$中所有满足$$f(\pmb x;\pmb w)=0$$的点组成一个分割**超平面(hyperplane)**，称为**决策边界(decision boundary)**或**决策平面(decision surface)**。决策边界将特征空间一分为二，划分成两个区域，每个区域对应一个类别。

> 超平面就是三维空间中的平面在$$D$$维空间的推广。$$D$$维空间中的超平面是$$D − 1$$维的。