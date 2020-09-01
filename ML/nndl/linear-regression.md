# 模型

**线性回归(Linear Regression)**是机器学习和统计学中最基础和最广泛应用的模型，是一种对自变量和因变量之间关系进行建模的回归分析。自变量数量为1时称为**简单回归**，自变量数量大于1时称为**多元回归**。

从机器学习的角度来看，自变量就是样本的特征向量$$x ∈ \mathbb{R}^D$$(每一维对应一个自变量)，因变量是标签$$y$$，这里$$y ∈ \mathbb{R}$$是连续值(实数或连续整数)。假设空间是一组参数化的线性函数：
$$
f(\boldsymbol x; \boldsymbol w,b)=\boldsymbol w^{\rm T}\boldsymbol x+b \tag{1}
$$
其中权重向量$$\boldsymbol w \in \mathbb{R}^D$$和偏置$$b\in \mathbb{R}$$都是可学习的参数，函数$$f(\boldsymbol x; \boldsymbol w，b) \in \mathbb{R}$$也称为**线性模型**。

为简单起见，将公式(1)写为
$$
f(\boldsymbol x;\hat{\boldsymbol w})=\hat{\boldsymbol w}^{\rm T}\hat{\boldsymbol x} \tag{2}
$$
其中$$\hat{\boldsymbol w}$$和$$\hat{\boldsymbol x}$$分别称为增广权重向量和增广特征向量:
$$
\hat{\boldsymbol x}=\boldsymbol x \oplus 1=\begin{bmatrix}x_1\\ \vdots \\x_D \\1
\end{bmatrix}\\
\hat{\boldsymbol w}=\boldsymbol w \oplus b=\begin{bmatrix}w_1\\ \vdots \\w_D \\b
\end{bmatrix}\\
$$
其中$$\oplus$$定义为两个向量的拼接操作。

之后将采用简化的表示方法，即直接用$$\boldsymbol w$$和$$\boldsymbol x$$表示増广权重向量和増广特征向量。





# 参数学习

给定一组包含$$N$$个训练样本的训练集$$\mathcal{D} = \{(\boldsymbol x^{(n)},y^{(n)})\}_{n=1}^N$$,我们希望能够学习一个最优的线性回归的模型参数$$\boldsymbol w$$.

这里介绍四种不同的参数估计方法: 经验风险最小化、 结构风险最小化、 最大似然估计、 最大后验估计.



## 经验风险最小化

由于线性回归的标签 $$y$$ 和模型输出都为连续的实数值,因此<u>平方损失函数</u>非常合适衡量真实标签和预测标签之间的差异.

根据经验风险最小化准则,训练集 $$\mathcal{D}$$ 上的经验风险定义为
$$
\mathcal{R}(\boldsymbol w)=\sum_{n=1}^N \mathcal{L}(y^{(n)},f(\boldsymbol x^{(n)};\boldsymbol w))\\
=\frac{1}{2}\sum_{n=1}^N(y^{(n)}-\boldsymbol w^{\rm T}\boldsymbol x^{(n)})^2\\
=\frac{1}{2}||\boldsymbol y-\boldsymbol X^{\rm T} \boldsymbol w||^2
$$
其中
$$
\boldsymbol y = [y^{(1)},\cdots,y^{(N)}]^{\rm T}\\
\boldsymbol X=\begin{bmatrix} x_1^{(1)} & x_1^{(2)} & \cdots & x_1^{(N)}\\
\vdots & \vdots & \ddots & \vdots\\
x_D^{(1)} & x_D^{(2)} & \cdots & x_D^{(N)}\\
1 & 1 & \cdots & 1
\end{bmatrix}
$$
风险函数$$\mathcal{R}(\boldsymbol w)$$是关于$$\boldsymbol w$$的凸函数，其对$$\boldsymbol w$$的偏导数为（需要证明）
$$
\frac{\partial \mathcal{R}(\boldsymbol w)}{\partial \boldsymbol w}=\frac{1}{2}\frac{\partial ||\boldsymbol y-\boldsymbol X^{\rm T} \boldsymbol w||^2}{\partial \boldsymbol w}\\
=-\boldsymbol X(\boldsymbol y-\boldsymbol X^{\rm T}\boldsymbol w)
$$
令该偏导数为0，得到最优参数
$$
\boldsymbol w^* = (\boldsymbol X \boldsymbol X^{\rm T})^{-1} \boldsymbol X \boldsymbol y\\
=()
$$
这种求解线性回归参数的方法也叫**最小二乘法( Least Square Method , LSM )**.





