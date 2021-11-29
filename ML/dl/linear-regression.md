**线性回归(linear regression)**是机器学习和统计学中最基础和最广泛应用的模型，是一种对自变量和因变量之间关系进行建模的回归分析。自变量数量为1时称为**简单回归**，自变量数量大于1时称为**多元回归**。

从机器学习的角度来看，自变量就是样本的特征向量 $\pmb x ∈ \mathbb{R}^D$ (每一维对应一个自变量)，因变量是标签 $y$，这里 $y ∈ \mathbb{R}$ 是连续值(实数或连续整数)。假设空间是一组参数化的线性函数：
$$
f(\pmb x; \pmb w,b)=\pmb w^{\rm T}\pmb x+b
$$
其中权重向量 $\pmb w \in \mathbb{R}^D$ 和偏置 $b\in \mathbb{R}$ 都是可学习的参数，函数 $f(\pmb x; \pmb w，b) \in \mathbb{R}$ 也称为**线性模型**。

为简单起见，将上式写为
$$
f(\pmb x;\hat{\pmb w})=\hat{\pmb w}^{\rm T}\hat{\pmb x}
$$
其中 $\hat{\pmb w}$ 和 $\hat{\pmb x}$ 分别称为增广权重向量和增广特征向量:
$$
\hat{\pmb x}=\pmb x \oplus 1=\begin{bmatrix}x_1\\ \vdots \\x_D \\1
\end{bmatrix}\\
\hat{\pmb w}=\pmb w \oplus b=\begin{bmatrix}w_1\\ \vdots \\w_D \\b
\end{bmatrix}\\
$$
其中 $\oplus$ 定义为两个向量的拼接操作。

之后将采用简化的表示方法，即直接用 $\pmb w$ 和 $\pmb x$ 表示増广权重向量和増广特征向量。



## 参数学习

给定一组包含 $N$ 个训练样本的训练集 $\mathcal{D} = \{(\pmb x^{(n)},y^{(n)})\}_{n=1}^N$，我们希望能够学习一个最优的线性回归的模型参数 $\pmb w$。

这里介绍四种不同的参数估计方法：经验风险最小化、结构风险最小化、最大似然估计、最大后验估计。

### 经验风险最小化（最小二乘法）

由于线性回归的标签 $y$ 和模型输出都为连续的实数值，因此<u>平方损失函数</u>非常合适衡量真实标签和预测标签之间的差异。

根据经验风险最小化准则，训练集 $\mathcal{D}$ 上的经验风险定义为
$$
\mathcal{R}(\pmb w)=\sum_{n=1}^N \mathcal{L}(y^{(n)},f(\pmb x^{(n)};\pmb w))\\
=\frac{1}{2}\sum_{n=1}^N(y^{(n)}-\pmb w^{\rm T}\pmb x^{(n)})^2\\
=\frac{1}{2}||\pmb y-X^{\rm T} \pmb w||^2
$$
其中
$$
\pmb y = [y^{(1)},\cdots,y^{(N)}]^{\rm T}\\
X=\begin{bmatrix} x_1^{(1)} & x_1^{(2)} & \cdots & x_1^{(N)}\\
\vdots & \vdots & \ddots & \vdots\\
x_D^{(1)} & x_D^{(2)} & \cdots & x_D^{(N)}\\
1 & 1 & \cdots & 1
\end{bmatrix}
$$
风险函数 $\mathcal{R}(\pmb w)$ 是关于 $\pmb w$ 的凸函数，其对 $\pmb w$ 的偏导数为
$$
\frac{\partial \mathcal{R}(\pmb w)}{\partial \pmb w}=\frac{1}{2}\frac{\partial ||\pmb y-X^{\rm T} \pmb w||^2}{\partial \pmb w}\\
=-X(\pmb y-X^{\rm T}\pmb w)
$$
令该偏导数为 $\pmb 0$，得到最优参数
$$
\pmb w^* = (X X^{\rm T})^{-1} X \pmb y\\
$$
建立这种求解线性回归参数的方法也叫**最小二乘法(Least Square Method , LSM)**。



@对平面直角坐标系上的点： $(1,1),(3,2),(5,5),(8,6),(9,7),(11,8)$ 进行线性回归，其中 $x$ 为自变量， $y$ 为因变量。
$$
X=\begin{bmatrix}1&3&5&8&9&11\\
1&1&1&1&1&1
\end{bmatrix},
\pmb y=(1,2,5,6,7,8)^{\rm T}\\
\pmb w^*=(X X^{\rm T})^{-1} X \pmb y=(0.7162,0.4165)^{\rm T}\\
$$

回归方程为 $y=0.7162x+0.4165$，如图所示。

<img src="https://i.loli.net/2020/10/28/zbmDUGLBrSJ21cw.png" alt="Screenshot from 2020-10-28 18-07-23.png" style="zoom:50%;" />




最小二乘法要求 $XX^{\rm T}\in \mathbb{R}^{(D+1)\times (D+1)}$ 必须存在逆矩阵。一种常见的 $XX^{\rm T}$ 不可逆的情况是样本数量 $N$ 小于特征数量 $(D+1)$，这时 $XX^{\rm T}$ 的秩为 $N$。

当 $XX^{\rm T}$ 不可逆时, 可以通过下面两种方法来估计参数：

1. 先使用主成分分析等方法来预处理数据，消除不同特征之间的相关性，然后再使用最小二乘法来估计参数；

2. 使用梯度下降法来估计参数：先初始化 $\pmb w =\pmb 0$，然后通过下面公式进行迭代：
   $$
   \pmb w ←\pmb w +αX(\pmb y −X^{\rm T}\pmb w),
   $$
   这种方法也称为最小均方(least mean squares, LMS)算法。



@对平面直角坐标系上的点： $(1,1),(1,2)$ 进行线性回归，其中 $x$ 为自变量， $y$ 为因变量。
$$
X=\begin{bmatrix}1&1\\1&1
\end{bmatrix},\pmb y=(1,2)^{\rm T} \\
XX^{\rm T}不可逆
$$
使用梯度下降法，设定初始值 $\pmb w_0=(0,0)^{\rm T}$， $\alpha=0.1$，
$$
\pmb w_1=\pmb w_0+αX(\pmb y −X^{\rm T}\pmb w_0)=(0.3,0.3)^{\rm T}\\
\pmb w_2=\pmb w_1+αX(\pmb y −X^{\rm T}\pmb w_1)=(0.48,0.48)^{\rm T}\\
\cdots\\
\pmb w_\infty = (0.75,0.75)^{\rm T}
$$
<img src="https://i.loli.net/2020/10/28/KBC6g5aUwTSHqZV.png" alt="Screenshot from 2020-10-28 20-06-19.png" style="zoom:50%;" />



### 结构风险最小化（岭回归和Lasso回归）

即使 $XX^{\rm T}$ 可逆，如果特征之间有较大的**多重共线性(multicollinearity)**，也会使得 $XX^{\rm T}$ 的逆在数值上无法准确计算。数据集 $X$ 上一些小的扰动就会导致 $(XX^{\rm T})^{-1}$ 发生大的改变，进而使得最小二乘法的计算变得很不稳定。

> 共线性(collinearity)指一个特征可以通过其他特征的线性组合来较准确地预测

为了解决这个问题，**岭回归(ridge regression)**给 $XX^{\rm T}$ 的对角线元素都加上一个常数 $λ$ 使得 $(XX^{\rm T}+ λI)$ 满秩。最优的参数 $\pmb w^∗ $ 为
$$
\pmb w^*=(XX^{\rm T} +\lambda I)^{-1}X\pmb y
$$
其中 $\lambda >0$。



@求线性回归
$$
X=\begin{bmatrix}1&1.05\\1&1
\end{bmatrix},\pmb y=(1,2)^{\rm T} \\
\pmb w^*=(X X^{\rm T})^{-1} X \pmb y=(20,-19)^{\rm T}\\\quad\\
X=\begin{bmatrix}1&1.1\\1&1
\end{bmatrix},\pmb y=(1,2)^{\rm T} \\
\pmb w^*=(X X^{\rm T})^{-1} X \pmb y=(10,-9)^{\rm T}\\
$$
求岭回归，设 $\lambda =0.01$ 
$$
X=\begin{bmatrix}1&1.05\\1&1
\end{bmatrix},\pmb y=(1,2)^{\rm T} \\
\pmb w^*=(X X^{\rm T}+\lambda I)^{-1} X \pmb y=(1.857,-0.401)^{\rm T}\\\quad\\
X=\begin{bmatrix}1&1.1\\1&1
\end{bmatrix},\pmb y=(1,2)^{\rm T} \\
\pmb w^*=(X X^{\rm T}+\lambda I)^{-1} X \pmb y=(2.529,-1.149)^{\rm T}\\
$$
<img src="https://i.loli.net/2020/10/28/f7knMNPehYCw1Om.png" alt="Screenshot from 2020-10-28 21-28-53.png" style="zoom:50%;" />

求岭回归，设 $\lambda =0.1$ 
$$
X=\begin{bmatrix}1&1.05\\1&1
\end{bmatrix},\pmb y=(1,2)^{\rm T} \\
\pmb w^*=(X X^{\rm T}+\lambda I)^{-1} X \pmb y=(0.852,0.597)^{\rm T}\\\quad\\
X=\begin{bmatrix}1&1.1\\1&1
\end{bmatrix},\pmb y=(1,2)^{\rm T} \\
\pmb w^*=(X X^{\rm T}+\lambda I)^{-1} X \pmb y=(0.952,0.476)^{\rm T}\\
$$
<img src="https://i.loli.net/2020/10/28/Q6DzdBVx2MtjlrK.png" alt="Screenshot from 2020-10-28 21-29-39.png" style="zoom:50%;" />



@求线性回归
$$
X=\begin{bmatrix}1&2&3&4.05&5\\2&3&4&5&6 \\
1&1&1&1&1
\end{bmatrix},
\pmb y=(11,20,32,42,51)^{\rm T}\\
\pmb w^*=(X X^{\rm T})^{-1} X \pmb y=(17.143,-7.029,7.714)^{\rm T}\\\quad\\
X=\begin{bmatrix}1&2&3&4.1&5\\2&3&4&5&6 \\
1&1&1&1&1
\end{bmatrix},
\pmb y=(11,20,32,42,51)^{\rm T}\\
\pmb w^*=(X X^{\rm T})^{-1} X \pmb y=(8.571,1.543,-0.857)^{\rm T}\\
$$
求岭回归，设 $\lambda =0.1$ 
$$
X=\begin{bmatrix}1&2&3&4.05&5\\2&3&4&5&6 \\
1&1&1&1&1
\end{bmatrix},
\pmb y=(11,20,32,42,51)^{\rm T}\\
\pmb w^*=(X X^{\rm T}+\lambda I)^{-1} X \pmb y=(6.399,3.632,-2.536)^{\rm T}\\\quad\\
X=\begin{bmatrix}1&2&3&4.1&5\\2&3&4&5&6 \\
1&1&1&1&1
\end{bmatrix},
\pmb y=(11,20,32,42,51)^{\rm T}\\
\pmb w^*=(X X^{\rm T}+\lambda I)^{-1} X \pmb y=(6.372,3.628,-2.504)^{\rm T}\\
$$



岭回归的解 $\pmb w^*$ 可以看作结构风险最小化准则下的最小二乘法估计，其结构风险在经验风险的基础上增加了 $l_2$ 范数的正则化项：
$$
\mathcal{R}(\pmb w)=\frac{1}{2}||\pmb y-X^{\rm T} \pmb w||^2+\frac{1}{2}\lambda||\pmb w||^2
$$
类似地，**LASSO回归**的结构风险在经验风险的基础上增加了 $l_1$ 范数的正则化项：
$$
\mathcal{R}(\pmb w)=\frac{1}{2}||\pmb y-X^{\rm T} \pmb w||^2+\lambda\sum_{i=1}^{D+1}\sqrt{w_i^2+\varepsilon}
$$




### 最大似然估计

除了直接建立 $\pmb x$ 和标签 $y$ 之间的函数关系外，线性回归还可以从建立条件概率 $p(y|\pmb x)$ 的角度来进行参数估计。

在给定 $\pmb x$ 的条件下，假设标签 $y$ 为一个随机变量，由函数 $f(\pmb x;\pmb w)=\pmb w^{\rm T}\pmb x$ 加上一个随机噪声 $ε$ 决定，即
$$
y=\pmb w^{\rm T}\pmb x+\varepsilon
$$
其中 $\varepsilon$ 服从均值为0，方差为 $\sigma^2$ 的高斯分布。这样， $y$ 服从均值为 $\pmb w^{\rm T}\pmb x$，方差为 $\sigma^2$ 的高斯分布
$$
p(y|\pmb x;\pmb w,\sigma)=\mathcal{N}(y;\pmb w^{\rm T}\pmb x,\sigma^2) =\frac{1}{\sqrt{2\pi}\sigma}\exp(-\frac{(y-\pmb w^{\rm T}\pmb x)^2}{2\sigma^2})
$$
参数 $\pmb w$ 在训练集 $\mathcal{D}$ 上的似然函数为
$$
p(\pmb y|X;\pmb w,\sigma)=\prod_{n=1}^N p(y^{(n)}|\pmb x^{(n)};\pmb w,\sigma)
$$

> **最大似然估计(MLE)**方法是找到一组参数 $\pmb w$ 使得似然函数取最大值。

建立似然方程组
$$
\frac{\partial \ln p(\pmb y|X;\pmb w,\sigma)}{\partial \pmb w}=\pmb 0
$$
解得
$$
\pmb w^{MLE} = (X X^{\rm T})^{-1} X \pmb y\\
$$
可以看到，最大似然估计的解和最小二乘法的解相同。



### 最大后验估计

假设参数 $\pmb w$ 为一个随机向量，并服从一个先验分布 $p(\pmb w;\nu)$。为简单起见，一般令 $p(\pmb w;\nu)$ 为各向同性的高斯分布
$$
p(\pmb w;\nu)=\mathcal{N}(\pmb w;\pmb 0,\nu^2I)
$$
其中 $\nu^2$ 为每一维上的方差。根据贝叶斯公式，参数 $\pmb w$ 的后验分布为
$$
p(\pmb w|X,\pmb y;\nu,\sigma)=\frac{p(\pmb w,\pmb y|X;\nu,\sigma)}{\sum_{\pmb w}p(\pmb w,\pmb y|X;\nu,\sigma)}\propto p(\pmb w;\nu)p(\pmb y|X,\pmb w;\nu,\sigma)
$$
其中 $p(\pmb y|X,\pmb w;\nu,\sigma)$ 为 $\pmb w$ 的似然函数。

> 这种估计参数 $\pmb w$ 的后验概率分布的方法称为**贝叶斯估计**，采用贝叶斯估计的线性回归也称为**贝叶斯线性回归**。
>
> 为得到点估计，可以采用最大后验估计方法。**最大后验估计(MAP)**方法是找到参数使后验分布取最大值
> $$
> \pmb w^{MAP}=\arg\max_{\pmb w}p(\pmb y|X,\pmb w;\sigma)p(\pmb w;\nu)
> $$

令似然函数为前面定义的高斯密度函数，那么
$$
p(\pmb w|X,\pmb y;\nu,\sigma)\propto -\frac{1}{2\sigma^2}\|\pmb y-X^{\rm T}\pmb w \|^2-\frac{1}{2\nu^2}\pmb w^{\rm T}\pmb w
$$
可以看到，最大后验估计等价于 $\ell_2$ 正则化的的结构风险最小化，其中正则化系数 $\lambda=\sigma^2/\lambda^2$。

最大似然估计和贝叶斯估计可以分别看作频率学派和贝叶斯学派对需要估计的参数 $\pmb w$ 的不同解释。

