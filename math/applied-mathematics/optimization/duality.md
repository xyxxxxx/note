# Lagrange对偶函数

考虑标准形式的优化问题：
$$
\begin{align}
\min & \quad f(\pmb x)\\
{\rm s.t.} & \quad g_i(\pmb x)\le 0,\ i =1,\cdots,m\\
& \quad h_i(\pmb x)= 0,\ i =1,\cdots,p
\end{align}\tag{1}
$$
其中自变量$$\pmb x\in \mathbb{R}^n$$。设问题的定义域$$\mathcal{D}=\bold{dom}f\cap \bigcap_{i=1}^m\bold{dom}g_i \cap \bigcap_{i=1}^p\bold{dom}h_i$$是非空集合，优化问题的最优值为$$p^*$$。

Lagrange对偶的基本思想是在目标函数中考虑上述问题的约束条件，即添加约束条件的加权和，得到増广的目标函数。定义问题(1)的**Lagrange函数**$$L:\mathbb{R}^n\times \mathbb{R}^m\times \mathbb{R}^p\to \mathbb{R}$$为
$$
L(\pmb x,\pmb \lambda,\pmb \nu)=f(\pmb x)+\sum_{i=1}^m\lambda_ig_i(\pmb x)+\sum_{i=1}^p\nu_i h_i(\pmb x)
$$
其中定义域为$$\bold{dom}L=\mathcal{D}\times \mathbb{R}^m\times \mathbb{R}^p$$。$$\lambda_i$$称为第$$i$$个不等式约束$$g_i(\pmb x)\le 0$$对应的**Lagrange乘子**，$$\nu_i$$称为第$$i$$个等式约束$$h_i(\pmb x)=0$$对应的**Lagrange乘子**。向量$$\pmb\lambda,\pmb \nu$$称为问题(1)的Lagrange乘子向量。



## Lagrange对偶函数

定义**Lagrange对偶函数**或**对偶函数**$$\Lambda:\mathbb{R}^m\times \mathbb{R}^p\to \mathbb{R}$$为Lagrange函数关于$$\pmb x$$取得的最小值：即对$$\pmb \lambda\in \mathbb{R}^m, \pmb \nu\in \mathbb{R}^p$$，有
$$
\Lambda(\pmb \lambda,\pmb \nu)=\inf_{\pmb x\in\mathcal{D}}L(\pmb x,\pmb \lambda,\pmb \nu)
$$
如果Lagrange函数关于$$\pmb x$$无下界，则对偶函数取值为$$-\infty$$。因为对偶函数是一族关于$$(\pmb \lambda,\pmb \nu)$$的仿射函数的逐点下确界，所以即使原问题不是凸的，对偶函数也是凹函数。？



对偶函数构成了原问题最优值$$p^*$$的下界：即对任意$$\pmb\lambda\ge \pmb 0$$和$$\pmb\nu$$下式成立
$$
\Lambda(\pmb \lambda,\pmb \nu)\le p^*\tag{2}
$$
证明：设$$\tilde{\pmb x}$$是原问题的一个可行点，即$$g_i(\tilde{\pmb x})\le 0$$且$$h_i(\tilde{\pmb x})=0$$，对于$$\pmb \lambda\ge\pmb 0$$，有
$$
\sum_{i=1}^m\lambda_ig_i(\tilde{\pmb x})+\sum_{i=1}^p\nu_ih_i(\tilde{\pmb x})\le 0
$$
于是有
$$
L(\tilde{\pmb x},\pmb \lambda,\pmb \nu)=f(\tilde{\pmb x})+\sum_{i=1}^m\lambda_ig_i(\tilde{\pmb x})+\sum_{i=1}^p\nu_i h_i(\tilde{\pmb x})\le f(\tilde{\pmb x})
$$
因此
$$
\Lambda(\pmb \lambda,\pmb \nu)=\inf_{\pmb x\in\mathcal{D}}L(\pmb x,\pmb \lambda,\pmb \nu)\le L(\tilde{\pmb x},\pmb \lambda,\pmb \nu)\le f(\tilde{\pmb x})
$$
上式对任意可行点$$\tilde{\pmb x}$$成立。

对于$$x\in \mathbb{R}$$和只有一个不等式约束的某简单问题，下图描述了式(2)给出的下界。

![](https://i.loli.net/2020/12/10/IuMQhvpbOkLNA6U.png)

图1 实线表示目标函数$$f$$，虚线表示约束函数$$g$$，根据$$g\le 0$$得到可行域，即区间$$[-0.46,0.46]$$，图中用两条垂直点线表示。最优点和最优值分别为$$x^*=-0.46,p^*=1.54$$，图中用圆点表示。$$f$$附近的点线表示一系列Lagrange函数$$L(x,\lambda)$$，其中$$\lambda=0.1,0.2,\cdots,1.0$$。每个Lagrange函数都有一个极小值，均小于原问题最优目标值$$p^*$$。



![](https://i.loli.net/2020/12/10/dFBaxYVH4XgEJGt.png)

图2 图1中问题的对偶函数。函数$$f,g$$都不是凸函数，但是对偶函数是凹函数。水平虚线为原问题最优目标值$$p^*$$，即式(2)成立。



虽然式(2)成立，但是当$$\Lambda(\pmb \lambda,\pmb \nu)=-\infty$$时其意义不大。称满足$$\pmb\lambda\ge0$$以及$$(\pmb \lambda,\pmb \nu)\in \bold{dom}\Lambda$$（即$$\Lambda(\pmb \lambda,\pmb \nu)>-\infty$$）的$$(\pmb \lambda,\pmb \nu)$$是**对偶可行的**，此时对偶函数给出$$p^*$$的一个非平凡下界（即不是$$-\infty$$的下界）。



## 例子

**线性方程组的最小二乘解**

考虑问题
$$
\begin{align}
\min & \quad \pmb x^{\rm T}\pmb x\\
{\rm s.t.} & \quad A\pmb x=\pmb b\\
\end{align}
$$
其中$$A\in \mathbb{R}^{p\times n}$$。这个问题没有不等式约束，有$$p$$个（线性）等式约束，其Lagrange函数是$$L(\pmb x,\pmb \nu)=\pmb x^{\rm T}\pmb x+\pmb \nu^{\rm T}(A\pmb x-\pmb b)$$，定义域为$$\mathbb{R}^n\times \mathbb{R}^p$$，对偶函数为$$\Lambda(\pmb \nu)=\inf_{\pmb x\in\mathcal{D}}L(\pmb x,\pmb \nu)$$，由于$$L(\pmb x,\pmb \nu)$$是$$\pmb x$$的二次凸函数，可以通过求解如下的最优性条件得到函数的最小值，
$$
\nabla_{\pmb x} L(\pmb x,\pmb \nu)=2\pmb x+A^{\rm T}\pmb \nu=\pmb 0
$$
在点$$\pmb x=-(1/2)A^{\rm T}\pmb\nu$$处Lagrange函数达到最小值，因此对偶函数为
$$
\Lambda(\pmb \nu)=\inf_{\pmb x}L(\pmb x,\pmb \nu)=L(-(1/2)A^{\rm T}\pmb\nu,\pmb \nu)=-(1/4)\pmb \nu^{\rm T}AA^{\rm T}\pmb\nu-\pmb \nu^{\rm T}\pmb b
$$
它是一个二次凹函数，定义域为$$\mathbb{R}^p$$。根据对偶函数给出原问题下界的性质，则$$\forall \pmb\nu\in\mathbb{R}^p$$，有
$$
-(1/4)\pmb \nu^{\rm T}AA^{\rm T}\pmb\nu-\pmb \nu^{\rm T}\pmb b\le \inf \{\pmb x^{\rm T}\pmb x|A\pmb x=\pmb b\}
$$



**标准形式的线性规划**

考虑标准形式的线性规划问题
$$
\begin{align}
\min & \quad \pmb c^{\rm T}\pmb x\\
{\rm s.t.} & \quad A\pmb x=\pmb b\\
& \quad \pmb x\ge \pmb 0
\end{align}
$$
这个问题的不等式约束$$g_i(\pmb x)=-x_i,i=1,\cdots,n$$，其Lagrange函数是
$$
L(\pmb x,\pmb \lambda,\pmb \nu)=\pmb c^{\rm T}\pmb x-\sum_{i=1}^n\lambda_ix_i +\pmb \nu^{\rm T}(A\pmb x-\pmb b)\\
=-\pmb \nu^{\rm T}\pmb b+(\pmb c+A^{\rm T}\pmb \nu-\pmb \lambda)^{\rm T}\pmb x
$$
对偶函数为
$$
\Lambda(\pmb \lambda,\pmb \nu)=\inf_{\pmb x}L(\pmb x,\pmb \lambda,\pmb \nu)=-\pmb \nu^{\rm T}\pmb b+\inf_{\pmb x}(\pmb c+A^{\rm T}\pmb \nu-\pmb \lambda)^{\rm T}\pmb x
$$
由于线性函数只有一次项系数恒为零时才有下界，因此
$$
\Lambda(\pmb \lambda,\pmb \nu)=\begin{cases}-\pmb \nu^{\rm T}\pmb b,\quad \pmb c+A^{\rm T}\pmb \nu-\pmb \lambda=\pmb 0\\
-\infty,\quad\ \ 其它情况
\end{cases}
$$

> 注意到对偶函数$$\Lambda$$只有在$$\mathbb{R}^m\times \mathbb{R}^p$$上的一个仿射子集上才是有限值，实际上这是一种常见的情况。

只有当$$\pmb \lambda,\pmb \nu$$满足$$\pmb \lambda\ge \pmb 0$$和$$\pmb c+A^{\rm T}\pmb \nu-\pmb \lambda=\pmb 0$$时，下界才是非平凡的，此时$$-\pmb \nu^{\rm T}\pmb b$$给出了一个下界。






# Lagrange对偶问题

对于任意一组$$(\pmb \lambda,\pmb \nu)\in \bold{dom}\Lambda,\pmb\lambda\ge0$$，Lagrange对偶函数给出了优化问题(1)的最优值$$p^*$$的一个下界。一个自然的问题是：从Lagrange函数能够得到的最好下界是什么？

可以将这个问题表述为优化问题
$$
\begin{align}
\max & \quad \Lambda(\pmb \lambda,\pmb \nu)\\
{\rm s.t.} &\quad \pmb \lambda \ge \pmb 0
\end{align}\tag{3}
$$
上述问题称为问题(1)的**Lagrange对偶问题**或**对偶问题**，而问题(1)称为**原问题**。前面提到的对偶可行的概念，在这里的意义是对偶问题的一个可行解。称解$$(\pmb \lambda^*,\pmb \nu^*)$$是**对偶最优解**或者是**最优Lagrange乘子**，如果它是对偶问题的最优解。

对偶问题是一个凸优化问题，因为极大化的目标函数是凹函数，且约束集合是凸集。因此<u>对偶问题的凸性和原问题是否是凸优化问题无关</u>。



## 显式表达对偶约束

一般情况下，对偶函数的定义域
$$
\bold{dom}\Lambda=\{(\pmb \lambda,\pmb \nu)|\Lambda(\pmb \lambda,\pmb \nu)>-\infty \}
$$
的维数都小于$$m+p$$。事实上，很多情况下，我们可以识别出对偶问题目标函数$$\Lambda$$中隐含的等式约束。这样处理之后我们就得到一个等价的问题，在这个问题中隐含的等式约束都被显式地表达为优化问题的约束。考虑下面的例子：

**标准形式线性规划的Lagrange对偶**

标准形式线性规划
$$
\begin{align}
\min & \quad \pmb c^{\rm T}\pmb x\\
{\rm s.t.} & \quad A\pmb x=\pmb b\\
& \quad \pmb x\ge \pmb 0
\end{align}
$$
的Lagrange对偶函数为
$$
\Lambda(\pmb \lambda,\pmb \nu)=\begin{cases}-\pmb \nu^{\rm T}\pmb b,\quad \pmb c+A^{\rm T}\pmb \nu-\pmb \lambda=\pmb 0\\
-\infty,\quad\ \ 其它情况
\end{cases}
$$

对偶问题为

$$
\begin{align}
\max & \quad \Lambda(\pmb \lambda,\pmb \nu)=\begin{cases}-\pmb \nu^{\rm T}\pmb b,\quad \pmb c+A^{\rm T}\pmb \nu-\pmb \lambda=\pmb 0\\
-\infty,\quad\ \ 其它情况
\end{cases}\\
{\rm s.t.} &\quad \pmb \lambda \ge \pmb 0
\end{align}
$$
我们将对偶函数中隐含的等式约束显式表示来得到一个等价的问题
$$
\begin{align}
\max & \quad -\pmb \nu^{\rm T}\pmb b\\
{\rm s.t.} &\quad \pmb \lambda \ge \pmb 0\\
&\quad \pmb c+A^{\rm T}\pmb \nu-\pmb \lambda=\pmb 0
\end{align}
$$
变换为
$$
\begin{align}
\max & \quad -\pmb \nu^{\rm T}\pmb b\\
{\rm s.t.} &\quad \pmb c+A^{\rm T}\pmb \nu \ge \pmb 0\\
\end{align}
$$
这几个等价的问题都统称为标准形式线性规划的Lagrange对偶问题。



## 弱对偶性

Lagrange对偶问题的最优解用$$d^*$$表示，根据定义，这是原问题最优值$$p^*$$的最好下界。因此我们有以下简单但是非常重要的不等式
$$
d^*\le p^*\tag{4}
$$
即使原问题不是凸问题，上述不等式亦成立。这个性质称为**弱对偶性**。

即使当$$d^*,p^*$$无限时，弱对偶性不等式亦成立，例如如果原问题无下界，即$$p^*=-\infty$$，那么由弱对偶性必有$$d^*=-\infty$$，即Lagrange对偶问题没有可行解；反过来如果对偶问题无上界，即$$d^*=\infty$$，那么由弱对偶性必有$$p^*=\infty$$，即原问题没有可行解。

定义差值$$p^*-d^*$$是原问题的**最优对偶间隙**，它给出了原问题最优值以及通过Lagrange对偶函数所能得到的最好下界之间的差值。

<u>当原问题很难求解时，弱对偶不等式可以给出原问题最优值的一个下界，这是因为对偶问题总是凸问题，而且在很多情况下都可以进行有效的求解得到$$d^*$$。</u>



## 强对偶性和Slater约束准则

如果等式
$$
d^*= p^*\tag{5}
$$
成立，即最优对偶间隙为零，那么强对偶性成立。这说明从Lagrange对偶函数得到的最好下界是紧的。

对于一般情况，强对偶性不成立。但是如果原问题是凸问题，即可以表述为如下形式
$$
\begin{align}
\min & \quad f(\pmb x) \\
{\rm s.t.} & \quad g_i(\pmb x)\le 0,\ i=1,\cdots,m\\
& \quad A\pmb x=\pmb b
\end{align}
$$
其中函数$$f,g_i$$是凸函数，那么强对偶性通常（但不总是）成立。有很多研究成果给出了强对偶性成立的条件，这些条件称为约束准则。



一个简单的约束准则是Slater条件：存在一点$$\tilde{\pmb x}\in \bold{relint}\mathcal{D}$$使得
$$
g_i(\tilde{\pmb x})<0,\ i=1,\cdots,m\\
A\tilde{\pmb x}=\pmb b
$$
满足上述条件的点称为严格可行。Slater定理说明，<u>当原问题是凸问题，且Slater条件成立时，强对偶性成立</u>。

当不等式约束函数$$g_i$$中有一些是仿射函数时，Slater条件可以进一步改进。如果前$$k$$个不等式约束函数$$g_1,\cdots,g_k$$是仿射的，那么若存在一点$$\tilde{\pmb x}\in \bold{relint}\mathcal{D}$$使得
$$
g_i(\tilde{\pmb x})\le 0,\ i=1,\cdots,k\\
g_i(\tilde{\pmb x})< 0,\ i=k+1,\cdots,m\\
A\tilde{\pmb x}=\pmb b
$$
强对偶性成立。换言之，即仿射不等式不需要严格成立。注意到当所有约束条件都是线性等式或不等式且$$\bold{dom}f$$是开集时，改进的Slater条件就是可行性条件。



若Slater条件或其改进形式成立，意味着对于凸问题不仅强对偶性成立，而且当$$d^*>-\infty$$时对偶问题能够取到最优值，即存在一组对偶可行解$$(\pmb \lambda^*,\pmb\nu^*)$$使得$$\Lambda (\pmb \lambda^*,\pmb\nu^*)=d^*=p^*$$。



## 例子

**线性方程组的最小二乘解**

再次考虑问题
$$
\begin{align}
\min & \quad \pmb x^{\rm T}\pmb x\\
{\rm s.t.} & \quad A\pmb x=\pmb b\\
\end{align}
$$
其对偶问题为
$$
\max\quad -(1/4)\pmb \nu^{\rm T}AA^{\rm T}\pmb\nu-\pmb \nu^{\rm T}\pmb b
$$
它是一个凹二次函数的无约束极大化问题。

Slater条件此时是原问题的可行性条件，所以有$$d^*=p^*$$。





# Lagrange对偶的解释

## 价格解释









# 最优性条件

## 互补松弛性

设<u>强对偶性成立</u>，$$\pmb x^*$$是原问题的最优解，$$(\pmb \lambda^*,\pmb \nu^*)$$是对偶问题的最优解，这表明
$$
\begin{align}
f(\pmb x^*)&=\Lambda(\pmb \lambda^*,\pmb \nu^*)\\
&=\inf_{\pmb x}(f(\pmb x^*)+\sum_{i=1}^m\lambda_i^*g_i(\pmb x^*)+\sum_{i=1}^p\nu_i^* h_i(\pmb x^*))\\
&\le f(\pmb x^*)+\sum_{i=1}^m\lambda_i^*g_i(\pmb x^*)+\sum_{i=1}^p\nu_i^* h_i(\pmb x^*)\\
&\le f(\pmb x^*)
\end{align}
$$
第一个等式说明最优对偶间隙为零；第三个不等式说明Lagrange函数$$L(\pmb x,\pmb \lambda^*,\pmb \nu^*)$$在$$\pmb x^*$$处取得最小值（$$\pmb x^*$$可能只是所有最小点之一）；第四个不等式说明
$$
\sum_{i=1}^m\lambda_i^*g_i(\pmb x^*)=0
$$
实际上，求和的每一项都非正，因此有
$$
\lambda_i^*g_i(\pmb x^*)=0,\ i=1,\cdots,m
$$
上述条件称为**互补松弛条件**，也可以写成
$$
\lambda_i^*>0\Rightarrow g_i(\pmb x^*)=0\\
g_i(\pmb x^*)<0\Rightarrow \lambda_i^*=0\\
$$
互补松弛条件意味着在最优点处，第$$i$$个约束起作用（即取等），或者Lagrange乘子$$\pmb\lambda$$的第$$i$$项为零。



## KKT最优性条件

现在假设函数$$f,g_1,\cdots,g_m,h_1,\cdots,h_p$$可微（因此定义域是开集），但是并不假设这些函数是凸函数。



**非凸问题的KKT条件**

和前面一样，设<u>强对偶性成立</u>，$$\pmb x^*$$和$$(\pmb \lambda^*,\pmb \nu^*)$$分别是原问题和对偶问题的一对最优解。因为$$L(\pmb x,\pmb \lambda^*,\pmb \nu^*)$$在$$\pmb x^*$$处取得最小值，因此在此处的导数必须为零，即
$$
\nabla f(\pmb x^*)+\sum_{i=1}^m\lambda_i^*\nabla g_i(\pmb x^*)+\sum_{i=1}^p\nu_i^*\nabla h_i(\pmb x^*)=\pmb 0
$$
因此有
$$
\begin{align}
g_i(\pmb x^*)\le 0,&\quad i=1,\cdots,m\\
\lambda^*_i\ge 0,&\quad i=1,\cdots,m\\
\lambda^*_ig_i(\pmb x^*)= 0,&\quad i=1,\cdots,m\\
h_i(\pmb x^*)=0,&\quad i=1,\cdots,p\\
\nabla f(\pmb x^*)+\sum_{i=1}^m\lambda_i^*\nabla g_i(\pmb x^*)+\sum_{i=1}^p\nu_i^*\nabla h_i(\pmb x^*)=\pmb 0
\end{align}
$$
我们称上式为 **Karush-Kuhn-Tucker(KKT)** 条件。

总之，对于目标函数和约束函数可微的优化问题，如果强对偶性成立，那么任意一对原问题和对偶问题的最优解必须满足KKT条件。



**凸问题的KKT条件**

当原问题是凸问题时，满足KKT条件的解就是原问题和对偶问题的最优解。换言之，如果函数$$f,g_i$$是凸函数，$$h_i$$是仿射函数，$$\tilde{\pmb x},\tilde{\pmb \lambda},\tilde{\pmb \nu}$$是任意满足KKT条件的点，
$$
\begin{align}
g_i(\tilde{\pmb x})\le 0,&\quad i=1,\cdots,m\\
\tilde\lambda_i\ge 0,&\quad i=1,\cdots,m\\
\tilde\lambda_ig_i(\tilde{\pmb x})= 0,&\quad i=1,\cdots,m\\
h_i(\tilde{\pmb x})=0,&\quad i=1,\cdots,p\\
\nabla f(\tilde{\pmb x})+\sum_{i=1}^m\tilde\lambda_i\nabla g_i(\tilde{\pmb x})+\sum_{i=1}^p\tilde\nu_i\nabla h_i(\tilde{\pmb x})=\pmb 0
\end{align}
$$
那么$$\tilde{\pmb x},(\tilde{\pmb \lambda},\tilde{\pmb \nu})$$分别是原问题和对偶问题的最优解，对偶间隙为零。

为了说明这一点，注意到条件1, 4说明了$$\tilde{\pmb x}$$是原问题的可行解；条件2说明$$L(\pmb x,\tilde{\pmb\lambda},\tilde{\pmb\nu})$$是$$\pmb x$$的凸函数；条件5说明在$$\pmb x=\tilde{\pmb x}$$处，$$L(\pmb x,\tilde{\pmb\lambda},\tilde{\pmb\nu})$$的导数为零，即取得最小值。由此
$$
\begin{align}
\Lambda(\tilde{\pmb\lambda},\tilde{\pmb\nu})&=L(\tilde{\pmb x},\tilde{\pmb\lambda},\tilde{\pmb\nu})\\
&=f(\tilde{\pmb x})+\sum_{i=1}^m\tilde\lambda_i g_i(\tilde{\pmb x})+\sum_{i=1}^p\tilde\nu_i h_i(\tilde{\pmb x})\\
&=f(\tilde{\pmb x}) 
\end{align}
$$
最后一行成立是因为$$h_i(\tilde{\pmb x})=0$$和$$\tilde\lambda_ig_i(\tilde{\pmb x})=0$$。这说明对偶间隙为零，因此分别是原问题和对偶问题的最优解。

总之，对于目标函数和约束函数可微的**凸**优化问题，任意满足KKT条件的点分别是原问题和对偶问题的最优解，并且对偶间隙为零。

若某个凸优化问题具有可微的目标函数和约束函数，且满足Slater条件（强对偶性成立），那么KKT条件是最优性的充要条件。

KKT条件在优化领域有着重要的作用。在一些特殊的情形下可以解析求解KKT条件；更一般地，很多求解凸优化问题的方法可以理解为就是求解KKT条件的方法。



## KKT最优性条件的力学解释

可以从力学角度对KKT条件给出一个较好的解释。如下图所示，两个滑块通过三段弹簧连接，质点的位置用$$\pmb x\in \mathbb{R}^2$$描述，各参数见图。

![](https://i.loli.net/2020/12/10/MBPti1KlODW2unk.png)

设弹簧原长为0，弹性系数为$$k_1,k_2,k_3$$，弹性势能为
$$
f(x_1,x_2)=\frac{1}{2}k_1x_1^2+\frac{1}{2}k_2(x_2-x_1)^2+\frac{1}{2}k_3(l-x_2)^2
$$
在满足以下不等式约束
$$
w/2-x_1\le 0\\
w+x_1-x_2\le 0\\
w/2-l+x_2\le 0\tag{6}
$$
的条件下极小化弹性势能可以得到平衡位置$$\pmb x^*$$，即得到如下优化问题
$$
\begin{align}
\min & \quad \frac{1}{2}(k_1x_1^2+k_2(x_2-x_1)^2+k_3(l-x_2)^2)\\
{\rm s.t.} & \quad w/2-x_1\le 0\\
&\quad w+x_1-x_2\le 0\\
&\quad w/2-l+x_2\le 0
\end{align}
$$
这是一个二次规划问题。

引入Lagrange乘子$$\lambda_1,\lambda_2,\lambda_3$$，此问题的KKT条件包含运动约束(6)，非负约束$$\lambda_i\ge 0$$，互补松弛条件
$$
\lambda_1(w/2-x_1)=0\\
\lambda_2(w+x_1-x_2)=0\\
\lambda_3(w/2-l+x_2)=0
$$
以及零梯度条件
$$
\begin{bmatrix}k_1x_1-k_2(x_2-x_1)\\k_2(x_2-x_1)-k_3(l-x_2)
\end{bmatrix}+\lambda_1\begin{bmatrix}-1\\0
\end{bmatrix}+\lambda_2\begin{bmatrix}1\\-1
\end{bmatrix}+\lambda_3\begin{bmatrix}0\\1
\end{bmatrix}=\pmb 0\tag{7}
$$
原问题是凸问题，因此求解上述KKT条件即得到原问题和对偶问题的最优解。



现在我们从另一个角度，即受力分析的角度去理解方程(7)，如图所示

![](https://i.loli.net/2020/12/10/ftDw5RBNH2lAFbX.png)

Lagrange乘子即可以理解为墙或滑块之间的<u>压力</u>，那么松弛互补条件就可以理解为物体接触才能产生压力。



## 通过求解对偶问题来求解原问题

在互补松弛性的开始部分我们提到，如果强对偶性成立，$$(\pmb \lambda^*,\pmb \nu^*)$$是对偶最优解，那么原问题的最优解$$\pmb x^*$$也一定是$$L(\pmb x,\pmb \lambda^*,\pmb \nu^*)$$的最优解。这个性质可以让我们通过求解对偶问题来求解原问题。

更精确地，假设强对偶性成立，对偶最优解$$(\pmb \lambda^*,\pmb \nu^*)$$已知，$$L(\pmb x,\pmb \lambda^*,\pmb \nu^*)$$的最小点，即下列问题的解
$$
\min \quad f(\pmb x)+\sum_{i=1}^m\lambda_i^*g_i(\pmb x)+\sum_{i=1}^p\nu_i h_i^*(\pmb x)\tag{8}
$$
唯一（例如对于凸问题，$$L(\pmb x,\pmb \lambda^*,\pmb \nu^*)$$是$$\pmb x$$的严格凸函数），那么如果问题(8)的解是原问题的可行解，那么它就是原问题的最优解；如果它不是原问题的可行解，那么原问题的最优解不能达到。当对偶问题比原问题更易求解时，比如说对偶问题可以解析求解或者结构特殊时，上述方法很有意义。