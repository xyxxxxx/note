# 仿射集合和凸集

## 直线和线段

设 $x_1\neq x_2$ 是 $\mathbb{R}^n$ 空间的两个点，那么具有下列形式的点

$$
y=\theta x_1+(1-\theta)x_2,\ \theta\in\mathbb{R}
$$

组成**直线** $x_1x_2$。若限定 $\theta\in [0,1]$，则组成**闭线段** $x_1x_2$。

## 仿射集合

如果通过集合 $C\sube \mathbb{R}^n$ 中任意两个不同点的直线仍然在集合 $C$ 中，那么称集合 $C$ 是**仿射的**。

这个概念可以扩展到多个点的情况。如果 $\theta_1+\cdots+\theta_k=1$，我们称具有 $\theta_1x_1+\cdots+\theta_kx_k$ 形式的点为 $x_1,\cdots,x_k$ 的**仿射组合**。 $C$ 是一个仿射集合等价于 $\forall x_1,\cdots,x_k\in C$， $x_1,\cdots,x_k$ 的仿射组合仍在 $C$ 中。

如果 $C$ 是一个仿射集合并且 $x_0\in C$，则集合
$$
V=C-x_0=\{x-x_0|x\in C\}
$$
是一个子空间，即关于加法和数乘是封闭的。因此仿射集合 $C$ 可以表示为
$$
C=V+x_0=\{v+x_0|v\in V\}
$$
即一个子空间加上一个偏移。子空间 $V$ 与 $x_0$ 的选取无关，所以 $x_0$ 可以是 $C$ 中任意一点。我们定义仿射集合 $C$ 的**维数**为子空间 $V$ 的维数。

> $\mathbb{R}^3$ 上的所有平面和直线都是仿射集合，所有过原点的平面和直线都是子空间。

我们称由集合 $C\sube \mathbb{R}^n$ 中的点的所有仿射组合组成的集合为 $C$ 的**仿射包**，记为 $\bold{aff} C$ 
$$
\bold{aff}C=\{\theta_1x_1+\cdots+\theta_kx_k|x_1,\cdots,x_k\in C,\theta_1+\cdots+\theta_k=1 \}
$$
仿射包是包含 $C$ 的最小的仿射集合。

## 仿射维数与相对内部

我们定义集合 $C$ 的**仿射维数**为其仿射包的维数。仿射维数由于仿射包的性质，通常大于或等于其它维数，例如 $\mathbb{R}^2$ 上的单位圆环 $\{x\in\mathbb{R}^2|x_1^2+x_2^2=1\}$，它的仿射维数为2，但在其它大多数维数的定义下维数为1。

如果集合 $C\sube\mathbb{R}^n$ 的仿射维数小于 $n$，即其仿射包 $\bold{aff} C\neq \mathbb{R}^n$，我们定义集合 $C$ 的**相对内部**为 $\bold{aff} C$ 的内部，记作 $\bold{relint}C$，即
$$
\bold{relint}C=\{x\in C|B(x,r)\cap\bold{aff}C\sube C,r>0 \}
$$
其中 $B(x,r)=\{y|\|y-x\|\le r \}$，即以 $x$ 为中心， $r$ 为半径并由范数 $\|\cdot\|$ 定义的球（范数可以是任意范数，不影响结果）。我们于是可以定义集合 $C$ 的相对边界 $\bold{cl}C\backslash \bold{relint}C$，这里 $\bold{cl}C$ 表示 $C$ 的闭包。

> 集合 $C$ 的闭包 $\bold{cl}C$ 定义为包含 $C$ 的最小闭集合，例如在 $\mathbb{R}$ 上， $\bold{cl}(0,1)=[0,1]$。

@考虑 $\mathbb{R}^3$ 中处于 $(x_1,x_2)$ 平面的一个正方形
$$
C=\{x\in\mathbb{R}^3|-1\le x_1\le 1,-1\le x_2\le 1,x_3=0 \}
$$
其仿射包为 $(x_1,x_2)$ 平面，即 $\bold{aff}C=\{x\in\mathbb{R}^3|x_3=0 \}$， $C$ 的相对内部为
$$
\bold{relint}C=\{x\in\mathbb{R}^3|-1< x_1< 1,-1< x_2< 1,x_3=0 \}
$$
$C$ 的相对边界是其边框
$$
\bold{cl}C\backslash \bold{relint}C=\{x\in\mathbb{R}^3|\max\{|x_1|,|x_2|\}=1,x_3=0 \}
$$

## 凸集

集合 $C$ 被称为**凸集**，如果 $C$ 中任意两点间的线段仍在 $C$ 中，即 $\forall x_1,x_2\in C,\forall \theta\in [0,1]$，
$$
\theta x_1+(1-\theta)x_2\in C
$$
由于仿射集合包含穿过集合中任意两点的整条直线，因此仿射集合是凸集。下图展示了 $\mathbb{R}^2$ 上的一些简单的凸集和非凸集。

![Screenshot from 2020-10-22 10-58-03.png](https://i.loli.net/2020/10/22/2PihEgwJcHCQonZ.png)

我们称点 $\theta_1x_1+\cdots+\theta_kx_k$ 是点 $x_1,\cdots,x_k$ 的一个**凸组合**，其中 $\theta_1+\cdots+\theta_k=1,\  \theta_i\ge 0,i=1,\cdots,k$。与仿射集合类似， $C$ 是一个凸集等价于 $\forall x_1,\cdots,x_k\in C$， $x_1,\cdots,x_k$ 的凸组合仍在 $C$ 中。

我们称集合 $C$ 中所有点的凸组合的集合为其**凸包**，记作 $\bold{conv}C$ ：
$$
\bold{conv}C=\{\theta_1x_1+\cdots+\theta_kx_k|x_i\in C,\theta_i\ge 0,i=1,\cdots,k,\ \theta_1+\cdots+\theta_k=1 \}
$$
凸包是包含 $C$ 的最小凸集，如下图所示。

![Screenshot from 2020-10-22 11-15-12.png](https://i.loli.net/2020/10/22/KHxADQfR9me2s5Y.png)

凸组合的概念可以扩展到无穷级数、积分以及大多数形式的概率分布。假设 $\theta_1,\theta_2,\cdots$ 满足
$$
\theta_i\ge0,\ i=1,2,\cdots,\quad \sum_{i=1}^\infty\theta_i=1
$$
对于凸集 $C\sube \mathbb{R}^n$， $\forall x_1,x_2,\cdots\in C$，如果下面的级数收敛，则有
$$
\sum_{i=1}^\infty\theta_ix_i\in C
$$
更一般地，对于凸集 $C\sube \mathbb{R}^n$，假设 $p:\mathbb{R}^n\to \mathbb{R}$ 对所有 $x\in C$ 满足 $p(x)\ge 0$，并且 $\int_Cp(x){\rm d}x=1$，如果下面的积分存在，则有
$$
\int_Cp(x)x{\rm d}x\in C
$$
最一般的情况，对于凸集 $C\sube \mathbb{R}^n$， $x$ 是随机变量，并且 $P(x\in C)=1$，那么 $E(x)\in C$。

## 锥

如果对于任意 $x\in C$ 和 $\theta\ge0$ 都有 $\theta x\in C$，我们称集合 $C$ 是**锥**。如果集合 $C$ 是锥，并且是凸集，则称 $C$ 为**凸锥**，即对于任意 $x_1,x_2\in C$ 和 $\theta_1,\theta_2\ge0$，都有
$$
\theta_1x_1+\theta_2x_2\in C
$$
在 $\mathbb{R}^2$ 上，具有此类形式的点构成扇形，这个扇形以0为顶点，边通过 $x_1$ 和 $x_2$，如图所示。

![Screenshot from 2020-10-22 13-44-13.png](https://i.loli.net/2020/10/22/ZrOeg8JMIKyRt9F.png)

具有 $\theta_1x_1+\cdots+\theta_kx_k,\theta_1,\cdots,\theta_k\ge 0$ 形式的点称为 $x_1,\cdots,x_k$ 的**锥组合**。 $C$ 是一个凸锥等价于 $\forall x_1,\cdots,x_k\in C$， $x_1,\cdots,x_k$ 的锥组合仍在 $C$ 中。与凸组合类似，锥组合的概念可以扩展到无穷级数和积分中。

集合 $C$ 的**锥包**是 $C$ 中元素的所有锥组合的集合，即
$$
\{\theta_1x_1+\cdots+\theta_kx_k|x_i\in C,\theta_i\ge0,\ i=1,\cdots,k \}
$$
锥包是包含 $C$ 的最小的凸锥。

![Screenshot from 2020-10-22 13-50-44.png](https://i.loli.net/2020/10/22/zklhPxyoHLrmsU1.png)

# 重要的凸集

+ 空集 $\varnothing$ 、单点集 $\{x_0\}$ 、全空间 $\mathbb{R}^n$ 都是 $\mathbb{R}^n$ 的仿射子集
+ 任意直线是仿射的；通过原点的直线则是子空间，也是凸锥
+ 一条线段是凸的，但不是仿射的
+ 一条射线是凸的，但不是仿射的；以原点为端点的射线则是凸锥
+ 任意子空间是仿射的、凸锥

## 超平面与半空间

**超平面**是具有下面形式的集合
$$
\{\pmb x|\pmb a^{\rm T}\pmb x=b \}
$$
其中 $\pmb a\in\mathbb{R}^n,a\neq 0,b\in \mathbb{R}$。解析上，超平面是关于 $\pmb x$ 的线性方程的解空间；几何上，超平面可以解释为与给定向量 $\pmb a$ 的内积为常数的点的集合，或者法线方向为 $\pmb a$ 的超平面（常数 $b\in\mathbb{R}$ 决定了这个超平面从原点的偏移）。

一个超平面将 $\mathbb{R}^n$ 划分为两个**半空间**，半空间是具有下列形式的集合
$$
\{\pmb x|\pmb a^{\rm T}\pmb x\le b \}
$$
其中 $\pmb a\neq \pmb 0$，即线性不等式的解空间。半空间是凸的，但不是仿射的。半空间的边界是超平面 $\{\pmb x|\pmb a^{\rm T}\pmb x=b \}$，内部是 $\{\pmb x|\pmb a^{\rm T}\pmb x< b \}$，称为**开半空间**。

## Euclid球和椭球

$\mathbb{R}^n$ 上的**空间Euclid球**（简称为**球**）具有下面的形式
$$
B(\pmb x_c,r)=\{\pmb x|\|\pmb x-\pmb x_c\|_2\le r \}=\{\pmb x|(\pmb x-\pmb x_c)^{\rm T}(\pmb x-\pmb x_c)\le r^2 \}
$$
其中 $r>0$， $\|\cdot\|_2$ 表示Euclid范数。向量 $\pmb x_c$ 称为**球心**，标量 $r$ 称为**半径**。球的另一个常见表达式为
$$
B(\pmb x_c,r)=\{\pmb x_c+r\pmb u|\|\pmb u\|_2\le 1 \}
$$
球是凸集，证明略。

一类相关的凸集是**椭球**，具有如下形式
$$
\mathcal{E}=\{\pmb x|(\pmb x-\pmb x_c)^{\rm T}P^{-1}(\pmb x-\pmb x_c) \le 1 \}
$$
其中 $P=P^{\rm T}> 0$，即 $P$ 是对称正定矩阵。向量 $\pmb x_c$ 称为椭球的**中心**，矩阵 $P$ 决定了椭球从 $x_c$ 向各个方向扩展的幅度。 $\mathcal{E}$ 的半轴长度由 $\sqrt{\lambda_i}$ 给出，其中 $\lambda_i$ 为 $P$ 的特征值。椭球的另一个常见表达式为
$$
\mathcal{E}=\{\pmb x_c+A\pmb u| \|\pmb u\|_2\le 1  \}
$$
其中 $A$ 是非奇异的方阵。在此类表达形式中，我们可以不失一般性地假设 $A$ 对称正定。当矩阵 $A$ 为对称半正定矩阵，但奇异时，集合 $\mathcal{E}$ 称为**退化的椭球**，其仿射维数等于 $A$ 的秩，退化的椭球也是凸集。

## 范数球和范数锥

设 $\|\cdot\|$ 是 $\mathbb{R}^n$ 上的范数。由范数的一次齐次性和三角不等式可知，以 $r$ 为半径， $\pmb x_c$ 为球心的**范数球** $\{\pmb x|\|\pmb x-\pmb x_c\|\le r\}$ 是凸的。**范数锥**
$$
C=\{(\pmb x,t)|\|\pmb x\|\le t,\ t\ge 0 \}\sube \mathbb{R}^{n+1}
$$
是一个凸锥。

@**二阶锥**是由Euclid范数定义的范数锥，即
$$
C=\{(\pmb x,t)\in\mathbb{R}^{n+1}| \|\pmb x\|_2\le t \}
$$
二阶锥也称为**Lorentz锥**或**冰激凌锥**。下图显示了 $\mathbb{R}^3$ 上的一个二阶锥。

![Screenshot from 2020-10-22 14-52-13.png](https://i.loli.net/2020/10/22/n8BEqswfkjtKh15.png)

## 多面体

**多面体**被定义为有限个线性等式和不等式的解集，
$$
\mathcal{P}=\{\pmb x|\pmb a_i^{\rm T}\pmb x\le b_i,i=1,\cdots,m,\ \pmb c_j^{\rm T}\pmb x= d_j,j=1,\cdots,p \}
$$
因此，多面体是有限个半空间和超平面的交集。仿射集合（例如子空间、超平面、直线）、射线、线段和半空间都是多面体。显而易见，多面体是凸集。有界的多面体有时也称为**多胞形**，但也有一些作者反过来使用这两个概念。下图显示了一个由五个半空间的交集定义的多面体。

![Screenshot from 2020-10-22 15-00-30.png](https://i.loli.net/2020/10/22/AhSeufRb9Uan5j7.png)

多面体可以使用更紧凑的表达式表示
$$
\mathcal{P}=\{\pmb x|A\pmb x\le \pmb b,C\pmb x=\pmb d \}
$$
其中
$$
A=\begin{bmatrix} \pmb a_1^{\rm T}\\\vdots\\\pmb a_m^{\rm T}
\end{bmatrix},\quad
C=\begin{bmatrix} \pmb c_1^{\rm T}\\\vdots\\\pmb c_p^{\rm T}
\end{bmatrix}
$$

@**非负象限**是具有非负分量的点的集合，即
$$
\mathbb{R}_*^n=\{\pmb x\in\mathbb{R}^n|x_i\ge 0,i=1,\cdots,n \}=\{\pmb x\in\mathbb{R}^n|\pmb x\ge \pmb 0 \}
$$
非负象限既是多面体也是锥，称为**多面体锥**。

### 单纯形

设 $k+1$ 个点 $\pmb v_0,\cdots,\pmb v_k\in \mathbb{R}^n$ **仿射无关**，即 $\pmb v_1-\pmb v_0,\cdots,\pmb v_k-\pmb v_0$ **线性无关**，那么这些点决定了一个**单纯形**
$$
C=\bold{conv}\{\pmb v_0,\cdots,\pmb v_k\}=\{\theta_0\pmb v_0+\cdots+\theta_k\pmb v_k|\pmb \theta\ge 0,\pmb 1^{\rm T}\pmb \theta=1 \}
$$
这个单纯形的仿射维数为 $k$，因此也称为 $\mathbb{R}^n$ 空间的 $k$ 维单纯形。

@常见的单纯形：1维单纯形是一条线段，2维单纯形是一个三角形，3维单纯形是一个四面体

**单位单纯形**是由零向量和单位向量 $0,e_1,\cdots,e_n\in\mathbb{R}^n$ 决定的 $n$ 维单纯形，它可以表示为满足下列条件的向量的集合
$$
\pmb x\ge \pmb 0,\quad \pmb 1^{\rm T}\pmb x\le1
$$
**概率单纯形**是由单位向量 $e_1,\cdots,e_n\in\mathbb{R}^n$ 决定的 $n-1$ 维单纯形，它是满足下列条件的向量的集合
$$
\pmb x\ge \pmb 0,\quad \pmb 1^{\rm T}\pmb x=1
$$
概率单纯形中的向量对应有 $n$ 个取值的概率分布。

考虑用多面体来描述单纯形。定义 $\pmb y=(\theta_1,\cdots,\theta_k)$ 和
$$
B=[\pmb v_1-\pmb v_0\ \ \cdots\ \ \pmb v_k-\pmb v_0]\in \mathbb{R}^{n\times k}
$$
$\pmb x\in C$ 的充要条件是 $\exist \pmb y \ge \pmb 0,\pmb 1^{\rm T}\pmb y\le 1$，
$$
\pmb x=\pmb v_0+B\pmb y=(1-\theta_1-\cdots-\theta_k)\pmb v_0+\theta_1\pmb v_1+\cdots+\theta_k\pmb v_k
$$
注意到 $\pmb v_0,\cdots,\pmb v_k\in \mathbb{R}^n$ 仿射无关意味着矩阵 $B$ 的秩为 $k$，因此存在非奇异矩阵 $A\in \mathbb{R}^{n\times n}$ 使得
$$
AB=\begin{bmatrix}A_1\\A_2
\end{bmatrix}B=\begin{bmatrix}I\\0
\end{bmatrix}
$$

> 相当于做初等变换

用 $A$ 左乘 $\pmb x=\pmb v_0+B\pmb y$，得到
$$
A_1\pmb x=A_1\pmb v_0+\pmb y,\quad A_2\pmb x=A_2\pmb v_0
$$
这样我们得到 $\pmb x\in C$ 的充要条件
$$
\quad A_2\pmb x=A_2\pmb v_0,\quad A_1\pmb x\ge A_1\pmb v_0,\quad \pmb 1^{\rm T} A_1\pmb x\le 1+\pmb 1^{\rm T} A_1\pmb v_0
$$
这些关于 $\pmb x$ 的线性等式和不等式描述了一个多面体。

### 多面体的凸包描述

有限集合 $\{\pmb v_1,\cdots,\pmb v_k\}$ 的凸包是
$$
\bold{conv}\{\pmb v_1,\cdots,\pmb v_k\}=\{\theta_1\pmb v_1+\cdots+\theta_k\pmb v_k|\pmb \theta\ge 0,\pmb 1^{\rm T}\pmb \theta=1 \}
$$
它是一个有界的多面体，但无法简单地用线性等式和不等式的集合将其表示。

凸包表达式的一个扩展表示为
$$
\{\theta_1\pmb v_1+\cdots+\theta_k\pmb v_k|\theta_1+\cdots+\theta_m=1, \theta_i\ge 0,i=1,\cdots,k,\ m\le k \}
$$
这里只要求前 $m$ 个系数之和为1，其余系数可以取任意非负整数。可以证明上述集合与多面体一一对应。

考虑多面体采用线性表述和凸包表述，一个简单的例子是定义在 $\ell_{\infty}$ 范数空间 $\mathbb{R}^n$ 上的单位球
$$
C=\{\pmb x||x_i|\le 1,i=1,\cdots,n \}
$$
$C$ 的线性表述由 $2n$ 个线性不等式 $\pm \pmb e_i^{\rm T}\pmb x\le 1$ 表示，其中 $\pmb e_i$ 是第 $i$ 维的单位向量；而 $C$ 的凸包表述需要至少 $2^n$ 个点
$$
C=\bold{conv}\{\pmb v_1,\cdots,\pmb v_{2^n}\}
$$
其中 $\pmb v_1,\cdots,\pmb v_{2^n}$ 是以1和-1为分量的全部向量。

## 半正定锥

我们用 $\mathbb{S}^n$ 表示对称 $n\times n$ 矩阵的集合
$$
\mathbb{S}^n=\{X\in \mathbb{R}^{n\times n}|X=X^{\rm T} \}
$$
这是一个维数为 $n(n+1)/2$ 的向量空间；用 $\mathbb{S}_*^n$ 表示对称半正定矩阵的集合
$$
\mathbb{S}_*^n=\{X\in \mathbb{S} ^{n}|X\ge 0 \}
$$
用 $\mathbb{S}_+^n$ 表示对称正定矩阵的集合
$$
\mathbb{S}_+^n=\{X\in \mathbb{S} ^{n}|X> 0 \}
$$
集合 $\mathbb{S}_*^n$ 是一个凸锥，即 $\forall \theta_1,\theta_2\ge 0,\forall A,B\in \mathbb{S}_*^n$， $\theta_1A+\theta_2B\in \mathbb{S}_*^n$。证明如下： $\forall \theta_1,\theta_2\ge 0,\forall A,B\in \mathbb{S}_*^n$， $A\ge 0,B\ge 0$， $\forall x\in \mathbb{R}^{n} $，
$$
(\theta_1A+\theta_2B)^{\rm T}=\theta_1A^{\rm T}+\theta_2B^{\rm T}=\theta_1A+\theta_2B \\
\pmb x^{\rm T}(\theta_1A+\theta_2B)\pmb x=\theta_1\pmb x^{\rm T}A\pmb x+\theta_2\pmb x^{\rm T}B\pmb x\ge 0
$$

@对于凸锥 $\mathbb{S}_*^2$ 有

$$
X=\begin{bmatrix}x&y\\y&z
\end{bmatrix}\in\mathbb{S}_*^2\Leftrightarrow  x\ge 0,z\ge 0,xz\ge y^2
$$

如图显示了这个凸锥的边界，按 $(x,y,z)$ 表示在 $\mathbb{R}^3$ 上。

![Screenshot from 2020-10-22 16-34-16.png](https://i.loli.net/2020/10/22/n6QHG9teV3YxgaF.png)

# 保凸运算

利用保凸运算可以从凸集构造出其它凸集。保凸运算与前面的凸集实例一起构成了凸集的演算，可以用来确定或构建凸集。

## 交集

交集运算是保凸的：如果 $S_1,S_2$ 是凸集，那么 $S_1\cap S_2$ 也是凸集。这个性质可以扩展到无穷个集合的交集：如果 $\forall \alpha\in \mathcal{A}$， $S_\alpha$ 是凸集，那么 $\bigcap_{\alpha\in A}S_\alpha$ 也是凸集。

@多面体是半空间和超平面的交集，因而是凸集。

@半正定锥 $\mathbb{S}_*^2$ 可以表示为
$$
\bigcap_{\pmb z\neq \pmb 0} \{X\in \mathbb{S} ^{n}|\pmb z^{\rm T}X\pmb z\ge 0 \}
$$
$\forall\pmb z\neq\pmb 0$， $\pmb z^{\rm T}X\pmb z$ 是关于 $X$ 的线性函数，因此集合
$$
\{X\in \mathbb{S} ^{n}|\pmb z^{\rm T}X\pmb z\ge 0 \}
$$
实际上就是 $\mathbb{S} ^{n}$ 的半空间。由此可见，半正定锥是无穷个半空间的交集，因此是凸的。

上面这些例子中，我们通过将集合表示为有穷或无穷多个半空间的交集来表明集合的凸性。反过来我们也将看到，每一个闭的凸集 $S$ 是（通常是无穷多个）半空间的交集，事实上，闭凸集 $S$ 是所有包含它的半空间的交集。

## 仿射函数

函数 $f:\mathbb{R}^n\to \mathbb{R}^m$ 是**仿射的**，如果它是一个线性函数和一个常数的和，即具有 $f(\pmb x)=A\pmb x+\pmb b$ 的形式，其中 $A\in \mathbb{R}^{m\times n},b\in\mathbb{R}^m$。假设 $S\sube \mathbb{R}^n$ 是凸集，并且 $f:\mathbb{R}^n\to \mathbb{R}^m$ 是仿射函数，那么 $S$ 在 $f$ 下的象
$$
f(S)=\{f(\pmb x)|\pmb x\in S \}
$$
是凸的。类似地，如果 $g:\mathbb{R}^k\to \mathbb{R}^n$ 是仿射函数，那么 $S$ 在 $g$ 下的原象
$$
g^{-1}(S)=\{\pmb x|g(\pmb x)\in S \}
$$
是凸的。

@**伸缩**和**平移**：如果 $S\sube \mathbb{R}^n$ 是凸集， $\alpha\in \mathbb{R},a\in \mathbb{R}^n$，那么集合 $\alpha S$ 和 $S+a$ 是凸的。

@**投影**：如果 $S\sube \mathbb{R}^m\times \mathbb{R}^n$ 是凸集，那么
$$
T=\{x_1\in \mathbb{R}^m|(x_1,x_2)\in S \}
$$
是凸集。

@**和**：如果 $S_1,S_2$ 是凸集，那么
$$
S_1+S_2=\{x_1+x_2|x_1\in S_1,x_2\in S_2 \}
$$
是凸集。

@**笛卡尔积**：如果 $S_1,S_2$ 是凸集，那么
$$
S_1\times S_2=\{(x_1,x_2)|x_1\in S_1,x_2\in S_2 \}
$$
是凸集。

@**部分和**：如果 $S_1,S_2\sube \mathbb{R}^m\times \mathbb{R}^n$ 是凸集，那么
$$
S=\{(x,y_1+y_2)|(x,y_1)\in S_1,(x,y_2)\in S_2 \}
$$
@**多面体**：多面体 $\{\pmb x|A\pmb x\le \pmb b,C\pmb x=\pmb d \}$ 可以表示为非负象限和原点的笛卡尔积在仿射函数 $f(\pmb x)=(\pmb b-A\pmb x,\pmb d-C\pmb x)$ 下的原象
$$
\{\pmb x|A\pmb x\le \pmb b,C\pmb x=\pmb d \}=\{\pmb x|f(\pmb x)\in \mathbb{R}_*^m\times \{\pmb 0\} \}
$$
@**线性矩阵不等式(LMI)**：条件
$$
A(\pmb x)=x_1A_1+\cdots+x_nA_n\le B
$$
称为关于 $x$ 的线性矩阵不等式，其中 $B,A_i\in \mathbb{S} ^{m}$。

LMI的解 $\{\pmb x|A(\pmb x)\le B \}$ 是凸集。事实上，它是半正定锥 $\mathbb{S}_*^m$ 在仿射函数 $f(\pmb x)=B-A(\pmb x)$ 下的原象。

@**椭球**：椭球
$$
\mathcal{E}=\{\pmb x|(\pmb x-\pmb x_c)^{\rm T}P^{-1}(\pmb x-\pmb x_c) \le 1 \}
$$
是单位球 $\{\pmb u|\|\pmb u\|_2\le 1\}$ 在仿射函数 $f(\pmb u)=P^{1/2}\pmb u+\pmb x_c$ 下的象，其中 $P\in \mathbb{S}_+^2$。

## 线性分式和透视函数

本节讨论一类称为**线性分式**的函数，它比仿射函数更普遍，而且仍然保凸。

### 透视函数

我们定义 $P:\mathbb{R}^{n+1}\to \mathbb{R}^{n},P(z,t)=z/t$ 为**透视函数**，其定义域为 $\bold{dom}P=\mathbb{R}^{n}\times \mathbb{R}_+$。透视函数对向量进行伸缩，

……

