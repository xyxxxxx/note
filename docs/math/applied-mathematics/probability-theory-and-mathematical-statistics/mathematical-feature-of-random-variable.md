# 随机变量的数字特征

在上一章中，我们较为仔细地讨论了随机变量的概率分布，这种分布是随机变量的概率性质最完整的刻画。随机变量的数字特征，则是某些由随机变量的分布所决定的常数，它刻画了随机变量（或者说其分布）的某一方面的性质。

## 数学期望（均值）与中位数

### 数学期望的定义

先考虑最简单的一种情况：

**定义 1.1** 设随机变量 $X$ 只取有限个可能值 $a_1,\cdots,a_m$，其概率分布为 $P(X=a_i)=p_i,i=1,\cdots,m$，则 $X$ 的**数学期望**，记作 $E(X)$ 或 $EX$，定义为

$$
\begin{equation}
\tag{1.1}
E(X)=a_1p_1+a_2p_2+\cdots+a_mp_m
\end{equation}
$$

数学期望也常称为**均值**，即“随机变量取值的平均值”之意，当然这个平均是以概率为权的加权平均。

利用概率的统计定义，容易给均值这个名词一个自然的解释。假定将试验重复 $N$ 次，每次将 $X$ 的取值记录下来，设在这 $N$ 次中，有 $N_1$ 次取 $a_1$，$N_2$ 次取 $a_2$，…，$N_m$ 次取 $a_m$，则这 $N$ 次试验中 $X$ 的取值之和为 $a_1N_1+a_2N_2+\cdots+a_mN_m$，而平均每次试验中 $X$ 的取值，记作 $\overline{X}$，等于

$$
\overline{X}=(a_1N_1+a_2N_2+\cdots+a_mN_m)/N=a_1(N_1/N)+a_2(N_2/N)+\cdots+a_m(N_m/N)
$$

$N_i/N$ 是事件 $\{X=a_i\}$ 在这 $n$ 次试验中的频率。按照概率的统计定义，当 $N$ 很大时，$N_i/N$ 应很接近 $p_i$。因此，$\overline{X}$ 应接近于 (1.1) 右侧的值。也就是说，$X$ 的数学期望 $E(X)$ 正是在大量次数的试验之下，$X$ 在各次试验中取值的平均。

很自然地，如果 $X$ 为离散型变量，取无穷个值 $a_1,a_2,\cdot$，而概率分布为 $P(X=a_i)=p_i,i=1,2,\cdots$，因此我们仿照 (1.1)，将 $X$ 的数学期望 $E(X)$ 定义为级数之和：

$$
\begin{equation}
\tag{1.2}
E(X)=\sum_{i=1}^\infty a_ip_i
\end{equation}
$$

当然级数必须收敛，实际上我们还要求这个级数绝对收敛：

**定义 1.2** 如果

$$
\begin{equation}
\tag{1.3}
\sum_{i=1}^\infty |a_i|p_i<\infty
\end{equation}
$$

则 (1.2) 成立，即 (1.2) 右侧的级数之和为 $X$ 的数学期望。

!!! note "说明"
    根据[黎曼级数定理](https://zh.wikipedia.org/zh-hans/%E9%BB%8E%E6%9B%BC%E7%BA%A7%E6%95%B0%E5%AE%9A%E7%90%86)，条件收敛的无穷级数在重新排列各项后可以收敛到任何一个给定的值或者发散，只有绝对收敛的无穷级数才可以不论如何重新排列各项都不改变收敛值。$E(X)$ 作为刻画 $X$ 的某种特性的值，有其客观意义，不应与各项的人为排列次序有关。

对于连续型随机变量，以积分代替求和，以得到数学期望的定义：

**定义 1.3** 设 $X$ 有概率密度函数 $f(x)$。如果

$$
\begin{equation}
\tag{1.4}
\int_{-\infty}^\infty |x|f(x){\rm d}x<\infty
\end{equation}
$$

则称

$$
\begin{equation}
\tag{1.5}
E(X)=\int_{\infty}^\infty xf(x){\rm d}x
\end{equation}
$$

**例 1.1** 设 $X$ 服从泊松分布 $P(\lambda)$，则

$$
E(X)=\sum_{i=0}^\infty i\frac{\lambda^i}{i!}e^{-\lambda}=\lambda e^{-\lambda}\sum_{i=1}^\infty\frac{\lambda^{i-1}}{(i-1)!}=\lambda e^{-\lambda}\sum_{i=0}^\infty \frac{\lambda^i}{i!}=\lambda e^{-\lambda}e^\lambda=\lambda
$$

**例 1.2** 设 $X$ 服从负二项分布，则

$$
E(X)=p^r\sum_{i=0}^\infty i\begin{pmatrix}i+r-1\\r-1\end{pmatrix}(1-p)^i
$$

为求这个和，我们需要用到负指数二项展开式：

$$
(1-x)^{-r}=\sum_{i=0}^\infty\begin{pmatrix}i+r-1\\r-1\end{pmatrix}x^i
$$

两边对 $x$ 求导，得到

$$
r(1-x)^{-r-1}=\sum_{i=0}^\infty i\begin{pmatrix}i+r-1\\r-1\end{pmatrix}x^{i-1}
$$

在上式中令 $x=1-p$，然后两边同乘 $1-p$ 得到

$$
\sum_{i=0}^\infty i\begin{pmatrix}i+r-1\\r-1\end{pmatrix}(1-p)^i=rp^{-r-1}(1-p)
$$

故

$$
E(X)=p^r\cdot rp^{-r-1}(1-p)=r(1-p)/p
$$

当 $r=1$ 时，得到几何分布的期望为 $(1-p)/p$。

**例 1.3** 若 $X$ 服从 $[a,b]$ 区间的均匀分布，则

$$
E(X)=\frac{1}{b-a}\int_a^bx{\rm d}x=\frac{a+b}{2}
$$

**例 1.4** 若 $X$ 服从指数分布，则

$$
E(X)=\lambda\int_0^\infty xe^{-\lambda x}{\rm d}x=\lambda^{-1}\int_0^\infty te^{-t}{\rm d}t=\lambda^{-1}\Gamma(2)=\lambda^{-1}
$$

**例 1.5** 若 $X$ 服从正态分布 $N(\mu,\sigma^2)$，则

$$
E(X)=\frac{1}{\sqrt{2\pi}\sigma}\int_{-\infty}^\infty x\exp(-\frac{(x-\mu)^2}{2\sigma^2}){\rm d}x
$$

作变数代换 $x=\mu+\sigma t$，化为

$$
\begin{align}
E(X)&=\frac{1}{\sqrt{2\pi}}\int_{-\infty}^\infty(\mu+\sigma t)e^{-t^2/2}{\rm d}t\\
&=\mu\frac{1}{\sqrt{2\pi}}\int_{-\infty}^\infty e^{-t^2/2}{\rm d}t+\sigma\frac{1}{\sqrt{2\pi}}\int_{-\infty}^\infty te^{-t^2/2}{\rm d}t
\end{align}
$$

上式右侧第一项为 $\mu$，第二项为 0，因此

$$
E(X)=\mu
$$

因为数学期望是由随机变量的分布完全决定的，因此我们可以并且常常说某个分布 $F$ 的期望是多少，某个密度 $f$ 的期望是多少等。

### 数学期望的性质

**定理 1.1** 若干个随机变量之和的期望，等于各变量的期望之和，即

$$
E(X_1+X_2+\cdots+X_n)=E(X_1)+E(X_2)+\cdots+E(X_n)
$$

当然，这里需要假定各变量 $X_i$ 的期望都存在。

证明：先考虑 $n=2$ 的情况。若 $X_1,X_2$ 为离散型，分别以 $a_1,a_2,\cdots$ 和 $b_1,b_2,\cdots$ 记 $X_1$ 和 $X_2$ 的一切可能值，而记其联合分布为

$$
\begin{equation}
\tag{1.13}
P(X_1=a_i,X_2=b_j)=p_{ij},i,j=1,2,\cdots
\end{equation}
$$

当 $X_1=a_i,X_2=b_j$ 时，有 $X_1+X_2=a_i+b_j$，按照定义 1.1 和定义 1.2，有

$$
\begin{align}
E(X_1+X_2)&=\sum_{i,j}(a_i+b_j)p_{ij}=\sum_{i,j}a_ip_{ij}+\sum_{i,j}b_jp_{ij}\\
&=\sum_{i}(a_i\sum_jp_{ij})+\sum_{j}(b_j\sum_ip_{ij})=\sum_{i}a_iP(X_1=a_i)+\sum_{j}b_jP(X_1=a_i)\\
&=E(X_1)+E(X_2)
\end{align}
$$

这证明了所要的结果。

若 $(X_1,X_2)$ 为连续型，记其联合密度为 $f(x_1,x_2)$，则 $X_1+X_2$ 的密度函数为 $l(y)=\int_{-\infty}^\infty f(x,y-x){\rm d}x$，故按照定义 1.3，有

$$
\begin{align}
E(X_1+X_2)&=\int_{-\infty}^\infty yl(y){\rm d}y\\
&=\int_{-\infty}^\infty\int_{-\infty}^\infty yf(x,y-x){\rm d}x{\rm d}y\\
&=\int_{-\infty}^\infty {\rm d}x\int_{-\infty}^\infty yf(x,y-x){\rm d}y
\end{align}
$$

对于右侧积分作变数代换 $y=x+t$，得

$$
\begin{align}
E(X_1+X_2)&=\int_{-\infty}^\infty {\rm d}x\int_{-\infty}^\infty (x+t)f(x,t){\rm d}t\\
&=\int_{-\infty}^\infty x{\rm d}x\int_{-\infty}^\infty f(x,t){\rm d}t+\int_{-\infty}^\infty t{\rm d}t\int_{-\infty}^\infty f(x,t){\rm d}x
\end{align}
$$

左边这一项为 $X_1$ 的密度函数，右边这一项为 $X_2$ 的密度函数，于是证明了所要的结果。

一般情况可用归纳的方式得到。例如，记 $Y=X_1+X_2$，有

$$
E(X_1+X_2+X_3)=E(Y+X_3)=E(Y)+E(X_3)=E(X_1)+E(X_2)+E(X_3)
$$

等等。定理 1.1 证毕。

**定理 1.2** 若干个独立随机变量之积的期望，等于各变量的期望之积：

$$
E(X_1X_2\cdots X_n)=E(X_1)E(X_2)\cdots E(X_n)
$$

当然，这里也需要假定各变量 $X_i$ 的期望都存在。

证明：与定理 1.1 类似，只需要对 $n=2$ 的情况证明即可。先设 $X_1,X_2$ 都为离散型，其分布为 (1.13)。由独立性假定知 $p_{ij}=P(X_1=a_i)P(X_2=b_j)$。

因为当 $X_1=a_i,X_2=b_j$ 时有 $X_1X_2=a_ib_j$，故

$$
\begin{align}
E(X_1X_2)&=\sum_{i,j}a_ib_jp_{ij}\\
&=\sum_{i,j}a_ib_jP(X_1=a_i)P(X_2=b_j)\\
&=\sum_i a_iP(X_1=a_i)\sum_j b_jP(X_2=b_j)\\
&=E(X_1)E(X_2)
\end{align}
$$

这证明了所要的结果。若 $(X_1,X_2)$ 为连续型，则因独立性，其联合密度 $f(x_1,x_2)$ 等于各分量密度 $f_1(x_1)$ 和 $f_2(x_2)$ 之积，故

$$
\begin{align}
E(X_1X_2)&=\iint_{-\infty}^\infty x_1x_2f(x_1,x_2){\rm d}x_1{\rm d}x_2\\
&=\int_{-\infty}^\infty x_1f(x_1){\rm d}x_1\int_{-\infty}^\infty x_2f(x_2){\rm d}x_2\\
&=E(X_1)E(X_2)
\end{align}
$$

注意到这一段证明使用到公式

$$
E(X_1X_2)=\iint_{-\infty}^\infty x_1x_2f(x_1,x_2){\rm d}x_1{\rm d}x_2
$$

而这一公式并非从期望的定义直接得到，它也需要证明，但此处略去。

读者也许还会问：在以上两个定理中，如果一部分变量为离散型，一部分为连续型，结果如何？答案是结论仍成立。对于乘积的情况，由于有独立假定，证明不难；对于和的情况则要用到高等概率论，这些都不在这里细讲了。

**定理 1.3** 设随机变量 $X$ 为离散型，有分布 $P(X=a_i)=p_i,i=1,2,\cdots$，或者为连续型，有概率密度函数 $f(x)$，则

$$
\begin{equation}
\tag{1.17}
E(g(X))=\sum_i g(a_i)p_i,当 \sum_i |g(a_i)|p_i<\infty
\end{equation}
$$

或

$$
\begin{equation}
\tag{1.18}
E(g(X))=\int_{-\infty}^\infty g(x)f(x){\rm d}x,当 \int_{-\infty}^\infty ｜g(x)｜f(x){\rm d}x<\infty
\end{equation}
$$

这个定理的实质在于：为了计算 $X$ 的某一函数 $g(X)$ 的期望，不必先计算 $g(X)$ 的密度函数，而可以直接从 $X$ 的分布出发。这当然大大方便了计算，因为当 $g$ 较为复杂时，$g(x)$ 的密度很难求出。

证明略。

**推论** 若 $C$ 为常数，则

$$
\begin{equation}
\tag{1.19}
E(CX)=CE(X)
\end{equation}
$$

**例 1.6** 设 $X$ 服从二项分布 $B(n,p)$，求 $E(X)$。

此例不难由定义 1.1 直接计算，但作如下考虑更加简单：因为 $X$ 是 $n$ 次独立试验中某事件 $A$ 发生的次数，且在每次试验中 $A$ 发生的概率为 $p$，故如引入随机变量 $X_1,\cdots,X_n$，其中

$$
X_i=\begin{cases}1,&若在第i次试验中事件A发生\\
0,&若在第i次试验中事件A不发生\end{cases}
$$

则 $X_1,\cdots,X_n$ 独立，且

$$
X=X_1+\cdots+X_n
$$

按照定理 1.1，有 $E(X)=E(X_1)+\cdots+E(X_n)$。根据 $X_i$ 的定义有 $E(X_i)=1×p+0×(1-p)=p$，由此得到

$$
E(X)=np
$$

**例 1.8** 计算“统计三大分布”的期望值。

对自由度 $n$ 的卡方分布，直接用其密度函数的形式、$\Gamma$ 函数的公式以及数学期望的定义，不难算出其期望为 $n$。更加简单的方法是将 $X$ 表示 $X_1^2+\cdots+X_n^2$ 独立且各服从标准正态分布 $N(0,1)$。按照定理 1.3，有

$$
E(X_i^2)=\frac{1}{\sqrt{2\pi}}\int_{-\infty}^\infty x^2e^{-x^2/2}{\rm d}x=\frac{2}{\sqrt{2\pi}}\int_0^\infty x^2e^{-x^2/2}{\rm d}x
$$

将 $e^{-x^2/2}x^2{\rm d}x$ 写作 $-x{\rm d}(e^{-x^2/2})$，用分部积分得到

$$
\begin{align}
\int_0^\infty x^2e^{-x^2/2}{\rm d}x&=(-xe^{-x^2/2})\rvert_0^\infty-\int_0^\infty e^{-x^2/2}{\rm d}(-x)\\
&=\int_0^\infty e^{-x^2/2}{\rm d}x\\
&=\frac{1}{2}\int_{-\infty}^\infty e^{-x^2/2}{\rm d}x\\
&=\frac{\sqrt{2\pi}}2
\end{align}
$$

于是 $E(X_i^2)=1$，而 $E(X)=E(X_1^2)+\cdots+E(X_n^2)=n$。

对自由度 $n$ 的 $t$ 分布，由于其密度函数关于 0 对称，易见其期望为 0，但条件是自由度 $n$ 必须大于 1，因为

$$
\int_{-\infty}^\infty|x|(1+x^2/n)^{-\frac{n+1}{2}}{\rm d}x
$$

只有当 $n>1$ 时有限。

对自由度为 $m,n$ 的 F 分布，

### 条件数学期望（条件均值）

与条件分布的定义相似，随机变量 $Y$ 的条件数学期望，就是它在给定的某种附加条件之下的数学期望。对统计学而言，最重要的情况是：在给定了其他随机变量 $X,Z,\cdots$ 等的值 $x,z,\cdots$ 等的条件下，$Y$ 的条件期望，记为 $E(Y|X=x,Z=z,\cdots)$。以只有一个变量 $X$ 为例，就是 $E(Y|X=x)$，简记为 $E(Y|x)$。

如果知道了 $(X,Y)$ 的联合密度，则 $E(Y|x)$ 的定义就可以具体化为：先得到在给定 $X=x$ 之下 $Y$ 的条件密度函数 $f(y|x)$，再按照定义 1.3 算出

$$
E(Y|x)=\int_{-\infty}^\infty yf(y|x){\rm d}y
$$

如果说条件分布是变量 $X$ 与 $Y$ 的相依关系在概率上的完全刻画，那么条件期望则在一个很重要的方面刻画了两者的关系，它反映了随着 $X$ 的取值 $x$ 的变化，$Y$ 的平均的变化情况如何，而这常常是研究者所关心的主要内容。在统计学上，常把条件期望 $E(Y|x)$ 作为 $x$ 的函数，称为 $Y$ 对 $X$ 的**回归函数**，而**回归分析**，即关于回归函数的统计研究，构成了统计学的一个重要分支。

**例 1.9** 条件期望的一个最重要的例子是 $(X,Y)$ 服从二维正态分布 $N(a,b,\sigma_1^2,\sigma_2^2,\rho)$。根据前例的结果，在给定 $X=x$ 时 $Y$ 的条件分布为正态分布 $N(b+\rho\sigma_2\sigma_1^{-1}(x-a),\sigma_2^2(1-\rho^2))$，故有

$$
E(Y|x)=b+\rho\sigma_2\sigma_1^{-1}(x-a)
$$

它是 $x$ 的线性函数。如果 $\rho>0$，则 $E(Y|x)$ 随 $x$ 增加而增加，即 $Y$ 平均意义上有随 $X$ 的增加而增加的趋势，这就是我们之前提到的正相关的解释。如果 $\rho<0$，则为负相关；如果 $\rho=0$，$X$ 与 $Y$ 独立，$E(Y|x)$ 当然就与 $x$ 无关。

从条件数学期望的概念，可以得到求（无条件）数学期望的一个重要公式。这个公式与计算概率的全概率公式相当。回想全概率公式 $P(A)=\sum_iP(B_i)P(A|B_i)$，它可以理解为通过事件 $A$ 的条件概率 $P(A|B_i)$ 去计算（无条件）概率 $P(A)$。更确切地说，$P(A)$ 就是条件概率 $P(A|B_i)$ 的某种加权平均，权即为事件 $B_i$ 的概率。以此类推，变量 $Y$ 的（无条件）期望，应等于其条件期望 $E(Y|x)$ 对 $x$ 取加权平均，$x$ 的权与变量 $X$ 在 $x$ 点的概率密度 $f_1(x)$ 成比例，即

$$
\begin{equation}
\tag{1.27}
E(Y)=\int_{-\infty}^\infty E(Y|x)f_1(x){\rm d}x
\end{equation}
$$

此式很容易证明：以 $f(x,y)$ 记 $(X,Y)$ 的联合密度函数，则 $X,Y$ 的（边缘）密度函数分别为

$$
f_1(x)=\int_{-\infty}^\infty f(x,y){\rm d}y,f_2(x)=\int_{-\infty}^\infty f(x,y){\rm d}x
$$

按照定义，$E(Y)=\int_{-\infty}^\infty yf_2(y){\rm d}y$，可写为

$$
E(Y)=\int_{-\infty}^\infty\int_{-\infty}^\infty yf(x,y){\rm d}x{\rm d}y=\int_{-\infty}^\infty(\int_{-\infty}^\infty yf(x,y){\rm d}y){\rm d}x
$$

由于 $E(Y|x)=\int_{-\infty}^\infty yf(x,y){\rm d}y/f_1(x)$，有 $\int_{-\infty}^\infty yf(x,y){\rm d}y=E(Y|x)f_1(x)$，从而上式转换为 (1.27)。

(1.27) 可以给出另一种写法，$E(Y|x)$ 是 $x$ 的函数，记作 $g(x)$，则 (1.27) 变为

$$
E(Y)=\int_{-\infty}^\infty g(x)f_1(x){\rm d}x
$$

又根据 (1.18)，上式右侧就是 $E(g(X))$。根据 $g(x)$ 的定义，$g(X)$ 是 $E(Y|x)\rvert_{x=X}$，简写为 $E(Y|X)$，于是由 (1.28) 得

$$
\begin{equation}
\tag{1.29}
E(Y)=E(E(Y|X))
\end{equation}
$$

这个公式可以形象地叙述为：一个变量 $Y$ 的期望，等于其条件期望的期望。在求 $E(Y|X)$ 时，先设定 $X$ 为一定值 $x$，这样可算出 $E(Y|x)$，其为 $x$ 的函数，再把 $x$ 换成 $X$ 即得。

(1.29) 虽然可算作是概率论中一个比较高深的公式，但它的实际含义很简单：可以理解为一个“两步走”去计算期望的方法。因为在不少情况下，径直计算 $E(Y)$ 较困难，而在限定另一变量 $X$ 的值之后，计算条件期望 $E(Y|x)$ 则较容易。因此我们第一步算出 $E(Y|x)$，再借助 $X$ 的概率分布，通过 $E(Y|x)$ 算出 $E(Y)$。更直观一些，可以将求 $E(Y)$ 看作是在一个很大的范围求平均，限定 $X$ 的值是从这个很大的范围内界定了一个较小的部分，先对较小的这部分求平均，然后再对后者求平均。比如要求全校学生的平均身高，你可以先求出每个班的学生的平均身高，再对各班的平均值求一次平均。当然，在做后一次平均时，要考虑到各班人数的不同，因此应是以各班人数为权的加权平均，这个权的作用就相当于 (1.27) 中的 $f_1(x)$。

(1.29) 虽然来自 (1.27)，但因为其形式并不要求对 $X,Y$ 有特殊的假设，故可适用于更一般的情形。例如，$X$ 不必是一维的，如果 $X$ 是 $n$ 维随机向量 $(X_1,\cdots,X_n)$，有概率密度 $f(x_1,\cdots,x_n)$，则 (1.29) 有形式

$$
E(Y)=\int_{-\infty}^\infty\cdots\int_{-\infty}^\infty E(Y|x_1,\cdots,x_n)f(x_1,\cdots,x_n){\rm d}x_1\cdots{\rm d}x_n
$$

这里 $E(Y|x_1,\cdots,x_n)$ 就是在 $X_1=x_1,\cdots,X_n=x_n$ 的条件下，$Y$ 的条件期望。此外，$X,Y$ 都可以是离散型的。例如，设 $X$ 为一维离散型变量，有分布

$$
P(X=a_i)=p_i,i=1,2,\cdots
$$

则 (1.29) 有形式

$$
E(Y)=\sum_{i=1}^\infty p_iE(Y|a_i)
$$

### 中位数

**定义 1.4** 设连续型随机变量 $X$ 的分布函数为 $F(x)$，则满足条件

$$
\begin{equation}
\tag{1.32}
P(X\le m)=F(m)=1/2
\end{equation}
$$

的数 $m$ 称为 $X 或分布 $F$ 的中位数。

由于连续型变量取一个值的概率为 0，$P(X=m)=0$，由 (1.32) 知

$$
P(X\le m)=P(X<m)=P(X>m)=P(X\ge m)=1/2
$$

就是说，$m$ 这个点把 $X$ 的分布从概率上切成两半：$m$ 左边占一半，$m$ 右边也占一半，从概率上说，$m$ 这个点正好位于中央，这就是中位数名称的由来。

在实用上，中位数用得很多，特别有不少社会统计资料，常拿中位数来刻画某种量的代表性数值，有时它比数学期望更说明问题。例如，某社区的收入中位数告诉我们：有一半的人收人低于此值，另一半高于此值。我们直观上感觉到这个值对该社区的收入情况的确很具有代表性。和期望相比它的一个优点是：它受个别特大或特小值的影响很小，而期望则不然。举例而言，若该社区中有一人收人在百万元以上，则该社区的均值可能很高，而绝大多数人并不富裕，这个均值并不很有代表性。

从理论上说，中位数和期望相比还有一个优点，即它总存在，而期望则不是对任何随机变量都存在。

虽则中位数有这些优点，但在概率统计中，无论在理论和应用上，数学期望的重要性都超过中位数，其原因有以下两方面：

一是期望具有很多优良的性质，反映在前面的定理 1.1—1.3。这些性质使得在数学上处理均值很方便，例如 $E(X_1+X_2)=E(X_1)+E(X_2)$，这公式既简单又毫无条件（除了均值存在以外）。中位数则不然：$X_1+X_2$ 的中位数，与 $X_1,X_2$ 各自的中位数之间不存在简单的联系，这使中位数在数学上的处理很复杂且不方便。

二是中位数本身所固有的某些缺点。首先，中位数可以不唯一，例如考察下图的密度函数 $f$，它只在两个分开的区间 $(a,b)$ 和 $(c,d)$ 内不为 0，且在这两段区间上围成的面积都是 1/2，这时按中位数的定义 1.4，区间 $[b,c]$ 中任何一点 $m$ 都是中位数，它没有一个唯一的值。

![](https://s2.loli.net/2023/02/10/pkMlLmuZH65h8cJ.png)

另一个问题是：在 $X$ 为离散型的情况，虽然也可以定义中位数（其定义与定义 1.4 有所不同），但并不很理想，不一定完全符合“中位“这名词所应有的含义。

**例 1.10** 正态分布 $N(\mu,\sigma^2)$ 的中位数就是 $\mu$，这从 $N(\mu,\sigma^2)$ 的密度函数关于 $\mu$ 对称可以看出。指数分布的中位数 $m$ 是方程 $1-e^{-\lambda m}=1/2$ 的解，即 $m=\log 2/\lambda$。

## 方差和矩

### 方差和标准差

设随机变量 $X$ 有均值 $a=E(X)$。试验中，$X$ 取的值当然不一定怡好是 $a$，而会有所偏离，偏离的量 $X-a$ 本身也是随机的（因为 $X$ 是随机的）。我们要取这个偏离 $X-a$ 的某种有代表性的数字，来刻画这偏离即散布的程度大小如何。我们不能就取 $X-a$ 的均值，因为 $E(X-a)=E(X)-a=0$——正负偏离彼此抵消了。一种解决办法是取 $X-a$ 的绝对值 $|X-a|$ 以消除符号，再取其均值 $E|X-a|$，作为变量 $X$ 取值的散布程度的数字特征，这个量 $E|X-a|$ 叫做 $X$（或其分布）的**平均绝对差**，是常用于刻画散布度的数字特征之一。但是，由于绝对值在数学上处理甚不方便，人们就考虑了另一种做法：先把 $(X-a)$ 平方以消去符号，然后取其均值得 $E(X-a)^2$，把它作为 $X$ 取值散布度的衡量。这个量就叫做 $X$ 的“方差”（“方差”表示差的平方）。

**定义 2.1** 设 $X$ 为随机变量，分布为 $F$，则

$$
\begin{equation}
\tag{2.1}
{\rm Var}(X)=E(X-EX)^2
\end{equation}
$$

称为 $X$（或分布 $F$）的方差，其算术平方根 $\sqrt{{\rm Var}(X)}$ 称为 $X$（或分布 $F$）的标准差。

暂记 $EX=a$，由于 $(X-a)^2=X^2-2aX+a^2$，按照定理 1.1，得到

$$
\begin{equation}
\tag{2.2}
{\rm Var}(X)=E(X^2)-2aE(X)+a^2=E(X^2)-(EX)^2
\end{equation}
$$

方差的这个形式在计算上往往较为方便。

方差之所以成为刻画散布度的最重要的数字特征，原因之一是它具有一些优良的数字性质，反映在以下几个定理中：

**定理 2.1**

1. 常数的方差为 0。
2. 若 $C$ 为常数，则 ${\rm Var}(X+C)={\rm Var}(X)$。
3. 若 $C$ 为常数，则 ${\rm Var}(CX)=C^2{\rm Var}(X)$。

证明：

1. 若 $X 为常数 $a$，则 ${\rm Var}(X)=E(X^2)-(EX)^2=a^2-a^2=0$。
2. 因为 $E(X+C)=E(X)+C$，故有

    $$
    {\rm Var}(X+C)=E((X+C)-(EX+C))^2=E(X-EX)^2={\rm Var}(X)
    $$

3. 因为 $E(CX)=CE(X)$，故有

    $$
    {\rm Var}(CX)=E(CX-CEX)^2=C^2E(X-EX)^2=C^2{\rm Var}(X)
    $$

**定理 2.2** 独立随机变量之和的方差，等于各变量的方差之和：

$$
\begin{equation}
\tag{2.3}
{\rm Var}(X_1+\cdots+X_n)={\rm Var}(X_1)+\cdots+{\rm Var}(X_n)
\end{equation}
$$

证明：记 $E(X_i)=a_i,i=1,\cdots,n$，则因 $E(\sum_{i=1}^nX_i)=\sum_{i=1}^na_i$，有

$$
\begin{align}
{\rm Var}(X_1+\cdots+X_n)&=E(\sum_{i=1}^nX_i-\sum_{i=1}^na_i)^2=E(\sum_{i=1}^n(X_i-a_i))^2\\
&=\sum_{i,j=1}^nE((X_i-a_i)(X_j-a_j))
\end{align}
$$

其中有两类项：一类是 $i=j$，按照方差的定义，即为 ${\rm Var}(X_i)$；另一类项是 $i\neq j$，因 $X_i,X_j$ 独立，按照定理 1.2 有 $E(X_iX_j)=E(X_i)E(X_j)=a_ia_j$，故

$$
E((X_i-a_i)(X_j-a_j))=E(X_iX_j)-E(a_iX_j)-E(a_jX_i)+a_ia_j=a_ia_j-a_ia_j-a_ia_j+a_ia_j=0
$$

即证明了此定理。

这个定理是方差的一个极重要的性质，它与期望的定理 1.1 相似。但要注意的是：方差的定理要求各变量独立，而期望的定理则不要求。

**例 2.1** 设 $X$ 是一随机变量，$E(X)=a$，而 ${\rm Var}(X)=\sigma^2$。记 $Y=(X-a)/\sigma$，则 $E(Y)=0$，且按照定理 2.1 有 ${\rm Var}(Y)=1$。这样，对 $X$ 作一线性变换后，得到一个期望为 0、方差为 1 的变量 $Y$，常称 $Y$ 是 $X$ 的标准化。

**例 2.2** 设 $X$ 服从泊松分布 $P(\lambda)$，求其方差。前面已求出 $E(X)=\lambda$，又根据定理 1.3，有

$$
\begin{align}
E(X^2)&=\sum_{i=0}^\infty i^2e^{-\lambda}\lambda^i/i!\\
&=\sum_{i=0}^\infty i(i-1)e^{-\lambda}\lambda^i/i!+\sum_{i=0}^\infty ie^{-\lambda}\lambda^i/i!\\
&=\lambda^2\sum_{i=2}^\infty e^{-\lambda}\lambda^(i-2)/(i-2)!+\lambda\\
&=\lambda^2e^{-\lambda}\sum_{i=0}^\infty \lambda^(i)/i!+\lambda\\
&=\lambda^2e^{-\lambda}e^\lambda+\lambda\\
&=\lambda^2+\lambda\\
\end{align}
$$

于是 ${\rm Var}(X)=E(X^2)-(EX)^2=\lambda^2+\lambda-\lambda^2=\lambda$，即泊松分布 $P(\lambda)$ 的期望方差都等于其参数 $\lambda$。

**例 2.3** 设 $X$ 服从二项分布 $B(n,p)$，求 ${\rm Var}(X)$。

与例 1.6 类似地，将 $X$ 表示为 $n$ 个服从伯努利分布的独立随机变量 $X_1,\cdots,X_n$ 之和。因 $X_i$ 只取 1 和 0 两个值，概率分别为 $p$ 和 $1-p$，故 $E(X_i)=p,E(X_i^2)=p,i=1,\cdots,n$，因而有 ${\rm Var}(X_i)=p-p^2=p(1-p)$，从而

$$
{\rm Var}(X)=np(1-p)
$$

**例 2.5** 设 $X$ 服从正态分布 $N(\mu,\sigma^2)$，注意到 $E(X)=\mu$，有

$$
{\rm Var}(X)=E(X-\mu)^2=\frac{1}{\sqrt{2\pi}\sigma}\int_{-\infty}^\infty(x-\mu)^2e^{-(x-\mu)^2/2\sigma^2}{\rm d}x
$$

作变数代换 $x=\mu+\sigma t$，得

$$
{\rm Var}(X)=\sigma^2\frac{1}{\sqrt{2\pi}}\int_{-\infty}^\infty t^2e^{-t^2/2}{\rm d}t
$$

式中的积分已在例 1.8 中算过，为 $\sqrt{2\pi}$，故

$$
{\rm Var}(X)=\sigma^2
$$

**例 2.6** 指数分布的方差为 $1/\lambda^2$。区间 $[a,b]$ 上的均匀分布的方差为 $(b-a)^2/12$。这些都容易根据 (2.2) 直接算出。

**例 2.7** 求“统计三大分布”的方差。

先考虑卡方分布。设 $X\sim\chi_n^2$，将 $X$ 表示为 $X_1^2+\cdots+X_n^2$，$X_1,\cdots,X_n$ 独立同分布且有公共分布 $N(0,1)$，故

$$
{\rm Var}(X_i^2)=E(X_i^4)-(EX_i^2)^2=E(X_i^4)-1
$$

而

$$
E(X_i^4)=\frac{1}{\sqrt{2\pi}}\int_{-\infty}^\infty x^4e^{-x^2/2}{\rm d}x=\frac{2}{\sqrt{2\pi}}\int_0^\infty x^4e^{-x^2/2}{\rm d}x
$$

作变数代换 $x=\sqrt{2t}$，有

$$
\begin{align}
E(X_i^4)&=\frac{2}{\sqrt{2\pi}}2\sqrt{2}\int_0^\infty t^{3/2}e^{-t}{\rm d}t=\frac{4}{\sqrt{\pi}}\Gamma(\frac{5}{2})\\
&=\frac{4}{\sqrt{\pi}}\frac{3}{2}\frac{1}{2}\sqrt{\pi}=3
\end{align}
$$

故 ${\rm Var}(X_i^4)=3-1=2$，而 ${\rm Var}(X)=2n$。

然后考虑 $t$ 分布。设 $X=X_1/\sqrt{\frac{1}{n}X_2}$，$X_1,X_2$ 独立而 $X_2\sim\chi_n^2,X_1\sim N(0,1)$。前面已指出 $E(X)=0$，故由独立性有

$$
{\rm Var}(X)=E(X^2)=E(X_1^2)E(n/X_2)=nE(1/X_2)
$$

### 矩

**定义 2.2** 设 $X$ 为随机变量，$c$ 为常数，$k$ 为正整数，则 $E(X-c)^k$ 称为 $X$ 关于点 $c$ 的 $k$ 阶矩。

有两种情况比较重要：

1. $c=0$，此时 $a_k=E(X^k)$ 称为 $X$ 的 $k$ 阶原点矩。
2. $c=E(X)$，此时 $\mu_k=E(X-EX)^k$ 称为 $X$ 的 $k$ 阶中心矩。

一阶原点矩就是期望。一阶中心矩 $\mu_1=0$，二阶中心矩 $\mu_2$ 就是 ${\rm Var}(X)$。在统计学上，高于四阶的矩极少使用，三、四阶的矩有些应用，但也不是很多。

应用之一是用 $\mu_3$ 去衡量分布是否有偏。设 $X$ 的概率密度函数为 $f(x)$。若 $f$ 关于某点 $a$ 对称，即

$$
f(a+x)=f(a-x)
$$

如上图所示，则 $a$ 必等于 $E(X)$，且 $\mu_3=E(X-EX)^3=0$。如果 $\mu_3>0$，则称分布正偏或右偏；如果 $\mu_3<0$，则称分布负偏或左偏。特别地，对于正态分布而言有 $\mu_3=0$，故如果 $\mu_3$ 显著异于 0，则是分布与正态有较大偏离的标志。由于 $\mu_3$ 的次数是 $X$ 的次数的三次方，为抵消这一点，以 $X$ 的标准差的三次方，即 $\mu_2^{3/2}$ 去除 $\mu_3$，其商

$$
\begin{equation}
\tag{2.8}
\beta_1=\mu_3/\mu_2^{3/2}
\end{equation}
$$

称为 $X$ 或其分布的**偏度系数**。

应用之二是用 $\mu_4$ 去衡量分布（密度）在均值附近的陡峭程度如何。因为 $\mu_4=E(X-EX)^4$，容易看出，若 $X$ 的取值在概率上非常集中在 $E(X)$ 附近，则 $\mu_4$ 就倾向于小，否则就倾向于大。为抵消尺度的影响，类似于 $\mu_3$ 的情况，以标准差的四次方即去除，得到

$$
\begin{equation}
\tag{2.9}
\beta_2=\mu_4/\mu_2^2
\end{equation}
$$

称为 $X$ 或其分布的**峰度系数**。