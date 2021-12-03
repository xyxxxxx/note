设随机试验的样本空间为 $S=\{e_1,e_2,\cdots\}$， $X=X(e)$ 是定义在样本空间 $S$ 上的实值单值函数，称 $X=X(e)$ 为随机变量。



@将一枚硬币抛掷三次，观察正反面情况，样本空间为
$$
S=\{HHH,HHT,HTH,THH,HTT,THT,TTH,TTT\}
$$
随机变量 $X$ 可以定义为
$$
X=\begin{cases}3,\quad e=HHH\\
2,\quad e=HHT,HTH,THH\\
1,\quad e=HTT,THT,TTH\\
0,\quad e=TTT
\end{cases}
$$
$X$ 的定义域为样本空间，值域为实数集合 $\{0,1,2,3\}$ ； $X$ 的值表示三次抛掷中得到 $H$ 的总数。



随机变量的取值随试验的结果而定，而试验的各个样本点单独组成的事件上都定义了概率，因而随机变量的取值也对应着一定的概率。一般地，若 $L$ 是一个实数集合，事件 $B=\{e|X(e)\in L\}$，即 $B$ 是 $S$ 中使得 $X(e)\in L$ 的所有样本点 $e$ 所组成的事件，此时有
$$
P(B)=P(e|X(e)\in L)
$$
由于随机变量的取值在试验之前不能预知，且其取值有一定的概率，因此随机变量与普通函数有着本质的差异。





# 离散型随机变量

取值是有限多个或可列无限多个的随机变量称为离散型随机变量。设离散型随机变量 $X$ 所有可能的取值为 $x_k(k=1,2,\cdots)$， $X$ 取各个可能值得概率为
$$
P(X=x_k)=p_k,\ k=1,2,\cdots
$$
由概率的定义， $p_k$ 满足如下两个条件

1. $p_k\ge 0, k=1,2,\cdots$ 
2. $\sum_{k=1}^\infty p_k=1$ 

上式称为离散型随机变量 $X$ 的**分布律**，分布律也可以用表格表示
$$
\begin{array}{c|ccccc}
X & x_1 & x_2 & \cdots & x_n & \cdots\\
\hline
p_k & p_1 & p_2 & \cdots & p_n & \cdots
\end{array}
$$


## 伯努利分布

设试验 $E$ 只有两个可能结果：成功和失败，则 $E$ 称为**伯努利(Bernouli)试验**；设随机变量 $X$ 只可能取1和0两个值，当 $E$ 成功时取1，失败时取0，则称 $X$ 服从**伯努利分布(Bernouli distribution)**，其分布律为
$$
\begin{array}{c|ccccc}
X & 0 & 1\\
\hline
p_k & 1-p & p
\end{array}
$$
记作 $X\sim B(p)$。



## 二项分布

将上述伯努利试验独立地重复n次，以随机变量 $X$ 表示n次中成功的次数，则称 $X$ 服从**二项分布(binomial distribution)**，其分布律为
$$
P(X=k)=C_n^kp^k(1-p)^{n-k},\ k=0,1,\cdots, n
$$
记作 $X\sim b(n,p)$。

> $C_n^kp^k(1-p)^{n-k}$ 是 $(p+(1-p))^n$ 的二项式展开项，故称二项分布

@抛掷4枚均匀的硬币，各次结果独立，求得到2个正面的概率。
$$
X\sim b(4,\frac{1}{2}),\quad P(X=2)=C_4^2(\frac{1}{2})^2(\frac{1}{2})^2=\frac{3}{8}
$$


## 几何分布

独立地重复上述伯努利试验，以随机变量 $X$ 表示首次成功时的总试验次数，则称 $X$ 服从**几何分布(geometric distribution)**，其分布律为
$$
P(X=k)=(1-p)^{k-1}p
$$
记作 $X\sim G(p)$。



## *负二项分布

独立地重复上述伯努利试验，以随机变量 $X$ 表示第 $n$ 次成功时的总试验次数，则称 $X$ 服从**负二项分布**，其分布律为
$$
P(X=k)=C_{k-1}^{k-n}(1-p)^{k-n}p^n
$$
记作 $X\sim NB(n,p)$。



## 泊松分布

设随机变量 $X$ 的可能取值为所有非负整数，分布律为
$$
P(X=k)=\frac{\lambda^ke^{-\lambda}}{k!},\ k=0,1,2,\cdots
$$
其中 $\lambda>0$ 为常数，则称 $X$ 服从参数为 $\lambda$ 的**泊松分布(Poisson distribution)**，记作 $X\sim \pi(\lambda)$。

**泊松定理** 设 $\lambda>0$ 是一个常数， $n$ 是任意正整数，设 $np=\lambda$，则对于任一非负整数常数 $k$，有
$$
\lim_{n\to\infty}C_n^kp^k(1-p)^{n-k}=\frac{\lambda^ke^{-\lambda}}{k!}
$$

> 证明：
> $$
> \lim_{n\to\infty}C_n^kp^k(1-p)^{n-k}=\lim_{n\to\infty} \frac{n(n-1)\cdots(n-k+1)}{k!}(\frac{\lambda}{n})^k(1-\frac{\lambda}{n})^{n-k}\\
> =\lim_{n\to\infty}\frac{\lambda^k}{k!}(1\cdot(1-\frac{1}{n})\cdots(1-\frac{k-1}{n}))(1-\frac{\lambda}{n})^{n}(1-\frac{\lambda}{n})^{-k}\\
> =\lim_{n\to\infty}\frac{\lambda^k}{k!}\cdot 1\cdot e^{-\lambda}\cdot 1=\frac{\lambda^ke^{-\lambda}}{k!}
> $$
> 

泊松定理表明当 $n$ 较大时，以 $n,p$ 为参数的二项分布的概率可以由以 $\lambda =np$ 为参数的泊松分布近似计算。

@计算机硬件公司生产某种特殊芯片，次品率为0.1%，假设各芯片成为次品相互独立，求在1000只产品中至少有2只次品的概率。

$X\sim b(1000,0.001)$ 近似有 $X\sim \pi(1)$，因此
$$
P(X\ge 2)=1-P(X=0)-P(X=1)\\
=1-e^{-1}-e^{-1}\approx0.264
$$

> 经验表明当 $n\ge20,p\le0.05$ 时用泊松分布近似二项分布的效果颇佳。





# 连续型随机变量

设 $X$ 是一个随机变量， $x$ 是任意实数，函数
$$
F(x)=P(X\le x),\ -\infty<x<\infty
$$
称为 $X$ 的**分布函数(cumulative function)**。

分布函数具有以下性质：

1. $F(x)$ 是一个单调非减函数
2. $0\le F(x)\le 1$，且 $\lim_{x\to -\infty} F(x)=0,\lim_{x\to +\infty} F(x)=1$ 
3. $F(x)$ 是右连续的，即 $\lim_{\varepsilon\to 0}F(x+\varepsilon)=F(x)$ 

对于随机变量 $X$ 的分布函数 $F(x)$，若存在非负函数 $f(x)$，使得对于任意实数 $x$ 有
$$
F(x)=\int_{-\infty}^xf(t){\rm d}t
$$
则称 $X$ 为**连续型随机变量**，其中函数 $f(x)$ 称为 $X$ 的**概率密度函数(probability density function)**。由上式知连续型随机变量的分布函数为连续函数。

概率密度函数具有以下性质：

1. $f(x)\ge 0$ 

2. $\int_{-\infty}^{+\infty}f(x){\rm d}x=1$ 

3. 对于任意实数 $x_1\le x_2$，
   $$
   P(x_1<X\le x_2)=F(x_2)-F(x_1)=\int_{x_1}^{x_2}f(x){\rm d}x
   $$

4. 若 $f(x)$ 在点 $x$ 处连续，则有 $F'(x)=f(x)$ 

由性质2知曲线 $y=f(x)$ 与 $x$ 轴之间的面积等于1；由性质3知概率 $P(x_1<X\le x_2)$ 等于区间 $(x_1,x_2]$ 上曲线 $y=f(x)$ 之下的曲边梯形的面积；由性质4知 $f(x)$ 的连续点处有
$$
f(x)=\lim_{\Delta x\to 0^+}\frac{F(x+\Delta x)-F(x)}{\Delta x}\\
=\lim_{\Delta x\to 0^+}\frac{P(x<X\le x+\Delta x)}{\Delta x}
$$
若忽略高阶无穷小，则有
$$
P(x<X\le x+\Delta x)\approx f(x)\Delta x
$$
需要注意的是，对于连续型随机变量 $X$，它取任一指定实数值 $a$ 的概率均为0，即 $P(X=a)=0$，据此有
$$
P(a< x< b)=P(a< x\le b)=P(a\le x< b)=P(a\le x\le b)
$$
我们看到 $P(X=a)=0$，但 $X=a$ 并非不可能事件，因此 $P(A)=0$ 是 $A$ 是不可能事件的必要不充分条件。



## 均匀分布

若连续型随机变量 $X$ 具有概率密度
$$
f(x)=\cases{
\frac{1}{b-a},\ a<x<b\\
0,\quad\ 其它
}
$$
则称 $X$ 在区间 $(a,b)$ 上服从**均匀分布(uniform distribution)**，记作 $X\sim U(a,b)$。

$X$ 的分布函数为
$$
F(x)=\cases{0,\quad\ x<a\\
\frac{x-a}{b-a},\ a\le x<b\\
1,\quad\ x\ge b
}
$$


## 指数分布

若连续型随机变量 $X$ 具有概率密度
$$
f(x)=\cases{\lambda e^{-\lambda x},\ x>0\\
0,\quad\quad\ \ 其它
}
$$
其中 $\lambda>0$ 为常数，则称 $X$ 服从参数为 $\lambda$ 的**指数分布(exponential distribution)**，记作 $X\sim Exp(\lambda)$。

$X$ 的分布函数为
$$
F(x)=\cases{1-e^{-x/\theta},\ x>0\\
0,\quad\quad\quad\ \ 其它
}
$$
对于任意 $s,t>0$，有
$$
P(X>s+t|X>s)=\frac{P(X>s+t\and X>s)}{P(X>s)}=\frac{P(X>s+t)}{P(X>s)}\\
=\frac{1-F(s+t)}{1-F(s)}=\frac{e^{-(s+t)/\theta}}{e^{-s/\theta}}=e^{-t/\theta}=P(X>t)
$$
上式称为随机变量 $X$ 的**无记忆性**。若 $X$ 是某一元件的寿命，则元件能够继续使用的时长的概率分布与元件已经使用的时长无关。



## 正态分布

若连续型随机变量 $X$ 具有概率密度
$$
f(x)=\frac{1}{\sqrt{2\pi}\sigma}e^{-(x-\mu)^2/(2\sigma^2)},\ -\infty<x<+\infty
$$
其中 $\mu,\sigma\ (\sigma>0)$ 为常数，则称 $X$ 服从参数为 $\mu,\sigma$ 的**正态分布(normal distribution)**或**高斯分布(Gauss distribution)**，记作 $X\sim N(\mu,\sigma^2)$。

正态分布的概率密度曲线具有以下性质：

1. 曲线 $y=f(x)$ 关于 $x=\mu$ 对称

2. 当 $x=\mu$ 时取到最大值
   $$
   f(\mu)=\frac{1}{\sqrt{2\pi}\sigma}
   $$

$X$ 的分布函数为
$$
F(x)=\frac{1}{\sqrt{2\pi}\sigma}\int_{-\infty}^x e^{-(t-\mu)^2/(2\sigma^2)}{\rm d}t
$$
特别地，当 $\mu=0,\sigma=1$ 时称随机变量 $X$ 服从**标准正态分布**，其概率密度函数和分布函数分别用 $\varphi(x),\Phi(x)$ 表示，即
$$
\varphi(x)=\frac{1}{\sqrt{2\pi}}e^{-t^2/2}\\
\Phi(x)=\frac{1}{\sqrt{2\pi}}\int_{-\infty}^xe^{-t^2/2}{\rm d}t
$$

**性质**

+ 若 $X\sim N(\mu,\sigma^2)$，则 $Z=\frac{X-\mu}{\sigma}\sim N(0,1)$ 

+ 有限个相互独立的正态随机变量的线性组合仍然服从正态分布

  


## 伽马分布

若连续型随机变量 $X$ 具有概率密度
$$
f(x)=\begin{cases} \frac{1}{\theta^\alpha\Gamma(\alpha)}x^{\alpha-1}e^{-x/\theta},\quad x>0\\
0,\quad\quad\quad\quad\quad\quad\quad\ 其它
\end{cases}
$$
其中常数 $\alpha>0,\theta>0$，则称 $X$ 服从参数为 $\alpha,\theta$ 的**伽马分布(gamma distribution)**，记作 $X\sim \Gamma(\alpha,\theta)$。





# 多维随机变量（随机向量）

## 二维随机变量

设 $E$ 是一个随机试验，其样本空间是 $S$，设 $X=X(e)$ 和 $Y=Y(e)$ 是定义在 $S$ 上的随机变量，它们构成一个向量 $(X,Y)$，称为二维随机变量或二维随机向量。二维随机变量 $(X,Y)$ 的性质不仅与 $X,Y$ 本身有关，还依赖于这两个随机变量的相互关系，因此需要将 $(X,Y)$ 作为一个整体研究。

设二维随机变量 $(X,Y)$，对于任意实数 $x,y$，二元函数：
$$
F(x,y)=P(X\le x\cap Y\le y)\triangleq P(X\le x,Y\le y)
$$
称为 $(X,Y)$ 的**分布函数**，或称为 $X,Y$ 的**联合分布函数**。

分布函数 $F(x,y)$ 具有以下基本性质：

1. $F(x,y)$ 关于 $x,y$ 单调非减

2. $0\le F(x,y)\le 1$，且对于任意固定的 $y$， $F(-\infty,y)=0$ ；对于任意固定的 $x$， $F(x,-\infty)=0$ ； $F(-\infty,-\infty)=0,F(+\infty,+\infty)=1$ 

3. $F(x,y)$ 关于 $x,y$ 右连续

4. 对任意 $x_1<x_2,y_1<y_2$，以下不等式成立：
   $$
   F(x_2,y_2)-F(x_2,y_1)-F(x_1,y_2)+F(x_1,y_1)\ge 0
   $$
   



如果二维随机变量 $(X,Y)$ 全部能取到的值是有限对或可列无限多对，则称 $(X,Y)$ 是**二维离散型随机变量**。设二维离散型随机变量 $(X,Y)$ 的所有可能取值为 $(x_i,y_i),\ i,j=1,2,\cdots$，记 $P(X=x_i,Y=y_j)=p_{ij},\ i,j=1,2,\cdots$，称为 $(X,Y)$ 的**分布律**，或者 $X,Y$ 的**联合分布律**，用表格表示为：

![Screenshot from 2020-10-13 20-10-23.png](https://i.loli.net/2020/10/13/IHnrayGK9LkeCBb.png)

由概率的定义有：
$$
p_{ij}\ge 0,\quad \sum_{i=1}^\infty\sum_{j=1}^\infty p_{ij}=1
$$


@设随机变量 $X$ 在整数1,2,3,4中等可能地取一个值，另一个随机变量 $Y$ 在整数 $1\sim X$ 中等可能地取一个值，求 $(X,Y)$ 的分布律。
$$
P(X=i,Y=j)=P(Y=j|X=i)P(X=i)=\frac{1}{i}\cdot\frac{1}{4},\quad i=1,2,3,4,j\le i
$$
因此 $(X,Y)$ 的分布律为

![Screenshot from 2020-10-13 20-17-16.png](https://i.loli.net/2020/10/13/FwZ7TlXdcWKA9D8.png)



根据分布函数的定义，离散型随机变量 $X$ 和 $Y$ 的联合分布函数为
$$
F(x,y)=\sum_{x_i\le x} \sum_{y_j\le y}p_{ij}
$$
对于分布函数 $F(x,y)$，如果存在非负的函数 $f(x,y)$，使得对于任意 $x,y$ 都有
$$
F(x,y)=\int_{-\infty}^y\int_{-\infty}^xf(u,v){\rm d}u{\rm d}v
$$
则称 $(X,Y)$ 是**二维连续型随机变量**，函数 $f(x,y)$ 称为 $(X,Y)$ 的**概率密度**，或者 $X,Y$ 的联合概率密度。

概率密度 $f(x,y)$ 具有以下性质：

1. $f(x,y)\ge 0$ 

2. $\int_{-\infty}^{+\infty}\int_{-\infty}^{+\infty}f(x,y){\rm d}x{\rm d}y=1$ 

3. 设 $G$ 是 $xOy$ 平面上的区域，点 $(X,Y)$ 落在 $G$ 内的概率为
   $$
   P((X,Y)\in G)=\iint_G f(x,y){\rm d}x{\rm d}y
   $$
   
4. 若 $f(x,y)$ 在点 $(x,y)$ 处连续，则
   $$
\frac{\partial^2 F(x,y)}{\partial x\partial y}=f(x,y)
   $$

   

@ 设二维随机变量 $(X,Y)$ 具有概率密度
$$
f(x,y)=\begin{cases}2e^{-(2x+y)},\quad x,y>0\\
0,\quad\quad\quad\quad\ 其它
\end{cases}
$$
求概率 $P(Y\le X)$。
$$
P(Y\le X)=\int_{0}^{+\infty} \int_{0}^x 2e^{-(2x+y)}{\rm d}y{\rm d}x=1/3
$$


以上关于二维随机变量的讨论不难推广到 $n(n>2)$ 维的情形。

   

## 边缘分布   

二维随机变量 $(X,Y)$ 作为一个整体具有分布函数 $F(x,y)$，而随机变量 $X,Y$ 也各自有分布函数，分别记为 $F_X(x),F_Y(y)$，称为 $(X,Y)$ 关于 $X,Y$ 的**边缘分布函数**。边缘分布函数与分布函数具有如下关系：
$$
F_X(x)=F(x,\infty)\\
F_Y(y)=F(\infty,y)
$$
对于离散型随机变量 $(X,Y)$， $X,Y$ 各自的分布律分别为
$$
p_{i*}=\sum_{j=1}^{\infty}p_{ij}=P(X=x_i),\quad i=1,2,\cdots\\
p_{*j}=\sum_{i=1}^{\infty}p_{ij}=P(Y=y_j),\quad j=1,2,\cdots\\
$$
称为 $(X,Y)$ 关于 $X,Y$ 的**边缘分布律**。

对于连续型随机变量 $(X,Y)$，设其概率密度为 $f(x,y)$，则 $X,Y$ 各自的概率密度分别为
$$
f_X(x)=\int_{-\infty}^{+\infty}f(x,y){\rm d}y\\
f_Y(y)=\int_{-\infty}^{+\infty}f(x,y){\rm d}x\\
$$
称为 $(X,Y)$ 关于 $X,Y$ 的**边缘概率密度**。



@设随机变量 $X,Y$ 具有联合概率密度
$$
f(x,y)=\begin{cases}6,\quad x^2\le y\le x\\
0,\quad 其它
\end{cases}
$$
求边缘概率密度 $f_X(x),f_Y(y)$。
$$
f_X(x)=\int_{-\infty}^{+\infty}f(x,y){\rm d}y=\begin{cases}\int_{x^2}^x 6{\rm d}y=6(x-x^2),\ 0\le x\le 1\\
0,\quad\quad\quad\quad\quad\quad\quad\quad 其它
\end{cases}\\
f_Y(y)=\int_{-\infty}^{+\infty}f(x,y){\rm d}x=\begin{cases}\int_{y}^{\sqrt y} 6{\rm d}x=6(\sqrt{y}-y),\ 0\le y\le 1\\
0,\quad\quad\quad\quad\quad\quad\quad\quad 其它
\end{cases}\\
$$


@设二维随机变量 $(X,Y)$ 的概率密度为
$$
f(x,y)=\frac{1}{2\pi\sigma_1\sigma_2\sqrt{1-\rho^2}}\exp(\frac{-1}{2(1-\rho^2)}(\frac{(x-\mu_1)^2}{\sigma_1^2}+\frac{(y-\mu_2)^2}{\sigma_2^2}-2\rho\frac{(x-\mu_1)(y-\mu_2)}{\sigma_1\sigma_2}))
$$
其中 $\mu_1,\mu_2,\sigma_1,\sigma_2,\rho$ 都是常数（均值、标准差、相关系数），则称 $(X,Y)$ 服从参数为 $\mu_1,\mu_2,\sigma_1,\sigma_2,\rho$ 的**二维正态分布**，记作 $(X,Y)\sim N(\mu_1,\mu_2,\sigma_1^2,\sigma_2^2,\rho)$。求二维正态随机变量的边缘概率密度。

过程略。求得 $f_X(x),f_Y(y)$ 分别为 $N(\mu_1,\sigma_1^2),N(\mu_2,\sigma_2^2)$ 的概率密度，而与 $\rho$ 无关。由此可知已知 $X,Y$ 边缘分布并不足以确定 $X,Y$ 的联合分布。



## 条件分布

设二维离散型随机变量 $(X,Y)$，考虑条件概率
$$
P(X=x_i|Y=y_j)=\frac{P(X=x_i,Y=y_j)}{P(Y=y_j)}=\frac{p_{ij}}{p_{*j}},\ i=1,2,\cdots
$$
易知上述条件概率具有分布律的性质：

1. $P(X=x_i|Y=y_j)\ge 0$ 
2. $\sum_{i=1}^\infty P(X=x_i|Y=y_j) =1$ 

上述条件概率称为在 $Y=y_j$ 条件下随机变量 $X$ 的**条件分布律**。同样地，对于固定的 $i$，且 $P(X=x_i)>0$，则称
$$
P(Y=y_j|X=x_i)=\frac{P(X=x_i,Y=y_j)}{P(X=x_i)}=\frac{p_{ij}}{p_{i*}},\ j=1,2,\cdots
$$
为在 $X=x_i$ 条件下随机变量 $Y$ 的**条件分布律**。

对于连续型随机变量 $(X,Y)$，设其概率密度为 $f(x,y)$， $(X,Y)$ 关于 $Y$ 的边缘概率密度为 $f_Y(y)$。若对于固定的 $y$， $f_Y(y)>0$，则分别称
$$
f_{X|Y}(x|y)=\frac{f(x,y)}{f_Y(y)}\\
F_{X|Y}(x|y)=\int_{-\infty}^x \frac{f(x,y)}{f_Y(y)}{\rm d}x
$$
为在 $Y=y$ 条件下随机变量 $X$ 的**条件概率密度**和**条件分布函数**。



## 随机变量的独立性

设 $F(x,y)$ 和 $F_X(x),F_Y(y)$ 分别是二维随机变量 $(X,Y)$ 的分布函数和边缘分布函数。若对于所有 $x,y$ 有
$$
F(x,y)=F_X(x)F_Y(y)
$$
则称随机变量 $X,Y$ 是**相互独立**的。

设二维离散型随机变量 $(X,Y)$，那么 $X,Y$ 相互独立等价于：对于 $(X,Y)$ 的所有可能取值 $(x_i,y_i)$，有
$$
P(X=x_i,Y=y_j)=P(X=x_i)P(Y=y_j)
$$
设二维连续型随机变量 $(X,Y)$ 的概率密度和边缘概率密度分别为 $f(x,y),f_X(x),f_Y(y)$，那么 $X,Y$ 相互独立等价于：
$$
f(x,y)=f_X(x)f_Y(y)
$$
在平面上几乎处处成立。



以上关于二维随机变量的讨论不难推广到 $n(n>2)$ 维的情形。

**定理** 设 $(X_1,X_2,\cdots,X_m)$ 和 $(Y_1,Y_2,\cdots,Y_n)$ 相互独立，即
$$
F(x_1,x_2,\cdots,x_m,y_1,y_2,\cdots,y_n)=F_1(x_1,x_2,\cdots,x_m)F_2(y_1,y_2,\cdots,y_n)
$$
其中 $F,F_1,F_2$ 分别为 $(X_1,X_2,\cdots,X_m,Y_1,Y_2,\cdots,Y_n),(X_1,X_2,\cdots,X_m),(Y_1,Y_2,\cdots,Y_n)$ 的分布函数，则 $X_i,Y_j\ (i=1,2,\cdots,m,j=1,2,\cdots,n)$ 相互独立；又若 $h,g$ 为连续函数，则 $h(X_1,X_2,\cdots,X_m),g(Y_1,Y_2,\cdots,Y_n)$ 相互独立。



## 两个随机变量的函数的分布

### $Z=X+Y$ 的分布

设二维连续型随机变量 $(X,Y)$ 具有概率密度 $f(x,y)$。则 $Z=X+Y$ 仍为连续型随机变量，概率密度为
$$
f_{X+Y}(z)=\int_{-\infty}^{+\infty}f(z-y,y){\rm d}y\\
或\ f_{X+Y}(z)=\int_{-\infty}^{+\infty}f(x,z-x){\rm d}x\\
$$
又若 $X,Y$ 相互独立，设 $(X,Y)$ 关于 $X,Y$ 的边缘密度分别为 $f_X(x),f_Y(y)$，则上式化为
$$
f_{X+Y}(z)=\int_{-\infty}^{+\infty}f_X(z-y)f_Y(y){\rm d}y\\
或\ f_{X+Y}(z)=\int_{-\infty}^{+\infty}f_X(x)f_Y(z-x){\rm d}x\\
$$
这两个公式称为 $f_X,f_Y$ 的卷积公式，记为 $f_X*f_Y$，即
$$
f_X*f_Y=\int_{-\infty}^{+\infty}f_X(z-y)f_Y(y){\rm d}y=\int_{-\infty}^{+\infty}f_X(x)f_Y(z-x){\rm d}x
$$


@设 $X,Y\sim N(0,1)$，且 $X,Y$ 相互独立，求 $Z=X+Y$ 的概率密度。
$$
f_X(x)=f_Y(y)=\frac{1}{\sqrt{2\pi}}e^{-x^2/2},\ x,y\in\mathbb{R}\\
f_Z(z)=\int_{-\infty}^{+\infty}f_X(x)f_Y(z-x){\rm d}x\\
=\frac{1}{2\pi}\int_{-\infty}^{+\infty}e^{-x^2/2}e^{-(z-x)^2/2}{\rm d}x\\
=\frac{1}{2\pi}e^{-z^2/4}\int_{-\infty}^{+\infty}e^{-(x-z/2)^2}{\rm d}x\\
(令t=x-z/2)=\frac{1}{2\pi}e^{-z^2/4}\int_{-\infty}^{+\infty}e^{-t^2}{\rm d}t\\
=\frac{1}{2\pi}e^{-z^2/4}\sqrt{\pi}=\frac{1}{2\sqrt{\pi}}e^{-z^2/4}
$$
即 $Z\sim N(0,2)$。

根据本例可知，设 $X\sim N(\mu_1,\sigma_1^2),Y\sim N(\mu_2,\sigma_2^2)$ 且 $X,Y$ 相互独立，则 $Z=X+Y\sim N(\mu_1+\mu_2,\sigma_1^2+\sigma_2^2)$。这个结论还能进一步推广为，<u>有限个相互独立的正态随机变量的线性组合仍然服从正态分布</u>。



@设 $X\sim \Gamma(\alpha,\theta),Y\sim \Gamma(\beta,\theta)$，则 $X+Y\sim \Gamma(\alpha+\beta,\theta)$。这个结论还能进一步推广到 $n$ 个随机变量的情形。



### $Z=Y/X,Z=XY$ 的分布





### $M=\max\{X,Y\},N=\min\{X,Y\}$ 的分布







# 随机变量的数字特征

## 数学期望

设离散型随机变量 $X$ 的分布律为 $P(X=x_k)=p_k,\quad k=1,2,\cdots$，若级数
$$
E(X)=\sum_{k=1}^\infty x_kp_k
$$
绝对收敛，则称该级数为随机变量 $X$ 的**数学期望**，简称**期望**，也称**均值**。

设连续型随机变量 $X$ 的概率密度为 $f(x)$，若积分
$$
E(X)=\int_{-\infty}^{+\infty}xf(x){\rm d}x
$$
绝对收敛，则称该积分的值为随机变量 $X$ 的数学期望。



**定理** 设随机变量 $Y$ 是 $X$ 的函数，即 $Y=g(X)$， $g$ 为连续函数，则有
$$
E(Y)=E(g(X))
$$


**性质**

+ $E(C)=C$，其中 $C$ 为常数
+ $E(CX)=CE(X)$ 
+ $E(X+Y)=E(X)+E(Y)$ （可以推广到任意有限个随机变量）
+ $E(XY)=E(X)E(Y)$，其中 $X,Y$ 相互独立（可以推广到任意有限个随机变量）



## 方差

设随机变量 $X$，若 $E([X-E(X)]^2)$ 存在，则称其为 $X$ 的**方差**，记作
$$
Var(X)=E([X-E(X)]^2)
$$
实际应用中还引入 $\sigma(X)=\sqrt{Var(X)}$，称为**标准差**。

方差用于衡量随机变量与其期望的偏离程度， $Var(X)$ 较小意味着 $X$ 的取值集中于 $E(X)$ 附近，而 $Var(X)$ 较大则意味着 $X$ 的取值，较为分散。

方差可按下式计算
$$
Var(X)=E(X^2)-E^2(X)
$$

**性质**

+ $Var(C)=0$，其中 $C$ 为常数

+ $Var(CX)=C^2Var(X),Var(X+C)=Var(X)$ 

+ $$
  Var(X+Y)=Var(X)+Var(Y)+2E([X-E(X)][Y-E(Y)])=Var(X)+Var(Y)+2Cov(X,Y)
  $$

+ $Var(X+Y)=Var(X)+Var(Y)$，其中 $X,Y$ 相互独立（可以推广到任意有限个随机变量）

+ $Var(X)=0$ 的充要条件是 $X$ 以概率1取常数 $E(X)$ 



## 协方差及相关系数

$E([X-E(X)][Y-E(Y)])$ 称为随机变量 $X,Y$ 的**协方差**，记作
$$
Cov(X,Y)=E([X-E(X)][Y-E(Y)])\\
$$
随机变量 $X,Y$ 的**相关系数**定义为
$$
\rho_{XY}=\frac{Cov(X,Y)}{\sqrt{Var(X)Var(Y)}}
$$
由定义可知
$$
Cov(X,Y)=Cov(Y,X)=E(XY)-E(X)E(Y) \\
D(X+Y)=D(X)+D(Y)+2Cov(X,Y)\\
$$

**性质**

+ $Cov(aX,bY)=abCov(X,Y)$，其中 $a,b$ 是常数

+ $Cov(X_1+X_2,Y)=Cov(X_1,Y)+Cov(X_2,Y)$ 

+ $|\rho_{XY}|\le 1$ 

+ $|\rho_{XY}|= 1$ 的充要条件是，存在常数 $a,b$ 使得
  $$
  P(Y=a+bX)=1
  $$
  

## 矩和协方差矩阵

设随机变量 $X,Y$，若
$$
E(X^k),\quad k=1,2,\cdots
$$
存在，称之为 $X$ 的** $k$ 阶原点矩**，简称** $k$ 阶矩**；若
$$
E([X-E(X)]^k),\quad k=2,3\cdots
$$
存在，称之为 $X$ 的** $k$ 阶中心矩**；若
$$
E(X^kY^l),\quad k,l=1,2,\cdots
$$
存在，称之为 $X,Y$ 的** $k+l$ 阶混合矩**；若
$$
E([X-E(X)]^k[Y-E(Y)]^l),\quad k,l=1,2,\cdots
$$
存在，称之为 $X,Y$ 的** $k+l$ 阶混合中心矩**。

显然， $E(X)$ 是 $X$ 的一阶原点矩， $Var(X)$ 是 $X$ 的二阶中心矩， $Cov(X,Y)$ 是 $X,Y$ 的二阶混合中心矩。



设 $n$ 维随机变量 $(X_1,X_2,\cdots,X_n)$ 的二阶混合中心矩
$$
c_{ij}=Cov(X_i,X_j)=E([X_i-E(X_i)][X_j-E(X_j)]),\ i,j=1,2,\cdots,n
$$
都存在，则称矩阵
$$
C=\begin{bmatrix}c_{11} & c_{12} & \cdots & c_{1n}\\
c_{21} & c_{22} & \cdots & c_{2n}\\
\vdots & \vdots & &\vdots\\
c_{n1} & c_{n2} & \cdots & c_{nn}
\end{bmatrix}
$$
为 $(X_1,X_2,\cdots,X_n)$ 的**协方差矩阵**。由于 $c_{ij}=c_{ji},i,j=1,2,\cdots,n$，因而上述矩阵是一个对称矩阵。

由此可以将 $n$ 维正态随机变量 $(X_1,X_2,\cdots,X_n)$ 的概率密度化简为
$$
f(x_1,x_2,\cdots,x_n)=\frac{1}{(2\pi)^{n/2}(\det C)^{1/2}}\exp(-\frac{1}{2}(\pmb x-\pmb \mu)^{\rm T}C^{-1}(\pmb x-\pmb \mu))
$$
其中 $C$ 是 $(X_1,X_2,\cdots,X_n)$ 的协方差矩阵。




## 矩母函数，特征函数，分布函数的拉普拉斯变换

设随机变量 $X$，若 $E(e^{tX}),t\in \mathbb{R}$ 存在，则称其为 $X$ 的**矩母函数(moment-generating function)**，记作
$$
M_X(t)=E(e^{tX}),\quad t\in\mathbb{R}
$$
<u>矩母函数完全定义了随机变量的分布</u>，因此我们可以用矩母函数表示随机变量的分布。

然而，随机变量不一定存在矩母函数，因此理论上更方便的方法是定义**特征函数(characteristic function)**
$$
\varphi_X(t)=E(e^{itX}),\quad t\in\mathbb{R},\ i为虚数单位
$$
<u>特征函数同样完全定义了随机变量的分布</u>，并且对于任意随机变量总是存在。特征函数可以视作随机变量 $iX$ 的矩母函数。



**性质**

+ 对 $M_X$ 逐次求导并计算 $t=0$ 的值可以得到 $X$ 的各阶矩，即
  $$
  M_X^n(t)=E(X^ne^{tX})\\
  M_X^n(0)=E(X^n),\quad n\ge 1\\
  $$

+ 若 $X_1,X_2$ 独立，则有
  $$
  M_{X_1+X_2}(t)=M_{X_1}(t)M_{X_2}(t)
  $$
  

类似地，定义随机变量 $X_1,X_2,\cdots,X_n$ 的**联合矩母函数**为
$$
M_X(t_1,t_2,\cdots,t_n)=E(\exp(\sum_{i=1}^n t_iX_i)),\quad t_i\in\mathbb{R},\ i=1,2,\cdots,n
$$
**联合特征函数**为
$$
\varphi_X(t_1,t_2,\cdots,t_n)=E(\exp(i\sum_{i=1}^n t_iX_i)),\quad t_i\in\mathbb{R},\ i=1,2,\cdots,n
$$


对于只取非负值的随机变量，使用分布函数 $F(x)$ 的拉普拉斯变换有时比特征函数更加方便，定义为
$$
\tilde{F}(s)=\int_0^\infty 
$$




## 常见随机变量的特征

$$
\begin{array}{c|cccc}
X & E(X) & Var(X) & M_X(t) & \varphi(t)\\
\hline
{\rm Bernoulli} & p & p(1-p) & 1-p+pe^t & 1-p+pe^{it}\\
{\rm binomial} & np & np(1-p) & (1-p+pe^t)^n & (1-p+pe^{it})^n\\
{\rm geometric} & 1/p & (1-p)/p^2 & \frac{pe^t}{1-e^t+pe^t} & \frac{pe^{it}}{1-e^{it}+pe^{it}}\\
{\rm negative\ binomial} & n/p & n(1-p)/p^2 & (\frac{pe^t}{1-e^t+pe^t})^n & (\frac{pe^{it}}{1-e^{it}+pe^{it}})^n\\
{\rm Poisson} & \lambda & \lambda & e^{\lambda(e^t-1)} & e^{\lambda(e^{it}-1)}\\
\hline
{\rm uniform} & (a+b)/2 & (b-a)^2/12 & \frac{e^{tb}-e^{ta}}{t(b-a)} & \frac{e^{itb}-e^{ita}}{it(b-a)}\\
{\rm exponential}  & 1/\lambda & 1/\lambda^2 & (1-t\lambda^{-1})^{-1} & (1-it\lambda^{-1})^{-1}\\
{\rm normal} & \mu & \sigma^2 & e^{t\mu+\frac{1}{2}\sigma^2t^2} & e^{it\mu-\frac{1}{2}\sigma^2t^2} \\
{\rm gamma} &  &  & (1-t\theta)^{-k} & (1-it\theta)^{-k}\\
{\rm chi-squared} &  &  & (1-2t)^{-k/2} & (1-2it)^{-k/2} \\
\end{array}
$$

