设随机试验的样本空间为$$S=\{e_1,e_2,\cdots\}$$，$$X=X(e)$$是定义在样本空间$$S$$上的实值单值函数，称$$X=X(e)$$为随机变量。



@将一枚硬币抛掷三次，观察正反面情况，样本空间为
$$
S=\{HHH,HHT,HTH,THH,HTT,THT,TTH,TTT\}
$$
随机变量$$X$$可以定义为
$$
X=\begin{cases}3,\quad e=HHH\\
2,\quad e=HHT,HTH,THH\\
1,\quad e=HTT,THT,TTH\\
0,\quad e=TTT
\end{cases}
$$
$$X$$的定义域为样本空间，值域为实数集合$$\{0,1,2,3\}$$；$$X$$的值表示三次抛掷中得到$$H$$的总数。



随机变量的取值随试验的结果而定，而试验的各个样本点单独组成的事件上都定义了概率，因而随机变量的取值也对应着一定的概率。一般地，若$$L$$是一个实数集合，事件$$B=\{e|X(e)\in L\}$$，即$$B$$是$$S$$中使得$$X(e)\in L$$的所有样本点$$e$$所组成的事件，此时有
$$
P(B)=P(e|X(e)\in L)
$$
由于随机变量的取值在试验之前不能预知，且其取值有一定的概率，因此随机变量与普通函数有着本质的差异。





# 离散型随机变量

取值是有限多个或可列无限多个的随机变量称为离散型随机变量。设离散型随机变量$$X$$所有可能的取值为$$x_k(k=1,2,\cdots)$$，$$X$$取各个可能值得概率为
$$
P(X=x_k)=p_k,\ k=1,2,\cdots
$$
由概率的定义，$$p_k$$满足如下两个条件

1. $$p_k\ge 0, k=1,2,\cdots$$
2. $$\sum_{k=1}^\infty p_k=1$$

上式称为离散型随机变量$$X$$的**分布律**，分布律也可以用表格表示
$$
\begin{array}{c|ccccc}
X & x_1 & x_2 & \cdots & x_n & \cdots\\
\hline
p_k & p_1 & p_2 & \cdots & p_n & \cdots
\end{array}
$$


## 伯努利分布

设试验$$E$$只有两个可能结果：成功和失败，则$$E$$称为**伯努利(Bernouli)试验**；设随机变量$$X$$只可能取1和0两个值，当$$E$$成功时取1，失败时取0，则称$$X$$服从**伯努利分布(Bernouli distribution)**，其分布律为
$$
\begin{array}{c|ccccc}
X & 0 & 1\\
\hline
p_k & 1-p & p
\end{array}
$$
记作$$X\sim B(p)$$。



## 二项分布

将上述伯努利试验独立地重复n次，以随机变量$$X$$表示n次中成功的次数，则称$$X$$服从**二项分布(binomial distribution)**，其分布律为
$$
P(X=k)=C_n^kp^k(1-p)^{n-k},\ k=0,1,\cdots, n
$$
记作$$X\sim b(n,p)$$。

> $$C_n^kp^k(1-p)^{n-k}$$是$$(p+(1-p))^n$$的二项式展开项，故称二项分布

@抛掷4枚均匀的硬币，各次结果独立，求得到2个正面的概率。
$$
X\sim b(4,\frac{1}{2}),\quad P(X=2)=C_4^2(\frac{1}{2})^2(\frac{1}{2})^2=\frac{3}{8}
$$


## 几何分布

独立地重复上述伯努利试验，以随机变量$$X$$表示首次成功时的总试验次数，则称$$X$$服从**几何分布(geometric distribution)**，其分布律为
$$
P(X=k)=(1-p)^{k-1}p
$$
记作$$X\sim G(p)$$。



## *负二项分布

独立地重复上述伯努利试验，以随机变量$$X$$表示第$$n$$次成功时的总试验次数，则称$$X$$服从**负二项分布**，其分布律为
$$
P(X=k)=C_{k-1}^{k-n}(1-p)^{k-n}p^n
$$
记作$$X\sim NB(n,p)$$。



## 泊松分布

设随机变量$$X$$的可能取值为所有非负整数，分布律为
$$
P(X=k)=\frac{\lambda^ke^{-\lambda}}{k!},\ k=0,1,2,\cdots
$$
其中$$\lambda>0$$为常数，则称$$X$$服从参数为$$\lambda$$的**泊松分布(Poisson distribution)**，记作$$X\sim \pi(\lambda)$$。

**泊松定理** 设$$\lambda>0$$是一个常数，$$n$$是任意正整数，设$$np=\lambda$$，则对于任一非负整数常数$$k$$，有
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

泊松定理表明当$$n$$较大时，以$$n,p$$为参数的二项分布的概率可以由以$$\lambda =np$$为参数的泊松分布近似计算。

@计算机硬件公司生产某种特殊芯片，次品率为0.1%，假设各芯片成为次品相互独立，求在1000只产品中至少有2只次品的概率。

$$X\sim b(1000,0.001)$$近似有$$X\sim \pi(1)$$，因此
$$
P(X\ge 2)=1-P(X=0)-P(X=1)\\
=1-e^{-1}-e^{-1}\approx0.264
$$

> 经验表明当$$n\ge20,p\le0.05$$时用泊松分布近似二项分布的效果颇佳。





# 连续型随机变量

设$$X$$是一个随机变量，$$x$$是任意实数，函数
$$
F(x)=P(X\le x),\ -\infty<x<\infty
$$
称为$$X$$的**分布函数(cumulative function)**。

分布函数具有以下性质：

1. $$F(x)$$是一个单调非减函数
2. $$0\le F(x)\le 1$$，且$$\lim_{x\to -\infty} F(x)=0,\lim_{x\to +\infty} F(x)=1$$
3. $$F(x)$$是右连续的，即$$\lim_{\varepsilon\to 0}F(x+\varepsilon)=F(x)$$

对于随机变量$$X$$的分布函数$$F(x)$$，若存在非负函数$$f(x)$$，使得对于任意实数$$x$$有
$$
F(x)=\int_{-\infty}^xf(t){\rm d}t
$$
则称$$X$$为**连续型随机变量**，其中函数$$f(x)$$称为$$X$$的**概率密度函数(probability density function)**。由上式知连续型随机变量的分布函数为连续函数。

概率密度函数具有以下性质：

1. $$f(x)\ge 0$$

2. $$\int_{-\infty}^{+\infty}f(x){\rm d}x=1$$

3. 对于任意实数$$x_1\le x_2$$，
   $$
   P(x_1<X\le x_2)=F(x_2)-F(x_1)=\int_{x_1}^{x_2}f(x){\rm d}x
   $$

4. 若$$f(x)$$在点$$x$$处连续，则有$$F'(x)=f(x)$$

由性质2知曲线$$y=f(x)$$与$$x$$轴之间的面积等于1；由性质3知概率$$P(x_1<X\le x_2)$$等于区间$$(x_1,x_2]$$上曲线$$y=f(x)$$之下的曲边梯形的面积；由性质4知$$f(x)$$的连续点处有
$$
f(x)=\lim_{\Delta x\to 0^+}\frac{F(x+\Delta x)-F(x)}{\Delta x}\\
=\lim_{\Delta x\to 0^+}\frac{P(x<X\le x+\Delta x)}{\Delta x}
$$
若忽略高阶无穷小，则有
$$
P(x<X\le x+\Delta x)\approx f(x)\Delta x
$$
需要注意的是，对于连续型随机变量$$X$$，它取任一指定实数值$$a$$的概率均为0，即$$P(X=a)=0$$，据此有
$$
P(a< x< b)=P(a< x\le b)=P(a\le x< b)=P(a\le x\le b)
$$
我们看到$$P(X=a)=0$$，但$$X=a$$并非不可能事件，因此$$P(A)=0$$是$$A$$是不可能事件的必要不充分条件。



## 均匀分布

若连续型随机变量$$X$$具有概率密度
$$
f(x)=\cases{
\frac{1}{b-a},\ a<x<b\\
0,\quad\ 其它
}
$$
则称$$X$$在区间$$(a,b)$$上服从**均匀分布(uniform distribution)**，记作$$X\sim U(a,b)$$。

$$X$$的分布函数为
$$
F(x)=\cases{0,\quad\ x<a\\
\frac{x-a}{b-a},\ a\le x<b\\
1,\quad\ x\ge b
}
$$


## 指数分布

若连续型随机变量$$X$$具有概率密度
$$
f(x)=\cases{\lambda e^{-\lambda x},\ x>0\\
0,\quad\quad\ \ 其它
}
$$
其中$$\lambda>0$$为常数，则称$$X$$服从参数为$$\lambda$$的**指数分布(exponential distribution)**，记作$$X\sim Exp(\lambda)$$。

$$X$$的分布函数为
$$
F(x)=\cases{1-e^{-x/\theta},\ x>0\\
0,\quad\quad\quad\ \ 其它
}
$$
对于任意$$s,t>0$$，有
$$
P(X>s+t|X>s)=\frac{P(X>s+t\and X>s)}{P(X>s)}=\frac{P(X>s+t)}{P(X>s)}\\
=\frac{1-F(s+t)}{1-F(s)}=\frac{e^{-(s+t)/\theta}}{e^{-s/\theta}}=e^{-t/\theta}=P(X>t)
$$
上式称为随机变量$$X$$的**无记忆性**。若$$X$$是某一元件的寿命，则元件能够继续使用的时长的概率分布与元件已经使用的时长无关。



## 正态分布

若连续型随机变量$$X$$具有概率密度
$$
f(x)=\frac{1}{\sqrt{2\pi}\sigma}e^{-(x-\mu)^2/(2\sigma^2)},\ -\infty<x<+\infty
$$
其中$$\mu,\sigma\ (\sigma>0)$$为常数，则称$$X$$服从参数为$$\mu,\sigma$$的**正态分布(normal distribution)**或**高斯分布(Gauss distribution)**，记作$$X\sim N(\mu,\sigma^2)$$。

正太分布具有以下性质：

1. 曲线$$y=f(x)$$关于$$x=\mu$$对称

2. 当$$x=\mu$$时取到最大值
   $$
   f(\mu)=\frac{1}{\sqrt{2\pi}\sigma}
   $$

$$X$$的分布函数为
$$
F(x)=\frac{1}{\sqrt{2\pi}\sigma}\int_{-\infty}^x e^{-(t-\mu)^2/(2\sigma^2)}{\rm d}t
$$
特别地，当$$\mu=0,\sigma=1$$时称随机变量$$X$$服从**标准正态分布**，其概率密度函数和分布函数分别用$$\varphi(x),\Phi(x)$$表示，即
$$

$$




## 伽马分布



