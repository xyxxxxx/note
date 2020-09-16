# 概率,probability,確率

$$
P(B-A)=P(B)-P(A), if\;A\subseteq B
\\P(A\bigcup B)=P(A)+P(B)-P(AB)
\\P(B\vert A)=\frac{P(AB)}{P(A)}
\\P(A_1A_2\cdots A_n)=P(A_1)P(A_2\vert A_1)P(A_3\vert A_1A_2) \cdots
\\全概率公式 P(A)=P(B_1)P(A\vert B_1)+P(B_2)P(A\vert B_2)+\cdots+P(B_n)P(A\vert B_n)
\\贝叶斯公式 P(B_i\vert A)=\frac{P(B_iA)}{P(A)}=\frac{P(B_i)P(A\vert B_i)}{\sum P(B_k)P(A\vert B_k)}
\\独立P(AB)=P(A)P(B)
$$





# 离散型随机变量,discrete random variable,確率変数

> 随机变量的取值亦可是随机变量

## 二项分布,binomial distribution,二項分布

$$
X\sim B(n,p)
\\P(X=k)=C_n^kp^k(1-p)^{n-k}
\\E(X)=np,Var(X)=np(1-p)
\\可加性~if~X_1\sim B(n_1,p),X_2\sim B(n_2,p),then ~X_1+X_2\sim B(n_1+n_2,p)
$$

> n个伯努利分布相加



## 几何分布geometric distribution

$$
X\sim G(p)\\
P(X=k)=(1-p)^{k-1}p\\
E(X)=\frac{1}{p},Var(X)=\frac{1-p}{p^2}
$$



## 超几何分布hyper geometric distribution

$$
X\sim H(n,K,N)\\
P(X=k)=\frac{C_K^kC_{N-K}^{n-k}}{C_N^n}\\
E(X)=n\frac{K}{N}
$$



## 负二项分布

$$
X\sim NB(n,p)\\
P(X=k)=C_{k-1}^{k-n}(1-p)^{k-n}p^n\\
E(X)=n\frac{1}{p},Var(X)=n\frac{1-p}{p^2}
$$

> n个几何分布相加



## 泊松分布Poisson distribution$$

$$
X\sim P(\lambda)\\
P(X=k)=\frac{e^{-\lambda}\lambda^k}{k!}\\
E(X)=Var(X)=\lambda\\

泊松定理 \lim_{n \to \infty}C_n^kp^k(1-p)^{n-k}=\frac{\lambda^ke^{-\lambda}}{k!}\\
可加性 if \; X_1\sim P(\lambda_1),X_2\sim P(\lambda_2),then \; X_1+X_2\sim P(\lambda_1+\lambda_2)
$$

> 二项分布在n较大，p较小时的近似





# 连续型随机变量continuous random variable

概率密度函数（probability density function，確率密度関数）$$f(x)$$，累积分布函数（cumulative distribution function）$$F(x)$$
$$
F(x)=\int_{-\infty}^{x}f(t)dt\\
F'(x)=f(x)\\
$$


## 均匀分布uniform distribution

$$
X\sim U(a,b)\\
f(x)=\frac{1}{b-a},~X\in[a,b]\\
E(X)=\frac{a+b}{2},~Var(X)=\frac{(b-a)^2}{12}
$$



## 指数分布exponential distribution

$$
X\sim Exp(\lambda)\\
f(x)=\lambda e^{-\lambda x}\quad(x>0)\\
E(X)=\frac{1}{\lambda},~Var(X)=\frac{1}{\lambda^2}
$$



## 正态分布,normal distribution,正規分布

$$
标准正态分布X\sim N(0,1)\\
\varphi(x)=\frac{1}{\sqrt{2\pi}}e^{-\frac{x^2}{2}}\\
E(X)=0, ~Var(X)=1\\
正态分布X\sim N(\mu,\sigma^2)\\
E(X)=\mu,~Var(X)=\sigma^2
$$

**定理** 设随机变量$$X_1,X_2,\cdots,X_n$$相互独立，且$$X_i\sim N(\mu_i,\sigma_i^2)$$，则$$\sum X_i\sim N(\sum\mu_i,\sum \sigma_i^2)$$



### 二元正态分布

**二元正态分布的性质**

+ 二元正态分布的边缘分布为一元正态分布
+ 如果$$(X,Y)\sim N(\mu_1,\mu_2,\sigma_1^2,\sigma_2^2,\rho)$$，则X,Y相互独立当且仅当$$Cov(X,Y)=0$$
+ 如果$$(X,Y)\sim N(\mu_1,\mu_2,\sigma_1^2,\sigma_2^2,\rho)$$，则$$(a_1X+b_1Y,a_2X+b_2Y)$$也服从二元正态分布，其中$$aX+bY\sim N(a\mu_1+b\mu_2,a^2\sigma_1^2+b^2\sigma_2^2+2ab\rho\sigma_1\sigma_2)$$





# 多维随机变量,n-dimensional random variable

## 二维离散型随机变量

**联合分布$$P(X=x_i,Y=y_j)$$**

**边缘分布$$P(X=x_i)$$**



**独立性**
$$
P(X=x_i,Y=y_i)=P(X=x_i)P(Y=y_i)
$$
表明随机变量$$X,Y$$独立



### 多项分布

$$
(X_1,X_2,\cdots,X_m)\sim M(n,p_1,p_2,\cdots,p_m)\\
P(X_1=k_1,\cdots,X_m=k_m)=\frac{n!}{k_1!k_2!\cdots k_m!}p_1^{k_1}p_2^{k_2}\cdots p_m^{k_m}\\
$$

> 多项分布的边缘分布为二项分布



## 二维连续型随机变量



![xzcngoj5y3htbewfe](C:\Users\Xiao Yuxuan\Documents\pic\xzcngoj5y3htbewfe.PNG)

**独立性**
$$
f(x_i,y_i)=f_X(x_i)f_Y(y_i)
$$
表明随机变量$$X,Y$$独立



### 二维均匀分布



### 二元正态分布

$$
(X,Y)\sim N(\mu_1,\mu_2,\sigma_1^2,\sigma_2^2,\rho)
$$



## 条件分布

$$
离散\quad P(y|x)=P(Y=y|X=x)=\frac{P(x,y)}{P(x)}\\
连续\quad f(y|x)=\frac{f(x,y)}{f_X(x)}
$$



### 条件独立

$$
P(x, y|z) = P(x|z)P(y|z)\\
f(x,y|z)=f(x|z)f(y|z)
$$



### 条件数学期望

![c0dji35yt24nefjvwr](C:\Users\Xiao Yuxuan\Documents\pic\c0dji35yt24nefjvwr.PNG)

![gjin24ojfevwfjrgq](C:\Users\Xiao Yuxuan\Documents\pic\gjin24ojfevwfjrgq.PNG)

![c9i0ieyijo5n3jgrebtoh](C:\Users\Xiao Yuxuan\Documents\pic\c9i0ieyijo5n3jgrebtoh.PNG)



### 全期望公式

$$
E(X)=E(E(X|Y))
$$

![drie9f0rijgt42grb](C:\Users\Xiao Yuxuan\Documents\pic\drie9f0rijgt42grb.PNG)

![dbgjhitnj35otgrwv](C:\Users\Xiao Yuxuan\Documents\pic\dbgjhitnj35otgrwv.PNG)

![npkn35bfdfwerfed](C:\Users\Xiao Yuxuan\Documents\pic\npkn35bfdfwerfed.PNG)

![vcxjiotn35itojneor](C:\Users\Xiao Yuxuan\Documents\pic\vcxjiotn35itojneor.PNG)

![gsroijn436hgdrt4](C:\Users\Xiao Yuxuan\Documents\pic\gsroijn436hgdrt4.PNG)





![vcbf45jiontgfdf](C:\Users\Xiao Yuxuan\Documents\pic\vcbf45jiontgfdf.PNG)

![ejti3to4n3rvrfwwvwfb](C:\Users\Xiao Yuxuan\Documents\pic\ejti3to4n3rvrfwwvwfb.PNG)

![hrt903jierwnvirvb](C:\Users\Xiao Yuxuan\Documents\pic\hrt903jierwnvirvb.PNG)



## 随机变量函数

### 期望和方差,expectation & variance,平均と分散

$$
E(X)=\sum x_kp_k=\int_{\mathbb{R}} xf(x){\rm d}x\\
E(X+Y)=E(X)+E(Y)\\
独立E(XY)=E(X)E(Y)\\
Var(X)=E(X^2)-E^2(X),\sigma(X)=\sqrt{Var(X)}\\
Var(aX+b)=a^2Var(X)\\
Var(X\pm Y)=Var(X)+Var(Y)\pm 2Cov(X,Y)\\
独立Var(X+Y)=Var(X)+Var(Y)
$$

对于离散型随机变量
$$
Var(X)=\sum_{i=1}^np_i(x_i-\mu)^2
$$
对于连续型随机变量
$$
Var(X)=\int_{-\infty}^{+\infty}(x-\mu)^2f(x){\rm d}x
$$




![130rjitogrfobqqferwvfwin](C:\Users\Xiao Yuxuan\Documents\pic\130rjitogrfobqqferwvfwin.PNG)

**Jensen不等式**

对于凸函数$$g$$，
$$
g(E(X))\le E(g(X))
$$
等式当且仅当$$X$$是常数或$$g$$是线性时成立



### 协方差,covariance

$$
协方差Cov(X,Y)=E(XY)-E(X)E(Y)\\
Cov(aX+b,cY+d)=acCov(X,Y)\\
分配律Cov(X+Y,Z)=Cov(X,Z)+Cov(Y,Z)\\
独立Cov(X,Y)=0
\\相关系数Corr(X,Y)=\frac{Cov(X,Y)}{\sqrt{Var(X)Var(Y)}}\quad\\
Corr(X,Y)>0正相关,Corr(X,Y)<0负相关,Corr(X,Y)=0不相关
$$

=∫+∞−∞(t−μ)2fX(t)dt

对于$$M$$维和$$N$$维随机变量$$\pmb X$$和$$\pmb Y$$，
$$
Cov(\pmb X,\pmb Y)=E((\pmb X-E(\pmb X))(\pmb Y-E(\pmb Y))^{\rm T})
$$
即$$Cov(\pmb X,\pmb Y)$$的第$$(m,n)$$个元素是$$X_m$$和$$Y_n$$的协方差。



### 相关和独立

> 不相关可能不独立
>
> 相关系数反映的是<u>线性相关意义</u>下的相关程度，不能表达非线性的相关关系

![sdfvj2it4njhoy3j47i](C:\Users\Xiao Yuxuan\Documents\pic\sdfvj2it4njhoy3j47i.PNG)



## 独立随机变量的和分布

### 离散型

$$
P(Z=z_k)=\sum_{x_i\in X(\Omega)}P(X=x_i)P(Y=z_k-x_i)
$$



### 连续型

$$
卷积公式~f_X*f_Y=\int_{-\infty}^{+\infty}f_X(z-y)f_Y(y)dy
$$

![dbgnh35yj6hoiy5g](C:\Users\Xiao Yuxuan\Documents\pic\dbgnh35yj6hoiy5g.PNG)

![fsvebjit35yno6435yt24gr](C:\Users\Xiao Yuxuan\Documents\pic\fsvebjit35yno6435yt24gr.PNG)

## 最大值和最小值分布

![cxvbeihno63y5mknjhto](C:\Users\Xiao Yuxuan\Documents\pic\cxvbeihno63y5mknjhto.PNG)





# 大数定律与中心极限定理

## 大数定律,large numbers law

设$$X_1,X_2,\cdots$$是相互独立，服从同一分布的随机变量序列，且具有数学期望$$E(X_k)=\mu(k=1,2,\cdots)$$，作前n个变量的算数平均$$\frac{1}{n}\sum_{k=1}^{n}X_k$$，则对于任意$$\varepsilon>0$$，有
$$
\lim_{n\to \infty}P(\left | \frac{1}{n}\sum_{k=1}^nX_k-\mu \right |<\varepsilon)=1
$$

> 证明：使用切比雪夫不等式

**切比雪夫不等式**
$$
\forall \varepsilon>0,\ P(|X-\mu)|\ge \varepsilon)\le \frac{Var(X)}{\varepsilon^2}
$$

> 证明：以连续型随机变量$$X$$为例
> $$
> Var(X)=\int_{-\infty}^{+\infty}(t-\mu)^2f_X(t){\rm d}t\\
> \ge \int_{-\infty}^{\mu-\varepsilon}(t-\mu)^2f_X(t){\rm d}t+\int_{\mu+\varepsilon}^{+\infty}(t-\mu)^2f_X(t){\rm d}t\\
> \ge \int_{-\infty}^{\mu-\varepsilon}\varepsilon^2f_X(t){\rm d}t+\int_{\mu+\varepsilon}^{+\infty}\varepsilon^2f_X(t){\rm d}t\\
> = \varepsilon^2(\int_{-\infty}^{\mu-\varepsilon}f_X(t){\rm d}t+\int_{\mu+\varepsilon}^{+\infty}f_X(t){\rm d}t)\\
> = \varepsilon^2P(X\le\mu-\varepsilon {\rm~or~} X\ge\mu+\varepsilon)\\
> = \varepsilon^2 P(|X-\mu|\ge \varepsilon)\\
> \therefore P(|X-\mu|\ge \varepsilon) \le \frac{Var(X)}{\varepsilon^2}
> $$



> 大数定律的包含的意义是：频率收敛于概率



## 中心极限定理

独立同分布的中心极限定理

设随机变量$$X_1,X_2,\cdots,X_n,\cdots$$相互独立，服从同一分布，且具有数学期望和方差：$$E(X_k)=\mu,Var(X_k)=\sigma^2>0(k=1,2,\cdots)$$，则随机变量的算数平均$$\overline{X}=\frac{1}{n}\sum_{k=1}^{n}X_k$$满足
$$
\overline{X} \sim N(\mu,\sigma^2/n)
$$
李雅普诺夫定理