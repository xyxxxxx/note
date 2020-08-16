# 基础概念与常用符号





# 函数

……





# 极限论

……





# 连续函数

……





# 一元微分

## 导数

记作$$f'(x),\frac{{\rm d}f}{{\rm d}x}$$



### 运算法则

$$
\begin{align}
&(f+g)'=f'+g'\\
&(cf)'=cf'\\
&(fg)'=f'g+fg'\\
&(\frac{f}{g})'=\frac{f'g-fg'}{g^2}\\
&(f(g))'=f'(g)g'\\
&(f^{-1})'=\frac{1}{f'}
\end{align}
$$



### 常用函数的导数

$$
\begin{align}
&c'=0\\
&x'=1\\
&(x^p)'=px^{p-1}\\
&(a^x)'=a^x{\rm ln}a,(e^x)'=e^x \\
&({\rm log}_ax)'=\frac{1}{x{\rm ln}a},({\rm ln}x)'=\frac{1}{x}\\
&({\rm sin}x)'={\rm cos}x,({\rm cos}x)'=-{\rm sin}x\\
&({\rm tan}x)'={\rm sec^2}x,({\rm cot}x)'=-{\rm csc^2}x\\
&({\rm sec}x)'={\rm tan}x{\rm sec}x,({\rm cot}x)'=-{\rm cot}x{\rm csc}x\\
&({\rm arcsin}x)'=\frac{1}{\sqrt{1-x^2}},({\rm arccos}x)'=\frac{-1}{\sqrt{1-x^2}}\\
&({\rm arctan}x)'=\frac{1}{1+x^2},({\rm arccot}x)'=\frac{-1}{1+x^2}\\
\end{align}
$$



### 特殊的求导方法

- 对数求导法
- 参数方程求导法
- 隐函数求导法



### 高阶导数

$$
\begin{align}
&(f+g)^{(n)}=f^{(n)}+g^{(n)}\\
&(cf)^{(n)}=cf^{(n)}\\
&(fg)^{(n)}=\sum_{k=0}^n{\rm C}_n^kf^{(k)}g^{(n-k)}\\
\end{align}
$$



## 微分

（与导数相似）



## 微分中值定理

**极大值 驻点**

**拉格朗日中值定理 设函数$f$在闭区间$[a,b]$上连续，在开区间$(a,b)$内可导，则$\exists \xi \in (a,b)$使得**
$$
f'(\xi)=\frac{f(b)-f(a)}{b-a}
$$
**柯西中值定理 设函数$f$和$g$在闭区间$[a,b]$上连续，在开区间$(a,b)$内可导，且$g'(x)\neq0$，则$\exists \xi \in (a,b)$使得**
$$
\frac{f'(\xi)}{g'(\xi)}=\frac{f(b)-f(a)}{g(b)-g(a)}
$$



## 洛必达法则

**如果$\lim_{x \to c}f(x)=\lim_{x \to c}g(x)=0$或$\lim_{x \to c}\vert f(x)\vert =\lim_{x \to c}\vert g(x)\vert =\infty$，并且$f(x),g(x)$在$x=c$为端点的开区间可导，$g(x)\neq0$，那么**
$$
\lim_{x \to c}\frac{f(x)}{g(x)}=\lim_{x \to c}\frac{f'(x)}{g'(x)}
$$
**如果$\lim_{x \to \infty}f(x)=\lim_{x \to \infty}g(x)=0$，并且$f(x),g(x)$在$\Bbb R$上可导，$g(x)\neq0$，那么**
$$
\lim_{x \to \infty}\frac{f(x)}{g(x)}=\lim_{x \to \infty}\frac{f'(x)}{g'(x)}
$$
不符合分数形式的不定式，可以通过通分、取对数等方法转化为分数形式，再以本法求值。



## 函数极值

……



## 函数图形

**上凸函数 下凸函数**



## 泰勒公式

设函数$f$$x_0$$(a,b)$$1$$$n+1$$$\forall x \in (a,b)$
$$
f(x)=f(x_0)+f'(x_0)(x-x_0)+\frac{1}{2!}f''(x_0)(x-x_0)^2+\cdots+\frac{1}{n!}f^{n}(x_0)(x-x_0)^n+\frac{1}{(n+1)!}f^{n+1}(\xi)(x-x_0)^{n+1}\\
e^x = 1+x+\frac{x^2}{2!}+\frac{x^3}{3!}+\cdots
$$





# 一元积分

## 不定积分

如果存在函数$F(x)$，使得$F'(x)=f(x)$在区间$I$上处处成立，则称在区间$I$上$F(x)$是$f(x)$的一个**原函数**。

在区间$I$上，所有满足${\rm d}F(x)=f(x){\rm d}x$的函数$F(x)$构成的函数族称为$f(x){\rm d}x$在区间$I$上的**不定积分**。

**基本积分公式**
$$
\int0{\rm d}x=C\\
\int {\rm d}x=x\\
\int x^p{\rm d}x=\frac{1}{p+1}x^{p+1}+C\\
\int \frac{{\rm d}x}{x}=\ln \vert x\vert+C\\
\int e^x{\rm d}x=e^x+C\\
\int a^x{\rm d}x=\frac{1}{\ln a}a^x+C\\
\int \sin x{\rm d}x=-\cos x+C\\
\int \cos x{\rm d}x=\sin x+C\\
\int \sec^2x{\rm d}x=\tan x+C\\
\int \csc^2x{\rm d}x=-\cot x+C\\
\int \tan x \sec x{\rm d}x=\sec x+C\\
\int \cot x \csc x {\rm d}x=-\csc x+C\\
\int \frac{{\rm d}x}{\sqrt{1-x^2}}=\arcsin x+C=-\arccos x+C\\
\int \frac{{\rm d}x}{1+x^2}=\arctan x+C=-{\rm acrcot}x+C
$$



### 换元积分法

**第一换元法** $\int f(x){\rm d}x=\int g(u){\rm d}u=G(u)+C=G(u(x))+C$

**第二换元法** $\int f(x){\rm d}x=\int f(\varphi (t))\varphi' (t){\rm d}t=G(\varphi^{-1}(x))+C$



### 分部积分法

$\int u{\rm d}v=uv-\int v{\rm d}u$

> u——幂函数 v——可积



### 有理函数的积分

分式：化为最简分式求得

三角函数：通过三角恒等式求得



## 定积分

### **定积分**

**积分中值定理 设$f\in C[a,b]$，则存在点$\xi \in [a,b]$，使得**
$$
\int_a^b f(x){\rm d}x=f(\xi)(b-a)
$$
**设$f\in C[a,b]$，函数$g(x)$在区间$[a,b]$上可积且不变号，则存在点$\xi \in [a,b]$，使得**
$$
\int_a^b f(x)g(x){\rm d}x=f(\xi)\int_a^b g(x){\rm d}x
$$



### 牛顿—莱布尼兹公式

**设$f\in C[a,b]$，如果$G(x)$是$f(x)$在区间$[a,b]$上的一个原函数，则**
$$
\int_a^bf(x){\rm d}x=G(b)-G(a)
$$



### 换元积分法与分部积分法

> 与不定积分的换元积分法相似，注意积分上下限为变量的上下限。



### 几何应用

曲率
$$
k=\left | \frac{y''}{[1+(y')^2]^{3/2}} \right |
$$



### 无穷积分

**收敛性的判定**

1. 对于无穷积分$\int_a^{+\infty}f(x){\rm d}x$：
   如果能找到非负函数$g(x)$，满足$0 \le f(x) \le g(x)(x \ge a)$，且无穷积分$\int_a^{+\infty}g(x){\rm d}x$收敛，则无穷积分$\int_a^{+\infty}f(x){\rm d}x$收敛；
   如果能找到非负函数$g(x)$，满足$0 \le g(x) \le f(x)(x \ge a)$，且无穷积分$\int_a^{+\infty}g(x){\rm d}x$发散，则无穷积分$\int_a^{+\infty}f(x){\rm d}x$发散；
2. 对于无穷积分$\int_a^{+\infty}f(x){\rm d}x,f(x) \ge 0$：
   如果能找到非负函数$g(x)$，极限$\lim_{x \to +\infty}\frac{f(x)}{g(x)}$存在，且无穷积分$\int_a^{+\infty}g(x){\rm d}x$收敛，则无穷积分$\int_a^{+\infty}f(x){\rm d}x$收敛；
   如果能找到非负函数$g(x)$，极限$\lim_{x \to +\infty}\frac{f(x)}{g(x)}$存在且不等于0，且无穷积分$\int_a^{+\infty}g(x){\rm d}x$发散，则无穷积分$\int_a^{+\infty}f(x){\rm d}x$发散；
3. 对于无穷积分$\int_a^{+\infty}f(x){\rm d}x,f(x) \ge 0$：
   如果$\exists p>1$，使得$\lim_{x \to \infty}x^pf(x)$存在，则$\int_a^{+\infty}f(x){\rm d}x$收敛；
   如果$\exists p\le 1$，使得$\lim_{x \to \infty}x^pf(x)$存在且不等于0，则$\int_a^{+\infty}f(x){\rm d}x$发散；
   （因为对于$f(x)=x^p,p \ge -1 $时$\int_a^{+\infty}f(x){\rm d}x$发散，$p<-1$时收敛）
4. 柯西收敛准则 无穷积分$\int_a^{+\infty}f(x){\rm d}x$收敛的充要条件是：对于任意正数$\epsilon$都能找到正数$N$，使得对任意满足$A_1>N,A_2>N$的实数$A_1,A_2$都有$\left | \int_{A_1}^{A_2}f(x){\rm d}x \right |<\epsilon$。





# 级数论

## 概念与性质

假设${u_n}$是一数列，数列中所有数依次相加：
$$
\sum_{n=1}^{\infty}u_n=u_1+u_2+\cdots+u_n+\cdots
$$
就称为级数。

**收敛 发散**

级数$\sum_{n=1}^{\infty}u_n,\sum_{n=1}^{\infty}v_n$收敛，则有

1. $$
   \lim_{x\to \infty }u_n=0
   $$

2. $$
   \sum_{n=1}^{\infty}(au_n+bv_n)=a\sum_{n=1}^{\infty}u_n+b\sum_{n=1}^{\infty}v_n
   $$

   

3. 求和时满足结合律

4. 级数是否收敛与前面有限项无关



### 特殊级数

几何级数
$$
\sum_{n=0}^{\infty}aq^{n-1}=a+aq+aq^2+\cdots+aq^n+\cdots=
\begin{cases}
\frac{a}{1-q},& \text{if $\left | q \right |<1$}\\
(发散),&\text{if $\left | q \right | \ge1$}
\end{cases}
$$
p级数
$$
\sum_{n=1}^{\infty}\frac{1}{n^p}=1+\frac{1}{2^p}+\frac{1}{3^p}+\cdots+\frac{1}{n^p}+\cdots=
\begin{cases}
(收敛),& \text{if $p>1$}\\
(发散),&\text{if $p \le1$}
\end{cases}
$$



## 正项级数的收敛判别

1. 对于正项级数$\sum_{n=1}^{\infty}u_n$：
   如果能找到收敛的正项级数$\sum_{n=1}^{\infty}v_n$，使得当n充分大时恒有$u_n \le cv_n$，其中c为正数，则$\sum_{n=1}^{\infty}u_n$收敛；
   如果能找到发散的正项级数$\sum_{n=1}^{\infty}v_n$，使得当n充分大时恒有$u_n \ge cv_n$，其中c为正数，则$\sum_{n=1}^{\infty}u_n$发散；

2. 对于正项级数$\sum_{n=1}^{\infty}u_n$：
   如果能找到收敛的正项级数$\sum_{n=1}^{\infty}v_n$，使得极限$\lim_{x \to +\infty}\frac{u_n}{v_n}$存在，则$\sum_{n=1}^{\infty}u_n$收敛；
   如果能找到发散的正项级数$\sum_{n=1}^{\infty}v_n$，使得极限$\lim_{x \to +\infty}\frac{u_n}{v_n}$存在且不等于0，则$\sum_{n=1}^{\infty}u_n$发散；

3. 对于正项级数$\sum_{n=1}^{\infty}u_n$：
   如果$\exists p>1$，使得当n充分大时恒有$0 \le u_n \le c\frac{1}{n^p}$，其中c为正数，或使得$\lim_{n \to \infty}n^pu_n$存在，则$\sum_{n=1}^{\infty}u_n$收敛；
   如果$\exists p\le1$，使得当n充分大时恒有$u_n \ge c\frac{1}{n^p}$，其中c为正数，或使得$\lim_{n \to \infty}n^pu_n$存在且不等于0,则$\sum_{n=1}^{\infty}u_n$发散；

   （因为对于$\sum_{n=1}^{\infty}\frac{1}{n^p},p\le 1$时发散，$p>1$时收敛）

   > 与无穷积分收敛的判别方法相似。

4. 对于正项级数$\sum_{n=1}^{\infty}u_n$，$\lim_{n \to \infty}\frac{u_{n+1}}{u_n}=q$存在，则有：
   如果$q<1$，则级数$\sum_{n=1}^{\infty}u_n$收敛；
   如果$q>1$，则级数$\sum_{n=1}^{\infty}u_n$发散；

   

## 任意项级数

### 交错级数

若交错级数$\sum_{n=1}^{\infty}(-1)^{n-1}u_n(u_n>0)$中的数列${u_n}$单调减少且趋向于0，则有

1. $\sum_{n=1}^{\infty}(-1)^{n-1}u_n$收敛
2. $0\le\sum_{n=1}^{\infty}(-1)^{n-1}u_n\le u_1$

## 函数级数



## 幂级数



## 傅里叶级数

## ……





# 常用解法

$$
泰勒公式f(x)=f(x_0)+f'(x_0)(x-x_0)+\frac{1}{2!}f''(x_0)(x-x_0)^2+\cdots+\frac{1}{n!}f^{n}(x_0)(x-x_0)^n+\frac{1}{(n+1)!}f^{n+1}(\xi)(x-x_0)^{n+1}\\
e^x = 1+x+\frac{x^2}{2!}+\frac{x^3}{3!}+\cdots\\
\sin x=x-\frac{x^3}{3!}+\frac{x^5}{5!}-\frac{x^7}{7!}+\cdots\\
\cos x=1-\frac{x^2}{2!}+\frac{x^4}{4!}-\frac{x^6}{6!}+\cdots\\
\ln (1+x)=x-\frac{x^2}{2}+\frac{x^3}{3}-\frac{x^4}{4}+\cdots\\
$$

