> 参考：
>
> [OI Wiki](https://oi-wiki.org/math/)

[toc]

# 基础

### 度

多项式 $f(x)$ 的最高次项的次数为该多项式的**度(degree)** ，记作 $\deg f$。



### 乘法

多项式最核心的操作是多项式乘法，即给定多项式 $f(x)$ 和 $g(x)$ 
$$
f(x)=a_0+a_1x+\cdots+a_nx^n\\
g(x)=b_0+b_1x+\cdots+b_mx^m
$$
那么
$$
Q(x)=\sum_{i=0}^n\sum_{j=0}^ma_ib_jx^{i+j}=c_0+c_1x+\cdots+c_{n+m}x^{n+m}
$$
上述过程的蛮力算法的时间复杂度为 $O(nm)$，采用FFT方法计算的时间复杂度为 $O(n\log n)$ （设 $m=n$ ）。

> 参见常见算法-多项式乘法



### 逆

设多项式 $f(x)$ 的度为 $n$，若存在 $g(x)$ 满足
$$
f(x)g(x)\equiv 1\mod x^n\\
\deg g\le \deg f
$$
则称 $g(x)$ 为 $f(x)$ 在模 $x^n$ 意义下的**逆元(inverse element)**，记作 $f^{-1}(x)$。



### 除法

设多项式 $g(x)\neq0$，则对于一个任意多项式 $f(x)$，<u>存在唯一</u>的多项式 $q(x),r(x)$，使得
$$
f(x)=g(x)q(x)+r(x)\\
\deg q=\deg f-\deg g\\
\deg r<\deg g\ 或\ r(x)=0
$$


### 指数函数和对数函数

多项式 $f(x)$ 的指数函数可以定义为级数
$$
\exp f(x)=e^{f(x)}=\sum_{i=0}^\infty \frac{f^i(x)}{i!}
$$
类似地，对数函数可以定义为
$$
\ln(1-f(x))=-\sum_{i=1}^{\infty}\frac{f^i(x)}{i}
$$




# 插值interpolation

## 拉格朗日插值法



## 牛顿插值法





# 快速傅里叶变换

快速傅里叶变换可以在 $O(n\log n)$ 的时间内计算两个 $n$ 度多项式的乘法。

## 多项式的系数表示和点值表示

对于 $n$ 度多项式 $f(x)$ 
$$
f(x)=a_0+a_1x+\cdots+a_nx^n\\
\iff f(x)=\{a_0,a_1,\cdots,a_n\}\quad 系数表示法\\
\iff f(x)=\{(x_0,y_0),(x_1,y_1),\cdots(x_n,y_n)\}\quad 点值表示法
$$
将多项式由系数表示法转换为点值表示法的过程就是**离散傅里叶变换(Discrete Fourier Transform, DFT)**；由点值表示法转换为系数表示法的过程就是**离散傅里叶逆变换(Inverse Discrete Fourier Transform, IDFT)**。而FFT则是通过取某些特殊的 $(x,y)$ 点值来加速上述过程。



## 单位复根

问题是计算两个 $n$ 度多项式的乘法，我们发现点值表示法可以轻松地完成这一任务。即给定
$$
f(x)=\{(x_0,f(x_0)),(x_1,f(x_1)),\cdots,(x_n,f(x_n))\}\\
g(x)=\{(x_0,g(x_0)),(x_1,g(x_1)),\cdots,(x_n,g(x_n))\}
$$
设 $F(x)=f(x)g(x)$，那么容易得到其点值表达式
$$
F(x)=\{(x_0,f(x_0)g(x_0)),(x_1,f(x_1)g(x_1)),\cdots,(x_n,f(x_n)g(x_n))\}
$$
但是我们已知和待求的都是系数表达式，因此整个过程是两次DFT，一组乘法，和一次IDFT。



由于在DFT和IDFT过程中都需要计算 $f(x)$，因此需要选择性质好、计算快的 $x$。FFT即选择的是 $n$ 次单位复根 $\{w_n^i=e^{2\pi i/n}|i=0,1,\cdots,n-1\}$，具有以下性质
$$
w_n^0=w_n^n=1\\
w_n^k=w_{2n}^{2k}\\
w_{2n}^{k+n}=-w_{2n}^k
$$


## 快速傅里叶变换

FFT算法的基本思想是分治。在DFT中，它分治地求 $f(w_n^k)$，方法是将多项式分为奇次项和偶次项分别处理。

例如，对于7度多项式
$$
f(x)=a_0+a_1x+a_2x^2+a_3x^3+a_4x^4+a_5x^5+a_6x^6+a_7x^7
$$
按照次数的奇偶来分成两组，然后右边提出一个 $x$ 
$$
f(x)=(a_0+a_2x^2+a_4x^4+a_6x^6)+(a_1x+a_3x^3+a_5x^5+a_7x^7)\\
=(a_0+a_2x^2+a_4x^4+a_6x^6)+x(a_1+a_3x^2+a_5x^4+a_7x^6)
$$
然后设
$$
G(x)=a_0+a_2x+a_4x^2+a_6x^3\\
H(x)=a_1+a_3x+a_5x^2+a_7x^3
$$
那么
$$
f(x)=G(x^2)+xH(x^2)
$$
利用单位复根的性质得到
$$
f(w_n^k)=G(w_n^{2k})+w_n^kH(w_n^{2k})\\
=G(w_{n/2}^k)+w_n^kH(w_{n/2}^{k})
$$
同理可得
$$
f(w_n^{k+n/2})=G(w_n^{2k+n})+w_n^{k+n/2}H(w_n^{2k+n})\\
=G(w_{n/2}^k)-w_n^kH(w_{n/2}^{k})
$$
因此当我们求得 $G(w_{n/2}^k)$ 和 $H(w_{n/2}^{k})$ 时，就可以同时求出 $f(w_n^k)$ 和 $f(w_n^{k+n/2})$。于是继续对 $G$ 和 $H$ 递归DFT操作。

> 这里有 $T(n)=2T(n/2)+O(n)$，其中 $T(n)$ 指任务“对 $n-1$ 度多项式求在 $w_n^0,\cdots,w_n^{n-1}$ 这 $n$ 个位置上的值”



**度** 非零多项式 $f(x)=a_0+a_1x^1+\cdots+a_nx^n$ 的最高次称为度，记作 $deg(f(x))=n$ 

性质

![yi3jhgnjvhtt42rg47i](C:\Users\Xiao Yuxuan\Documents\pic\yi3jhgnjvhtt42rg47i.PNG)

> f(x)=0，则deg(f)不存在





# 卷积及其应用

## 卷积

$$
给定向量\;a=(a_0,a_1,\cdots,a_{m-1}),b=(b_0,b_1,\cdots,b_{n-1})\\
卷积\; a*b=(c_0,c_1,\cdots,c_{m+n-2})
\\其中c_k=\sum_{i+j=k}a_ib_j,k=0,1,\cdots,m+n-2\\
$$

卷积应用于信号平滑处理

## 卷积与多项式乘法

$$
A(x)=a_0+a_1x+a_2x^2+\cdots+a_{m-1}x^{m-1}\\
B(x)=b_0+b_1x+b_2x^2+\cdots+b_{n-1}x^{n-1}\\
C(x)=A(x)B(x)=a_0b_0+(a_0b_1+a_1b_0)x+\cdots+a_{m-1}b_{n-1}x^{m+n-2},\\其中x^k的系数为c_k=\sum_{i+j=k}a_ib_j
$$



