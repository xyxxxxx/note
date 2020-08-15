# 基础



# 插值interpolation

## 多项式的系数表示和点值表示

以二次函数为例
$$
f(x)=a_0+a_1x+a_2x^2 \Leftrightarrow f(x)=\{a_0,a_1,a_2 \}\\
\Leftrightarrow f(x)=\{(x_0,y_0),(x_1,y_1),(x_2,y_2) \}
$$
将多项式由系数表示法转换为点值表示法的过程就是DFT；由点值表示法转换为系数表示法的过程就是IDFT

FFT则是通过取某些特殊的(x,y)点值来加速DFT过程

## 快速傅里叶变换



# 求逆



# 开方



# 除法

**度** 非零多项式$f(x)=a_0+a_1x^1+\cdots+a_nx^n$的最高次称为度，记作$deg(f(x))=n$

性质

![yi3jhgnjvhtt42rg47i](C:\Users\Xiao Yuxuan\Documents\pic\yi3jhgnjvhtt42rg47i.PNG)

> f(x)=0，则deg(f)不存在

**带余除法** 设多项式$g(x)\neq0$，则对于一个任意多项式$f(x)$，<u>存在唯一</u>的多项式$q(x),r(x)$，使得$f(x)=g(x)q(x)+r(x)$，其中$r(x)=0或deg(r(x))<deg(g(x))$



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



