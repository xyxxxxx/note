# 复变函数

## 复数

**复数complex** z=x+iy=Re z+iIm z

**实部Re, 虚部Im** Re z=x, Im z=y

**复共轭complex conjugation/c.c.** $$\overline{z}=x-iy$$
$$
x=\frac{z+\overline{z}}{2},y=\frac{z-\overline{z}}{2i}
$$
**复平面complex plane, 实轴, 虚轴**

复向量←→复平面的点←→复数z=x+iy

平面坐标(x,y)←→极坐标($$\rho,\varphi$$)，$$\varphi$$辐角，记作Arg z，其中arg z$$\in [0,2\pi)$$，$$\rho$$模，记作$$|z|$$
$$
x=\rho \cos \varphi,y=\rho \sin \varphi\\
\rho^2=x^2+y^2,\tan\varphi=y/x\\
$$
**三角式**
$$
z=\rho(\cos \varphi+i\sin\varphi)
$$
**指数式**
$$
z=\rho e^{i\varphi}
$$
**无限远点** 模无穷大，没有辐角

**复数运算**

加法的交换律和结合律成立，且有$$|z_1+z_2|\le |z_1|+|z_2|$$

减法$$|z_1-z_2|\ge |z_1|-|z_2|$$

乘法的交换律，结合律和分配律成立

## 复变函数

**复变函数** $$w=f(z),z\in E$$

**初等函数**
$$
e^z=e^{x+iy}=e^x(\cos y+i\sin y)\\
\sin z=\frac{1}{2i}(e^{iz}-e^{-iz})\\
\cos z=\frac{1}{2}(e^{iz}+e^{-iz})\\
sh~z=\frac{1}{2}(e^x-e^{-x}) \\
ch~z=\frac{1}{2}(e^x+e^{-x})\\
\ln z=\ln |z|+iArg~z\\
z^s=e^{s\ln z}
$$

+ sin z和cos z具有实周期$$2\pi$$；模可以大于1
+ e^z, sh z, ch z具有虚周期$$2\pi i$$
+ ln z有无限多值

## 导数

$$
\lim_{\Delta z→0}\frac{\Delta w}{\Delta z}
$$

复变函数f(z)可导的充分必要条件：f(z)的偏导数存在，且连续，且满足**C-R条件**：
$$
\frac{\part u}{\part x}=\frac{\part v}{\part y}\\
\frac{\part v}{\part x}=-\frac{\part u}{\part y}
$$
**初等函数导数**
$$
\frac{d}{dz}z^n=nz^{n-1}\\
\frac{d}{dz}e^z=e^z\\
\frac{d}{dz}\sin z=\cos z\\
\frac{d}{dz}\cos z=-\sin z\\
\frac{d}{dz}\ln z=\frac{1}{z}
$$

## 解析函数

**解析** f(z)在点z0及其邻域上处处可导

**解析函数** f(z)在区域B上的每一点都解析

> 函数在某一区域上可导和解析是等价的







## 平面标量场

## 多值函数





# 积分



# 幂级数展开



# 留数定理





# 傅里叶变换

## 傅里叶级数

**周期函数的傅里叶展开**

函数$$f(x)$$以$$2l$$为周期，即$$f(x+2l)=f(x)$$

则取三角函数族
$$
1,\cos\frac{\pi x}{l},\cos\frac{2\pi x}{l},\cdots,\cos\frac{k\pi x}{l},\cdots\\
\sin\frac{\pi x}{l},\sin\frac{2\pi x}{l},\cdots,\sin\frac{k\pi x}{l},\cdots
$$
将$$f(x)$$展开为级数
$$
f(x)=a_0+\sum_{k=1}^{\infty}(a_k\cos\frac{k\pi x}{l}+b_k\sin\frac{k\pi x}{l})
$$
函数族是正交的，即任意两个函数的乘积在一个周期上的积分等于0

展开系数为
$$
a_k=\frac{1}{\delta_kl}\int_{-l}^{l}f(\xi)\cos\frac{k\pi \xi}{l}d\xi\\
b_k=\frac{1}{l}\int_{-l}^{l}f(\xi)\sin\frac{k\pi \xi}{l}d\xi\\
\delta_k=2(k=0),1(k\neq0)
$$
傅里叶级数**平均收敛**于$$f(x)$$

> 平均收敛并不一定收敛

**Dirichlet conditions** 若$$f(x)$$满足：(1)处处连续，或在每个周期内只有有限个第一类间断点；(2)每个周期内只有有限个极值点，则级数**收敛**，且级数和$$=(f(x+0)+f(x-0))/2$$



**奇函数和偶函数的傅里叶展开**

若周期函数是奇函数，则$$a_0=a_k=0$$，有
$$
f(x)=\sum_{k=1}^{\infty}b_k\sin\frac{k\pi x}{l}
$$
展开系数为
$$
b_k=\frac{2}{l}\int_{0}^{l}f(\xi)\sin\frac{k\pi \xi}{l}d\xi\\
$$


若周期函数是偶函数，则$$b_k=0$$，有
$$
f(x)=a_0+\sum_{k=1}^{\infty}a_k\cos\frac{k\pi x}{l}
$$
展开系数为
$$
a_k=\frac{2}{\delta_kl}\int_0^lf(\xi)\cos\frac{k\pi \xi}{l}d\xi
$$


**定义在有限区间上函数的傅里叶展开**

（延拓为周期函数



**复数形式的傅里叶级数**

取复指数函数族
$$
e^{-ik\pi x/l},\cdots,e^{-i\pi x/l},1,e^{i\pi x/l},\cdots,e^{ik\pi x/l}
$$
$$f(x)$$展开为级数
$$
f(x)=\sum_{k=-\infty}^{\infty}c_ke^{ik\pi x/l}
$$
函数族是正交的

## 傅里叶变换

### 实数形式的傅里叶变换

条件：$$f(x)$$在任一有限区间满足**Dirichlet conditions**，并在$$(-\infty,\infty)$$区间上绝对可积.

将$$f(x)$$看作是某个周期函数，满足$$2l→\infty$$，则
$$
f(x)=a_0+\sum_{k=1}^{\infty}(a_k\cos \omega_kx+b_k\sin\omega_kx)
$$
代入傅里叶系数表达式，取$$l→\infty$$的极限，得到**傅里叶积分**和**傅里叶变换式**
$$
f(x)=\int_0^\infty A(\omega)\cos \omega xd\omega+\int_0^\infty B(\omega)\sin \omega xd\omega\\
A(\omega)=\frac{1}{\pi}\int_{-\infty}^{\infty}f(\xi)\cos \omega \xi d\xi\\
B(\omega)=\frac{1}{\pi}\int_{-\infty}^{\infty}f(\xi)\sin \omega \xi d\xi
$$

**傅里叶积分定理** 若函数$$f(x)$$在$$(-\infty,\infty)$$区间上满足：(1) $$f(x)$$在任一有限区间上满足Dirichlet conditions；(2) $$\int_{-\infty}^\infty|f(x)|dx$$收敛（绝对可积），则$$f(x)$$可以表示为傅里叶积分，且傅里叶积分值$$=(f(x+0)+f(x-0))/2$$

傅里叶积分又可改写为
$$
f(x)=\int_0^\infty C(\omega)\cos[\omega x-\varphi(\omega)]d\omega\\
振幅谱C(\omega)=\sqrt{A^2(\omega)+B^2(\omega)}\\
相位谱\varphi(\omega)=\arctan(B(\omega)/A(\omega))
$$
<img src="C:\Users\Xiao Yuxuan\Documents\pic\ealgn5b3jkhbt3kg.PNG" alt="ealgn5b3jkhbt3kg" style="zoom: 67%;" />

奇函数$$f(x)$$的傅里叶积分是**傅里叶正弦积分**，$$B()$$是其**傅里叶正弦变换**
$$
f(x)=\int_0^\infty B(\omega)\sin \omega xd\omega\\
B(\omega)=\frac{2}{\pi}\int_{0}^{\infty}f(\xi)\sin \omega \xi d\xi
$$
偶函数$$f(x)$$的傅里叶积分是**傅里叶余弦积分**，$$A()$$是其**傅里叶余弦变换**
$$
f(x)=\int_0^\infty A(\omega)\cos \omega xd\omega\\
A(\omega)=\frac{2}{\pi}\int_{0}^{\infty}f(\xi)\cos \omega \xi d\xi
$$


### 复数形式的傅里叶变换

傅里叶积分和傅里叶变换
$$
f(x)=\int_{-\infty}^{\infty}F(\omega)e^{i\omega x}d\omega\\
F(\omega)=\frac{1}{2\pi}\int_{-\infty}^{\infty}f(x)e^{-i\omega x}dx
$$
记作$$F(\omega)=\mathscr{F}[f(x)],f(x)=\mathscr{F}^{-1}[F(\omega)]$$，分别称为傅里叶变换的**像函数**和**原函数**



**基本性质**

+ **导数定理**
  $$
  \mathscr{F}[f'(x)]=i\omega F(\omega)
  $$

+ 

$$
\mathscr{F}[\int f(x)dx]=\frac{1}{i\omega}F(\omega)
$$

+ **相似性定理**

$$
\mathscr{F}[f(ax)]=\frac{1}{a}F(\frac{\omega}{a})
$$

+ **延迟定理**
  $$
  \mathscr{F}[f(x-x_0)]=e^{-i\omega x_0}F(\omega)
  $$

+ 

$$
\mathscr{F}[e^{i\omega_0x}f(x)]=f(\omega-\omega_0)
$$

+ **卷积定理**
  $$
  F_1(\omega)=\mathscr{F}[f_1(x)],F_2(\omega)=\mathscr{F}[f_2(x)]\\
  则\mathscr{F}[f_1(x)*f_2(x)]=2\pi F_1(\omega)F_2(\omega)\\
  其中f_1(x)*f_2(x)=\int_{-\infty}^{\infty}f_1(u)f_2(x-u)du
  $$

## 单位阶跃函数和$$\delta$$函数

### 单位阶跃函数u(t)

定义 
$$
u=
\begin{cases}
1,x>0\\任意值,x=0\\0,x<0
\end{cases}
$$
卷积性质











# 拉普拉斯变换

> 拉普拉斯变换主要用于求解线性微分方程/积分方程

## 拉普拉斯变换

常用于初始值问题——已知某个物理量在初始时刻$$t=0$$的值$$f(0)$$，置$$f(t)=0,t<0$$, 求解$$f(t), t>0$$

构造函数$$g(t)=e^{-\sigma t}f(t)$$以保证$$g(t)$$在区间$$(-\infty,\infty)$$上绝对可积，对其进行傅里叶变换
$$
G(\omega)=\frac{1}{2\pi} \int_{-\infty}^\infty g(t)e^{-\omega t}dt=\frac{1}{2\pi}\int_0^\infty f(t)e^{-(\sigma+i\omega)t}dt\\
记s=\sigma+i\omega,G(\omega)=F(s)/2\pi,则\\
F(s)=\mathscr{L}[f(t)]=\int_{-\infty}^\infty f(t)e^{-st}dt\\
f(t)=\mathscr{L^{-1}}[F(s)]=\frac{1}{2\pi i}\int_{\sigma-i\infty}^{\sigma+i\infty}F(s)e^{is}ds\\
$$
即为拉氏变换，$$F(s)$$和$$f(t)$$分别为**像函数和原函数**



**基本性质**

+ $$F(s)$$是在$$\sigma>\sigma_0$$的半平面上的解析函数

+ $$|s|→\infty, |Arg~s|\le\frac{\pi}{2}-\varepsilon$$时，$$F(s)$$存在且满足
  $$
  \lim_{p→\infty}F(s)=0
  $$

+ **线性定理** 

$$
F_1(s)=\mathscr{L}[f_1(t)],F_2(s)=\mathscr{L}[f_2(t)]\\
则c_1F_1(s)+c_2F_2(s)=\mathscr{L}[c_1f_1(t)+c_2f_2(t)]
$$

+ **导数定理**

$$
\mathscr{L}[f'(t)]=[s\mathscr{L}[f(t)]-f(0)]
$$

+ **积分定理**
  $$
  \mathscr{L}[\int_0^t f(\tau)d\tau]=\frac{1}{s}\mathscr{L}[f(t)]
  $$

+ **相似性定理**
  $$
  \mathscr{L}[f(at)]=\frac{1}{a}F(\frac{s}{a})
  $$
  **位移定理**
  $$
  \mathscr{L}[e^{-\lambda t}f(t)]=F(s+\lambda)
  $$

+ **延迟定理**
  $$
  \mathscr{L}[f(t-t_0)]=e^{-st_0}F(s)
  $$

+ **卷积定理**
  $$
  F_1(s)=\mathscr{L}[f_1(t)],F_2(s)=\mathscr{L}[f_2(t)]\\
  则F_1(s)F_2(s)=\mathscr{L}[f_1(t)*f_2(t)]\\
  其中f_1(t)*f_2(t)=\int_0^t f_1(\tau)f_2(t-\tau)d\tau即卷积
  $$





## 反演





## 应用

步骤：

1. 对方程实施拉普拉斯变换
2. 根据变换后的方程求解像函数
3. 对解出的像函数进行反演





# 推导

$$
e^{i\theta}=\cos \theta+i\sin\theta\\
\cos \theta=(e^{i\theta}+e^{-i\theta})/2,\sin\theta=(e^{i\theta}-e^{-i\theta})/2i
$$


$$
\mathscr{L}[1]=\int_0^\infty e^{-st}dt=(\frac{-1}{s}e^{-st})|_0^\infty=\frac{1}{s}\\
\mathscr{L}[e^{-at}]=\int_0^\infty e^{-(s+a)t}dt=\frac{1}{s+a}\\
\mathscr{L}[\sin(\omega t)]=\frac{1}{2i}\mathscr{L}[e^{i\omega t}-e^{-i\omega t}]=\frac{1}{2i}(\frac{1}{s-i\omega}-\frac{1}{s+i\omega})=\frac{\omega}{s^2+\omega^2}\\
\mathscr{L}[\cos(\omega t)]=\frac{1}{2}\mathscr{L}[e^{i\omega t}+e^{-i\omega t}]=\frac{1}{2}(\frac{1}{s+i\omega}+\frac{1}{s-i\omega})=\frac{s}{s^2+\omega^2}\\
\mathscr{L}[t^n]=\frac{n}{s}\mathscr{L}[t^{n-1}]=\cdots=\frac{n!}{s^n}\mathscr{L}[1]=\frac{n!}{s^{n+1}}
$$

