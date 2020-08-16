# 基本概念

表示未知函数、其导数与自变量之间的关系的方程称为**微分方程**，未知函数是一元函数的称为**常微分方程**，未知函数是多元函数的称为**偏微分方程**，未知函数的最高阶导数的阶称为微分方程的**阶**，代入微分方程使其成为恒等式的的函数叫做微分方程的**解**，如果微分方程中的解含有任意常数，且任意常数的个数与微分方程的阶数相同，这样的解叫做微分方程的**通解**。为确定通解的任意常数，需要给定**初始条件**即$$x_0$$处$$y,y',\cdots,y^{n}$$的值，最后得到微分方程的**特解**。





# 常微分方程

一般地，n阶常微分方程的形式是
$$
F(x,y,y',\cdots,y^{n})=0
$$
其中最高阶项必须出现。这里讨论最高阶项可分离的情形。



## 一阶常微分方程

一阶常微分方程一般写作
$$
y'=f(x,y) \quad {\rm or} \quad P(x,y)dx+Q(x,y)dy=0 \\
{\rm or} \quad \frac{dy}{dx}=-\frac{P(x,y)}{Q(x,y)} \quad {\rm or} \quad \frac{dx}{dy}=-\frac{Q(x,y)}{P(x,y)}
$$



### 可分离变量的微分方程

一般地，如果一阶常微分方程能写成
$$
f(x)dx=g(y)dy
$$
的形式，原方程就称为可分离变量的微分方程。对两端积分得到
$$
G(y)=F(x)+C
$$
即为该微分方程的**隐式通解**。



### 齐次方程

如果一阶常微分方程能写成
$$
\frac{dy}{dx}=\varphi(\frac{y}{x})
$$
的形式，原方程就称为**齐次方程**。引入新的参数$$u=\frac{y}{x}$$即化为可分离变量的方程。

**可化为齐次的方程**

对于方程
$$
\frac{dy}{dx}=\frac{ax+by+c}{a_1x+b_1y+c_1}
$$
如果系数行列式$$\begin{vmatrix} a&b \\a_1&b_1  \end{vmatrix}\neq0$$，则令$$x=X+h,y=Y+k$$消去常数项再令$$u=\frac{y}{x}$$；否则令
$$
\frac{a_1}{a}=\frac{b_1}{b}=\lambda \quad and \quad v=ax+by
$$



### 一阶线性微分方程

如果一阶常微分方程能写成
$$
\frac{dy}{dx}+P(x)y=Q(x)
$$
的形式，原方程就称为**一阶线性微分方程**。如果$$Q(x)\equiv 0$$，则称为**齐次**方程，其通解为
$$
\begin{align}
&\frac{dy}{dx}+P(x)y=0  \\
\Rightarrow &\frac{dy}{y}=-P(x)dx \\
\Rightarrow &\ln \left | y \right |=-\int P(x)dx+C_1 \\
\Rightarrow &y=Ce^{-\int P(x)dx} 
\end{align}
$$

在此基础上，使用常数变异法来求**非齐次**方程的通解
$$
\begin{align}
&y=ue^{-\int P(x)dx}\\
\Rightarrow &\frac{dy}{dx}=u'e^{-\int P(x)dx}-uP(x)e^{-\int P(x)dx} \\
\Rightarrow &u'=Q(x)e^{\int P(x)dx} \\
\Rightarrow &u=\int Q(x)e^{\int P(x)dx}dx+C \\
\Rightarrow &y=e^{-\int P(x)dx}(\int Q(x)e^{\int P(x)dx}dx+C) \\
\end{align}
$$

> 也可用积分因子$$\mu (x)=e^{\int P(x)dx}$$乘方程构建全微分

**伯努利方程** 
$$
\begin{align}
&\frac{dy}{dx}+P(x)y=Q(x)y^n\\
\Rightarrow & y^{-n}\frac{dy}{dx}+P(x)y^{1-n}=Q(x)\\
\Rightarrow & \frac{dz}{dx}+(1-n)P(x)z=(1-n)Q(x) \quad (z=y^{1-n})\\
\end{align}
$$



### 全微分方程

一阶微分方程写成
$$
\begin{align}
&du(x,y)=P(x,y)dx+Q(x,y)dy=0\\
\Rightarrow & u(x,y)\equiv C\\
\end{align}
$$

> 全微分的充要条件是$$\frac{\partial P}{\partial y}=\frac{\partial Q}{\partial x}$$



## 高阶常微分方程

### 降阶

$$
\begin{align}
&y^{(n)}=f(x)\\
\Rightarrow & y^{(n-1)}=\int f(x)dx+C_1\\
\Rightarrow & \cdots
\end{align}
$$

$$
\begin{align}
&y''=f(x,y')\\
\Rightarrow & p'=f(x,p) \quad (p=y')\\
\Rightarrow & \cdots
\end{align}
$$

$$
\begin{align}
&y''=f(y,y')\\
\Rightarrow & p\frac{dp}{dy}=f(y,p) \quad (p=y')\\
\Rightarrow & \cdots
\end{align}
$$



### 二阶线性常微分方程

$$
\frac{d^2y}{dx^2}+P(x)\frac{dy}{dx}+Q(x)y=f(x)
$$

叫做**二阶线性常微分方程**，当$$f(x)\equiv0$$时方程是**齐次**的，否则是**非齐次**的。

**定理 如果函数$$y_1(x)$$和$$y_2(x)$$是方程$$\frac{d^2y}{dx^2}+P(x)\frac{dy}{dx}+Q(x)y=0$$的两个解，那么$$y=C_1y_1(x)+C_2y_2(x)$$也是该方程的解**

**定理 如果函数$$y_1(x)$$和$$y_2(x)$$是方程$$\frac{d^2y}{dx^2}+P(x)\frac{dy}{dx}+Q(x)y=0$$的两个线性无关的特解，那么$$y=C_1y_1(x)+C_2y_2(x)$$就是该方程的通解。**

**定理 如果$$y^*(x)$$是方程$$\frac{d^2y}{dx^2}+P(x)\frac{dy}{dx}+Q(x)y=f(x)$$的特解，$$Y(x)$$是对应的齐次方程的通解，那么$$y=y^*(x)+Y(x)$$就是该方程的通解。**

> 上述定理可推广至n阶线性常微分方程

**常数变易法**

如果齐次方程的通解是$$Y=C_1y_1+C_2y_2$$，那么令
$$
\begin{align}
&y=v_1y_1+v_2y_2\\
\Rightarrow &y'=v_1'y_1+v_1y_1'+v_2'y_2+v_2y_2'\\
再设& v_1'y_1+v_2'y_2=0\\
\Rightarrow &y'=v_1y_1'+v_2y_2'\\
\Rightarrow &y''=v_1y_1''+v_1'y_1'+v_2y_2''+v_2'y_2'\\
代入&二阶线性常微分方程\\
\Rightarrow &v_1'y_1'+v_2'y_2'=f\\
如果&W=
\begin{vmatrix}
y_1 & y_2\\
y_1' & y_2'
\end{vmatrix} \neq0 \\
则&v_1'=-\frac{y_2 f}{W},v_2'=\frac{y_1 f}{W}\\
积分得&v_1=C_1+\int(-\frac{y_2f}{W})dx,v_2=C_2+\int(\frac{y_1f}{W})dx\\
代入原式即得
\end{align}
$$



### **二阶常系数齐次线性微分方程**

求方程$$y''+py'+qy=0$$的通解的步骤如下

1. 求特征方程$$r^2+pr+q=0$$的根$$r_1,r_2$$；

2. | 实根$$r_1=r_2$$                       | $$y=C_1e^{r_1x}+C_2e^{r_2x}$$                         |
   | ------------------------------------- | ----------------------------------------------------- |
   | 实根$$r_1\neq r_2$$                   | $$y=(C_1+C_2x)e^{r_1x}$$                              |
   | 共轭复根$$r_{1,2}=\alpha \pm i\beta$$ | $$y=e^{\alpha x}(C_1\cos(\beta x)+C_2\sin(\beta x))$$ |



### **高阶常系数齐次线性微分方程**

与二阶类似，求方程$$(D^n+p_1D^{n-1}+\cdots+p_{n-1}D+p_n)y=0$$的通解的步骤如下

1. 求特征方程$$r^n+p_1r^{n-1}+\cdots+p_{n-1}r+p_n=0$$的根$$r$$；

2. | 单实根$$r$$                              | 给出1项$$Ce^{rx}$$                                           |
   | ---------------------------------------- | ------------------------------------------------------------ |
   | k重实根$$r$$                             | 给出k项$$e^{rx}(C_1+C_2x+\cdots+C_kx^{k-1})$$                |
   | 一对复根$$r_{1,2}=\alpha \pm i\beta$$    | 给出2项$$e^{\alpha x}(C_1\cos(\beta x)+C_2\sin(\beta x))$$   |
   | 一对k重复根$$r_{1,2}=\alpha \pm i\beta$$ | 给出2k项$$e^{\alpha x}((C_1+C_2x+\cdots+C_kx^{k-1})\cos(\beta x)+(D_1+D_2x+\cdots+D_kx^{k-1})\sin(\beta x))$$ |



### **二阶常系数非齐次线性微分方程**

求方程$$y''+py'+qy=f(x)$$的特解，仅给出2种常见f(x)形式的情形

$$f(x)=e^{\lambda x}P_m(x)$$**型**

特解形式为$$y^*=x^kQ_m(x)e^{\lambda x}$$，其中$$Q_m(x)$$表示x的m次多项式，k根据$$\lambda$$是特征方程的非根/单根/重根取0/1/2。

$$f(x)=e^{\lambda x}(P_n(x)\cos \omega x+P_l(x)\sin \omega x)$$**型**

特解形式为$$y^*=x^k(Q_{m1}(x)\cos \omega x+Q_{m2}(x)\sin \omega x)e^{\lambda x}$$，其中$$Q_m(x)$$表示x的m次多项式，$$m=\max{\{n,l\}}$$，k根据$$\lambda+i\omega$$是特征方程的非根/单根取0/1。



### 欧拉方程

形如
$$
x^ny^{(n)}+p_1x^{n-1}y^{(n-1)}+\cdots+p_{n-1}xy'+p_ny=f(x)
$$
的方程称为**欧拉方程**，作变换$$x=e^t$$后，$$x^ky^{(k)}=D(D-1)\cdots(D-k+1)y$$。





# 偏微分方程





# 汇总与例题

$$
\begin{align}
FIRST\; ORDER\\
f(x)dx=g(y)dy \quad& 分离变量\\

\\\frac{dy}{dx}=\frac{ax+by+c}{a_1x+b_1y+c_1}  \quad& 化为齐次\\

\\\frac{dy}{dx}+P(x)y=0 \quad& 一阶线性齐次\\
&通解 y=Ce^{-\int P(x)dx} \\

\\\frac{dy}{dx}+P(x)y=Q(x) \quad& 一阶线性非齐次\\
&通解 y=e^{-\int P(x)dx}(\int Q(x)e^{\int P(x)dx}dx+C)

\\\frac{dy}{dx}+P(x)y=Q(x)y^n \quad& 伯努利方程\\
& /y^n再令z=y^{1-n}

\\du(x,y)=P(x,y)dx+Q(x,y)dy=0 \quad& 全微分方程\\
& u(x,y)\equiv C\\

SECOND/HIGHER\; ORDER
\\y^{(n)}=f(x) \quad &降阶(直接积分)
\\y''=f(x,y') \quad &降阶(p=y')
\\y''=f(y,y')\quad &降阶(p=y')\\

\\y''+py'+qy=0 \quad& 常系数线性齐次 \\
&C_1e^{r_1x}+C_2e^{r_2x} / (C_1+C_2x)e^{r_1x} / e^{\alpha x}(C_1\cos(\beta x)+C_2\sin(\beta x))\\
(D^n+p_1D^{n-1}+\cdots+p_{n-1}D+p_n)y=0 \quad& 高阶常系数线性齐次\\
&Ce^{rx} / e^{rx}(C_1+C_2x+\cdots+C_kx^{k-1}) \\
&/e^{\alpha x}(C_1\cos(\beta x)+C_2\sin(\beta x)) /e^{\alpha x}((C_1+C_2x+\cdots+C_kx^{k-1})\cos(\beta x)+(D_1+D_2x+\cdots+D_kx^{k-1})\sin(\beta x))


\\y''+py'+qy=f(x) \quad& 常系数线性非齐次\\
f(x)=e^{\lambda x}P_m(x) \quad&y^*=x^kQ_m(x)e^{\lambda x}\\
f(x)=e^{\lambda x}(P_n(x)\cos \omega x+P_l(x)\sin \omega x) \quad&y^*=x^k(Q_{m1}(x)\cos \omega x+Q_{m2}(x)\sin \omega x)e^{\lambda x}\\

\\x^ny^{(n)}+p_1x^{n-1}y^{(n-1)}+\cdots+p_{n-1}xy'+p_ny=f(x) \quad& 欧拉方程\\
&x=e^t,x^ky^{(k)}=D(D-1)\cdots(D-k+1)y
\\微分方程组
\\\frac{d \pmb x}{dt}=\pmb A \pmb x +\pmb b \quad& e^{-t \pmb A}\pmb x求导
\end{align}
$$

$$
\begin{align}
\\e.g.\quad y''-2y'+5y=0\\




\end{align}
$$

$$
一阶\\
@完全非线性(p=y',dy=pdx解得p，回代即得)\\
1.\;y=xy'+y'+y'^2\\
A:y=Cx+C+C^2 \; or \; y=-\frac{(x+1)^2}{4}\\
2.\;y'^2+2xy'-2y=0\\
A:y=Cx+\frac{C^2}{2} \; or \; y=-\frac{x^2}{2}\\
@Riccati方程（y=y_1+\frac{1}{u}）\\
1.\;x^2y'-x^2y^2+xy+1=0(y=\frac{1}{x} \text{ is a particular solution)}\\
[A:y=\frac{1+Cx^2}{x(1-Cx^2)}]\\
2.\;y'+(2x^2+1)y+y^2+(x^4+x^2+2x)=0 (y=-x^2 \text{ is a particular solution)}\\
[A:y=-x^2+\frac{1}{Ce^x-1}]\\

二阶\\
@欧拉方程(x=e^t,x^ky^{(k)}=D(D-1)\cdots(D-k+1)y)\\
1.\;x^2y''-xy'+y=x^3\\
[A:y=(C_1+C_2\ln x)x+\frac{1}{4}x^3]\\
2.\;y''+\frac{1}{x}y'+\frac{4}{x^2}y=(\frac{2\log x}{x})^2\\
[A:y=C_1\cos (2\ln x)+C_2 \sin (2\ln x)+(\ln x)^2-\frac{1}{2}]\\
3.\;x^2y''-xy'-8y=x^2\\
[A:y=C_1x^4+C_2x^{-2}-\frac{1}{8}x^2]\\
4.\;x^3y'''-3x^2y''+6xy'-6y=2x^4e^x\\
[A:y=C_1x+C_2x^2+C_3x^3+2xe^x]\\
\\@常系数线性非齐次\\(通解：Ce^{rx},e^{\alpha x}(C\cos \beta x +C\sin \beta x),重根替换C;特解:照搬，替换同次多项式，\lambda/\lambda+iw \rightarrow补x^k)\\
1.\;y''+2y'-3y=e^x\cos x\\
[A:y=e^x(\frac{-1}{17}\cos x+\frac{4}{17}\sin x)+C_1e^x+C_2e^{-3x}]\\
2.\;y^{(4)}-2y^{(3)}+2y'-y=9e^{-2x}\\
[A:y=\frac{1}{3}e^{-2x}+(C_1+C_2x+C_3x^2)e^x+C_4e^{-x}]\\
3.\;y'''-3y''+2y'=3x^2\\
[A:y=\frac{1}{2}x^3+\frac{9}{4}x^2+\frac{21}{4}x+C_1e^x+C_2e^{2x}+C_3]\\
4.\;y''+4y'+4y=4e^{2x}\\
[A:y=\frac{1}{4}e^{2x}+(C_1+C_2x)e^{-2x}]\\
$$



