# 凸函数

## 基本性质

### 定义

函数 $f:\mathbb{R}^n\to \mathbb{R}$ 是**凸**的，如果 $\textbf{dom}\ f$ 是凸集，且 $\forall\pmb x,\pmb y\in \textbf{dom}\ f,\forall \theta\in[0,1]$，有

$$
\begin{equation}
\tag{3.1}
f(\theta\pmb x+(1-\theta)\pmb y)\le \theta f(\pmb x)+(1-\theta)f(\pmb y)
\end{equation}
$$

从几何意义上看，上述不等式意味着点 $(\pmb x,f(\pmb x))$ 和 $(\pmb y,f(\pmb y))$ 之间的线段，即从 $\pmb x$ 到 $\pmb y$ 的**弦**，在函数 $f$ 的图像上方（如下图所示）。称函数 $f$ 是**严格凸**的，如果上述不等式当 $\pmb x\neq \pmb y$ 以及 $0<\theta<1$ 时严格成立。称函数 $f$ 是凹的，如果函数 $-f$ 是凸的；称函数 $f$ 是**严格凸**的，如果 $-f$ 严格凸。

对于仿射函数，不等式 (3.1) 的等号总是成立，因此所有的仿射函数（包括线性函数）是既凸且凹的。反过来，若某个函数是既凸且凹的，则其是仿射函数。

函数是**凸**的，当且仅当其在与其定义域相交的任何直线上都是凸的。换言之，函数 $f$ 是凸的，当且仅当 $\forall \pmb x\in\textbf{dom}\ f,\forall \pmb v$，函数 $g(t)=f(\pmb x+t\pmb v)$ 是凸的（其定义域为 $\{t|\pmb x+t\pmb v\in\textbf{dom}\ f\}$）。这个性质非常有用，因为它容许我们通过将函数限制在直线上来判断其是否为凸函数。

对凸函数的**分析**已经相当地透彻，这里不再继续深入。例如有这样一个简单的结论：凸函数在其定义域相对内部是连续的；它只可能在相对边界上不连续。

### 扩展值延伸

通常可以定义凸函数在定义域外的值为 $\infty$，从而将这个凸函数延伸至全空间 $\mathbb{R}^n$。如果 $f$ 是凸函数，我们定义它的**扩展值延伸** $\tilde{f}:\mathbb{R}^n\to \mathbb{R}\cup\{\infty\}$ 如下

$$
\tilde{f}=\begin{cases}
f(\pmb x)& \pmb x\in\textbf{dom}\ f\\
\infty& \pmb x\notin\textbf{dom}\ f
\end{cases}
$$

延伸函数 $\tilde{f}$ 定义在全空间 $\mathbb{R}^n$ 上，取值集合为 $\mathbb{R}\cup\{\infty\}$。我们也可以从延伸函数 $\tilde{f}$ 的定义中确定原函数 $f$ 的定义域，即 $\textbf{dom}\ f=\{\pmb x|\tilde{f}(\pmb x)<\infty \}$。

这种延伸可以简化符号描述，这样我们就不需要明确描述定义域或者每次提到 $f(\pmb x)$ 时都限定“$\forall \pmb x\in\textbf{dom}\ f$”。以基本不等式 (3.1) 为例，对于延伸函数 $\tilde{f}$，可以描述为：$\forall \pmb x,\pmb y,\forall \theta\in(0,1)$，有

$$
\tilde{f}(\theta\pmb x+(1-\theta)\pmb y)\le \theta\tilde{f}(\pmb x)+(1-\theta)\tilde{f}(\pmb y)
$$

（当 $\theta=0$ 或 $\theta=1$ 时不等式总成立。）当然此时我们应当利用扩展运算和序来理解这个不等式。若 $\pmb x$ 和 $\pmb y$ 都在 $\textbf{dom}\ f$ 内，上述不等式即为不等式 (3.1)；如果有任何一个在 $\textbf{dom}\ f$ 外，上述不等式的右端为 $\infty$，不等式仍然成立。

在不会造成歧义的情况下，这里将用同样的符号来表示一个凸函数及其延伸函数。即假设所有的凸函数都隐含地被延伸了，也就是在定义域外都被定义为 $\infty$。

类似地，可以通过定义凹函数在定义域外都为 $-\infty$ 对其进行延伸。

### 一阶条件

假设 $f$ 可微（即其梯度 $\nabla f$ 在开集 $\textbf{dom}\ f$ 内处处存在），则函数 $f$ 是凸函数的充要条件是 $\textbf{dom}\ f$ 是凸集且 $\forall\pmb x,\pmb y\in \textbf{dom}\ f$，有

$$
\begin{equation}
\tag{3.2}
f(\pmb y)\ge f(\pmb x)+\nabla f(\pmb x)^{\rm T}(\pmb y-\pmb x)
\end{equation}
$$

下图描述了上述不等式。

![](https://s2.loli.net/2023/01/03/mUD6uB84VAFzopf.png)

由 $f(\pmb x)+\nabla f(\pmb x)^{\rm T}(\pmb y-\pmb x)$ 得出的仿射函数 $\pmb y$ 即为函数 $f$ 在点 $\pmb x$ 附近的泰勒近似。不等式 (3.2) 表明，对于一个凸函数，其一阶泰勒近似实质上是原函数的一个全局下估计。反之，如果某个函数的一阶泰勒近似总是其全局下估计，则这个函数是凸的。

不等式 (3.2) 说明从一个凸函数的**局部信息**（即它在某点的函数值和导数），我们可以得到一些**全局信息**（如它的全局下估计）。这也许是凸函数的最重要的信息，由此可以解释凸函数以及凸优化问题的一些非常重要的性质。下面是一个简单的例子：由不等式 (3.2) 可以知道，如果 $\nabla f(\pmb x)=\pmb 0$，那么 $\forall\pmb y\in\textbf{dom}\ f$，$f(\pmb y)\ge f(\pmb x)$，即 $\pmb x$ 是函数 $f$ 的全局极小点。

严格凸性同样可以由一阶条件刻画：函数 $f$ 严格凸的充要条件是 $\textbf{dom}\ f$ 是凸集且 $\forall\pmb x,\pmb y\in\textbf{dom}\ f,\pmb x\neq \pmb y$，有

$$
\begin{equation}
\tag{3.3}
f(\pmb y)>f(\pmb x)+\nabla f(\pmb x)^{\rm T}(\pmb y-\pmb x)
\end{equation}
$$

对于凹函数，亦存在与凸函数相反的一阶条件。

#### 一阶凸性条件的证明

为了证明式 (3.2)，先考虑 $n=1$ 的情况：我们证明可微函数 $f:\mathbb{R}\to \mathbb{R}$ 是凸函数的充要条件是 $\forall x,y\in\textbf{dom}\ f$，有

$$
\begin{equation}
\tag{3.4}
f(y)\ge f(x)+f'(x)(y-x)
\end{equation}
$$

首先假设 $f$ 是凸函数，且 $x,y\in\textbf{dom}\ f$。因为 $\textbf{dom}\ f$ 是凸集（某个区间），对于任意的 $0<t\le 1$，我们有 $x+t(y-x)\in\textbf{dom}\ f$，由函数 $f$ 的凸性可得

$$
f(x+t(y-x))\le (1-t)f(x)+tf(y)
$$

将上式两端同除 $t$ 可得

$$
f(y)\ge f(x)+\frac{f(x+t(y-x))-f(x)}{t}
$$

令 $t\to 0$，可以得到不等式 (3.4)。

为了证明充分性，假设 $\forall x,y\in\textbf{dom}\ f$（某个区间），函数满足不等式 (3.4)。选择任意 $x\neq y,0\le\theta\le 1$，令 $z=\theta x+(1-\theta)y$，两次应用不等式 (3.4) 可得

$$
f(x)\ge f(z)+f'(z)(x-z),\quad\quad f(y)\ge f(z)+f'(z)(y-z)
$$

将第一个不等式乘 $\theta$，第二个不等式乘 $1-\theta$，并将两者相加可得

$$
\theta f(x)+(1-\theta)f(y)\ge f(z)
$$

从而说明函数 $f$ 是凸的。

现在来证明一般情况，即 $f:\mathbb{R}^n\to \mathbb{R}$。设 $\pmb x,\pmb y\in\mathbb{R}^n$，考虑过这两点的直线上的函数 $f$，即函数 $g(t)=f(t\pmb y+(1-t)\pmb x)$，此函数对 $t$ 求导可得 $g'(t)=\nabla f(t\pmb y+(1-t)\pmb x)^{\rm T}(\pmb y-\pmb x)$。

首先假设函数 $f$ 是凸的，则函数 $g$ 是凸的，由前面的讨论可得 $g(1)\ge g(0)+g'(0)$，即

$$
f(\pmb y)\ge f(\pmb x)+\nabla f(\pmb x)^{\rm T}(\pmb y-\pmb x)
$$

再假设此不等式对于任意 $\pmb x$ 和 $\pmb y$ 均成立，因此若 $t\pmb y+(1-t)\pmb x\in\textbf{dom}\ f$ 以及 $u\pmb y+(1-u)\pmb x\in\textbf{dom}\ f$，有

$$
f(t\pmb y+(1-t)\pmb x)\ge f(u\pmb y+(1-u)\pmb x)+\nabla f(u\pmb y+(1-u)\pmb x)^{\rm T}(\pmb y-\pmb x)(t-u)
$$

即 $g(t)\ge g(u)+g'(u)(t-u)$，说明函数 $g$ 是凸的，从而 $\forall \theta\in[0,1]$，有 $g(0\theta+1(1-\theta))\le \theta g(0)+(1-\theta)g(1)$，即

$$
f(\theta\pmb x+(1-\theta)\pmb y)\le \theta f(\pmb x)+(1-\theta)f(\pmb y)
$$

### 二阶条件

现在假设函数 $f$ 二阶可微，即对于开集 $\textbf{dom}\ f$ 内的任意一点，它的 Hessian 矩阵活着二阶导数 $\nabla^2f$ 存在，则函数 $f$ 是凸函数的充要条件是，其 Hessian 矩阵是半正定阵：即 $\forall \pmb x\in\textbf{dom}\ f$，有

$$
\nabla^2f(\pmb x)⪰0
$$

对于 $\mathbb{R}$ 上的函数，上式可以简化为一个简单的条件 $f''(x)\ge 0$（$\textbf{dom}\ f$ 是凸的，即一个区间），此条件说明函数 $f$ 的导数是非减的。条件 $\nabla^2f(\pmb x)⪰0$ 从集合上可以理解为函数图像在点 $\pmb x$ 处具有正（向上）的曲率。二阶条件的证明略。

严格凸的条件可以部分由二阶条件刻画。如果 $\forall \pmb x\in\textbf{dom}\ f$，有 $\nabla^2f(\pmb x)≻0$，则函数 $f$ 严格凸。反过来则不一定成立：例如，函数 $f:\mathbb{R}\to\mathbb{R}$，其表达式为 $f(x)=x^4$，它是严格凸的，但是在 $x=0$ 处，其二阶导数为零。

类似地，函数 $f$ 是凹函数的充要条件是，$\textbf{dom}\ f$ 是凸集且 $\forall\pmb x\in\textbf{dom}\ f$，$\nabla^2f(\pmb x)⪯0$。

@考虑二次函数 $f:\mathbb{R}^n\to\mathbb{R}$，其定义域为 $\textbf{dom}\ f=\mathbb{R}^n$，其表达式为

$$
f(\pmb x)=\frac{1}{2}\pmb x^{\rm T}P\pmb x+\pmb q^{\rm T}\pmb x+r
$$

其中 $P\in\mathbb{S}^n,\pmb q\in\mathbb{R}^n,r\in\mathbb{R}$。因为 $\forall\pmb x$，$\nabla^2f(\pmb x)=P$，所以函数 $f$ 是凸的当且仅当 $P⪰0$。

对于二次函数，严格凸比较容易表达：函数 $f$ 是严格凸的当且仅当 $P≻0$。

!!! note "说明"
    在判断函数的凸性和凹性时，不管是一阶条件还是二阶条件，$\textbf{dom}\ f$ 必须是凸集这个前提条件必须满足。例如，考虑函数 $f(x)=1/x^2$，其定义域为 $\textbf{dom}\ f=\{x\in\mathbb{R}|x\neq 0\}$，$\forall x\in \textbf{dom}\ f$ 均满足 $f''(x)>0$，但是函数 $f(x)$ 并不是凸函数。

### 例子

前文已经提到所有的线性函数和仿射函数均为凸函数（同时也是凹函数），并描述了凸和凹的二次函数。本节给出更多的凸函数和凹函数的例子。首先考虑 $\mathbb{R}$ 上的一些函数，其自变量为 $x$。

* **指数函数**：$\forall a\in\mathbb{R}$，函数 $e^{ax}$ 在 $\mathbb{R}$ 上是凸的。
* **幂函数**：当 $a\ge 1$ 或 $a\le 0$ 时，$x^a$ 在 $\mathbb{R}_+$ 上是凸函数；当 $0\le a\ge 1$ 时，$x^a$ 在 $\mathbb{R}_{+}$ 上是凹函数。
* **绝对值幂函数**：当 $p\ge 1$ 时，函数 $|x|^p$ 在 $\mathbb{R}$ 上是凸函数。
* **对数函数**：函数 $\log x$ 在 $\mathbb{R}_+$ 上是凹函数。
* **负熵**：函数 $x\log x$ 在 $\mathbb{R}_*$ 上是凸函数（定义 $0\log 0=0$）。

我们可以通过基本不等式 (3.1) 或者二阶导数半正定或半负定来判断上述函数是凸的或凹的。以函数 $f(x)=x\log x$ 为例，其导数和二阶导数为

$$
f'(x)=\log x+1,\quad f''(x)=1/x
$$

即 $\forall x>0$，有 $f''(x)>0$，所以负熵是严格凸的。

下面给出 $\mathbb{R}^n$ 上的一些例子。

* **范数**：$\mathbb{R}^n$ 上的任意范数均为凸函数。
* **二次-线性分式函数**：函数 $f(x,y)=x^2/y$，其定义域为

    $$
    \textbf{dom}\ f=\mathbb{R}×\mathbb{R}_+=\{(x,y)\in\mathbb{R}^2|y>0\}
    $$

    是凸函数（如下图所示）。

![](https://s2.loli.net/2023/01/03/EVKTqU2vcsWN5Qu.png)

* **指数和的对数**：函数 $f(\pmb x)=\log (e^{x_1}+\cdots+e^{x_n})$ 在 $\mathbb{R}^n$ 上是凸函数。这个函数可以看成最大值函数的可微（实际上是解析）近似，因为对任意 $\pmb x$，下面的不等式成立

    $$
    \max{x_1,\cdots,x_n}\le f(\pmb x)\le \max{x_1,\cdots,x_n}+\log n
    $$

* **几何平均**：几何平均函数 $f(\pmb x)=(\prod_{i=1}^nx_i)^{1/n}$ 在定义域 $\mathbb{R}_+^n$ 上是凹函数。
* **对数-行列式**：函数 $f(X)=\log\det X$ 在定义域 $\mathbb{S}_+^n$ 上是凹函数。

判断上述函数的凸性（或者凹性）可以有多种途径，可以直接验证不等式 (3.1) 是否成立，亦可以验证其 Hessian 矩阵是否半正定，或者可以将函数转换到与其定义域相交的任意直线上，通过得到的单变量函数判断原函数的凸性。

@**范数**：如果函数 $f:\mathbb{R}^n\to\mathbb{R}$ 是范数，对任意的 $0\le\theta\le1$，有

$$
f(\theta\pmb x+(1-\theta)\pmb y)\le f(\theta\pmb x)+f((1-\theta)\pmb y)=\theta f(\pmb x)+(1-\theta)f(\pmb y)
$$

上述不等式可以由三角不等式得到。

@**二次-线性分式函数**：为了说明二次-线性分式函数 $f(x,y)=x^2/y$ 是凸的，我们注意到，对于 $y>0$，有

$$
\nabla^2f(\pmb x,\pmb y)=\frac{2}{y^3}\begin{bmatrix}
y^2 & -xy\\
-xy & x^2
\end{bmatrix}=\frac{2}{y^3}\begin{bmatrix}y\\-x\end{bmatrix}\begin{bmatrix}y\\-x\end{bmatrix}^{\rm T}⪰0
$$

@**指数和的对数**

### Jensen 不等式及其扩展

## 保凸运算
