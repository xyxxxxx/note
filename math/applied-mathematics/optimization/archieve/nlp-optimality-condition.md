# 最优性条件

最优性条件即非线性规划的最优解满足的必要条件和充分条件，这些条件十分重要，为各种算法的推导和分析提供了理论基础。



## 无约束极值问题的极值条件

**无约束极值问题** 考虑非线性规划问题
$$
\min\quad f(\pmb x),\ \pmb x\in\mathbb{R}^n
$$
其中 $f(\pmb x)$ 是定义在 $\mathbb{R}^n$ 上的实函数。这个问题是求 $f(\pmb x)$ 在 $n$ 维欧氏空间中的极小点，称为**无约束极值问题**。

**定理** 设函数 $f(\pmb x)$ 在点 $\overline{\pmb x}$ 可微，如果存在方向 $\pmb d$，使得 $\pmb d^{\rm T}\nabla f(\overline{\pmb x})<0$，则 $\exist \delta >0, \forall \lambda\in (0,\delta),f(\overline{\pmb x}+\lambda \pmb d)<f(\overline{\pmb x})$。

**局部极小点的一阶必要条件** 设函数 $f(\pmb x)$ 在点 $\overline{\pmb x}$ 可微，若 $\overline{\pmb x}$ 是局部极小点，则梯度 $\nabla f(\overline{\pmb x})=\pmb 0$。

**局部极小点的二阶必要条件** 设函数 $f(\pmb x)$ 在点 $\overline{\pmb x}$ 处二阶可微，若 $\overline{\pmb x}$ 是局部极小点，则则梯度 $\nabla f(\overline{\pmb x})=\pmb 0$，且Hessian矩阵 $\nabla^2 f(\overline{\pmb x})$ 半正定。

**局部极小点的二阶充分条件** 设函数 $f(\pmb x)$ 在点 $\overline{\pmb x}$ 处二阶可微，若梯度 $\nabla f(\overline{\pmb x})=\pmb 0$，且Hessian矩阵 $\nabla^2 f(\overline{\pmb x})$ 正定，则 $\overline{\pmb x}$ 是局部极小点。

**凸函数的全局最小点的充要条件** 设 $f(\pmb x)$ 是定义在 $\mathbb{R}^n$ 上的可微凸函数， $\overline{\pmb x}\in \mathbb{R}^n$，则 $\overline{\pmb x}$ 是全局极小点的充要条件是梯度 $\nabla f(\overline{\pmb x})=\pmb 0$。



@利用极值条件解下列问题：
$$
\min\quad f(\pmb x)=\frac{1}{3}x_1^3+\frac{1}{3}x_2^3-x_2^2-x_1
$$
计算偏导数
$$
\frac{\partial f}{\partial x_1}=x_1^2-1,\quad \frac{\partial f}{\partial x_2}=x_2^2-2x_2
$$
令 $\nabla f(\overline{\pmb x})=\pmb 0$，解得驻点
$$
\pmb x^{(1)}=\begin{bmatrix}1\\0\end{bmatrix},\quad
\pmb x^{(2)}=\begin{bmatrix}1\\2\end{bmatrix},\quad
\pmb x^{(3)}=\begin{bmatrix}-1\\0\end{bmatrix},\quad
\pmb x^{(4)}=\begin{bmatrix}-1\\2\end{bmatrix}
$$
计算Hessian矩阵
$$
\nabla^2 f(\pmb x)=\begin{bmatrix}2x_1 & 0\\0 & 2x_2-2
\end{bmatrix}
$$
代入各驻点
$$
\nabla^2 f(\pmb x^{(1)} )=\begin{bmatrix}2 & 0\\0 & -2
\end{bmatrix},\quad \nabla^2 f(\pmb x^{(2)} )=\begin{bmatrix}2 & 0\\0 & 2
\end{bmatrix},\\
\nabla^2 f(\pmb x^{(3)} )=\begin{bmatrix}-2 & 0\\0 & -2
\end{bmatrix},\quad \nabla^2 f(\pmb x^{(4)} )=\begin{bmatrix}-2 & 0\\0 & 2
\end{bmatrix}
$$
根据二阶必要条件， $\pmb x^{(1)},\pmb x^{(3)},\pmb x^{(4)}$ 不是极小点；根据二阶充分条件， $\pmb x^{(2)}$ 是极小点。



## 约束极值问题的最优性条件

**约束极值问题** **约束极值问题**一般表示为
$$
\begin{align}
\min & \quad f(\pmb x),\ \pmb x\in\mathbb{R}^n\\
{\rm s.t.} & \quad g_i(\pmb x)\ge 0,\ i =1,\cdots,m,\\
&\quad h_j(\pmb x)=0,\ j=1,\cdots,l.
\end{align}
$$
其中 $g_i(x)\ge 0$ 称为**不等式约束**， $h_j(x)=0$ 称为**等式约束**。集合
$$
S=\{\pmb x|g_i(\pmb x)\ge 0,\ i=1,\cdots,m;h_j(\pmb x)=0,\ j=1,\cdots,l \}
$$
称为**可行域**。

由于在约束极值问题中，自变量的取值受到限制，目标函数在无约束情况下的极小点很可能不在可行域内，因此一般不能用无约束极值条件处理约束问题。



### 下降方向和可行方向

**下降方向** 设 $f(\pmb x)$ 是定义在 $\mathbb{R}^n$ 上的实函数， $\overline{\pmb x}\in \mathbb{R}^n$， $\pmb d$ 是非零向量。若 $\exist \delta >0,\forall \lambda \in (0,\delta)$，都有
$$
f(\overline{\pmb x}+\lambda \pmb d)<f(\overline{\pmb x})
$$
则称 $\pmb d$ 是函数 $f(\pmb x)$ 在 $\overline{\pmb x}$ 处的**下降方向**。

**可行方向** 设集合 $S\sub \mathbb{R}^n,\overline{\pmb x}\in {\rm cl}S$， $\pmb d$ 为非零向量，若存在数 $\delta >0$，使得 $\forall \lambda \in (0,\delta)$，都有
$$
\overline{\pmb x}+\lambda \pmb d\in S
$$
则称 $\pmb d$ 为集合 $S$ 在 $\overline{\pmb x}$ 处的**可行方向**。集合 $S$ 在 $\overline{\pmb x}$ 处的所有可行方向组成的集合称为在 $\overline{\pmb x}$ 处的**可行方向锥**。

**定理** 考虑非线性规划问题
$$
\begin{align}
\min & \quad f(\pmb x)\\
{\rm s.t.} &\quad \pmb x\in S
\end{align}
$$
设 $S$ 是 $\mathbb{R}^n$ 上的非空集合， $\overline{\pmb x}\in S$， $f(\pmb x)$ 在 $\overline{\pmb x}$ 处可微。如果 $\overline{\pmb x}$ 是局部最优解，那么 $F_0\cap D=\varnothing$。其中
$$
F_0=\{\pmb d|\pmb d^{\rm T} \nabla f(\overline{\pmb x})<0  \}\\
D=\{集合S在\overline{\pmb x}处的可行方向锥\}
$$


### 不等式约束极值问题的一阶最优性条件

**起作用约束，不起作用约束** 考虑**不等式约束极值问题**
$$
\begin{align}
\min & \quad f(\pmb x)\\
{\rm s.t.} & \quad g_i(\pmb x)\ge 0,\ i =1,\cdots,m
\end{align}
$$
在某一点 $\overline{\pmb x}\in S$ 处，一些约束（下标集用 $I$ 表示）成立等式
$$
\quad g_i(\overline{\pmb x})= 0,\ i\in I
$$
称为在 $\overline{\pmb x}$ 处**起作用约束**；另一些约束成立严格不等式
$$
\quad g_i(\overline{\pmb x})> 0,\ i\notin I
$$
称为在 $\overline{\pmb x}$ 处**不起作用约束**。

在研究某一点处的可行方向时，可以只考虑这一点的起作用约束，用符号 $I$ 表示起作用约束下标集，即
$$
I=\{i|g_i(\overline{\pmb x})= 0\}
$$


**一阶最优性必要条件** 设在不等式约束极值问题中， $\overline{\pmb x}\in S$， $f(\pmb x)$ 和 $g_i(\pmb x),(i\in I)$ 在 $\overline{\pmb x}$ 处可微， $g_i(\pmb x),(i\notin I)$ 在 $\overline{\pmb x}$ 处连续。如果 $\overline{\pmb x}$ 是局部最优解，那么 $F_0\cap G_0 =\varnothing$，其中 $G_0=\{\pmb d|\pmb d^{\rm T}\nabla g_i(\overline{\pmb x})>0, i\in I \}$。

> $G_0$ 是 $D$ 的另一种表示方法。

**Fritz John条件（上述条件的代数表达）** 设在不等式约束极值问题中， $\overline{\pmb x}\in S,I=\{i|g_i(\overline{\pmb x})= 0\},f,g_i(i\in I)$ 在 $\overline{\pmb x}$ 处可微， $g_i(i\notin I)$ 在 $\overline{\pmb x}$ 处连续，如果 $\overline{\pmb x}$ 是局部最优解，则存在不全为零的非负数 $w_0,w_i(i\in I)$，使得
$$
w_0 \nabla f(\overline{\pmb x})-\sum_{i\in I}w_i\nabla g_i(\overline{\pmb x})=\pmb 0
$$


@已知 $\overline{\pmb x}=(3,1)^{\rm T}$ 是以下非线性规划问题的最优解：
$$
\begin{align}
\min & \quad f(\pmb x)=(x_1-7)^2+(x_2-3)^2 \\
{\rm s.t.} & \quad g_1(\pmb x)=10-x_1^2-x_2^2 \ge 0\\
& \quad g_2(\pmb x)=4-x_1-x_2 \ge 0\\
& \quad g_3(\pmb x)=x_2 \ge 0\\
\end{align}
$$
验证在 $\overline{\pmb x}$ 处满足Fritz John条件（如图所示）。

![Screenshot from 2020-10-13 14-18-56.png](https://i.loli.net/2020/10/13/2fCdJA6KMeq5hgw.png)

在点 $\overline{\pmb x}=(3,1)^{\rm T}$，前两个约束是起作用约束，即 $I=\{1,2\}$，计算目标函数及起作用函数的梯度，得到
$$
\nabla f(\overline{\pmb x}) = \begin{bmatrix}-8\\-4
\end{bmatrix},\quad
\nabla g_1(\overline{\pmb x}) = \begin{bmatrix}-6\\-2
\end{bmatrix},\quad
\nabla g_2(\overline{\pmb x}) = \begin{bmatrix}-1\\-1
\end{bmatrix}
$$
解方程组
$$
w_0\begin{bmatrix}-8\\-4
\end{bmatrix}-w_1\begin{bmatrix}-6\\-2
\end{bmatrix}-w_2\begin{bmatrix}-1\\-1
\end{bmatrix}=\begin{bmatrix}0\\0
\end{bmatrix}
$$
存在非零解，因此满足Fritz John条件。



Fritz John条件允许 $w_0$ 为零，为了使 $w_0\neq 0$，我们对约束函数进一步施加限制：

**Kuhn-Tucker必要条件** 设在不等式约束极值问题中， $\overline{\pmb x}\in S,I=\{i|g_i(\overline{\pmb x})= 0\},f,g_i(i\in I)$ 在 $\overline{\pmb x}$ 处可微， $g_i(i\notin I)$ 在 $\overline{\pmb x}$ 处连续，向量组 $\{\nabla g_i(\overline{\pmb x})|i\in I\}$ 线性无关，如果 $\overline{\pmb x}$ 是局部最优解，则存在非负数 $w_i(i\in I)$，使得
$$
\nabla f(\overline{\pmb x})-\sum_{i\in I}w_i\nabla g_i(\overline{\pmb x})=\pmb 0
$$
若 $g_i(i\notin I)$ 在 $\overline{\pmb x}$ 处可微，则K-T条件可以写成等价形式：
$$
\nabla f(\overline{\pmb x})-\sum_{i\in I}w_i\nabla g_i(\overline{\pmb x})=\pmb 0\\
w_ig_i(\overline{\pmb x})=0,\quad i=1,\cdots,m\quad (互补松弛条件)\\
w_i\ge 0,\quad i=1,\cdots,m
$$
**互补松弛条件**的含义是，当 $i\notin I$ 时， $g_i(\overline{\pmb x})\neq 0$，因此 $w_i=0$ ；当 $i\in I$ 时， $g_i(\overline{\pmb x})= 0$，因此 $w_i\ge 0$。



@对于非线性规划问题
$$
\begin{align}
\min & \quad f(\pmb x)=(x_1-1)^2+x_2 \\
{\rm s.t.} & \quad g_1(\pmb x)=-x_1-x_2+2 \ge 0\\
& \quad g_2(\pmb x)=x_2 \ge 0\\
\end{align}
$$
求满足K-T条件的点。

计算目标函数和约束函数的梯度
$$
\nabla f(\pmb x) = \begin{bmatrix}2x_1-2\\1
\end{bmatrix},\quad
\nabla g_1(\pmb x) = \begin{bmatrix}-1\\-1
\end{bmatrix},\quad
\nabla g_2(\pmb x) = \begin{bmatrix}0\\1
\end{bmatrix}
$$
K-T条件为
$$
\begin{align}
& 2x_1-2+w_1=0,\\
& 1+w_1-w_2=0,\\
& w_1(-x_1-x_2+2)=0,\\
& w_2x_2=0,\\
& w_1,w_2\ge 0
\end{align}
$$
问题转化为求解上述非线性方程组，一般而言求解非线性方程组比较复杂，但这个方程组相对比较简单。

若 $w_2=0$，代入得
$$
\begin{align}
& 2x_1-2+w_1=0,\\
& w_1=-1,\\
& w_1(-x_1-x_2+2)=0,\\
& w_1\ge 0
\end{align}
$$
无解。同理，若 $x_2=0$，代入得
$$
\begin{align}
& 2x_1-2+w_1=0,\\
& 1+w_1-w_2=0,\\
& w_1(-x_1+2)=0,\\
& w_1,w_2\ge 0
\end{align}
$$
解得 $w_1=0,w_2=1,x_1=1,x_2=0$，即得到K-T点
$$
\overline{\pmb x}=\begin{bmatrix}1\\0
\end{bmatrix}
$$

> 需要验证 $\overline{\pmb x}\in S$ 

**凸优化的一阶最优性充分条件** 设在不等式约束极值问题中， $f$ 是凸函数， $g_i(i=1,\cdots,m)$ 是凹函数， $\overline{\pmb x}\in S,I=\{i|g_i(\overline{\pmb x})= 0\}$，且在 $\overline{\pmb x}$ 处K-T条件成立，则 $\overline{\pmb x}$ 为全局最优解。



@上个例子中的K-T点是全局最优解。



### 约束极值问题的一阶最优性条件

考虑约束极值问题
$$
\begin{align}
\min & \quad f(\pmb x),\quad \pmb x\in\mathbb{R}^n \\
{\rm s.t.} & \quad \pmb g(\pmb x)\ge\pmb 0\\
& \quad \pmb h(\pmb x)=\pmb 0\\
\end{align}
$$
**正则点** 设 $\overline{\pmb x}$ 为可行点，不等式约束中在 $\overline{\pmb x}$ 处起作用约束的下标集记作 $I$，如果向量组 $\{\nabla g_i(\overline{\pmb x}),\nabla h_j(\overline{\pmb x})|i\in I,j=1,2,\cdots,l\}$ 线性无关，就称 $\overline{\pmb x}$ 为约束 $\pmb g(\pmb x)\ge\pmb 0$ 和 $\pmb h(\pmb x)=\pmb 0$ 的正则点。

引入等式约束后，推导一阶最优性条件的难点在于可行移动的描述：当 $h_j$ 为非线性函数时，在任何可行点均不存在可行方向。因此为描述可行移动，需要考虑超曲面 $S=\{\pmb x|\pmb h(\pmb x)=\pmb 0\}$ 上的可行曲线。

**超曲线** 如果 $\forall t\in[t_0,t_1],\pmb h(\pmb x(t))=\pmb 0$，点集 $\{\pmb x(t)|t_0\le t\le t_1\}$ 称为超曲面 $S=\{\pmb x|\pmb h(\pmb x)=\pmb 0\}$ 上的一条**超曲线**。显然，超曲线以 $t$ 为参数，如果导数 $\pmb x'(t)=\frac{{\rm d}\pmb x(t)}{{\rm d}t}$ 存在，则称曲线是可微的。

**超切平面** 曲线 $\pmb x(t)$ 的导数 $\pmb x'(t)$ 称为在 $\pmb x(t)$ 处的**切向量**，曲面 $S$ 在点 $\pmb x$ 处的所有可微曲线的切向量的集合，称为曲面 $S$ 在点 $\pmb x$ 处的**超切平面**，记作 $T(\pmb x)$。

**定理** 设 $\overline{\pmb x}$ 是超曲面 $S=\{\pmb x|\pmb h(\pmb x)=\pmb 0\}$ 上的一个正则点（即 $\nabla h_1(\overline{\pmb x}),\nabla h_2(\overline{\pmb x}),\cdots,\nabla h_l(\overline{\pmb x})$ 线性无关），则超切平面 $T(\overline{\pmb x})$ 等于子空间 $H=\{\pmb d|\pmb d^{\rm T}\nabla h(\overline{\pmb x})=\pmb 0 \}$。



**一阶最优性必要条件** 设在约束极值问题中， $\overline{\pmb x}\in S,I=\{i|g_i(\overline{\pmb x})= 0\},f,g_i(i\in I)$ 在 $\overline{\pmb x}$ 处可微， $g_i(i\notin I)$ 在 $\overline{\pmb x}$ 处连续， $h_j$ 在 $\overline{\pmb x}$ 处连续可微，且 $\nabla h_1(\overline{\pmb x}),\nabla h_2(\overline{\pmb x}),\cdots,\nabla h_l(\overline{\pmb x})$ 线性无关，如果 $\overline{\pmb x}$ 是局部最优解，则在 $\overline{\pmb x}$ 处有
$$
F_0\cap G_0\cap H_0=\varnothing
$$
其中 $F_0,G_0,H_0$ 定义为
$$
F_0=\{\pmb d|\pmb d^{\rm T} \nabla f(\overline{\pmb x})<0  \}\\
G_0=\{\pmb d|\pmb d^{\rm T}\nabla g_i(\overline{\pmb x})>0, i\in I \}\\
H_0=\{\pmb d|\pmb d^{\rm T}\nabla h_i(\overline{\pmb x})=0, j=1,2,\cdots,l \}
$$
**Fritz John条件（上述条件的代数表达）** 设在约束极值问题中， $\overline{\pmb x}\in S,I=\{i|g_i(\overline{\pmb x})= 0\},f,g_i(i\in I)$ 在 $\overline{\pmb x}$ 处可微， $g_i(i\notin I)$ 在 $\overline{\pmb x}$ 处连续， $h_j$ 在 $\overline{\pmb x}$ 处连续可微，如果 $\overline{\pmb x}$ 是局部最优解，则存在不全为零的非负数 $w_0,w_i(i\in I),v_j(j=1,2,\cdots,l)$，使得
$$
w_0 \nabla f(\overline{\pmb x})-\sum_{i\in I}w_i\nabla g_i(\overline{\pmb x})-\sum_{j=1}^lv_j\nabla h_j(\overline{\pmb x})=\pmb 0,\ w_0,w_i\ge 0,i\in I
$$
**Kuhn-Tucker必要条件** 设在约束极值问题中， $\overline{\pmb x}\in S,I=\{i|g_i(\overline{\pmb x})= 0\},f,g_i(i\in I)$ 在 $\overline{\pmb x}$ 处可微， $g_i(i\notin I)$ 在 $\overline{\pmb x}$ 处连续， $h_j$ 在 $\overline{\pmb x}$ 处连续可微，向量组 $\{\nabla g_i(\overline{\pmb x}),\nabla h_j(\overline{\pmb x})|i\in I,j=1,\cdots,l\}$ 线性无关，如果 $\overline{\pmb x}$ 是局部最优解，则存在非负数 $w_i(i\in I),v_j(j=1,\cdots,l)$，使得
$$
\nabla f(\overline{\pmb x})-\sum_{i\in I}w_i\nabla g_i(\overline{\pmb x})-\sum_{j=1}^lv_j\nabla h_j(\overline{\pmb x})=\pmb 0,\ w_i\ge 0,i\in I
$$
若 $g_i(i\notin I)$ 在 $\overline{\pmb x}$ 处可微，则K-T条件可以写成等价形式：
$$
\nabla f(\overline{\pmb x})-\sum_{i=1}^mw_i\nabla g_i(\overline{\pmb x})-\sum_{j=1}^lv_j\nabla h_j(\overline{\pmb x})=\pmb 0\\
w_ig_i(\overline{\pmb x})=0,\quad i=1,\cdots,m\quad (互补松弛条件)\\
w_i\ge 0,\quad i=1,\cdots,m
$$
**凸优化的一阶最优性充分条件** 设在约束极值问题中， $f$ 是凸函数， $g_i(i=1,\cdots,m)$ 是凹函数， $h_j(j=1,\cdots,l)$ 是线性函数， $\overline{\pmb x}\in S,I=\{i|g_i(\overline{\pmb x})= 0\}$，且在 $\overline{\pmb x}$ 处K-T条件成立，则 $\overline{\pmb x}$ 为全局最优解。



@求下列非线性规划问题的最优解（如图所示）
$$
\begin{align}
\min & \quad (x_1-2)^2+(x_2-1)^2 \\
{\rm s.t.} & \quad -x_1^2+x_2 \ge 0\\
& \quad -x_1-x_2+2 \ge 0\\
\end{align}
$$
![Screenshot from 2020-10-13 16-26-29.png](https://i.loli.net/2020/10/13/FrtpJZKAa2EOojz.png)

计算目标函数和约束函数的梯度
$$
\nabla f(\pmb x) = \begin{bmatrix}2x_1-4\\2x_2-2
\end{bmatrix},\quad
\nabla g_1(\pmb x) = \begin{bmatrix}-2x_1\\1
\end{bmatrix},\quad
\nabla g_2(\pmb x) = \begin{bmatrix}-1\\-1
\end{bmatrix}
$$
K-T条件为
$$
\begin{align}
& 2x_1-4+2w_1x_1+w_2=0,\\
& 2x_2-2-w_1+w_2=0,\\
& w_1(-x_1^2+x_2)=0,\\
& w_2(-x_1-x_2+2)=0,\\
& w_1,w_2\ge 0
\end{align}
$$
解上述非线性方程组，验证 $\overline{\pmb x}\in S$，得 $w_1=2/3,w_2=2/3,x_1=1,x_2=1$。由于本题中 $f$ 是凸函数， $g_i$ 是凹函数，因此 $\overline{\pmb x}$ 是全局最优解。



### 二阶条件

目标函数和约束函数的二阶导数反映函数的曲率特性，对稳定算法的设计具有重要意义。对于无约束问题，二阶条件由目标函数的Hessian矩阵给出；然而约束问题比无约束问题要复杂得多，研究目标函数的Hessian矩阵仍不足够。

**超切锥** 设 $S$ 是 $\mathbb{R}^n$ 上的非空集合，点 $\overline{\pmb x}\in {\rm cl}S$，集合
$$
T=\{\pmb d|\exist \pmb x^{(k)}\in S,\pmb x^{(k)}\to \overline{\pmb x},\lambda_k>0,使得\pmb d = \lim_{k\to \infty}\lambda_k(\pmb x^{(k)}-\overline{\pmb x}) \}
$$
下图给出了超切锥的两个示例

![Screenshot from 2020-10-13 17-22-58.png](https://i.loli.net/2020/10/13/GxrdqKuveBigQUb.png)

……





# 