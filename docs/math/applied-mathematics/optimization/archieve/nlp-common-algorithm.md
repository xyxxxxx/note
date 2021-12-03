考虑无约束问题
$$
\min\quad f(\pmb x),\quad \pmb x\in\mathbb{R}^n
$$
其中函数 $f(\pmb x)$ 具有一阶连续偏导数。



# 最速下降法

**最速下降法**的目标是从任意一点出发，总是选择一个使目标函数值下降最快的方法，以利于尽快到达极小点。



## 最速下降方向

可微函数 $f(\pmb x)$ 在点 $\pmb x$ 处沿方向 $\pmb p$ 的变化率可用方向导数表示
$$
{\rm D}f(\pmb x;\pmb p)=\pmb p^{\rm T} \nabla f(\pmb x)
$$
因此，求函数 $f(\pmb x)$ 在点 $\pmb x$ 处下降最快的方向，即求解下列非线性规划问题
$$
\begin{align}
\min &\quad \pmb p^{\rm T} \nabla f(\pmb x)\\
{\rm s.t.}&\quad ||\pmb p||\le 1
\end{align}
$$
根据柯西不等式，有
$$
|\pmb p^{\rm T} \nabla f(\pmb x)|\le ||\nabla f(\pmb x)||\ ||\pmb p||\le ||\nabla f(\pmb x)||\\
\Rightarrow \pmb p^{\rm T} \nabla f(\pmb x)\ge -||\nabla f(\pmb x)||
$$
因此最优解为
$$
\pmb p=-\frac{\nabla f(\pmb x)}{||\nabla f(\pmb x)||}
$$
即在点 $\pmb x$ 处<u>**负梯度方向**为**最速下降方向**</u>。

需要指出的是，上面定义的最速下降方向是在 $\ell_2$ 范数 $||\pmb p||$ 不大于1的限制下得到的，即欧氏度量意义下的最速下降方向。若改用其它度量，得到的最速下降方向与会有所不同。通常使用的最速下降法指欧氏度量意义下的最速下降法，即**梯度下降法**。



## 梯度下降算法

梯度下降法的迭代公式是
$$
\pmb x^{(k+1)}=\pmb x^{(k)}+\alpha_k\pmb p^{(k)}
$$
其中 $\pmb p^{(k)}$ 是从 $\pmb x^{(k)}$ 出发的搜索方向，这里取负梯度方向，即
$$
\pmb p^{(k)}=-\nabla f(\pmb x^{(k)})
$$
$\alpha_k$ 是从 $\pmb x^{(k)}$ 出发沿方向 $\pmb p^{(k)}$ 进行线搜索的步长，即 $\alpha_k$ 满足
$$
f(\pmb x^{(k)}+\alpha_k\pmb p^{(k)})=\min_{\alpha \ge 0}f(\pmb x^{(k)}+\alpha\pmb p^{(k)})
$$
计算步骤如下：

1. 给定初点 $\pmb x^{(1)}\in\mathbb{R}^n$，允许误差 $\varepsilon >0$，置 $k=1$ 

2. 计算搜索方向 $\pmb p^{(k)}=-\nabla f(\pmb x^{(k)})$ 

3. 若 $||\pmb p^{(k)}||\le \varepsilon$，则停止计算；否则从 $\pmb x^{(k)}$ 出发，沿 $\pmb p^{(k)}$ 进行线搜索，求 $\alpha_k$ 使得
   $$
   f(\pmb x^{(k)}+\alpha_k\pmb p^{(k)})=\min_{\alpha \ge 0}f(\pmb x^{(k)}+\alpha\pmb p^{(k)})
   $$

4. 令 $\pmb x^{(k+1)}=\pmb x^{(k)}+\alpha_k\pmb p^{(k)}$，置 $k=k+1$，goto 2



@用梯度下降法解下列非线性规划问题：
$$
\min\quad f(\pmb x)=2x_1^2+x_2^2
$$
初点 $\pmb x^{(1)}=(1,1)^{\rm T},\varepsilon = 0.1$。

计算梯度
$$
\nabla f(\pmb x)=\begin{bmatrix}4x_1\\2x_2 \end{bmatrix}
$$
第1次迭代，
$$
\pmb p^{(1)}=-\nabla f(\pmb x^{(1)})=\begin{bmatrix}-4\\-2 \end{bmatrix},\ ||\pmb p^{(1)}||=2\sqrt{5}>0.1\\
\min_{\alpha\ge 0}\quad\varphi (\alpha)\triangleq f(\pmb x^{(1)}+\alpha\pmb p^{(1)})=f(\begin{bmatrix}1-4\alpha\\1-2\alpha \end{bmatrix})=2(1-4\alpha)^2+(1-2\alpha)^2\\
令\varphi'(\alpha)=-16(1-4\alpha)-4(1-2\alpha)=0\Rightarrow
\alpha_1=\frac{5}{18}\\
\pmb x^{(2)}=\pmb x^{(1)}+\alpha_1\pmb p^{(1)}=\begin{bmatrix}-1/9 \\4/9 \end{bmatrix}
$$

> 这里线搜索使用解析法。

类似地，第2, 3次迭代，
$$
\pmb x^{(3)}=\pmb x^{(2)}+\alpha_2\pmb p^{(2)}=\frac{2}{27}\begin{bmatrix}1\\1 \end{bmatrix}\\
\pmb x^{(4)}=\pmb x^{(3)}+\alpha_3\pmb p^{(3)}=\frac{2}{243}\begin{bmatrix}-1\\4 \end{bmatrix}\\
$$
达到精度要求 $||\nabla f(\pmb x^{(4)})||=\frac{8}{243}\sqrt{5}<0.1$，于是近似解 $\overline{\pmb x}=\frac{2}{243}(-1,4)^{\rm T}$。实际上，问题的最优解为 $\pmb x^*=(0,0)^{\rm T}$。



**定理** 设 $f(\pmb x)$ 是连续可微实函数，解集合 $\Omega=\{\overline{\pmb x}|\nabla f(\overline{\pmb x})=\pmb 0 \}$，最速下降算法产生的序列 $\{\pmb x^{(k)}\}$ 包含于某个紧集，则序列 $\{\pmb x^{(k)}\}$ 的每个聚点 $\hat{\pmb x}\in \Omega$。

**定理** 设 $f(\pmb x)$ 存在连续二阶偏导数， $\overline{\pmb x}$ 是局部极小点，Hessian矩阵 $\nabla^2 f(\overline{\pmb x})$ 的最小特征值 $a>0$，最大特征值为 $A$，算法产生的序列 $\{\pmb x^{(k)}\}$ 收敛于点 $\overline{\pmb x}$，则目标函数值的序列 $\{f(\pmb x^{(k)})\}$ 以不大于
$$
(\frac{A-a}{A+a})^2
$$
的收敛比线性地收敛于 $f(\overline{\pmb x})$。

**条件数** 在以上定理中，令 $r=A/a$，则
$$
(\frac{A-a}{A+a})^2=(\frac{r-1}{r+1})^2<1
$$
$r$ 称为对称正定矩阵 $\nabla^2 f(\overline{\pmb x})$ 的**条件数**。以上定理表明，<u>条件数越小，收敛越快；条件数越大，收敛越慢</u>。

**锯齿现象** 用梯度下降法极小化目标函数时，相邻的两个搜索方向是正交的，因为
$$
\varphi(\alpha)= f(\pmb x^{(k)}+\alpha\pmb p^{(k)}), \\
\pmb p^{(k)}=-\nabla f(\pmb x^{(k)}), \\
令\ \varphi'(\alpha)=\pmb p^{(k){\rm T}} \nabla f(\pmb x^{(k)}+\alpha\pmb p^{(k)})=0\Rightarrow \alpha=\alpha_k \\
\Rightarrow \pmb p^{(k){\rm T}}\nabla f(\pmb x^{(k+1)})=0\Rightarrow -\pmb p^{(k){\rm T}}\pmb p^{(k+1)}=0
$$
即方向 $\pmb p^{(k)},\pmb p^{(k+1)}$ 正交，这表明迭代产生的序列 $\{\pmb x^{(k)}\}$ 所循路径是“之”字形的，如下图所示。特别是当 $\pmb x^{(k)}$ 接近极小点 $\overline{\pmb x}$ 时，每次迭代移动的步长很小，于是出现了**锯齿现象**，影响了收敛速率。

![Banana-SteepDesc.gif](https://upload.wikimedia.org/wikipedia/commons/thumb/6/60/Banana-SteepDesc.gif/400px-Banana-SteepDesc.gif)

当条件数比较大时，锯齿现象的影响尤为严重，原因略。



从局部看，最速下降方向是函数值下降最快的方向，选择这样的方向进行搜索是有利的；但从全局看，由于锯齿现象的影响，收敛速率大为减慢。梯度下降法并不是收敛最快的方法，相反，从全局看，它的收敛是比较慢的。因此梯度下降法一般适用于计算过程的前期迭代，而不适用于后期接近极小点的情形。





# 牛顿法

这里将线搜索中的牛顿法推广到求解一般无约束问题的牛顿法。

设 $f(\pmb x)$ 是二次可微实函数， $\pmb x\in\mathbb{R}^n$，又设 $\pmb x^{(k)}$ 是 $f(\pmb x)$ 的极小点的一个估计，把 $f(\pmb x)$ 在 $\pmb x^{(k)}$ 展成 Taylor 级数，并取二阶近似
$$
f(\pmb x)\approx \phi(\pmb x)=f(\pmb x^{(k)})+\nabla f(\pmb x^{(k)})^{\rm T}(\pmb x-\pmb x^{(k)})+\frac{1}{2}(\pmb x-\pmb x^{(k)})^{\rm T}\nabla^2 f(\pmb x^{(k)})(\pmb x-\pmb x^{(k)})
$$
其中 $\nabla^2 f(\pmb x^{(k)})$ 是 Hessian 矩阵。为求 $\phi(\pmb x)$ 的驻点，令
$$
\nabla\phi(\pmb x)=\pmb 0\\
即\ \nabla f(\pmb x^{(k)})+\nabla^2 f(\pmb x^{(k)})(\pmb x-\pmb x^{(k)})=\pmb 0
$$
设 $\nabla^2 f(\pmb x^{(k)})$ 可逆，则得到牛顿法的迭代公式
$$
\pmb x^{(k+1)}=\pmb x^{(k)}-(\nabla^2 f(\pmb x^{(k)}))^{-1}\nabla f(\pmb x^{(k)})
$$
**定理** 设 $f(\pmb x)$ 为二次连续可微函数， $\pmb x\in\mathbb{R}^n$，







# 共轭梯度法





# 拟牛顿法

