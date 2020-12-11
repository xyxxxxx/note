凸集和凸函数是数学优化中的基本概念，关于凸集和凸函数的一些定理在最优化问题的理论证明和算法研究中具有重要作用。



# 凸集

**凸集** 设$$S$$是$$n$$维欧氏空间$$\mathbb{R}^n$$中的一个集合，若对$$S$$中任意两点，连接它们的线段上的任意一点仍属于$$S$$；换言之，对$$S$$中任意两点$$\pmb x^{(1)},\pmb x^{(2)}$$和任意$$\lambda \in[0,1]$$，都有
$$
\lambda\pmb x^{(1)} +(1-\lambda)\pmb x^{(2)}\in S
$$
则称$$S$$为**凸集**。

![凸集](https://upload.wikimedia.org/wikipedia/commons/thumb/6/6b/Convex_polygon_illustration1.svg/220px-Convex_polygon_illustration1.svg.png)![非凸集](https://upload.wikimedia.org/wikipedia/commons/thumb/6/6c/Convex_polygon_illustration2.svg/220px-Convex_polygon_illustration2.svg.png)

**超平面(hyperplane)，半空间** $$n$$维欧氏空间$$\mathbb{R}^n$$上的集合$$H=\{\pmb x|\pmb p^{\rm T}\pmb x=\alpha \}$$称为**超平面**，其中$$\pmb p$$为$$n$$维列向量，$$\alpha$$为实数。集合$$H^-=\{\pmb x|\pmb p^{\rm T}\pmb x\le\alpha \}$$和$$H^+=\{\pmb x|\pmb p^{\rm T}\pmb x\ge\alpha \}$$称为**半空间**。

**定理** 超平面和半空间是凸集。

凸集具有以下性质：设$$S_1,S_2$$是$$\mathbb{R}^n$$中的两个凸集，$$\beta$$是实数，则

+ $$\beta S_1=\{\beta\pmb x|\pmb x\in S_1 \}$$是凸集
+ $$S_1\cap S_2$$是凸集
+ $$S_1+S_2=\{\pmb x^{(1)}+\pmb x^{(2)}|\pmb x^{(1)}\in S_1,\pmb x^{(2)}\in S_2  \}$$是凸集
+ $$S_1-S_2=\{\pmb x^{(1)}-\pmb x^{(2)}|\pmb x^{(1)}\in S_1,\pmb x^{(2)}\in S_2  \}$$是凸集



**凸锥**  设集合$$C\sub \mathbb{R}^n$$，若$$\forall \pmb x\in C$$，$$\forall \lambda \ge 0$$，都有$$\lambda \pmb x\in C$$，则称$$C$$为**锥**；若$$C$$亦为凸集，称为**凸锥**。

**多面集** 有限个半空间的交$$\{\pmb x|A\pmb x\le \pmb b\}$$称为**多面集**，其中$$A$$为$$m\times n$$矩阵，$$\pmb b$$为$$m$$维向量；若$$\pmb b=\pmb 0$$，称为**多面锥**。

@集合$$S=\{\pmb x|x_1+2x_2\le 4,x_1-x_2\le 1,x_1\ge 0,x_2\ge 0\}$$是多面集，如图所示

![Screenshot from 2020-10-12 20-00-37.png](https://i.loli.net/2020/10/12/zOADjc8Qesky4Wi.png)



**极点** 设$$S$$为非空凸集，$$\pmb x\in S$$，若$$\pmb x$$不能表示成$$S$$中两个不同点的凸组合；换言之，若假设$$\pmb x=\lambda\pmb x^{(1)} +(1-\lambda)\pmb x^{(2)},0<\lambda<1,\pmb x^{(1)},\pmb x^{(2)}\in S$$，则必有$$\pmb x = \pmb x^{(1)}=\pmb x^{(2)}$$，则称$$\pmb x$$是凸集$$S$$的**极点**。

@多边形的极点是它的所有顶点，圆的极点是它的圆周上的所有点。

**紧集** 对于$$n$$维欧氏空间$$\mathbb{R}^n$$的子集，如果它是闭集且是有界的，则称其为**紧集**。

**定理** 紧凸集中的任意一点都能表示为极点的凸组合。



**极方向** 设$$S$$为$$\mathbb{R}^n$$上的闭凸集，$$\pmb d$$为非零向量，若对$$S$$中的每一个$$\pmb x$$，都有射线
$$
\{\pmb x+\lambda \pmb d|\lambda \ge 0\}\sub S
$$
则称向量$$\pmb d$$为$$S$$的**方向**。设$$\pmb d^{(1)},\pmb d^{(2)}$$是$$S$$的两个方向，且$$\forall \lambda >0,\ \pmb d^{(1)}\neq\lambda\pmb d^{(2)}$$，则称$$\pmb d^{(1)},\pmb d^{(2)}$$是$$S$$的两个不同的方向。若方向$$\pmb d$$不能表示为该集合的两个不同方向的正的线性组合，则称$$\pmb d$$为$$S$$的**极方向**。

@对于集合$$S=\{(x_1,x_2)|x_2\ge |x_1| \}$$，凡是与向量$$(0,1)^{\rm T}$$夹角小于等于$$45°$$的向量，都是它的方向，其中$$(1,1)^{\rm T},(-1,1)^{\rm T}$$是它的2个极方向。



## 凸集分离定理

**集合分离** 设$$S_1,S_2$$是$$\mathbb{R}^n$$上的两个非空集合，$$H=\{\pmb x|\pmb p^{\rm T}\pmb x=\alpha \}$$为超平面，如果$$\forall x\in S_1,\pmb p^{\rm T}\pmb x\ge \alpha$$，$$\forall x\in S_2,\pmb p^{\rm T}\pmb x\le \alpha$$，则称超平面$$H$$分离集合$$S_1,S_2$$。

**定理** 设$$S$$是$$\mathbb{R}^n$$上的闭凸集，$$\pmb y\notin S$$，则存在唯一的点$$\overline{\pmb x}\in S$$，使得
$$
||\pmb y-\overline{\pmb x}||=\inf_{\pmb x\in S}||\pmb y-\pmb x||
$$
**定理** 设$$S$$是$$\mathbb{R}^n$$上的非空闭凸集，$$\pmb y\notin S$$，则存在非零向量$$\pmb p$$及数$$\varepsilon >0$$，使得$$\forall \pmb x\in S, \pmb p^{\rm T}\pmb y\ge\varepsilon+\pmb p^{\rm T}\pmb x$$。

此定理表明，当$$S$$为非空闭凸集，$$\pmb y\notin S$$时，$$\pmb y$$与$$S$$是可分离的。

**定理** 设$$S$$是$$\mathbb{R}^n$$上的非空凸集，$$\pmb y\in \partial S$$，则存在非零向量$$\pmb p$$，使得$$\forall \pmb x\in {\rm cl}S, \pmb p^{\rm T}\pmb y\ge\pmb p^{\rm T}\pmb x$$，其中$$\partial S$$表示$$S$$的边界，$${\rm cl}S$$表示$$S$$和其边界的并集。



**定理** 设$$S_1,S_2$$是$$\mathbb{R}^n$$上的两个非空凸集，$$S_1\cap S_2=\varnothing$$，则存在非零向量$$\pmb p$$，使得
$$
\inf \{\pmb p^{\rm T}\pmb x|\pmb x\in S_1 \}\ge \sup \{\pmb p^{\rm T}\pmb x|\pmb x\in S_2 \}
$$
**Farkas定理** 设$$A$$为$$m\times n$$矩阵，$$\pmb c$$为$$n$$维向量，则$$A\pmb x\le \pmb 0,\pmb c^{\rm T}\pmb x>0$$有解的充要条件是$$A^{\rm T}\pmb y=\pmb c,\pmb y\ge \pmb 0$$无解。

> $$\pmb y\ge \pmb 0$$即向量$$\pmb y$$的每个分量均大于等于0。

**Gordan定理** 设$$A$$为$$m\times n$$矩阵，那么$$A\pmb x<\pmb 0$$有解的充要条件是不存在非零向量$$\pmb y\ge 0$$，使$$A^{\rm T}\pmb y=\pmb 0$$。





# 凸函数

**凸函数，凹函数** 设$$S$$是$$\mathbb{R}^n$$上的非空凸集，$$f$$是定义在$$S$$上的实函数。如果$$\forall \pmb x^{(1)},\pmb x^{(2)}\in S,\forall \lambda \in (0,1)$$，都有
$$
f(\lambda \pmb x^{(1)}+(1-\lambda)\pmb x^{(2)})\le \lambda f(\pmb x^{(1)})+(1-\lambda) f(\pmb x^{(2)})
$$
则称$$f$$是$$S$$上的**凸函数**（或称下凸函数）；上式去掉等于号则为**严格凸函数**。若$$f$$是$$S$$上的**凸函数**，则$$-f$$是$$S$$上的**凹函数**（或称上凸函数）。

**定理** 设$$f_1,f_2,\cdots,f_n$$是定义在凸集$$S$$上的凸函数，实数$$\lambda_1,\lambda_2,\cdots,\lambda_n\ge 0$$，则$$\sum_i f_i\lambda_i$$也是定义在凸集$$S$$上的凸函数。

**定理** 设$$S$$是$$\mathbb{R}^n$$上的非空凸集，$$f$$是定义在$$S$$上的凸函数，$$\alpha$$是一个实数，则水平集$$S_\alpha=\{\pmb x|\pmb x\in S,f(\pmb x)\le\alpha \}$$是凸集。

**定理** 设$$S$$是$$\mathbb{R}^n$$上的凸集，$$f$$是定义在$$S$$上的凸函数，则$$f$$在$$S$$的内部连续。



**方向导数** 设$$S$$是$$\mathbb{R}^n$$上的集合，$$f$$是定义在$$S$$上的实函数，$$\overline{\pmb x}\in {\rm int}S$$，$$\pmb d$$是非零向量，$$f$$在$$\overline{\pmb x}$$处沿方向$$\pmb d$$的**方向导数**定义为
$$
{\rm D}f(\overline{\pmb x};\pmb d)=\lim_{\lambda \to 0}\frac{f(\overline{\pmb x}+\lambda \pmb d)-f(\overline{\pmb x})}{\lambda}
$$
其中$${\rm int}S$$表示集合$$S$$的内部。同理可定义右侧导数和左侧导数。若方向$$\pmb d$$的第$$j$$个分量为1，其余$$n-1$$个分量为0，则有
$$
{\rm D}f(\overline{\pmb x};\pmb d)=\frac{\partial f(\overline{\pmb x})}{\partial x_j}
$$
若$$f$$在$$\overline{\pmb x}$$处可微，则有
$$
{\rm D}f(\overline{\pmb x};\pmb d)=\pmb d^{\rm T} \nabla f(\overline{\pmb x})
$$
**定理** 设$$f$$是一个凸函数，$$\pmb x\in\mathbb{R}^n$$，$$f(\pmb x)$$在$$\pmb x$$处取有限值，则$$f$$在$$\pmb x$$处沿任何方向的右侧导数和左侧导数都存在。



凸函数的根本重要性在于下面的基本性质：

**定理** 设$$S$$是$$\mathbb{R}^n$$上的非空凸集，$$f$$是定义在$$S$$上的凸函数，则$$f$$在$$S$$上的局部极小点是全局极小点，且极小点的集合为凸集。



## 凸函数的判别

**凸函数判定的一阶充要条件** 设$$S$$是$$\mathbb{R}^n$$上的非空开凸集，$$f$$是定义在$$S$$上的可微函数，则$$f(\pmb x)$$是凸函数的充要条件是$$\forall \pmb x^{(1)},\pmb x^{(2)}\in S$$，都有
$$
f(\pmb x^{(2)})\ge f(\pmb x^{(1)})+\nabla f(\pmb x^{(1)})^{\rm T}(\pmb x^{(2)}-\pmb x^{(1)})
$$
$$f(\pmb x)$$是严格凸函数对应于上式去掉等于号。



**凸函数判定的二阶充要条件** 设$$S$$是$$\mathbb{R}^n$$上的非空开凸集，$$f$$是定义在$$S$$上的二阶可微函数，则$$f(\pmb x)$$是凸函数的充要条件是$$\forall \pmb x\in S$$，Hessian矩阵半正定。



**严格凸函数判定的二阶充分条件** 设$$S$$是$$\mathbb{R}^n$$上的非空开凸集，$$f$$是定义在$$S$$上的二阶可微函数，如果$$\forall \pmb x\in S$$，Hessian矩阵正定，则$$f(\pmb x)$$为严格凸函数。



@给定二次函数
$$
f(x_1,x_2)=2x_1^2+x_2^2-2x_1x_2+x_1+1=\frac{1}{2}(x_1,x_2)\begin{bmatrix}4 & -2\\
-2 & 2
\end{bmatrix}(x_1,x_2)^{\rm T}+x_1+1
$$
其Hessian矩阵
$$
\nabla^2f(\pmb x)=\begin{bmatrix}4 & -2\\
-2 & 2
\end{bmatrix}
$$
是正定的，因此$$f(\pmb x)$$是严格凸函数。





# 凸优化

**凸优化** 考虑以下极小化问题：
$$
\begin{align}
\min & \quad f(\pmb x)\\
{\rm s.t.} & \quad g_i(\pmb x)\le 0,\ i =1,\cdots,m,\\
&\quad h_j(\pmb x)=0,\ j=1,\cdots,l.
\end{align}
$$
其中$$f(\pmb x)$$是凸函数，$$g_i(\pmb x)$$是凸函数，$$h_i(\pmb x)$$是线性函数。问题的可行域是
$$
S=\{\pmb x|g_i(\pmb x)\le 0,\ i=1,\cdots,m;h_j(\pmb x)=0,\ j=1,\cdots,l \}
$$
由于$$g_i(\pmb x)$$是凸函数，因此满足$$g_i(\pmb x)\le 0$$的点的集合是凸集；线性函数$$h_i(\pmb x)$$既是凸函数又是凹函数，因此满足$$h_j(\pmb x)=0$$的点的集合也是凸集；$$S$$是$$m+l$$个凸集的交，因此也是凸集。这样，上述极小化问题是求凸函数在凸集上的极小点，称为**凸优化**问题。

凸优化是非线性规划中的一种重要的特殊情形，具有以下性质：

+ 凸优化的局部极小点是全局最小点，且极小点的集合是凸集
+ 如果凸优化的目标函数是严格凸函数，那么局部极小点是唯一的

解决一般的凸优化问题已经称为一项成熟的技术，就像最小二乘法和线性规划那样。如果某个问题可以表述为凸优化问题，那么我们就能迅速有效地进行求解，也就是说这个问题事实上已经得到解决。这样，判断某个问题是否属于凸优化问题或者是否可以转换为凸优化问题就成了最具挑战性的工作。



