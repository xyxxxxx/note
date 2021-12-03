**数学优化（mathmatical optimization）**问题，也叫最优化问题，是指在一定约束条件下，求解一个目标函数的最大值（最小值）问题。

数学优化问题的定义为：给定一个目标函数（也称为代价函数） $f:\mathcal{A}\rightarrow \mathbb{R}$，寻找一个变量 $\boldsymbol x^*\in \mathcal{D}\sub \mathcal{A}$，使得对于所有 $\mathcal{D}$ 中的 $\boldsymbol x$，都满足 $f(\boldsymbol x^*)\le f(\boldsymbol x)$ （最小化）或 $f(\boldsymbol x^*)\ge f(\boldsymbol x)$ （最大化），其中 $\mathcal{D}$ 为变量 $\boldsymbol x$ 的约束集，也叫**可行域**； $\mathcal{D}$ 中的变量被称为**可行解**。

# 类型

## 离散优化和连续优化

根据输入变量 $\boldsymbol x$ 的值域是否为实数域，数学优化问题可以分为**离散优化问题**和**连续优化问题**。

### 离散优化

1. **组合优化(combinatorial optimization)** 其目标是从一个有限集合中找出使得目标函数最优的元素。在一般的组合优化问题中，集合中的元素之间存在一定的关联，可以表示为图结构。典型的组合优化问题有旅行商问题、最小生成树问题、图着色问题等。很多机器学习问题都是组合优化问题，比如特征选择、聚类问题、超参数优化问题以及结构化学习中的标签预测问题等.
2. **整数规划(integer programming)** 输入变量 $\boldsymbol x ∈ \mathbb Z^D$ 为整数向量。常见的整数规划问题通常为**整数线性规划(integer linear programming , ILP)**。

### 连续优化

**连续优化(continuous optimization)**问题是目标函数的输入变量为连续变量 $\boldsymbol x ∈ \mathbb{R}^D$，即目标函数为实函数。

## 无约束优化和约束优化

在连续优化问题中，根据是否有变量的约束条件，可以将优化问题分为无约束优化问题和约束优化问题。**无约束优化(unconstrained optimization)**问题的可行域通常为整个实数域 $\mathcal{D}=\mathbb{R}^D$ ；而**约束优化(constrained optimization)**问题中变量 $\boldsymbol x$ 需要满足一些等式或不等式的约束。约束优化问题通常使用**拉格朗日乘数法**来进行求解。

## 线性优化和非线性优化

如果目标函数和所有的约束函数都为线性函数，则该问题为**线性规划(linear programming)**问题。相反，如果目标函数或任何一个约束函数为非线性函数，则该问题为**非线性规划(Nonlinear Programming)**问题。

在非线性优化问题中，有一类比较特殊的问题是**凸优化(Convex Optimization)**问题。在凸优化问题中，变量 $\boldsymbol x$ 的可行域为**凸集(convex set)**，即对于集合中任意两点，它们的连线全部位于集合内部。目标函数 $f$ 也必须为凸函数,即满足
$$
f(α\boldsymbol x + (1 − α)\boldsymbol y) ≤ αf(\boldsymbol x) + (1 − α)f(\boldsymbol y),\ ∀α ∈ [0, 1].
$$

此外还需要等式约束函数为线性函数，不等式约束函数为凸函数。

# 优化算法

优化问题一般都可以通过迭代的方式来求解：通过猜测一个初始的估计 $\boldsymbol x_0$，然后不断迭代产生新的估计 $\boldsymbol x_1,\boldsymbol x_2,\cdots,\boldsymbol x_t$，希望 $\boldsymbol x_t$ 最终收敛到期望的最优解 $\boldsymbol x^*$。

优化算法中常用的迭代方法有线性搜索和置信域方法等。线性搜索的策略是寻找方向和步长，具体算法有梯度下降法、牛顿法、共轭梯度法等。

## 全局最小解和局部最小解

对于很多非线性优化问题，会存在若干个**局部最小值(Local Minima)**，其对应的解称为**局部最小解(Local Minimizer)**。局部最小解 $\boldsymbol x^∗$ 定义为：存在一个 $δ > 0$ , 对于所有的满足 $‖\boldsymbol x −\boldsymbol x^∗ ‖ ≤ δ$ 的 $\boldsymbol x$，都有 $ f(\boldsymbol x^∗)≤f(\boldsymbol x)$。

如果对于所有的 $\boldsymbol x∈D$，都有 $f(\boldsymbol x^∗) ≤ f(\boldsymbol x)$ 成立，则 $\boldsymbol x^∗$ 为**全局最小解(global minimizer)**。

求局部最小解一般是比较容易的，但很难保证其为全局最小解。对于线性规划或凸优化问题，局部最小解就是全局最小解。要确认一个点 $\boldsymbol x^∗$ 是否为局部最小解，通过比较它的邻域内有没有更小的函数值是不现实的。如果函数 $f(\boldsymbol x)$ 是二次连续可微的，我们可以通过检查目标函数在点 $\boldsymbol x^∗$ 的梯度 $∇f(\boldsymbol x^∗)$ 和Hessian矩阵 $∇^2 f(\boldsymbol x^∗)$ 来判断：

**定理 局部最小解的一阶必要条件** 如果 $\boldsymbol x^∗$ 为局部最小解并且函数 $f$ 在 $\boldsymbol x^∗$ 的邻域内一阶可微，则在 $∇f(\boldsymbol x^∗) = 0$。

函数 $f(\boldsymbol x)$ 的一阶偏导数为0的点也称为**驻点(stationary point)**或**临界点(critical point)**，驻点不一定为局部最小解。

**定理 局部最小解的二阶必要条件** 如果 $\boldsymbol x^∗$ 为局部最小解并且函数 $f$ 在 $\boldsymbol x^∗$ 的邻域内二阶可微，则除了 $∇f(\boldsymbol x^∗) = 0$， $∇^2 f(\boldsymbol x^∗)$ 为半正定矩阵。

> $$
> ∇f(\boldsymbol x) = [\frac{\partial f(\boldsymbol x)}{\partial x_1},\frac{\partial f(\boldsymbol x)}{\partial x_2},\cdots,\frac{\partial f(\boldsymbol x)}{\partial x_n}]^{\rm T}\\
> ∇^2f(\boldsymbol x) = [\frac{\partial^2 f(\boldsymbol x)}{\partial x_1^2},\frac{\partial^2 f(\boldsymbol x)}{\partial x_2^2},\cdots,\frac{\partial^2 f(\boldsymbol x)}{\partial x_n^2}]^{\rm T}
> $$

## 梯度下降法

**梯度下降法（gradient descent method）**，也称为**最速下降法（steepest descend method）**，经常用来求解无约束优化的最小值问题。

对于函数 $f(\boldsymbol x)$ , 如果 $f(\boldsymbol x)$ 在点 $\boldsymbol x_t$ 附近是连续可微的，那么 $f(\boldsymbol x)$ 下降最快的方向是 $f(\boldsymbol x)$ 在 $\boldsymbol x_t$ 点的梯度方向的反方向。根据泰勒一阶展开公式，有
$$
f(\pmb x_{t+1})=f(\pmb x_t +\Delta \pmb x)\approx f(\pmb x_t)+\Delta \pmb x^{\rm T}\nabla f(\pmb x_t)
$$
取 $\Delta \pmb x = -\alpha\nabla f(\pmb x_t)$，当 $\alpha$ 足够小时， $f(\pmb x_{t+1}) < f(\pmb x_t)$ 成立。

这样我们就可以从一个初始值 $\pmb x_0$ 出发，通过迭代公式
$$
\pmb x_{t+1}=\pmb x_t-\alpha_t\nabla f(\pmb x_t),\ t\ge 0
$$
生成序列 $\pmb x_0, \pmb x_1,\pmb x_2,\cdots$，使得
$$
f(\pmb x_0)\ge f(\pmb x_1)\ge f(\pmb x_2)\ge \cdots
$$
如果顺利的话，序列 $(\pmb x_n)$ 收敛到局部最小解 $\pmb x^*$。注意，每次迭代步长 $α$ 可以改变，但其取值必须合适，如果过大就不会收敛，如果过小则收敛速度太慢。

梯度下降法为一阶收敛算法，当靠近局部最小解时梯度变小，收敛速度会变慢，并且可能以 “之字形” 的方式下降。如果目标函数为二阶连续可微，我们可以采用牛顿法。**牛顿法(Newton’s method)**为二阶收敛算法，收敛速度更快，但是每次迭代需要计算Hessian矩阵的逆矩阵，复杂度较高.
相反，如果我们要求解一个最大值问题，就需要向梯度正方向迭代进行搜索，逐渐接近函数的局部最大解，这个过程则被称为**梯度上升法(Gradient Ascent Method)**。

## 牛顿法

## 拉格朗日乘数法

