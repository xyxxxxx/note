# 数学优化

!!! abstract "参考"
    * 《Convex Optimization》（Stephen Boyd）

**数学优化(mathematical optimization)**是应用数学的一个分支，研究在特定情况下最大化或最小化某一特定函数或变量。

**数学优化问题**可以写成如下形式：

$$
\begin{align}
\min &\quad f(\pmb x)\\
{\rm s.t.} &\quad g_i(\pmb x)\le b_i,\ i=1,\cdots,m
\end{align}
$$

其中，向量 $\pmb x=(x_1,\cdots,x_n)$ 称为问题的**优化变量**，函数 $f:\mathbb{R}^n\to \mathbb{R}$ 称为**目标函数**，函数 $g_i:\mathbb{R}^n\to \mathbb{R}, i=1,\cdots,m$ 称为**约束函数**，常数 $b_1,\cdots,b_m$ 称为约束上限或约束边界。如果在所有满足约束的向量中向量 $\pmb x^*$ 对应的目标函数值最小，则称 $\pmb x^*$ 为问题的**最优解**。

数学优化包含以下分支：

* 线性规划：目标函数和约束函数是线性函数
* 整数规划：在线性规划的基础上，所有变量的取值限定于整数
* 非线性规划：目标函数或约束函数中含有非线性函数
* 组合优化：从有限集合中寻找最优解
