# 随机过程

随机过程被认为是概率论的“动力学”部分，它的研究对象是随时间演变的随机现象。

设 $T$ 是一无限实数集，我们把随机变量族 $\{X(t),t\in T\}$ 称为**随机过程(stochastic process)**，其中对于 $\forall t\in T$， $X(t)$ 是一随机变量。 $T$ 称为**参数集**， $t$ 常被看作时间， $X(t)$ 称为过程在 $t$ 时刻的**状态**。对于一切 $t\in T$， $X(t)$ 的所有可能取值的集合称为随机过程的**状态空间**。

对随机过程 $\{X(t),t\in T\}$ 进行一次试验，其结果是 $t$ 的函数，记作 $x(t),t\in T$，称为随机过程的一个**样本函数**或**样本曲线**。所有不同的实验结果构成样本函数族。

在后面的叙述中，为简便起见，以 $X(t)$ 表示随机过程。



@考虑抛掷一个均匀骰子的试验：(1) 设 $X_n$ 是第 $n$ 次 $(n\ge 1)$ 掷出的点数，那么 $\{X_n,n=1,2,\cdots\}$ 构成一随机过程，属于**伯努利过程**或**伯努利随机序列**；(2) 设 $Y_n$ 是前 $n$ 次抛掷中出现的最大点数，那么 $\{Y_n,n=1,2,\cdots\}$ 也构成一随机过程。它们的状态空间都是 $\{1,2,3,4,5,6\}$。



随机过程的两种描述方式——随机变量族和样本函数族——在本质上是一致的，前者连结了概率论，常应用于理论分析；后者连结了统计学，常应用于实际测量和数据处理。

随机过程依照其在任意时刻的状态是连续型或离散型随机变量分为**连续型随机过程**和**离散型随机过程**。随机过程还可依照参数集 $T$ 是连续区间或离散集合分为**连续参数随机过程**和**离散参数随机过程**或**随机序列**。





# 随机过程的统计描述

## 随机过程的分布函数族

给定随机过程 $X(t)$，对于每一个固定的 $t\in T$，随机变量 $X(t)$ 的分布函数一般与 $t$ 有关，记作
$$
F_X(x,t)=P(X(t)\le x),\ x\in\mathbb{R}
$$
称为随机过程 $X(t)$ 的**一维分布函数**， $\{F_X(x,t),t\in T\}$ 称为**一维分布函数族**。

为了描述随机过程在不同时刻状态之间的统计联系，一般可取 $n$ 个任意不同时刻，引入 $n$ 维随机变量 $(X(t_1),X(t_2),\cdots,X_(t_n))$，其分布函数记作
$$
F_X(x_1,x_2,\cdots,x_n;t_1,t_2,\cdots,t_n)=P(X(t_1)\le x_1,X(t_2)\le x_2,\cdots,X(t_n)\le x_n),\ x_i\in\mathbb{R},i=1,2,\cdots,n
$$