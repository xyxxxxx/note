> 参考：
>
> 最优化理论与算法 第2版 陈宝林

**数学优化(mathematical optimization)**是应用数学的一个分支，研究在特定情况下最大化或最小化某一特定函数或变量。

数学优化的问题形式为：给定一个**目标函数**（也称为代价函数）$$f:\mathcal{A}\rightarrow \mathbb{R}$$，寻找一个变量$$\boldsymbol x^*\in \mathcal{D}\sub \mathcal{A}$$，使得对于所有$$\mathcal{D}$$中的$$\boldsymbol x$$，都满足$$f(\boldsymbol x^*)\le f(\boldsymbol x)$$ （最小化）或$$f(\boldsymbol x^*)\ge f(\boldsymbol x)$$ （最大化），其中$$\mathcal{D}$$为变量$$\boldsymbol x$$的约束集，也叫**可行域**，$$\mathcal{D}$$中的变量被称为**可行解**，$$\pmb x^*$$称为**最优解**。

数学优化包含以下分支：

+ 线性规划：目标函数和约束函数（确定集合$$\mathcal{A}$$的函数）是线性函数
+ 整数规划：在线性规划的基础上，所有变量的取值限定于整数
+ 非线性规划：目标函数或约束函数中含有非线性函数
+ 组合优化：从有限集合中寻找最优解

