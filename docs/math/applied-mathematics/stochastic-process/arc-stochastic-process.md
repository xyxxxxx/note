**随机过程(Stochastic Process)**是一组随机变量 $X_t$ 的集合, 其中 $t$ 属于一个**索引(index)**集合 $\mathcal{T}$。索引集合 $\mathcal{T}$ 可以定义在时间域或者空间域，但一般为时间域，以实数或正数表示。当 $t$ 为实数时，随机过程为**连续随机过程**；当 $t$ 为整数时，为**离散随机过程**。日常生活中的很多例子包括股票的波动、语音信号、身高的变化等都可以看作随机过程。常见的和时间相关的随机过程模型包括伯努利过程、随机游走(random walk)、马尔可夫过程等。 和空间相关的随机过程通常称为**随机场(random field)**。比如一张二维的图片，每个像素点(变量)通过空间的位置进行索引，这些像素就组成了一个随机过程。

# 马尔可夫过程

随机过程中，**马尔可夫性质(Markov property)**是指一个随机过程在给定现在状态及所有过去状态情况下，其未来状态的条件概率分布仅依赖于当前状态。以离散随机过程为例，假设随机变量 $X_0, X_1, ⋯, X_T$ 构成一个随机过程，这些随机变量的所有可能取值的集合被称为**状态空间(state space)**。如果 $X_{t+1}$ 对于过去状态的条件概率分布仅是 $X_t$ 的一个函数，则
$$
P(X_{t+1}=x_{t+1}|X_{0:t}=x_{0:t})=P(X_{t+1}=x_{t+1}|X_t=x_t)
$$

## 马尔可夫链

离散时间的马尔可夫过程也称为**马尔可夫链(Markov chain)**。如果一个马尔可夫链的条件概率
$$
P(X_{t+1}=s|X_t=s')=m_{ss'}
$$
只和状态 $s$ 和 $s'$ 相关，和时间 $t$ 无关，则称为**时间同质的马尔可夫链（time-homogeneous Markov chain）**，其中 $m_{ss'}$ 称为**状态转移概率**。如果状态空间大小 $K$ 是有限的，状态转移概率可以用一个矩阵 $\pmb M ∈\mathbb{R}^{K×K}$ 表示, 称为**状态转移矩阵(Transition Matrix)**，其中元素 $m_{ij}$ 表示状态 $s_i$ 转移到状态 $s_j$ 的概率。

**平稳分布** 假设状态空间大小为 $K$，向量 $\pmb π=[π_1,⋯,π_K]^{\rm T}$ 为状态空间中的一个分布，满足 $0\le \pi_k \le 1$ 和 $\sum\pi_k=1$。对于状态转移矩阵为 $\pmb M$ 的时间同质的马尔可夫链，若存在一个分布 $\pmb π$ 满足
$$
\pmb \pi=\pmb M \pmb \pi
$$
则称分布 $\pmb π$ 为该马尔可夫链的**平稳分布(stationary distribution)**。根据特征向量的定义可知， $\pmb π$ 为矩阵 $\pmb M$ 的(归一化)的对应特征值为1的特征向量。

如果一个马尔可夫链的状态转移矩阵 $\pmb M$ 满足**所有状态可遍历性**以及**非周期性**, 那么对于任意一个初始状态分布 $\pmb π^{(0)}$，在经过一定时间的状态转移之后,都会收敛到平稳分布，即
$$
\pmb \pi=\lim_{T\rightarrow \infty}\pmb M^T\pmb \pi^{(0)}
$$
**细致平稳条件（detailed balance condition）** 给定一个状态空间中的分布 $\pmb π ∈ [0, 1]^K$，如果一个状态转移矩阵为 $\pmb M ∈ \mathbb{R}^{K×K}$ 的马尔可夫链满足
$$
π_i m_{ij}= π_j m_{ji}
$$

则该马尔可夫链经过一定时间的状态转移后一定会收敛到分布 $\pmb π$。

# 高斯过程

**高斯过程(Gaussian process)**也是一种应用广泛的随机过程模型。假设有一组连续随机变量 $X_0,X_1,⋯,X_T$，如果由这组随机变量构成的任一有限集合
$$
X_{t_1,\cdots,t_N}=[X_{t_1},\cdots,X_{t_N}]^{\rm T}
$$
都服从一个多元正态分布，那么这组随机变量为一个高斯过程。