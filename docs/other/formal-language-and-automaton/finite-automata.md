# 确定型有穷自动机

## 确定型有穷自动机DFA

**定义** **确定型有穷自动机（deterministic finite automata, DFA）**是一个五元组
$$
M=(Q,\Sigma,\delta,q_0,F)
$$
其中

+ $Q$ 是**内部状态（internal state）**的有限集合
+ $\Sigma$ 是符号的有限集合，称为**输入字母表（input alphabet）**
+ $\delta: Q\times \Sigma$ 是一个全函数，称为**状态转移函数（state transition function）**
+ $q_0\in Q$ 是**初态（initial state）**
+ $F\sube Q$ 是**终态（final state）**集合

确定型有穷<u>接受器</u>按照下面的方式工作：开始时自动机处于初态 $q_0$，读入装置位于输入符号串最左端的符号上。每次迁移读入一个输入符号，状态转移函数决定自动机转移到另一个状态。读入装置读完符号串时，如果自动机处于它的一个终态，那么自动机就接收这个符号串，否则自动机拒绝接受这个符号串。

为了可视化DFA，我们使用**转换图（transition graph）**。转换图中的顶点表示状态，边表示转移，有一个没有起始点的箭头指向初态，终态用双圈表示。

@例如，DFA
$$
M=(\{q_0,q_1,q_2\},\{0,1\},\delta,q_0,\{q_1\})
$$
其中， $\delta$ 定义为
$$
\delta(q_0,0)=q_0,\delta(q_0,1)=q_1\\
\delta(q_1,0)=q_0,\delta(q_1,1)=q_2\\
\delta(q_2,0)=q_2,\delta(q_2,1)=q_1
$$
对应的转换图如图所示

![Screenshot from 2020-09-10 16-54-13.png](https://i.loli.net/2020/09/10/PH5lYyu3oX1aWCL.png)

## 语言和DFA接受的语言

**定义** DFA $M=(Q,\Sigma,\delta,q_0,F)$ 接收的语言是定义在 $\Sigma$ 上被 $M$ 接受的所有符号串的集合，表示为
$$
L(M)=\{w\in \Sigma^*:\delta^*(q_0,w)\in F\}
$$
其中 $\delta^*(q_0,w)=\delta(\cdots\delta(\delta(q_0,a_1),a_2)\cdots,a_n)$，对于 $w=a_1a_2\cdots a_n$ 

@例如，DFA

![Screenshot from 2020-09-10 17-02-50.png](https://i.loli.net/2020/09/10/V4GJMyrd3AhEstb.png)

状态 $q_2$ 称为**陷阱状态（trap state）**。我们看出这个自动机接受任意个 $a$ 加上一个 $b$ 的所有符号串，而不接受其它符号串，集合表示为
$$
L=\{a^nb:n\ge 0\}
$$

**定理** 设 $M=(Q,\Sigma,\delta,q_0,F)$ 是DFA， $G_M$ 是与之对应的转换图，那么对于任意 $q_i,q_j\in Q$ 和 $w\in \Sigma^+$， $\delta^*(q_i,w)=q_j$ 当且仅当 $G_M$ 中有一条从 $q_i$ 到 $q_j$ 并标有 $w$ 的通道。

> 此定理用于确保DFA和转换图的等价性，证明略。

除了转换图，我们还可以用状态转移表来表示函数 $\delta$，下表与上面的转换图是等价的，其中行表示当前状态，列表示当前输入符号，正文表示下一个状态。
$$
\begin{array}{|c|c|}
\hline
& a & b\\
\hline
q_0 & q_0 & q_1 \\
\hline
q_1 & q_2 & q_2 \\
\hline
q_2 & q_2 & q_2 \\
\hline
\end{array}
$$

## 正则语言

**定义** 一种语言 $L$ 是**正则语言（regular language）**，当且仅当存在某个确定型有穷接受器 $M$ 满足
$$
L=L(M)
$$
@证明语言
$$
L=\{awa:w\in\{a,b\}^*\}
$$
是正则语言，方法是为其构造一个DFA

![Screenshot from 2020-09-10 17-19-11.png](https://i.loli.net/2020/09/10/kd3jtvUGXRfbPsu.png)

> 严格的证明需要证明 $L\sube L(M)$ 以及 $L(M )\sube L$，这里直接作显然处理。

@证明语言
$$
L^2=\{aw_1aaw_2a:w_1,w_2\in\{a,b\}^*\}
$$
是正则语言，方法是构造DFA

![Screenshot from 2020-09-10 17-27-35.png](https://i.loli.net/2020/09/10/ocHjWNbMeItRZVP.png)

> 这里暗含了结论：正则语言 $L$ 的自连接 $L^n$ 也是正则语言。

# 非确定型有穷自动机

在非确定型有穷自动机中，自动机的每一步迁移都不是唯一的，而是一个由所有可能的迁移构成的集合。

**定义** **非确定型有穷自动机（nondeterministic finite accepter, NFA）**是一个五元组
$$
M=(Q,\Sigma,\delta,q_0,F)
$$
其中，只有 $\delta$ 的定义与DFA不同：
$$
\delta:Q\times (\Sigma\cup \{\lambda\})\to 2^Q
$$
NFA的定义与DFA有三个主要的不同点：

1. NFA的定义中， $\delta$ 的值域是幂集 $2^Q$，函数值是 $Q$ 的一个子集

   @例如，当前状态是 $q_1$，读入符号 $a$，并且有 $\delta(q_1,a)=\{q_0,q_2\}$，那么NFA的下一个状态是 $q_0$ 或 $q_2$ 

2. 允许 $\lambda$ 作为 $\delta$ 的第二个参数，即NFA可以无输入转移

3. 集合 $\delta(q_i,a)$ 可以为空，即此种情况的状态转移函数无定义

NFA也可以用转移图表示，@例如

![Screenshot from 2020-09-10 17-42-57.png](https://i.loli.net/2020/09/10/ojXOgQxnDAGY6Tb.png)

**定义** 设 $M=(Q,\Sigma,\delta,q_0,F)$ 是NFA， $G_M$ 是与之对应的转换图，那么扩展的转移函数定义为：对于任意 $q_i,q_j\in Q$ 和 $w\in \Sigma^+$， $q_j\in \delta^*(q_i,w)$ 当且仅当 $G_M$ 中有一条从 $q_i$ 到 $q_j$ 并标有 $w$ 的通道。

**定义** NFA $M=(Q,\Sigma,\delta,q_0,F)$ 接收的语言是定义在 $\Sigma$ 上被 $M$ 接受的所有符号串的集合，表示为
$$
L(M)=\{w\in \Sigma^*:\delta^*(q_0,w)\cap F\neq \varnothing\}
$$

## 为什么需要非确定型

许多确定型算法要求在某些阶段做出选择，如回溯算法：当同时存在几种策略时，我们选择其中一种走下去，直到能够判断这种策略是否是最佳的，如果不是就回退到分歧点。非确定型算法可以解决这个问题而不需要回溯，因此非确定型机器可以看作是回溯算法查找的模型。

非确定型有助于简单地解决一些问题，如下图的NFA。这个NFA接受的语言是 $\{a^3\}\cup \{a^{2n}:n\ge 1\}$，像这种不同集合的并集组成的语言，使用非确定型接受器更加自然，尽管也可以找到接受这种语言的DFA。

![Screenshot from 2020-09-11 09-51-24.png](https://i.loli.net/2020/09/11/vOGtNg4RpyJuhbV.png)

我们下面将指明两种自动机并没有本质上的区别，因此使用NFA会使得有些结论更容易得到。

# 确定型有穷自动机和非确定型有穷自动机的等价性

**定义** 对于两个有穷自动机 $M_1$ 和 $M_2$，如果
$$
L(M_1)=L(M_2)
$$
那么这两个有穷自动机等价。

> 容易看出DFA接受的任何语言NFA也可以接受，因为DFA可以看做一种特殊的NFA；反过来，也容易为 $|Q|$ 个状态的NFA构造 $2^{|Q|}$ 个状态的DFA。

**定理** 设 $L(M_N)$ 是被NFA $M_N=(Q_N,\Sigma,\delta_N,q_0,F_N)$ 接受的语言，那么一定存在一个DFA $M_D=(Q_D,\Sigma,\delta_D,\{q_0\},F_D)$，满足 $L(M_D)=L(M_N)$。

> 构造程序与证明略。

@例如，将下图的NFA转化为等价的DFA

![Screenshot from 2020-09-11 10-01-05.png](https://i.loli.net/2020/09/11/NvTU4ByIgeKSkno.png)

确定状态转移函数
$$
\begin{array}{|c|c|c|}
\hline
& a & b\\
\hline
\{q_0\} & \{q_1,q_2\} & \varnothing \\
\hline
\{q_1,q_2\} & \{q_1,q_2\} & \{q_0\} \\
\hline
\end{array}
$$
于是得到

![Screenshot from 2020-09-11 10-01-16.png](https://i.loli.net/2020/09/11/42txbWaw1RvTPpH.png)

@将下图的NFA转化为等价的DFA

![Screenshot from 2020-09-11 10-36-27.png](https://i.loli.net/2020/09/11/AIoP4VfbRpzyFaU.png)

确定状态转移函数
$$
\begin{array}{|c|c|c|}
\hline
& 0 & 1\\
\hline
\{q_0\} & \{q_0,q_1\} & \{q_1\} \\
\hline
\{q_0,q_1\} & \{q_0,q_1,q_2\} & \{q_1,q_2\} \\
\hline
\{q_1\} & \{q_2\} & \{q_2\} \\
\hline
\{q_0,q_1,q_2\} & \{q_0,q_1,q_2\} & \{q_1,q_2\} \\
\hline
\{q_1,q_2\} & \{q_2\} & \{q_2\} \\
\hline
\{q_2\} & \varnothing & \{q_2\} \\
\hline
\varnothing & \varnothing & \varnothing \\
\hline
\end{array}
$$
得到

![Screenshot from 2020-09-11 10-36-33.png](https://i.loli.net/2020/09/11/DrWpOoXHLBGMYZu.png)

# *化简DFA

**定义** 对于所有的 $w\in \Sigma^*$，如果
$$
\delta^*(p,w)\in F \Rightarrow \delta^*(q,w)\in F
$$
并且
$$
\delta^*(p,w)\notin F \Rightarrow \delta^*(q,w)\notin F
$$
则称DFA的两个状态 $p$ 和 $q$ 是**不可区分的（indistinguishable）**，否则是**可区分的（distinguishable）**。

不可区分具有等价性。

化简方法为，将所有不可区分状态构成的集合合并为一个状态，@例如将下图的DFA化简

> 具体的reduce程序略。

![Screenshot from 2020-09-11 10-34-18.png](https://i.loli.net/2020/09/11/2DJNLgdmrnatyMK.png)

**定理** 给定DFA $M$，应用程序reduce可以得到另一个DFA $\hat M$，满足 $L(M)=L(\hat M)$，并且 $\hat M$ 是所有接受 $L(M)$ 的DFA中状态数最少的。

> 证明略。

