参考[MIT 18.06SC Linear Algebra, Fall 2011](https://www.youtube.com/watch?v=7UJ4CFRGd-U&list=PL221E2BBF13BECF6C)

# 方程组与矩阵

**方程组**
$$
\begin{cases}
    2x-y=0 \\ 
    -x+2y-z=-1 \\ 
    -3y+4z=4 \\
\end{cases}
$$

**矩阵表示**
$$
A\pmb x=\pmb b\\
\begin{bmatrix}2&-1&0\\-1&2&-1\\0&-3&4
\end{bmatrix}\begin{bmatrix}x\\y\\z
\end{bmatrix}=\begin{bmatrix}0\\-1\\4
\end{bmatrix}
$$

> 右乘（列）向量为组合列，相应地左乘（行）向量为组合行。

**row picture**

> $n$ 维空间中 $n$ 个超平面的交点。

超平面 $2x-y=0,-x+2y-z=-1和-3y+4z=4$ 的交点。

**column picture**

> $n$ 维空间中对 $n$ 个向量进行线性组合得到给定向量。

将向量 $[2,-1,0]^{\rm T},[-1,2,-3]^{\rm T},[0,-1,4]^{\rm T}$ 线性组合为向量 $[0,-1,4]^{\rm T}$。

下列问题等价：

1. 对于任意的 $\pmb b$，是否可以解方程 $A\pmb x=\pmb b$ ？
2. 对列的线性组合是否可以充满整个 $n$ 维空间？

# 高斯消元法

$$
\begin{cases}
    x+2y+z=2 \\ 
    3x+8y+z=12 \\ 
    4y+z=2 \\
\end{cases}\to\cdots\to\begin{cases}
    x+2y+z=2 \\ 
    2y-2z=6 \\ 
    5z=-10 \\
\end{cases} \\

\begin{bmatrix}[1]&2&1&|2\\3&8&1&|12\\0&4&1&|2
\end{bmatrix}\to\begin{bmatrix}[1]&2&1&|2\\0&[2]&-2&|6\\0&4&1&|2
\end{bmatrix}\to\begin{bmatrix}[1]&2&1&|2\\0&[2]&-2&|6\\0&0&[5]&|-10
\end{bmatrix}
$$

其中[]中的元素称为pivot，最后一列为増广列。消元完成后即可逆序求解方程组。

回顾上面的矩阵变换过程，用矩阵乘法表示为
$$
\begin{bmatrix}1&0&0\\-3&1&0\\0&0&1
\end{bmatrix}
\begin{bmatrix}[1]&2&1\\3&8&1\\0&4&1
\end{bmatrix}=\begin{bmatrix}[1]&2&1\\0&[2]&-2\\0&4&1
\end{bmatrix}
$$
以左乘向量理解左乘矩阵每一行的含义，例如左乘矩阵的第2行表示(-3倍的第1行)与(1倍的第2行)组合。

> 高斯消元法的时间复杂度为 $O(n^3)$。

# 矩阵乘法

$A_{m\times n}B_{n\times p}=C_{m\times p}$，其中 $c_{ij}=\sum_{k}a_{ik}b_{kj}$。这是矩阵乘法的定义。（ $A$ 的行乘 $B$ 的列）

从另外的角度理解矩阵乘法，将 $B_{n\times p}$ 看作 $p$ 个列向量，每个列向量对 $A_{m\times n}$ 的 $n$ 个 $m$ 维的列做线性组合，最终得到 $p$ 个（ $m$ 维列向量的）线性组合的结果；或者将 $A_{m\times n}$ 看作 $m$ 个行向量，每个行向量对 $B_{n\times p}$ 的 $n$ 个 $p$ 维的行做线性组合，最终得到 $m$ 个（ $p$ 维行向量的）线性组合的结果。

此外，还可以用 $A$ 的列乘 $B$ 的行，即外积， $n$ 个外积求和即为结果。

上述对矩阵的划分就是矩阵分块的操作。

> 矩阵乘法的计算复杂度为
>
> + 蛮力算法： $O(mnp)$ 
> + 

# 逆矩阵

回顾操作
$$
\begin{bmatrix}1&0&0\\-3&1&0\\0&0&1
\end{bmatrix}
\begin{bmatrix}[1]&2&1\\3&8&1\\0&4&1
\end{bmatrix}=\begin{bmatrix}[1]&2&1\\0&[2]&-2\\0&4&1
\end{bmatrix}
$$
左乘矩阵之后，我们如何再将其回复原状？也就是
$$
\begin{bmatrix}?
\end{bmatrix}\begin{bmatrix}1&0&0\\-3&1&0\\0&0&1
\end{bmatrix}
\begin{bmatrix}[1]&2&1\\3&8&1\\0&4&1
\end{bmatrix}=\begin{bmatrix}?
\end{bmatrix}\begin{bmatrix}[1]&2&1\\0&[2]&-2\\0&4&1
\end{bmatrix}=\begin{bmatrix}[1]&2&1\\3&8&1\\0&4&1
\end{bmatrix}
$$
考虑到
$$
\begin{bmatrix}1&0&0\\3&1&0\\0&0&1
\end{bmatrix}\begin{bmatrix}1&0&0\\-3&1&0\\0&0&1
\end{bmatrix}=\begin{bmatrix}1&0&0\\0&1&0\\0&0&1
\end{bmatrix}
$$
相当于在第2行减去(3倍的第1行)，再加上(3倍的第1行)，矩阵回复原状，因此称这两个矩阵互为逆矩阵，标记为 $A^{-1}A=I$。

$A$ 存在逆矩阵 $A^{-1}\iff AA^{-1}=I\iff A^{-1}A=I$，称 $A$ 为invertable, nonsingular，尽管第2个等价符号并不容易证明。

考虑一个singular矩阵
$$
A=\begin{bmatrix}1&3\\2&6
\end{bmatrix}
$$
给出它没有逆矩阵的理由：

+ 考虑 $AB$，将 $B$ 看作2个列向量，每个列向量对 $A$ 的2个2维的列做线性组合，组合结果必定是 $[1,2]^{\rm T}$ 的倍数，因而必定不是 $[1,0]^{\rm T}$ 或 $[0,1]^{\rm T}$ ；考虑 $BA$ 亦然
+ 考虑方程 $A\pmb x=0$ 有非零解 $\pmb x=[3,-1]^{\rm T}$ （也就是 $A$ 的列的线性组合可以得到0），如果存在 $A^{-1}$，那么等式两边同乘 $A^{-1}$ 得到 $A^{-1}A\pmb x=\pmb x=0$，矛盾
+ determinant为0

考虑方程 $AX=I$，即
$$
\begin{bmatrix}1&3\\2&7
\end{bmatrix}\begin{bmatrix}a&b\\c&d
\end{bmatrix}=\begin{bmatrix}1&0\\0&1
\end{bmatrix}
$$
用高斯消元的方法
$$
\begin{bmatrix}1&3\\2&7
\end{bmatrix}\begin{bmatrix}a\\b
\end{bmatrix}=\begin{bmatrix}1\\0
\end{bmatrix},\begin{bmatrix}1&3\\2&7
\end{bmatrix}\begin{bmatrix}c\\d
\end{bmatrix}=\begin{bmatrix}0\\1
\end{bmatrix}
$$
矩阵变换为
$$
\begin{bmatrix}1&3&|1&0\\2&7&|0&1
\end{bmatrix}\to\begin{bmatrix}1&3&|1&0\\0&1&|-2&1
\end{bmatrix}\to\begin{bmatrix}1&0&|7&-3\\0&1&|-2&1
\end{bmatrix}\\
[A|I]\to^{行变换}\cdots\to[I|A^{-1}]
$$
上述过程可以视作 $E$ 左乘分块矩阵 $[A|I]$，若 $EA=I$，则有 $EI=E=A^{-1}$。

$AB$ 的逆矩阵为 $B^{-1}A^{-1}$ （如果 $A,B$ 可逆），因为 $(AB)(B^{-1}A^{-1})=(B^{-1}A^{-1})(AB)=I$。

对于 $AA^{-1}=I$，等式两边转置有 $(A^{-1})^{\rm T}A^{\rm T}=I$，因此 $(A^{-1})^{\rm T}=(A^{\rm T})^{-1}$，即同一矩阵的逆和转置可以交换顺序。

# LU分解

> 参见[LU分解](https://zh.wikipedia.org/wiki/LU%E5%88%86%E8%A7%A3)

考虑高斯消元
$$
\begin{bmatrix}1&0\\-4&1
\end{bmatrix}\begin{bmatrix}2&1\\8&7
\end{bmatrix}=\begin{bmatrix}2&1\\0&3
\end{bmatrix}\\
\Rightarrow I\begin{bmatrix}2&1\\8&7
\end{bmatrix}=\begin{bmatrix}1&0\\4&1
\end{bmatrix}\begin{bmatrix}2&1\\0&3
\end{bmatrix}
$$
上式即为LU分解，其中 $L$ 表示下三角(lower triangular)矩阵，对角线元素为1， $U$ 表示上三角(upper triangular)矩阵，对角线元素为pivot。

进一步地，
$$
\begin{bmatrix}2&1\\8&7
\end{bmatrix}=\begin{bmatrix}1&0\\4&1
\end{bmatrix}\begin{bmatrix}2&1\\0&3
\end{bmatrix}=\begin{bmatrix}1&0\\4&1
\end{bmatrix}\begin{bmatrix}2&0\\0&3
\end{bmatrix}\begin{bmatrix}1&1/2\\0&1
\end{bmatrix}
$$
上式称为LDU分解，其中 $D$ 表示对角矩阵。

考虑例子
$$
A={\begin{bmatrix}1&2&3\\2&5&7\\3&5&3\\\end{bmatrix}}\\
L_{{1}}A={\begin{bmatrix}1&0&0\\-2&1&0\\-3&0&1\\\end{bmatrix}}\times {\begin{bmatrix}1&2&3\\2&5&7\\3&5&3\\\end{bmatrix}}={\begin{bmatrix}1&2&3\\0&1&1\\0&-1&-6\\\end{bmatrix}}\\
L_{{2}}(L_{{1}}A)={\begin{bmatrix}1&0&0\\0&1&0\\0&1&1\\\end{bmatrix}}\times {\begin{bmatrix}1&2&3\\0&1&1\\0&-1&-6\\\end{bmatrix}}={\begin{bmatrix}1&2&3\\0&1&1\\0&0&-5\\\end{bmatrix}}=U\\
L=L_{{1}}^{{-1}}L_{{2}}^{{-1}}={\begin{bmatrix}1&0&0\\2&1&0\\3&0&1\\\end{bmatrix}}\times {\begin{bmatrix}1&0&0\\0&1&0\\0&-1&1\\\end{bmatrix}}={\begin{bmatrix}1&0&0\\2&1&0\\3&-1&1\\\end{bmatrix}}
$$
我们发现一个好的性质，即计算 $L$ 时只需将 $L_i^{-1}$ 的左下角元素集中到一个矩阵即可。

上述LU分解使用高斯消元法。对于 $n$ 阶方阵，高斯消元总共需要 $\sum_{i=1}^n i(i-1)$ 次操作（认为一次乘法和一次加法为一次操作），即 $O(n^3)$ 次操作。

上述LU分解假设高斯消元的过程中pivot不为0，如果出现pivot为0，则可以通过交换行的方式解决，即左乘置换矩阵 $P^{-1}$，称为 $PLU$ 分解。事实上，所有 $n$ 阶可逆矩阵都有 $n$ 个非零pivot，也都有 $PLU$ 分解。

 

# 置换矩阵

> 参见[置换矩阵](https://zh.wikipedia.org/wiki/%E7%BD%AE%E6%8D%A2%E7%9F%A9%E9%98%B5)

形如
$$
{\begin{bmatrix}1&0&0\\0&0&1\\0&1&0\\\end{bmatrix}}
$$
的矩阵称为置换矩阵。 $n$ 阶置换矩阵共有 $n!$ 个，构成一个关于矩阵乘法的群。

置换矩阵是正交矩阵，即 $PP^{\rm T}=I,P^{-1}=P^{\rm T}$。

# 转置

对称矩阵定义为 $A^{\rm T}=A$。 $AA^{\rm T}$ 总是对称的，因为 $(AA^{\rm T})^{\rm T}=AA^{\rm T}$。

# 向量空间

向量空间 $\mathbb{R}^2$ 的子空间包括：

1. $\mathbb{R}^2$ 
2. 经过零点的直线
3. 零点

向量空间 $\mathbb{R}^3$ 的子空间包括：

1. $\mathbb{R}^3$ 
2. 经过零点的平面
3. 经过零点的直线
4. 零点

考虑矩阵
$$
A={\begin{bmatrix}1&3\\2&3\\4&1\\\end{bmatrix}}
$$
其列向量的所有线性组合在 $\mathbb{R}^3$ 中构成一个子空间，称为列空间。行空间同理。

# $A\pmb x=\pmb b, A\pmb x=\pmb 0$ 

考虑方程组
$$
\begin{bmatrix}1&1&2\\2&1&3\\3&1&4\\4&1&5\end{bmatrix}\begin{bmatrix}x_1\\x_2\\x_3\end{bmatrix}=\begin{bmatrix}0\\0\\0\\0\end{bmatrix}
$$
它的解组成 $\mathbb{R}^3$ 的一个子空间，称为nullspace。矩阵 $A$ 的秩为2，因此nullspace的维数为1。

> 秩为1的nullspace是一条直线，秩为2是一个平面，等等。

考虑方程组
$$
\begin{bmatrix}1&1&2\\2&1&3\\3&1&4\\4&1&5\end{bmatrix}\begin{bmatrix}x_1\\x_2\\x_3\end{bmatrix}=\begin{bmatrix}b_1\\b_2\\b_3\\b_4\end{bmatrix}
$$
它是否有解？事实上，它对于部分 $\pmb b$ 无解，对于其它 $\pmb b$ 有无穷解。

考虑矩阵 $A$ 的列空间，是 $\mathbb{R}^4$ 的一个2维子空间，对于所有不属于列空间的 $\pmb b$，方程组无解；对于所有属于列空间的 $\pmb b$，方程组有无穷解。

解可以拆分为 $A\pmb x=\pmb 0$ 的通解和 $A\pmb x=\pmb b$ 的一个特解。

> $a_1x_1+a_2x_2=b$ 的解可以拆分为通解和一个特解，即通过原点的直线的偏移。

高斯消元法解方程组 $A\pmb x=\pmb 0$，其中
$$
A=\begin{bmatrix}1&2&2&2\\2&4&6&8\\3&6&8&10\end{bmatrix}\to\begin{bmatrix}[1]&2&2&2\\0&0&[2]&4\\0&0&0&0\end{bmatrix}\to \begin{bmatrix}[1]&2&0&-2\\0&0&[1]&2\\0&0&0&0\end{bmatrix}\\
  \quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad{\rm rref(row\ reduced\ echelon\ form)}
$$
这里有2个pivot，pivot的数量称为矩阵的秩；pivot所在的列称为pivot列，其余列称为自由列，这里我们可以为 $x_2,x_4$ （对应自由列）取任意值，然后求解 $x_1,x_3$。秩和自由列数量（nullspace维数）之和即为 $\pmb x$ 维数。

事实上，对于任意rref形式的矩阵
$$
R=\begin{bmatrix}I&F\\0&0
\end{bmatrix}
$$
都有
$$
R\pmb x=0\Rightarrow \begin{bmatrix}I&F
\end{bmatrix}\begin{bmatrix}\pmb x_{pivot}\\\pmb x_{free}
\end{bmatrix}=0\Rightarrow \pmb x_{pivot}=-F\pmb x_{free}
$$
其中 $\pmb x_{free}$ 可以任意确定。令 $\pmb x_{free}=[1,0,\cdots]^{\rm T},\cdots,[\cdots,0,1]^{\rm T}$，那么通解为
$$
\pmb x=\begin{bmatrix}-F\\I
\end{bmatrix}\pmb \lambda
$$

高斯消元法解方程组 $A\pmb x=\pmb b$，其中
$$
[A|\pmb b]=\begin{bmatrix}1&2&2&2&|1\\2&4&6&8&|5\\3&6&8&10&|6\end{bmatrix}\to\begin{bmatrix}[1]&2&2&2&|1\\0&0&[2]&4&|3\\0&0&0&0&|0\end{bmatrix}
$$
方程组有解的条件的几种表述为：

1. $\pmb b$ 属于 $A$ 的列空间
2. 如果 $A$ 的行组合得到0，那么 $b$ 的同样的组合得到0
3. $A$ 的秩等于 $[A|\pmb b]$ 的秩

如果方程组可解，解步骤为：

1. 得到非齐次方程组的一个特解：令所有自由变量为0，解得剩余pivot变量
2. 求齐次方程组 $A\pmb x=0$ 的通解，即nullspace
3. 特解+通解即为方程组的解

如果 $A$ 列满秩，那么 $A$ 的rref形式为 $R=\begin{bmatrix}I\\0\end{bmatrix}$，没有自由变量，nullspace仅包含零向量， $A\pmb x=\pmb b$ 的解为 $\pmb x_p$ 或无解。

如果 $A$ 行满秩，那么 $A$ 的rref形式为 $R=\begin{bmatrix}I&F\end{bmatrix}$， $A\pmb x=\pmb b$ 一定有解；如果 $F$ 至少有1列，则有自由变量， $A\pmb x=\pmb b$ 有无穷解（通解+特解）

如果 $A$ 列满秩且行满秩，即 $m=n=r$，那么 $A$ 的rref形式为 $R=I$， $A$ 可逆， $A\pmb x=\pmb b$ 的解为唯一解 $\pmb x_p=A^{-1}\pmb b$。

综上所述，

![](https://i.loli.net/2020/12/20/A5W9hLMgFipcKBj.png)

> 考虑二阶齐次方程 $y''+y=0$，其（通）解为 $y=c_1\cos x+c_2\sin x$。
>
> 我们同样可以把解空间视作2维线性空间， $\cos x$ 和 $\sin x$ 是它的一组基。
>
> 考虑二阶非齐次方程 $y''+y=x$， $y=x$ 是它的一个特解，特解+通解即为它的解。

# 线性无关

线性无关的定义为（略）。

若 $m$ 维向量 $\pmb v_1,\cdots,\pmb v_n$ 线性无/有关，设 $A=[\pmb v_1,\cdots,\pmb v_n]$，那么 $A\pmb x=0$ 只有零解/有非零解， $A$ 的秩等于/小于 $n$，没有/有自由变量。

线性空间的一组基(basis)定义为满足下列条件的一组向量：

1. 线性无关
2. span整个空间

例如 $\mathbb{R}^3$ 的一组基为 $[1,0,0]^{\rm T},[0,1,0]^{\rm T},[0,0,1]^{\rm T}$。 $\mathbb{R}^n$ 的一组基和一个 $n$ 阶可逆矩阵一一对应。

线性空间的任意一组基都具有相同数量的向量，该数量称为线性空间的维数。

 

# 矩阵的四个子空间

![](https://i.loli.net/2020/12/22/Gd4ZiaT5rvYLcbD.png)

对于矩阵 $A_{m\times n}$，列空间 $\in \mathbb{R}^m$ 的维数为 $r$，nullspace $\in \mathbb{R}^n$ 的维数为 $n-r$ ；行空间 $\in \mathbb{R}^n$ 的维数为 $r$，左nullspace $\in \mathbb{R}^m$ 的维数为 $m-r$。

考虑对 $A_{3\times 4}$ 做行变换
$$
\begin{bmatrix}-1&2&0\\1&-1&0\\-1&0&1\end{bmatrix}\begin{bmatrix}1&2&3&1\\1&1&2&1\\1&2&3&1\end{bmatrix}=\begin{bmatrix}1&0&1&1\\0&1&1&0\\0&0&0&0\end{bmatrix}
$$
可以得到行空间的一组基（第3个矩阵的前2行）和左nullspace的一组基（第1个矩阵的第3行）。

# 矩阵空间

矩阵空间可以视作特殊的向量空间。例如 $\mathbb{R}^{3\times 3}$ 表示所有3阶实方阵构成的空间，它的一组基为
$$
\begin{bmatrix}1&0&0\\0&0&0\\0&0&0\end{bmatrix},\begin{bmatrix}0&1&0\\0&0&0\\0&0&0\end{bmatrix},\cdots,\begin{bmatrix}0&0&0\\0&0&0\\0&0&1\end{bmatrix}
$$
所有3阶对称实方阵 $S$ 构成 $\mathbb{R}^{3\times 3}$ 的一个6维子空间，所有3阶上三角方阵 $U$ 构成 $\mathbb{R}^{3\times 3}$ 的一个6维子空间； $S\bigcap U$ 即所有3阶对角方阵，构成一个3维子空间； $S+U$ 可以是任意3阶方阵，构成（9维）原空间。实际上有以下关系成立
$$
{\rm dim}(S)+{\rm dim}(U)={\rm dim}(S\bigcap U)+{\rm dim}(S+U)
$$

# 秩1矩阵

考虑秩为1的矩阵
$$
A=\begin{bmatrix}1&4&5\\2&8&10\end{bmatrix}=\begin{bmatrix}1\\2\end{bmatrix}\begin{bmatrix}1&4&5\end{bmatrix}
$$
实际上，每个秩1矩阵都可以写作两个向量的外积。

秩为 $r$ 的矩阵可以表示为 $r$ 个秩1矩阵之和，因此秩1矩阵可以看作是矩阵空间中的砖块。

# 图

考虑ABCDE五个人之间的关系，如果两人是朋友那么用线段连接，

![](https://i.loli.net/2020/12/20/D5oaQACrRbxFe9L.png)

上述关系用矩阵表示为

|      | A          | B          | C          | D          | E          |
| ---- | ---------- | ---------- | ---------- | ---------- | ---------- |
| A    | 0          | 1          | 1          | $0$ | $0$ |
| B    | 1          | 0          | 1          | $0$ | $0$ |
| C    | 1          | 1          | 0          | 1          | $0$ |
| D    | $0$ | $0$ | 1          | 0          | 1          |
| E    | $0$ | $0$ | $0$ | 1          | 0          |

求任意两人之间的距离，例如求AE之间的距离。从图中可以看到AE的距离为3，那么如何用矩阵运算得到这一结果？

矩阵
$$
R=\begin{bmatrix}0&1&1&0&0\\1&0&1&0&0\\1&1&0&1&0\\0&0&1&0&1\\0&0&0&1&0\end{bmatrix},
R^2=\begin{bmatrix}2&1&1&1&0\\1&2&1&1&0\\1&1&3&0&1\\1&1&0&2&0\\0&0&1&0&1\end{bmatrix},
R^3=\begin{bmatrix}2&3&4&1&1\\3&2&4&1&1\\4&4&2&4&0\\1&1&4&0&2\\1&1&0&2&0\end{bmatrix}
$$
$R$ 表示从某点（ABCDE对应行12345）到另一点（ABCDE对应列12345）距离为1的路线数， $R^2$ 表示距离为2的路线数，……由于 $R^3$ 的第5行第1列元素（或第1行第5列元素）首次不为零，因此距离为3。

上例将无向图换成有向图亦成立。

考虑以下电路图，u, v, w, a, b 5条通路上皆有电阻，

![](https://i.loli.net/2020/12/20/TwBPIG7xSikrhAn.png)

以图中箭头方向为正方向，计算5条通路的电势差为
$$
A\pmb x=\begin{bmatrix}1&0&-1&0\\1&-1&0&0\\0&1&-1&0\\0&0&1&-1\\0&1&0&-1\end{bmatrix}\begin{bmatrix}x_1\\x_2\\x_3\\x_4\end{bmatrix}=\begin{bmatrix}x_1-x_3\\x_1-x_2\\x_2-x_3\\x_3-x_4\\x_2-x_4\end{bmatrix}
$$
其中 $x_1,x_2,x_3,x_4$ 分别为A, B, C, D点的电势；电流为
$$
CA\pmb x={\rm diag}(c_1,c_2,c_3,c_4,c_5)\begin{bmatrix}x_1-x_3\\x_1-x_2\\x_2-x_3\\x_3-x_4\\x_2-x_4\end{bmatrix}=\begin{bmatrix}c_1(x_1-x_3)\\c_2(x_1-x_2)\\c_3(x_2-x_3)\\c_4(x_3-x_4)\\c_5(x_2-x_4)\end{bmatrix}
$$
其中 $c_1,\cdots,c_5$ 分别为5条通路的电导；基尔霍夫电流定律表示为
$$
A^{\rm T}CA\pmb x=\begin{bmatrix}1&1&0&0&0\\0&-1&1&0&1\\-1&0&-1&1&0\\0&0&0&-1&-1\end{bmatrix}\begin{bmatrix}c_1(x_1-x_3)\\c_2(x_1-x_2)\\c_3(x_2-x_3)\\c_4(x_3-x_4)\\c_5(x_2-x_4)\end{bmatrix}=0
$$
可以看到矩阵 $A$ 的nullspace和左nullspace的物理意义，分别为不产生电势差的电势，和满足基尔霍夫电流定律的电流。nullspace的维数为1，代表一个位置的电势确定则所有位置的电势确定（相等）；左nullspace的维数为2，对应图中的2个回路。同样可以看到欧拉公式成立：点数+回路数=边数+1。

找到矩阵 $A^{\rm T}$ 的3个pivot列，它们对应的电流不会形成回路，将是图包含的最大树。

# 正交

如果 $\pmb x^{\rm T}\pmb y=0$，那么称 $\pmb x,\pmb y$ 正交。如果 $\forall \pmb s\in S,\forall \pmb t\in T,\pmb s^{\rm T}\pmb t=0$，那么称子空间 $S,T$ 正交。

考虑矩阵的四个子空间，行空间与nullspace正交，列空间与左nullspace正交。因为 $\forall \pmb x \in$ nullspace， $A\pmb x=\pmb 0=\begin{bmatrix}\pmb r_1\\\pmb r_2\\\vdots\\\pmb r_m \end{bmatrix}\pmb x$，因此 $\forall \pmb r\in$ 行空间， $\pmb r^{\rm T} \pmb x=0$，行空间与nullspace正交；又因为行空间和nullspace的维数之和为 $n$，称行空间和nullspace在 $\mathbb{R}^n$ 中互补。

# 最小二乘法

我们经常遇到这样一类问题，对几个物理量进行多次测量，每次测量得到一个线性关系，但同时每次测量结果又包含了噪声，这时想要这几个物理量的估计值。例如问题[示例](https://zh.wikipedia.org/wiki/%E6%9C%80%E5%B0%8F%E4%BA%8C%E4%B9%98%E6%B3%95#%E7%A4%BA%E4%BE%8B)。

![](https://upload.wikimedia.org/wikipedia/commons/thumb/b/b0/Linear_least_squares_example2.svg/220px-Linear_least_squares_example2.svg.png)

（图中红点代表 $\pmb b$，绿线代表 $\pmb e$，与蓝线的交点代表 $A\pmb x$，蓝线的参数为 $\pmb x$ ）

该问题可以抽象为近似求解方程
$$
A_{m\times n}\pmb x=\pmb b
$$
其中 $m>n$，或者 $m>>n$。该方程的最小二乘解为
$$
A^{\rm T}A\hat{\pmb x}=A^{\rm T}\pmb b\Rightarrow \hat{\pmb x}=(A^{\rm T}A)^{-1}A^{\rm T}\pmb b
$$
最小二乘的意义是残差的平方和最小，即 $\min \|A\pmb x-\pmb b\|^2=\min \|\pmb e\|^2$。

> 直接求解 $\frac{\partial \|A\pmb x-\pmb b\|^2}{\partial\pmb x}=0$ 得到上式
>
> 参考[投影](#投影)，可以将 $A(A^{\rm T}A)^{-1}A^{\rm T}\pmb b$ 理解为 $\pmb b$ 在 $C(A)$ 的投影；

对矩阵 $A_{m\times n}$，考虑 $A^{\rm T}A$ 的性质：

+ $n$ 阶方阵
+ 对称
+ 秩等于 $A$ 的秩。因此如果 $A$ 的秩为 $n$，那么 $A^{\rm T}A$ 满秩，可逆
+ nullspace等于 $A$ 的nullspace

> 如果 $A$ 的秩为 $n$，或者说 $A$ 的列向量线性无关，证明 $A^{\rm T}A$ 可逆即证明 $A^{\rm T}A\pmb x=\pmb 0\Rightarrow \pmb x=\pmb 0$ 
> $$
> A^{\rm T}A\pmb x=\pmb 0\\
> \Rightarrow \pmb x^{\rm T}A^{\rm T}A\pmb x=\pmb 0\\
> \Rightarrow (A\pmb x)^{\rm T}A\pmb x=\pmb 0\\
> \Rightarrow A\pmb x=\pmb 0\\
> \Rightarrow \pmb x=\pmb 0\quad (A的列向量线性无关)
> $$
> 得证。

# 投影

![](https://i.loli.net/2020/12/22/814iDFGyKje3Ysq.png)

考虑 $\mathbb{R}^2$ 上的点 $\pmb b$ 和经过点 $\pmb a$ 的一维子空间，设 $\pmb b$ 在该子空间上的投影为 $\pmb p=\hat x\pmb a$，那么有上述关系，化简为
$$
\hat x=\frac{\pmb a^{\rm T}\pmb b}{\pmb a^{\rm T}\pmb a}\\
\pmb p=\pmb a\frac{\pmb a^{\rm T}\pmb b}{\pmb a^{\rm T}\pmb a}
$$
现在换一个角度观察上式，将 $\pmb p$ 视作由 $\pmb b$ 通过某个线性变换得到，即
$$
\pmb p=\frac{\pmb a\pmb a^{\rm T}}{\pmb a^{\rm T}\pmb a}\pmb b=P\pmb b
$$
可以看到：

+ 这里 $P$ 由向量 $\pmb a$ 与其自身作外积得到，因此秩为1，且行空间和列空间皆为该一维子空间（ $P\pmb b$ 一定属于 $P$ 的列空间，因此 $\pmb p$ 属于该一维子空间）
+ $P$ 是对称方阵
+ 考虑 $P^2\pmb b=P\pmb p=\pmb p$，因为既然 $P$ 将 $\pmb b$ 投影到 $\pmb p$，那么再投影一次也不会改变结果，即 $P^2=P$ 

为什么需要投影？还是前面最小二乘法的问题： $A\pmb x=\pmb b$ 无解，即 $\pmb b$ 不属于 $A$ 的列空间。那么通过投影，我们可以找到 $\pmb p\in C(A)$，并且 $\pmb p$ 与 $\pmb b$ 最接近，这时我们转而求解一个有解的近似方程 $A\hat{\pmb x}=\pmb p$。

将上述投影的例子升维：考虑 $\mathbb{R}^3$ 上的点 $\pmb b$ 和经过点 $\pmb a_1,\pmb a_2$ 的二维子空间，设 $\pmb b$ 在该子空间上的投影为 $\pmb p=\hat x_1\pmb a_1+\hat x_2\pmb a_2=A\hat{\pmb x}$，那么有关系
$$
\pmb a_1^{\rm T}(\pmb b-A\hat{\pmb x})=0,\pmb a_2^{\rm T}(\pmb b-A\hat{\pmb x})=0\\
\Rightarrow A^{\rm T}(\pmb b-A\hat{\pmb x})=\pmb 0 \Rightarrow A^{\rm T}\pmb e=\pmb 0\\
\Rightarrow A^{\rm T}A\hat{\pmb x}=A^{\rm T}\pmb b\\
\Rightarrow \hat{\pmb x}=(A^{\rm T}A)^{-1}A^{\rm T}\pmb b\\
\Rightarrow \pmb p=A (A^{\rm T}A)^{-1}A^{\rm T}\pmb b=P\pmb b
$$
可以看到：

+ 根据上式有 $\pmb e\in N(A^{\rm T})$，即 $\pmb e\perp C(A)$ （因为 $\pmb e$ 是该二维子空间的'垂线'）。 $\pmb b=\pmb p+\pmb e$，即任一向量分解为其在正交子空间 $C(A)$ 和 $N(A^{\rm T})$ 上的投影之和，而 $\pmb p=P\pmb b$，有 $\pmb e=(I-P)\pmb b$，因此 $I-P$ 可以理解为投影到 $N(A^{\rm T})$ 之操作。
+ 这里 $A$ 是3×2矩阵，因此 $(A^{\rm T}A)^{-1}$ 不能拆分；如果 $A$ 是方阵，即子空间为三维（即 $\mathbb{R}^3$ 本身），那么 $(A^{\rm T}A)^{-1}$ 可以拆分，结论将退化为 $\pmb p=\pmb b$，即投影即是自身。
+ $P$ 是对称方阵，因为 $(A^{\rm T}A)^{-1 {\rm T}}=(A^{\rm T}A)^{{\rm T}-1}=(A^{\rm T}A)^{-1}$ 
+ $P^2=P$，因为 $P^2=A (A^{\rm T}A)^{-1}A^{\rm T}A (A^{\rm T}A)^{-1}A^{\rm T}=A (A^{\rm T}A)^{-1}A^{\rm T}=P$ 
+ 如果 $\pmb b\in C(A)$，那么 $\pmb b$ 位于要投影的子空间上， $P\pmb b=\pmb b$ ；从另一个角度看，如果 $\pmb b \in C(A)$，那么 $\pmb b=A\pmb c$，因此 $P\pmb b=A (A^{\rm T}A)^{-1}A^{\rm T}A\pmb c=A\pmb c=\pmb b$。
  如果 $\pmb b \perp C(A)$，那么 $\pmb b$ 投影到原点， $P\pmb b=\pmb 0$ ；从另一个角度看，如果 $\pmb b \perp C(A)$，那么 $\pmb b\in N(A^{\rm T})$， $A^{\rm T}\pmb b=\pmb 0$，因此 $P\pmb b=A (A^{\rm T}A)^{-1}A^{\rm T}\pmb b=\pmb 0$。

# 标准正交基

一组 $n$ 个 $n$ 维正交向量 $\pmb q_1,\pmb q_2,\cdots,\pmb q_n$ 满足
$$
\pmb q_i\pmb q_j=\begin{cases}1,\quad i=j\\
0,\quad i\neq j
\end{cases}
$$
矩阵 $Q=[\pmb q_1,\pmb q_2,\cdots,\pmb q_n]$ 称为正交矩阵，满足 $Q^{\rm T}Q=\begin{bmatrix}\pmb q_1^{\rm T}\\\vdots\\\pmb q_n^{\rm T}\end{bmatrix}[\pmb q_1,\cdots,\pmb q_n]=I$，因此 $Q^{-1}=Q^{\rm T}$。

> orthonormal表示向量之间的正交关系，而orthogonal matrix表示正交矩阵（方阵）

假设在上一部分中矩阵 $A$ 的各列正交，那么

+ $P=A (A^{\rm T}A)^{-1}A^{\rm T}=A A^{\rm T}$，对称且 $P^2=P$ 
+ $P=I$，如果 $A$ 是方阵
+ $\hat{\pmb x}=A^{\rm T}\pmb b$ 

可以看到，正交向量将带来极大的简化。

Gram-Schmidt正交化： $\pmb a_1,\pmb a_2,\cdots,\pmb a_r$ 是一组线性无关的向量，对其规范正交化，取
$$
\begin{align}
\pmb b_1=&\pmb a_1\\
\pmb b_2=&\pmb a_2-\frac{\pmb b_1^{\rm T}\pmb a_2}{\pmb b_1^{\rm T}\pmb b_1}\pmb b_1 \\
\cdots&\cdots\\
\pmb b_r=&\pmb a_r-\frac{\pmb b_1^{\rm T}\pmb a_r}{\pmb b_1^{\rm T}\pmb b_1}\pmb b_1-\cdots-\frac{\pmb b_{r-1}^{\rm T}\pmb a_r}{\pmb b_{r-1}^{\rm T}\pmb b_{r-1}}\pmb b_{r-1}
\end{align}
$$
再将它们单位化即取 $\pmb e_i=\frac{1}{\Vert \pmb b_i \Vert} \pmb b_i$，得到 $A=[\pmb a_1,\pmb a_2,\cdots,\pmb a_r]$ 的列空间的一组规范正交基。

将上述过程用矩阵记录，即得到矩阵 $A$ 的 $QR$ 分解。

# 行列式

性质：

1. ${\rm det}I=1$ 

2. 交换行列式的两行或两列，行列式反号

3. 行列式的某行或某列乘以 $c$，则行列式乘以 $c$ 

4. 行列式对于任意一行或列是线性的，即

   $$
   \begin{vmatrix}a+a'&b+b'\\c&d\end{vmatrix}=
   \begin{vmatrix}a&b\\c&d\end{vmatrix}+
   \begin{vmatrix}a'&b'\\c&d\end{vmatrix}
   $$

+ 对于方阵 $A$， ${\rm det}A=0\iff A$ 不可逆。

  考虑对 $A$ 进行高斯消元法，在此过程中 ${\rm det}A$ 不变，最后得到 $n$ 个pivot时，表明 $A$ 可逆，并且 ${\rm det}A$ 等于 $n$ 个pivot之积，即不等于零；最后得到全零的行时，表明 $A$ 不可逆，并且 ${\rm det}A=0$ 

+ ${\rm det}AB={\rm det}A{\rm det}B$ 

+ ${\rm det}A^{\rm T}={\rm det}A$ 

余子式展开略……

伴随矩阵 $AA^*={\rm det}AI$ 

克莱姆法则……

> 克莱姆法则计算逆矩阵的运算开销惊人，因此实际应用中不会使用（而是使用高斯消元法）。
>
> 使用高斯消元法计算行列式的时间复杂度为

行列式的意义是广义的体积

# 特征值和特征向量

> 参考[如何理解矩阵特征值？ - 马同学的回答](https://www.zhihu.com/question/21874816/answer/181864044)

对于矩阵 $A$，每将其左乘一个向量 $\pmb x$ 都会得到一个新的向量 $A\pmb x$，就像是一个函数 $f(\pmb x)=A\pmb x$。绝大多数情形下， $A\pmb x$ 都会指向不同于 $\pmb x$ 的方向，而如果 $A\pmb x=\lambda\pmb x\ (\pmb x\neq \pmb 0)$，则称 $\lambda$ 为特征值， $\pmb x$ 为特征向量，特征值可以是负数、零、复数。

+ 考虑 $\lambda=0$ 的情形，解 $A\pmb x=\pmb 0$，意味着nullspace即为特征值0对应的特征空间。
+ 考虑 $P$ 是将 $\mathbb{R}^3$ 上的向量投影到二维子空间的投影矩阵，那么属于该二维子空间的任意向量都是特征值1对应的特征向量，该二维子空间是特征值1对应的特征空间；垂直于该二维子空间的任意向量都是特征值0对应的特征向量，nullspace是特征值0对应的特征空间。
+ 考虑置换矩阵 $P=\begin{bmatrix}0&1\\1&0\end{bmatrix}$，特征值1对应的特征向量为 $[c,c]^{\rm T}$，特征值-1对应的特征向量为 $[c,-c]^{\rm T}$ 
+ 考虑矩阵 $R=\begin{bmatrix}0&-1\\1&0\end{bmatrix}$，其将 $\mathbb{R}^2$ 上的向量逆时针旋转90°，特征值i对应的特征向量为 $[ci,c]^{\rm T}$，特征值-i对应的特征向量为 $[c,ci]^{\rm T}$ 

将 $A\pmb x=\lambda\pmb x$ 变换为 $(A-\lambda I)\pmb x=\pmb 0$，如果 $\lambda$ 为某特征值， $\pmb x$ 为对应的特征向量，那么方程有非零解，等价于 $A-\lambda I$ 不满秩， $|A-\lambda I|=0$。 $|A-\lambda I|=0$ 是关于 $\lambda$ 的 $n$ 次方程，那么根据代数基本定理，特征值为该方程的 $n$ 个复根，其中可能包含重根，而虚部不为0的复根以共轭方式成对出现。若 $A$ 是对称矩阵，则 $n$ 个复根都是实根。

# 矩阵对角化

若 $n$ 阶方阵 $A$ <u>有 $n$ 个线性无关的特征向量</u>，将它们作为方阵 $S$ 的各列，有
$$
AS=A[\pmb x_1,\cdots,\pmb x_n]=[\lambda_1\pmb x_1,\cdots,\lambda_n\pmb x_n]=[\pmb x_1,\cdots,\pmb x_n]{\rm diag}(\lambda_1,\cdots,\lambda_n)=S\Lambda\\
\Rightarrow S^{-1}AS=\Lambda\ or\ A=S\Lambda S^{-1}
$$
上述操作即为方阵 $A$ 的对角化，注意可对角化的条件。

+ $A^2=S\Lambda S^{-1}S\Lambda S^{-1}=S\Lambda^2 S^{-1},A^k=S\Lambda^k S^{-1}$，若 $|\lambda_i|<1$， $\lim_{k\to \infty}A^k=0$ 

+ $\forall \pmb u\in\mathbb{R}^n$ 可以表示为 $\pmb u=c_1\pmb x_1+\cdots+c_n\pmb x_n$，因此 $A^k\pmb u=A^k(c_1\pmb x_1+\cdots+c_n\pmb x_n)=c_1\lambda_1^k\pmb x_1+\cdots+c_n\lambda_n^k\pmb x_n$，若 $|\lambda_i|<1$，则该分量会指数减小，反之则指数增加。

  例如Fibonacci数列有递推公式 $\begin{bmatrix}F_{n+2}\\F_{n+1}\end{bmatrix}=\begin{bmatrix}1&1\\1&0\end{bmatrix}\begin{bmatrix}F_{n+1}\\F_n\end{bmatrix}$，矩阵 $A$ 的特征值为 $\frac{1\pm\sqrt{5}}{2}$，1.618对应的分量会指数增加而-.618对应的分量会指数减小直到趋于零，当 $n$ 较大时相当于 $\begin{bmatrix}F_{n+2}\\F_{n+1}\end{bmatrix}=1.618\begin{bmatrix}F_{n+1}\\F_n\end{bmatrix}$。

  > 因此可以将特征向量理解为（发生）变化的方向，而特征值即为变化的倍率
  >
  > 事实上数列递推公式的所有求解技巧都是矩阵对角化技巧

如果 $A$ 有 $n$ 个不同的特征值，那么一定有 $n$ 个线性无关的特征向量；如果有重复的特征值则不一定。 考虑矩阵 $A=\begin{bmatrix}2&1\\0&2\end{bmatrix}$，特征值 $\lambda=2$ 的代数重数为2，但线性无关的特征向量只有 $[1,0]^{\rm T}$。

# 矩阵的微分方程

考虑线性微分方程组：
$$
\begin{cases}
u_1'=-u_1+2u_2\\
u_2'=u_1-2u_2
\end{cases}
$$
边界条件为 $t=0$ 时， $\pmb u=[1,0]^{\rm T}$。

微分方程 $u'=au$ 的解为 $u=Ce^{at}$，类似地 $\pmb u'=A\pmb u$ 的解为 $\pmb u=e^{At}\pmb u(0)$。

另一方面求解 $\pmb u'=A\pmb u$，首先将 $\pmb u$ 解耦为特征向量的线性组合 $C_1(t)\pmb x_1+C_2(t)\pmb x_2$，回代得 $C_1'(t)=\lambda_1C_1(t)\Rightarrow C_1(t)=Ce^{\lambda_1t}$，因此解得 $\pmb u=C_1e^{\lambda_1t}\pmb x_1+C_2e^{\lambda_2t}\pmb x_2$。

将上述过程用矩阵表示为，将 $\pmb u$ 解耦为特征向量的线性组合 $\pmb u=S\pmb v$，回代则有 $S\pmb v'=AS\pmb v\Rightarrow \pmb v'=\Lambda\pmb v$，于是原微分方程组解耦为 $\pmb v$ 的各分量的微分方程，解得 $\pmb v=e^{\Lambda t}\pmb v(0)\Rightarrow \pmb u=Se^{\Lambda t}S^{-1}\pmb u(0)$。于是有 $e^{At}=Se^{\Lambda t}S^{-1}$。

系数矩阵 $A=\begin{bmatrix}-1&2\\1&-2\end{bmatrix}$ 的特征值为 $\lambda=0,-3$，对应的特征向量为 $[2,1]^{\rm T},[1,-1]^{\rm T}$，因此通解为 $\pmb u=C_1[2,1]^{\rm T}+C_2e^{-3t}[1,-1]^{\rm T}$，根据边界条件解得 $C_1=C_2=1/3$， $\pmb u=1/3[e^{-3t}+2,-e^{-3t}+1]^{\rm T}$。

可以看到 ${\rm Re}\lambda_i<0$ 导致该项衰减， $>0$ 导致该项爆炸， $=0$ 时该项稳定。

$e^x$ 的泰勒展开为 $e^x=1+x+\frac{1}{2}x^2+\cdots$，于是类似地，定义 $e^{A}=I+A+\frac{1}{2}A^2+\cdots$，可以验证 $e^{At}=Se^{\Lambda t}S^{-1}$ 成立，并且 $e^{\Lambda t}={\rm diag}(e^{\lambda_1t},\cdots,e^{\lambda_nt})$。

此外， $\frac{1}{1-x}=1+x+x^2+\cdots,0<x<1$，类似地， $(I-A)^{-1}=I+A+A^2+\cdots,0<A<I$，当 $A$ 较小时近似有 $(I-A)^{-1}=I+A$。

对于二阶微分方程 $y''+py'+qy=0$，同样可以将其化为 $\begin{bmatrix}y''\\y'\end{bmatrix}=\begin{bmatrix}-p&-q\\1&0\end{bmatrix}\begin{bmatrix}y'\\y\end{bmatrix}$，特征方程为 $\lambda^2+p\lambda+q=0$。这就是高阶常系数齐次线性微分方程的求解方法。

> 参见微分方程。

小结：当计算 $A^n$ 时， $|\lambda|=1$ 时稳定；而当计算 $e^A$ 时， ${\rm Re}\lambda<0$ 时稳定。

# 马尔可夫矩阵

$$
\begin{bmatrix}.1&.01&.3\\.2&.99&.3\\.7&0&.4\end{bmatrix}
$$

是一个马尔可夫矩阵，其特点为

1. 所有元素不小于0
2. 所有列的元素和为1

马尔可夫矩阵的特征值具有性质：

1. 1是其特征值，对应的特征向量的所有元素同号
2. 其它特征值的绝对值均小于1

假设马尔可夫矩阵可对角化，那么 $A^k\pmb u=A^k(c_1\pmb x_1+\cdots+c_n\pmb x_n)=c_1\lambda_1^k\pmb x_1+\cdots+c_n\lambda_n^k\pmb x_n=c_1\pmb x_1$，这里设 $\lambda_1=1$，其它 $|\lambda|<1$。也就是说马尔可夫矩阵将空间中的任意向量变换到接近特征值1对应的特征空间的方向。

考虑以下例子：某种膜材料将某封闭容器分为两部分，左侧有 $m$ 个A分子而右侧有 $n$ 个，每经过固定周期左侧分子中的10%进入右侧，而右侧分子中的20%进入左侧，那么递推关系可以表示为
$$
\begin{bmatrix}m\\n\end{bmatrix}_{k+1}=\begin{bmatrix}.9&.2\\.1&.8\end{bmatrix}\begin{bmatrix}m\\n\end{bmatrix}_k
$$
设定初始值 $\begin{bmatrix}m\\n\end{bmatrix}_0=\begin{bmatrix}0\\10000\end{bmatrix}$，则 $\begin{bmatrix}m\\n\end{bmatrix}_1=\begin{bmatrix}2000\\8000\end{bmatrix},\begin{bmatrix}m\\n\end{bmatrix}_2=\begin{bmatrix}3400\\6600\end{bmatrix}$ ……

求马尔可夫矩阵的特征值，已知 $\lambda_1=1$，迹=1.7，则 $\lambda_2=0.7$，对应的特征向量分别为 $[2,1]^{\rm T},[1,-1]^{\rm T}$。那么稳定状态与 $[2,1]^{\rm T}$ 同向，即 $\begin{bmatrix}m\\n\end{bmatrix}_\infty=\begin{bmatrix}6667\\3333\end{bmatrix}$ 

# 傅里叶级数

任意函数 $f(x)$ 的傅里叶级数为
$$
f(x)=a_0+a_1\cos(x)+a_2\cos(2x)+\cdots+b_1\sin(x)+b_2\sin(2x)+\cdots
$$
该函数可以视作无限维向量，相应的正交基为 $1,\cos(x),\cos(2x),\cdots,\sin(x),\sin(2x),\cdots$。考虑向量的正交定义为向量的点积为0，而向量的点积定义为向量各分量的积之和，那么类似地，函数的点积可以定义为函数各分量（即所有可能的 $x$ 值对应的函数值）的积之和，即函数积的积分 $f^{\rm T}g=\int_0^{2\pi}fg{\rm d}x$，函数的正交即定义为该积分为0。

 因此求特定参数，例如 $a_1$，就可以将等式两边同乘 $\cos(x)$ 并作积分，得到
$$
\int_{0}^{2\pi} f(x)\cos(x){\rm d}x=a_1\int_{0}^{2\pi}\cos^2(x){\rm d}x=\pi a_1\\
\Rightarrow a_1=\frac{1}{\pi}\int_{0}^{2\pi} f(x)\cos(x){\rm d}x
$$

# 对称矩阵

（实）对称矩阵 $A$ 具有以下性质

+ 特征值是<u>实数</u>
+ 有 $n$ 个<u>正交的</u>特征向量，因此可以用正交矩阵相似对角化，即 $A=Q\Lambda Q^{-1}$ 

> 特征值为何是实数？考虑 $A\pmb x=\lambda\pmb x$，对等式两边取共轭有 $\overline{A}\overline{\pmb x}=\overline{\lambda}\overline{\pmb x}\Rightarrow A\overline{\pmb x}=\overline{\lambda}\overline{\pmb x}$ （也就是说，如果 $A$ 有特征值和特征向量 $\lambda,\pmb x$，那么就有特征值和特征向量 $\overline{\lambda},\overline{\pmb x}$ ） $\Rightarrow \overline{\pmb x}^{\rm T}A^{\rm T}=\overline{\lambda}\overline{\pmb x}^{\rm T}\Rightarrow \overline{\pmb x}^{\rm T}A=\overline{\lambda}\overline{\pmb x}^{\rm T}\Rightarrow \overline{\pmb x}^{\rm T}A\pmb x=\overline{\lambda}\overline{\pmb x}^{\rm T}\pmb x$，又有 $A\pmb x=\lambda\pmb x\Rightarrow \overline{\pmb x}^{\rm T}A\pmb x=\lambda\overline{\pmb x}^{\rm T}\pmb x$，联立有 $\overline{\lambda}\overline{\pmb x}^{\rm T}\pmb x=\lambda\overline{\pmb x}^{\rm T}\pmb x\Rightarrow \overline{\lambda}=\lambda$。

上述推导分别用到了性质 $\overline{A}=A$ 和 $A^{\rm T}=A$，但如果仅有 $A^{\rm H}=A$，上述推导依然成立，满足后一条件的矩阵是实对称矩阵的超集，称为埃尔米特矩阵。

# 复矩阵

考虑复向量 $\pmb z\in\mathbb{C}^n$，根据性质 $z\bar{z}=|z|^2$， $\pmb z$ 应该与其共轭转置作内积以得到向量长度，即复向量的 $\overline{\pmb z}^{\rm T}\pmb z$ 相当于实向量的 $\pmb x^{\rm T}\pmb x$，记作 $\pmb z^{\rm H}\pmb z$。

因此复数域上的内积记作 $\pmb x^{\rm H}\pmb y$，其大多数情况下是复数，但当向量与自身作内积时是非负实数。

埃尔米特矩阵已在前面定义为 $A^{\rm H}=A$，与实对称矩阵一样有实特征值和正交的特征向量（注意这里的正交指 $\pmb x^{\rm H}\pmb y=0$ ）。

复数域上的 $U^{\rm H}U=I$ 对应实数域上的正交矩阵，称为酉矩阵。

考虑矩阵
$$
F_n=\begin{bmatrix}1&1&1&\cdots&1\\1&w&w^2&\cdots&w^{n-1}\\1&w^2&w^4&\cdots&w^{2(n-1)}\\\vdots&\vdots&\vdots&\ddots&\vdots\\1&w^{n-1}&w^{2(n-1)}&\cdots&w^{(n-1)(n-1)}\end{bmatrix}
$$
其中 $w=e^{\frac{2\pi}{n}i}$，例如
$$
F_4=\begin{bmatrix}1&1&1&1\\1&i&-1&-i\\1&-1&1&-1\\1&-i&-1&i\end{bmatrix}
$$
的任意两列的内积为0，因此是一个酉矩阵

（需要FFT介绍）

# 正定矩阵与二次型

若实对称矩阵 $A$ 满足下列条件之一

+ 所有特征值为正
+ 所有pivot为正（前k个pivot的积等于第k个主子式的值）
+ 各阶主子式为正
+ $\forall \pmb x\neq\pmb 0,\pmb x^{\rm T}A\pmb x>0$ 

则称 $A$ 正定，记作 $A>0$。若将上述条件改为非负，则称 $A$ 半正定，记作 $A\ge 0$。

每个实对称矩阵 $A$ 都对应一个二次型 $\pmb x^{\rm T}A\pmb x$，矩阵元素即为二次型的系数。

每个二次型又对应一个多元二次曲线，例如正定二次型对应开口朝上的抛物面，不定二次型在有些维度开口朝上，另一些维度开口朝下。

对于正定二次型， $\pmb x=\pmb 0$ 是全局最小点，因此应有 $f(\pmb x)=\pmb x^{\rm T}A\pmb x$ 在该点的梯度为0，二阶偏导数大于0。实际上 $\frac{\partial}{\partial \pmb x}\pmb x^{\rm T} A \pmb x=(A+A^{\rm T})\pmb x$ 在该点梯度就是0，而二阶偏导数为 $\frac{\partial}{\partial \pmb x}(A+A^{\rm T})\pmb x=A^{\rm T}+A=2A$，是正定矩阵。这也就是正定矩阵记作 $>0$ 的原因。

性质：

+ 对于列满秩矩阵 $A$， $A^{\rm T}A$ 是正定矩阵，因为 $\forall \pmb x\neq \pmb 0,A\pmb x\neq \pmb 0,\pmb x^{\rm T}A^{\rm T}A\pmb x=(A\pmb x)^{\rm T}A\pmb x> \pmb 0$ ；对于列不满秩矩阵 $A$， $A^{\rm T}A$ 是半正定矩阵
+ 若 $A>0,B>0$，则 $A+B>0$ 

（合同对角化 $\pmb x^{\rm T}A\pmb x\Rightarrow \pmb x^{\rm T}Q^{\rm T}\Lambda Q\pmb x=\pmb y^{\rm T}\Lambda \pmb y$ ）

# 相似矩阵与若尔当标准形

矩阵 $A,B$ 相似即存在可逆矩阵 $M$，满足 $B=M^{-1}AM$。

前面相似对角化部分又 $S^{-1}AS=\Lambda$，即 $A$ 与 $\Lambda$ 相似。<u>一族相似矩阵具有相同的特征值</u>，因为 $|\lambda I-M^{-1}AM|=|M^{-1}(\lambda I-A)M|=|M^{-1}||\lambda I-A||M|=|\lambda I-A|$，而 $\Lambda$ 是当中性质最好的那个（可对角化条件下）。

考虑以下矩阵
$$
\begin{bmatrix}2&0\\0&2\end{bmatrix},\begin{bmatrix}2&1\\0&2\end{bmatrix}
$$
我们发现前者的矩阵族仅包含其自身，因为 $M^{-1}2IM=2I$ （因此<u>具有相同特征值的矩阵不一定同属一族</u>）；后者与前者不属于同一矩阵族，因此也就无法相似对角化。

事实上，后者称为若尔当标准形，所有特征值为2,2的矩阵（前者除外）都可以相似到若尔当标准形（都不可对角化）。

考虑以下矩阵
$$
\begin{bmatrix}0&1&0&0\\0&0&1&0\\0&0&0&0\\0&0&0&0\end{bmatrix}
$$
显然其特征值为0,0,0,0，称代数重数为4，但只能找到2个特征向量 $[1,0,0,0]^{\rm T},[0,0,0,1]^{\rm T}$，称几何重数为2。事实上，上述矩阵就是一个若尔当标准形，包含2个若尔当块 $\begin{bmatrix}0&1&0\\0&0&1\\0&0&0\end{bmatrix}$ 和 $[0]$。

若尔当块是形如 $J=\begin{bmatrix}\lambda&1&\\&\lambda&1\\&&\ddots&1\\&&&\lambda\end{bmatrix}_n$ 的矩阵块，其仅有一个特征向量 $\pmb x_1=[1,0,\cdots,0]^{\rm T}$，因为 $J-\lambda I$ 的秩为 $n-1$ ；但与此同时 $(J-\lambda I)\pmb x=\pmb x_1$ 有解 $\pmb x_2=[0,1,\cdots,0]^{\rm T}$， $(J-\lambda I)\pmb x=\pmb x_2$ 又有解 $\pmb x_3=[0,0,1,\cdots,0]^{\rm T}$ ……最后用矩阵表示为 $(J-\lambda I)[\pmb x_1\ \pmb x_2\ \cdots\ \pmb x_n]=[\pmb 0\ \pmb x_1\ \cdots\ \pmb x_{n-1}]\Rightarrow [\pmb x_1\ \pmb x_2\ \cdots\ \pmb x_n]^{}(J-\lambda I)[\pmb x_1\ \pmb x_2\ \cdots\ \pmb x_n]=\begin{bmatrix}0&1&\\&0&1\\&&\ddots&1\\&&&0\end{bmatrix}_n$。此方法也是求若尔当标准形的方法。

所有矩阵都可以相似到一个若尔当标准形；若矩阵可对角化。则若尔当标准形是特殊的对角阵形式。

# 奇异值分解

<u>任意</u>（实）矩阵 $A_{m\times n}$ 可以做奇异值分解 $A_{m\times n}=U_m\Sigma_{m\times n} V^{\rm T}_n$，其中 $U,V$ 是正交矩阵， $\Sigma$ 是对角矩阵，对角元素为非负数，且从大到小排列。

> 对上式作变换
> $$
> A=U\Sigma V^{\rm T}\iff AV=U\Sigma\iff A[\pmb v_1\ \cdots\ \pmb v_r]=[\sigma_1\pmb u_1\ \cdots\ \sigma_r\pmb u_r]
> $$
> $R(A)=R(\Sigma)=r$， $U\Sigma$ 相当于对 $\Sigma$ 做行变换，因此只有前 $r$ 列有非零值， $AV$ 仅需计算 $V$ 的前 $r$ 个列向量。

考虑 $A^{\rm T}A=V\Sigma^{\rm T}U^{\rm T}U\Sigma V^{\rm T}=V\Sigma^{\rm T}\Sigma V^{\rm T}$，其中 $A^{\rm T}A$ 为半正定矩阵， $\Sigma^{\rm T}\Sigma$ 为对角阵，因此问题转换为半正定矩阵（对称且所有特征值非负）的合同对角化。尽管也可以通过相似的方法求 $U$，但这样求得的 $U,V$ 不一定能够使等式成立，因此一般将 $V$ 回代求 $U$。

> 实际上 $\pmb v_1\ \cdots\ \pmb v_r$ 是 $A$ 的行空间的一组基， $\pmb u_1\ \cdots\ \pmb u_r$ 是列空间的一组基。因为 $A\pmb v_i$ 可以视作对 $A$ 的列向量的线性组合，因此必定属于列空间；若 $\pmb v_1\ \cdots\ \pmb v_r$ 线性无关，因此 $A\pmb v_1\ \cdots\ A\pmb v_r$ 也线性无关。
>
> 相应地， $\pmb v_{r+1}\ \cdots\ \pmb v_n$ 是 $N(A)$ 的一组基， $\pmb u_{r+1}\ \cdots\ \pmb u_m$ 是 $N(A^{\rm T})$ 的一组基。

# 线性变换

线性变换 $T$ 将一个线性空间 $S_1$ 映射到另一个线性空间 $S_2$，且满足
$$
T(\pmb v+\pmb w)=T(\pmb v)+T(\pmb w)\\
T(c\pmb v)=cT(\pmb v)
$$
设 $\pmb v_1\ \cdots\ \pmb v_n$ 是 $S_1$ 的一组基，那么 $\forall \pmb v\in S_1,\pmb v =c_1\pmb v_1+\cdots+c_n\pmb v_n$，设定 $\pmb v=[c_1,\cdots,c_n]$ 为上式的坐标表示；再设 $\pmb w_1\ \cdots\ \pmb w_m$ 是 $S_2$ 的一组基，那么 $[T(\pmb v_1)\ \cdots\ T(\pmb v_n)]=[\pmb w_1\ \cdots\ \pmb w_m]A$， $A$ 为线性变换对应的矩阵。

若选用 $S_1$ 的另一组基 $\pmb v_1'\ \cdots\ \pmb v_n'$，或 $S_2$ 的另一组基 $\pmb w_1'\ \cdots\ \pmb w_m'$，对应的矩阵为 $A'$ 会与 $A$ 有所不同。

在（各）一组给定基下，线性变换与矩阵一一对应。矩阵是线性变换基于坐标的描述。

考虑 $T:\mathbb{R}^n\to \mathbb{R}^n$， $\pmb v_1\ \cdots\ \pmb v_n$ 和 $\pmb w_1\ \cdots\ \pmb w_n$ 选用同一组基，选择两组不同的基（原基和新基）时，有 $[T(\pmb v_1)\ \cdots\ T(\pmb v_n)]=[\pmb v_1\ \cdots\ \pmb v_n]A$， $[T(\pmb v_1')\ \cdots\ T(\pmb v_n')]=[\pmb v_1\ \cdots\ \pmb v_n]A'$，那么 $A$ 和 $A'$ 相似。因为对任意向量 $\pmb x=c_1\pmb v_1+\cdots+c_n\pmb v_n=c_1'\pmb v_1'+\cdots+c_n'\pmb v_n'$ 即 $V\pmb c=V'\pmb c'$，又有 $T(\pmb x)=VA\pmb c=V'A'\pmb c'$，因此 $V'^{-1}VAV^{-1}V'\pmb c'=A'\pmb c'$，而 $\pmb c'$ 可以取 $\mathbb{R}^n$ 中任意值，因此 $V'^{-1}VAV^{-1}V'=A'$。

# 基变换

设某向量在原基上的坐标表示为 $\pmb x$，在新基上的坐标表示为 $\pmb y$，而新基在原基上的坐标表示为 $W$，则有
$$
\pmb x=W\pmb y\\
\pmb y=W^{-1}\pmb x
$$

# 图片压缩

![](https://i.loli.net/2020/12/29/VxDhjQH34bfuCdc.png) 

512x512的8位灰度图片可以视作一个向量。该线性空间的一组自然基为 $[1,0,\cdots,0]^{\rm T},[0,1,\cdots,0]^{\rm T},\cdots,[0,0,\cdots,1]^{\rm T}$，但考虑到图片的特性，相邻的像素点的值高概率相等或近似，因此更好的一组基为 $[1,1,\cdots,1]^{\rm T},[1,\cdots,1,-1,\cdots,-1]^{\rm T},\cdots,[1,-1,\cdots,1,-1]^{\rm T}$。事实上，JPEG选择的一组基为傅里叶基
$$
\begin{bmatrix}1\\1\\1\\1\\1\\1\\1\\1\end{bmatrix},\begin{bmatrix}1\\w\\w^2\\w^3\\w^4\\w^5\\w^6\\w^7\end{bmatrix},\cdots
$$
JPEG的下一代——JPEG-2000选择的则是小波基
$$
\begin{bmatrix}1\\1\\1\\1\\1\\1\\1\\1\end{bmatrix},\begin{bmatrix}1\\1\\1\\1\\-1\\-1\\-1\\-1\end{bmatrix},\begin{bmatrix}1\\1\\-1\\-1\\0\\0\\0\\0\end{bmatrix},\begin{bmatrix}0\\0\\0\\0\\1\\1\\-1\\-1\end{bmatrix},\begin{bmatrix}1\\-1\\0\\0\\0\\0\\0\\0\end{bmatrix},\begin{bmatrix}0\\0\\1\\-1\\0\\0\\0\\0\end{bmatrix},\cdots
$$
由于基变换公式为 $\pmb x=W\pmb y,\pmb y=W^{-1}\pmb x$，那么好的 $W$ 需要

+ 计算快（傅里叶基有快速傅里叶变换，小波基有快速小波变换）
+ 坐标少，即多数坐标可以忽略

> 参见信号处理

# 左逆，右逆，伪逆

对于 $m>n=r$ 的矩阵 $A$，存在左逆 $A^{-1}_{left}=(A^{\rm T}A)^{-1}A^{\rm T}$ 使得 $A^{-1}_{left}A=I_n$ ；对于 $n>m=r$ 的矩阵 $A$，存在右逆 $A^{-1}_{right}=A^{\rm T}(AA^{\rm T})^{-1}$ 使得 $AA^{-1}_{right}=I_m$。

回顾投影矩阵 $P=A (A^{\rm T}A)^{-1}A^{\rm T}$，即 $AA^{-1}_{left}$ 是投影到 $A$ 的列空间的矩阵；类似地， $A^{-1}_{right}A$ 是投影到 $A$ 的行空间的矩阵。

在奇异值分解部分我们讲到，若 $\pmb v_1\ \cdots\ \pmb v_r$ 是行空间的一组基，则 $A\pmb v_1\ \cdots\ A\pmb v_r$ 是列空间的一组基， $\forall \pmb v\in$ 行空间， $\pmb v$ 和 $A\pmb v$ 在这两组基上的坐标相等；那么 $A$ 是行空间到列空间的双射，定义 $A$ 的伪逆 $A^+$，满足 $\pmb v=A^{+}(A\pmb v)$。

对于任意矩阵 $A$，作奇异值分解 $A=U\Sigma V^{\rm T}$， $\Sigma=\begin{bmatrix}\sigma_1\\&\ddots\\&&\sigma_r\\&&&0&\cdots\\&&&\vdots\end{bmatrix}_{m\times n}$，则 $\Sigma^+=\begin{bmatrix}\frac{1}{\sigma_1}\\&\ddots\\&&\frac{1}{\sigma_r}\\&&&0&\cdots\\&&&\vdots\end{bmatrix}_{n\times m}$，我们发现 $\Sigma\Sigma^+=\begin{bmatrix}I_r&0\\0&0\end{bmatrix}_m,\Sigma^+\Sigma=\begin{bmatrix}I_r&0\\0&0\end{bmatrix}_n$ 分别是投影到 $\Sigma$ 的列空间和行空间的矩阵（亦即 $\Sigma^+$ 兼具左逆和右逆的特性）； $A^+=V\Sigma^+U^{\rm T}$。

