参考[MIT 18.06SC Linear Algebra, Fall 2011](https://www.youtube.com/watch?v=7UJ4CFRGd-U&list=PL221E2BBF13BECF6C)

[toc]

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

> $$n$$维空间中$$n$$个超平面的交点。

超平面$$2x-y=0,-x+2y-z=-1和-3y+4z=4$$的交点。



**column picture**

> $$n$$维空间中对$$n$$个向量进行线性组合得到给定向量。

将向量$$[2,-1,0]^{\rm T},[-1,2,-3]^{\rm T},[0,-1,4]^{\rm T}$$线性组合为向量$$[0,-1,4]^{\rm T}$$。



下列问题等价：

1. 对于任意的$$\pmb b$$，是否可以解方程$$A\pmb x=\pmb b$$？
2. 对列的线性组合是否可以充满整个$$n$$维空间？



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



# 矩阵乘法

$$A_{m\times n}B_{n\times p}=C_{m\times p}$$，其中$$c_{ij}=\sum_{k}a_{ik}b_{kj}$$。这是矩阵乘法的定义。（$$A$$的行乘$$B$$的列）

从另外的角度理解矩阵乘法，将$$B_{n\times p}$$看作$$p$$个列向量，每个列向量对$$A_{m\times n}$$的$$n$$个$$m$$维的列做线性组合，最终得到$$p$$个（$$m$$维列向量的）线性组合的结果；或者将$$A_{m\times n}$$看作$$m$$个行向量，每个行向量对$$B_{n\times p}$$的$$n$$个$$p$$维的行做线性组合，最终得到$$m$$个（$$p$$维行向量的）线性组合的结果。

此外，还可以用$$A$$的列乘$$B$$的行，即外积，$$n$$个外积求和即为结果。

上述对矩阵的划分就是矩阵分块的操作。



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
相当于在第2行减去(3倍的第1行)，再加上(3倍的第1行)，矩阵回复原状，因此称这两个矩阵互为逆矩阵，标记为$$A^{-1}A=I$$。



$$A$$存在逆矩阵$$A^{-1}\iff AA^{-1}=I\iff A^{-1}A=I$$，称$$A$$为invertable, nonsingular，尽管第2个等价符号并不容易证明。

考虑一个singular矩阵
$$
A=\begin{bmatrix}1&3\\2&6
\end{bmatrix}
$$
给出它没有逆矩阵的理由：

+ 考虑$$AB$$，将$$B$$看作2个列向量，每个列向量对$$A$$的2个2维的列做线性组合，组合结果必定是$$[1,2]^{\rm T}$$的倍数，因而必定不是$$[1,0]^{\rm T}$$或$$[0,1]^{\rm T}$$；考虑$$BA$$亦然
+ 考虑方程$$A\pmb x=0$$有非零解$$\pmb x=[3,-1]^{\rm T}$$（也就是$$A$$的列的线性组合可以得到0），如果存在$$A^{-1}$$，那么等式两边同乘$$A^{-1}$$得到$$A^{-1}A\pmb x=\pmb x=0$$，矛盾
+ determinant为0



考虑方程$$AX=I$$，即
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
上述过程可以视作$$E$$左乘分块矩阵$$[A|I]$$，若$$EA=I$$，则有$$EI=E=A^{-1}$$。



$$AB$$的逆矩阵为$$B^{-1}A^{-1}$$（如果$$A,B$$可逆），因为$$(AB)(B^{-1}A^{-1})=(B^{-1}A^{-1})(AB)=I$$。

对于$$AA^{-1}=I$$，等式两边转置有$$(A^{-1})^{\rm T}A^{\rm T}=I$$，因此$$(A^{-1})^{\rm T}=(A^{\rm T})^{-1}$$，即同一矩阵的逆和转置可以交换顺序。



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
上式即为LU分解，其中$$L$$表示下三角(lower triangular)矩阵，对角线元素为1，$$U$$表示上三角(upper triangular)矩阵，对角线元素为pivot。

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
上式称为LDU分解，其中$$D$$表示对角矩阵。



考虑例子
$$
A={\begin{bmatrix}1&2&3\\2&5&7\\3&5&3\\\end{bmatrix}}\\
L_{{1}}A={\begin{bmatrix}1&0&0\\-2&1&0\\-3&0&1\\\end{bmatrix}}\times {\begin{bmatrix}1&2&3\\2&5&7\\3&5&3\\\end{bmatrix}}={\begin{bmatrix}1&2&3\\0&1&1\\0&-1&-6\\\end{bmatrix}}\\
L_{{2}}(L_{{1}}A)={\begin{bmatrix}1&0&0\\0&1&0\\0&1&1\\\end{bmatrix}}\times {\begin{bmatrix}1&2&3\\0&1&1\\0&-1&-6\\\end{bmatrix}}={\begin{bmatrix}1&2&3\\0&1&1\\0&0&-5\\\end{bmatrix}}=U\\
L=L_{{1}}^{{-1}}L_{{2}}^{{-1}}={\begin{bmatrix}1&0&0\\2&1&0\\3&0&1\\\end{bmatrix}}\times {\begin{bmatrix}1&0&0\\0&1&0\\0&-1&1\\\end{bmatrix}}={\begin{bmatrix}1&0&0\\2&1&0\\3&-1&1\\\end{bmatrix}}
$$
我们发现一个好的性质，即计算$$L$$时只需将$$L_i^{-1}$$的左下角元素集中到一个矩阵即可。



上述LU分解使用高斯消元法。对于$$n$$阶方阵，高斯消元总共需要$$\sum_{i=1}^n i(i-1)$$次操作（认为一次乘法和一次加法为一次操作），即$$O(n^3)$$次操作。



上述LU分解假设高斯消元的过程中pivot不为0，如果出现pivot为0，则可以通过交换行的方式解决，即左乘置换矩阵$$P^{-1}$$，称为$$PLU$$分解。事实上，所有$$n$$阶可逆矩阵都有$$n$$个非零pivot，也都有$$PLU$$分解。

 

# 置换矩阵

> 参见[置换矩阵](https://zh.wikipedia.org/wiki/%E7%BD%AE%E6%8D%A2%E7%9F%A9%E9%98%B5)

形如
$$
{\begin{bmatrix}1&0&0\\0&0&1\\0&1&0\\\end{bmatrix}}
$$
的矩阵称为置换矩阵。$$n$$阶置换矩阵共有$$n!$$个，构成一个关于矩阵乘法的群。

置换矩阵必定为正交矩阵，即$$PP^{\rm T}=I,P^{-1}=P^{\rm T}$$。



# 转置

对称矩阵定义为$$A^{\rm T}=A$$。$$AA^{\rm T}$$总是对称的，因为$$(AA^{\rm T})^{\rm T}=AA^{\rm T}$$。



# 向量空间

向量空间$$\mathbb{R}^2$$的子空间包括：

1. $$\mathbb{R}^2$$
2. 经过零点的直线
3. 零点

向量空间$$\mathbb{R}^3$$的子空间包括：

1. $$\mathbb{R}^3$$
2. 经过零点的平面
3. 经过零点的直线
4. 零点

考虑矩阵
$$
A={\begin{bmatrix}1&3\\2&3\\4&1\\\end{bmatrix}}
$$
其列向量的所有线性组合在$$\mathbb{R}^3$$中构成一个子空间，称为列空间。行空间同理。



# $$A\pmb x=\pmb b, A\pmb x=\pmb 0$$

考虑方程组
$$
\begin{bmatrix}1&1&2\\2&1&3\\3&1&4\\4&1&5\end{bmatrix}\begin{bmatrix}x_1\\x_2\\x_3\end{bmatrix}=\begin{bmatrix}0\\0\\0\\0\end{bmatrix}
$$
它的解组成$$\mathbb{R}^3$$的一个子空间，称为nullspace。矩阵$$A$$的秩为2，因此nullspace的维数为1。

> 秩为1的nullspace是一条直线，秩为2是一个平面，等等。



考虑方程组
$$
\begin{bmatrix}1&1&2\\2&1&3\\3&1&4\\4&1&5\end{bmatrix}\begin{bmatrix}x_1\\x_2\\x_3\end{bmatrix}=\begin{bmatrix}b_1\\b_2\\b_3\\b_4\end{bmatrix}
$$
它是否有解？事实上，它对于部分$$\pmb b$$无解，对于其它$$\pmb b$$有无穷解。

考虑矩阵$$A$$的列空间，是$$\mathbb{R}^4$$的一个2维子空间，对于所有不属于列空间的$$\pmb b$$，方程组无解；对于所有属于列空间的$$\pmb b$$，方程组有无穷解。

解可以拆分为$$A\pmb x=\pmb 0$$的通解和$$A\pmb x=\pmb b$$的一个特解。

> $$a_1x_1+a_2x_2=b$$的解可以拆分为通解和一个特解，即通过原点的直线的偏移。



高斯消元法解方程组$$A\pmb x=\pmb 0$$，其中
$$
A=\begin{bmatrix}1&2&2&2\\2&4&6&8\\3&6&8&10\end{bmatrix}\to\begin{bmatrix}[1]&2&2&2\\0&0&[2]&4\\0&0&0&0\end{bmatrix}\to \begin{bmatrix}[1]&2&0&-2\\0&0&[1]&2\\0&0&0&0\end{bmatrix}\\
  \quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad{\rm rref(row\ reduced\ echelon\ form)}
$$
这里有2个pivot，pivot的数量称为矩阵的秩；pivot所在的列称为pivot列，其余列称为自由列，这里我们可以为$$x_2,x_4$$（对应自由列）取任意值，然后求解$$x_1,x_3$$。秩和自由列数量（nullspace维数）之和即为$$\pmb x$$维数。

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
其中$$\pmb x_{free}$$可以任意确定。令$$\pmb x_{free}=[1,0,\cdots]^{\rm T},\cdots,[\cdots,0,1]^{\rm T}$$，那么通解为
$$
\pmb x=\begin{bmatrix}-F\\I
\end{bmatrix}\pmb \lambda
$$


高斯消元法解方程组$$A\pmb x=\pmb b$$，其中
$$
[A|\pmb b]=\begin{bmatrix}1&2&2&2&|1\\2&4&6&8&|5\\3&6&8&10&|6\end{bmatrix}\to\begin{bmatrix}[1]&2&2&2&|1\\0&0&[2]&4&|3\\0&0&0&0&|0\end{bmatrix}
$$
方程组有解的条件的几种表述为：

1. $$\pmb b$$属于$$A$$的列空间
2. 如果$$A$$的行组合得到0，那么$$b$$的同样的组合得到0
3. $$A$$的秩等于$$[A|\pmb b]$$的秩

如果方程组可解，解步骤为：

1. 得到非齐次方程组的一个特解：令所有自由变量为0，解得剩余pivot变量
2. 求齐次方程组$$A\pmb x=0$$的通解，即nullspace
3. 特解+通解即为方程组的解



如果$$A$$列满秩，那么$$A$$的rref形式为$$R=\begin{bmatrix}I\\0
\end{bmatrix}$$，没有自由变量，nullspace仅包含零向量，$$A\pmb x=\pmb b$$的解为$$\pmb x_p$$或无解。

如果$$A$$行满秩，那么$$A$$的rref形式为$$R=\begin{bmatrix}I&F
\end{bmatrix}$$， $$A\pmb x=\pmb b$$一定有解；如果$$F$$至少有1列，则有自由变量，$$A\pmb x=\pmb b$$有无穷解（通解+特解）

如果$$A$$列满秩且行满秩，即$$m=n=r$$，那么$$A$$的rref形式为$$R=I$$，$$A$$可逆，$$A\pmb x=\pmb b$$的解为唯一解$$\pmb x_p=A^{-1}\pmb b$$。

综上所述，

![](https://i.loli.net/2020/12/20/A5W9hLMgFipcKBj.png)

> 考虑二阶齐次方程$$y''+y=0$$，其（通）解为$$y=c_1\cos x+c_2\sin x$$。
>
> 我们同样可以把解空间视作2维线性空间，$$\cos x$$和$$\sin x$$是它的一组基。
>
> 考虑二阶非齐次方程$$y''+y=x$$，$$y=x$$是它的一个特解，特解+通解即为它的解。



# 线性无关

线性无关的定义为（略）。

若$$m$$维向量$$\pmb v_1,\cdots,\pmb v_n$$线性无/有关，设$$A=[\pmb v_1,\cdots,\pmb v_n]$$，那么$$A\pmb x=0$$只有零解/有非零解，$$A$$的秩等于/小于$$n$$，没有/有自由变量。



线性空间的一组基(basis)定义为满足下列条件的一组向量：

1. 线性无关
2. span整个空间

例如$$\mathbb{R}^3$$的一组基为$$[1,0,0]^{\rm T},[0,1,0]^{\rm T},[0,0,1]^{\rm T}$$。$$\mathbb{R}^n$$的一组基和一个$$n$$阶可逆矩阵一一对应。

线性空间的任意一组基都具有相同数量的向量，该数量称为线性空间的维数。

 

# 矩阵的四个子空间

对于矩阵$$A_{m\times n}$$，列空间$$\in \mathbb{R}^m$$的维数为$$r$$，nullspace$$\in \mathbb{R}^n$$的维数为$$n-r$$；行空间$$\in \mathbb{R}^n$$的维数为$$r$$，左nullspace$$\in \mathbb{R}^m$$的维数为$$m-r$$。

考虑对$$A_{3\times 4}$$做行变换
$$
\begin{bmatrix}-1&2&0\\1&-1&0\\-1&0&1\end{bmatrix}\begin{bmatrix}1&2&3&1\\1&1&2&1\\1&2&3&1\end{bmatrix}=\begin{bmatrix}1&0&1&1\\0&1&1&0\\0&0&0&0\end{bmatrix}
$$
可以得到行空间的一组基（第3个矩阵的前2行）和左nullspace的一组基（第1个矩阵的第3行）。



# 矩阵空间

矩阵空间可以视作特殊的向量空间。例如$$\mathbb{R}^{3\times 3}$$表示所有3阶实方阵构成的空间，它的一组基为
$$
\begin{bmatrix}1&0&0\\0&0&0\\0&0&0\end{bmatrix},\begin{bmatrix}0&1&0\\0&0&0\\0&0&0\end{bmatrix},\cdots,\begin{bmatrix}0&0&0\\0&0&0\\0&0&1\end{bmatrix}
$$
所有3阶对称实方阵$$S$$构成$$\mathbb{R}^{3\times 3}$$的一个6维子空间，所有3阶上三角方阵$$U$$构成$$\mathbb{R}^{3\times 3}$$的一个6维子空间；$$S\bigcap U$$即所有3阶对角方阵，构成一个3维子空间；$$S+U$$可以是任意3阶方阵，构成（9维）原空间。实际上有以下关系成立
$$
{\rm dim}(S)+{\rm dim}(U)={\rm dim}(S\bigcap U)+{\rm dim}(S+U)
$$


# 秩1矩阵

考虑秩为1的矩阵
$$
A=\begin{bmatrix}1&4&5\\2&8&10\end{bmatrix}=\begin{bmatrix}1\\2\end{bmatrix}\begin{bmatrix}1&4&5\end{bmatrix}
$$
实际上，每个秩1矩阵都可以写作两个向量的外积。

秩为$$r$$的矩阵可以表示为$$r$$个秩1矩阵之和，因此秩1矩阵可以看作是矩阵空间中的砖块。



# 图

考虑ABCDE五个人之间的关系，如果两人是朋友那么用线段连接，

![](https://i.loli.net/2020/12/20/D5oaQACrRbxFe9L.png)

上述关系用矩阵表示为

|      | A          | B          | C          | D          | E          |
| ---- | ---------- | ---------- | ---------- | ---------- | ---------- |
| A    | 0          | 1          | 1          | $$0$$ | $$0$$ |
| B    | 1          | 0          | 1          | $$0$$ | $$0$$ |
| C    | 1          | 1          | 0          | 1          | $$0$$ |
| D    | $$0$$ | $$0$$ | 1          | 0          | 1          |
| E    | $$0$$ | $$0$$ | $$0$$ | 1          | 0          |

求任意两人之间的距离，例如求AE之间的距离。从图中可以看到AE的距离为3，那么如何用矩阵运算得到这一结果？

矩阵
$$
R=\begin{bmatrix}0&1&1&0&0\\1&0&1&0&0\\1&1&0&1&0\\0&0&1&0&1\\0&0&0&1&0\end{bmatrix},
R^2=\begin{bmatrix}2&1&1&1&0\\1&2&1&1&0\\1&1&3&0&1\\1&1&0&2&0\\0&0&1&0&1\end{bmatrix},
R^3=\begin{bmatrix}2&3&4&1&1\\3&2&4&1&1\\4&4&2&4&0\\1&1&4&0&2\\1&1&0&2&0\end{bmatrix}
$$
$$R$$表示从某点（ABCDE对应行12345）到另一点（ABCDE对应列12345）距离为1的路线数，$$R^2$$表示距离为2的路线数，……由于$$R^3$$的第5行第1列元素（或第1行第5列元素）首次不为零，因此距离为3。

上例将无向图换成有向图亦成立。



考虑以下电路图，u, v, w, a, b5条通路上皆有电阻，

![](https://i.loli.net/2020/12/20/TwBPIG7xSikrhAn.png)

以图中箭头方向为正方向，计算5条通路的电势差为
$$
A\pmb x=\begin{bmatrix}1&0&-1&0\\1&-1&0&0\\0&1&-1&0\\0&0&1&-1\\0&1&0&-1\end{bmatrix}\begin{bmatrix}x_1\\x_2\\x_3\\x_4\end{bmatrix}=\begin{bmatrix}x_1-x_3\\x_1-x_2\\x_2-x_3\\x_3-x_4\\x_2-x_4\end{bmatrix}
$$
其中$$x_1,x_2,x_3,x_4$$分别为A, B, C, D点的电势；电流为
$$
CA\pmb x={\rm diag}(c_1,c_2,c_3,c_4,c_5)\begin{bmatrix}x_1-x_3\\x_1-x_2\\x_2-x_3\\x_3-x_4\\x_2-x_4\end{bmatrix}=\begin{bmatrix}c_1(x_1-x_3)\\c_2(x_1-x_2)\\c_3(x_2-x_3)\\c_4(x_3-x_4)\\c_5(x_2-x_4)\end{bmatrix}
$$
其中$$c_1,\cdots,c_5$$分别为5条通路的电导；基尔霍夫电流定律表示为
$$
A^{\rm T}CA\pmb x=\begin{bmatrix}1&1&0&0&0\\0&-1&1&0&1\\-1&0&-1&1&0\\0&0&0&-1&-1\end{bmatrix}\begin{bmatrix}c_1(x_1-x_3)\\c_2(x_1-x_2)\\c_3(x_2-x_3)\\c_4(x_3-x_4)\\c_5(x_2-x_4)\end{bmatrix}=0
$$
可以看到矩阵$$A$$的nullspace和左nullspace的物理意义。





参考[如何理解矩阵特征值？ - 马同学的回答](https://www.zhihu.com/question/21874816/answer/181864044)





