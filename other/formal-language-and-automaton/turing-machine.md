比较有穷自动机和下推自动机，我们发现它们的区别在于存储空间——有穷自动机没有存储空间，下推自动机的存储空间机制为栈。如果我们给自动机更加灵活的存储机制，将可能实现更强大的功能，并且对应更大的语言族。

# 标准图灵机

如图所示，图灵机是一种自动机，它采用**带（tape）**作为临时存储。带可以被划分为若干单元，其中每一个单元内包含一个符号，图灵机的**读写头（read-write head）**能够在带上左右移动，每次移动能够读写一个符号。

![Screenshot from 2020-09-14 13-41-54.png](https://i.loli.net/2020/09/14/HD5Z4kW8dNgj3a9.png)

**定义** 图灵机$$M$$由一个七元组定义
$$
M=(Q,\Sigma,\Gamma,\delta,q_0,\square,F)
$$
其中

+ $$Q$$是控制部件内部状态的有穷集
+ $$\Sigma$$是输入字母表
+ $$\Gamma$$是要给有穷符号集，称为**带字母表（tape alphabet）**
+ $$\delta:Q\times \Gamma \to Q\times \Gamma \times \{L,R\}$$的有穷子集，称为状态转移函数
+ $$q_0\in Q$$是控制部件的初始状态
+ $$\square\in \Gamma$$是**空白符（blank）**
+ $$F\sube Q$$是终止状态集合



@下图展示了转移函数
$$
\delta(q_0,a)=(q_1,d,R)
$$
执行前后的情况。

![Screenshot from 2020-09-14 13-49-34.png](https://i.loli.net/2020/09/14/lvp2HG5mkziuFEV.png)



@考虑图灵机
$$
Q=\{q_0,q_1\}\\
\Sigma=\{a,b\}\\
\Gamma=\{a,b,\square\}\\
F=\{q_1\}\\
\delta(q_0,a)=\{q_0,b,R\}\\
\delta(q_0,b)=\{q_0,b,R\}\\
\delta(q_0,\square)=\{q_1,\square,L\}\\
$$
如图所示，如果该图灵机处于初态$$q_0$$并且读写头指向符号$$a$$，则读写头将$$a$$写为$$b$$，然后向右移动；如果指向符号$$b$$，则直接向右移动；如果指向空白符，则向左移动一个单元，并状态转移为终态$$q_1$$。

![Screenshot from 2020-09-14 13-55-23.png](https://i.loli.net/2020/09/14/UNDJT7sKCqVQwYn.png)



@考虑图灵机的状态转移函数
$$
\delta(q_0,a)=\{q_1,a,R\}\\
\delta(q_0,b)=\{q_1,b,R\}\\
\delta(q_0,\square)=\{q_1,\square,R\}\\
\delta(q_1,a)=\{q_0,a,L\}\\
\delta(q_1,b)=\{q_0,b,L\}\\
\delta(q_1,\square)=\{q_0,\square,L\}\\
$$
该图灵机会永远不停地工作，其读写头反复地左右移动，我们称其进入了**无穷循环（infinite loop）**。



**定义** **标准图灵机（standard Turing machine）**是具有以下特性的图灵机：

1. 图灵机的带在左右方向上都没有限制，并且允许任意数目的左移和右移
2. 对于每一个格局，$$\delta$$最多之定义了一种迁移，因而它是确定型的
3. 没有特定的输入文件和输出设备，带的部分内容即是图灵机的输入和输出



我们使用瞬时描述展示图灵机的格局，下图的格局的瞬时描述为
$$
a_1a_2\cdots a_{k-1}qa_{k}a_{k+1}\cdots a_n
$$
![Screenshot from 2020-09-14 14-06-29.png](https://i.loli.net/2020/09/14/bkmLno6FItsfTZS.png)



**定义** 图灵机$$M=(Q,\Sigma,\Gamma,\delta,q_0,\square,F)$$，$$a_1a_2\cdots a_{k-1}q_1a_{k}a_{k+1}\cdots a_n$$是$$M$$的一个瞬时描述，其中$$a_i\in \Gamma,q_i\in Q$$。那么
$$
a_1a_2\cdots a_{k-1}q_1a_{k}a_{k+1}\cdots a_n\vdash a_1a_2\cdots a_{k-1}bq_2a_{k+1}\cdots a_n \iff \delta(q_i,a_k)=(q_2,b,R)\\
a_1a_2\cdots a_{k-1}q_1a_{k}a_{k+1}\cdots a_n\vdash a_1a_2\cdots q_2a_{k-1}ba_{k+1}\cdots a_n \iff \delta(q_i,a_k)=(q_2,b,L)\\
$$
针对某一初始格局$$x_1q_ix_2$$，如果
$$
x_1q_ix_2\vdash^* y_1q_jay_2
$$
由于$$\delta(q_j,a)$$没有定义，则$$M$$将停机。我们称导致停机状态的格局序列为**计算（computation）**。

图灵机可能永不停机，表示为
$$
x_1qx_2\vdash^*\infty
$$



## 作为语言接受器的图灵机

以下讨论中，图灵机视作语言接受器，符号串$$w$$记录在带上，其它位置为空白符，初始情况下图灵机的状态为$$q_0$$，读写头位于$$w$$的最左符号上。如果经过一系列的移动，图灵机最终进入终态并停机，则认为$$w$$能够被接受。

**定义** 图灵机$$M=(Q,\Sigma,\Gamma,\delta,q_0,\square,F)$$，则$$M$$接收的语言为
$$
L(M)=\{w\in \Sigma^+:\exist q_f\in F,x_1,x_2\in \Gamma^*,q_0w\vdash^*x_1q_fx_2\}
$$


@基于$$\Sigma = \{0,1\}$$，设计一个图灵机，它能够接受正则表达式00*表示的语言。

设置内部状态$$Q=\{q_0,q_1\}$$，终态$$F=\{q_1\}$$，转移函数为
$$
\delta(q_0,0)=(q_0,0,R)\\
\delta(q_0,\square)=(q_1,\square,R)
$$
一旦读到1，由于$$\delta(q_0,1)$$没有定义，图灵机将在非终态$$q_0$$下停机。



@基于$$\Sigma=\{a,b\}$$，设计一个图灵机，它能够接受
$$
L=\{a^nb^n:n\ge 1\}
$$
方案为
$$
Q=\{q_0,q_1,q_2,q_3,q_4\}\\
F=\{q_4\}\\
\Sigma=\{a,b\}\\
\Gamma=\{a,b,x,y,\square\}\\
\delta(q_0,a)=(q_1,x,R)\\
\delta(q_1,a)=(q_1,a,R)\\
\delta(q_1,y)=(q_1,y,R)\\
\delta(q_1,b)=(q_2,y,L)\\
\delta(q_2,y)=(q_2,y,L)\\
\delta(q_2,a)=(q_2,a,L)\\
\delta(q_2,x)=(q_0,x,R)\\
\delta(q_0,y)=(q_3,y,R)\\
\delta(q_3,y)=(q_3,y,R)\\
\delta(q_3,\square)=(q_4,\square,R)\\
$$
每一次执行，图灵机将第一个$$a$$与第一个$$b$$匹配
$$
q_0aabb\vdash xq_1abb\vdash xaq_1bb\vdash xq_2ayb\\
\vdash q_2xayb\vdash xq_0ayb \vdash xxq_1yb\\
\vdash xxyq_1b\vdash xxq_2yy \vdash xq_2xyy\\
\vdash xxq_0yy\vdash xxyq_3y \vdash xxyyq_3\\
\vdash xxyy\square q_4\square
$$



……




# 其它形式的图灵机