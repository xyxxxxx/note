
# 常用符号

$${\pmb A}$$ $${\pmb B}$$ $${\pmb C}$$ 给定矩阵

$${\pmb P}$$ 可逆矩阵

$${\pmb Q}$$ 正交矩阵

$${\pmb R}$$ 三角矩阵

齐次线性方程组
$$
\left\{ 
\begin{array}{l}
a_{11}x_1+a_{12}x_2+⋯+a_{1n}x_n=0 \\ 
a_{21}x_1+a_{22}x_2+⋯+a_{2n}x_n=0 \\ 
⋯⋯\\ 
a_{n1}x_1+a_{n2}x_2+⋯+a_{nn}x_n=0
\end{array}
\right.
$$
非齐次线性方程组
$$
\left\{ 
\begin{array}{l}
a_{11}x_1+a_{12}x_2+⋯+a_{1n}x_n=b_1 \\ 
a_{21}x_1+a_{22}x_2+⋯+a_{2n}x_n=b_2 \\ 
⋯⋯\\ 
a_{n1}x_1+a_{n2}x_2+⋯+a_{nn}x_n=b_n
\end{array}
\right.
$$
其中$$b_i$$不全为零。





# 行列式

## 二阶与三阶行列式

$$
\begin{vmatrix}
a_{11}&a_{12}\\\
a_{21}&a_{22}
\end{vmatrix}
=a_{11}a_{22}-a_{12}a_{21}
$$

$$
\begin{vmatrix}
a_{11}&a_{12}&a_{13}\\\
a_{21}&a_{22}&a_{23}\\\
a_{31}&a_{32}&a_{33}
\end{vmatrix}
=a_{11}a_{22}a_{33}+a_{12}a_{23}a_{31}+a_{13}a_{21}a_{32}-a_{11}a_{23}a_{32}-a_{12}a_{21}a_{33}-a_{13}a_{22}a_{31}
$$



## 逆序数与对换

对于n个自然数元素组成的排列$$p_1p_2⋯p_n$$，令排在$$p_i$$之前且大于其的元素有$$t_i$$个，则排列的逆序数为
$$
t=t_1+t_2+⋯+t_n=\sum_{t=1}^nt_i
$$
逆序数为奇数的排列为奇排列，逆序数为偶数的排列为偶排列。

**定理1 一个排列中任意两个元素对换，排列改变奇偶性。**



## n阶行列式

$$
\begin{vmatrix}
a_{11}&a_{12}&⋯&a_{1n}\\\
a_{21}&a_{22}&⋯&a_{2n}\\\
⋮&⋮&&⋮\\\
a_{n1}&a_{n2}&⋯&a_{nn}
\end{vmatrix}
=\mathrm{det}(a_{ij})
=\sum(-1)^ta_{1p_1}a_{2p_2}⋯a_{np_n}
$$



## 行列式的性质

$$D^T$$为$$D$$的转置行列式。

**性质1 $$D^T=D$$** 

**性质2 互换行列式的2行/列，行列式变号。**

**性质3 行列式某行/列所有元素乘以$$k$$，等于行列式乘以$$k$$。**

**性质4** 
$$
\begin{vmatrix}
a_{11}&a_{12}&⋯&(a_{1i}+a_{1i}')&⋯&a_{1n}\\\
a_{21}&a_{22}&⋯&(a_{2i}+a_{2i}')&⋯&a_{2n}\\\
⋮&⋮&&⋮&&⋮\\\
a_{n1}&a_{n2}&⋯&(a_{ni}+a_{ni}')&⋯&a_{nn}
\end{vmatrix}
=
\begin{vmatrix}
a_{11}&a_{12}&⋯&a_{1i}&⋯&a_{1n}\\\
a_{21}&a_{22}&⋯&a_{2i}&⋯&a_{2n}\\\
⋮&⋮&&⋮&&⋮\\\
a_{n1}&a_{n2}&⋯&a_{ni}&⋯&a_{nn}
\end{vmatrix}
+
\begin{vmatrix}
a_{11}&a_{12}&⋯&a_{1i}'&⋯&a_{1n}\\\
a_{21}&a_{22}&⋯&a_{2i}'&⋯&a_{2n}\\\
⋮&⋮&&⋮&&⋮\\\
a_{n1}&a_{n2}&⋯&a_{ni}'&⋯&a_{nn}
\end{vmatrix}
$$
一般计算方法：利用性质4将行列式化为上/下三角行列式，再做一次乘法。

**性质5 分块行列式**
$$
\begin{vmatrix}
A&\\\
C&B
\end{vmatrix}
=\begin{vmatrix}A\end{vmatrix}\begin{vmatrix}B\end{vmatrix}
$$




## 行列式的展开

$$n$$阶行列式中去掉第$$i$$行和第$$j$$列后的$$n-1$$阶行列式为$$a_{ij}$$的**余子式**$$M_{ij}$$，**代数余子式**$$A_{ij}=(-1)^{i+j}M_{ij}$$

**定理3 行列式等于其任一行/列的各元素与其对应的代数余子式乘积之和，即**
$$
D=a_{i1}A_{i1}+a_{i2}A_{i2}+⋯+a_{in}A_{in}
$$
**克拉默法则 对于线性方程组**
$$
\left\{ 
\begin{array}{l}
a_{11}x_1+a_{12}x_2+⋯+a_{1n}x_n=b_1 \\ 
a_{21}x_1+a_{22}x_2+⋯+a_{2n}x_n=b_2 \\ 
⋯⋯\\ 
a_{n1}x_1+a_{n2}x_2+⋯+a_{nn}x_n=b_n
\end{array}
\right.
$$


**如果的系数行列式即**
$$
D=\begin{vmatrix}
a_{11}&⋯&a_{1n}\\\
⋮&&⋮\\\
a_{n1}&⋯&a_{nn}
\end{vmatrix}≠0
$$
**那么方程组有唯一解**
$$
x_i=\frac{D_i}{D}
$$
**其中**
$$
D_i=\begin{vmatrix}
a_{11}&⋯&a_{1,j-1}&b_1&a_{1,j-1}&⋯&a_{1n}\\\
⋮&&⋮&⋮&⋮&&⋮\\\
a_{n1}&⋯&a_{n,j-1}&b_n&a_{n,j-1}&⋯&a_{nn}
\end{vmatrix}
$$
**定理4 如果线性方程组的系数行列式$$D≠0$$，则具有唯一解；如果$$D=0$$则无解或有复数解。**





# 矩阵

由$$m×n$$个数$$a_{ij}$$排成的$$m$$行$$n$$列的数表称为$$m$$行$$n$$列矩阵，简称$$m×n$$矩阵，记作
$$
\pmb{A}=\begin{bmatrix}
a_{11}&a_{12}&⋯&a_{1n}\\\
a_{21}&a_{22}&⋯&a_{2n}\\\
⋮&⋮&&⋮\\\
a_{n1}&a_{n2}&⋯&a_{nn}
\end{bmatrix}
$$
这$$m×n$$个数称为矩阵$$\pmb{A}$$的**元素**，简称**元**。以数$$a_{ij}$$为$$(i,j)$$元的矩阵简记作$$(a_{ij})$$或$$(a_{ij})_{m×n}$$。

行列数都等于$$n$$的矩阵称为$$n$$阶矩阵或$$n$$阶**方阵**，$$n$$阶矩阵$${\pmb A}$$也记作$${\pmb A_n}$$。

只有一行的矩阵称为行矩阵或**行向量**，只有一列的矩阵称为列矩阵或**列向量**。

两个矩阵行列数相等时，称它们是**同型矩阵**。如果$${\pmb A}$$和$${\pmb B}$$是同型矩阵且对应元素相等，则称矩阵$${\pmb A}$$和矩阵$${\pmb B}$$相等，记作$$\pmb{A=B}$$。

元素都是$$0$$的矩阵称为**零矩阵**。

$$n$$个变量$$x_1,x_2,⋯,x_n$$和$$m$$个变量$$y_1,y_2,⋯,y_m$$之间的关系式
$$
\left\{ 
\begin{array}{l}
y_1=a_{11}x_1+a_{12}x_2+⋯+a_{1n}x_n \\ 
y_2=a_{21}x_1+a_{22}x_2+⋯+a_{2n}x_n \\ 
⋯⋯\\ 
y_n=a_{n1}x_1+a_{n2}x_2+⋯+a_{nn}x_n
\end{array}
\right.
$$
表示从变量$$x_1,x_2,⋯,x_n$$到变量$$y_1,y_2,⋯,y_m$$的**线性变换**。

**恒等变换**对应$$n$$阶方阵
$$
\pmb{E}=\begin{bmatrix}
1&0&⋯&0\\\
0&1&⋯&0\\\
⋮&⋮&&⋮\\\
0&0&⋯&1
\end{bmatrix}
$$
称为$$n$$阶**单位矩阵**，简称单位阵，$$(i,j)$$元为
$$
\delta_{ij} =
        \begin{cases}
        1,  & \text{if $i=j$} \\
        0, & \text{if $i≠j$}
        \end{cases}
$$
$$n$$阶方阵
$$
\pmb{\Lambda}=\begin{bmatrix}
\lambda_1&0&⋯&0\\\
0&\lambda_2&⋯&0\\\
⋮&⋮&&⋮\\\
0&0&⋯&\lambda_n
\end{bmatrix}
$$
称为**对角矩阵**，也记作$${\pmb \Lambda}={\rm diag}(\lambda_1,\lambda_2,⋯,\lambda_n)$$。

矩阵
$$
\begin{bmatrix}
{\rm cos}\varphi&-{\rm sin}\varphi\\\
{\rm sin}\varphi&{\rm cos}\varphi
\end{bmatrix}
$$
对应把向量$$\vec{OP}$$逆时针旋转$$\varphi$$角的旋转变换。



## 矩阵运算

设有两个$$m×n$$矩阵$${\pmb A}$$和$${\pmb B}$$，那么矩阵$${\pmb A}$$与$$\pmb{B}$$的和记作$$\pmb{ A+B}$$，规定为
$$
\pmb{A+B}=\begin{bmatrix}
a_{11}+b_{11}&a_{12}+b_{12}&⋯&a_{1n}+b_{1n}\\\
a_{21}+b_{21}&a_{22}+b_{22}&⋯&a_{2n}+b_{2n}\\\
⋮&⋮&&⋮\\\
a_{m1}+b_{m1}&a_{m2}+b_{m2}&⋯&a_{mn}+b_{mn}
\end{bmatrix}
$$
矩阵加法满足交换律和结合律。记$$-{\pmb A}=(-a_{ij})$$。

数$$\lambda$$与矩阵$$\pmb{A}$$的乘积记作$$\lambda{\pmb A}$$或$${\pmb A}\lambda$$，规定为
$$
\lambda{\pmb A}={\pmb A}\lambda=
\begin{bmatrix}
\lambda a_{11}&\lambda a_{12}&⋯&\lambda a_{1n}\\\
\lambda a_{21}&\lambda a_{22}&⋯&\lambda a_{2n}\\\
⋮&⋮&&⋮\\\
\lambda a_{m1}&\lambda a_{m2}&⋯&\lambda a_{mn}
\end{bmatrix}
$$
数乘矩阵满足结合律和分配律。

设$$\pmb{A}=(a_{ij})$$是$$m×s$$矩阵，$$\pmb{B}=(b_{ij})$$是$$s×n$$矩阵，那么矩阵$${\bf A}$$和$${\bf B}$$的乘积是$$m×n$$矩阵$$\pmb{C}=(c_{ij})$$，其中
$$
c_{ij}=\sum_{k=1}^{s}{a_{ik}b_{kj}}
$$
记作$$\pmb{C=AB}$$。

矩阵乘法不满足交换律（一般$$\pmb{AB≠BA}$$），满足结合律和分配律。如果$$\pmb{AB=BA}$$，则称$$\pmb{A}$$与$$\pmb{B}$$可交换。

对于单位矩阵$$\pmb{E}$$，有
$$
\pmb{EA=AE=A}
$$
设$$\pmb{A}$$为$$n$$阶方阵，定义矩阵的幂

$$\pmb{A^{k+1}=A^kA}$$

转置矩阵记作$${\pmb A^ \rm T}$$，满足
$$
{({\pmb{AB}})^{\rm T}=
\pmb{B}^{\rm T}\pmb{A}^{\rm T}}
$$
由$$n$$阶方阵$$\pmb{A}$$的元素构成的行列式称为方阵$$\pmb{A}$$的行列式，记作$$\vert \pmb{A}\vert$$或$${\rm det}{\pmb A}$$。满足以下运算规律：
$$
\vert \pmb{A} ^{\rm T} \vert=\vert \pmb{A}\vert\\
\vert \lambda \pmb{A}\vert=\lambda^n\vert \pmb{A}\vert\\
\vert \pmb{AB}\vert=\vert \pmb{A}\vert\vert \pmb{B}\vert
$$
行列式$$\vert \pmb{A}\vert$$的各个元素的代数余子式$$A_{ij}$$所构成的如下矩阵
$$
\pmb{A}^*=\begin{bmatrix}
A_{11}&A_{12}&⋯&A_{1n}\\\
A_{21}&A_{22}&⋯&A_{2n}\\\
⋮&⋮&&⋮\\\
A_{n1}&A_{n2}&⋯&A_{nn}
\end{bmatrix}
$$
称为矩阵$$\pmb{A}$$的伴随矩阵，有
$$
\pmb{AA}^*=
\pmb{A}^*\pmb{A}=
\vert \pmb{A}\vert \pmb{E}
$$



## 逆矩阵

对于$$n$$阶矩阵$$\pmb{A}$$，如果有一个$$n$$阶矩阵$$\pmb{B}$$使
$$
\pmb{AB}=
\pmb{E}\; or \;
\pmb{BA}=\pmb{E}
$$
则称矩阵$$\pmb{A}$$可逆，矩阵$$\pmb{A}$$和矩阵$$\pmb{B}$$互为逆矩阵，记作$$\pmb{B=A^{-1}}$$。

**定理 矩阵$$\pmb{A}$$可逆$$\iff$$$$\vert \pmb{A}\vert \neq 0$$，且**
$$
{\pmb A}^{-1}=\frac{1}{\vert {\pmb A} \vert}{\pmb A}^*
$$
$$\vert \pmb{A}\vert=0$$时，称$$\pmb{A}$$为**奇异矩阵**，否则称为**非奇异矩阵**（即为可逆矩阵）。

逆矩阵满足以下运算规律
$$
({\pmb A \pmb B})^{-1}=\pmb B^{-1}\pmb A^{-1}
$$



## 分块矩阵

如果矩阵$$\pmb{A}$$和$$\pmb{B}$$行列数相同且有相同的分块法，则分块加法和乘法的方法与不分块时一致。





# 矩阵的初等变换

## 初等变换

以下变换称为矩阵的**初等行变换**

- 对调两行
- 以非零数乘某行所有元素
- 把某行元素的$$k$$倍加到另一行

**初等列变换**同理，统称为**初等变换**。

如果矩阵$$\pmb{A}$$经有限次初等（行/列）变换变成矩阵$$\pmb{B}$$，则称$$\pmb{A}$$和$$\pmb{B}$$（行/列）等价，记作$$\pmb{A} \sim \pmb{B}$$。等价关系具有反身性，对称性和传递性。

对于$$m ×n$$矩阵$$\pmb{A}$$，总可经过初等变换将其化为**标准形**
$$
\pmb{F}=\begin{bmatrix}
{\pmb E}_r&{\pmb O}\\\
{\pmb O}&{\pmb O}
\end{bmatrix}_{m×n}
$$
**定理 $$\pmb{A} \sim \pmb{B}$$的充要条件是存在$$m$$阶可逆矩阵$$\pmb{P}$$和$$n$$阶可逆矩阵$$\pmb{Q}$$使$$\pmb{PAQ=B}$$。**

单位阵$$\pmb{E}$$经过一次初等变换得到的矩阵称为初等矩阵。对于$$m ×n$$矩阵$$\pmb{A}$$，左乘$$m$$阶初等矩阵相当于进行初等行变换，右乘$$n$$阶初等矩阵相当于进行初等列变换。

方阵$${\pmb A}$$可逆$$\iff {\pmb A=P_1P_2 \cdots P_i}$$（$$\pmb{P}_i$$为初等矩阵）$$\iff \pmb{A} \sim^r \pmb{E}$$



## 秩

矩阵$$\pmb{A}$$的标准形$$\pmb{F}$$中$$r$$即为其秩。

$$m ×n$$矩阵$$\pmb{A}$$中任取$$k$$行$$k$$列，位于行列交叉处的$$k^2$$的元素组成的$$k$$阶行列式称为矩阵$$\pmb{A}$$的$$k$$阶子式。

设矩阵$$\pmb{A}$$有一不等于$$0$$的子式$$D$$，且所有$$r+1$$阶子式全等于$$0$$，则$$D$$为矩阵$$\pmb{A}$$的最高阶非零子式，$$r$$称为矩阵的秩，记作$$R(\pmb{A})$$。可逆矩阵为满秩矩阵，不可逆矩阵为降秩矩阵。

**性质 **

- $$0 \le R(\pmb{A}_{m×n}) \le {\rm min} \{ m,n\}$$
- $$R({\pmb A}^{\rm T})=R({\pmb A})$$
- **若$$\pmb{A} \sim \pmb{B}$$，则$$R({\pmb A})=R(\pmb B)$$。**
- $$\pmb{P},\pmb{Q}$$可逆，$$R(\pmb{PAQ})=R({\pmb A})$$
- $${\rm max}\{R({\pmb A}),R({\pmb B})\} \le R(\pmb{ A,B}) \le R({\pmb A})+R({\pmb B})$$
- $$R(\pmb{ A+B}) \le R({\pmb A})+R({\pmb B})$$ 
- $$R(\pmb{ AB}) \le {\rm min}\{R({\pmb A}),R({\pmb B})\} $$
- 若$${\pmb A_{m×n}\pmb B_{n×i}=\pmb O}$$，则$$R({\pmb A})+R({\pmb B}) \le n$$



## 线性方程组的解

**定理 $$n$$元线性方程组$$\pmb{Ax=b}$$**

1. **无解的充要条件是$$R(\pmb{ A}) < R({\pmb A,\pmb b})$$**
2. **有唯一解的充要条件是$$R(\pmb{ A}) = R({\pmb A,\pmb b}) = n$$**
3. **有无穷解的充要条件是$$R(\pmb{ A}) = R({\pmb A,\pmb b}) < n$$**

求解线性方程组的步骤：

1. 对于非齐次线性方程组，将其增广矩阵化为行阶梯形，若此时$$R(\pmb{A})<R(\pmb{B})$$，则无解
2. 若$$R(\pmb{A})=R(\pmb{B})$$，进一步把$$\pmb{B}$$化为最简形
3. 若$$R(\pmb{A})=R(\pmb{B})=r$$，写出含$$n-r$$个参数的通解

定理 矩阵方程$$\pmb{AX=B}$$有解的充要条件是$$R(\pmb{ A}) = R({\pmb A,\pmb B})$$





# 向量空间

## 向量组及线性组合

$$n$$个有次序的数$$\pmb{a_1,a_2,\cdots,a_n}$$组成的数组称为$$n$$维向量，这$$n$$个数称为该向量的分量。若干个同维的列向量组成的集合称为向量组。

对于给定的向量组$$\pmb{A:a_1,a_2,\cdots,a_n}$$，对于任何一组实数$$k_1,k_2,\cdots,k_m$$，$$k_1{\pmb a}_1+k_2{\pmb a}_2+\cdots+k_m{\pmb a}_m$$称为向量组$$\pmb{A}$$的一个**线性组合**。给定向量组$$\pmb{A}$$和向量$$\pmb{b}$$，如果存在一组数$$\lambda_1,\lambda_2,\cdots,\lambda_m$$，使$${\pmb b}=k_1{\pmb a}_1+k_2{\pmb a}_2+\cdots+k_m{\pmb a}_m$$，则称$${\pmb b}$$能由向量组$$\pmb{A}$$线性表示，亦即方程组$$x_1{\pmb a}_1+x_2{\pmb a}_2+\cdots+x_m{\pmb a}_m=\pmb{b}$$有解。

**定理 向量$$\pmb{b}$$能被向量组$$\pmb{A:a_1,a_2,\cdots,a_n}$$线性表示的充要条件是矩阵$${\pmb A}=({\pmb a}_1,{\pmb a}_2,\cdots,{\pmb a}_m)$$的秩等于矩阵$${\pmb B}=({\pmb a}_1,{\pmb a}_2,\cdots,{\pmb a}_m,{\pmb b})$$的秩。**

设有向量组$$A:\pmb{a_1,a_2,\cdots,a_n}$$和向量组$$B:\pmb{b_1,b_2,\cdots,b_n}$$，若B组每个向量都能由A组线性表示，则称向量组B能由向量组A线性表示。若向量组A和向量组B能互相线性表示，则称这两个向量组等价。

**定理 向量组$$\pmb{B}$$能被向量组$$\pmb{A}$$线性表示的充要条件是$$R({\pmb A})=R({\pmb A},{\pmb B})$$**。

向量组$$\pmb{B}$$能被向量组$$\pmb{A}$$线性表示$$\iff$$存在矩阵$${\pmb K}$$，使$$\pmb{ B=AK} \iff$$方程$$\pmb{AX=B}$$有解



## 向量组的线性相关

对于给定的向量组$$\pmb{A:a_1,a_2,\cdots,a_m}$$，如果存在不全为零的数$$k_1,k_2,\cdots,k_m$$使$$k_1{\pmb a}_1+k_2{\pmb a}_2+\cdots+k_m{\pmb a}_m=\bf 0$$，则称向量组$$\pmb{A}$$是**线性相关**的，否则称其**线性无关**。

**定理 向量组$$\pmb{a_1,a_2,\cdots,a_m}$$线性相关的充要条件是它所构成的矩阵$${\pmb A}$$的秩小于$${\pmb m}$$；线性无关的充要条件是$$R({\pmb A})=m$$。**



## 向量组的秩

对于给定的向量组$$\pmb{A:a_1,a_2,\cdots,a_m}$$，如果能选出$$r$$个向量$$\pmb{a_1,a_2,\cdots,a_r}$$，满足

1. 向量组$$\pmb{A_0:a_1,a_2,\cdots,a_r}$$线性无关
2. 向量组$$\pmb{A}$$中任意$$r+1$$个向量都线性相关

则称向量组$${\pmb A_0}$$是向量组$${\pmb A}$$的**最大线性无关向量组**，$$r$$为向量组$${\pmb A}$$的秩，记作$$R_A$$。

**定理 矩阵的秩等于其列向量组的秩，也等于其行向量组的秩。**



## 线性方程组的解

**定理 设$$m ×n$$矩阵$$\pmb{A}$$的秩$$R({\pmb A})=r$$，则n元齐次线性方程组$$\pmb{Ax=0}$$的解集$$S$$的秩$$R_s=n-r$$。**

n元非齐次线性方程组$$\pmb{Ax=b}$$的解表示为$$\pmb{Ax=b}$$的1个特解与$$\pmb{Ax=0}$$的通解。



## 向量空间

设$$V$$为$$n$$维向量的集合，如果集合$$V$$非空，且对于向量的加法及数乘两种运算封闭，则称为**向量空间**。

设$$V$$为向量空间，如果$$r$$个向量$$\pmb{a_1,a_2,\cdots,a_r} \in V$$，且满足

1. $$\pmb{a_1,a_2,\cdots,a_r}$$线性相关
2. $$V$$中任一向量都可由$$\pmb{a_1,a_2,\cdots,a_r}$$线性表示

则称向量组$$\pmb{a_1,a_2,\cdots,a_r}$$为向量空间的$$V$$的一个**基（base）**，$$r$$称为向量空间$$V$$的维数，称其为$$r$$维向量空间。

如果再向量空间$$V$$中取定一个基$$\pmb{a_1,a_2,\cdots,a_r}$$，那么$$V$$中任一向量$${\pmb x}$$可唯一表示为
$$
\pmb x=\lambda_1 \pmb a_1 +\lambda_2 \pmb a_2+\cdots+\lambda_r \pmb a_r
$$
数组$$\lambda_1,\lambda_2,\cdots,\lambda_r$$称为向量$$\pmb x$$在基$$\pmb{a_1,a_2,\cdots,a_r}$$上的**坐标（coordinate）**。

$$\pmb{e_1,e_2,\cdots,e_n}$$称为$$\Bbb R^n$$中的**自然基**或**标准基**，向量在自然基上的坐标称为**笛卡尔坐标（Cartesian coordinate）**。



## 范数

**范数(Norm)**将向量空间内的所有向量映射到非负实数。对于一个$$N$$维向量$$\boldsymbol v$$ ,一个常见的范数函数为$$\mathcal{l}_p$$范数
$$
\mathcal{l}_p(\boldsymbol v)=||\boldsymbol v||_p=(\sum_{n=1}^N|v_n|^p)^{1/p}
$$
其中$$p\ge 0$$为一个标量的参数，常用的$$p$$的取值有1，2，∞等。
$$
\mathcal{l}_1(\boldsymbol v)=||\boldsymbol v||_1=\sum_{n=1}^N|v_n|\\
\mathcal{l}_2(\boldsymbol v)=||\boldsymbol v||_2=||\boldsymbol v||=\sqrt{\sum_{n=1}^Nv_n^2}
$$
$$\mathcal{l}_2$$范数又称为Euclidean范数或者Frobenius范数。从几何角度，向量也可以表示为从原点出发的一个带箭头的有向线段，其$$\mathcal{l}_2$$范数为线段的长度，也常称为向量的**模**。
$$
‖\boldsymbol v‖_∞ = \max\{v_1, v_2, ⋯ , v_N \}
$$
下图给出了常见范数的示例，其中红线表示不同范数的$$ \mathcal{l}_p = 1 $$的点

![Screenshot from 2020-09-01 19-02-39.png](https://i.loli.net/2020/09/02/2VMxbcDkFCmasPn.png)





# 矩阵相似与二次型

## 向量的内积、长度与正交

$$[\pmb x,\pmb y]=\pmb x^T \pmb y=x_1y_1+x_2y_2+\cdots+x_ny_n$$称为向量$${\pmb x}$$与$${\pmb y}$$的**内积**。当$$[\pmb x,\pmb y]=0$$时，称向量$${\pmb x}$$与$${\pmb y}$$**正交**。

$$\Vert x \Vert=\sqrt{x_1^2+x_2^2+\cdots+x_n^2}$$称为向量$${\pmb x}$$的**长度**（**$$\mathcal{l}_2$$范数**）。

**定理 若$$n$$维向量$$\pmb{a_1,a_2,\cdots,a_r}$$是一组两两正交的非零向量，则$$\pmb{a_1,a_2,\cdots,a_r}$$线性无关。**

设$$n$$维向量$$\pmb{e_1,e_2,\cdots,e_n}$$是向量空间$$V(V \subset \Bbb R^n)$$的一个基，如果$$\pmb{e_1,e_2,\cdots,e_n}$$两两正交且都是单位向量，则称其是$$V$$的一个**规范正交基**。

施密特正交化：$$\pmb{a_1,a_2,\cdots,a_r}$$是向量空间$$V$$的一个基，对其规范正交化，取
$$
\begin{align}
\pmb b_1=&\pmb a_1\\
\pmb b_2=&\pmb a_2-\frac{[\pmb b_1,\pmb a_2]}{[\pmb b_1 , \pmb b_1]}\pmb b_1 \\
\cdots&\cdots\\
\pmb b_r=&\pmb a_r-\frac{[\pmb b_1,\pmb a_r]}{[\pmb b_1 , \pmb b_1]}\pmb b_1-\cdots-\frac{[\pmb b_{r-1},\pmb a_r]}{[\pmb b_{r-1} , \pmb b_{r-1}]}\pmb b_{r-1}
\end{align}
$$
再将他们单位化即取$$\pmb e_i=\frac{1}{\Vert \pmb b_i \Vert} \pmb b_i$$，就是$$V$$的一个规范正交基。

如果n阶矩阵$$\pmb Q$$满足
$$
\pmb Q^ {\rm T} \pmb Q=\pmb E \quad {\rm i.e.} \quad \pmb Q^{-1}= \pmb Q^ {\rm T}
$$
则称$${\pmb Q}$$为**正交矩阵**。方阵$${\pmb Q}$$为正交矩阵的充要条件是$${\pmb Q}$$的列向量都是单位向量，且两两正交。

若$${\pmb Q}$$为正交矩阵，则线性变换$$\pmb{ y=Qx}$$称为正交变换。正交变换不改变向量的长度。



## 方阵的特征值与特征向量

对于n阶矩阵$${\pmb A}$$，如果数$$\lambda$$和n维非零列向量$${\pmb x}$$使式
$$
\pmb A \pmb x = \lambda \pmb x
$$
成立，则$$\lambda$$称为矩阵$${\pmb A}$$的**特征值**，$${\pmb x}$$称为$${\pmb A}$$对应于特征值$$\lambda$$的**特征向量**。
$$
\pmb A \pmb x = \lambda \pmb x \iff (\pmb A -\lambda \pmb E )\pmb x= \pmb 0
$$
该其次线性方程组有非零解的充要条件是系数行列式
$$
\vert \pmb A -\lambda \pmb E \vert =0 \\
\rm{i.e.}\;\begin{vmatrix}
a_{11}-\lambda &a_{12}&\cdots&a_{1n}\\
a_{21} &a_{22}-\lambda&\cdots&a_{2n}\\
\vdots &\vdots&&\vdots\\
a_{n1} &a_{n2}&\cdots&a_{nn}-\lambda\\
\end{vmatrix}=0
$$
上式称为矩阵$${\pmb A}$$的**特征方程**，其左端称为矩阵$${\pmb A}$$的**特征多项式**。

设矩阵$${\pmb A}$$的特征值为$$\lambda_1,\lambda_2,\cdots,\lambda_n$$，则

1. $$\lambda_1+\lambda_2+\cdots+\lambda_n=a_{11}+a_{22}+\cdots+a_{nn}$$
2. $$\lambda_1\lambda_2\cdots\lambda_n=\vert \pmb A \vert$$

定理 设$$\lambda_1,\lambda_2,\cdots,\lambda_m$$是方阵$${\pmb A}$$的m个特征值，$$\pmb{p_1,p_2,\cdots,p_m}$$是对应的特征向量。如果$$\lambda_1,\lambda_2,\cdots,\lambda_m$$各不相等，则$$\pmb{p_1,p_2,\cdots,p_m}$$线性无关。



## 相似矩阵

设$${\pmb A, \pmb B}$$都是n阶矩阵，如果有可逆矩阵$${\pmb P}$$，使
$$
\pmb{P^{-1}AP=B}
$$
则称$${\pmb B}$$是$${\pmb A}$$的**相似矩阵**。

若n阶矩阵$${\pmb A}$$与$${\pmb B}$$相似，则它们的特征多项式相同，因而特征值相同。

**定理 n阶矩阵$${\pmb A}$$与对角阵相似的充要条件是$${\pmb A}$$有n个线性无关的特征向量。**

n阶矩阵$${\pmb A}$$的n个特征值互不相等，则$${\pmb A}$$与对角阵相似。如果特征方程有重根，则不一定。



## 对称矩阵的对角化

**定理 对称矩阵的特征值为实数。**

**定理 设$$\lambda_1,\lambda_2$$是n阶对称矩阵$${\pmb A}$$的两个相异的特征值，$$\pmb{p_1,p_2}$$是对应的特征向量，则$$\pmb{p_1,p_2}$$正交。**

**定理 对于n阶对称矩阵$${\pmb A}$$，必有正交阵$${\pmb Q}$$使得$$\pmb{Q^{-1}AQ=\Lambda}$$，其中$${\pmb \Lambda}$$是n个特征值为对角元的对角阵。**

对角化步骤：……

>求对角矩阵的n次方



## 二次型

含有n个变量$$x_1,x_2,\cdots,x_n$$的二次齐次函数$$f=\sum_{i,j=1}^na_{ij}x_ix_j$$称为**二次型**。只含平方项的二次型称为二次型的**标准型**。

二次型可记作$$f=\pmb x^ {\rm T} \pmb A \pmb x $$，其中对称阵$${\pmb A}$$称为二次型f的矩阵，f称为$${\pmb A}$$的二次型，$${\pmb A}$$的秩即为二次型f的秩。

设$${\pmb A, \pmb B}$$都是n阶矩阵，如果有可逆矩阵$${\pmb P}$$，使
$$
\pmb P^{\rm T}\pmb A\pmb P=\pmb B
$$
则称$${\pmb A}$$与$${\pmb B}$$**合同**。

定理 任给二次型$$f=\sum_{i,j=1}^na_{ij}x_ix_j$$，总有正交变换$$\pmb{x=Qy}$$使f化为标准型
$$
f=\lambda_1 y_1^2+\lambda_2 y_2^2+\cdots+\lambda_n y_n^2
$$
其中$$\lambda_1,\lambda_2,\cdots,\lambda_n$$是矩阵$${\pmb A}=(a_{ij})$$的特征值。



### 配方法

> 步骤：$$f=(ax_1+bx_2+cx_3)^2+(dx_2+ex_3)^2+fx_3^2$$
>
> 或换元法



### 正定二次型

**惯性定理 设二次型$$f={\pmb x^ {\rm T}\pmb A \pmb x}$$的秩为r，有两个可逆变换$$\pmb{x=P_1y}$$和$$\pmb{x=P_2z}$$使**
$$
f=k_1 y_1^2+k_2 y_2^2+\cdots+k_r y_n^2(k_i \neq 0)\\
f=\lambda_1 z_1^2+\lambda_2 z_2^2+\cdots+\lambda_n z_n^2(\lambda_i \neq0)
$$
**则$$k_i$$和$$\lambda_i$$中正数个数相等。正系数的个数称为正惯性指数，负系数的个数称为负惯性指数。**

设二次型$$f={\pmb x^ {\rm T}\pmb A \pmb x}$$，如果对任何$$\pmb {x \neq 0}$$都有$$f(\pmb x)>0$$，则称f为**正定二次型**，并称$$\pmb A$$是**正定矩阵**。

**定理 n元二次型$$f={\pmb x^ {\rm T}\pmb A \pmb x}$$正定的充要条件是它的标准形的n个系数全为正，即正惯性指数等于n。**

对称阵$${\pmb A}$$正定的充要条件是$${\pmb A}$$的特征值全为正。

**定理 对称阵$${\pmb A}$$正定的充要条件是$${\pmb A}$$的各阶主子式都为正；对称阵$${\pmb A}$$负定的充要条件是$${\pmb A}$$的奇数阶主子式为负，偶数阶主子式为正。**

> 实对称矩阵$${\pmb A}$$正定$$\iff$$$${\pmb P^{\rm T}\pmb  A\pmb P}$$正定$$\iff$$特征值全正$$\iff$$$$p=n \iff$$$${\pmb A}$$与$${\pmb E}$$合同$$\iff$$$${\pmb A=\pmb P^{\rm T} \pmb P}$$



# 线性空间与线性变换





# 矩阵微积分

**矩阵微积分（Matrix Calculus）**是多元微积分的一种表达方式，即使用矩阵和向量表示因变量每个成分关于自变量每个成分的偏导数。



## 偏导数

**标量关于向量的偏导数** 对于M维向量$$\boldsymbol x\in \mathbb{R}^M$$和函数$$y=f(\boldsymbol x)\in \mathbb{R}$$，$$y$$关于$$\boldsymbol x$$的偏导数为
$$
\frac{\partial y}{\partial \boldsymbol x}=[\frac{\partial y}{\partial  x_1},\cdots,\frac{\partial y}{\partial x_M}]^{\rm T}
$$
$$y$$关于$$\boldsymbol x$$的二阶偏导数为
$$
\boldsymbol H=\frac{\partial^2 y}{\partial \boldsymbol x^2}=\begin{bmatrix} \frac{\partial^2 y}{\partial x_1^2} & \cdots & \frac{\partial^2 y}{\partial  x_1 \partial x_M}\\
\vdots & \ddots & \vdots \\
\frac{\partial^2 y}{\partial x_M \partial x_1} & \cdots & \frac{\partial^2 y}{\partial  x_M^2}
\end{bmatrix}
\in \mathbb{R}^{M\times M}
$$
称为函数$$f(\boldsymbol x)$$的**Hessian矩阵**，也写作$$\nabla^2 f(\boldsymbol x)$$

**向量关于标量的偏导数** 对于标量$$x\in \mathbb{R}$$和函数$$\boldsymbol y=f(x)\in \mathbb{R}^N$$，$$\boldsymbol y$$关于$$x$$的偏导数为
$$
\frac{\partial \boldsymbol y}{\partial x}=[\frac{\partial y_1}{\partial  x},\cdots,\frac{\partial y_N}{\partial x}]
$$
**向量关于向量的偏导数** 对于M维向量$$\boldsymbol x\in \mathbb{R}^M$$和函数$$\boldsymbol y=f(\boldsymbol x)\in \mathbb{R}^N$$，$$\boldsymbol y$$关于$$\boldsymbol x$$的偏导数为
$$
\frac{\partial \boldsymbol y}{\partial \boldsymbol x}=\begin{bmatrix} \frac{\partial y_1}{\partial  x_1} & \cdots & \frac{\partial y_N}{\partial  x_1}\\
\vdots & \ddots & \vdots \\
\frac{\partial y_1}{\partial  x_M} & \cdots & \frac{\partial y_N}{\partial  x_M}
\end{bmatrix}
\in \mathbb{R}^{M\times N}
$$
称为函数$$f(\boldsymbol x)$$的**雅可比矩阵（Jacobian Matrix）**的转置。



## 导数计算法则

**加法法则**

若$$\boldsymbol x\in \mathbb{R}^M$$，$$\boldsymbol y=f(\boldsymbol x)\in \mathbb{R}^N$$，$$\boldsymbol z=g(\boldsymbol x)\in \mathbb{R}^N$$，则
$$
\frac{\partial (\boldsymbol y+ \boldsymbol z)}{\partial \boldsymbol x}=\frac{\partial \boldsymbol y}{\partial \boldsymbol x}+\frac{\partial \boldsymbol z}{\partial \boldsymbol x} \in \mathbb{R}^{M\times N}
$$
**乘法法则**

1. 若$$\boldsymbol x\in \mathbb{R}^M$$，$$\boldsymbol y=f(\boldsymbol x)\in \mathbb{R}^N$$，$$\boldsymbol z=g(\boldsymbol x)\in \mathbb{R}^N$$，则
   $$
   \frac{\partial \boldsymbol y^{\rm T} \boldsymbol z}{\partial \boldsymbol x}=\frac{\partial \boldsymbol y}{\partial \boldsymbol x}\boldsymbol z+\frac{\partial \boldsymbol z}{\partial \boldsymbol x}\boldsymbol y \in \mathbb{R}^{M}
   $$

2. 若$$\boldsymbol x\in \mathbb{R}^M$$，$$\boldsymbol y=f(\boldsymbol x)\in \mathbb{R}^S$$，$$\boldsymbol z=g(\boldsymbol x)\in \mathbb{R}^T$$，$$\boldsymbol A \in \mathbb{R}^{S\times T}$$和$$\boldsymbol x$$无关，则
   $$
   \frac{\partial \boldsymbol y^{\rm T}\boldsymbol A \boldsymbol z}{\partial \boldsymbol x}=\frac{\partial \boldsymbol y}{\partial \boldsymbol x}\boldsymbol A\boldsymbol z+\frac{\partial \boldsymbol z}{\partial \boldsymbol x}\boldsymbol A^{\rm T} \boldsymbol y \in \mathbb{R}^{M}
   $$

3. 若$$\boldsymbol x\in \mathbb{R}^M$$，$$y=f(\boldsymbol x)\in \mathbb{R}$$，$$\boldsymbol z=g(\boldsymbol x)\in \mathbb{R}^N$$，则
   $$
   \frac{\partial y \boldsymbol z}{\partial \boldsymbol x}=y\frac{\partial \boldsymbol z}{\partial \boldsymbol x}+\frac{\partial y}{\partial \boldsymbol x}\boldsymbol z^{\rm T} \in \mathbb{R}^{M\times N}
   $$

**链式法则（Chain Rule）**

1. 若$$x\in \mathbb{R}$$，$$\boldsymbol y=f(x)\in \mathbb{R}^M$$，$$\boldsymbol z=g(\boldsymbol y)\in \mathbb{R}^N$$，则
   $$
   \frac{\partial \boldsymbol z}{\partial x}=\frac{\partial \boldsymbol y}{\partial x}\frac{\partial \boldsymbol z}{\partial \boldsymbol y} \in \mathbb{R}^{1\times N}
   $$

2. 若$$\boldsymbol x\in \mathbb{R}^M$$，$$\boldsymbol y=f(\boldsymbol x)\in \mathbb{R}^K$$，$$\boldsymbol z=g(\boldsymbol y)\in \mathbb{R}^N$$，则
   $$
   \frac{\partial \boldsymbol z}{\partial \boldsymbol x}=\frac{\partial \boldsymbol y}{\partial \boldsymbol x}\frac{\partial \boldsymbol z}{\partial \boldsymbol y} \in \mathbb{R}^{M\times N}
   $$

3. 若$$\boldsymbol X\in \mathbb{R}^{M\times N}$$，$$\boldsymbol y=f(\boldsymbol X)\in \mathbb{R}^K$$，$$z=g(\boldsymbol y)\in \mathbb{R}$$，则
   $$
   \frac{\partial z}{\partial x_{ij}}=\frac{\partial \boldsymbol y}{\partial x_{ij}}\frac{\partial z}{\partial \boldsymbol y} \in \mathbb{R}
   $$



## 常见导数

$$
\frac{\partial \boldsymbol x}{\partial \boldsymbol x}=\boldsymbol I\\
\frac{\partial ||\boldsymbol x||^2}{\partial \boldsymbol x}=2\boldsymbol x\\
\frac{\partial \boldsymbol A \boldsymbol x}{\partial \boldsymbol x}=\boldsymbol A^{\rm T}\\
\frac{\partial \boldsymbol A \boldsymbol x}{\partial \boldsymbol x^{\rm T}}=\frac{\partial \boldsymbol x^{\rm T} \boldsymbol A}{\partial \boldsymbol x}=\boldsymbol A\\
\frac{\partial \boldsymbol x^{\rm T} \boldsymbol A \boldsymbol x}{\partial \boldsymbol x}=(\boldsymbol A+\boldsymbol A^{\rm T})\boldsymbol x\\
$$





# 归纳

泰勒公式 伴随矩阵/adjoint matrix $$A^*$$  正交矩阵/orthogonal matrix $$Q$$

相抵/equivalent 相似/similar 合同/congruent 可交换/commutable



## 定理

n阶方阵A可逆$$\iff \vert {\pmb A} \vert \neq0 \iff A与I相抵 \iff R({\pmb A})=n$$$$\iff$$A的n个特征值全不为零

n阶方阵A可对角化$$\iff $$ 所有特征值的几何重数达到代数重数

A正定$$\iff$$$$P^TAP$$正定 $$\iff$$ 特征值全正 $$\iff$$ $$p=n/与I合同/A=C^TC$$ $$\iff$$ **各阶顺序主子式>0** 

特征多项式有n个复根，且是A的化零多项式

相似矩阵有相同的特征值，迹，行列式

实对称阵**一定**可以通过**正交矩阵**对角化，且特征值是**实数**，且不同特征值下的特征向量**正交**

$$\vert AB \vert=\vert A \vert \vert B\vert $$

$$R({\pmb A})+R({\pmb B})-n \le R(\pmb{AB}) \le R(\pmb{A}) or R(\pmb{B}) \le R(\pmb{A,B}) \le R(\pmb{A})+R(\pmb{B}),R(\pmb{A+B}) \le R(\pmb{A})+R(\pmb{B})$$

$$tr{\pmb A}=\sum a_{ii}=\sum \lambda_i,tr(\pmb{AB})=tr(\pmb{BA}),tr(\pmb{ABC})=tr(\pmb{BCA})$$

$$\vert {\pmb A} \vert=\Pi \lambda_i$$

$$\frac{\partial Ax}{\partial x}=A^T,\frac{\partial Ax}{\partial x^T}=A,\frac{\partial x^TAx}{\partial x}=(A+A^T)x\\$$



## 解线性方程组

（克莱姆法则：$$x_j=\frac{D_j}{D}$$）A

高斯消元法

齐次线性方程组$$Ax=0$$：A可逆/$$r(A)=n$$只有零解；A不可逆/$$r(A)<n$$有非零解；

非齐次线性方程组$$Ax=b$$：$$r(A)=r(B)$$有解；$$r(A)=r(B)=n$$有唯一解；



## 求特征值和特征向量（相似对角化（求n次方））

> 注意复数域上的求法：酉空间$$(a,b)=\sum a \bar{b},\vert x \vert =\sqrt{x \bar{x}}$$
>
> 要求正交矩阵相似对角化一定要**检查正交**！



## 求逆矩阵

$$\vert A \; I \vert $$初等变换



## 求文字行列式

递推/数学归纳法



## 扩展

对称阵的合同对角化：正交矩阵的相似对角化，初等变换法

QR分解$$A=QR$$

若尔当标准型

奇异值分解$$A=U\Sigma V^T$$

最小二乘解 $$A^TAx=A^Tb$$   广义逆

酉矩阵 正规矩阵（可以酉对角化） 埃尔米特矩阵 

幂零矩阵$$A^k=0 \iff$$所有特征值为0

> 任给一个n阶方阵A，它的特征多项式是一个n阶多项式，故有n个特征值，其中相同的特征值累加重数。如果都是实根，如果总几何重数等于总代数重数/有n个线性无关的特征向量，则可以相似对角化，在此基础上如果$$AA^T=A^TA$$，则可以正交相似对角化，在此基础上包含实对称阵，其对应实二次型；如果总几何重数等于总代数重数，则可以相似到若尔当标准型。
>
> 如果有复根，如果$$AA^H=A^HA$$，则可以酉相似对角化，在此基础上包含埃尔米特矩阵，其对应埃尔米特二次型。
>