> 参考[Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf)

# 向量空间

## 向量

**标量(scalar)**是一个实数，只有大小，没有方向。标量一般用斜体小写英文字母$$a,b,c$$来表示。**向量(vector)**是由一组实数组成的有序数组，同时具有大小和方向。一个$$N$$维向量$$\pmb a$$是由$$N$$个有序实数组成，表示为
$$
\pmb a = [a_1 , a_2 , ⋯ , a_N ]
$$



## 向量运算

### 内积

$$[\pmb x,\pmb y]=\pmb x^T \pmb y=x_1y_1+x_2y_2+\cdots+x_ny_n$$称为向量$${\pmb x}$$与$${\pmb y}$$的**内积**。当$$[\pmb x,\pmb y]=0$$时，称向量$${\pmb x}$$与$${\pmb y}$$**正交**。



### 外积







### 范数

**范数(norm)**将向量空间内的所有向量映射到非负实数。对于一个$$N$$维向量$$\boldsymbol v$$ ,一个常见的范数函数为$$\ell_p$$范数
$$
\ell_p(\boldsymbol v)=||\boldsymbol v||_p=(\sum_{n=1}^N|v_n|^p)^{1/p}
$$
其中$$p\ge 0$$为一个标量的参数，常用的$$p$$的取值有1，2，∞等。
$$
\ell_1(\boldsymbol v)=||\boldsymbol v||_1=\sum_{n=1}^N|v_n|\\
\ell_2(\boldsymbol v)=||\boldsymbol v||_2=||\boldsymbol v||=\sqrt{\sum_{n=1}^Nv_n^2}
$$
$$\ell_2$$范数又称为Euclidean范数或者Frobenius范数。从几何角度，向量也可以表示为从原点出发的一个带箭头的有向线段，其$$\ell_2$$范数为线段的长度，也常称为向量的**模**。
$$
‖\boldsymbol v‖_∞ = \max\{v_1, v_2, ⋯ , v_N \}
$$
下图给出了常见范数的示例，其中红线表示不同范数的$$\ell_p = 1 $$的点

![Screenshot from 2020-09-01 19-02-39.png](https://i.loli.net/2020/09/02/2VMxbcDkFCmasPn.png)



## 向量组

若干个同维的列向量组成的集合称为向量组。

### 线性组合

对于给定的向量组$$\pmb{A:a_1,a_2,\cdots,a_n}$$，对于任何一组实数$$k_1,k_2,\cdots,k_m$$，$$k_1{\pmb a}_1+k_2{\pmb a}_2+\cdots+k_m{\pmb a}_m$$称为向量组$$\pmb{A}$$的一个**线性组合**。给定向量组$$\pmb{A}$$和向量$$\pmb{b}$$，如果存在一组数$$\lambda_1,\lambda_2,\cdots,\lambda_m$$，使$${\pmb b}=k_1{\pmb a}_1+k_2{\pmb a}_2+\cdots+k_m{\pmb a}_m$$，则称$${\pmb b}$$能由向量组$$\pmb{A}$$线性表示，亦即方程组$$x_1{\pmb a}_1+x_2{\pmb a}_2+\cdots+x_m{\pmb a}_m=\pmb{b}$$有解。

**定理** 向量$$\pmb{b}$$能被向量组$$\pmb{A:a_1,a_2,\cdots,a_n}$$线性表示的充要条件是矩阵$${\pmb A}=({\pmb a}_1,{\pmb a}_2,\cdots,{\pmb a}_m)$$的秩等于矩阵$${\pmb B}=({\pmb a}_1,{\pmb a}_2,\cdots,{\pmb a}_m,{\pmb b})$$的秩。

设有向量组$$A:\pmb{a_1,a_2,\cdots,a_n}$$和向量组$$B:\pmb{b_1,b_2,\cdots,b_n}$$，若B组每个向量都能由A组线性表示，则称向量组B能由向量组A线性表示。若向量组A和向量组B能互相线性表示，则称这两个向量组**等价**。



### 线性相关

对于给定的向量组$$\pmb{A:a_1,a_2,\cdots,a_m}$$，如果存在不全为零的数$$k_1,k_2,\cdots,k_m$$使$$k_1{\pmb a}_1+k_2{\pmb a}_2+\cdots+k_m{\pmb a}_m=\bf 0$$，则称向量组$$\pmb{A}$$是**线性相关**的，否则称其**线性无关**。

**定理** 向量组$$\pmb{a_1,a_2,\cdots,a_m}$$线性相关的充要条件是$$R(\pmb A)<m$$；线性无关的充要条件是$$R({\pmb A})=m$$。



### 秩

对于给定的向量组$$\pmb{A:a_1,a_2,\cdots,a_m}$$，如果能选出$$r$$个向量$$\pmb{a_1,a_2,\cdots,a_r}$$，满足

1. 向量组$$\pmb{A_0:a_1,a_2,\cdots,a_r}$$线性无关
2. 向量组$$\pmb{A}$$中任意$$r+1$$个向量都线性相关

则称向量组$${\pmb A_0}$$是向量组$${\pmb A}$$的**最大线性无关向量组**，$$r$$为向量组$${\pmb A}$$的**秩**，记作$$R_A$$。

**定理** 矩阵的秩等于其列向量组的秩，也等于其行向量组的秩。



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

$$\pmb e_1=\{1,0,\cdots\},\pmb e_2,\cdots,\pmb e_n$$称为$$\Bbb R^n$$中的**自然基**或**标准基**，向量在自然基上的坐标称为**笛卡尔坐标（Cartesian coordinate）**。



设向量组$$\pmb c_1,\pmb c_2,\cdots,\pmb c_n$$是向量空间$$V(V \subset \Bbb R^n)$$的一个基，如果$$\pmb c_1,\pmb c_2,\cdots,\pmb c_n$$两两正交且都是单位向量，则称其是$$V$$的一个**规范正交基**。

施密特正交化：$$\pmb{a_1,a_2,\cdots,a_r}$$是向量空间$$V$$的一个基，对其规范正交化，取
$$
\begin{align}
\pmb b_1=&\pmb a_1\\
\pmb b_2=&\pmb a_2-\frac{[\pmb b_1,\pmb a_2]}{[\pmb b_1 , \pmb b_1]}\pmb b_1 \\
\cdots&\cdots\\
\pmb b_r=&\pmb a_r-\frac{[\pmb b_1,\pmb a_r]}{[\pmb b_1 , \pmb b_1]}\pmb b_1-\cdots-\frac{[\pmb b_{r-1},\pmb a_r]}{[\pmb b_{r-1} , \pmb b_{r-1}]}\pmb b_{r-1}
\end{align}
$$
再将他们单位化即取$$\pmb c_i=\frac{1}{\Vert \pmb b_i \Vert} \pmb b_i$$，就是$$V$$的一个规范正交基。

向量组$$\pmb c_1,\pmb c_2,\cdots,\pmb c_n$$构成一个正交矩阵。



# 线性空间



## 线性变换

**线性变换(linear transformation)**或**线性映射(linear mapping)**是指从线性空间$$\mathcal{X}$$到线性空间$$\mathcal{Y}$$的一个映射
函数$$f∶\mathcal{X}→\mathcal{Y}$$，并满足对于$$\mathcal{X}$$中任何两个向量$$\pmb u$$和$$\pmb v$$以及任何标量c，有
$$
f(\pmb u +\pmb v) = f(\pmb u) + f(\pmb v)\\
f(c\pmb v) = cf(\pmb v)
$$
两个有限维欧氏空间的映射函数$$f∶\mathbb{R}^N →\mathbb{R}^M$$可以表示为
$$
\left\{ 
\begin{array}{l}
y_1=a_{11}x_1+a_{12}x_2+⋯+a_{1n}x_n \\ 
y_2=a_{21}x_1+a_{22}x_2+⋯+a_{2n}x_n \\ 
⋯⋯\\ 
y_m=a_{m1}x_1+a_{m2}x_2+⋯+a_{mn}x_n
\end{array}
\right.
$$




# 矩阵

## 矩阵

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

两个矩阵行列数相等时，称它们是**同型矩阵**。如果$${\pmb A}$$和$${\pmb B}$$是同型矩阵且对应元素相等，则称矩阵$${\pmb A}$$和矩阵$${\pmb B}$$相等，记作$$\pmb{A=B}$$。

元素都是$$0$$的矩阵称为**零矩阵**。
$$
\pmb{I}=\begin{bmatrix}
1&0&⋯&0\\\
0&1&⋯&0\\\
⋮&⋮&&⋮\\\
0&0&⋯&1
\end{bmatrix}
$$
称为$$n$$阶**单位矩阵**，简称单位阵。
$$
\pmb{\Lambda}=\begin{bmatrix}
\lambda_1&0&⋯&0\\\
0&\lambda_2&⋯&0\\\
⋮&⋮&&⋮\\\
0&0&⋯&\lambda_n
\end{bmatrix}
$$
称为**对角矩阵**，也记作$${\pmb \Lambda}={\rm diag}(\lambda_1,\lambda_2,⋯,\lambda_n)$$。



矩阵与线性变换一一对应：
$$
\left\{ 
\begin{array}{l}
y_1=a_{11}x_1+a_{12}x_2+⋯+a_{1n}x_n \\ 
y_2=a_{21}x_1+a_{22}x_2+⋯+a_{2n}x_n \\ 
⋯⋯\\ 
y_m=a_{m1}x_1+a_{m2}x_2+⋯+a_{mn}x_n
\end{array}
\right. \Leftrightarrow \pmb y=\pmb A \pmb x
$$



## 基本运算

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

对于单位矩阵$$\pmb{I}$$，有
$$
\pmb{IA=AI=A}
$$
设$$\pmb{A}$$为$$n$$阶方阵，定义矩阵的幂
$$
\pmb A^{k+1}=\pmb A^k \pmb A
$$


$$\pmb A=\{a_{ij}\}\in \R^{m\times n}$$转置记作$${\pmb A^ \rm T}$$，定义为$$({\pmb A^ \rm T})_{ij}=a_{ji},{\pmb A^ \rm T}\in\R^{n\times m}$$。转置满足以下性质

+ $${({\pmb{AB}})^{\rm T}=\pmb{B}^{\rm T}\pmb{A}^{\rm T}}$$





## 逆矩阵

对于$$n$$阶矩阵$$\pmb{A}$$，如果有一个$$n$$阶矩阵$$\pmb{B}$$使
$$
\pmb{AB}=
\pmb{I}\; or \;
\pmb{BA}=\pmb{I}
$$
则称矩阵$$\pmb{A}$$**可逆**，矩阵$$\pmb{A}$$和矩阵$$\pmb{B}$$互为**逆矩阵**，记作$$\pmb{B=A^{-1}}$$。

### 性质

+ 矩阵$$\pmb{A}$$可逆$$\iff 矩阵\pmb A为非奇异矩阵 \iff \vert \pmb{A}\vert \neq 0$$
+ $$({\pmb A \pmb B})^{-1}=\pmb B^{-1}\pmb A^{-1}$$
+ $$(\pmb A^{-1})^{\rm T}=(\pmb A^{\rm T})^{-1}$$
+ 



## 行列式

$$n$$阶方阵$$\pmb{A}$$的行列式是一个将其映射到标量的函数，记作$$\vert \pmb{A}\vert$$或$${\rm det}{\pmb A}$$，定义为
$$
{\rm det}\pmb A=\begin{vmatrix}
a_{11}&a_{12}&⋯&a_{1n}\\\
a_{21}&a_{22}&⋯&a_{2n}\\\
⋮&⋮&&⋮\\\
a_{n1}&a_{n2}&⋯&a_{nn}
\end{vmatrix}
=\sum(-1)^ta_{1p_1}a_{2p_2}⋯a_{np_n}
$$
其中$$t$$为排列$$p_1p_2⋯p_n$$的逆序数。

行列式可以看作有向面积或体积的概念在欧氏空间中的推广。在N维欧氏空间中，行列式描述的是一个线性变换对 “体积” 所造成的影响。

### 性质

+ $${\rm det}\pmb A^{\rm T}={\rm det}\pmb A$$

+ 互换行列式的两行/列，行列式变号

+ 如果行列式某行/列所有元素乘以$$k$$，行列式乘以$$k$$；$${\rm det}(\lambda\pmb{A}) =\lambda^n{\rm det} \pmb{A}$$

+ $$
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

+ $${\rm det}(\pmb A\pmb B)={\rm det}(\pmb A){\rm det}(\pmb B)$$

+ $$
  \begin{vmatrix}
  \pmb A&\\\
  \pmb C&\pmb B
  \end{vmatrix}
  =\begin{vmatrix}\pmb A\end{vmatrix}\begin{vmatrix}\pmb B\end{vmatrix}
  $$


### 展开

$$n$$阶行列式中去掉第$$i$$行和第$$j$$列后的$$n-1$$阶行列式为$$a_{ij}$$的**余子式**$$M_{ij}$$，**代数余子式**$$A_{ij}=(-1)^{i+j}M_{ij}$$

**定理** 行列式等于其任一行/列的各元素与其对应的代数余子式乘积之和，即
$$
{\rm det}\pmb A=a_{i1}A_{i1}+a_{i2}A_{i2}+⋯+a_{in}A_{in}
$$



## 秩，迹

一个矩阵$$\pmb A$$的列秩是$$\pmb A$$的线性无关的列向量数量，行秩是$$\pmb A$$的线性无关的行向量数量。一个矩阵的列秩和行秩总是相等的，简称为**秩(rank)**，记作$$R(\pmb A)$$。

**性质**

- $$0 \le R(\pmb{A}_{m×n}) \le {\rm min} \{ m,n\}$$
- $$R({\pmb A}^{\rm T})=R({\pmb A})$$
- $$\pmb{P},\pmb{Q}$$可逆，$$R(\pmb{PAQ})=R({\pmb A})$$
- $$R(\pmb A_{n\times n})<n \iff {\rm det}A=0$$
- $${\rm max}\{R({\pmb A}),R({\pmb B})\} \le R(\pmb{ A,B}) \le R({\pmb A})+R({\pmb B})$$
- $$R(\pmb{ A+B}) \le R({\pmb A})+R({\pmb B})$$ 
- $$R(\pmb{ AB}) \le {\rm min}\{R({\pmb A}),R({\pmb B})\} $$
- 若$${\pmb A_{m×n}\pmb B_{n×i}=\pmb O}$$，则$$R({\pmb A})+R({\pmb B}) \le n$$



方阵$$\pmb A$$的对角线元素之和称为它的**迹(trace)**，记为$${\rm tr}(\pmb A)$$。

**性质**

+ $${\rm tr}(\pmb A\pmb B) = {\rm tr}(\pmb B\pmb A)$$



## 特征值与相似矩阵

对于n阶矩阵$${\pmb A}$$，如果数$$\lambda$$和n维非零列向量$${\pmb x}$$使式
$$
\pmb A \pmb x = \lambda \pmb x
$$
成立，则$$\lambda$$称为矩阵$${\pmb A}$$的**特征值**，$${\pmb x}$$称为$${\pmb A}$$对应于特征值$$\lambda$$的**特征向量**。
$$
\pmb A \pmb x = \lambda \pmb x \iff (\pmb A -\lambda \pmb I )\pmb x= \pmb 0
$$
该其次线性方程组有非零解的充要条件是系数行列式
$$
p(\lambda)= \vert \pmb A -\lambda \pmb I \vert =\begin{vmatrix}
a_{11}-\lambda &a_{12}&\cdots&a_{1n}\\
a_{21} &a_{22}-\lambda&\cdots&a_{2n}\\
\vdots &\vdots&&\vdots\\
a_{n1} &a_{n2}&\cdots&a_{nn}-\lambda\\
\end{vmatrix}=0
$$
上式称为矩阵$${\pmb A}$$的**特征方程**，其左端称为矩阵$${\pmb A}$$的**特征多项式**。

**性质**

设矩阵$${\pmb A}$$的特征值为$$\lambda_1,\lambda_2,\cdots,\lambda_n$$，则

+ $$\lambda_1+\lambda_2+\cdots+\lambda_n={\rm tr}(\pmb A)$$
+ $$\lambda_1\lambda_2\cdots\lambda_n={\rm det}\pmb A$$
+ $$p(\pmb A)=\pmb 0$$



设$${\pmb A, \pmb B}$$都是n阶矩阵，如果有可逆矩阵$${\pmb P}$$，使
$$
\pmb{P^{-1}AP=B}
$$
则称$${\pmb B}$$是$${\pmb A}$$的**相似矩阵**。

**性质**

+ 相似矩阵有相同的特征多项式和相同的特征值
+ n阶矩阵$${\pmb A}$$与对角阵相似的充要条件是$${\pmb A}$$有n个线性无关的特征向量
  + 若特征方程有n个互不相等的根，则$${\pmb A}$$与对角阵相似
  + 若特征方程有重根，则$$\pmb A$$与相应的若尔当标准型相似



## 二次型与正定矩阵

含有n个变量$$x_1,x_2,\cdots,x_n$$的二次齐次函数$$f=\sum_{i,j=1}^na_{ij}x_ix_j$$称为**二次型**。只含平方项的二次型称为二次型的**标准型**。

二次型可记作$$f=\pmb x^ {\rm T} \pmb A \pmb x $$，其中对称阵$${\pmb A}$$称为二次型f的矩阵，f称为$${\pmb A}$$的二次型，$${\pmb A}$$的秩即为二次型f的秩。



**定理** 任给二次型$$f=\sum_{i,j=1}^na_{ij}x_ix_j$$，总有正交变换$$\pmb{x=Qy}$$使f化为标准型
$$
f=\lambda_1 y_1^2+\lambda_2 y_2^2+\cdots+\lambda_n y_n^2
$$
其中$$\lambda_1,\lambda_2,\cdots,\lambda_n$$是矩阵$$\pmb A$$的特征值。



### 正定二次型

**惯性定理** 设二次型$$f={\pmb x^ {\rm T}\pmb A \pmb x}$$的秩为r，有两个可逆变换$$\pmb{x=P_1y}$$和$$\pmb{x=P_2z}$$使
$$
f=k_1 y_1^2+k_2 y_2^2+\cdots+k_r y_n^2(k_i \neq 0)\\
f=\lambda_1 z_1^2+\lambda_2 z_2^2+\cdots+\lambda_n z_n^2(\lambda_i \neq0)
$$
则$$k_i$$和$$\lambda_i$$中正数个数相等。正系数的个数称为正惯性指数，负系数的个数称为负惯性指数。

设二次型$$f={\pmb x^ {\rm T}\pmb A \pmb x}$$，如果对任何$$\pmb {x \neq 0}$$都有$$f(\pmb x)>0$$，则称f为**正定二次型**，并称$$\pmb A$$是**正定矩阵**。

**性质**

+ n元二次型$$f={\pmb x^ {\rm T}\pmb A \pmb x}$$正定的充要条件是它的标准形的n个系数全为正，即正惯性指数等于n
+ 对称阵$${\pmb A}$$正定的充要条件是$${\pmb A}$$的特征值全为正
+ 对称阵$${\pmb A}$$正定的充要条件是$${\pmb A}$$的各阶主子式都为正；对称阵$${\pmb A}$$负定的充要条件是$${\pmb A}$$的奇数阶主子式为负，偶数阶主子式为正
+ 对称阵$${\pmb A}$$正定的充要条件是存在可逆矩阵$$\pmb P$$，使得$$\pmb A=\pmb P^{\rm T}\pmb P$$



## 矩阵的特殊运算

### 有穷级数

$$
\pmb I+\pmb X+\pmb X^2+\cdots+\pmb X^{n-1} = (\pmb X^n-\pmb I)(\pmb  X-\pmb I)^{-1}
$$



### 无穷级数

对于一维的超越函数，可以以无穷级数的方式定义以方阵为参数的此函数，例如
$$
\exp(\pmb A)=\pmb I+\pmb A+\frac{\pmb A^2}{2!}+\cdots\\
\sin(\pmb A)=\pmb A-\frac{\pmb A^3}{3!}+\frac{\pmb A^5}{5!}-\cdots
$$
**性质**

+ 如果$$\pmb A$$能够特征分解为$$\pmb P\pmb B\pmb P^{-1}$$，那么$$f(\pmb A)=\pmb Pf(\pmb B)\pmb P^{-1}$$
+ 如果？，$$\lim_{n\to \infty}\pmb A^n\to \pmb 0$$
+ $$f(\pmb A\pmb B)\pmb A=\pmb Af(\pmb B\pmb A)$$

指数函数的一些性质包含

+ $$\exp(\pmb A)\exp(\pmb B)=\exp(\pmb A+\pmb B)$$，如果方阵$$\pmb A,\pmb B$$是可交换的

+ $$\exp^{-1}(\pmb A)=\exp(-\pmb A)$$

+ $$
  \frac{{\rm d}}{{\rm d}t}e^{t\pmb A}=\pmb Ae^{t\pmb A}
  $$

+ $$
  \frac{{\rm d}}{{\rm d}t}{\rm tr}(e^{t\pmb A})={\rm tr}(\pmb Ae^{t\pmb A})
  $$

+ $${\rm det}(\exp(\pmb A))=\exp({\rm tr}(\pmb A))$$



### Kronecker积

矩阵$$\pmb A_{m\times n}$$和$$\pmb B_{p\times q}$$的Kronecker积记作$$\pmb A\otimes \pmb B$$，定义为
$$
\pmb A\otimes \pmb B=\begin{bmatrix}a_{11}\pmb B & \cdots & a_{1n}\pmb B\\
\vdots & \ddots & \vdots\\
a_{m1}\pmb B & \cdots & a_{mn}\pmb B\\
\end{bmatrix}
$$


### Vec算子

vec算子将矩阵$$\pmb A_{m\times n}$$的所有列汇集到一列，定义为
$$
{\rm vec}(\pmb A)=[a_{11},a_{21},\cdots,a_{m1},\cdots,a_{1n},a_{2n},\cdots,a_{mn}]^{\rm T}
$$


### 范数

$$
||\pmb A||_1=\max_j\sum_i|a_{ij}|\\
||\pmb A||_2=\sqrt{\max{\rm eig}(\pmb A^{\rm T}\pmb A)}\quad{\rm eig}(\cdot)表示矩阵的所有特征值的集合 \\
||\pmb A||_p=(\max_{||\pmb x||_p=1}||\pmb A\pmb x||_p)^{1/p}\\
||\pmb A||_\infty=\max_i\sum_j|a_{ij}|\\
||\pmb A||_F=\sqrt{\sum_{i,j}|a_{ij}|^2}\\
||\pmb A||_{\max}=\max_{ij}|a_{ij}|\\
$$





## 矩阵的特殊类型

**正交矩阵**

如果n阶矩阵$$\pmb Q$$满足
$$
\pmb Q^ {\rm T} \pmb Q=\pmb E \quad {\rm i.e.} \quad \pmb Q^{-1}= \pmb Q^ {\rm T}
$$
则称$${\pmb Q}$$为正交矩阵。方阵$${\pmb Q}$$为正交矩阵的充要条件是$${\pmb Q}$$的列向量都是单位向量，且两两正交。

若$${\pmb Q}$$为正交矩阵，则线性变换$$\pmb{ y=Qx}$$称为正交变换。正交变换不改变向量的长度。



**伴随矩阵**

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
\vert \pmb{A}\vert \pmb{I}
$$



**幂零矩阵**



## 线性方程组的解

**定理** $$n$$元线性方程组$$\pmb{Ax=b}$$

1. 无解的充要条件是$$R(\pmb{ A}) < R({\pmb A,\pmb b})$$
2. 有唯一解的充要条件是$$R(\pmb{ A}) = R({\pmb A,\pmb b}) = n$$
3. 有无穷解的充要条件是$$R(\pmb{ A}) = R({\pmb A,\pmb b}) < n$$

求解线性方程组的步骤：

1. 对于非齐次线性方程组，将其增广矩阵化为行阶梯形，若此时$$R(\pmb{A})<R(\pmb{B})$$，则无解
2. 若$$R(\pmb{A})=R(\pmb{B})$$，进一步把$$\pmb{B}$$化为最简形
3. 若$$R(\pmb{A})=R(\pmb{B})=r$$，写出含$$n-r$$个参数的通解

**定理** 矩阵方程$$\pmb{AX=B}$$有解的充要条件是$$R(\pmb{ A}) = R({\pmb A,\pmb B})$$



# 矩阵分解

## 特征分解



## 若尔当标准型



## 奇异值分解





# 酉空间





