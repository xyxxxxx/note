# 向量空间

## 向量



## 向量空间



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

对于单位矩阵$$\pmb{I}$$，有
$$
\pmb{IA=AI=A}
$$
设$$\pmb{A}$$为$$n$$阶方阵，定义矩阵的幂
$$
\pmb A^{k+1}=\pmb A^k \pmb A
$$
转置矩阵记作$${\pmb A^ \rm T}$$，满足
$$
{({\pmb{AB}})^{\rm T}=
\pmb{B}^{\rm T}\pmb{A}^{\rm T}}
$$


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

+ $$
  ({\pmb A \pmb B})^{-1}=\pmb B^{-1}\pmb A^{-1}
  $$

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

+ $$|\pmb A\pmb B|=|\pmb A||\pmb B|$$

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


## 秩，迹，范式

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



方阵$$\pmb A$$的对角线元素之和称为它的**迹(trace)**，记为$$tr(\pmb A)$$。

**性质**

+ $$tr(\pmb A\pmb B) = tr(\pmb B\pmb A)$$
  

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
\vert \pmb A -\lambda \pmb I \vert =0 \\
\rm{i.e.}\;\begin{vmatrix}
a_{11}-\lambda &a_{12}&\cdots&a_{1n}\\
a_{21} &a_{22}-\lambda&\cdots&a_{2n}\\
\vdots &\vdots&&\vdots\\
a_{n1} &a_{n2}&\cdots&a_{nn}-\lambda\\
\end{vmatrix}=0
$$
上式称为矩阵$${\pmb A}$$的**特征方程**，其左端称为矩阵$${\pmb A}$$的**特征多项式**。

**性质**

设矩阵$${\pmb A}$$的特征值为$$\lambda_1,\lambda_2,\cdots,\lambda_n$$，则

+ $$\lambda_1+\lambda_2+\cdots+\lambda_n=tr(\pmb A)$$
+ $$\lambda_1\lambda_2\cdots\lambda_n={\rm det}\pmb A$$



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







## 矩阵类型

**正交矩阵**





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







