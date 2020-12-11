> 参考[Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf)

# 向量空间

## 向量

**标量(scalar)**是一个实数，只有大小，没有方向。标量一般用斜体小写英文字母$$a,b,c$$来表示。**向量(vector)**是由一组实数组成的有序数组，同时具有大小和方向。一个$$N$$维向量$$\pmb a$$是由$$N$$个有序实数组成，表示为
$$
\pmb a = [a_1 , a_2 , ⋯ , a_N ]
$$



## 向量运算

### 内积

两个$$N$$维向量$$\pmb a，\pmb b$$的**内积(inner product)**定义为
$$
\lang \pmb a,\pmb b \rang =a_1b_1+a_2b_2+\cdots+a_nb_n=\pmb a^{\rm T} \pmb b
$$
当$$\lang\pmb a,\pmb b\rang=0$$时，称向量$${\pmb a}$$与$${\pmb b}$$**正交(orthogonal)**。内积也称为**点积(dot product)**或**标量积(scalar product)**。



### 外积

两个向量$$\pmb a\in \mathbb{R}^M$$和$$\pmb b\in \mathbb{R}^N$$的**外积(outer product)**是一个$$M\times N$$矩阵，定义为
$$
\pmb a \otimes\pmb b=\begin{bmatrix}a_1b_1 & a_1b_2 &\cdots& a_1b_N\\
a_2b_1 & a_2b_2 &\cdots& a_2b_N\\
\vdots & \vdots & \ddots & \vdots\\
a_Mb_1 & a_Mb_2 &\cdots& a_Mb_N
\end{bmatrix}=\pmb a \pmb b^{\rm T}
$$
外积也称为叉积或矢量积。



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

对于给定的向量组$$A:\pmb{a_1,a_2,\cdots,a_n}$$，对于任何一组实数$$k_1,k_2,\cdots,k_m$$，$$k_1{\pmb a}_1+k_2{\pmb a}_2+\cdots+k_m{\pmb a}_m$$称为向量组$$A$$的一个**线性组合**。给定向量组$$A$$和向量$$\pmb{b}$$，如果存在一组数$$\lambda_1,\lambda_2,\cdots,\lambda_m$$，使$${\pmb b}=k_1{\pmb a}_1+k_2{\pmb a}_2+\cdots+k_m{\pmb a}_m$$，则称$${\pmb b}$$能由向量组$$A$$线性表示，亦即方程组$$x_1{\pmb a}_1+x_2{\pmb a}_2+\cdots+x_m{\pmb a}_m=\pmb{b}$$有解。

**定理** 向量$$\pmb{b}$$能被向量组$$A:\pmb{a_1,a_2,\cdots,a_n}$$线性表示的充要条件是矩阵$$A=({\pmb a}_1,{\pmb a}_2,\cdots,{\pmb a}_m)$$的秩等于矩阵$$B=({\pmb a}_1,{\pmb a}_2,\cdots,{\pmb a}_m,{\pmb b})$$的秩。

设有向量组$$A:\pmb{a_1,a_2,\cdots,a_n}$$和向量组$$B:\pmb{b_1,b_2,\cdots,b_n}$$，若B组每个向量都能由A组线性表示，则称向量组B能由向量组A线性表示。若向量组A和向量组B能互相线性表示，则称这两个向量组**等价**。



### 线性相关

对于给定的向量组$$A:\pmb{a_1,a_2,\cdots,a_m}$$，如果存在不全为零的数$$k_1,k_2,\cdots,k_m$$使$$k_1{\pmb a}_1+k_2{\pmb a}_2+\cdots+k_m{\pmb a}_m=\bf 0$$，则称向量组$$A$$是**线性相关**的，否则称其**线性无关**。

**定理** 向量组$$\pmb{a_1,a_2,\cdots,a_m}$$线性相关的充要条件是$$R(A)<m$$；线性无关的充要条件是$$R(A)=m$$。



### 秩

对于给定的向量组$$\pmb a_1,\pmb a_2,\cdots,\pmb a_m$$，如果能选出$$r$$个向量$$\pmb a_1,\pmb a_2,\cdots,\pmb a_r$$，满足

1. 向量组$$\pmb a_1,\pmb a_2,\cdots,\pmb a_r$$线性无关
2. 向量组$$A$$中任意$$r+1$$个向量都线性相关

则称向量组$${A_0}$$是向量组$$A$$的**最大线性无关向量组**，$$r$$为向量组$$A$$的**秩(rank)**，记作$$R_A$$。

**定理** 矩阵的秩等于其列向量组的秩，也等于其行向量组的秩。



## 向量空间

设$$V$$为$$n$$维向量的集合，如果集合$$V$$非空，且对于向量的加法及数乘两种运算封闭，则称为**向量空间**。

设$$V$$为向量空间，如果$$r$$个向量$$\pmb a_1,\pmb a_2,\cdots,\pmb a_r \in V$$，且满足

1. $$\pmb a_1,\pmb a_2,\cdots,\pmb a_r$$线性相关
2. $$V$$中任一向量都可由$$\pmb a_1,\pmb a_2,\cdots,\pmb a_r$$线性表示

则称向量组$$\pmb a_1,\pmb a_2,\cdots,\pmb a_r$$为向量空间的$$V$$的一个**基(base)**，$$r$$称为向量空间$$V$$的维数，称其为$$r$$维向量空间。

如果再向量空间$$V$$中取定一个基$$\pmb a_1,\pmb a_2,\cdots,\pmb a_r$$，那么$$V$$中任一向量$${\pmb x}$$可唯一表示为
$$
\pmb x=\lambda_1 \pmb a_1 +\lambda_2 \pmb a_2+\cdots+\lambda_r \pmb a_r
$$
数组$$\lambda_1,\lambda_2,\cdots,\lambda_r$$称为向量$$\pmb x$$在基$$\pmb a_1,\pmb a_2,\cdots,\pmb a_r$$上的**坐标(coordinate)**。

$$\pmb e_1=\{1,0,\cdots\},\pmb e_2,\cdots,\pmb e_n$$称为$$\Bbb R^n$$中的**自然基**或**标准基**，向量在自然基上的坐标称为**笛卡尔坐标(Cartesian coordinate)**。



设向量组$$\pmb c_1,\pmb c_2,\cdots,\pmb c_n$$是向量空间$$V(V \subset \Bbb R^n)$$的一个基，如果$$\pmb c_1,\pmb c_2,\cdots,\pmb c_n$$两两正交且都是单位向量，则称其是$$V$$的一个**规范正交基**。

**施密特正交化**：$$\pmb a_1,\pmb a_2,\cdots,\pmb a_r$$是向量空间$$V$$的一个基，对其规范正交化，取
$$
\begin{align}
\pmb b_1=&\pmb a_1\\
\pmb b_2=&\pmb a_2-\frac{\lang\pmb b_1,\pmb a_2\rang}{\lang\pmb b_1 , \pmb b_1\rang}\pmb b_1 \\
\cdots&\cdots\\
\pmb b_r=&\pmb a_r-\frac{\lang\pmb b_1,\pmb a_r\rang}{\lang\pmb b_1 , \pmb b_1\rang}\pmb b_1-\cdots-\frac{\lang\pmb b_{r-1},\pmb a_r\rang}{\lang\pmb b_{r-1} , \pmb b_{r-1}\rang}\pmb b_{r-1}
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
A=\begin{bmatrix}
a_{11}&a_{12}&⋯&a_{1n}\\\
a_{21}&a_{22}&⋯&a_{2n}\\\
⋮&⋮&&⋮\\\
a_{n1}&a_{n2}&⋯&a_{nn}
\end{bmatrix}
$$
这$$m×n$$个数称为矩阵$$A$$的**元素**，简称**元**。以数$$a_{ij}$$为$$(i,j)$$元的矩阵简记作$$(a_{ij})$$或$$(a_{ij})_{m×n}$$。

行列数都等于$$n$$的矩阵称为$$n$$阶矩阵或$$n$$阶**方阵**，$$n$$阶矩阵$$A$$也记作$${A_n}$$。

两个矩阵行列数相等时，称它们是**同型矩阵**。如果$$A$$和$$B$$是同型矩阵且对应元素相等，则称矩阵$$A$$和矩阵$$B$$相等，记作$$A=B$$。

元素都是$$0$$的矩阵称为**零矩阵**。
$$
I=\begin{bmatrix}
1&0&⋯&0\\\
0&1&⋯&0\\\
⋮&⋮&&⋮\\\
0&0&⋯&1
\end{bmatrix}
$$
称为$$n$$阶**单位矩阵**，简称单位阵。
$$
\Lambda=\begin{bmatrix}
\lambda_1&0&⋯&0\\\
0&\lambda_2&⋯&0\\\
⋮&⋮&&⋮\\\
0&0&⋯&\lambda_n
\end{bmatrix}
$$
称为**对角矩阵**，也记作$$\Lambda={\rm diag}(\lambda_1,\lambda_2,⋯,\lambda_n)$$。



矩阵与线性变换一一对应：
$$
\left\{ 
\begin{array}{l}
y_1=a_{11}x_1+a_{12}x_2+⋯+a_{1n}x_n \\ 
y_2=a_{21}x_1+a_{22}x_2+⋯+a_{2n}x_n \\ 
⋯⋯\\ 
y_m=a_{m1}x_1+a_{m2}x_2+⋯+a_{mn}x_n
\end{array}
\right. \Leftrightarrow \pmb y=A \pmb x
$$



## 基本运算

设有两个$$m×n$$矩阵$$A$$和$$B$$，那么矩阵$$A$$与$$B$$的和记作$$A+B$$，规定为
$$
A+B=\begin{bmatrix}
a_{11}+b_{11}&a_{12}+b_{12}&⋯&a_{1n}+b_{1n}\\\
a_{21}+b_{21}&a_{22}+b_{22}&⋯&a_{2n}+b_{2n}\\\
⋮&⋮&&⋮\\\
a_{m1}+b_{m1}&a_{m2}+b_{m2}&⋯&a_{mn}+b_{mn}
\end{bmatrix}
$$
矩阵加法满足交换律和结合律。记$$-A=(-a_{ij})$$。



数$$\lambda$$与矩阵$$A$$的乘积记作$$\lambda A$$或$$A\lambda$$，规定为
$$
\lambda A=A\lambda=
\begin{bmatrix}
\lambda a_{11}&\lambda a_{12}&⋯&\lambda a_{1n}\\\
\lambda a_{21}&\lambda a_{22}&⋯&\lambda a_{2n}\\\
⋮&⋮&&⋮\\\
\lambda a_{m1}&\lambda a_{m2}&⋯&\lambda a_{mn}
\end{bmatrix}
$$
数乘矩阵满足结合律和分配律。



设$$A=(a_{ij})$$是$$m×s$$矩阵，$$B=(b_{ij})$$是$$s×n$$矩阵，那么矩阵$$A$$和$$B$$的乘积是$$m×n$$矩阵$$\pmb{C}=(c_{ij})$$，其中
$$
c_{ij}=\sum_{k=1}^{s}{a_{ik}b_{kj}}
$$
记作$$C=AB$$。

矩阵乘法不满足交换律(一般$$AB≠BA$$)，满足结合律和分配律。如果$$AB=BA$$，则称$$A$$与$$B$$可交换。

对于单位矩阵$$I$$，有
$$
IA=AI=A
$$
设$$A$$为$$n$$阶方阵，定义矩阵的幂
$$
A^{k+1}=A^k A
$$


$$A=\{a_{ij}\}\in \R^{m\times n}$$转置记作$${A^ \rm T}$$，定义为$$({A^ \rm T})_{ij}=a_{ji},{A^ \rm T}\in\R^{n\times m}$$。转置满足以下性质

+ $$(AB)^{\rm T}=B^{\rm T}A^{\rm T}$$



## 逆矩阵

对于$$n$$阶矩阵$$A$$，如果有一个$$n$$阶矩阵$$B$$使
$$
AB=
I\; or \;
BA=I
$$
则称矩阵$$A$$**可逆**，矩阵$$A$$和矩阵$$B$$互为**逆矩阵(inverse matrix)**，记作$$B=A^{-1}$$。

### 性质

+ 矩阵$$A$$可逆$$\iff 矩阵A为非奇异矩阵 \iff | A| \neq 0$$
+ $$({A B})^{-1}=B^{-1}A^{-1}$$
+ $$(A^{-1})^{\rm T}=(A^{\rm T})^{-1}$$



## 行列式

$$n$$阶方阵$$A$$的行列式是一个将其映射到标量的函数，记作$$| A|$$或$${\rm det}A$$，定义为
$$
|A|={\rm det}A=\begin{vmatrix}
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

+ 行列式的一行（列）元素全为0，行列式的值为0

+ 互换行列式的两行（列），行列式变号；行列式的两行（列）对应成比例或相同，行列式的值为0

+ 如果行列式某行（列）所有元素乘以$$k$$，行列式乘以$$k$$；$$|\lambda A| =\lambda^n|A|$$

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

+ 将一行（列）的$$k$$倍加进另一行（列）里，行列式的值不变

+ $$|A^{\rm T}|=|A|$$

+ $$|AB|=|A||B|$$

+ $$
  \begin{vmatrix}
  A&\\\
  C&B
  \end{vmatrix}
  =\begin{vmatrix}A\end{vmatrix}\begin{vmatrix}B\end{vmatrix}
  $$



### 展开

$$n$$阶行列式中去掉第$$i$$行和第$$j$$列后的$$n-1$$阶行列式为$$a_{ij}$$的**余子式**$$M_{ij}$$，**代数余子式**$$A_{ij}=(-1)^{i+j}M_{ij}$$

**定理** 行列式等于其任一行/列的各元素与其对应的代数余子式乘积之和，即
$$
|A|=a_{i1}A_{i1}+a_{i2}A_{i2}+⋯+a_{in}A_{in}
$$



## 秩，迹

一个矩阵$$A$$的列秩是$$A$$的线性无关的列向量数量，行秩是$$A$$的线性无关的行向量数量。一个矩阵的列秩和行秩总是相等的，简称为**秩(rank)**，记作$$R(A)$$。

**性质**

- $$0 \le R(A_{m×n}) \le {\rm min} \{ m,n\}$$
- $$R(A^{\rm T})=R(A)$$
- $$P,Q$$可逆，$$R(PAQ)=R(A)$$
- $$R(A_{n\times n})<n \iff |A|=0$$
- $${\rm max}\{R(A),R(B)\} \le R(A,B) \le R(A)+R(B)$$
- $$R(A+B) \le R(A)+R(B)$$ 
- $$R(AB) \le {\rm min}\{R(A),R(B)\} $$
- 若$$A_{m×n}B_{n×i}=0$$，则$$R(A)+R(B) \le n$$



方阵$$A$$的对角线元素之和称为它的**迹(trace)**，记为$${\rm tr}(A)$$。

**性质**

+ $${\rm tr}(A+B) = {\rm tr}(A)+{\rm tr}(B),\ {\rm tr}(rA)=r\cdot{\rm tr}(A)$$
+ $${\rm tr}(A)={\rm tr}(A^{\rm T})$$
+ $${\rm tr}(AB) = {\rm tr}(BA)$$
+ 对于方阵$$A,B,C$$，$${\rm tr}(ABC) = {\rm tr}(BCA) = {\rm tr}(CAB)$$



## 特征值与相似矩阵

对于$$n$$阶矩阵$$A$$，如果数$$\lambda$$和$$n$$维非零列向量$${\pmb x}$$使式
$$
A \pmb x = \lambda \pmb x
$$
成立，则$$\lambda$$称为矩阵$$A$$的**特征值**，$${\pmb x}$$称为$$A$$对应于特征值$$\lambda$$的**特征向量**。
$$
A \pmb x = \lambda \pmb x \iff (A -\lambda I )\pmb x= \pmb 0
$$
该其次线性方程组有非零解的充要条件是系数行列式
$$
p(\lambda)= | A -\lambda I | =\begin{vmatrix}
a_{11}-\lambda &a_{12}&\cdots&a_{1n}\\
a_{21} &a_{22}-\lambda&\cdots&a_{2n}\\
\vdots &\vdots&&\vdots\\
a_{n1} &a_{n2}&\cdots&a_{nn}-\lambda\\
\end{vmatrix}=0
$$
上式称为矩阵$$A$$的**特征方程**，其左端称为矩阵$$A$$的**特征多项式**。

**性质**

设矩阵$$A$$的特征值为$$\lambda_1,\lambda_2,\cdots,\lambda_n$$，则

+ $$\lambda_1+\lambda_2+\cdots+\lambda_n={\rm tr}(A)$$
+ $$\lambda_1\lambda_2\cdots\lambda_n=|A|$$
+ $$p(A)=0$$



设$${A, B}$$都是$$n$$阶矩阵，如果有可逆矩阵$${P}$$，使
$$
P^{-1}AP=B
$$
则称$$B$$是$$A$$的**相似矩阵**。

**性质**

+ 相似矩阵有相同的：秩，行列式，迹，特征多项式和特征值（特征向量一般不同）
+ $$n$$阶矩阵$$A$$与对角阵相似的充要条件是$$A$$有$$n$$个线性无关的特征向量（参见可对角化矩阵）



## 二次型与正定矩阵

含有$$n$$个变量$$x_1,x_2,\cdots,x_n$$的二次齐次函数$$f=\sum_{i,j=1}^na_{ij}x_ix_j$$称为**二次型**。只含平方项的二次型称为二次型的**标准型**。

二次型可记作$$f=\pmb x^ {\rm T} A \pmb x $$，其中对称阵$$A$$称为二次型$$f$$的矩阵，$$f$$称为$$A$$的二次型，$$A$$的秩即为二次型$$f$$的秩。



**定理** 任给二次型$$f=\sum_{i,j=1}^na_{ij}x_ix_j$$，总有正交变换$$\pmb x=Q\pmb y$$使$$f$$化为标准型
$$
f=\lambda_1 y_1^2+\lambda_2 y_2^2+\cdots+\lambda_n y_n^2
$$
其中$$\lambda_1,\lambda_2,\cdots,\lambda_n$$是矩阵$$A$$的特征值。



### 正定二次型

**惯性定理** 设二次型$$f={\pmb x^ {\rm T}A \pmb x}$$的秩为r，有两个可逆变换$$\pmb x=P_1\pmb y$$和$$\pmb x=P_2\pmb z$$使
$$
f=k_1 y_1^2+k_2 y_2^2+\cdots+k_r y_n^2(k_i \neq 0)\\
f=\lambda_1 z_1^2+\lambda_2 z_2^2+\cdots+\lambda_n z_n^2(\lambda_i \neq0)
$$
则$$k_i$$和$$\lambda_i$$中正数个数相等。正系数的个数称为**正惯性指数**，负系数的个数称为**负惯性指数**。

设二次型$$f={\pmb x^ {\rm T}A \pmb x}$$，如果对任何$$\pmb {x \neq 0}$$都有$$f(\pmb x)>0$$，则称$$f$$为**正定二次型**，并称$$A$$是**正定矩阵**。

**性质**

+ n元二次型$$f={\pmb x^ {\rm T}A \pmb x}$$正定的充要条件是它的标准形的n个系数全为正，即正惯性指数等于$$n$$
+ 对称阵$$A$$正定的充要条件是$$A$$的特征值全为正
+ 对称阵$$A$$正定的充要条件是$$A$$的各阶主子式都为正；对称阵$$A$$负定的充要条件是$$A$$的奇数阶主子式为负，偶数阶主子式为正
+ 对称阵$$A$$正定的充要条件是存在可逆矩阵$$P$$，使得$$A=P^{\rm T}P$$



## 矩阵的特殊运算

### 有穷级数

$$
I+X+X^2+\cdots+X^{n-1} = (X^n-I)(X-I)^{-1}
$$



### 无穷级数

对于一维的超越函数，可以以无穷级数的方式定义以方阵为参数的此函数，例如
$$
\exp(A)=I+A+\frac{A^2}{2!}+\cdots\\
\sin(A)=A-\frac{A^3}{3!}+\frac{A^5}{5!}-\cdots
$$
**性质**

+ 如果$$A$$能够特征分解为$$PBP^{-1}$$，那么$$f(A)=Pf(B)P^{-1}$$
+ 如果？，$$\lim_{n\to \infty}A^n\to \pmb 0$$
+ $$f(AB)A=Af(BA)$$

指数函数的一些性质包含

+ $$\exp(A)\exp(B)=\exp(A+B)$$，如果方阵$$A,B$$是可交换的

+ $$\exp^{-1}(A)=\exp(-A)$$

+ $$
  \frac{{\rm d}}{{\rm d}t}e^{tA}=Ae^{tA}
  $$

+ $$
  \frac{{\rm d}}{{\rm d}t}{\rm tr}(e^{tA})={\rm tr}(Ae^{tA})
  $$

+ $${\rm det}(\exp(A))=\exp({\rm tr}(A))$$



### Kronecker积

矩阵$$A_{m\times n}$$和$$B_{p\times q}$$的Kronecker积记作$$A\otimes B$$，定义为
$$
A\otimes B=\begin{bmatrix}a_{11}B & \cdots & a_{1n}B\\
\vdots & \ddots & \vdots\\
a_{m1}B & \cdots & a_{mn}B\\
\end{bmatrix}
$$



### Vec算子

vec算子将矩阵$$A_{m\times n}$$的所有列汇集到一列，定义为
$$
{\rm vec}(A)=[a_{11},a_{21},\cdots,a_{m1},\cdots,a_{1n},a_{2n},\cdots,a_{mn}]^{\rm T}
$$



### 范数

$$
||A||_1=\max_j\sum_i|a_{ij}|\\
||A||_2=\sqrt{\max{\rm eig}(A^{\rm T}A)}\quad{\rm eig}(\cdot)表示矩阵的所有特征值的集合 \\
||A||_p=(\max_{||\pmb x||_p=1}||A\pmb x||_p)^{1/p}\\
||A||_\infty=\max_i\sum_j|a_{ij}|\\
||A||_F=\sqrt{\sum_{i,j}|a_{ij}|^2}\\
||A||_{\max}=\max_{ij}|a_{ij}|\\
$$



## 矩阵的特殊类型

### 可对角化矩阵

$$n$$阶方阵$$A$$是可对角化的，当且仅当下列条件之一成立：

1. $$A$$有$$n$$个线性无关的特征向量
2. $$A$$的所有特征值的几何重数等于相应的代数重数，即几何重数之和为$$n$$

可对角化和不可对角化的例子参见特征分解和若尔当标准型。



### 对称矩阵symmetric matrix, 反对称矩阵anti-symmetric matrix

如果$$n$$阶方阵$$A$$满足
$$
A^{\rm T}=A
$$
则称$$A$$为对称矩阵。如果$$n$$阶方阵$$A$$满足
$$
A^{\rm T}=-A
$$
则称$$A$$为反对称矩阵。

**性质**

+ 每个实方阵都可写作两个实对称矩阵的积，每个复方阵都可写作两个复对称矩阵的积

+ 实对称矩阵有$$n$$个线性无关的特征向量，因此必可对角化
+ 实对称矩阵的特征值都是实数，特征向量都是实向量
+ 实对称矩阵的不同特征值所对应的特征向量是正交的



### 正交矩阵orthogonal matrix

如果$$n$$阶方阵$$Q$$满足
$$
Q^ {\rm T} Q=I \quad {\rm i.e.} \quad Q^{-1}= Q^ {\rm T}
$$
则称$${Q}$$为正交矩阵。方阵$${Q}$$为正交矩阵的充要条件是$${Q}$$的列向量都是单位向量，且两两正交。

**性质**

+ 线性变换$$\pmb y=Q\pmb x$$称为正交变换，正交变换不改变向量的长度
+ $$|Q|=\pm 1$$



### 埃尔米特矩阵Hermitian matrix, 反埃尔米特矩阵anti-Hermitian matrix

如果$$n$$阶复方阵$$A$$满足
$$
A^{\rm H}=A
$$
则称$$A$$为埃尔米特矩阵，其中$$A^H$$是$$A$$的共轭转置。如果$$n$$阶复方阵$$A$$满足
$$
A^{\rm H}=-A
$$
则称$$A$$为反埃尔米特矩阵。



### 酉矩阵unitary matrix

如果$$n$$阶复方阵$$U$$满足
$$
UU^{\rm H}=U^{\rm H}U=I_n
$$
则称$$U$$为酉矩阵，其中$$U^{\rm H}$$是$$U$$的共轭转置。



### 正规矩阵normal matrix

如果$$n$$阶复方阵$$A$$满足
$$
AA^{\rm H}=A^{\rm H}A
$$
则称$$A$$为正规矩阵，其中$$A^{\rm H}$$是$$A$$的共轭转置。

在复系数矩阵中，所有的酉矩阵、埃尔米特矩阵和反埃尔米特矩阵都是正规的；在实系数矩阵中，所有的正交矩阵、对称矩阵和反对称矩阵都是正规的。

**谱定理** 矩阵$$A$$正规当且仅当它可以写成$$A=U\Lambda U^{\rm H}$$的形式，其中$$\Lambda={\rm diag}(\lambda_1,\lambda_2,\cdots)$$为对角矩阵，$$U$$为酉矩阵。



### 伴随矩阵

行列式$$| A|$$的各个元素的代数余子式$$A_{ij}$$所构成的如下矩阵
$$
A^*=\begin{bmatrix}
A_{11}&A_{12}&⋯&A_{1n}\\\
A_{21}&A_{22}&⋯&A_{2n}\\\
⋮&⋮&&⋮\\\
A_{n1}&A_{n2}&⋯&A_{nn}
\end{bmatrix}
$$
称为矩阵$$A$$的伴随矩阵，有
$$
AA^*=
A^*A=
|A| {I}
$$



### 正定矩阵positive-definite matrix

如果对于任意的$$n$$维非零实向量$$\pmb x$$，$$n$$阶实对称阵$$A$$都满足
$$
\pmb x^{\rm T}A\pmb x>0
$$
则称$$A$$是正定的；如果对于任意的$$n$$维非零复向量$$\pmb x$$，$$n$$阶埃尔米特矩阵$$A$$都满足
$$
\pmb x^{\rm H}A\pmb x>0
$$
则称$$A$$为正定的。若上式取大于等于，则称为半正定的；若上式取小于，则称为负定的。

对于正定阵$$A$$，我们记作$$A\ge 0$$，半正定阵记作$$A>0$$。对于一般的埃尔米特矩阵$$A,B$$，$$A\ge B\iff A-B\ge 0$$，这样就定义了一个在埃尔米特矩阵集合上的偏序关系，类似地，也可以定义大于、小于。

+ 每个正定阵都是可逆的，它的逆也是正定阵，且有$$A\ge B>0\iff B^{-1}\ge A^{-1}>0$$
+ 如果$$A>0$$，$$r$$为正实数，那么$$rA>0$$；如果$$A,B>0$$，那么$$A+B,ABA,BAB>0$$，$$AB>0$$（如果$$AB=BA$$）
+ $$A>0$$当且仅当存在唯一的$$B>0$$使得$$B^2=A$$，根据唯一性可以记作$$B=A^{1/2}$$，且有$$A>B>0\iff A^{1/2}>B^{1/2}>0$$
+ ……



### 幂零矩阵nilpotent matrix

如果存在一个正整数$$q$$，使得$$n$$阶方阵$$A$$满足
$$
A^q=0
$$
则称$$A$$为幂零矩阵。

**性质**

+ 满足$$A^q=0$$的最小整数小于等于$$n$$
+ $$A$$是幂零的当且仅当$$A$$的所有特征值为零



## 线性方程组的解

**定理** $$n$$元线性方程组$$A\pmb x=\pmb b$$

1. 无解的充要条件是$$R(A) < R(A,\pmb b)$$
2. 有唯一解的充要条件是$$R(A) = R(A,\pmb b) = n$$
3. 有无穷解的充要条件是$$R(A) = R(A,\pmb b) < n$$

求解线性方程组的步骤：

1. 对于非齐次线性方程组，将其增广矩阵化为行阶梯形，若此时$$R(A)<R(B)$$，则无解
2. 若$$R(A)=R(B)$$，进一步把$$B$$化为最简形
3. 若$$R(A)=R(B)=r$$，写出含$$n-r$$个参数的通解

**定理** 矩阵方程$$AX=B$$有解的充要条件是$$R(A) = R(A,B)$$



# 矩阵分解

## 特征分解eigendecomposition

特征分解（也称为谱分解）将<u>可对角化矩阵</u>分解为其特征值和特征向量的矩阵之积。

$$A$$是一个$$n$$阶方阵，且有$$N$$个线性独立的特征向量$$\pmb p_i,\ i=1,2,\cdots,N$$。这样$$A$$可以分解为
$$
A=P\Lambda P^{-1}
$$
其中$$P$$为$$n$$阶方阵，且其第$$i$$列为特征向量$$\pmb p_i,\ i=1,2,\cdots,N$$；$$\Lambda$$为对角阵，其对角线上元素为特征值。



@特征分解矩阵
$$
A=\begin{bmatrix}1&2&0\\0&3&0\\2&-4&2
\end{bmatrix}
$$

$$
|\lambda I-A|=\begin{vmatrix}\lambda-1&-2&0\\0&\lambda-3&0\\-2&4&\lambda-2
\end{vmatrix}=(\lambda-1)(\lambda-2)(\lambda-3)\\
\lambda_1=1,\lambda_2=2,\lambda_3=3\\
\lambda_1I-A=\begin{bmatrix}0&-2&0\\0&-2&0\\-2&4&-1 
\end{bmatrix},\ v_1=[-1,0,2]^{\rm T}\\
\lambda_2I-A=\begin{bmatrix}1&-2&0\\0&-1&0\\-2&4&0 
\end{bmatrix},\ v_2=[0,0,1]^{\rm T}\\
\lambda_3I-A=\begin{bmatrix}2&-2&0\\0&0&0\\-2&4&1 
\end{bmatrix},\ v_3=[-1,-1,2]^{\rm T}\\
P=[v_1,v_2,v_3]\\
AP=A[v_1,v_2,v_3]=[\lambda_1v_1,\lambda_2v_2,\lambda_3v_3]=[v_1,v_2,v_3]{\rm diag(\lambda_1,\lambda_2,\lambda_3)}=P\Lambda \Rightarrow\\
P^{-1}AP=\Lambda
$$



**实对称矩阵**

$$n$$阶实对称矩阵$$A$$可被分解成
$$
A=Q\Lambda Q^{\rm T}
$$
其中$$Q$$为正交矩阵，$$\Lambda$$为实对角矩阵。

**正规矩阵**

$$n$$阶正规矩阵$$A$$可被分解成
$$
A=U\Lambda U^{\rm H}
$$
其中$$U$$为酉矩阵。若$$A$$为埃尔米特矩阵，则$$\Lambda$$为实对角矩阵。



## 若尔当标准型Jordan normal form

对于不可对角化矩阵，我们可以将其相似到一个接近于对角矩阵的矩阵，称为若尔当标准型。

对于域$$\mathbb{K}$$上的$$n$$阶方阵$$A$$，只要其特征值都在$$\mathbb{K}$$中，就存在一个可逆矩阵$$P$$，使得
$$
A=PJP^{-1}
$$
其中若尔当标准型$$J$$满足：

+ 对角元素为$$A$$的特征值
+ 特征值$$\lambda_i$$的几何重数是它的所有若尔当块的个数
+ 特征值$$\lambda_i$$的代数重数是它的所有若尔当块的维数之和



@求矩阵的若尔当标准型
$$
A=\begin{bmatrix}5&4&2&1\\0&1&-1&-1\\-1&-1&3&0\\1&1&-1&2
\end{bmatrix}
$$

$$
|\lambda I-A|=\begin{vmatrix}\lambda-5&-4&-2&-1\\0&\lambda-1&1&1\\1&1&\lambda-3&0\\-1&-1&1&\lambda-2
\end{vmatrix}=(\lambda-1)(\lambda-2)(\lambda-4)^2\\
\lambda_1=1,\lambda_2=2,\lambda_3=4\\
\lambda_1I-A=\begin{bmatrix}-4&-4&-2&-1\\0&0&1&1\\1&1&-2&0\\-1&-1&1&-1 
\end{bmatrix},\ v_{1}=[1,-1,0,0]^{\rm T},\\
\lambda_2I-A=\begin{bmatrix}-3&-4&-2&-1\\0&1&1&1\\1&1&-1&0\\-1&-1&1&0 
\end{bmatrix},\ v_{2}=[1,-1,0,1]^{\rm T},\\
\lambda_3I-A=\begin{bmatrix}-1&-4&-2&-1\\0&3&1&1\\1&1&1&0\\-1&-1&1&2 
\end{bmatrix},\ v_{31}=[-1,0,1,-1]^{\rm T},\\
几何重数1未达到代数重数2，因此不能对角化\\
\begin{bmatrix}-1&-4&-2&-1\\0&3&1&1\\1&1&1&0\\-1&-1&1&2 
\end{bmatrix}v_{311}=[-1,0,1,-1]^{\rm T},\ v_{311}=[1,0,0,0]^{\rm T}\\
P=[v_1,v_2,v_{31},v_{311}]\\
P^{-1}AP=\begin{bmatrix}1&0&0&0\\0&2&0&0\\0&0&4&1\\0&0&0&4 
\end{bmatrix}
$$

各特征值的几何重数和代数重数如图

![](https://i.loli.net/2020/12/07/bsjwugixI56tAqD.png)



## 奇异值分解singular value decomposition

对于域$$\mathbb{K}$$上的$$m\times n$$矩阵$$A$$，存在奇异值分解
$$
A=U\Sigma V^{\rm H}
$$
其中$$U$$是$$m$$阶酉矩阵，$$\Sigma$$是$$m\times n$$非负实数对角矩阵，$$V$$是$$n$$阶酉矩阵，$$\Sigma$$对角线上的元素称为$$A$$的奇异值。常见的做法是将奇异值由大到小排列，这样$$A$$唯一确定一个$$\Sigma$$。



@对以下矩阵进行奇异值分解
$$
M=\begin{bmatrix}1&0&0&0&2\\0&0&3&0&0\\0&0&0&0&0\\0&4&0&0&0
\end{bmatrix}
$$

$$
U=\begin{bmatrix}0&0&1&0\\0&1&0&0\\0&0&0&1\\1&0&0&0
\end{bmatrix},\Sigma=\begin{bmatrix}4&0&0&0&0\\0&3&0&0&0\\0&0&\sqrt{5}&0&0\\0&0&0&0&0
\end{bmatrix},V^{\rm H}=\begin{bmatrix}0&1&0&0&0\\0&0&1&0&0\\\sqrt{0.2}&0&0&0&\sqrt{0.8}\\0&0&0&1&0\\\sqrt{0.8}&0&0&0&-\sqrt{0.2}
\end{bmatrix}
$$



# 酉空间





