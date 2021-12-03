# 代数系统

## 运算operation

**二元运算automorphism** $f:S\times S → S$，需要满足：

1. S中任意两个元素都可以进行该运算，且结果唯一
2. S对该运算封闭

**一元运算** $f:S→S$ 
$$
交换律\quad x*y=y*x\\
结合律associative~law\quad (x*y)*z=x*(y*z)\\
幂等律\quad x*x=x\\
左分配律\quad x*(y\circ z)=(x*y)\circ (x*z)\\
右分配律\quad (y\circ z)*x=(y*x)\circ (z*x)\\
吸收率\quad x*(x\circ y)=x,x\circ(x*y)=x\\
单位元identity\quad e*x=x*e=x\\
零元\quad \theta*x=x*\theta=\theta\\
逆元inverse\quad x*y=y*x=e\\
消去律\quad x*y=x*z|y*x=z*x,x\neq \theta\Rightarrow y=z
$$
**定理** 设*为S上的二元运算， $e_l,e_r$ 分别为左单位元和右单位元，则 $e_l=e_r=e$，e为唯一的单位元

**定理** 设*为S上的二元运算， $\theta_l,\theta_r$ 分别为左零元和右零元，则 $\theta_l=\theta_r=\theta$， $\theta$ 为唯一的零元

**定理** 设*为S上可结合的二元运算，对于x存在 $y_l,y_r$ 分别为左逆元和右逆元，则 $y_l=y_r=y$，y为唯一的逆元



## 代数系统

**代数系统** 非空集合S和S上k个一元或二元运算 $f_1,f_2,\cdots,f_k$ 组成的系统称为一个代数系统，记作 $\langle S,f_1,f_2,\cdots,f_k \rangle$ . 如 $\langle \mathbb{Z},+,0 \rangle,\langle P(S),\bigcup,\bigcap,\sim,\varnothing,S \rangle$ 

**同类型代数系统** 两个代数系统中运算的个数相同，对应运算的元数相同，且代数常熟个数相同. 如 $V_1=\langle \mathbb{R},+,\cdot,-,0,1\rangle和V_2=\langle P(B),\bigcup,\bigcap,\sim,\varnothing,B\rangle$ 



**格** $V=\langle S,*,\circ\rangle$，其中 $*和\circ$ 是满足交换律，结合律，幂等律和吸收律的二元运算

**子代数系统** 代数系统 $V=\langle S,f_1,f_2,\cdots,f_k \rangle$， $B \subseteq S$，B对 $f_1,\cdots,f_k$ 封闭，且含有与S相同的常数，则B是V的子代数系统

**积代数** 同类型代数系统 $V_1=\langle A,\circ \rangle,V_2=\langle B,*\rangle$，在集合 $A\times B$ 上定义如下二元运算 $\cdot$  $\forall \langle a_1,b_1 \rangle,\langle a_2,b_2 \rangle \in A\times B$ 有
$$
\langle a_1,b_1 \rangle\cdot \langle a_2,b_2 \rangle=\langle a_1\circ a_2,b_1*b_2 \rangle
$$
称 $V=\langle A\times B, \cdot\rangle$ 为V1和V2的积代数，记作 $V_1\times V_2$ 

**定理** $V_1=\langle A,\circ \rangle,V_2=\langle B,*\rangle,V_1\times V_2=\langle A\times B,\cdot \rangle$，

1. 如果 $\circ, *$ 是可交换|可结合|幂等的，则 $\cdot$ 也是可交换|可结合|幂等的
2. 如果e1和e2| $\theta_1,\theta_2$ 分别为 $\circ,*$ 运算的单位元|零元，那么 $\langle e_1,e_2\rangle|\langle \theta_1,\theta_2\rangle$ 也是 $\cdot$ 运算的单位元|零元
3. 如果x,y分别为 $\circ,*$ 运算的可逆元素，则 $\langle x,y\rangle$ 也是 $\cdot$ 运算的可逆元素，其逆元为 $\langle x^{-1},y^{-1}\rangle$ 



## 同态和同构

**同态，同构** $V_1=\langle A,\circ \rangle,V_2=\langle B,*\rangle$ 同类， $f:A→B$ 且 $\forall x,y\in A$ 有 $f(x\circ y)=f(x)*f(y)$，则称f是V1到V2的同态映射，简称同态；如果是满射，则称为同构，记作 $V_1\cong V_2$ 

如果 $\circ$ 具有交换律，结合律，幂等律，则*也具有相同性质； $f(e_1)=e_2,f(\theta_1)=\theta_2,f(x^{-1})=f(x)^{-1}$ 

同构是代数系统集合中的<u>等价关系</u>



# 群group

## 群的定义

**半群semigroup** $V=\langle S,*\rangle$，其中*是可结合的二元运算

**幺半群|独异点** $V=\langle S,*,e\rangle$，其中e是单位元

**群group** $G=\langle S,*\rangle$ 是幺半群，且 $\forall a\in S,a^{-1}\in S$ 

> 证明群：封闭性，结合律，单位元，逆元存在

**有限群，无限群，阶** 群G为有穷集|无穷集，G的基数称为阶

**平凡群** 只定义单位元

**交换群|Abel群Abelian group** 群G的二元运算可交换

**幂** $a\in G$，定义
$$
a^n=\begin{cases}
e,&n=0\\
a^{n-1}a,&n>0\\
(a^{-1})^m,&n<0,n=-m
\end{cases}
$$
**阶** $a\in G$ 使得等式 $a^k=e$ 成立的最小正整数k称为a的阶，称a为k阶元，记作 $|a|=k$ ；若不存在这样的k则称a为无限阶元

**定理**
$$
\begin{align}
&\forall a\in G,(a^{-1})^{-1}=a\\
&\forall a,b \in G, (ab)^{-1}=b^{-1}a^{-1}\\
&\forall a\in G,a^ma^n=a^{m+n}\\
&\forall a\in G,(a^m)^n=a^{mn}\\
若G为交换群，&(ab)^n=a^nb^n

\end{align}
$$
**消去律** 群G必然满足消去律，即 $\forall a,b,c\in G$，

1. $ab=ac\Rightarrow b=c$ 
2. $ba=ca\Rightarrow b=c$ 

**定理** 群G， $a\in G,~|a|=r$，k为整数，则

1. $a^k=e$ 当且仅当 $r|k$ 
2. $|a^{-1}|=|a|$ 





## 子群与群的陪集分解

**子群** 群G，H是G的非空子集，且关于G中的运算构成群，则H是G的子群，记作 $H\le G$ ；H是G的真子群记作 $H<G$ 

**子群判定定理一** $H\le G$ 当且仅当

1. $\forall a,b \in H,ab\in H$ 
2. $\forall a \in H,a^{-1}\in H$ 

**子群判定定理二** $H\le G$ 当且仅当 $\forall a,b \in H,ab^{-1}\in H$ 

**子群判定定理三** H是有穷集， $H\le G$ 当且仅当 $\forall a,b \in H,ab\in H$ 



**生成子群** $a\in G$，令 $H=\{a^k|k\in \mathbb{Z}\}$，则 $H\le G$，称为a生成的子群，记作 $\langle a \rangle$ 

**中心center** 群G，令 $C=\{ a|a\in G \land\forall x \in G:ax=xa  \}$，则 $C\le G$，称为G的中心



**子群格** 群G，令 $S=\{H|H\le G \}$，在S上定义关系R：
$$
\forall A,B\in S,ARB\Leftrightarrow A\le B\\
$$
则 $\langle S,R\rangle$ 构成偏序集，称为群G的子群格



**陪集coset** $H\le G,a\in G$，令
$$
Ha=\{ha|h\in H \}
$$
称Ha是子群H在G中的右陪集，a为代表元素

**定理** $H\le G$，

1. $He=H$ 
2. $\forall a\in G,a\in Ha$ 

**定理** $H\le G,~\forall a,b \in G$ 有
$$
a=Hb\Leftrightarrow ab^{-1}\in H \Leftrightarrow Ha=Hb
$$

> 右陪集中任何元素都可以作为代表元素

**定理** $H\le G$，定义二元关系R， $\forall a,b \in G$，
$$
\langle a,b \rangle \in R\Leftrightarrow ab^{-1}\in R
$$
则R是G上的等价关系，且 $[a]_R=Ha$ 

**推论** $H\le G$，

1. $\forall a,b \in G,Ha=Hb或Ha\bigcap Hb=\varnothing$ 
2. $\bigcup \{Ha|a\in G \}=G$ 

> 即给定群G的一个子群H，集合 $\{Ha|a\in G\}$ 恰好构成G的一个划分

类似地可以定义H的左陪集



**正规子群|不变子群invariant subgroup** $H\le G$， $\forall a\in G,aH=Ha$，则H为G的正规子群，记作 $H\unlhd G$ . 任何群的平凡子群 $\{e\}$ 和G都是正规的

**指数**  H在G中的陪集数称为指数，记作 $[G:H]$ 

**拉格朗日定理** 有限群G， $H\le G$，则
$$
|G|=|H|\cdot [G:H]
$$
**推论** 

+ $|G|=n$，则 $\forall a\in G,~|a|$ 是n的因子，且 $a^n=e$ 

+ $|G|$ 为素数，则 $\exists a\in G,G=\langle a\rangle$ ；G为Abel群

  



## 循环群与置换群

**循环群cyclic group** $\exists a\in G$ 使得 $G=\langle a \rangle$，则称G是循环群，a是生成元；若a是n阶元，则G为**n阶循环群**；若a为无限阶元，则G为**无限循环群**

**定理** 设循环群 $G=\langle a \rangle$ 

+ 若G是无限循环群，则G只有2个生成元： $a和a^{-1}$ 
+ 若G是n阶循环群，则G含有 $\phi(n)$ 个生成元，对于任何小于n且与n互素的自然数r， $a'$ 是G的生成元

**定理** 

1. $G=\langle a\rangle$ 是循环群，则G的子群仍是循环群
2. $G=\langle a\rangle$ 是无限循环群，则G的子群除{e}以外都是无限循环群
3. $G=\langle a\rangle$ 是n阶循环群，则对于n的每个正因子d，G恰好有一个d阶子群

> 求循环群子群方法：
>
> 无限循环群 $G=\langle a\rangle$，子群 $\langle a^m\rangle,m\in \mathbb{N}$ 
>
> n阶循环群 $G=\langle a\rangle$，对于n的每个正因子d， $\langle a^{n/d}\rangle$ 是唯一d阶子群



e.g. 置换 $g=(1234)$，则 $G=\langle g \rangle$，是4阶循环群



**n元置换permutation** $S=\{1,2,\cdots,n \}$，S上任何双射函数 $\sigma:S→S$ 称为S上的n元置换；n元置换 $\sigma,\tau$， $\sigma \circ\tau$ 也是n元置换，称为 $\sigma和\tau$ 的**乘积**

> 每个n元置换即对应一个排列

**k阶轮换** $S=\{1,2,\cdots,n\}$ 上的n元置换 $\sigma$ 满足
$$
\sigma(i_1)=i_2,\sigma(i_2)=i_3,\cdots,\sigma(i_k)=i_1
$$
而对其余元素保持不变，则 $\sigma$ 为S上的k阶轮换，记作 $(i_1i_2\cdots i_k)$ . 若k=2，则称为**对换**

**奇置换，偶置换** n元置换可以表示成奇数|偶数个对换之积；奇置换数=偶置换数

**n元对称群** 所有n元置换构成的集合 $S_n$ 关于置换的乘法构成一个群，称为n元对称群

**n元交错群** 所有n元偶置换的集合 $A_n$， $A_n$ 是 $S_n$ 的子群

**n元置换群** $S_n$ 的所有子群统称为n元置换群



e.g. 3元对称群 $S_3$ 

![svijhnuj46oy53tfwokijvgrw](C:\Users\Xiao Yuxuan\Documents\pic\svijhnuj46oy53tfwokijvgrw.PNG)

子群： $S_3,A_3=\{(1),(123),(132) \},\{(1),(12)\},\{(1),(13)\},\{(1),(23)\},\{(1)\}$ 



**Polya定理** 设 $N=\{1,2,\cdots,n\}$ 是被着色物体的集合， $G=\{\sigma_1,\sigma_2,\cdots,\sigma_g \}$ 是N上的置换群，用m种颜色对N种元素进行着色，则在G作用下的着色方案数是
$$
M=\frac{1}{|G|}\sum_{k=1}^g m^{c(\sigma_k)}
$$
其中 $c(\sigma_k)$ 是置换 $\sigma_k$ 的轮换表达式中包含1-轮换在内的轮换个数



# 群的同态

## 同构isomorphism

**同构isomorphism** 群 $\langle G,\circ \rangle,\langle H,*\rangle$， $f:G→H$ 是双射且 $\forall x,y\in G$ 有 $f(x\circ y)=f(x)*f(y)$，则称f是G到H的同构映射，简称同构，记作 $G\cong H$ 

**定理** 任意n阶循环群都同构于 $\langle \mathbb{Z_n},+ \rangle$ 

**定理** 任意无限循环群都同构于 $\langle \mathbb{Z},+ \rangle$ 

**定理** $G\cong H$，G是循环群，则H也是循环群

**定理** 双射 $f:A→B$，群 $\langle A,\cdot \rangle$，定义B上的二元运算 $x*y=f(f^{-1}(x)\cdot f^{-1}(y))$，则 $\langle B,* \rangle$，且f是群同构映射



## 可逆变换

**可逆变换** 群 $\langle G,\cdot \rangle$，将G到G的可逆映射称为G上可逆变换，G上所有可逆变换在映射合成之下构成群，记作 $I(G)$ 

**自同构automorphism** 群G到G自身的同构映射称为G的自同构

**定理** G的所有自同构的集合 ${\rm Aut}(G)$ 是 $I(G)$ 的一个子群



e.g. 群 $\langle \mathbb{Z_4},\oplus \rangle$ 
$$
I(G)=\{(0,1,2,3),(0,1,3,2),\cdots \}共24项\\
Aut(G)=\{(0,1,2,3),(0,3,2,1) \}\\
L=\{(0,1,2,3),(1,2,3,0),(2,3,0,1),(3,0,1,2) \}
$$


**左乘变换** 群G， $a\in G$，定义变换 $\lambda_a$，即
$$
\lambda_a(x)=ax,\quad x\in G
$$
则 $\lambda_a$ 是G上的可逆变换，称为左乘变换

**定理** 群G，G中所有元素的左乘变换的集合 $L=\{\lambda_a|a\in G\}$ 是 $I(G)$ 的一个子群

> L共有 $|G|$ 项，即乘法表中的每一行

**定理** 群G，其左乘变换集合L，则 $G\cong L$ 

**Cayley定理** 每个群G都同构于其上所有可逆变换构成的群 $I(G)$ 的一个子群

**推论** 每个n阶有限群必定同构于n阶对称群 $S_n$ 的一个子群



e.g. 群 $\langle \mathbb{Z_4},\oplus \rangle$ 同构于 $S_4$ 中 $(2,3,4,1)$ 生成的循环群



**内自同构inner automorphism** 群G， $a\in G$，通过a可导出映射 $\gamma:G→G$ :
$$
\gamma_a(x)=axa^{-1},\quad x\in G
$$
则 $\gamma_a$ 必为G到G的同构映射，称为a导出的内自同构

**不变子集** 映射 $f:A→A$，T是A的子集， $f(T)\subseteq T$，则T是f的不变子集

**不变子群|正规子群normal subgroup** 群G，H是G的子群， $\forall a\in G,\forall h\in H$，都有
$$
aha^{-1}\in H
$$
则H是G的不变子群或正规子群，记作 $H\unlhd G$ 

> $G\unlhd G,\{e\}\unlhd G$，称为平凡的不变子群
>
> $\{(1),(1,2,3),(1,3,2) \}$ 是S_3的不变子群

**定理** $H\unlhd G$ 当且仅当 $\forall a\in G$， $aH=Ha$ 

> $H\unlhd G,aH=H\Leftrightarrow a\in H$ 

**定理** N, H都是群G的不变子群，则NH也是G的不变子群

**定理** $N_1,N_2,\cdots,N_k$ 都是群G的不变子群，则 $N=\bigcap_i N_i$ 也是G的不变子群

> N, H都是群G的子群，NH不一定是G的子群，如S_3的子群 $N=\{(1),(1,2)\},H=\{(1),(1,3)\},NH=\{(1),(1,2),(1,3),(1,3,2)\}$ 

> 子群的子群是子群；不变子群的不变子群



**换位子群** 群G，所有形如 $a_1b_1a_1^{-1}b_1^{-1}\cdots a_nb_na_n^{-1}b_n^{-1},n=1,2,\cdots$ 的集合称为G的换位子群，换位子群是不变子群

**单群** 不含非平凡的不变子群的群称为单群



## 同态homomorphism

> 同构犹如两群互为复制品，大小结构完全相同
>
> 同态则为群到群的映射，保持运算，但像比原像通常要小

**同态homomorphism** 群 $\langle G,\circ \rangle,\langle H,*\rangle$， $f:G→H$， $\forall x,y\in G$ 有 $f(x\circ y)=f(x)*f(y)$，则称f是G到H的同态映射，简称同态

**定理** 同态映射 $f:G→H$，则 $f(e_G)=e_H,f^{-1}(a)=f(a^{-1})$ 

**核kernel** 同态映射 $f:G→H$，H中 $e_H$ 的原像 $f^{-1}(e_H)$ 是G的<u>不变子群</u>，称为核，记作Ker(f)



**定理** 同态映射的复合是同态映射

**定理** 同态映射 $f:G→H,g:H→K$，则
$$
Ker(gf)=f^{-1}(Ker(g))\\
Img(gf)=g(Img(f))
$$
**定理** 同态映射 $f:G→H$，若A是G的子群，则 $f(A)$ 是H的子群；若B是H的子群，则 $f^{-1}(A)$ 是G的子群

**同态像** 同态映射 $f:G→H$ 是满射，则G同态于H，H是G的同态像

**定理** <u>满</u>同态映射 $f:G→H$，若A是G的不变子群，则 $f(A)$ 是H的不变子群；若B是H的不变子群，则 $f^{-1}(A)$ 是G的不变子群

**定理** 同态映射 $f:G→H$ 是单射当且仅当 $Ker(f)=\{e_G\}$ 

**推论** 满同态映射 $f:G→H$ 是同构映射当且仅当 $Ker(f)=\{e_H\}$ 

**定理** 同态映射 $f:G→H$，

1. B为H的子群，则 $f(f^{-1}(B))=B\bigcap Img(f)$ 
2. A为G的子群，则 $f^{-1}(f(A))=AKer(f)$ 



e.g. G=<Z,+>, H=<Z_6,O+>, f=x mod 6,
$$
A=\{\cdots,-2,0,2,\cdots \},\quad f(A)=\{0,2,4\}是H的(不变)子群\\
B=\{1,3,5\},\quad f^{-1}(B)=\{\cdots-1,1,3,5,\cdots\}是G的(不变)子群\\
f(f^{-1}(B))=\{1,3,5\}\\
f^{-1}(f(A))=\{\cdots,-2,0,2,\cdots \}
$$



## 商群quotient group

群G的不变子群N，等价关系 $a\sim b$ 当且仅当 $a^{-1}b\in N$ 

**商群quotient group** 群<G,*>的不变子群N， $G/N$ 代表G对N的所有陪集构成的集合，规定 $\forall aN,bN\in G/N$，运算#
$$
aN\#bN=(a*b)N
$$
且 $(G/N,\#)$ 是个群，称为G对N的商群



e.g. 

![op3jygvrnj42grvth](C:\Users\Xiao Yuxuan\Documents\pic\op3jygvrnj42grvth.PNG)

![dajiph6u4nhtjgogr2](C:\Users\Xiao Yuxuan\Documents\pic\dajiph6u4nhtjgogr2.PNG)



![grwji6uwrfwvwrfgh](C:\Users\Xiao Yuxuan\Documents\pic\grwji6uwrfwvwrfgh.PNG)

![gtjihn4u76fwrfgwht](C:\Users\Xiao Yuxuan\Documents\pic\gtjihn4u76fwrfgwht.PNG)

![2t48jih3u6nhjvw3rf1](C:\Users\Xiao Yuxuan\Documents\pic\2t48jih3u6nhjvw3rf1.PNG)

![vwfrjinojy43ygr2ioj24](C:\Users\Xiao Yuxuan\Documents\pic\vwfrjinojy43ygr2ioj24.PNG)

![wrfiphk4gnjh3t3j6](C:\Users\Xiao Yuxuan\Documents\pic\wrfiphk4gnjh3t3j6.PNG)



**自然同态** 群G，不变子群N，则映射 $f:G→G/N,f(a)=aN$ 是满同态映射，且 $Ker(f)=N$，称为自然同态

**同态基本定理fundamental theorem of homomorphism** 群G，H， $f:G→H$ 是满同态映射， $Ker(f)=K$，则有映射 $\varphi:G/K→H$ 使得
$$
\forall aK\in G/K,\varphi(aK)=f(a)
$$
且 $\varphi$ 是G/K到H的同构映射



e.g. G=<Z_6,O+>, N=<0,2,4,O+>, G/N=<024,135>, 则
$$
f是满同态映射(同态由商群定义给出)\\
Ker(f)=\{024\}
$$
e.g. G=<Z,+>, H=<Z_6,O+>, f=x mod 6, Ker(f)=<6k>=K, G/K=<[0],[1],[2],[3],[4],[5]>
$$
\forall aK\in G/K, \varphi(aK)=f(a)=a [mod~6],且\varphi为同构映射
$$







# 环ring

## 环的定义

**环ring** 代数系统 $\langle R,+,\cdot \rangle$， $+和\cdot$ 是二元运算，若满足

1. $\langle R,+\rangle$ 构成交换群（可交换
2. $\langle R,\cdot\rangle$ 构成半群（可结合
3. $\cdot$ 关于+适合分配律

则 $\langle R,+,\cdot \rangle$ 是一个环；+称为环加， $\cdot$ 称为环乘；环加单位元记作0，逆元记作-x，n次幂记作nx，环乘单位元记作1，逆元记作 $x^{-1}$，n次幂记作 $x^n$ 

> 环加单位元零元在环乘中满足 $0\cdot a=a\cdot 0=0$ ；只含有0的环称为**零环zero ring**
>
> 环加的逆元一定存在（<R,+>构成群），但环乘的逆元不一定存在

**定理** 环 $\langle R,+,\cdot \rangle$ ：

1.  $\forall a \in R,a0=0a=0$ 
2.  $\forall a,b \in R,(-a)b=a(-b)=-ab$ 
3.  $\forall a,b,c\in R,a(b-c)=ab-ac,(b-c)a=ba-ca$ 
4.  $\forall a_1,\cdots,a_n,b_1,\cdots,b_m\in R,(\sum a_i)(\sum b_j)=\sum \sum a_ib_j$ 



e.g.模n整数环，实矩阵环



**零环zero ring** 



**整环，域** 环 $\langle R,+,\cdot\rangle$ ：

1. 若环乘适合交换律，则R是**交换环**
2. 若环乘存在单位元，则R是含幺环
3. 若 $\forall a,b\in R,ab=0\Rightarrow a=0\lor b=0$，则R是无零因子环
4. 若R同时是交换环，含幺环，无零因子环，则R是整环
5. 若R是整环，且R中至少有2个元素， $\forall a\in R^*=R-\{0\},a^{-1}\in R$，则R是域

| $\mathbb{Z}$ | $2\mathbb{Z}$    | 实矩阵集合 | $\mathbb{Z}_6$ |
| -------------- | ------------------ | ---------- | ---------------- |
| 整环           | 交换环，无零因子环 | 含幺环     | 交换环，含幺环   |
| $\mathbb{R}$ |                    |            |                  |
| 域             |                    |            |                  |

**定理** 整环的环乘满足消去律



## 子环subring

**子环subring** 环 $\langle R,+,\cdot \rangle$，S是R的非空子集，若将 $+,\cdot$ 派生到S上， $\langle S,+,\cdot \rangle$ 也是个环，则其为 $\langle R,+,\cdot \rangle$ 的子环

**判定** 

1. 环 $\langle R,+,\cdot \rangle$，S是R的非空子集，则S是R的子环当且仅当 $\forall a,b\in S, -a, a+b,a\cdot b\in S$ 
2. 环 $\langle R,+,\cdot \rangle$，S是R的非空子集，则S是R的子环当且仅当 $\forall a,b\in S, a\cdot b\in S$，且 $\langle S,+\rangle$ 是 $\langle R,+\rangle$ 的子群
3. 环 $\langle R,+,\cdot \rangle$，S是R的非空子集，则S是R的子环当且仅当 $\forall a,b\in S, a-b,a\cdot b\in S$ 
4. $S_i,i \in I$ 都是环R的子环，则 $S=\bigcap_{i\in I}S_i$ 也是R的子环

**生成子环** 环R， $a\in R$，R的子环族 $A=\{S是R的子环|a \in S \}$，则 $\bigcap_{S\in A}S$ 称为元素a生成的子环，记作 $\langle a \rangle$ . 即 $\langle a \rangle$ 使包含a的子环中的最小者；同理可以由R的非空子集T生成子环，记作 $\langle T\rangle$ 

**推论** 环R， $a\in R$，则R中形如 $ma,m_1a+m_2a^2,\cdots,m_1a+m_2a^2+\cdots+m_ta^t,\cdots$ 的元素（正整数t，整数m_i）构成的集合即为 $\langle a \rangle$ 

环R， $a,b\in R$，则R中形如 $ma+nb,m_1a^2+m_2ab+n_1ba+n_2b^2,\cdots$ 的元素（整数m_i,n_i）构成的集合即为 $\langle a,b \rangle$ 



## 理想ideal

**理想idealイデアル** 环 $\langle R,+,\cdot\rangle$，A是R的非空子集，若

1. $\langle A,+\rangle$ 是 $\langle R,+\rangle$ 的子群
2. $\forall a\in A,\forall x\in R,xa\in A,ax\in A$ 

则称A是R的理想

> R和{0}是R的理想，称为平凡理想

**定理** $A_1,A_2,\cdots,A_k$ 都是环R的理想，则 $A=\bigcap_i A_i$ 也是R的理想

**生成理想** 环R，T是R的非空子集，令 $B=\{I是R的理想，T\subseteq I \}$ 得到的理想 $\bigcap_{I\in B}I$ 称为R的由子集T生成的理想，记作 $(T)$ ；特别地，如果T只有一个元素a，则记作(a)

**定理** T是环R的非空子集，R中所有形如下式的元素集合即为(T)
$$
\{\sum na+\sum xa+\sum ay+ \sum xay\}
$$
其中a为T中元素，n为整数，xy为R中元素

**推论** 环R， $a\in R$，形如下式的元素集合即为(a)
$$
\{na+\sum xa+\sum ay+\sum xay \}
$$
若 $e \in R$，则化简为
$$
\{\sum xay \}
$$
**定理** A，B是环R的理想，则 $A+B=A\bigcup B$，其中
$$
A+B=\{x\in R | x=a+b,a\in A,b \in B \}
$$

>  A=[4], B=[6], 则 $A+B=A\bigcup B=[2]$ 
>
>  $(a_1,a_2,\cdots,a_n)$ 即 $(a_1)+(a_2)+\cdots+(a_n)$ 



## 理想与商环

**商环|剩余环quotient ring商環** 环 $\langle R,+,\cdot \rangle$，A是R的理想. 作为群，得商群 $\overline{R}=R/A$，加法为#，再定义乘法 $\odot$ ：
$$
(a+A)\odot(b+A)=ab+A\\
(a+A)\#(b+A)=a+b+A\\
$$
则 $(\overline{R},\#,\odot)$ 是环R关于A的商环

**单环|单纯环** 只有平凡理想的环称为单环



e.g. 整数环的商环
$$
\mathbb{Z/Z}=\{\mathbb{Z}\}\\
\mathbb{Z/\{0\}}=\{\cdots,\{-1\},\{0\},\{1\},\cdots\}\\
\mathbb{Z/(5)}=\{[0],[1],[2],[3],[4] \}\\
\mathbb{Z/(4,6)}=\mathbb{Z/(gcd(4,6))}=\mathbb{Z/(2)}
$$

> 商环也是对环的一种划分

运算表

![g25kpyo6uktnfqje25y](C:\Users\Xiao Yuxuan\Documents\pic\g25kpyo6uktnfqje25y.PNG)

实系数多项式环的商环
$$
\mathbb{P/P}=\{\mathbb{P}\}\\
\mathbb{P/\{0\}}=\{\{0\},\{1\},\{\pi\},\cdots,\{X\},\{2X\},\{\pi X\},\cdots\}=\{\{p\}|p\in \mathbb{P}\}\\
\mathbb{P/(2)}=\mathbb{P/P}=\{\mathbb{P}\}\\
\mathbb{P/}(X)=\{\{Xp+C\}|C\in \mathbb{R},p\in \mathbb{P}\}
$$
整系数多项式环的商环
$$
\mathbb{P/P}=\{\mathbb{P}\}\\
\mathbb{P/\{0\}}=\{\{0\},\{1\},\{2\},\cdots,\{X\},\{2X\},\{3X+1\},\cdots\}=\{\{p\}|p\in \mathbb{P}\}\\
\mathbb{P/(2)}=\mathbb{P/\{2p|p\in \mathbb{P}\}}=\{\{2p\},\{2p+1\},\{2p+X\},\cdots,\{2p+X+1\},\cdots\}\\
\mathbb{P/}(X)=\mathbb{P/}\{Xp|p\in \mathbb{P}\}=\{\{Xp+Z\}|Z\in \mathbb{Z},p\in \mathbb{P}\}\\
\mathbb{P/}(2,X)=\mathbb{P/}\{2p_1+Xp_2|p\in \mathbb{P}\}=\{\{Xp|常数项为奇数\},\{Xp|常数项为偶数\}\}
$$

> $\mathbb{Z/(5)}=\{[0],[1],[2],[3],[4] \}\\$，表示任意整数除以5余数为0,1,2,3或4，每一余数组成一陪集
>
> 而 $\mathbb{P/}(X)=\{\{Xp+Z\}|Z\in \mathbb{Z},p\in \mathbb{P}\}$ 表示任意整系数多项式除以x余数为任意整数，同样每一整数组成一陪集

> $\mathbb{P/}(2,X)=\mathbb{P/}\{2p_1+Xp_2|p\in \mathbb{P}\}=\{\{Xp|常数项为奇数\},\{Xp|常数项为偶数\}\}$，表示任意整系数多项式先除以X降次到1，再除以2，得到余数0,1



> 实数环是1维的，m×n实矩阵环是mn维的，实系数多项式环是无限维的







## 环同态映射

**环同态映射|环同构映射** 环R,S，映射 $\varphi:R→S$ 满足
$$
\varphi(a+b)=\varphi(a)\#\varphi(b)\\
\varphi(a\cdot b)=\varphi(a)\odot\varphi(b)
$$
则 $\varphi$ 为环同态映射；特别地， $\varphi$ 是满射时则S是R的同态像， $\varphi$ 是双射时 $\varphi$ 为环同构映射，记作 $R\cong S$ 



e.g.多项式环到实数环的映射 $\varphi:f(x)→f(a)$ 即环同态映射

Z到<Z_3,O+,O*>的映射 $\varphi(x)=x (\mod 3)$ 



**像|核** 环R，S，环同态映射 $\varphi:R→S$，集合
$$
Img(\varphi)=\{s\in S|\exists r\in R,s=\varphi(r) \}\\
$$
是映射 $\varphi$ 的像，集合
$$
Ker(\varphi)=\{r\in R|\varphi(r)=0 \}
$$
是映射 $\varphi$ 的核

**定理** 环同态映射 $\varphi:R→S$，则 $Img(\varphi)$ 是环S的子环

**定理** 环同态映射 $\varphi:R→S$，则 $Ker(\varphi)$ 是环R的理想

**定理** 环同态映射 $\varphi:R→S$ 是满射，R有恒等元e，则S有恒等元 $\varphi(e)$ 

**定理** 环同态映射 $\varphi:R→S$ 是满射，R可交换，则S可交换

**定理** 环同态映射的复合是同态映射

**定理** A是环R的理想，则 $\varphi:r→r+A$ 是R到R/A的满的同态映射，称为自然映射

**环的同态基本定理** 环同态映射 $\varphi:R→S$ 是满射， $Ker(\varphi)=A$，则 $R/A$ 同构于S



e.g. 多项式环P到实数环R的映射 $\varphi:f(x)→f(a)$ 是满的环同态映射，则 $P/Ker(\varphi)\cong R$，其中 $Ker(\varphi)=((x-a))$ 



# 域

## 除环和域



## 理想和商环



## 嵌入问题



## 交换环上的多项式











e.g.

| $\langle \mathbb{Z}_{12},\oplus \rangle$ | 0    | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    | 9    | 10   | 11   |
| ------------------------------------------ | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 0                                          | 0    | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    | 9    | 10   | 11   |
| 1                                          | 1    | 2    | 3    | 4    | 5    | 6    |      |      |      |      |      |      |
| 2                                          | 2    | 3    | 4    | 5    | 6    |      |      |      |      |      |      |      |
| 3                                          | 3    | 4    | 5    | 6    |      |      |      |      |      |      |      |      |
| 4                                          | 4    | 5    | 6    |      |      |      |      |      |      |      |      |      |
| 5                                          | 5    | 6    |      |      |      |      |      |      |      |      |      |      |
| 6                                          | 6    |      |      |      |      |      |      |      |      |      |      |      |
| 7                                          | 7    |      |      |      |      |      |      |      |      |      |      |      |
| 8                                          | 8    |      |      |      |      |      |      |      |      |      |      |      |
| 9                                          | 9    |      |      |      |      |      |      |      |      |      |      |      |
| 10                                         | 10   |      |      |      |      |      |      |      |      |      |      |      |
| 11                                         | 11   |      |      |      |      |      |      |      |      |      |      |      |

features：

+ 运算表每一行|列都是群中元素的排列

  

$|G|=12$，平凡子群 $G,\{0\}$ 

| \|\|1,11 $\langle 1\rangle$ | 12   | 4,8 $\langle 4\rangle$ | 3    |
| ----------------------------- | ---- | ------------------------ | ---- |
| 2,10 $\langle 2\rangle$     | 6    | 5,7                      | 12   |
| 3,9 $\langle 3\rangle$      | 4    | 6 $\langle 6\rangle$   | 2    |
| 12,0 $\langle 12\rangle$    | 1    |                          |      |

+ $|a^{-1}|=|a|$ 
+ |a|是 $|G|$ 的因子
+ G是12阶循环群，生成元有 $\phi(12)=4$ 个，即1,5,7,11
+ 对于12的因子1,2,3,4,6,12，G都恰有一个该阶的子群



子群格

<img src="C:\Users\Xiao Yuxuan\Documents\pic\hyiet34t9vsfj31rg4u67.PNG" alt="hyiet34t9vsfj31rg4u67" style="zoom:67%;" />



$H=\{0,6\},~[G:H]=6$，其陪集

| H0,H6 | 0,6  | H3,H9  | 3,9  |
| ----- | ---- | ------ | ---- |
| H1,H7 | 1,7  | H4,H10 | 4,10 |
| H2,H8 | 2,8  | H5,H11 | 5,11 |
|       |      |        |      |

+ $a\in Ha$ 

+ 任何元素都可以作为代表元素

+ Ha和Hb相等或无交集因此构成一个G的划分

+ 拉格朗日定理：12=2*6

  

