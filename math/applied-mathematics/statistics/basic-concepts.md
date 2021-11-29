## 基本概念

> 参见
>
> 概率论与数理统计, 陈希孺, P150~158
>
> 数理统计学教程, 陈希孺, 第1章



## 常用统计量

**样本均值(sample mean) $\overline{X}$ **
$$
\overline{X}=\frac{1}{n}(X_1+X_2+\cdots+X_n)
$$
**性质**

+ 若总体 $X\sim N(\mu,\sigma^2)$，则 $\overline{X}\sim N(\mu,\frac{\sigma^2}{n})$ 
+ 若总体不服从正态分布， $E(X)=\mu,Var(X)=\sigma^2$，则由中心极限定理，n较大时 $\overline{X}$ 渐进分布为 $N(\mu,\frac{\sigma^2}{n})$ 



**样本方差(sample variance) $S^2$ **
$$
S^2=\frac{1}{n-1}\sum_{i=1}^n(X_i-\overline{X})^2
$$
上式中的 $n-1$ 称为**自由度**，自由度可以解释为：

1. 共有 $n$ 个 $X_i$，则应该有 $n$ 个自由度，但已有1个自由度用于 $\overline{X}$，故剩余 $n-1$ 个自由度
2. 将 $\overline{X}=\frac{1}{n}\sum_{i=1}^nX_i$ 代入上式，将其整理为二次型 $\sum_{i,j=1}^na_{ij}X_iX_j$，则方阵 $A=(a_{ij})$ 的秩为 $n-1$，即为自由度

前一个解释比较形象，后一个解释则在数学上最严谨。



**样本k阶原点矩sample moment of order k about the origin $a_k$ **
$$
a_k=\frac{1}{n}\sum_{i=1}^nX_i^k
$$


**样本k阶中心矩sample central moment of order k $b_k$ **
$$
b_k=\frac{1}{n}\sum_{i=1}^n(X_i-\overline{X})^k
$$



**分位点** 连续型随机变量的分布函数 $F(x)$， $F(a)=P(X\le a)=\alpha$，称a为该分布的 $\alpha$ 分位点



## 常用统计分布

** $\chi^2$ 分布(chi-squared distribution)** $X_1,X_2,\cdots,X_n$ 相互独立且服从标准正态分布，则随机变量
$$
Y=X_1^2+\cdots+X_n^2
$$
的分布称为自由度为n的 $\chi^2$ 分布，记作 $Y\sim\chi^2(n)$ 

**定理** $X_1,X_2,\cdots,X_n$ 是来自正态总体 $N(\mu,\sigma^2)$ 的样本，则

1. $\overline{X}和S^2$ 相互独立
2. $\overline{X}\sim N(\mu,\sigma^2/n)$ 
3. $\frac{(n-1)S^2}{\sigma^2}\sim\chi^2(n-1)$ 

**性质**：

$X\sim\chi^2(n),则E(X)=n,Var(X)=2n$ 

$X_1\sim\chi^2(m),X_2\sim\chi^2(n)，则X_1+X_2\sim\chi^2(m+n)$ 



**t分布** $X_1\sim N(0,1),X_2\sim \chi^2(n),X_1,X_2$ 相互独立，则随机变量
$$
Y=\frac{X_1}{\sqrt{X_2/n}}
$$
的分布称为自由度为n的t分布，记作 $Y\sim t(n)$ 

> t>30时，t分布基本等同于正态分布



**F分布** $X_1\sim \chi^2(m),X_2\sim \chi^2(n),X_1,X_2$ 相互独立，则随机变量
$$
Y=\frac{X_1/m}{X_2/n}
$$
的分布称为自由度为m与n的F分布，记作 $Y\sim F(m,n)$ 


