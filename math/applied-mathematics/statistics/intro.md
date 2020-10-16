## 基础概念

> 参见概率论与数理统计, 陈希孺, P150~158



## 常用统计量

**样本均值(sample mean) $$\overline{X}$$**
$$
\overline{X}=\frac{1}{n}(X_1+X_2+\cdots+X_n)
$$
**性质**

+ 若总体$$X\sim N(\mu,\sigma^2)$$，则$$\overline{X}\sim N(\mu,\frac{\sigma^2}{n})$$
+ 若总体不服从正态分布，$$E(X)=\mu,Var(X)=\sigma^2$$，则由中心极限定理，n较大时$$\overline{X}$$渐进分布为$$N(\mu,\frac{\sigma^2}{n})$$



**样本方差(sample variance) $$S^2$$**
$$
S^2=\frac{1}{n-1}\sum_{i=1}^n(X_i-\overline{X})^2
$$
**定理** 设总体X具有二阶中心矩，$$E(X)=\mu$$，$$Var(X)=\sigma^2<+\infty$$，$$X_1,X_2,\cdots,X_n$$为样本，$$\overline{X}和S^2$$分别是样本均值和样本方差，则$$E(S^2)=\sigma^2$$。 <u>样本方差是总体方差的无偏估计，样本均值是总体均值的无偏估计</u>。



**样本k阶原点矩sample moment of order k about the origin $$a_k$$**
$$
a_k=\frac{1}{n}\sum_{i=1}^nX_i^k
$$


**样本k阶中心矩sample central moment of order k $$b_k$$**
$$
b_k=\frac{1}{n}\sum_{i=1}^n(X_i-\overline{X})^k
$$



## 常用统计分布

**$$\chi^2$$分布(chi-squared distribution)** $$X_1,X_2,\cdots,X_n$$相互独立且服从标准正态分布，则随机变量
$$
Y=X_1^2+\cdots+X_n^2
$$
的分布称为自由度为n的$$\chi^2$$分布，记作$$Y\sim\chi^2(n)$$

$$X\sim\chi^2(n),则E(X)=n,Var(X)=2n$$

$$X_1\sim\chi^2(m),X_2\sim\chi^2(n)，则X_1+X_2\sim\chi^2(m+n)$$

**定理** $$X_1,X_2,\cdots,X_n$$是来自正态总体$$N(\mu,\sigma^2)$$的样本，则

1. $$\overline{X}和S^2$$相互独立
2. $$\overline{X}\sim N(\mu,\sigma^2/n)$$
3. $$\frac{(n-1)S^2}{\sigma^2}\sim\chi^2(n-1)$$



**t分布** $$X_1\sim N(0,1),X_2\sim \chi^2(n),X_1,X_2$$相互独立，则随机变量
$$
Y=\frac{X_1}{\sqrt{X_2/n}}
$$
的分布称为自由度为n的t分布，记作$$Y\sim t(n)$$

> t>30时，t分布基本等同于正态分布



**F分布** $$X_1\sim \chi^2(m),X_2\sim \chi^2(n),X_1,X_2$$相互独立，则随机变量
$$
Y=\frac{X_1/m}{X_2/n}
$$
的分布称为自由度为m与n的F分布，记作$$Y\sim F(m,n)$$



**分位点** 连续型随机变量的分布函数$$F(x)$$，$$F(a)=P(X\le a)=\alpha$$，称a为该分布的$$\alpha$$分位点

