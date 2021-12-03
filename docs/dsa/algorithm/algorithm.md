> 参考：
>
> 算法设计与分析（第2版）/屈婉玲



**算法** 算法是有限条指令的序列，是解决某个问题的一系列运算或操作

# 算法时间复杂度

**算法时间复杂度** 基本运算所做运算次数

**基本运算** 比较，位乘……

**输入规模**

**最坏时间复杂度** W(n) 和 **平均时间复杂度** A(n)





# 伪码描述

```
赋值 ←
分支 if ... then ... [else ...]
循环 while, for, repeat until
转向 goto
输出 return
调用 过程名
注释 //
```

伪码不是代码，仅给出算法的主要步骤；允许过程调用





# 函数的阶

$$
o\; 低于 \quad O \; 不高于 \\
\omega \; 高于 \quad \Omega \;不低于\\
\Theta \;同阶\\
$$

阶具有传递性
阶：对数函数<幂函数<指数函数
有限函数之和的阶为最高阶函数的阶

（数学定义，证明略）





# 基本函数类

阶
$$
指数级+ \; n!\\
指数级 \; 2^n, 3^n\\
多项式级 \; n,n^2,n\log n\\
对数多项式级 \;\log n,\log^2n,\log\log n
$$

对数函数
$$
\log_a n=\Theta (\log n)\\
\log n=o(n^\epsilon),\epsilon>0\\
a^{\log_b n}=n^{\log_b a}
$$
指数函数
$$
Stirling公式\; n!=\sqrt{2 \pi n}(\frac{n}{e})^n(1+\Theta(\frac{1}{n}))\\
n!=o(n^n)\\
n!=\omega(2^n)\\
\log(n!)=\Theta(n\log n)
$$
取整函数
$$
\lfloor x \rfloor, \lceil x \rceil\\
x-1< \lfloor x \rfloor \leq x \leq \lceil x \rceil <x+1\\
\lfloor \frac{n}{2} \rfloor +\lceil \frac{n}{2} \rceil =n
$$
阶排序例
$$
2^{2^n}>\\
n!>\\
n2^n>2^n>\\
(\log n)^n=n^{\log \log n}>\\
n^2>\\
\log (n!)=\Theta(n \log n)>\\
n>\\
\log^2 n>\log n>\\
\log \log n>\\
1
$$





# 求和方法

数列求和
$$
\sum_{k=1}^n a_k=\frac{n(a_1+a_n)}{2}\\
\sum_{k=0}^naq^k=\frac{a(1-q^{n+1})}{1-q}\rightarrow \frac{a}{1-q}(q<1)\\
\sum_{k=1}^n \frac{1}{k}=\ln n+O(1)
$$
其他求和方法

差分法 放缩法 放缩积分法





# 递归方程求解

迭代法 差分法 数学归纳法

### 递归树

### 主定理master theorem

$$
对于递推方程\;T(n)=aT(n/b)+O(f(n))\\
(1){\rm if}\;f(n)=O(n^{\log_b a-\epsilon}),\;T(n)=\Theta (n^{\log _b a})\\
{\rm e.g. kd-search}\;T(n)=2T(n/4)+O(1)=O(n^{1/2})\\
---\\
(2){\rm if}\;f(n)=\Omega(n^{\log_b a+\epsilon}),\;T(n)=\Theta (f(n))\\
{\rm e.g.quickselect(aver.)}\;T(n)=T(n/2)+O(n)=O(n)\\
---\\
(3){\rm if}\;f(n)=\Theta(n^{\log_b a }\log^k n),\;T(n)=\Theta (n^{\log_b a }\log^{k+1} n)\\
{\rm e.g.binarysearch}\;T(n)=T(n/2)+O(1)=O(\log n)\\
{\rm mergesort}\;T(n)=2T(n/2)+O(n)=O(n\log n)\\
{\rm STL mergesort}\;T(n)=2T(n/2)+O(n\log n)=O(n\log^2 n)\\
$$

取整函数：数学归纳证明