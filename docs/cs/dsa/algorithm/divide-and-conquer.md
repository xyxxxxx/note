# 基本思想

**步骤**

1. 将P归约为k个子问题
2. 递归求解每个子问题
3. 把子问题的解进行综合
4. 给出递归基

**特点**
将原问题归约位规模小的，性质相同的子问题；
子问题规模足够小时可直接求解；
算法可以递归或迭代实现；

# 分析技术

第二类递推方程 $T(n)=aT(n/b)+f(n)$

1. if $f(n)=O(1)$
   $$
   T(n)=\begin{cases}
   \Theta(n^{\log_b a}) &a \neq1\\
   \Theta(\log n) &a=1
   \end{cases}
   $$

2. if $f(n)=O(n)$ 
   $$
   T(n)=\begin{cases}
   \Theta(n) &a<b\\
   \Theta(n\log n) &a=b\\
   \Theta(n^{\log_b a}) &a>b
   \end{cases}
   $$

# 改善途径

1. 减少子问题数量
2. 减少归并工作量

# 典型应用

### 整数序列最大区间总和

蛮力算法

$O(n^3)$

递增算法：

$O(n^2)$

分治算法

$T(n)=2T(n/2)+O(n)=O(n\log n)$

### 求斐波那契数列第n项

直接递归

$O(\phi^n)$

迭代

$O(n)$

矩阵相乘

$O(\log n)$

### 平面n个点之间最短线段

蛮力算法

$O(n^2)$

递归算法

$O(n\log^2 n)$

递归算法（排序递归）

$O(n\log n)$

### 平面点集的凸包

