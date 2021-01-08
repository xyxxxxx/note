> 参考：
>
> 离散数学, 屈婉玲
>
> [OI Wiki](https://oi-wiki.org/math/)

[toc]

# 加法原理和乘法原理

**加法原理** 完成一个工程可以有$$n$$类办法，$$a_i$$代表第$$i$$类方法的数目，那么完成这件事共有$$S=a_1+a_2+\cdots+a_n$$种不同的方法。

**乘法原理** 完成一个工程需要分$$n$$个步骤，$$a_i$$代表第$$i$$个步骤的不同方法数目，那么完成这件事共有$$S=a_1×a_2×\cdots×a_n$$种不同的方法。





# 排列和组合

## 排列

$$
A_n^k=n(n-1)\cdots(n-k+1)=\frac{n!}{(n-k)!}
$$



## 组合

$$
C_n^k=\begin{pmatrix}n\\k\end{pmatrix}=\frac{n!}{k!(n-k)!}=\frac{A_n^k}{k!}
$$

**性质**

$$
\begin{pmatrix}n\\k\end{pmatrix}=\begin{pmatrix}n\\n-k\end{pmatrix} \\
\begin{pmatrix}n+1\\k+1\end{pmatrix}=\begin{pmatrix}n\\k\end{pmatrix}+\begin{pmatrix}n\\k+1\end{pmatrix} \\
\sum_{k=0}^{n}\begin{pmatrix}n\\k\end{pmatrix}=2^n\\
\sum_{r=0}^{k}\begin{pmatrix}n+r-1\\r\end{pmatrix}=\begin{pmatrix}n+k\\k\end{pmatrix}\\
\begin{pmatrix}m+n\\k\end{pmatrix}=\sum_{i=0}^k \begin{pmatrix}m\\i\end{pmatrix}\begin{pmatrix}n\\k-i\end{pmatrix}\\
\sum_{r=0}^{n}\begin{pmatrix}n\\r\end{pmatrix}^2=\begin{pmatrix}2n\\n\end{pmatrix}\\
$$




# 二项式定理

$$
(1+x)^n=\sum_{k=0}^{n}\begin{pmatrix}n\\k\end{pmatrix}x^k=\begin{pmatrix}n\\0\end{pmatrix}+\begin{pmatrix}n\\1\end{pmatrix}x+\cdots+\begin{pmatrix}n\\n\end{pmatrix}x^n\\
(x+y)^n=\sum_{k=0}^{n}\begin{pmatrix}n\\k\end{pmatrix}x^ky^{n-k}=\begin{pmatrix}n\\0\end{pmatrix}y^n+\begin{pmatrix}n\\1\end{pmatrix}xy^{n-1}+\cdots+\begin{pmatrix}n\\n\end{pmatrix}x^n
$$





# 多项式定理





# 递推方程

## 公式解法



## 生成函数





# Catalan数

以下问题的解为Catalan数列：

1. Cn表示长度2n的dyck word的个数。Dyck word是一个有*n*个X和*n*个Y组成的字串，且所有的前缀字串皆满足X的个数大于等于Y的个数。

   XXYY XYXY

   XXXYYY XXYXYY XXYYXY XYXXYY XYXYXY

2. Cn表示所有在n × n格点中不越过对角线的单调路径的个数。一个单调路径从格点左下角出发，在格点右上角结束，每一步均为向上或向右

   ![Catalan number 4x4 grid example.svg](https://upload.wikimedia.org/wikipedia/commons/thumb/f/f4/Catalan_number_4x4_grid_example.svg/1920px-Catalan_number_4x4_grid_example.svg.png) 

3. Cn表示n个+1和n个-1构成2n项数列$$a_1,a_2,...,a_{2n}$$的个数，其部分和满足$$a_1+a_2+...+a_k>=0,k=1,2,...,2n$$

4. Cn表示n组括号的合法运算式的个数（理解为X=放一个 Y=乘最后两个元素）

   ((A1A2)A3)  (A1(A2A3)) 

   (((A1A2)A3)A4) ((A1A2)(A3A4)) ((A1(A2A3))A4) (A1((A2A3)A4)) (A1(A2(A3A4))

5. Cn表示2n个高矮不同的人排成两排，每排必须是从矮到高排列，而且第二排比对应的第一排的人高，总共的排列方式种数（理解为依次放入队列，X=放第一排，Y=放第二排）

   | 34   | 24   |
   | ---- | ---- |
   | 12   | 13   |

   | 456  | 356  | 346  | 256  | 246  |
   | ---- | ---- | ---- | ---- | ---- |
   | 123  | 124  | 125  | 134  | 135  |

6. Cn在圆上选择2n个点，将这些点成对连接起来使得所得到的n条线段不相交的方法数

7. Cn表示对角线不相交的情况下，将一个n+2边凸多边形区域分成三角形区域的方法数

   ![Catalan-Hexagons-example.svg](https://upload.wikimedia.org/wikipedia/commons/thumb/a/a8/Catalan-Hexagons-example.svg/1920px-Catalan-Hexagons-example.svg.png) 

8. Cn表示n个节点组成不同构二叉树的方案数，或2n+1个节点组成不同构满二叉树的方案数（根据递推公式得到）

   ![Catalan number binary tree example.png](https://upload.wikimedia.org/wikipedia/commons/0/01/Catalan_number_binary_tree_example.png) 

9. Cn表示n个不同的数依次进栈，不同的出栈结果种数

    

## Catalan数列

| $$C_0$$ | $$C_1$$ | $$C_2$$ | $$C_3$$ | $$C_4$$ | $$C_5$$ | $$C_6$$ | ...  |
| ------- | ------- | ------- | ------- | ------- | ------- | ------- | ---- |
| 1       | 1       | 2       | 5       | 14      | 42      | 132     | ...  |

### 通项公式

$$
C_n=\frac{\begin{pmatrix}2n\\n\end{pmatrix}}{n+1}=\begin{pmatrix}2n\\n\end{pmatrix}-\begin{pmatrix}2n\\n-1\end{pmatrix}
$$

### 递推关系式

$$
C_{n+1}=\sum_{i=0}^nC_iC_{n-i}=\frac{4n+2}{n+2}C_n
$$





# Stirling数

第一类Stirling数



第二类Stirling数





# Bell数









