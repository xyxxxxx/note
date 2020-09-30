# 加法法则和乘法法则



# 排列和组合

## 排列

$$
A_n^k=n(n-1)\cdots(n-k+1)=\frac{n!}{(n-k)!}
$$

## 组合

$$
C_n^k=\frac{n!}{k!(n-k)!}=\frac{A_n^k}{k!}
$$


$$
C_{n+1}^{k+1}=C_{n}^{k}+C_{n}^{k+1}\\
\sum_{k=0}^{n}C_n^k=2^n\\
\sum_{r=0}^{k}C_{n+r-1}^r=C_{n+k}^k\\
C_{m+n}^k=\sum_{i=0}^k C_n^iC_m^{k-i}\\
\sum_{r=0}^{n}(C_n^r)^2=C_{2n}^{n}\\
$$




# 二项式定理

$$
(1+x)^n=\sum_{k=0}^{n}C_n^kx^k=C_n^0+C_n^1x+\cdots+C_n^nx^n\\
(x+y)^n=\sum_{k=0}^{n}C_n^kx^ky^{n-k}=C_n^0y^n+C_n^1xy^{n-1}+\cdots+C_n^nx^n
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

| C_0  | C_1  | C_2  | C_3  | C_4  | C_5  | C_6  | ...  |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 1    | 1    | 2    | 5    | 14   | 42   | 132  | ...  |

### 通项公式

$$
C_n=\frac{C_{2n}^{n}}{n+1}=C_{2n}^{n}-C_{2n}^{n+1}
$$

### 递推关系式

$$
C_{n+1}=\sum_{i=0}^nC_iC_{n-i}=\frac{2(2n+1)}{n+2}C_n
$$





# Stirling数

第一类Stirling数



第二类Stirling数

