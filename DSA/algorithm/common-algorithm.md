[toc]

# 排序算法sorting

> 排序算法时间复杂度下界为$$O(n\log n)$$

## 冒泡排序bubble sort

$$T(n)=\frac{n(n-1)}{2}=O(n^2)$$



## 选择排序selection sort

SelectSort()

输入：包含n个元素的数组A

输出：排好序的数组A

```
for i ← 1 to n-1 do
	for j ← i+1 to n do
		if A[j]<A[i] then swap(A[j],A[i])	
```

$$T(n)=O(n^2)$$



## 插入排序insertion sort

InsertSort(A,n)

输入：包含n个元素的数组A

输出：排好序的数组A

```
for j ← 2 to n do
	x ← A[j]
	i ← j-1
	while i>0 and x<A[i] do
		A[i+1] ← A[i]
		i ← i-1
	A[i+1] ← x
```

$$W(n)=W(n-1)+n-1=O(n^2),\quad T(n)=O(n^2)$$



## 堆排序heap sort

> 最优

Heapify(A,i)

输入：堆结构A，A的结点i

输出：完成一条路径的调整

```
l ← left(i)
r ← right(i)
if l <= heapsize[A] and A[l] > A[i]
	then largest ← l
	else largest ← i
if r <= heapsize[A] and A[r] > A[i]
	then largest ← r
if largest != i
	then exchange A[i]←→A[largest]
		Heapify(A,largest)
```

Build-Heap(A)

输入：数组A

输出：堆A

```
heapsize[A]←length[A]
for i ← floor(length[A]/2) downto 1 do
	Heapify(A,i)
```

Heap-sort(A)

输入：数组A

输出：排好序的数组A

```
Build-Heap(A)
for i ← length[A] downto 2 do
	exchange A[1]←→A[i]
	heapsize[A]← heapsize[A]-1
	Heapify(A,i)
```

$$T(n)=O(n\log n)$$



## 归并排序merge sort

> 最优

Mergesort(A,p,r)

输入：数组A[p..r], 1<=p<=r<=n

输出：排好序的区间A[p..r]

```
if p<r
then q ← floor((p+r)/2)
Mergesort(A,p,q)
Mergesort(A,q+1,r)
Merge(A,p,q,r)
```

Merge(A,p,q,r)

输入：排好序的区间A[p..q]和A[q+1..r]

输出：排好序的区间A[p..r]

```
x ← q-p+1, y ← r-q
copy A[p..q] into B[1..x], copy A[q+1..r] into C[1..y]
i←1, j←1, k←p
while i<=x and j<=y do
	if B[i]<=C[j]
	then A[k]=B[i]
		i←i+1
	else
		A[k]=C[j]
		j←j+1
	k←k+1
if i>x 
then copy C[j..y] into A[k..r]
else copy B[i..y] into A[k..r]
```

$$W(n)=2W(n/2)+n-1=O(n\log n)$$



## 快速排序quick sort

> 平均时间最优

QuickSort(A,p,r)

输入：区间A[p..r]

输出：排好序的区间A[p..r]

```
if p<r
then q ← Partition(A,p,r)
	swap(A[p],A[q])
	QuickSort(A,p,q-1)
	QuickSort(A,q+1,r)
```

Partition(A,p,r)

输入：区间A[p..r]

输出：A[p]在区间应处位置

```
![xjcvioh46j7ioyjpegrhw](C:\Users\Xiao Yuxuan\Documents\pic\xjcvioh46j7ioyjpegrhw.PNG)x ← A[p],i←p+1,j←r
while i<j do
	while A[j]>x do
		j←j-1
	while A[i]<x do
		i←i+1
	if i<j
	swap(A[i],A[j])
	else return j
```

最坏情况$$T(n)=O(n^2)$$，平均情况$$T(n)=O(n\log n)$$



## 桶排序bucket sort

> 桶排序因为假设输入数据服从均匀分布，平均情况下时间复杂度达到$$O(n)$$

<img src="C:\Users\Xiao Yuxuan\Documents\pic\xjcvioh46j7ioyjpegrhw.PNG" alt="xjcvioh46j7ioyjpegrhw" style="zoom:80%;" />

<img src="C:\Users\Xiao Yuxuan\Documents\pic\jksgfiopj2345tiotgnmo.PNG" alt="jksgfiopj2345tiotgnmo" style="zoom:80%;" />

$$T(n)=\Theta(n)+\sum_{i=0}^{n-1}O(n_i^2)$$，平均情况$$T(n)=\Theta(n)$$





# 查找算法list search

## 顺序查找linear search

### 改进顺序查找

算法 search(L,x)

```
j ← 1
while j <= n and x > L[j] do j ← j+1
if x < L[j] or j > n then j ← 0
return j
```



## 二分查找binary search

BinarySearch(T,x)

输入：排好序的数组T，数x

输出：目标元素位置j

```
l ← 1, r ← n
while l<=r do
	m ← floor((l+r)/2)
	if T[m]=x then return m
	else if T[m]>x then r ← m-1
		else l ← m+1
return 0
```



## 斐波那契查找Fibnacci search

参考Data Structure





# 选择算法

## 最值

Findmax （最优）

输入：数组L[1..n]

输出: max

```
max ← L[1]
for i←2 to n do
	if max<L[i]
	then max←L[i]
return max	
```

$$T(n)=n-1$$



## 双最值

Findmaxmin （最优）

输入：数组L[1..n]

输出: max, min

```
将n个元素两两分组分为floor(n/2)组
每组比较，得到floor(n/2)个较小和较大元素
在ceil(n/2)个较小元素中求最小值
在ceil(n/2)个较大元素中求最大值
```

$$W(n)=\lceil 3n/2 \rceil-2$$



## 第二最值 

Findsecond （最优）

输入：数组L[1..n]

输出: max, second

```
k ← n
将k个元素两两分组分为floor(k/2)组
每组比较，淘汰较小的数，并将其记录在较大的数的链表中
if k mod 2==1
	then k←floor(k/2)+1
else k ← k/2
if k>1
	then goto 2
max	← 最后剩余元素
second ← max链表中的最大元素
```

$$W(n)=n+\lceil \log n \rceil-2$$



## 第k最值 

Select(S,k) （阶最优）

输入：数组S[1..n]，正整数k

输出：S中的第k小元素

```
//P43
```

$$W(n)\leq W(n/5)+W(7n/10)+O(n)=O(n)$$





# 贪心算法

## Huffman algorithm

输入：字符集$$C=\{x_1,x_2,\cdots,x_n\}$$，每个字符的频率$$f(x_i)$$
输出：Q

```
n ← |C|
Q ← C
for i←1 to n-1 do
	z← allocate-node()
	z.left ← Q中最小元x
	z.right ← Q中最小元y
	f(z)←f(x)+f(y)
	Insert(Q,z)
return Q	
```





# 图算法graph search

## breadth-first search



## depth-first search



## Prim algorithm

输入：图$$G=(V,E,W)$$
输出：最小生成树

算法Prim(G,E,W)

```
S ← {1}; T←emptyset
while V-S!=emptyset do
	从V-S中选择j使得j到S中顶点的边e的权最小;
	T←T\bigcup {e};S←S\bigcup {j}
```

$$T(n)=O(n^2)$$



## Kruskal algorithm

输入：图$$G=(V,E,W)$$
输出：最小生成树

算法Kruskal

```
按照权从小到大排序G中的边，使E={e1,e2,...,em}
T ← emptyset
repeat
	e←E中的最短边
	if e的两端点不在同一连通分支
	then T←T\bigcup {e}; E←E-{e}
until T包含n-1条边	
```

$$T(n)=O(m\log n)$$

> 当$$m=\Theta(n^2)$$时，Prim算法效率更高；当$$m=\Theta(n)$$时，Kruskal算法效率更高



## Bellman-Ford algorithm

> 一般情形下的单源最短路径算法



## Dijkstra algorithm

> 所有边权重非负的单源最短路径算法，时间复杂度低于Bellman-Ford algorithm

输入：带权有向图$$G=(V,E,W)$$，源点$$s\in V$$
输出：数组L，$$\forall j \in V-\{s\}$$，$$L[j]$$表示s到j的最短路径上j前一个结点的标号

```
S←{s}
dist[s]←0
for i \in V-{s} do
	dist[i]←w(s,i)
while V-S!=\empty do
	从V-S中取出具有相对S的最短路径的结点j,k是该路径上连接j的结点
	S←S \bigcup {j}; L[j]←k
	for i \in V-S do
		if dist[j]+w(j,i)<dist[i]
		then dist[i]←dist[j]+w[j,i]
```

$$T(n)=O(n^2)$$



## Floyd-Warshall algorithm

> 所有节点之间的最短路径



## Ford-Fulkerson algorithm

> 最大流





# 字符串匹配算法





# 矩阵算法

## 高斯消元法



## 矩阵乘法





# 多项式算法

## 多项式计算

计算$$A(x)=a_0+a_1x+\cdots+a_{n-1}x^{n-1}$$

**蛮力算法**

$$T(n)=O(n^2)$$

**减治算法**
$$
A_{i}(x)=xA_{i-1}(x)+a_{n-i}
$$
$$T(n)=T(n-1)+O(1)=O(n)$$

**分治算法**
$$
A(x)=A_{even}(x^2)+xA_{odd}(x^2)
$$
$$T(n)=2T(n/2)+O(1)=O(n)$$，计算n个多项式时$$T(n)=2T(n/2)+O(n)=O(n\log n)$$

> 参见离散数学-多项式-快速傅里叶变换



## 多项式乘法(卷积计算)

考虑多项式乘法$$C(x)=A(x)B(x)$$，其中$$A(x)$$的系数向量为$$(a_0,a_1,\cdots,a_{n-1})$$，$$B(x)$$的系数向量为$$(b_0.b_1,\cdots,b_{n-1})$$，求$$C(x)$$的系数向量

### 蛮力算法

```c++
for(int i = 0; i < n; ++i)
	for(int j = 0; j < n; ++j){
		c[i+j] += a[i] * b[j];
	}
```

$$T(n)=O(n^2)$$



### 插值

```
求值A(x_j)和B(x_j),j=0,1,...,2n-1
计算C(x_j)
插值求得C(x)的系数
```



### 快速傅里叶变换FFT

> https://zhuanlan.zhihu.com/p/76622485
>
> https://zhuanlan.zhihu.com/p/31584464
>
> http://blog.miskcoo.com/2015/04/polynomial-multiplication-and-fast-fourier-transform



选择2n个数为1的2n次根$$\omega_j=e^{\frac{j}{n}\pi i},j=0,1,\cdots,2n-1$$

```c++
求值A(omega_j)和B(omega_j) //O(nlogn)
计算C(omega_j)=d_j; //O(n)
构造多项式D(x)=d_0+d_1*x+...+d_{2n-1}*x^{2n-1} 
计算D(omega_j) //O(nlogn)
D(omega_j)=2n*c_{2n-j}, D(omega_0)=2n*c_0
```





# 数论算法number theory

## 欧几里得算法Euclidian algorithm

算法Euclid(m,n)

输入：非负整数m,n，m,n不全为0

输出：m,n的最大公约数

```
while m>0 do
	r ← n mod m
	n ← m
	m ← r
return n
```

**多项式计算**

```
y ← P[0]; power ← 1
for i ← 1 to n do
	power ← power *x
	y ← y+P[i]*power
return y
```



# 其他

## 幂乘

计算$$a^n$$



### 蛮力算法

做n-1次乘法，$$T(n)=O(n)$$



### 快速幂/二进制取幂

计算$$a,a^2,\cdots,a^{2^{\lfloor \log_2 n\rfloor}}$$，再相乘得到

```c++
long long binpow(long long a, long long n) {
  long long res = 1;
  while (n > 0) {
    if (n & 1) res = res * a;//位运算n&1，取二进制n最后一位
    a = a * a;
    n >>= 1;
  }
  return res;
}
```

$$T(n)=O(\log n)+O(\log n)=O(\log n)$$



## Hanoi塔

 Hanoi(A,C,n)

```
if n=1 then move (A,C)
else
	Hanoi (A,B,n-1)
	move(A,C)
	Hanoi (B,C,n-1)
```

$$
T(n)=2T(n-1)+1\\
T(1)=1, \;T(n)=2^n-1
$$



## 随机打乱数组次序

```c++
void shuffle(int A[],int n)
{
    while (n>1)
        swap(A[rand()%n],A[--n]);
}
```

从后向前，依次将各元素与随机选取的某一元素交换





**2-subset**

NP-complete



**货郎算法**

设有m个城市，已知其中任意两个城市之间的道路距离。一个货郎需要到每个城市巡回卖货，他从某个城市出发，每个城市恰好经过一次最后回到出发的城市，求最短路程及路线。

或 如何在带权完全图G中找一条最短哈密顿回路

