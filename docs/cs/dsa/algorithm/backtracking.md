# 设计思想

特征：

+ 可求解搜索问题和优化问题
+ 搜索空间是一棵**树**，每个结点对应了部分向量，满足约束条件的树叶对应了可行解，在优化问题中不一定是最优解
+ 搜索过程一般采用**深度优先**，**宽度优先**，函数优先或深度-宽度结合等策略**隐含遍历搜索树**. 隐含遍历指不真正访问每个结点，而从搜索树中进行裁剪
+ 判定条件：满足条件则分支扩张解向量；不满足约束条件则回溯到父节点
+ 结点状态：**白结点**（尚未访问），**灰结点**（正在访问该结点的子树），**黑结点**（该结点的子树遍历完成）

使用条件——**多米诺性质**：

假设$P(x_1,x_2,\cdots,x_i)$是关于向量$<x_1,x_2,\cdots,x_i>$的某个性质，那么$P(x_1,x_2,\cdots,x_{k+1})为真 \Rightarrow P(x_1,x_2,\cdots,x_{k})为真 ~~(0<k<n)$. 其中n代表解向量的维数 

 **对称性**是回溯法裁减搜索空间的有效方法

**效率估计方法**——**Monte Carlo方法**：用于估计搜索树真正访问的结点数

**改进方法**

+ 根据树的分支情况设计优先策略
+ 利用搜索树的对称性裁剪子树
+ 分解为子问题，搜索子问题的解再进行组合

算法时间复杂度：$W(n)=p(n)f(n)$，p(n)为每个结点工作量，f(n)为结点个数

# 分支限界算法

目标函数，约束条件，可行解，最优解

分支限界算法的**基本思想**：

1. 设立**代价函数**：以该点为根的搜索树中所有可行解的值的上界/下界；父结点的代价不小于/不大于子结点的代价
2. 设立**界**：当前得到可行解的目标函数的最大值/最小值
3. 搜索中当某个结点不满足约束条件或代价函数小于当时的界，则停止分支。回溯父结点
4. 更新界 

 

# 典型应用

### 八皇后问题

完全8叉树

### 0-1背包问题

完全二叉树（子集树）

### 货郎问题

排列树

### 装载问题

### 着色问题

### 分支限界：背包问题P125

### 分支限界：最大团问题P127

### 分支限界：货郎问题P127

### 分支限界：圆排列问题P129  $\Game$

### 分支限界：连续邮资问题P131 $\Game$

边搜索边生成搜索树
