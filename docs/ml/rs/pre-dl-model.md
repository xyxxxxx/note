

# CF

**协同过滤(Collaborative Filtering, CF)**是最经典的推荐系统算法。顾名思义，协同过滤就是协同大家的反馈、评价和意见共同对海量信息进行过滤，从中筛选出目标用户可能感兴趣的物品的过程。

考虑在某平台上，用户对物品给出了以下的评价：

| 用户\物品 | a    | b    | c    | d    |
| --------- | ---- | ---- | ---- | ---- |
| u         | 好评 | 差评 | 好评 | 好评 |
| v         |      | 好评 | 差评 | 差评 |
| w         | 好评 | 好评 | 差评 |      |
| x         | 差评 |      | 好评 |      |
| y         | 好评 | 好评 | ?    | 差评 |

上述矩阵称为**共现矩阵**。这时求问号处的评价更接近好评还是差评，即平台是否该将物品c推荐给用户y？

为便于计算，将上述数据转换为数值，即：

| 用户\物品 | a    | b    | c     | d    |
| --------- | ---- | ---- | ----- | ---- |
| u         | 1    | -1   | 1     | 1    |
| v         | 0    | 1    | -1    | -1   |
| w         | 1    | 1    | -1    | 0    |
| x         | -1   | 0    | 1     | 0    |
| y         | 1    | 1    | 0 (?) | -1   |

## UserCF

**基于用户的协同过滤(UserCF)**基于用户相似度进行推荐，它符合实际生活中的“兴趣相似的同好喜欢的物品，我也很可能喜欢”的观念。在上例中，其具体过程为：

1. 计算共现矩阵行向量（即用户）两两之间的相似性，构建 $m\times m$ 维用户相似度矩阵，计算方法可以为余弦相似度（向量夹角），皮尔逊相关系数或使用物品平均分的类似皮尔逊相关系数。
   $$
   余弦相似度\ {\rm sim}(\pmb u,\pmb v)=\cos(\pmb u,\pmb v)=\frac{\pmb u\cdot\pmb v}{\|\pmb u\|\cdot\|\pmb v\|}\\
   皮尔逊相关系数\ {\rm sim}(\pmb u,\pmb v)=\frac{\sum_{i\in I}(r_{u,i}-\bar{r}_u)(r_{v,i}-\bar{r}_v)}{\sqrt{\sum_{i\in I}(r_{u,i}-\bar{r}_u)}\sqrt{\sum_{i\in I}(r_{v,i}-\bar{r}_v)}}\\
   类似皮尔逊相关系数\ {\rm sim}(\pmb u,\pmb v)=\frac{\sum_{i\in I}(r_{u,i}-\bar{r}_i)(r_{v,i}-\bar{r}_i)}{\sqrt{\sum_{i\in I}(r_{u,i}-\bar{r}_i)}\sqrt{\sum_{i\in I}(r_{v,i}-\bar{r}_i)}}\\
   $$

   > 理论上，任何合理的向量相似度定义方法都可以作为标准，因此在这里也可以做出各种改进。

2. 对于特定用户y，找到与其相似度最高的 $k$ 个用户（ $k$ 为超参数）。

3. 利用Top $k$ 相似用户对物品c的评价进行预测，或者更进一步，从所有的（y未评价的）物品中选出 $k_1$ 个推荐给用户y，最常用的方式是利用加权平均得到y对每个物品的评价预测，排序并从中选取最大的 $k_1$ 个值。
   $$
   r_{u,i}=\frac{\sum_{s\in S}(w_{u,s}\cdot r_{s,i})}{\sum_{s\in S}w_{u,s}}
   $$
   其中权重 $w_{u,s}$ 是用户 $u$ 与其相似用户 $s$ 的相似度， $r_{s,i}$ 是用户 $s$ 对物品 $i$ 的评分。

UserCF包含以下2个缺点：

1. 在互联网的实际应用场景下，用户数往往远大于物品数，而UserCF需要维护用户相似度矩阵，即一个 $m\times m$ 矩阵，该矩阵的存储开销非常大，且增长十分迅速。
2. 用户的历史数据向量往往十分稀疏，对于只有几次点击、购买或评价行为的用户而言，找到相似用户的准确率非常低。

## ItemCF

**基于物品的协同过滤(ItemCF)**则基于物品相似度进行推荐，它符合实际生活中的“与我喜欢的物品相似的物品，我也很可能喜欢”的观念。在上例中，其具体过程为：

1. 计算共现矩阵列向量（即物品）两两之间的相似性，构建 $n\times n$ 维物品相似度矩阵，计算方法同上。

2. 对于特定用户y，找到其历史行为数据中的正反馈（和负反馈）物品列表。

3. 利用正反馈（和负反馈）物品列表计算物品c的相似度，或者更进一步，计算所有（y未评价的）物品的相似度，排序并从中选取最大的 $k$ 个值。注意这里如果某个物品与多个正反馈（和负反馈物品）相似，其相似度是叠加的。
   $$
   r_{u,i}=\sum_{t\in T}(w_{i,t}\cdot r_{u,t})
   $$

## 比较

两种CF方法由于实现思路上的不同，其具体应用场景上也有所不同。

+ UserCF基于用户相似度进行推荐，使得其具备更强的社交属性，用户能够快速获知与自己兴趣相似的人最近喜欢的是什么，即使某个兴趣点以前不在自己的兴趣范围内。SNS上的分享推荐即具有此种属性，例如闺蜜之间推荐新化妆品，死宅之间推荐新游戏。此外UserCF也擅长发现热点和跟踪热点，因此非常适用于热搜和新闻推荐等场景。
+ ItemCF基于物品相似度进行推荐，更适用于兴趣变化较为稳定的应用，例如在电商平台上用户倾向于寻找特定类别的商品，视频网站上用户观看的视频类型也往往比较稳定。

## 小结

CF是一个非常直观、可解释性很强的模型，但它有一个严重的问题——<u>热门的物品具有很强的头部效应</u>，容易和大量物品产生相似性；而<u>尾部的物品由于特征向量稀疏</u>，很少与其它物品产生相似性，从而很少被推荐。

举例来说，考虑以下共现矩阵：

图2.3

可以看到，物品D与所有其它物品都具有相似性，因此在以ItemCF为基础构建的推荐系统中被推荐给所有对A、B、C有过正反馈的用户，但其原因仅为D是一个热门商品。对D的推荐会进一步增加D的热门程度，从而导致头部效应，或者说马太效应。另一方面，商品A、B、C之间无法计算相似度，原因为它们的特征向量非常稀疏。由此印证了前面的问题，即头部效应明显，而处理稀疏向量的能力弱。

此外，CF仅利用了用户和物品的交互信息，而未利用用户年龄、性别、教育水平、职业等用户信息，商品分类、描述等商品信息，当前日期、时间、位置等环境信息，这无疑造成了有效信息的遗漏。

# FM

**因子分解(矩阵分解, Factorization Machine, FM)**针对CF的头部效应明显、处理稀疏向量能力弱的问题而被提出。其在CF的共现矩阵的基础上加入了隐向量的概念，加强了模型处理稀疏矩阵的能力。

FM期望为每一个用户和物品生成一个隐向量，将用户和物品定位到隐向量的表示空间上（如图2.4），用户和物品距离相近表明兴趣接近，可以推荐。

图2.4

在FM的算法框架下，用户和物品的隐向量通过分解CF的共现矩阵得到（如图2.5）。 $m\times n$ 维的共现矩阵 $R$ 被分解为 $m\times k$ 维的用户矩阵 $U$ 和 $k\times n$ 维的物品矩阵 $V$，其中 $m$ 是用户数量， $n$ 是物品数量， $k$ 是隐向量维度。 $k$ 的大小决定了隐向量表达能力的强弱， $k$ 越大，表达能力越强，但泛化能力越弱，矩阵分解的计算量越大。在实际应用中根据经验选取 $k$ 的值。

基于用户矩阵 $U$ 和物品矩阵 $V$，用户 $u$ 对物品 $i$ 的预测评分为
$$
\hat{r}_{u,i}=q_i^{\rm T}p_u
$$
其中 $p_u,q_i$ 分别为用户 $u$ 、物品 $i$ 对应的行向量。

## 矩阵分解方法

矩阵分解的主要方法有三种：特征分解、**奇异值分解**和**梯度下降**。其中特征分解只能用于方阵，因此不适用于共现矩阵。

奇异值分解方法如下：

对共现矩阵做奇异值分解 $M=U\Sigma V^{\rm T}$，取对角阵 $\Sigma$ 的前 $k$ 个奇异值，即只保留前 $k$ 行和前 $k$ 列，删除 $U,V$ 的对应维度，则有 $M\approx U_{m\times k}\Sigma_{k\times k} V^{\rm T}_{k\times n}$，即得到用户矩阵 $U$ 和物品矩阵 $V$。

看起来奇异值分解很好地解决了矩阵分解的问题，但其存在两点缺陷：

1. 奇异值分解的应用条件是被分解矩阵是稠密的。

