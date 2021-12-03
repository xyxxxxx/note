

**词性(part of speech, POS, word classes, syntactic categories)**包含了词和它的上下文的信息，在成分分析、命名实体检测等任务中十分重要。

本章将介绍两种**词性标注(part of speech tagging)**模型：生成型的**隐马尔可夫模型(Hidden Markov Model, HMM)**和区分型的**最大熵马尔可夫模型(Maximum Entropy Markov Model, MEMM)**。在循环网络部分还会介绍第三种基于循环网络的算法。这三种模型大体上有相当的表现，但使用时需要彼此权衡。

# 英文词类

尽管每一类词都具有语义上的倾向，但词类本身是基于传统的定义，而不是通过句法或形态函数（即根据词的分布性质或形态性质）定义。

词类可以分为宽泛的两大类：**封闭类(closed class)**和**开放类(open class)**。封闭类具有相对固定的成员，例如很少出现新的介词；相对地，名词和动词就是开放类，因为不断地有新的名词和动词被创造或借用。每个语料库都有不同的开放类词，但很可能共享一个封闭类词集。封闭类词通常是功能词，例如代词you, it，介词of, in，连接词and等，一般很短、出现频率高，且有语法结构上的作用。

世界上的语言有四种主要的开放类：名词、动词、形容词和副词。其特征整理如下：

**名词(noun)**

| 共通                                       | 英文                                     | 中文                                        | 日文                                  |
| ------------------------------------------ | ---------------------------------------- | ------------------------------------------- | ------------------------------------- |
| 包含人、地点、物品、组织、抽象概念等       | 包含动名词<br />e.g. tagging             | 多数动词可以直接用做名词<br />e.g. 生产基地 | 包含动词的第一连用形<br />e.g. turiai |
| 可以笼统地理解为各种物                     | 有所有格<br />e.g. IBM’s annual revenue  | 有所有格<br />e.g. 谷歌的年营收             | 有所有格<br />e.g. IBMの年間収益      |
| 分为专有名词和普通名词，前者是特殊的实体名 | 普通名词的大部分有单复数<br />e.g. goats |                                             |                                       |
|                                            | 有限定词<br />e.g. the paper             |                                             |                                       |
|                                            | 专有名词的首字母需要大写<br />e.g. Alice |                                             |                                       |

**动词(verb)**

| 共通           | 英文                                           | 中文 | 日文                                               |
| -------------- | ---------------------------------------------- | ---- | -------------------------------------------------- |
| 表示动作或过程 | 包含单数第三人称形式、进行时、过去式、过去分词 |      | 包含未然形、连用形、终止形、连体形、假定形、命令形 |

**形容词(adjective)**

| 共通           | 英文 | 中文 | 日文                                               |
| -------------- | ---- | ---- | -------------------------------------------------- |
| 表示事物的属性 |      |      | 包含形容词和形容动词                               |
|                |      |      | 包含未然形、连用形、终止形、连体形、假定形、命令形 |

**副词(adverb)**

| 共通                                       | 英文 | 中文 | 日文             |
| ------------------------------------------ | ---- | ---- | ---------------- |
| 修饰动词                                   |      |      | 即形容词的连用形 |
| 包括位置副词、时间副词、程度副词和方法副词 |      |      |                  |

不同语言之间的封闭类的差异比开放类大得多，一些重要的封闭类在英文中的表示为：

​        prepositions: on, under, over, near, by, at, from, to, with
​        particles: up, down, on, off, in, out, at, by
​        determiners: a, an, the
​        conjunctions: and, but, or, as, if, when
​        pronouns: she, who, I, others
​        auxiliary verbs: can, may, should, are
​        numerals: one, two, three, first, second, third

**介词(preposition)**出现在名词短语之前，语义上表示空间、时间等关系。

**助词(particle)**与名词结合使用，通常有扩展的词义。可以作为单独的句法或语义单元的动词和助词组合称为**动词短语(phrasal verb)**，动词短语的词义通常是**不能组合的(noncompositional)**，即不能根据动词和助词的词义来预测。

一类与名词共同出现、标志名词短语开始位置的词称为**限定词(determiner)**。限定词中的一类是**冠词(article)**，英文中只有三个：a, an, the；其它的限定词包括this, that等。冠词在英文中出现得如此频繁，事实上，the是大部分书面英文语料库的最频繁的词，a, an也紧随其后。

**连词(conjunction)**连接两个短语、从句或句子。**并列连词(coordinating conjunction)**，例如and, or, but，连接两个相同的元素；**从属连词(Subordinating conjunction)**表示元素有嵌入的状态。

**代词(pronoun)**通常用作一些名词短语、实体或事件的速记表达。**人称代词(personal pronoun)**指代人或实体，例如I, you, me, she；**物主代词(possesive pronoun)**指代从属或其它抽象关系，例如my, your, his；**wh代词(wh-pronoun)**用于特定的疑问形式，或作为补语。

英文动词中的一个封闭类称为**辅助动词(auxiliary)**。英文的辅助动词包含**系动词(copula)**be, do, have和它们的变形，以及**情态动词(modal verb)**would, could, should, must...

# Penn树库词性标注集

英文的一个重要的标注集是45标注Penn树库词性标注集(Marcus et al., 1993)，被用于标注非常多的语料库，如下图所示。

![](https://i.loli.net/2021/01/18/VPOkdZY3K78whCz.png)

下面是一些例子：

​        The/DT grand/JJ jury/NN commented/VBD on/IN a/DT number/NN of/IN other/JJ topics/NNS ./.
​        There/EX are/VBP 70/CD children/NNS there/RB
​        Preliminary/JJ findings/NNS were/VBD reported/VBN in/IN today/NN ’s/POS New/NNP England/NNP Journal/NNP of/IN Medicine/NNP ./.

被标注了词性的语料库是标注算法的重要训练和测试集，有三个被标注的语料库常被使用。**Brown**语料库长一百万词，由1961年在美国发表的500篇不同体裁的文章组成；**WSJ**语料库长一百万词，由1989年华尔街日报发表的文章组成；**Switchboard**语料库长两百万词，由1990-1991年搜集的通话记录组成。

# 词性标注

**词性标注(part-of-speech tagging)**是为输入文本的每个词赋一个词性标记的过程。标记算法输入分词过的词序列和一个标注集，输出一个标注序列，长度与词序列相等。

标注是一个**消歧义(disambiguation)**的任务；词是**有歧义的(ambiguous)**，可能有多种可能的词性，而目标就是找到语境下的正确标注。例如book可以是动词或名词，that可以是限定词或补语。词性在多大程度上会有歧义？如下图所示，大部分的词（85~86%）是无歧义的，但词汇表中剩下的14~15%的词是非常常用的词，因此文本中55~67%的token都是歧义的。

![](https://i.loli.net/2021/01/18/yNwzAvHRYi5jDqg.png)

最有歧义的常用词包含that, back, down, put, set, etc，下列是back的6种不同词性的示例：

​        earnings growth took a back/JJ seat
​        a small building in the back/NN
​        a clear majority of senators back/VBP the bill
​        Dave began to back/VB toward the door
​        enable the country to buy back/RP about debt
​        I was twenty-one back/RB then

最简单的基线算法是对于任意一个给定的歧义词，选择该词在训练集中频率最高的标注。在WSJ语料库上的训练和测试显示，这样一个基线能够达到92.34%的准确率；作为对比，最先进的标注方法（HMMs, MEMMs, 神经网络, 基于规则的算法）在该数据集上可以达到97%的准确率。

# HMM词性标注

**HMM(Hidden Markov Model)**是一个同步的序列到序列模型，也是一个概率模型，计算几种可能的标签序列的概率并选择最佳的序列。

HMM基于增强的马尔可夫链，hidden表示标注并非直接观察到，而是通过词序列推断得到。

## HMM模型

HMM由以下部分组成：

+ $Q=\{q_1,q_2,\cdots,q_N\}$ ：N个**状态(state)**的集合
+ $A=\begin{bmatrix}a_{11}&\cdots\\\cdots&a_{NN} \end{bmatrix}$ ：**状态转移矩阵(transition probability matrix)**，其中 $a_{ij}$ 代表从状态 $i$ 转移到状态 $j$ 的概率
+ $O=o_1o_2\cdots o_T$ ： $T$ 次**观察(observation)**的序列，每一个 $o_i$ 都属于词汇表 $V$ 
+ $B=b_i(o_t)$ ：**观察似然(observation likelihood)**的序列，也称为**发射概率(emission probabilities)**，表示观察 $o_t$ 产生于状态 $q_i$ 的概率
+ $\pi = \pi_1, \pi_2,\cdots, \pi_N$ ：初始概率分布， $\sum\pi_i =1$ 

一阶的HMM使用了两个简化假定：第一，当前状态的概率取决于前一状态
$$
P(q_i|q_1\cdots q_{i-1})=P(q_i|q_{i-1})
$$
第二，观察 $o_i$ 的概率仅取决于产生该观察的状态 $q_i$，而不是其它的状态或观察
$$
P(o_i|q_1\cdots q_To_1\cdots o_T)=P(o_i|q_i)
$$

## HMM标注器的组成

HMM包含两部分：A和B概率。A概率矩阵包含了标注转移矩阵 $P(t_i|t_{i-1})$，例如情态动词之后高概率接续一个动词的基本形VB，因此 $P(VB|MD)$ 应该非常高。我们通过计数计算该转移概率的最大似然估计，即
$$
P(t_i|t_{i-1})=\frac{C(t_{i-1},t_i)}{C(t_{i-1})}
$$
例如在WSJ语料库中，
$$
P(VB|MD)=\frac{C(MD,VB)}{C(MD)}=\frac{10471}{13124}=.80
$$
HMM需要在已标记的训练语料库上计数，例如可以使用WSJ语料库。

B发射概率 $P(w_i|t_i)$，代表给定一个标注，给定词属于该标注的概率。发射概率的最大似然估计为
$$
P(w_i|t_i)=\frac{C(t_i,w_i)}{C(t_i)}
$$
例如在WSJ语料库中，
$$
P(will|MD)=\frac{C(MD,will)}{C(MD)}=\frac{4046}{13124}=.31
$$
> 根据贝叶斯公式 $P(t_i|w_i)=P(w_i|t_i)P(t_i)/P(w_i)$， $w_i$ 已经由文本给定因此 $P(w_i)=1$， $P(t_i)$ 作为先验可以利用额外的条件即 $P(t_i|t_{i-1})$，先验和似然相乘即为后验。

HMM的A转移概率和B观察似然绘制如下图（简化为只有3个状态）

![](https://i.loli.net/2021/01/19/VKkfFG4miOS5pqE.png)

## HMM标注作为解码

HMM的**解码(decoding)**过程定义为：给定HMM $\lambda=(A,B)$ 和观察序列 $O=o_1o_2\cdots o_T$，寻找最可能的状态序列 $Q=q_1q_2\cdots q_T$。

对于词性标注任务，HMM解码就是在给定观察序列 $w_1^n$ 的条件下，选择最可能的标注序列 $t_1^n$ ：
$$
\hat t_1^n=\arg\max_{t_1^n} P(t_1^n|w_1^n)
$$
使用贝叶斯公式转化为
$$
\hat t_1^n=\arg\max_{t_1^n} \frac{P(w_1^n|t_1^n)P(t_1^n)}{P(w_1^n)}=\arg\max_{t_1^n} P(w_1^n|t_1^n)P(t_1^n)
$$
HMM标注器使用的两个简化假定为：每个词的出现概率仅取决于其自身的标注，而与附近的词和标注无关；标注仅取决于前一个标注，称为bigram假定：
$$
P(w_1^n|t_1^n)=\prod_{i=1}^n P(w_i|t_i)\\
P(t_1^n)\approx \prod_{i=1}^n P(t_i|t_{i-1})
$$
应用上述假定，有
$$
\hat t_1^n=\arg\max_{t_1^n} P(w_1^n|t_1^n)P(t_1^n)\approx \arg\max_{t_1^n} \prod_{i=1}^n P(w_i|t_i)P(t_i|t_{i-1})\\
\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad {\rm emission}↑\quad\quad\quad ↑{\rm transition}
$$
此公式完全对应先前定义的A转移概率和B发射概率。

## Viterbi算法

HMM的解码算法称为Viterbi算法，如下图所示：

![Screenshot from 2021-01-19 10-54-34.png](https://i.loli.net/2021/01/19/AHUWbgRNpeX9imD.png)

Viterbi算法首先建立一个概率矩阵，称为lattice，其中每列代表一个观察，每行代表一个状态。下图展示了句子Janet will back the bill的lattice：

![](https://i.loli.net/2021/01/19/u8GqN4JQpKmZMUY.png)

lattice的每个元素 $v_t(j)$，代表在已知观察序列 $o_1\cdots o_t$ 和可能的状态序列 $q_1\cdots q_{t-1}$ 的条件下， $q_t=j$ 的最大概率，形式上表示为
$$
v_t(j)=\max_{q_1\cdots q_{t-1}}P(q_1\cdots q_{t-1}o_1\cdots o_t,q_t=j|\lambda)
$$
最可能的路径是对所有可能的先前状态序列取最大值，就像其它动态规划算法一样，Viterbi迭代地计算每个元素。若已知 $t-1$ 时刻所有状态的概率 $v_{t-1}(j)$，则
$$
v_t(j)=\max_{i=1}^N v_{t-1}(i)a_{ij}b_j(o_t)
$$
其中3个因子分别为

+ $v_{t-1}(i)$ ：前一时刻的**Viterbi路径概率**
+ $a_{ij}$ ：从状态 $q_i$ 转移到 $q_j$ 的**转移概率**
+ $b_j(o_t)$ ：给定状态 $j$ 条件下观察 $o_t$ 的**状态观察似然**

## 示例

以标注句子Janet will back the bill为例，正确标注（目标）为

​        Janet/NNP will/MD back/VB the/DT bill/NN

HMM由下图的两个矩阵定义。矩阵数据来源于WSJ语料库中的计数（稍作了简化）。

![](https://i.loli.net/2021/01/19/bzV1KS5fD6slkAq.png)

下图在图8.6的基础上增加了计算过程，注意 $\pi$ 表示序列开始符SOS：

![](https://i.loli.net/2021/01/19/whYzE4XABOGecCa.png)

## 扩展HMM算法到trigram

实际的HMM标注器会在上述简单模型的基础上做一些扩展，其中之一就是更宽的标注上下文。上述标注器中一个标注的概率仅取决于前一个标注，而实践中会使用更多历史标注。

将算法从bigram扩展到trigram可以小幅（大概0.5个百分点）提升性能，但需要对Viterbi算法做出非常大的改变。图8.7代表的转移概率矩阵会增加到 $N^2$ 行，而对于lattice中的每个元素需要计算前两列的所有路径组合中的最大值。

除了增加上下文宽度，HMM标注器还有一些其它的高级特征，例如使用序列结束符EOS告知句子的结束位置。SOS和EOS与任何常规的词都不同，因此需要使用特殊的标注“句子边界”。

使用trigram标注器的一个问题是数据的稀疏性。就像语言模型一样，根据最大似然估计
$$
P(t_i|t_{i-1}t_{i-2})=\frac{C(t_{i-2}t_{i-1}t_i)}{C(t_{i-2}t_{i-1})}
$$
许多测试集中出现标注序列 $t_{i-2},t_{i-1},t_i$ 没有出现在训练集，那么这样一个trigram的概率就是0，即永远不会发生。和语言模型一样，标准的解决方法是插值：
$$
{\rm trigram}\quad \hat{P}(t_i|t_{i-1}t_{i-2})=\frac{C(t_{i-2}t_{i-1}t_i)}{C(t_{i-2}t_{i-1})}\\
{\rm bigram}\quad \hat{P}(t_i|t_{i-1})=\frac{C(t_{i-1}t_i)}{C(t_{i-1})}\\
{\rm unigram}\quad \hat{P}(t_i)=\frac{C(t_i)}{N}\\
P(t_i|t_{i-1}t_{i-2})=\lambda_3\hat{P}(t_i|t_{i-1}t_{i-2})+\lambda_2\hat{P}(t_i|t_{i-1})+\lambda_1\hat{P}(t_i)
$$
其中 $\lambda_1+\lambda_2+\lambda_3=1$。 $\lambda$ 的值由**删除插值法(deleted interpolation)**确定(Jelinek and Mercer, 1980)，算法如下图所示。

![Screenshot from 2021-01-19 14-57-06.png](https://i.loli.net/2021/01/19/nFr1LsflQtNeci4.png)

> 考虑上面的三种情形：（需要实验验证）
>
> + $\frac{C(t_1,t_2,t_3)-1}{C(t_1,t_2)-1}$ 最大，例如等于1，那么 $t_1,t_2\to t_3$，表示这是一个连接紧密的trigram
> + $\frac{C(t_2,t_3)-1}{C(t_2)-1}$ 最大，表示 $t_2t_3$ 是一个比较紧密的bigram
> + $\frac{C(t_3)-1}{N-1}$ 最大，表示 $t_3$ 倾向于是单独出现

## 束搜索

Viterbi算法的复杂度为 $O(N^2T)$，对于trigram的情形增加至 $O(N^3T)$，因此当状态数增加到较大时，Viterbi算法会变得很慢。

一种常用的方法是使用**束搜索(beam search)**解码。在束搜索中，我们计算 $t$ 时刻所有可能状态的概率，但仅保留其中最好的几个状态，其余状态将直接去除。实现束搜索的最简单方法是保留固定数量的状态，该数量称为**束宽(beam width)** $\beta$，下图展示了 $\beta=2$ 的情形。同样也可以设定束宽为状态数的固定比例，或设定概率阈值等。

![](https://i.loli.net/2021/01/19/i71rKFtGVUZMIkd.png)

## 未知词

为了实现词性标注器的高准确率，模型处理未知词的能力也十分重要。专有名词总是不断涌现，新的名词和动词也快速进入各种语言。一种有用的识别词性的特征是词形：首字母大写的词为专有名词。

但是对于猜测未知词的词形，最大的信息来源是词法学：-s可能是名词复数，-ed可能是动词过去式，-able可能是形容词等等。我们在训练过程中建立词后缀（定长）与标注的统计关系，即计算给定长度为 $i$ 的后缀的条件下标注 $t_i$ 的概率(Samuelsson 1993, Brants 2000)：
$$
P(t_i|l_{n-i+1}\cdots l_n)
$$
可以使用后退法，即逐次使用更短的后缀来平滑概率。

由于未知词不太可能属于封闭类，可以只计算训练集中词频不大于10的词的后缀概率（因此不包含封闭类词以及使用方法灵活的常用词）。对于首字母大写和小写的词的处理方式也不一样。

结合所有特征，一个trigram HMM标注器可以在Penn树库上达到96.7%的标注准确率，仅略逊于最好的MEMM和神经标注器的表现。

# 最大熵马尔可夫模型

尽管HMM可以达到很高的准确率，但我们看到它需要一些结构上的改进以处理未知词、后退、后缀等等。我们希望能用一种简洁的方式增加任意的特征，但这对于HMM这样的生成型模型而言十分困难。我们使用过一种区分型模型——逻辑回归，但逻辑回归并不是一个序列模型，它只为单次的观察分类。然而，我们可以简单地通过将当前词的类别作为下一个词的特征之一传入，从而将逻辑回归转化为区分型序列模型。当我们如此应用逻辑回归时，该方法称为**最大熵马尔可夫模型(maximum entropy Markov model, MEMM)**。

给定词序列 $w_1^n$ 和标注序列 $t_1^n$，HMM通过贝叶斯公式计算使 $P(T|W)$ 取最大值的最佳标注序列：
$$
\hat{T}=\arg\max_{T}P(T|W)\\
=\arg\max_{T}P(W|T)P(T)\\
=\arg\max_{T}\prod P(w_i|t_i)\prod P(t_i|t_{i-1})\\
$$
但MEMM则直接计算后验，训练其区分可能的标注序列：
$$
\hat{T}=\arg\max_{T}P(T|W)\\
=\arg\max_{T}\prod P(t_i|w_i,t_{i-1})
$$
下图展示了HMM和MEMM计算方向上的差别：

![](https://i.loli.net/2021/01/19/RlavpwPOC2gjhSF.png)

## MEMM的特征

使用区分型序列模型的原因就是它可以更简单地统合非常多的特征，如下图所示：

> 因此HMM的所有计算都是基于两种概率 $P(tag|tag)$ 和 $P(word|tag)$，如果我们想要加入新的特征就必须把新的知识编码到这两种概率之中，使得结构更加复杂。

![](https://i.loli.net/2021/01/19/eQY48czrmGEOhTy.png)

一个基本的MEMM词性标注器使用当前词自身、邻近词、先前的标注以及各种组合，使用类似下面的特征模板：
$$
\lang t_i,w_{i-2}\rang,\lang t_i,w_{i-1}\rang,\lang t_i,w_i\rang,\lang t_i,w_{i+1}\rang,\lang t_i,w_{i+2}\rang\\
\lang t_i,t_{i-1}\rang,\lang t_i,t_{i-2},t_{i-1}\rang\\
\lang t_i,t_{i-1},w_i\rang,\lang t_i,w_{i-1},w_i\rang,\lang t_i,w_i,w_{i+1}\rang
$$
特征模板用于自动从训练集和测试集中提取

