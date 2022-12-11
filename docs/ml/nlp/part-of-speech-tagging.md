# 词性标注和命名实体检测

亚历山大里亚的狄俄尼索斯·特拉克斯（公元前 100 年），也可能是另外某个人，写了一篇希腊语的语法概述，总结了他那个时代的语言学知识。书中将词划分为了 8 种**词性（part of speech，POS）**：名词、动词、代词、介词、副词、连词、分词和冠词，这一分类成为了接下来 2000 年欧洲语言的描述基础。词性的超越两个千年的持久性展现了它们在人类语言模型中的中心地位。

专有名词是另一个重要且从古至今研究的语言学类别。相对于词性通常被赋给单个的词或语素，专有名词则经常是一整个多词短语，例如人名 `Marie Curie`，地名 `New York City` 或组织名 `Stanford University`。我们使用术语**命名实体（named entity）**来表示，粗略地讲，任何一个可以用一个专有名词来指代的事物：一个人，一个地点，一个组织，甚至可以扩展到本身不是实体的事物。

词性和命名实体是句子结构和含义的有用线索。知道是一个词是名词还是动词能够告诉我们可能的相邻词（英文中的名词之前是限定词或形容词，动词之前是名词）和语法结构（动词对名词有依赖性链接），使得词性标注成为解析的关键方面。知道一个类似于 `Washington` 的命名实体是一个人、一个地点还是一所学校的名称对于许多 NLP 任务（例如问答系统、立场检测或信息提取）都十分重要。

在本章中我们将介绍**词性标注（part-of-speech tagging）**任务，取一段词序列并为每个词赋予一个词性，以及**命名实体检测（named entity recognition）**任务，为词或短语赋予标签如 `PERSON`、`LOCATION` 或 `ORGANIZATION`。

我们为输入词序列中的每个词 $x_i$ 赋予一个标签 $y_i$，使得输出序列 $Y$ 与输入序列 $X$ 的长度相等，这样的任务被称为**序列贴标签（sequence labeling）**任务。我们将介绍传统的序列贴标签算法，生成式的**隐马尔可夫模型（Hidden Markov Model，HMM）**和区分式的**条件随机域（Conditional Random Fiel，CRF）**。在后面的章节我们还会介绍基于 RNN 和 transformer 的现代序列打标签器。

## 英文词类

尽管每一类词都具有语义上的倾向，但词类是根据它们与相邻词的语法关系或词缀的形态学特性来定义的。

词性可以分为两个大类：**封闭类（closed class）**和**开放类（open class）**。封闭类具有相对固定的成员，例如很少出现新的介词；相对地，名词和动词就是开放类，因为不断地有新的名词和动词被创造或借用。每个语料库都有不同的开放类词，但很可能共享一个封闭类词集。封闭类词通常是**功能词（function word）**，例如代词 `you`、`it`，介词 `of`、`in`，连接词 `and` 等，一般很短、出现频率高，并且有语法结构上的作用。

世界上的语言有四种主要的开放类：**名词（noun）**、**动词（verb）**、**形容词（adjective）**和**副词（adverb）**，以及一种规模更小的开放类：**感叹词（interjection）**。其特征整理如下：

**名词（noun）**

| 共通                                                                             | 英文                                     | 中文                                        | 日文                                    |
| -------------------------------------------------------------------------------- | ---------------------------------------- | ------------------------------------------- | --------------------------------------- |
| 包含人、地点、物品、组织、抽象概念等                                             | 包含动名词<br />e.g. tagging             | 多数动词可以直接用做名词<br />e.g. 生产基地 | 包含动词的第一连用形<br />e.g. 釣り合い |
| 可以笼统地理解为各种物                                                           | 有所有格<br />e.g. IBM’s annual revenue  | 有所有格<br />e.g. 谷歌的年营收             | 有所有格<br />e.g. IBMの年間収益        |
| 分为**专有名词（proper noun）**和**普通名词（common noun）**，前者是特殊的实体名 | 普通名词的大部分有单复数<br />e.g. goats |                                             |                                         |
|                                                                                  | 有限定词<br />e.g. the paper             |                                             |                                         |
|                                                                                  | 专有名词的首字母需要大写<br />e.g. Alice |                                             |                                         |

**动词（verb）**

| 共通           | 英文                                                                                         | 中文 | 日文                                                                                                                                                                                                                                                      |
| -------------- | -------------------------------------------------------------------------------------------- | ---- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 表示动作或过程 | 包含原形、单数第三人称形式、进行时、过去式、过去分词<br />e.g. eat, eats, eating, ate, eaten |      | 包含基本形（连体形）、第一连用形、第二连用形、过去式、未然形、假定形、命令形、意志形、可能形、被动形、使役形、被动使役形<br />e.g. 食べる、食べ、食べて、食べた、食べない、食べれば、食べろ、食べよう、食べられる、食べられる、食べさせる、食べさせられる |

**形容词（adjective）**

| 共通                 | 英文 | 中文 | 日文                                        |
| -------------------- | ---- | ---- | ------------------------------------------- |
| 描述事物的属性或品质 |      |      | 包含形容词和形容动词<br />e.g. 美しい、綺麗 |
| 一些语言没有形容词   |      |      |                                             |

**副词（adverb）**

| 共通                                                                                                                                  | 英文 | 中文 | 日文             |
| ------------------------------------------------------------------------------------------------------------------------------------- | ---- | ---- | ---------------- |
| 修饰动词或形容词                                                                                                                      |      |      | 即形容词的连用形 |
| 包括**方向副词（directional adverb）**、**时间副词（temporal adverb）**、**程度副词（degree adverb）**和**方法副词（manner adverb）** |      |      |                  |

**感叹词（interjection）**

感叹词是一个更小的开放类，其包含问候（`hello`、`goodbye`）和问题回复（`yes`、`no`、`uh-huh`）。

不同语言之间的封闭类的差异要比开放类大得多，一些重要的封闭类在英文中的表示为：

<pre>
    prepositions: on, under, over, near, by, at, from, to, with
    particles: up, down, on, off, in, out, at, by
    determiners: a, an, the
    conjunctions: and, but, or, as, if, when
    pronouns: she, who, I, others
    auxiliary verbs: can, may, should, are
    numerals: one, two, three, first, second, third
</pre>

**介词（preposition）**

英文的介词出现在名词之前，因此被称为 **preposition**。它们在语义上指示空间或时间关系，不论是字面意义的（`on it`、`before then`、`by the house`）还是比喻义的（`on time`、`with gusto`、`beside herself`），也指示其他关系。

**助词（particle）**

助词像是一个介词或副词，与动词结合使用，通常有扩展的词义。可以作为单独的句法或语义单元的动词和助词组合称为**动词短语（phrasal verb）**，动词短语的词义通常是**不能组合的（noncompositional）**，即不能根据动词和助词的词义来预测，例如 `turn down` 表示拒绝，`go on` 表示继续。

**限定词（determiner）**

一类与名词共同出现、标志名词短语开始位置的词称为限定词。限定词中的一类是**冠词（article）**，用于标记语段的属性，在英文中只有三个：`a`、`an`、`the`；其他的限定词包括 `this`、`that` 等。冠词在英文中出现得如此频繁，事实上，`the` 是大部分书面英文语料库中最频繁的词，`a` 和 `an` 也紧随其后。

**连词（conjunction）**

连词连接两个短语、从句或句子。**并列连词（coordinating conjunction）**，例如 `and`、`or`、`but`，连接两个地位等同的元素；**从属连词（subordinating conjunction）**表示元素之一有嵌入的状态。

**代词（pronoun）**

代词用作指代一些名词短语、实体或事件的速记表达。**人称代词（personal pronoun）**指代人或实体，例如 `I`、`you`、`me`、`it`；**物主代词（possesive pronoun）**指代人和某个对象的从属或其他抽象关系，例如 `my`、`your`、`his`；**wh 代词（wh-pronoun）**用于特定的疑问形式，或作为补语。

**辅助动词（auxiliary，auxiliary verb）**

辅助动词标示了主要动词的语义特征，例如时态、是否完成（体）、是否被否定（极性），以及动作是否是必要的、可能的、建议的或想要的（语气）。英文的辅助动词包含**系动词（copula）** `be`，动词 `do`、`have` 和它们的变形，以及用于标示由主要动词描述的事件相关联的语气的**情态动词（modal verb）** `may`、`can`、`should`、`must` 等。

英文的一个重要的标注集是 45-tag Penn Treebank 词性标注集（Marcus et al., 1993）(如下图所示)，已经被用于标注非常多的语料库，如 Penn Treebank 语料库。

![](https://s2.loli.net/2022/12/11/8R3iyfzF2H7s9WU.png)

下面是一些例子：

<pre>
​    There/EX are/VBP 70/CD children/NNS there/RB
​    Preliminary/JJ findings/NNS were/VBD reported/VBN in/IN today/NN ’s/POS
        New/NNP England/NNP Journal/NNP of/IN Medicine/NNP ./.
</pre>

被标注了词性的语料库是标注算法的重要训练和测试集，有三个被标注的语料库常被使用。**Brown**语料库长一百万词，由1961年在美国发表的500篇不同体裁的文章组成；**WSJ**语料库长一百万词，由1989年华尔街日报发表的文章组成；**Switchboard**语料库长两百万词，由1990-1991年搜集的通话记录组成。

## 词性标注

**词性标注（part-of-speech tagging）**是为输入文本的每个词赋予一个词性的过程。输入一个（分词过的）词序列 $x_1,x_2,\cdots,x_n$ 和一个标注集，输出一个标注序列 $y_1,y_2,\cdots,y_n$，长度与词序列相等，如下图所示。

![](https://s2.loli.net/2022/12/11/UgvBxayIOEQ1Mdl.png)

标注是一个**消歧义（disambiguation）**任务；词是**有歧义的（ambiguous）**，可能有多种可能的词性，而目标就是找到语境下的正确标注。例如 `book` 可以是动词或名词，`that` 可以是限定词或补语。

词性标注算法的正确率是极其高的。一项研究达成了来自 Universal Dependency treebank 的跨 15 种语言的超过 97% 的正确率（Wu 和 Dredze，2019）。在多个英文 treebank 上的正确率也达到了 97%（与算法无关，HMMs、CRFs 和 BERT 都表现接近）。97% 这个数字也差不多是人类在这项任务上的表现，至少对于英文是这样（Manning，2011）。

词性在多大程度上会有歧义？如下图所示，大部分的词（85~86%）是无歧义的，但词汇表中剩下的 14~15% 的词是非常常用的词，因此文本中55~67%的token都是歧义的。

![](https://i.loli.net/2021/01/18/yNwzAvHRYi5jDqg.png)

最有歧义的常用词包含that, back, down, put, set, etc，下列是back的6种不同词性的示例：

<pre>
    earnings growth took a back/JJ seat
    a small building in the back/NN
    a clear majority of senators back/VBP the bill
    Dave began to back/VB toward the door
    enable the country to buy back/RP about debt
    I was twenty-one back/RB then
</pre>

最简单的基线算法是对于任意一个给定的歧义词，选择该词在训练集中频率最高的标注。在WSJ语料库上的训练和测试显示，这样一个基线能够达到92.34%的准确率；作为对比，最先进的标注方法（HMMs, MEMMs, 神经网络, 基于规则的算法）在该数据集上可以达到97%的准确率。

## HMM词性标注

**HMM（Hidden Markov Model）**是一个同步的序列到序列模型，也是一个概率模型，计算几种可能的标签序列的概率并选择最佳的序列。

HMM基于增强的马尔可夫链，hidden表示标注并非直接观察到，而是通过词序列推断得到。

### HMM模型

HMM由以下部分组成：

* $Q=\{q_1,q_2,\cdots,q_N\}$ ：N个**状态（state）**的集合
* $A=\begin{bmatrix}a_{11}&\cdots\\\cdots&a_{NN} \end{bmatrix}$ ：**状态转移矩阵（transition probability matrix）**，其中 $a_{ij}$ 代表从状态 $i$ 转移到状态 $j$ 的概率
* $O=o_1o_2\cdots o_T$ ： $T$ 次**观察（observation）**的序列，每一个 $o_i$ 都属于词汇表 $V$ 
* $B=b_i(o_t)$ ：**观察似然（observation likelihood）**的序列，也称为**发射概率（emission probabilities）**，表示观察 $o_t$ 产生于状态 $q_i$ 的概率
* $\pi = \pi_1, \pi_2,\cdots, \pi_N$ ：初始概率分布， $\sum\pi_i =1$ 

一阶的HMM使用了两个简化假定：第一，当前状态的概率取决于前一状态

$$
P(q_i|q_1\cdots q_{i-1})=P(q_i|q_{i-1})
$$

第二，观察 $o_i$ 的概率仅取决于产生该观察的状态 $q_i$，而不是其他的状态或观察

$$
P(o_i|q_1\cdots q_To_1\cdots o_T)=P(o_i|q_i)
$$

### HMM标注器的组成

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

### HMM 标注作为解码

HMM 的**解码（decoding）**过程定义为：给定HMM $\lambda=(A,B)$ 和观察序列 $O=o_1o_2\cdots o_T$，寻找最可能的状态序列 $Q=q_1q_2\cdots q_T$。

对于词性标注任务，HMM解码就是在给定观察序列 $w_1^n$ 的条件下，选择最可能的标注序列 $t_1^n$ ：

$$
\hat t_1^n=\arg\max_{t_1^n} P(t_1^n|w_1^n)
$$

使用贝叶斯公式转化为

$$
\hat t_1^n=\arg\max_{t_1^n} \frac{P(w_1^n|t_1^n)P(t_1^n)}{P(w_1^n)}=\arg\max_{t_1^n} P(w_1^n|t_1^n)P(t_1^n)
$$

HMM标注器使用的两个简化假定为：每个词的出现概率仅取决于其自身的标注，而与附近的词和标注无关；标注仅取决于前一个标注，称为二元序列假定：

$$
P(w_1^n|t_1^n)=\prod_{i=1}^n P(w_i|t_i)\\
P(t_1^n)\approx \prod_{i=1}^n P(t_i|t_{i-1})
$$

应用上述假定，有

$$
\hat t_1^n=\arg\max_{t_1^n} P(w_1^n|t_1^n)P(t_1^n)\approx \arg\max_{t_1^n} \prod_{i=1}^n P(w_i|t_i)P(t_i|t_{i-1})\\
\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad {\rm emission}↑\quad\quad\quad ↑{\rm transition}
$$

此公式完全对应先前定义的 A 转移概率和 B 发射概率。

### Viterbi算法

HMM 的解码算法称为 Viterbi 算法，如下图所示：

![](https://i.loli.net/2021/01/19/AHUWbgRNpeX9imD.png)

Viterbi 算法首先建立一个概率矩阵，称为 lattice，其中每列代表一个观察，每行代表一个状态。下图展示了句子 `Janet will back the bill` 的 lattice：

![](https://i.loli.net/2021/01/19/u8GqN4JQpKmZMUY.png)

lattice的每个元素 $v_t(j)$，代表在已知观察序列 $o_1\cdots o_t$ 和可能的状态序列 $q_1\cdots q_{t-1}$ 的条件下， $q_t=j$ 的最大概率，形式上表示为

$$
v_t(j)=\max_{q_1\cdots q_{t-1}}P(q_1\cdots q_{t-1}o_1\cdots o_t,q_t=j|\lambda)
$$

最可能的路径是对所有可能的先前状态序列取最大值，就像其他动态规划算法一样，Viterbi迭代地计算每个元素。若已知 $t-1$ 时刻所有状态的概率 $v_{t-1}(j)$，则

$$
v_t(j)=\max_{i=1}^N v_{t-1}(i)a_{ij}b_j(o_t)
$$

其中3个因子分别为

* $v_{t-1}(i)$ ：前一时刻的 **Viterbi 路径概率**
* $a_{ij}$ ：从状态 $q_i$ 转移到 $q_j$ 的**转移概率**
* $b_j(o_t)$ ：给定状态 $j$ 条件下观察 $o_t$ 的**状态观察似然**

### 示例

以标注句子Janet will back the bill为例，正确标注（目标）为

<pre>
    Janet/NNP will/MD back/VB the/DT bill/NN
</pre>

HMM由下图的两个矩阵定义。矩阵数据来源于WSJ语料库中的计数（稍作了简化）。

![](https://i.loli.net/2021/01/19/bzV1KS5fD6slkAq.png)

下图在图8.6的基础上增加了计算过程，注意 $\pi$ 表示序列开始符SOS：

![](https://i.loli.net/2021/01/19/whYzE4XABOGecCa.png)

### 扩展 HMM 算法到三元序列

实际的 HMM 标注器会在上述简单模型的基础上做一些扩展，其中之一就是更宽的标注上下文。上述标注器中一个标注的概率仅取决于前一个标注，而实践中会使用更多历史标注。

将算法从二元序列扩展到三元序列可以小幅（大概0.5个百分点）提升性能，但需要对 Viterbi 算法做出非常大的改变。图8.7代表的转移概率矩阵会增加到 $N^2$ 行，而对于 lattice 中的每个元素需要计算前两列的所有路径组合中的最大值。

除了增加上下文宽度，HMM 标注器还有一些其他的高级特征，例如使用序列结束符 EOS 告知句子的结束位置。SOS 和 EOS 与任何常规的词都不同，因此需要使用特殊的标注“句子边界”。

使用三元序列标注器的一个问题是数据的稀疏性。就像语言模型一样，根据最大似然估计

$$
P(t_i|t_{i-1}t_{i-2})=\frac{C(t_{i-2}t_{i-1}t_i)}{C(t_{i-2}t_{i-1})}
$$

许多测试集中出现标注序列 $t_{i-2},t_{i-1},t_i$ 没有出现在训练集，那么这样一个三元序列的概率就是 0，即永远不会发生。和语言模型一样，标准的解决方法是插值：

$$
{\rm trigram}\quad \hat{P}(t_i|t_{i-1}t_{i-2})=\frac{C(t_{i-2}t_{i-1}t_i)}{C(t_{i-2}t_{i-1})}\\
{\rm bigram}\quad \hat{P}(t_i|t_{i-1})=\frac{C(t_{i-1}t_i)}{C(t_{i-1})}\\
{\rm unigram}\quad \hat{P}(t_i)=\frac{C(t_i)}{N}\\
P(t_i|t_{i-1}t_{i-2})=\lambda_3\hat{P}(t_i|t_{i-1}t_{i-2})+\lambda_2\hat{P}(t_i|t_{i-1})+\lambda_1\hat{P}(t_i)
$$

其中 $\lambda_1+\lambda_2+\lambda_3=1$。 $\lambda$ 的值由**删除插值法（deleted interpolation)**确定(Jelinek and Mercer, 1980)，算法如下图所示。

![](https://i.loli.net/2021/01/19/nFr1LsflQtNeci4.png)

> 考虑上面的三种情形：（需要实验验证）
>
> * $\frac{C(t_1,t_2,t_3)-1}{C(t_1,t_2)-1}$ 最大，例如等于1，那么 $t_1,t_2\to t_3$，表示这是一个连接紧密的三元序列
> * $\frac{C(t_2,t_3)-1}{C(t_2)-1}$ 最大，表示 $t_2t_3$ 是一个比较紧密的二元序列
> * $\frac{C(t_3)-1}{N-1}$ 最大，表示 $t_3$ 倾向于是单独出现

### 束搜索

Viterbi算法的复杂度为 $O(N^2T)$，对于三元序列的情形增加至 $O(N^3T)$，因此当状态数增加到较大时，Viterbi算法会变得很慢。

一种常用的方法是使用**束搜索（beam search)**解码。在束搜索中，我们计算 $t$ 时刻所有可能状态的概率，但仅保留其中最好的几个状态，其余状态将直接去除。实现束搜索的最简单方法是保留固定数量的状态，该数量称为**束宽（beam width)** $\beta$，下图展示了 $\beta=2$ 的情形。同样也可以设定束宽为状态数的固定比例，或设定概率阈值等。

![](https://i.loli.net/2021/01/19/i71rKFtGVUZMIkd.png)

### 未知词

为了实现词性标注器的高准确率，模型处理未知词的能力也十分重要。专有名词总是不断涌现，新的名词和动词也快速进入各种语言。一种有用的识别词性的特征是词形：首字母大写的词为专有名词。

但是对于猜测未知词的词形，最大的信息来源是词法学：`-s` 可能是名词复数，`-ed` 可能是动词过去式，`-able` 可能是形容词等等。我们在训练过程中建立词后缀（定长）与标注的统计关系，即计算给定长度为 $i$ 的后缀的条件下标注 $t_i$ 的概率（Samuelsson 1993, Brants 2000)：

$$
P(t_i|l_{n-i+1}\cdots l_n)
$$

可以使用后退法，即逐次使用更短的后缀来平滑概率。

由于未知词不太可能属于封闭类，可以只计算训练集中词频不大于 10 的词的后缀概率（因此不包含封闭类词以及使用方法灵活的常用词）。对于首字母大写和小写的词的处理方式也不一样。

结合所有特征，一个三元序列 HMM 标注器可以在 Penn 树库上达到 96.7% 的标注准确率，仅略逊于最好的 MEMM 和神经标注器的表现。

## 最大熵马尔可夫模型

尽管HMM可以达到很高的准确率，但我们看到它需要一些结构上的改进以处理未知词、后退、后缀等等。我们希望能用一种简洁的方式增加任意的特征，但这对于HMM这样的生成型模型而言十分困难。我们使用过一种区分型模型——逻辑回归，但逻辑回归并不是一个序列模型，它只为单次的观察分类。然而，我们可以简单地通过将当前词的类别作为下一个词的特征之一传入，从而将逻辑回归转化为区分型序列模型。当我们如此应用逻辑回归时，该方法称为**最大熵马尔可夫模型(maximum entropy Markov model，MEMM)**。

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

### MEMM的特征

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
