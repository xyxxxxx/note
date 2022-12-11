# 嵌入

出现在相似上下文中的词倾向于拥有相似的含义。这种词的分布的相似性与词的含义的相似性之间的关联称为**分布假定（distributional hypothesis）**。该假定于 1950 年代由语言学家 Joos（1950）、Harris（1954）和 Firth（1957）首次提出，他们注意到同义词倾向于出现在相同的语境中，而两个词的含义的差异程度就大致对应了它们的语境的差异程度。

本章我们介绍**向量语义（vector semantics）**，其实例化了上述语言学假定，直接从词在文本中的分布学习到它们的含义的表示，称为**嵌入（embedding）**。这些表示被用在每一个使用含义的 NLP 应用中，而我们这里要介绍的**静态嵌入（static embedding）**也构成了更加强大的动态或**上下文化嵌入（contextualized embedding）**（例如 BERT）的基础。

## 词语义

我们应当如何表示词的含义？在之前的 n 元序列模型和传统 NLP 应用中，我们对于词的表示就是词的字符串或者词汇表中的索引号。这种表示和哲学领域的一个习惯十分相似，即以小号大写字母拼写词来表示词的含义，例如猫的含义表示为 ᴄᴀᴛ，狗的含义表示为 ᴅᴏɢ。

但是这样的表示并不能让人满意，你可能看过下面这个古老的哲学笑话：

<pre>
    Q: What’s the meaning of life?
    A: ʟɪꜰᴇ
</pre>

我们当然能做得更好！我们想要的词义表示模型应当能够告诉我们，一些词具有相似的含义（例如“猫”和“狗”）而一些词具有相反的含义（例如“冷”和“热”），一些词具有褒义（例如“鼓励”）而一些词具有贬义（例如“怂恿”），一些词是从不同角度对同一个事件的描述（例如“买”、“卖”和“付款”）。

更普遍地，词义模型应能够做出有用的推断，以帮助我们解决词义相关的任务如问答、总结、检测短语或俗语、对话等。

在这一节中我们先总结一些需要的概念，利用对于词义的语言学研究（称为**词汇语义学（lexical semantics）**）的结果。

**词元和词义**

我们从词典中某个词的定义开始：

<pre>
    mouse (N)
    1. any of numerous small rodents...
    2. a hand-operated device that controls a cursor...
</pre>

这里的 `mouse` 就是**词元（lemma）**，也称为**词典形（citation form）**。`mouse` 是 `mice` 的词元，并且词典中也没有`mice` 这一屈折形式。类似地，`sing` 是 `sing`、`sang`、`sung` 的词元。在许多语言如西文中，不定式作为动词的词元，例如 `dormir` 是 `duermes` 的词元。`mice`、`sang`、`sung`则称为（词元的不同）**词形（wordform）**。

上面这个例子显示，同一词元可以有多个含义，我们将每一种含义称为一个**词义（word sense）**。词元的多义性可能使解释变得困难，**词义消歧（word sense disambiguation）**任务就是确定在特定上下文中一个词使用何种词义。

!!! note "说明"
    这里将 word sense 翻译为“词义”，word meaning 也翻译为“词的含义”或“词义”（在尽量避免歧义的条件下）。

**同义词/近义词**

关于词义很重要的一点是不同词义之间的关系。例如当两个词具有相同（或几乎相同）的词义时，我们称它们为**同义词（或近义词, synonymy）**，例如下面的词对：

<pre>
    couch/sofa    vomit/throw up    car/automobile
</pre>

同义词的一个更加正式的定义是，两个词是同义词，当它们可以任何句子中互相替换而不改变句子的真值条件。在这种情况下，我们经常称这两个词有相同的**命题意义（propositional meaning）**。

但就算同义词的替换不改变句子的真值，这些词在词义上也不是完全相同——事实上，几乎没有两个词的词义是完全相同的。语义学的基本原理之一，**对比原理（principle of contrast）**认为语言形式上的差异必然（或多或少地）带来含义上的差异。例如，$H_2O$ 常在化学、生物等学科的语境中使用，而不适合出现在大多数日常情景中。这种体裁（风格/使用场景）上的区别也是词义的一部分。因此在实践中，同义词（synonymy）更接近于描述词义相近的关系（即近义词）。

!!! note "说明"
    下文将不再区分“同义词”和“近义词”。

**词相似度**

尽管很少有词有很多同义词，但大多数词都有很多**相似（similar）**词。“猫”和“狗”不是同义词，但在一定程度上是相似词。从同义到相似，我们从处理词义的关系转换到处理词的关系（即相似度）。处理词避免了必须对词义采用特殊的表示，因而简化了我们的任务。

**相似度（similarity）**这个概念在大型的语义任务中十分有用。已知两个词的相似程度有助于计算两个短语或句子的相似程度，而这些是自然语言理解任务中的重要组成部分。一种获得词相似度的定量结果的方法是请人进行打分，一些数据集就是从这样的实验中得到，例如 SimLex-999 数据集（Hill 等人，2015）给出了从 0 到 10 的相似度度量，下面是一些例子：

![](https://i.loli.net/2021/01/05/Af4EaWxjuiTwbGZ.png)

**词相关性**

两个词的词义除了相似之外还会在其他方面相关，这当中的一种关系就是词**相关性（relatedness）**（Budanitsky
and Hirst, 2006），在心理学领域中习惯称为**关联（association）**。

考虑词“茶杯”和“咖啡”的含义，它们并不相似（即没有共同的特征），但它们明显相关（即共同参与到喝咖啡这样一个日常事件中）。类似地，“外科医生”和“手术刀”不相似，但相关。

一种常见的词之间的相关性是它们同属一个**语义场（semantic field）**。所谓语义场就是涉及一个特定语义情景并且有结构性关联的一组词，例如医院的语义场（“外科医生”，“手术刀”，“护士”，“麻醉”，“医院”），餐厅的语义场（“服务员”，“菜单”，“盘子”，“食物”，“厨师”），或房屋的语义场（“门”，“屋顶”，“厨房”，“家庭”，“床”）。

语义场也和**话题模型（topic model）**有关，话题模型是在大型文本集上进行无监督学习以推断出若干组相关词。语义场和话题模型都是对于发现文本的话题结构非常有用的工具。

**语义框架和语义角色**

**语义框架（semantic frame）**的概念和语义场关系紧密。所谓语义框架就是代表一类特定事件中的不同角度或不同参与者的一组词。例如，商业交易就是一类事件，其中一个实体用金钱向另一实体交易商品或服务，在这之后商品易手或服务实行。这一事件可以用动词编码为买（从甲方的角度），卖（从乙方的角度），付款（着眼于钱的交付）等。框架内包括**语义角色（semantic role）**（例如甲方、乙方、商品、金钱），而词可以在句子中扮演这些角色。

搭建了上述语义框架后，模型就能够知道“甲从乙买书”可以转述为“乙卖书给甲”，并且甲扮演了买方而乙扮演了卖方。这种释义的能力在问答系统中非常重要，并且也能帮助机器翻译进行角度转换。

**褒贬**

最后，词包含情感含义或**褒贬（或隐含义, connotation）**。例如一些词具有**褒义（positive connotation）**（例如“不错”、“喜欢”）而另一些词具有**贬义（negative connotation）**（例如“糟糕”、“讨厌”）。即使是含义在某些方面相似的词也有可能在褒贬上不同。语言中包含的积极或消极评价称为**情感（sentiment）**（例如“爱”、“恨”），而词的情感在情感分析、立场识别，以及 NLP 在政治语言、顾客评价的应用等重要任务上都扮演了重要角色。

在情感含义上的早期工作（Osgood 等人，1957）发现词在情感含义上有三个维度的变化，通常称为：

* **valence**：刺激的愉悦程度
* **arousal**：刺激唤起的情感强度
* **dominance**：刺激执行的控制程度

因此“高兴”和“满意”的 valence 高，而“难过”和“恼怒”的 valence 低；“激动”和“疯狂”的 arousal 高，而"放松"和“平静”的 arousal 低；“重要”和“控制”的 dominance 高，而“敬畏”和“影响”的 dominance 低。每个词被表示为 3 个数字，分别对应它在这 3 个维度上的值，例如：

![](https://i.loli.net/2021/01/06/Q4WXN8tb6DGrTeJ.png)

Osgood 等人（1957）注意到使用这 3 个数字表示一个词，则模型将所有词表示为三维空间中的一个点，词与三维空间中的向量一一对应。这种革命性的想法就是后面将要介绍的向量语义模型的雏形。

## 向量语义

那么我们如何建立一个计算模型，使得它可以处理词的各种词义？找到一个完美的、可以全方位地处理词的所有词义的模型是困难的，目前的最佳模型依然是向量语义模型，其灵感可以追溯到 1950 年代的语言学和哲学工作（即上文提到的 Osgood 等人和 Joos 等人的工作）。

当时，哲学家 Ludwig Wittgenstein 因为质疑为所有词的含义建立一个完全的形式理论的可能性，而提出“词的含义就是词在语言中的用法”（Wittgenstein, 1953, PI 43）。也就是说，除了从逻辑上定义每个词，还可以根据人们对词的使用和理解进行定义。

看一个说明上述方法的例子。假设你不知道粤语词 `ongchoy` 的含义，但是你看到它出现在一些句子和上下文中：

!!! note "说明"
    这是一个外来语的例子。在英文语境中，`ongchoy` 或 `Ong Choy` 是一个外来语，其来自于粤语的“蕹菜”，中文也称为“空心菜”。

* ongchoy 和蒜一起炒很好吃
* ongchoy 和米饭十分搭配
* ……ongchoy 叶子和咸酱……

再假设你看过一些其他的词出现在相似的上下文中：

* ……蒜炒菠菜盖饭……
* ……莙荙菜（chard）的根和叶子很好吃……
* ……羽衣甘蓝（collard greens）和其他咸的叶子菜……

`ongchoy` 与“米饭”、“蒜”、“好吃”和“咸”出现在一起，就如同“菠菜”、“莙荙菜”和“羽衣甘蓝”一样，因此读者可以推测 `ongchoy` 也是一种类似的叶子菜。

我们在计算中可以采用同样的方法：计数出现在 `ongchoy` 的上下文的词，我们会发现“炒”、“吃”、“蒜”之类的词，而类似的词也出现在“菠菜”或“羽衣甘蓝”的上下文中，这一现象可以帮助发现 `ongchoy` 和这些词的相似性。

因此，向量语义结合了两种直觉：向量直觉和用法直觉，将词 $w$ 的含义表示为高维语义空间的一个点。向量语义有非常多的版本，每一种对于向量的值的确定方法都略有不同，但都与上下文的词的计数有关。

用向量表示词称为**嵌入（embedding）**，因为词嵌入在一个特殊的向量空间中。下图展示了情感分析任务中学习到的嵌入的可视化，即将一些选定的词从原有的 60 维空间投影到 2 维空间的位置。

![](https://i.loli.net/2021/01/06/82aBNRUkMCHq73z.png)

可以看到正面和负面词（也和中立功能词）落在了空间的不同部分。这显示了向量语义的一个重要的优点：提供了一个细粒度的可以实现词相似度的词义模型。在基于朴素贝叶斯方法的情感分类器中，如果测试集文本中的情感词未曾出现在训练集中，则无法提供任何信息；但如果对词进行嵌入，则分类器可以为未知词赋予情感，只要它见过一些有相似含义的词。向量语义模型也十分切实可行，因为可以自动学习而无需复杂的人工贴标签和监督。

由于上述优点，向量语义模型已经成为 NLP 表示词义的标准方法。本章中我们将介绍两种最常用的模型：其一是 **tf-idf** 模型，通常用作基线，其得到的向量长而稀疏；其二是 **word2vec** 模型，其发展而来的一族模型可以构建更短而稠密的具有良好语义性质的向量。

## 词和向量

### 向量和文本

我们通常使用**共现矩阵（co-occurence matrix）**来表现词共同出现（co-occur）的频率，例如**词-文本矩阵（term-document matrix）**的每一行代表一个词而每一列代表一个文本。下图展示了 4 个选定的词在莎士比亚的 4 场戏剧中的出现次数：

![](https://i.loli.net/2021/01/06/CulYMhXTjdEINez.png)

这里每个文本对应一个列向量，每个维度代表一个词。最初词-文本矩阵被定义为**信息检索（Information Retrieval，IR）**任务中寻找相似文本的方法，因为相似文本有相似的词，因而有相似的向量。

实际上，信息检索任务是从一个文本集中找到查询 $q$ 的最佳匹配文本 $d$。因为查询 $q$ 也表示为一个 $|V|$ 维向量，因此，我们需要计算向量相似度的方法。后面我们将会介绍 tf-idf 词权重和余弦相似度。

### 词作为向量

类似地，图 6.2 中的每个词对应一个行向量，每个维度代表一个文本。相似的词出现在相似的文本中，因而有相似的向量。

但是更常用的方法是使用**词-词矩阵（term-term matrix, word-word matrix）**，这是一个 $|V|\times |V|$ 矩阵，每个元素记录该列的词出现在该行的词（目标词）的上下文的频率。上下文可以是整个文本，那么元素值代表两个词出现在同一个文本的次数（同一文本多次出现应求乘积）；但最常用的方法是使用目标词附近的一个窗口，例如从目标词往左数 4 个词，往右数 4 个词，称为一个 ±4 词窗口，那么元素值代表该列的词出现在该行的词的这一窗口中的次数。下面展示了一些例子：

<pre>
​    is traditionally followed by <b>cherry</b> pie, a traditional dessert
​    often mixed, such as <b>strawberry</b> rhubarb pie. Apple pie
​    computer peripherals and personal <b>digital</b> assistants. These devices usually
    a computer. This includes <b>information</b> available on the internet
</pre>

如果我们对每一个词的每一次出现都对其上下文的词进行计数，就能得到一个词-词矩阵。下图展示了一个简化的词-词矩阵的一部分，计算自维基百科语料库（Davies, 2015）。可以看到词 `cherry` 和 `strawberry` 比较相似（`pie` 和 `sugar` 都频繁出现在上下文窗口中），而 `digital` 和 `information` 比较相似。下图绘制了 `digital` 和 `information` 的两个分量，意在说明此种表示下余弦相似度的合理性。

![](https://i.loli.net/2021/01/06/FBUsyj5Omh9efQ6.png)

注意词-词矩阵的向量维度 $|V|$ 通常在 10000~50000 之间（选择训练语料库中词频最高的词；保留词频在第 50000 之后的词一般没有帮助）。词-词矩阵也是一个稀疏矩阵，可以使用对于稀疏矩阵更高效的存储和计算算法。

## 余弦相似度

计算两个词的相似度也就是对它们的向量进行运算。目前为止最常用的相似度指标是向量夹角的余弦值：

$$
\cos(v,w)=\frac{vw}{|v||w|}=\frac{\sum v_iw_i}{\sqrt{\sum v_i^2}\sqrt{\sum w_i^2}}
$$

对于一些应用，我们预先归一化每一个向量，通过除以它的长度，来创建长度为 1 的单位向量。对于单位向量，点积等于余弦相似度。

再来对这个部分词-词矩阵计算词相似度：

![](https://i.loli.net/2021/01/06/A53sPiJoyKTcxtO.png)

注意在这种表示下所有的分量都是非负整数，因此余弦相似度的取值范围为 [0,1] 区间。

## TF-IDF

如图 6.5 所示的词-词矩阵包含了词共同出现的原始频率，但这样一个频率并不是词之间关系的最佳描述，一个问题就是原始频率非常有偏向（skewed）而不太有区分性（discriminative）。例如当我们考虑何种上下文为 `cherry` 和 `strawberry` 共有而不为 `digital` 和 `information` 共有时，我们当然不会认为 `the`、`it` 或 `they` 这样的词具有好的区分度，因为它们会频繁出现在所有词的上下文中。

换言之，在目标词的上下文中频繁出现的词是重要的，但在所有词的上下文中都频繁出现的词又是不重要的，因此我们就要解决这个矛盾。有两种常用的方法：在这一节我们介绍 tf-idf 加权（weighting），通常在维度为文本时使用；在下一节我们将介绍 PPMI 算法，通常在维度为词时使用。

tf-idf 加权计算两项的积，这两项分别代表上述的两种直觉。

第一项称为**词频（term frequency）**（Luhn，1957），即词 $t$ 在文本 $d$ 中的频率，我们可以直接使用原始计数：

$$
{\rm tf}_{t,d}={\rm count}(t,d)
$$

实践中通常采用对数对计数进行压缩。这里的思路是一个词在文本中出现了 100 次，不代表这个词就与该文本的含义 100 倍地相关。由于不能取 0 的对数，我们对所有计数加一：

$$
{\rm tf}_{t,d}=\lg({\rm count}(t,d)+1)
$$

第二项为那些只在部分文本中出现的词赋予更高的权重，因为这些词在区分这些文本和其他文本时十分有用；相对而言，那些在整个集合中频繁出现的词就作用不大。**文本频率（document frequency）** ${\rm df}_t$ 指词 $t$ 出现在多少个文本中，**集合频率（collection frequency）**则指词 $t$ 在整个集合中的出现次数。例如莎士比亚的 37 部戏剧中，`Romeo` 和 `action` 有相同的集合频率，但有完全不同的文本频率：

![](https://i.loli.net/2021/01/06/X7tKyzewFVpuJWQ.png)

因此当我们的目标是寻找关于罗密欧的浪漫试炼的文本时，词 `Romeo` 就应该有相当大的权重。我们通过 **idf（inverse document frequency）**词权重来强调这些有区分度的词，定义为 $N/{\rm df}_t$，其中 $N$ 是集合中的文本总数，而 ${\rm df}_t$ 是词 $t$ 的文本频率。${\rm df}_t$ 越高，则权重越低；当 ${\rm df}_t$ 达到 $N$ 时，权重最低取 1。

由于许多集合的文本数量非常多，我们同样对 idf 值取对数，即：

$$
{\rm idf}_t=\lg (\frac{N}{{\rm df}_t})
$$

下面是莎士比亚语料库中一些词的 df 和 idf 值，可以看到像 `Romeo` 这样仅出现在一部戏剧中的词最有信息量，因而 idf 取到最大值；而像 `good` 和 `sweet` 这样出现在全部 37 部戏剧中的词则完全没有区分度，因而 idf 值直接取 0。

![](https://i.loli.net/2021/01/06/XuKi8R12IOfQ76T.png)

tf-idf 值是上面两项的乘积，即

$$
w_{t,d}={\rm tf}_{t,d}\times {\rm idf}_t
$$

为图 6.2 的莎士比亚词-文本矩阵应用 tf-idf 加权后的结果如下图，可以看到词 `good` 这一行的 tf-idf 值为 0（出现在 37/37 部戏剧中，idf=0），而 `fool` 这一行的 tf-idf 值也非常小（出现在 36/37 部戏剧中，idf=0.0119）。

![](https://i.loli.net/2021/01/06/GawyHiDJtqYrmR9.png)

tf-idf 加权是为信息检索的共现矩阵加权的一种方法，但也应用在 NLP 的其他许多地方。它也是一个好的基线，是首次尝试的选择。

## 点间互信息

正点间互信息（positive pointwise mutual information，PPMI）是 tf-idf 之外的另一个加权函数，用于词-词矩阵。PPMI 基于以下直觉：衡量两个词之间的关联的最佳方法，是询问这两个词在语料库中共同出现的频繁程度在多大程度上高于先验给出的它们偶然共同出现的频繁程度。

**点间互信息（pointwise mutual information，PMI）**（Fano，1961）是 NLP 中最重要的概念之一。它是对于两个事件 $x$ 和 $y$ 发生的频繁程度的度量，当它们相互独立时有：

$$
I(x,y)=\log_2\frac{P(x,y)}{P(x)P(y)}
$$

一个目标词 $w$ 和一个上下文词 $c$ 之间的点间互信息（Church and Hanks，1989）于是被定义为：

$$
{\rm PMI}(w,c)=\log_2\frac{P(w,c)}{P(w)P(c)}
$$

分子告诉我们观察到这两个词在一起的频繁程度（假定我们使用最大似然估计计算概率），分母告诉我们假定这两个词独立出现的情况下它们共同出现的频繁程度。因此，这个比值提供了一个对于这两个词在多大程度上比随机情况下共同出现得多的估计。不论何时只要我们需要找到强关联的词，PMI 都是一个有用的工具。

PMI 的取值范围从负无穷到正无穷，但负的 PMI 值（表示共同出现得比随机情况下少）往往是不可靠的，除非我们的语料库特别大。要分辨出两个单独出现的概率各为 $10^{-6}$ 的词同时出现得比随机情况下少，我们需要确定它们同时出现的概率明显小于 $10^{-12}$，而这样的粒度需要一个巨大的语料库。而且，以人类的判断来评估这样的“无相关性”分数是否可行也尚不明确。因此，我们更常使用将所有负值替换为 0 的正 PMI（称为 PPMI）：

$$
{\rm PPMI}(w,c)=\max(\log_2\frac{P(w,c)}{P(w)P(c)}, 0)
$$

更形式化地，假定我们有一个 W（word）行 C（context）列的共现矩阵 F，其中 $f_ij$ 给出词 $w_i$ 与上下文词 $c_j$ 共同出现的次数。它可以转换为一个 PPMI 矩阵，其中 ${\rm PPMI}_{ij}$ 给出词 $w_i$ 与上下文词 $c_j$ 的 PPMI 值如下：

$$
\displaylines{
p_{ij}=\frac{f_{ij}}{\sum_{i}\sum_{j}f_{ij}},p_{i*}=\frac{\sum_{j}f_{ij}}{\sum_{i}\sum_{j}f_{ij}},p_{*j}=\frac{\sum_{i}f_{ij}}{\sum_{i}\sum_{j}f_{ij}}\\
{\rm PPMI}_{ij}=\max(\log_2\frac{p_{ij}}{p_{i*}p_{*j}},0)
}
$$

让我们来看一些计算实例。使用如下矩阵，并且为便于计算，假定只有这些词和上下文词是重要的：

![](https://s2.loli.net/2022/12/06/XwcopAbKiEtkQd2.png)

例如我们计算 PPMI(information,data)：

$$
\displaylines{
P(w={\rm information},c={\rm data})=\frac{3982}{11716}=.3399\\
P(w={\rm information})=\frac{7703}{11716}=.6575\\
P(c={\rm data})=\frac{5673}{11716}=.4842\\
{\rm PPMI}({\rm information,data})=\log_2(.3399/(.6575*.4842))=.0944
}
$$

下图展示了从图 6.10 的计数计算得到的联合概率，以及 PPMI 值。不意外地，`cherry` 和 `strawberry` 与 `pie` 和 `sugar` 高度关联，而 `data` 与 `information` 低度关联。

![](https://s2.loli.net/2022/12/06/GLQa9FirVveBo7p.png)

PMI 具有偏向于低频事件的问题，非常罕见的词往往会有非常高的 PMI 值。减少这种偏差的一种方法是稍微改变 $P(c)$ 的计算方法：

$$
\displaylines{
{\rm PPMI}_\alpha(w,c)=\max(\log_2\frac{P(w,c)}{P(w)P_\alpha(c)}, 0)\\
P_\alpha(c)=\frac{{\rm count}(c)^{\alpha}}{\sum_c {\rm count}(c)^{\alpha}}
}
$$

Levy 等人（2015）发现设 $\alpha=0.75$ 提升了在各种任务中嵌入的表现。这一方法有用是因为设 $\alpha=0.75$ 提高了罕见上下文词的概率，从而降低了它们的 PMI。

另一种可能的解决方法是拉普拉斯平滑：在计算 PMI 之前，为每个计数增加一个小的常数 $k$（取值通常在 0.1～3 之间），从而折减了所有的非零值。$k$ 越大，则非零值折减得越多。

## tf-idf 和 PPMI 向量模型的应用

总而言之，我们目前的向量语义模型是将目标词表示为一个向量，其维度对应于一个大集合中的文本（词-文本矩阵）或某个邻近窗口的词计数（词-词矩阵）。每一维的值都是计数，由 tf-idf（对于词-文本矩阵）或 PPMI（对词-词矩阵）加权，并且非常稀疏（大部分的值都是 0）。

模型计算两个词 $x$ 和 $y$ 之间的相似度，通过计算它们的 tf-idf 或 PPMI 向量的余弦相似度；高余弦相似度，高相似度。这整个模型有时被称为 tf-idf 模型或 PPMI 模型。

tf-idf 模型经常用于文本函数，例如判断两个文本是否相似。我们可以将文本表示为文本中所有词的向量的平均值，或者说**中心（centroid）**，然后就可以计算余弦相似度。

!!! note "说明"
    一个向量集合的中心对于其到集合中所有向量的距离（欧氏距离）平方之和取到最小值。

文本相似度也用于诸多任务中，例如信息检索、查重（plagiarism detection）、新闻推荐系统等。

tf-idf 模型和 PPMI 模型都可以用于计算词相似度，用于寻找词的释义、追踪词义的变化、自动发现不同语料库中的词义等任务中。例如我们可以通过计算任何目标词 $w$ 与其他 $|V-1|$ 个词的余弦相似度，来计算与 $w$ 最相似的 $k$ 个词。

## word2vec

本节我们将介绍另一种更强大的词表示法，嵌入到更短且稠密的向量。向量更短：维数 $d$ 在 50～1000 之间，而不是词汇表规模 $|V|$ 或文本数 $D$；维数 $d$ 并没有一个明确的解释。向量更稠密：值为实数并且可以为负。

实践证明，在任何 NLP 任务中稠密向量都比稀疏向量的表现更好。尽管我们还没有完全明白其原因，但有以下直觉：首先，短而稠密的向量更适合作为机器学习的特征，例如模型处理 50000 维的稀疏向量需要大量的参数，而大部分参数都难以得到较好的学习；其次，由于使用了更少的参数，稠密向量的泛化效果更好，有利于防止过拟合；最后，稠密向量识别同义词的效果优于稀疏向量。

本节我们介绍一种计算嵌入的方法：**SGNS（skip-gram with negative sampling）**算法。它是 word2vec 工具包使用的两种算法之一，因此有时也被称为 word2vec 方法（Mikolov 等人 2013, Mikolov 等人 2013a）。word2vec 方法快速、训练高效，并且容易获取代码和预训练模型。word2vec 嵌入是**静态嵌入（static embedding）**，意味着该方法为词汇表中的每个词学习到一个固定的嵌入。在后续章节我们会介绍学习动态**上下文嵌入（contextual embedding）**的方法，例如流行的 BERT 表示族，其中每个词的向量表示在不同的上下文中都不同。

word2vec 的思路是不去计数每一个词 $w$ 出现在目标词附近的频率，而是训练一个分类器完成二元预测任务：$w$ 是否可能出现在目标词附近？我们不是真正要去完成这个预测任务，而是要将分类器学习到的权重作为词嵌入。任何文本都可以直接作为分类器的训练数据，而实际出现在目标词附近的词就会有较高的权重。

这里具有革命性的思路是我们可以直接使用流动文本作为这样一个分类器的隐含的有监督训练数据，一个词 $c$ 出现在目标词附近这件事本身，天然就成为了问题“词 $c$ 是否可能出现在目标词附近？”的“正确答案”。这种方法，通常称为**自监督（self-supervision）**，避免了需要任何类型的人工标注的监督标记。这一想法最初在神经语言建模任务中提出，Bengio 等人（2003）和 Collobert 等人（2011）指出一个神经语言模型可以直接使用流动文本的下一个词作为它的监督标记，并且可以被用于为每个词学习一个嵌入表示，作为这一预测任务的一部分。

word2vec 模型相比神经网络模型简单很多，这体现在两方面：第一，word2vec 简化了任务，将词预测任务简化为二元分类任务；第二，word2vec 简化了架构，将多层神经网络模型简化为逻辑回归模型。skip-gram 的思路是：

1. 将目标词和其邻近词组合作为若干正例
2. 从词汇表随机抽取其他词与目标词组合作为若干反例
3. 使用逻辑回归训练区分上述两类的分类器
4. 使用回归权重作为嵌入

### 分类器

让我们先从分类任务开始。考虑下面的句子，其中目标词为 `apricot`，窗口为 ±2 大小：

<pre>
    ... lemon, a [tablespoon of apricot jam, a] pinch ...
                  c1         c2 w       c3   c4
</pre>

我们的目标是训练一个分类器，使其接受目标词 $w$ 和候选上下文词 $c$ 组成的一个元组 $(w,c)$，返回 $c$ 是 $w$ 的上下文词的概率值：

$$
P(+|w,c)
$$

如何计算这个概率值？skip-gram 模型假定嵌入相似度有关：<u>如果一个词的嵌入与目标词的嵌入相似，则该词可能出现在目标词附近</u>。为了计算这些稠密嵌入之间的相似度，我们又假定如果两个向量的点积大，则它们的相似度高：

$$
{\rm similarity}(w,c)=\pwb w\cdot\pwb c
$$

!!! note "说明"
    注意“相似”这一概念：tf-idf 模型的相似指二阶共现（即语义相似），而 skip-gram 模型的相似指一阶共现（即彼此相邻），衡量相似的方法都是计算点积（余弦相似度可以视为归一化的点积）。换言之，tf-idf 模型的向量空间中同义词彼此靠近，而 skip-gram 模型的向量空间中相邻出现的词彼此靠近。

点积取值在正负无穷之间，因此采用逻辑函数将其映射到 (0,1) 区间：

$$
\displaylines{
P(+|w,c)=\frac{1}{1+e^{-w\cdot c}}\\
P(-|w,c)=\frac{1}{1+e^{w\cdot c}}
}
$$

上式给出了一个词的概率，但窗口中存在多个上下文词。skip-gram 模型简单地假定所有的上下文词都是独立的，因此概率可以相乘：

$$
\displaylines{
P(+|w,c_{1:k})=\prod_{i=1}^k\frac{1}{1+e^{-w\cdot c_i}}\\
\log P(+|w,c_{1:k})=-\sum_{i=1}^k\log(1+e^{-w\cdot c_i})
}
$$

总之，skip-gram 模型训练了这样一个分类器，它接受目标词 $w$ 和包含 $L$ 个词 $c_{1:L}$ 的窗口，返回反映目标词和窗口的相似程度的概率值。概率值基于目标词和每一个上下文词的嵌入的点积，因此我们需要对词汇表中的所有词进行嵌入。

下图展示了我们需要的参数。skip-gram 模型实际上为每个词保存了两个嵌入，一个用于词作为目标词，另一个用于词作为上下文词。现在让我们看向如何学习这些嵌入。

![](https://s2.loli.net/2022/12/08/wPCc2mM4pG5Snz7.png)

## 学习 skip-gram 嵌入

skip-gram 嵌入的学习算法接受一个文本语料库和一个选定的词汇表规模 N 作为输入。它首先为词汇表中的每个词赋予一个随机嵌入向量，然后迭代地改变每个词的嵌入，使其更接近于词的上下文词的嵌入，而远离非上下文词的嵌入。还是考虑这个句子：

<pre>
    ... lemon, a [tablespoon of apricot jam, a] pinch ...
                  c1         c2 w       c3   c4
</pre>

从中可以得到 4 个正例，如下图左侧所示：

![](https://i.loli.net/2021/01/07/oAKJN2GHbPcyCOD.png)

为了训练分类器我们还需要反例，实际上使用负采样的 skip-gram（skip-gram with negative sampling，SGNS）模型使用 $k$ 倍于正例数量的反例（$k$ 是超参数，这里取 2），如上图右侧所示。反例中的每个候选词（也称为噪声词）从词汇表随机选择抽取（不能是目标词自身），服从加权的 unigram 频率分布：

$$
p_\alpha(w)=\frac{{\rm count}(w)^\alpha}{\sum_{w'}{\rm count}(w')^\alpha}
$$

实践中 $\alpha$ 通常取 0.75，可以让各概率值向中间靠拢，提高罕见词的概率。例如假设长度为 100 的文本中出现了 99 次 a，1 次 b，那么：

$$
\displaylines{
p_\alpha(a)=\frac{.99^.75}{.99^.75+.01^.75}=.97\\
p_\alpha(b)=\frac{.01^.75}{.99^.75+.01^.75}=.03
}
$$

现在我们有了初始的嵌入和正反例，接下来就是学习算法对嵌入进行调整，目标为：

* 最大化正例的元组的相似度
* 最小化反例的元组的相似度

因此优化问题的目标函数为：

$$
\displaylines{
\max\mathcal{L}=\sum_{(w,c)\in +}\log P(+|w,c)+\sum_{(w,c)\in -}\log P(-|w,c)\\
=-\sum_{(w,c)\in +}\log(1+e^{-w\cdot c})-\sum_{(w,c)\in -}\log(1+e^{w\cdot c})\\
即\min \mathcal{L}=\sum_{(w,c)\in +}\log(1+e^{-w\cdot c})+\sum_{(w,c)\in -}\log(1+e^{w\cdot c})
}
$$

可以使用随机梯度下降法进行优化。下图展示了训练的一次迭代：

![](https://s2.loli.net/2022/12/08/kGoF7nxpctyIJWE.png)

回想 skip-gram 模型为每个词学习了两个嵌入：**目标嵌入（target embedding）**和**上下文嵌入（context embedding）**，分别用于词作为目标词和词作为上下文词。所有嵌入存储在两个矩阵：**目标矩阵（target matrix）**$W$ 和**上下文矩阵（context matrix）**$C$ 中，矩阵 $W,C$ 即为模型参数。

训练完毕之后，我们可以用 $w_i+c_i$ 作为词 $i$ 的嵌入，也可以用 $w_i$，还可以选择 $w_i\oplus c_i$（$\oplus$ 表示向量拼接）。

与简单的基于计数的方法（例如 tf-idf）相同，上下文窗口长度 $L$ 也是一个影响嵌入表现的重要超参数，通常根据验证集调参。$L$ 越大，则训练集（正反例）的规模越大。

### 其他类型的静态嵌入

静态嵌入有很多种类型。word2vec 的一个扩展，**fasttext**（Bojanowski 等人，2017），解决了 word2vec 的一个问题，即没有好的方法来处理未知词。一个关联的问题是词稀疏性，例如在一些形态丰富的语言中，每个名词或动词的一些形式出现得很少。Fasttext 通过使用子词模型来处理这些问题，将每一个词表示为它自身加上构成它的 n 元序列，这里每一个词要加上特殊的边界符号 `<>`。例如，当 $n=3$ 时词 `where` 被表示为序列 `<where>` 加上字符 n 元序列：`<wh, whe, her, ere, re>`。然后为每个构成的 n 元序列都学习一个嵌入，于是词 `where` 被表示为所有构成它的 n 元序列的嵌入之和。未知词因此可以被表示。一个 fasttext d饿开源库，包含 157 种语言的预训练嵌入，可以在 https://fasttext.cc 获取。

另一种非常广泛使用的静态嵌入模型是 GloVe（Pennington 等人，2014），缩写自 Global Vectors，因为该模型基于抓取全局语料库统计数据。GloVe 基于词-词共现矩阵中概率的比值，结合了 PPMI 等基于计数的模型的思路，也抓取了 word2vec 等方法使用的线性结构。

后来发现像 word2vec 这样的稠密嵌入与像 PPMI 这样的稀疏嵌入实际上有一个优雅的数学关系，其中 word2vec 可以被视为隐式地优化 PPMI 矩阵的一个变形版本（Levy 和 Goldberg，2014c）。

## 嵌入可视化

嵌入可视化是一个重要目标，意在帮助我们理解、应用和改进嵌入方法。但我们应该如何可视化一个高维向量呢？

可视化一个词 $w$ 的含义的最简单方法是列出与该词最相似的几个词（通过计算所有词的向量 $w$ 的向量的余弦相似度并排序），例如 GloVe 嵌入中最接近 `frog` 的7个词分别为 `frogs`、`toad`、`litoria`、`leptodactylidae`、`rana`、`lizard` 和`eleutherodactylus`（Pennington 等人，2014）。

另一种可视化方法是使用聚类算法构造嵌入空间中相似的词的层次表示。下图使用一些名词的嵌入向量的层次聚类作为一种可视化方法（Rohde 等人，2006）。

<img src="https://s2.loli.net/2022/12/08/fOkm8YTXeqzP1D2.png" style="zoom:50%;" />

但最常用的可视化方法依然是投影，例如课本上将高维空间投影到二维，tensorboard 可以将高维空间投影到三维。如图 6.16 使用了一种名为 t-SNE 的投影方法（van der Maaten and Hinton，2008）。

## 嵌入的语义性质

本节我们简要总结一些已经研究过的嵌入的语义性质。

**不同类型的相似度或关联**

向量语义模型的一个重要超参数就是窗口长度，tf-idf 和 word2vec 模型都有这个参数。它在目标词的每一侧通常为 1～10 个词（上下文总共为 2～20 个词）。

对于这个参数的选择取决于表示的目标：短窗口计算得到的相似的词与目标词在语义上相似，且词性相同；长窗口计算得到的高余弦相似度的词则只与目标词在话题上相关，而并不一定相似。

例如 Levy 和 Goldberg（2014a）展示了使用 skip-gram 模型和 ±2 窗口时，与 Hogwarts（出自哈利波特系列）最相似的词是其他小说中的学校：Sunnydale（出自吸血鬼猎人巴菲）和 Evernight（出自某个吸血鬼系列）；使用 ±5 窗口时，相似度最高的词则是哈利波特相关话题下的词：Dumbledore、Malfoy 和 half-blood。

区分两种类型的相似或关联是有必要的（Schütze and Pedersen, 1993）：**一阶共现（first-order co-occurrence）**，也称为**组合关联（syntagmatic association）**指两个词通常彼此相邻，例如 `write` 与 `book` 或 `poem` 一阶共现；**二阶共现（second-order co-occurrence）**，也称为**聚合关联（paradigmatic association）**指两个词有相似的上下文词，例如 `write` 与 `say` 或 `remark` 二阶共现。

**类比/相对相似性**

嵌入的另一个语义性质就是可以抓住词义关系。在一个重要的早期向量空间认知模型中，Rumelhart 和 Abrahamson（1973）就提出了**平行四边形模型（parallelogram model）**，用于解决形式为“a 相对于 b 就像 a* 相对于什么？”的简单类比问题。系统接受一个类似于 `apple:tree::grape:?` 的问题，在问号处填入词 `vine`。在平行四边形模型中，如下图所示，从词 `apple` 到 `tree` 的向量（$\overrightarrow{{\rm tree}}-\overrightarrow{{\rm apple}}$）被加到词 `grape` 的向量（$\overrightarrow{{\rm grape}}$）上，最接近结果那一点的词被返回。

![](https://s2.loli.net/2022/12/09/SoUGZOegD93puRL.png)

在稀疏嵌入的早期工作中，学者指出稀疏向量的词义模型可以解决类似的类比问题（Turney 和 Littman，2005），但平行四边形模型后来受到更多关注还是因为它在 word2vec 或 GloVe 向量上的成功（Mikolov 等人 2013c，Levy 和 Goldberg 2014b，Pennington 等人 2014）。例如，表达式 $\overrightarrow{{\rm king}}-\overrightarrow{{\rm man}}+\overrightarrow{{\rm woman}}$ 的结果非常接近向量 $\overrightarrow{{\rm queen}}$。类似地，$\overrightarrow{{\rm Paris}}-\overrightarrow{{\rm France}}+\overrightarrow{{\rm Italy}}$ 的结果非常接近向量 $\overrightarrow{{\rm Rome}}$。嵌入模型因此就像是在提取一些关系的表示，例如 `MALE-FEMALE`、`CAPITAL-COUNTRY`，甚至是 `COMPARATIVE-SUPERLATIVE`，如下图所示：

![](https://s2.loli.net/2022/12/09/joUDsAJrwmygfI2.png)

这里有一些注意事项。例如，在 word2vec 或 GloVe 嵌入空间中由平行四边形算法返回的最接近的值通常是三个输入词之一或者它们形态变体（即 `cherry:red::potato:?` 返回 `potato` 或 `potatoes` 而不是 `brown`），因此这些词必须被显式地去掉。此外，尽管嵌入空间对于涉及高频词、短距离、存在某种关系的任务表现得很好，但对于其他关系则表现不佳（Linzen 2016，Gladkova 等人 2016，Schluter 2018，Ethayarajh 等人 2019a），而 Peterson 等人（2020）认为平行四边形模型总体上过于简单，难以对人类形成此类类比的认知过程进行建模。

嵌入对于研究语义随时间的变化也十分有用，通过计算不同年代的文本得到的嵌入空间。例如下图展示了一些英文词在两个世纪内的词义变化：

![](https://i.loli.net/2021/01/07/ID98XkhcBioC54N.png)

## 偏差（政治正确环节）

嵌入会复现文本中潜藏的偏见和刻板印象。例如 Bolukbasi 等人（2016）发现在新闻文本上训练的 word2vec 嵌入中 `man:computer programmer::woman:?` 的结果最接近于 `homemaker`，类似地，`father:doctor::mother:?` 的结果最接近于 `nurse`。This could result in what Crawford (2017) and Blodgett et al. (2020) call an allocational harm, when a system allocates resources (jobs or credit) unfairly to different groups. For example algorithms that use embeddings as part of a search for hiring potential programmers or doctors might thus incorrectly downweight documents with women’s names.

It turns out that embeddings don’t just reflect the statistics of their input, but also amplify bias; gendered terms become more gendered in embedding space than they were in the input text statistics (Zhao et al. 2017, Ethayarajh et al. 2019b, Jia et al. 2020), and biases are more exaggerated than in actual labor employment statistics (Garg et al., 2018).

以前的**内隐关联测试（Implicit Association Test, IAT）**发现，美国人会将非裔美国人的名字关联到不愉快的词（相对于欧洲裔美国人），将男性关联到数学而女性关联到艺术，将老人的名字关联到不愉快的词（Greenwald 等人1998, Nosek 等人2002a, Nosek 等人2002b），而Caliskan 等人（2017）通过GloVe嵌入和计算余弦相似度也复现了上述发现。

Embeddings also encode the implicit associations that are a property of human reasoning. The Implicit Association Test (Greenwald et al., 1998) measures people’s associations between concepts (like ‘flowers’ or ‘insects’) and attributes (like ‘pleasantness’ and ‘unpleasantness’) by measuring differences in the latency with which they label words in the various categories.7 Using such methods, people in the United States have been shown to associate African-American names with unpleasant words (more than European-American names), male names more with mathematics and female names with the arts, and old people’s names with unpleasant words (Greenwald et al. 1998, Nosek et al. 2002a, Nosek et al. 2002b). Caliskan et al. (2017) replicated all these findings of implicit associations using GloVe vectors and cosine similarity instead of human latencies. For example African-American names like ‘Leroy’ and ‘Shaniqua’ had a higher GloVe cosine with unpleasant words while European-American names (‘Brad’, ‘Greg’, ‘Courtney’) had a higher cosine with pleasant words. These problems with embeddings are an example of a representational harm (Crawford 2017, Blodgett et al. 2020), which is a harm caused by a system demeaning or even ignoring some social groups. Any embedding-aware algorithm that made use of word sentiment could thus exacerbate bias against African Americans.

最近的研究致力于移除这些偏见……但这些**纠偏（debiasing）**方法只能降低偏见，而不能将其完全去除（Gonen and Goldberg， 2019）。

历史文本的嵌入也会包含过去的偏见和刻板印象。……

## 评估向量模型

对于向量模型，最重要的评估指标自然是外部评估：将它们作为 NLP 任务的特征并观察是否相对于其他向量模型有更好的表现。

尽管如此也有一些内部评估方法，最常用的指标是测试向量模型的相似度表现，即对于一些给定的词对，计算模型的词相似度和人工打分的相关系数。**WordSim-353**（Finkelstein 等人，2002）就是一个常用的集合，包含了 353 个打分从 0 到 10 的名词对；**SimLex-999**（Hill 等人，2015）则是一个更复杂的数据集，包括具体和抽象的形容词、名词和动词对；**TOEFL 数据集**包含 80 道选择同义词的题目，例如 levied 的含义最接近于：imposed，believed，requested，correlated（Landauer and Dumais，1997）。这些数据集给出的词均没有上下文。

更实际一点的内部相似度任务包含上下文。**Stanford Contextual Word Similarity（SCWS）**数据集（Huang 等人，2012）给出了人对于 2003 对词在句子上下文中的评估；**Word-in-Context（WiC）**数据集（Pilehvar and Camacho-Collados，2019）给出了目标词和两个包含该词的句子，需要判断词在这两个上下文中的词义是相同的还是不同的。语义文本相似度任务（Agirre 等人 2012，Agirre 等人 2015）则评估句子级别的相似度算法的表现，使用人工打分相似度的句子对的集合。

另一类用于评估的任务称为类比任务。一些用于该任务的问题集合已经被创建（Mikolov 等人 2013a，Mikolov 等人 2013c，Gladkova 等人 2016），涵盖了形态学（`city:cities::child:children`）、词汇关系（`leg:table::spout:teapot`）和百科全书关系（`Beijing:China::Dublin:Ireland`）。

所有嵌入算法都受到内在变异性的负面影响。例如由于初始化和随机负采样中的随机性，word2vec 这样的算法对于同一个数据集也可能会产生不同的结果，并且集合中的单个文本可能严重影响嵌入的结果（Tiam 等人 2016，Hellrich 和 Hahn 2016,Antoniak 和 Mimno 2018）。因此当嵌入被用于研究特定语料库中的词关联时，最佳实践是使用对于文本的 bootstrap 采样训练多个嵌入，然后对结果求平均（Antoniak 和 Mimno，2018）。
