[toc]

许多语言处理任务都包含分类任务。本章我们将介绍朴素贝叶斯算法并将其应用到**文本分类(text categorization)**任务。

我们聚焦于一种常见的文本分类方法：**情感分析(sentiment analysis)**。互联网上对于电影、商品、餐厅等评价都包含了评价者的积极或消极态度，报纸上的社论等也包含了作者对于政治事件或人物的褒贬态度。情感分析任务就是要提取这一种积极或消极的方向。

最简单的情感分析任务是二分类，并且文本中的词就已经给出了很好的提示，例如great, richly, awesome和pathetic，或者awful和ridiculously都是非常有信息量的词。考虑下列例子：

​        \+ ...zany characters and richly applied satire, and some great plot twists
​        \- It was pathetic. The worst part about it was the boxing scenes...
​        \+ ...awesome caramel sauce and sweet toasty almonds. I love this place!
​        \- ...awful pizza and ridiculously overpriced...

**垃圾邮件识别(spam detection)**就是其在商业上的重要应用，其中二分类模型为一封邮件贴上spam或not-spam的标签。许多词汇的特征和其它一些特征被用到，例如当邮件中包含“WITHOUT ANY COST”或“Dear Winner”等短语时，我们就有理由怀疑它。垃圾邮件识别任务是朴素贝叶斯最早在文本分类上的应用(Sahami et al., 1998)。

另一个经典任务是为一段文本贴上图书馆分类标签，这是信息检索的一个重要部分。实际上朴素贝叶斯算法在1961年被发明时就是应用在学科分类任务上。

> 对于不同的语言我们应用不同的处理方式，因此大多数语言处理流水线的第一步是贴上**语言id(language id)**。

分类在句子或者词级别的任务上也十分重要，例如我们已经见过的点号消歧义(period disambiguation)就是一种分类任务。甚至语言模型本身就可以视作一种分类：词汇表中的每个词都视作一类，因此预测下一个词相当于将上文划分到下一个词对应的类中。

总的来说，分类的目标就是取单次观测，提取出有用的特征，然后将特征分类到若干个离散的类中。

人为设定规则在NLP的分类任务中十分重要，许多领域中基于人工规则的分类器都是最先进NLP系统的重要部分。尽管如此，人工规则也十分脆弱，其普适性不够强，划分效果也不够好，因此我们在大多数情况下用有监督的机器学习方法替代。习惯上我们用 $(d_1,c_1),\cdots,(d_N,c_N)$ 表示训练集，其中 $d$ 为文本， $c$ 为正确的类别。

本章介绍的朴素贝叶斯算法和之后介绍的逻辑回归算法分别代表了两种分类的思路：**Generative** classifiers like naive Bayes build a model of how a class could generate some input data. Given an observation, they return the class most likely to have generated the observation. **Discriminative** classifiers like logistic regression instead learn what features from the input are most useful to discriminate between the different possible classes. While discriminative systems are often more accurate and hence more commonly used, generative classifiers still have a role.



# 朴素贝叶斯分类器

朴素贝叶斯分类器在特征间相互作用的方式上做了简化，其想法源于词袋模型（见下图），即将一段文本简单地视作词+频率。

![](https://i.loli.net/2020/12/28/asg6SXbzpKhrnyj.png)

朴素贝叶斯是一种概率分类器，即对于特定文本 $d$，从所有类别中返回具有最大后验概率的类别，即
$$
\hat{c}=\arg\max_{c\in C} P(c|d)
$$
Mosteller and Wallace (1964) 最早将贝叶斯推断应用于文本分类，其直观想法是将贝叶斯公式应用于计算上述条件概率
$$
\hat{c}=\arg\max_{c\in C} P(c|d)=\arg\max_{c\in C} \frac{P(d|c)P(c)}{P(d)}=\arg\max_{c\in C} P(d|c)P(c)
$$
上式去掉分母的原因是文本是给定的，因此 $P(d)$ 是定值（也可以理解为1）。于是后验概率被拆分为先验概率 $P(c)$ 和似然 $P(d|c)$ 的积。尽管如此，上式还是难以计算，因此朴素贝叶斯分类器做了两个简化假定：

1. 词的位置并不重要（如同词袋模型所展示的）

2. **朴素贝叶斯假定(naive Bayes assumption)**：各特征条件独立，即 $P(f_i|c)$ 独立，有
   $$
   P(d|c)=P(f_1,f_2,\cdots,f_n|c)=P(f_1|c)\cdot P(f_2|c)\cdots P(f_n|c)
   $$

朴素贝叶斯分类器的最终计算公式如下：
$$
c_{NB}=\arg\max_{c\in C} P(c)\prod_{f\in F} P(f|c)=\arg\max_{c\in C} P(c)\prod_{i} P(w_i|c)
$$
与语言模型的计算类似，朴素贝叶斯的计算也都使用对数概率，以防止下溢，提高速度
$$
c_{NB}=\arg\max_{c\in C}(\log P(c)+\sum_{i}\log P(w_i|c))
$$
上式也可以看作多分类的（对于对数条件概率的）线性分类器。





# 训练朴素贝叶斯分类器

那么如何得到 $P(c)$ 和 $P(w_i|c)$？最直接的方法是最大似然估计法，即使用训练集上的频率作为概率
$$
\hat{P}(c)=\frac{N_c}{N_{all}}\\
\hat{P}(w_i|c)=\frac{{\rm count}(w_i,c)}{\sum_{w\in V}{\rm count}(w,c)}
$$

> 注意词汇表 $V$ 是所有类别下所有文本的词汇表。

这种方法存在一个问题：设想我们现在想要estimate the likelihood of the word “fantastic” given class positive，但训练集中positive类别下的文本均不包含词fantastic，反而是negative类别下有一个文本包含fantastic（可能是讽刺？），此时计算有
$$
\hat{P}({\rm fantastic}|{\rm positive})=\frac{{\rm count}({\rm fantastic,positive})}{\sum_{w\in V}{\rm count}(w,{\rm positive})}=0
$$

> 注意如果negative类别下的文本也均不包含词fantastic，那么fantastic不会进入词汇表，即不考虑这个词

既然朴素贝叶斯对所有似然求积，那么如果测试集有一个文本包含词fantastic，那么该文本属于positive的概率直接为0，不管文本的其它部分如何。

最简单的解决方法依然是拉普拉斯平滑。尽管在语言模型中拉普拉斯平滑已经基本上被更精巧的平滑算法替代，但在朴素贝叶斯分类器中依然普遍使用
$$
\hat{P}(w_i|c)=\frac{{\rm count}(w_i,c)+1}{\sum_{w\in V}({\rm count}(w,c)+1)}=\frac{{\rm count}(w_i,c)+1}{\sum_{w\in V}{\rm count}(w,c)+|V|}
$$


如果测试集中出现了不在词汇表中的词，即**未知词(unknown word)**，处理方法是直接从文本中删除该词。

此外，一些模型也选择忽略一些高频词，称为**停用词(stop word)**，如a, the等。既可以选择词汇表中频率最高的10~100个词作为停用词，也可以使用预定义的停用词表。但实践证明，在大部分文本分类应用下，使用停用词都无法改善模型效果，因此一般都不使用。

下面是训练过程的伪代码

![](https://i.loli.net/2020/12/28/ZE4TNDYLtp6gMV1.png)



# 示例

下面是一个训练和测试朴素贝叶斯模型（使用拉普拉斯平滑）的示例，样本分为positive和negative两类，训练集和测试集都非常迷你：

![](https://i.loli.net/2020/12/28/Y4tIlCWkvZ2NBRq.png)

计算类别的先验
$$
P(-)=\frac{3}{5},\ P(+)=\frac{2}{5}\\
$$
测试文本中的with在训练集中未出现，直接删除；计算剩余词的似然
$$
P({\rm predictable}|-)=\frac{1+1}{14+20},\ P({\rm predictable}|+)=\frac{0+1}{9+20}\\
P({\rm no}|-)=\frac{1+1}{14+20},\ P({\rm no}|+)=\frac{0+1}{9+20}\\
P({\rm fun}|-)=\frac{0+1}{14+20},\ P({\rm fun}|+)=\frac{1+1}{9+20}\\
$$
其中14为所有negative文本的长度之和，9为所有positive文本的长度之和，20为词汇表规模。由此计算后验
$$
P(-)P(S|-)=\frac{3}{5}\cdot\frac{2\cdot2\cdot1}{34^3}=6.1\times 10^{-5}\\
P(+)P(S|+)=\frac{2}{5}\cdot\frac{1\cdot1\cdot2}{29^3}=3.2\times 10^{-5}
$$
因此预测测试文本为negative。





# 情感分析上的优化

尽管标准的朴素贝叶斯算法能够很好地进行语义分析，我们依然能做一些小的改变以提高效果。

首先，对于情感分析和其它一些文本分类任务，在一个文本中出现了哪些词比这些词的频率更加重要（也就是认为同一个词的反复使用不会产生影响）。因此我们可以将一个文本中出现的所有词的词频设为1，这种方法称为**二元多项式朴素贝叶斯(binary multinomial naive Bayes)**或**二元朴素贝叶斯(binary NB)**。实现方法可以是对所有文本预处理为词的集合。

第二个重要的改善是处理否定式。否定式可以让一个positive的句子直接转变为negative，也可以反过来。一个最简单的情感分析中常用的基线方法是：为否定式的token（n't, not, no, never）之后的所有词加上前缀NOT_。这样，NOT_like, NOT_recommend这样的词就能明显与like, recommend区分开。

最后，当我们的训练集规模太小，以至于模型无法使用词汇表的很多词预测积极或消极的情感时，我们可以借助外部的**情感词汇表(sentiment lexicon)**，其中各词汇被预先标注了积极或消极的情感。4个最流行的情感词汇表分别是General Inquirer (Stone et al., 1966), LIWC (Pennebaker et al., 2007), the opinion lexicon of Hu and Liu (2004a) and the MPQA Subjectivity Lexicon (Wilson et al., 2005)。

例如MPQA Subjectivity Lexicon包括6885个词，其中2718个积极的词，4912个消极的词，每个词都标注了情感的强弱，下面是一些例子

​        \+ admirable, beautiful, confident, dazzling, ecstatic, favor, glee, great
​        \-  awful, bad, bias, catastrophe, cheat, deny, envious, foul, harsh, hate

使用这种词汇表的通常方法是增加两个特征：文本中被词汇表标记为positive/negative的总词数。如果我们有非常多的训练数据，并且测试数据与训练数据一致，仅使用这两个特征的效果不如使用所有词（特征）；但如果训练数据较少并且测试数据与训练数据有一些差别，仅使用这两个特征的泛化效果更好。





# 朴素贝叶斯应用于其它文本分类任务

朴素贝叶斯当然不一定要把训练数据中出现过的所有词作为特征。例如在垃圾邮件识别任务中，常用的方法并非将每一个词作为独立的特征，而是预定义一些特定的词和短语作为特征，并且还包含一些不纯粹是语言的特征。例如开源的SpamAssassin工具预定义了类似于one hundred percent guaranteed这样的短语，大额数字（用正则表达式匹配）等作为特征，此外还包括一些其它特征：

+ 发送地址有垃圾邮件发送记录
+ HTML的文字比例太低
+ 邮件标题全是大写字母
+ ……



而对于语言分类（预测language id）这样的任务，最有效的朴素贝叶斯特征并非词，而是字节n-gram，例如字节trigram: 'nya', ' Vo'。一个广泛使用的朴素贝叶斯模型`langid.py`(Lui and Baldwin, 2012)首先提取文本的所有长度1-4的字符n-gram，然后使用特征选择方法筛选最终的7000个最有信息量的特征。





# *朴素贝叶斯与语言模型





# 评价：精确率，召回率和F-值

模型对于任意一个测试样本的二分类结果有4种情况，用如下的**混淆矩阵(confusion matrix)**来表示：

![](https://i.loli.net/2020/09/04/aj4dY5ZuKnpDk9R.png)

精确率定义为
$$
P=\frac{TP}{TP+FP}
$$
召回率定义为
$$
R=\frac{TP}{TP+FN}
$$
F值定义为
$$
F=\frac{(1+\beta^2)PR}{\beta^2P+ R}
$$
当 $\beta=1$ 时，F值称为F1值，是精确率和召回率的调和平均数
$$
F_1=\frac{2PR}{P+R}
$$

> 假定上述混淆矩阵中，阳性代表新型冠状病毒检测呈阳性，那么精确率可以适当牺牲而召回率一定要保证（因为隔离未患病的人代价较小而放走患病的人代价巨大）。
>
> 假定上述混淆矩阵中，阳性代表邮件为垃圾邮件，那么精确率一定要保证而召回率可以适当牺牲（因为将正常邮件归入垃圾邮件的代价较大而将垃圾邮件归入正常邮件的代价较小）；若阳性代表邮件为正常邮件，则结论相反，召回率一定要保证而精确率可以适当牺牲。
>
> 通过上述两个例子可以看到，精确率和召回率哪一个更重要取决于具体问题的具体形式。



多分类情况下的精确率和召回率定义如下图

![](https://i.loli.net/2020/12/28/o2BdK9bWU1J4nYf.png)

宏平均和微平均定义如下图

![](https://i.loli.net/2020/12/28/XjdxBMm7kg98Nvw.png)

可以看到，微平均受主要类别（spam）的影响最大，而宏平均则一视同仁。应根据具体情况选择指标。



