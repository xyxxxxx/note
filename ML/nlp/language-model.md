[toc]

接下来我们将引入一个模型，它为下一个词的每一种可能计算一个概率，也为整个句子计算一个概率。例如模型计算下列文本的概率

​        all of a sudden I notice three guys standing on the sidewalk

将远大于下列文本的概率

​        on guys all I of notice sidewalk three a sudden standing the

当需要识别有噪音或歧义的输入的具体内容时，例如在语音识别任务中，概率就十分重要，模型会识别到你刚才说了I will be back soonish而不是I will be bassoon dish，因为前者的概率远大于后者。或者在拼写检查或语法检查任务中，模型会将Their are two midterms中的Their纠正为There，Everything has improve中的improve纠正为improved。

概率在机器翻译任务中也十分重要，考虑将以下句子翻译成英文

​        他  向 记者          介绍了         主要   内容


​        He to reporters introduced main content

我们可能先生成若干个粗翻译

​        he introduced reporters to the main contents of the statement
​        he briefed to reporters the main contents of the statement
​        **he briefed reporters on the main contents of the statement**

模型会认识到第3种表达最为合适，因为其概率最大。

上述这种对文本计算概率的模型称为**语言模型(language model, LM)**。这里将介绍最简单的语言模型——n元语法。





# n元语法

首先从计算条件概率$$P(w|h)$$开始，假设$$h$$为'its water is so transparent that'，$$w$$为'the'，我们计算
$$
P({\rm the}|{\rm its\ water\ is\ so\ transparent\ that})
$$
一种方法是计算相对频率：在一个非常大的语料库中计数'its water is so transparent that'和'its water is so transparent that the'出现的次数，那么
$$
P({\rm the}|{\rm its\ water\ is\ so\ transparent\ that})=\frac{C(\rm its\ water\ is\ so\ transparent\ that\ the)}{C({\rm its\ water\ is\ so\ transparent\ that})}
$$
尽管这种方法在一些情形下表现不错，但在大多数情形下语料库的规模都不够大：我们总是可以创造新的文本，而语料库永远不可能包含所有可能的文本。

类似地，计算绝对概率$$P(t)$$，假设$$t$$为'its water is so transparent'，一种计算方法是计数所有5个词的序列以及其中多少个是$$t$$。但是这样的计算量非常之大！

因此我们改进上述方法——使用条件概率的链式法则
$$
P(w_1^n)=P(w_1)P(w_2|w_1)\cdots P(w_n|w_1^{n-1})=\prod_{k=1}^nP(w_k|w_1^{k-1})
$$
然后对于每一个条件概率，将历史序列裁剪为给定长度的序列作为近似
$$
P(w_k|w_1^{k-1})\approx P(w_k|w_{k-n+1}^{k-1})
$$
这就是**n元语法(n-gram grammar)**模型。例如对于2元语法模型
$$
P({\rm the}|{\rm its\ water\ is\ so\ transparent\ that})=P({\rm the}|{\rm that})
$$
这种词的概率仅依赖前几个词的假定称为**Markov假定**。

> n-gram也指文本中连续的n个词

接下来的问题是如何估算n-gram概率？依然是计算相对频率
$$
P(y|x)=\frac{C(xy)}{C(x)}
$$

> 这种估计实际上是一种最大似然估计，也就是说这样得到的一种参数使得训练集出现的概率最高。



再来看一个更大的数据集now-defunct Berkeley Restaurant Project——上世纪的一个基于伯克利餐厅信息数据库的问答系统，下列是一些用户询问的标准化后的文本

​        can you tell me about any good cantonese restaurants close by
​        mid priced thai food is what i’m looking for
​        tell me about chez panisse
​        can you give me a listing of the kinds of food that are available
​        i’m looking for a good place to eat breakfast
​        when is caffe venezia open during the day

下图展示了bigram的频率和相对概率

![](https://i.loli.net/2020/12/22/WNXBeoKx5pVy9l4.png)

![](https://i.loli.net/2020/12/22/JjvKbws5F7VLemo.png)

这里再补上一些有用的概率
$$
P({\rm i}|\lang s\rang) = 0.25\\
P({\rm food}|{\rm english}) = 0.5\\
P({\rm english}|{\rm want}) = 0.0011\\
P(\lang/s\rang|{\rm food}) = 0.68
$$
现在我们可以计算诸如I want English food或I want Chinese food这样的文本的概率了，例如
$$
P({\rm \lang s\rang\ I\ want\ English\ food\ \lang /s\rang})=.25\times .33\times .0011\times .5\times .68=.000031\\
P({\rm \lang s\rang\ I\ want\ Chinese\ food\ \lang /s\rang})=.25\times .33\times .0065\times .52\times .68=.000190
$$
我们看到bigram的统计信息中包含了文法规则（例如want之后接to或名词），表达习惯（例如句子高概率以i开始），甚至语言之外的信息（例如want chinese是want English的约6倍）。

> 上面使用bigram的原因是便于教学展示，实际应用中一般使用trigram, 4-gram或5-gram。
>
> 实际应用中一般使用对数概率，因为概率本身的连乘很可能在数值计算中造成下溢。这样计算概率时就有
> $$
> p_1\times p_2\times p_3\times p_4=\exp(\log p_1+\log p_2+\log p_3+\log p_4)
> $$
> 





# 评价语言模型

评价语言模型的最佳方法是应用它并衡量其表现，称为**外部评价(extrinsic evaluation)**。然而运行大型NLP系统的代价十分高昂，因此也进行**内部评价(intrinsic evaluation)**，即用测试集输入模型并衡量指标。



## 困惑度

尽管我们可以选择测试集在模型上的概率作为指标，但实践中通常使用另一指标——**困惑度(perplexity, pp)**。对于一个测试集$$W=w_1w_2\cdots w_N$$
$$
PP(W)=P(w_1w_2\cdots w_N)^{-1/N}=\sqrt[N]{\frac{1}{P(w_1w_2\cdots w_N)}}
$$
使用链式法则，对于bigram模型，有
$$
PP(W)=\sqrt[N]{\prod_{i=1}^N\frac{1}{P(w_i|w_{i-1})}}
$$
注意到现在由于条件概率位于分母位置，那么序列的概率越高，则困惑度越低。在后面我们将看到困惑度与信息论中的熵的概念关系紧密。



最后比较对于不同的n值，n-gram模型的困惑度。我们在来自华尔街日报的3800万词的训练集上分别训练unigram, bigram和trigram模型，然后分别计算这三个模型在一个150万词的测试集上的困惑度，结果为

![](https://i.loli.net/2020/12/23/odYq2t6FO54cn3u.png)

可以看到，n-gram的n越大，条件概率的值总体上也越大（例如$$P({\rm to}|{\rm I\ want})>P({\rm to})$$）；或者更宽泛地说，n越大则模型提供的信息越多，困惑度相应地越小。





# 泛化

对于给定的训练语料库，有以下两种规律：

+ 概率通常编码了语料库的一些特征
+ n越大，n-gram模型对训练语料库的建模效果越好

为了展示高阶n-gram模型的能力，这里展示了在莎士比亚作品集上训练的unigram, bigram, trigram和4-gram随机生成的句子

![](https://i.loli.net/2020/12/23/HXqxodWuTL3698D.png)

可以看到模型的n越大，则生成的句子越连贯。unigram的句子的词与词之间没有关联，bigram的句子的词与词只有相邻的关联，trigram和4-gram的句子乍一看就很像莎士比亚的风格了。句子It cannot be but so.更是直接来自King John原文，这是因为莎士比亚作品集语料库并没有很大（N=884647, V=29066），因此n-gram的概率矩阵将变得非常稀疏。对于n-gram模型，一旦之前的词是It cannot be but，那么下一个词只可能选择(that, I, he, thou, so)中的一个；实际上，在很多情况下下一个词根本没有选择的余地。

> 没有选择的余地意味着模型生成的文本会照抄训练集的原文，这种现象可以理解为过拟合。

再来看在另一个训练集——华尔街日报语料库上训练的unigram, bigram, trigram随机生成的句子

![](https://i.loli.net/2020/12/23/3Cw4n7Siv2dLltj.png)

将这些句子与图3.3中的句子比较，尽管它们都是类似英文的句子，但它们的体裁/风格完全不同。因此在实际应用中，我们需要让训练集与测试集的

+ 体裁具有相似性，例如建立法律文件的机器翻译模型，则训练集必须使用法律文件的语料库。
+ 方言具有相似性，例如African American Vernacular English的推文会使用词finna表示即将，用den替代then等，这在其它方言中都是不存在的
+ 风格具有相似性，例如郭敬明的小说就具有特有的凡尔赛风格。
+ ……

还是稀疏性的问题，任何语料库即使再大也依然是有限规模，在其之上训练的模型也只能（在所有可能的表达中）统计到相当有限的表达。例如考虑WSJ Treebank3语料库中，denied the的下一个词，统计结果为

​        denied the allegations:   5
​        denied the speculation:  2
​        denied the rumors:          1
​        denied the report:            1

那么trigram模型随机生成的denied the xxx必定是上述之一；或者假设测试集中有下列短语

​        denied the offer
​        denied the loan

那么trigram模型计算上述文本的概率为0，进而无法计算困惑度。事实上，denied the的下一个词可以是很多名词。这显示了n-gram模型完全缺乏语言的创造力。



## 未知词

上面讨论了条件概率为0时将会出现的问题。但这里我们退一步，考虑绝对概率为0，即测试集出现了训练集中没有出现的词汇的情形。

在有些任务中我们不必担心上述问题，例如在语音识别或机器翻译任务中，我们有一个固定的(fixed)语音词典或词典，并且能够保证测试集不会出现词典以外的词。但在另一些情况下我们不得不处理没有出现过的词，称为**未知词(unknown word)**或**OOV(out of vocabulary)**词，测试集中OOV词所占的比例称为OOV率。如果在词汇表中增加一个`<UNK>`标记，可以匹配所有的OOV词，那么称词汇表是一个开放的词汇表(open vocabulary)。

处理未知词有两种方法，其一为：

1. 使用固定的词汇表
2. 在文本处理过程中将OOV词替换为`<UNK>`标记
3. 将`<UNK>`视作一个普通的词并计算概率

其二为：

1. 统计训练集中词频top k（或词频>k）的词，做成词汇表，将剩余词替换为`<UNK>`
2. 将`<UNK>`视作一个普通的词并计算概率

需要注意的是，使用的词汇表规模越小，被替换为`<UNK>`的词越多，即更多的词共享一个更大的概率，导致困惑度越小。因此在比较不同模型的困惑度时应当确保它们有相同的词汇表。



## 平滑smoothing

回到词的绝对概率不为0但条件概率为0的问题，即零频率问题。为了避免模型为没有见过的短语赋零概率，我们从其它事件分一点概率过来，这种操作称为**平滑(smoothing)**或**折减(discounting)**。这里将介绍一些平滑的方法。

### 拉普拉斯平滑/加1平滑

最简单的方法是在归一化之前将频率矩阵的所有元素+1，这种算法称为**拉普拉斯平滑(Laplace smoothing)**。在现在的n-gram模型中拉普拉斯平滑的效果已经不是很好，但它作为一种基础方法依然能帮助理解概念和作为基线，并且在文本分类任务中仍被使用。

还是以之前的Berkeley Restaurant Project bigram为例，与图3.1, 3.2对比，以下是拉普拉斯平滑后的结果

![](https://i.loli.net/2020/12/23/VHahDOWkqZo5umF.png)

![](https://i.loli.net/2020/12/23/ZiE8FGlSkXWtv9e.png)

比较平滑前后的结果，我们发现条件概率发生了非常大的改变：$$P({\rm to}|{\rm want})$$从.66下降到.26，$$P({\rm food}|{\rm chinese})$$更是从.52下降到.052。如此巨大的变化的原因是，V大于，甚至远大于这一行的N。



### 加k平滑

由于加1平滑对概率分布的影响太大，需要将1修改为一个较小的数$$k$$，如0.5, 0.1, 0.01等。这里$$k$$是一个超参数。

加k平滑方法对于一些任务有用，但在语言模型上表现不好(Gale and Church, 1994)。



### 后退和插值

另一种解决零频率问题的方法是使用更短的上文以获得信息。当高阶n-gram出现零频率时，**后退(backoff)**法依次使用更低阶的n-gram，直到得到非零条件概率；而**插值(interpolation)**法则是使用1-n阶的各阶n-gram，然后加权平均取最终条件概率。



简单线性插值法为
$$
\hat{P}(w_n|w_{n-2}w_{n-1})=\lambda_1 P(w_n|w_{n-2}w_{n-1})+\lambda_2 P(w_n|w_{n-1})+\lambda_3P(w_n),\ \lambda_1+\lambda_2+\lambda_3=1
$$
在更复杂一点的条件插值法中，参数$$\lambda$$与上文有关，
$$
\hat{P}(w_n|w_{n-2}w_{n-1})=\lambda_1(w_{n-2}^{n-1}) P(w_n|w_{n-2}w_{n-1})+\lambda_2(w_{n-2}^{n-1}) P(w_n|w_{n-1})+\lambda_3(w_{n-2}^{n-1})P(w_n)
$$
上述两种插值法的参数$$\lambda$$均为超参数，从**held-out**语料库学习得到。held-out语料库是不同于训练语料库的，专门用于学习超参数的训练集。选择的$$\lambda$$超参应使得held-out语料库具有最大的似然。

> 归一化处理？



在后退法中，如果n-gram模型有零频率，那么后退一步到(n-1)-gram模型，直到得到非零条件概率。但此时n-gram模型的概率之和不为1，并且不同的得到非零条件概率的阶数也应该对应不同的权重，因此需要具体的方法。**Katz后退**法如此计算
$$
P_{bo}(w_n|w_{n-N+1}^{n-1})=\begin{cases}P^*(w_n|w_{n-N+1}^{n-1}),\quad C(w_{n-N+1}^{n}>0)\\
\alpha(w_{n-N+1}^{n-1})P_{bo}(w_n|w_{n-N+2}^{n-1}),\quad otherwise
\end{cases}
$$
其中$$P^*$$是折减的概率，折减方法依然是归一化。

> 归一化处理？

Katz后退法经常和一种叫做Good-Turing的平滑方法结合使用，称为Good-Turing后退法。



## Kneser-Ney平滑

一种最常用、效果最好的n-gram平滑方法就是插值Knerser-Ney算法(Kneser and Ney 1995)。该方法源于一种称为**绝对折减(absolute discounting)**的方法。

我们先从Church and Gale (1991)这里借用一种思路。他们在AP newswire的一个2200万词的语料库（训练集）上建立了一个bigram模型，然后检查频率为某个值$$c$$的所有bigram在另一个2200万词（held-out集）上的平均频率是多少，下面是$$c$$取0到9的结果

![](https://i.loli.net/2020/12/24/LstYo2a9WwRGVUX.png)

我们发现当$$c$$大于等于2时，held-out集上的频率近似为$$c-0.75$$。绝对折减方法就是基于这个发现，将每个频率减去固定值$$d$$。插值绝对折减方法应用在bigram的计算公式为
$$
P_{AD}(w_i|w_{i-1})=\frac{C(w_{i-1}w_i)-d}{\sum C(w_{i-1}*)}+\lambda(w_{i-1})P(w_i)
$$
其中第一项是折减的bigram，第二项是unigram的插值项。我们可以令$$d=0.75$$，或者当bigram频率为1时单独令$$d=0.5$$。

> 归一化处理？



Kneser-Ney折减(Kneser and Ney, 1995)改进了绝对折减，通过改进对低阶n-gram分布的处理。考虑预测以下句子的下一个词，使用bigram和unigram的插值方法：

​        I can’t see without my reading _______________

词glasses的概率应该比其它词，例如Kong，大得多，但实际上由于Hong Kong是一个高频词，unigram模型中Kong的概率会比glasses高得多。我们这里想要的效果是Kong基本上紧跟在Hong之后，而glasses则有更加广泛的分布。

换言之，我们希望unigram的模型能够预测词$$w$$可以作为新的接续的概率，而不仅是出现概率。因此Kneser-Ney的想法是基于$$w$$接续在多少个不同的词之后设定接续概率$$P_{CONTINUATION}$$，这里假定如果在过去的统计中$$w$$接续在多个词之后，那么它就更有可能接续在另外的词之后
$$
P_{CONTINUATION}(w)\propto |\{v:C(vw)>0 \}|\\
\Rightarrow P_{CONTINUATION}(w)=\frac{|\{v:C(vw)>0 \}|}{|\{(*,*):C(**)>0 \}|}\\
$$
这样，Kong就会因为仅接在Hong之后而使得接续概率非常小。

最终，插值Kneser-Ney平滑法应用在bigram的公式即为
$$
P_{KN}(w_i|w_{i-1})=\frac{\max(C(w_{i-1}w_i)-d,0)}{\sum C(w_{i-1}*)}+\lambda(w_{i-1})P_{CONTINUATION}(w_i)
$$
其中归一化参数$$\lambda(w_{i-1})$$为
$$
\lambda(w_{i-1})=\frac{d}{\sum C(w_{i-1}*)}|\{w:C(w_{i-1}w)>0\}|
$$
该参数刻画了$$w_{i-1}$$的下一个词的多样性。设想，如果词$$w_{i-1}$$出现了100次，但接续在之后的词都是同一个，那么参数等于$$d/100$$。

> 这里使用了2种方法判断词$$w$$的后/前一个词的多样性：
>
> 1. 统计$$w$$出现的频率以及$$w$$与后/前一个词的组合种类数，如果为1则多样性最低，如果等于频率则多样性最高
> 2. 统计$$w$$与后/前一个词的组合种类数以及所有bigram的种类数，如果比值低则多样性低，比值高则多样性高

应用在n-gram的公式的递归式为
$$
P_{KN}(w_i|w_{i-n+1}^{i-1})=\frac{\max(C(w_{i-n+1}^{i-1}w_i)-d,0)}{\sum C(w_{i-n+1}^{i-1}*)}+\lambda(w_{i-n+1}^{i-1})P_{KN}(w_i|w_{i-n+2}^{i-1})
$$
?





Kneser-Ney平滑效果最好的版本是**修正的Kneser-Ney平滑(modified Kneser-Ney smoothing)**，来自于Chen and Goodman (1998)。该方法为unigram, bigram和3+-gram分别使用了不同的折减$$d$$值。





# 傻瓜后退

通过使用互联网上的文本，我们可以建立非常大的语言模型。2006年Google发布了一个非常大的n-gram统计数据，将从互联网爬取的总长1024908267229词的文本当中出现至少40次的长度为5的序列





# 困惑度与熵的关系

> 参考信息论

