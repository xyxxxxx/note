# 语言模型

接下来我们将引入一个模型，它为下一个词的每一种可能计算一个概率，也为整个句子计算一个概率。例如模型计算下列文本的概率

<pre>    all of a sudden I notice three guys standing on the sidewalk</pre>

将远大于下列文本的概率

<pre>    on guys all I of notice sidewalk three a sudden standing the</pre>

当需要识别有噪音或歧义的输入的具体内容时，例如在语音识别（speech recognition）任务中，概率就十分重要。模型会识别到你刚才说了 `I will be back soonish` 而不是 `I will be bassoon dish`，因为前者的概率远大于后者。或者在拼写检查（spelling correction）或语法检查（grammatical error correction）任务中，模型会将 `Their are two midterms` 中的 `Their` 纠正为 `There`，`Everything has improve` 中的 `improve` 纠正为 `improved`。

概率在机器翻译（machine translation）任务中也十分重要，考虑将以下句子翻译成英文：

<pre>
    他 向 记者       介绍了      主要  内容
    He to reporters introduced main content
</pre>

作为流程的一部分，我们可能先生成一组粗翻译：

<pre>
    he introduced reporters to the main contents of the statement
    he briefed to reporters the main contents of the statement
    <b>he briefed reporters on the main contents of the statement</b>
</pre>

一个词序列的概率模型会认为 `briefed reporters on` 是一个比 `briefed to reporters` 或 `introduced reporters to` 更有可能的英文短语，使得我们能够正确地选择上面加粗的句子。

上面这种对文本（词序列）计算概率的模型称为**语言模型(language model，LM)**。本章将介绍最简单的语言模型——**n 元语法（n-gram）**。尽管 n 元语法模型比最新的基于 RNN 和 transformer 的神经语言模型要简单得多，但它对于理解语言模型的基本概念依然是一个非常重要的基础工具。

## n 元语法

让我们从计算条件概率 $P(w|h)$ 开始，即给定一段历史（history），计算下一个词（word）的概率。假设 $h$ 为 `its water is so transparent that`，$w$ 为 `the`，我们计算：

$$
P({\rm the}|{\rm its\ water\ is\ so\ transparent\ that})
$$

一种方法是用相对频率估计概率：取一个非常大的语料库，计数 `its water is so transparent that` 和 `its water is so transparent that the` 出现的次数，那么

$$
P({\rm the}|{\rm its\ water\ is\ so\ transparent\ that})=\frac{C(\rm its\ water\ is\ so\ transparent\ that\ the)}{C({\rm its\ water\ is\ so\ transparent\ that})}
$$

尽管这种方法在一些情形下表现不错，但在大多数情形下语料库的规模都不够大，不足以给予我们好的估计。这是因为语言是创造性的，我们总是可以创造新的文本，而语料库永远不可能包含所有可能的文本。

类似地，我们可以计算绝对概率 $P(t)$，假设 $t$ 为 `its water is so transparent`，则计数所有 5 个词的序列以及其中多少个是 $t$。但是这样的计算量非常之大！

因此我们引入更聪明的方法——使用**概率的链式法则（chain rule of probability）**：

$$
P(w_{1:n})=P(w_1)P(w_2|w_1)\cdots P(w_n|w_{1:n-1})=\prod_{k=1}^nP(w_k|w_{1:k-1})
$$

然后对于每一个条件概率，将历史序列裁剪为给定长度的序列作为近似：

$$
P(w_k|w_{1:k-1})\approx P(w_k|w_{k-n+1:k-1})
$$

这就是 **n 元语法(n-gram grammar)**模型。例如对于二元语法（二元序列）模型：

$$
P({\rm the}|{\rm its\ water\ is\ so\ transparent\ that})=P({\rm the}|{\rm that})
$$

这种词的概率仅依赖前几个词的假定称为**马尔可夫假定（Markov assumption）**。马尔可夫模型是一类概率模型，其假定我们可以预测某个未来单元的概率，而无需向过去看得太远。

!!! note "说明"
    n-gram 既可以指 n 元语法，也可以指 n 元序列，即文本中连续的 n 个词。

接下来的问题是如何估计 n 元序列的概率？依然是计算相对频率（relative frequency）：

$$
P(w_n|w_{n-N+1:n-1})=\frac{C(w_{n-N+1:n-1}w_n)}{\sum_w C(w_{n-N+1:n-1}w)}=\frac{C(w_{n-N+1:n-1}w_n)}{C(w_{n-N+1:n-1})}
$$

!!! note "说明"
    这种估计实际上是一种最大似然估计（maximum likelihood estimation，MLE），也就是说这样得到的一组参数使得训练语料库本身出现的概率最大（在给定模型的情况下）。

让我们以一个由三句话组成的迷你语料库为例。我们首先需要增强每一句话，在句首和句尾分别增加一个特殊符号 `<s>` 和 `</s>`。

!!! note "说明"
    引入句首和句尾符号是因为，句子本身只有有限长度（而不是双向无限延伸），符号 `<s>` 和 `</s>` 标志着句子的开始与结束，概率 $P(w_1|<s>)$、$P(</s>|w_n)$ 计算的是句子的开始和结束事件。

<pre>
    &lt;s&gt; I am Sam &lt;/s&gt;
    &lt;s&gt; Sam I am &lt;/s&gt;
    &lt;s&gt; I do not like green eggs and ham &lt;/s&gt;
</pre>

计算该语料库中一些二元序列的概率：

$$
\begin{aligned}
P({\rm I}|<s>)=\frac{2}{3} && P({\rm Sam}|<s>)=\frac{1}{3} && P({\rm am}|{\rm I})=\frac{2}{3}\\
P(</s>|{\rm Sam})=\frac{1}{2} && P({\rm Sam}|{\rm am})=\frac{1}{2} && P({\rm do}|{\rm I})=\frac{1}{3} \\
\end{aligned}
$$

再来看一个更大的语料库 now-defunct Berkeley Restaurant Project——上世纪的一个基于伯克利餐厅数据库的问答系统，下面是一些文本归一化后的用户询问示例：

<pre>
​    can you tell me about any good cantonese restaurants close by
​    mid priced thai food is what i’m looking for
​    tell me about chez panisse
​    can you give me a listing of the kinds of food that are available
​    i’m looking for a good place to eat breakfast
​    when is caffe venezia open during the day
</pre>

下图展示了语料库中部分二元序列的频率。注意到其中大部分的值都是 0，实际上这里选择的词还是相互连贯的，随机选择一组词所产生的矩阵将会更加稀疏。

![](https://i.loli.net/2020/12/22/WNXBeoKx5pVy9l4.png)

下图展示了相应的相对频率：

![](https://i.loli.net/2020/12/22/JjvKbws5F7VLemo.png)

这里再补上一些有用的概率

$$
\displaylines{
P({\rm i}|<s>) = 0.25 \\
P({\rm food}|{\rm english}) = 0.5 \\
P({\rm english}|{\rm want}) = 0.0011 \\
P(</s>|{\rm food}) = 0.68
}
$$

现在我们可以计算诸如 `I want English food` 或 `I want Chinese food` 这样的句子的概率了，例如：

$$
\displaylines{
P({\rm <s>\ I\ want\ English\ food\ </s>})=.25\times .33\times .0011\times .5\times .68=.000031 \\
P({\rm <s>\ I\ want\ Chinese\ food\ </s>})=.25\times .33\times .0065\times .52\times .68=.000190
}
$$

二元文法的统计信息中捕获到的语言现象包括文法规则（例如 `want` 后面接 `to` 或名词）、任务特性（例如句子高概率以 `I` 开始），甚至语言之外的信息（例如 `want chinese` 是 `want English` 的约 6 倍）。

!!! note "实际应用"
    尽管出于教学目的，这里只描述了二元语法，但实践中更常用的是三元语法（trigram），甚至是四元（4-gram）、五元语法（5-gram）（如果有足够的训练数据）。对于这些更大的 n 元语法，我们需要在句首增加额外的上下文。例如对于三元语法，句首要增加两个 `<s>` 标记，计算第一个三元序列的概率为 $P(I|<s><s>)$。

    实践中总是以对数形式表示和计算语言模型的概率，即对数概率（log probabilities），因为概率在连乘过程中可能会造成数值下溢。这样计算概率时就有：

    $$
    p_1\times p_2\times p_3\times p_4=\exp(\log p_1+\log p_2+\log p_3+\log p_4)
    $$
    
## 评估语言模型

评估一个语言模型的表现的最佳方法是将它嵌入到应用中并衡量应用的提升，这种端到端的评估称为**外部评估（extrinsic evaluation）**。然而端到端地运行大型 NLP 系统十分昂贵，因此作为替代，我们采用一些指标来快速地评估语言模型的潜在提升，这种独立于任何应用的评估称为**内部评估（intrinsic evaluation）**。

### 困惑度

在实践中我们不采用原始概率作为评估语言模型的指标，而是采用一个称为**困惑度（perplexity，pp）**的变量。对于一个测试集 $W=w_1w_2\cdots w_N$，一个语言模型的困惑度计算为：

$$
PP(W)=P(w_1w_2\cdots w_N)^{-1/N}=\sqrt[N]{\frac{1}{P(w_1w_2\cdots w_N)}}
$$

使用链式法则，对于二元语法模型，有：

$$
PP(W)=\sqrt[N]{\prod_{i=1}^N\frac{1}{P(w_i|w_{i-1})}}
$$

注意到由于条件概率位于分母的位置，序列的概率越高，则困惑度越低。在后面我们将看到困惑度与信息论中的熵的概念关系紧密。

我们可以从另一个角度来理解困惑度——作为一种语言的**加权平均分支因子（weighted average branching factor）**。一种语言的分叉因数是在每一个位置可以填入的可能的词的数量。例如假设反复掷一枚均匀骰子的结果构成了一个序列，那么这种迷你语言的分支因子就是 6，计算得到的困惑度也是 6。

又假设这枚骰子不均匀，掷出 1 的概率为 $5/10$，而掷出 2 到 6 的概率都是 $1/10$，那么尽管这种迷你语言的分支因子还是 6，它的困惑度（或者加权分支因子）会减小，因为结果更加容易预测。

最后来看一个关于困惑度如何用于比较不同的 n 元语法模型的例子。我们在来自华尔街日报的 3800 万词的训练集上分别训练一元、二元和三元语法模型，然后分别计算这三个模型在一个 150 万词的测试集上的困惑度，结果为

|            | 一元序列 | 二元序列 | trigram |
| ---------- | ------- | ------ | ------- |
| perplexity | 962     | 170    | 109     |

可以看到，n 元语法的 n 越大，所能提供的关于词序列的信息就越多，条件概率的值总体上越大（例如 $P({\rm to}|{\rm I\ want})>P({\rm to})$ ），困惑度越低。

## 泛化

n 元语法模型，就像其他统计模型一样，依赖于训练语料库，这体现在以下两个方面：

1. 概率经常会编码训练语料库的一些特征
2. n 越大，n 元语法模型对训练语料库的建模效果越好

为了展示高阶 n 元语法模型的能力，下面展示了在莎士比亚作品集上训练的一元、二元、三元和四元语法模型随机生成的句子：

![](https://i.loli.net/2020/12/23/HXqxodWuTL3698D.png)

可以看到训练模型时提供的上下文越多，则生成的句子越连贯。一元语法模型的句子中词与词之间没有关联，二元语法模型的句子的词与词只有相邻的关联，三元和四元语法模型的句子乍一看就很像莎士比亚的风格了。句子 `It cannot be but so.` 更是直接来自 `King John` 原文，这是因为莎士比亚作品集语料库并没有很大（N=884647, V=29066），因此 n 元语法模型的概率矩阵非常稀疏。对于四元语法模型，一旦之前的词是 `It cannot be but`，那么下一个词只可能选择 `(that, I, he, thou, so)` 中的一个；实际上，在很多情况下下一个词根本没有选择的余地。

!!! note "说明"
    没有选择的余地意味着模型生成的文本会照抄训练语料库的原文，这种现象可以理解为过拟合。

再来看在另一个完全不同的语料库——华尔街日报语料库上训练的一元、二元和三元语法模型随机生成的句子：

![](https://i.loli.net/2020/12/23/3Cw4n7Siv2dLltj.png)

将这些句子与图 3.3 中的句子比较，尽管它们都是“类似英文的句子”，但它们的体裁/风格完全不同。因此在实际应用中，我们需要让训练集与测试集有相似的：

* 体裁：例如建立一个法律文件的机器翻译模型，则训练语料库必须由法律文件组成。
* 方言：这在处理社交媒体贴文或口语材料时尤为重要。例如有些推文会使用 African American English（AAL）的一些特征，如词 `finna` 表示即将，拼写 `den` 替代 `then` 等，这在其他方言中都是不存在的。
* 风格：例如郭敬明的小说就具有特有的凡尔赛风格。
* ……

还是稀疏性的问题，任何语料库即使再大也依然是有限规模，在其之上训练的模型也只能（在所有可能的表达中）统计到相当有限的表达，一些可以完美接受的词序列必定不在其中。例如考虑 WSJ Treebank3 语料库中，二元序列 `denied the` 的下一个词，统计结果为：

<pre>
​    denied the allegations:  5
​    denied the speculation:  2
​    denied the rumors:       1
​    denied the report:       1
</pre>

那么三元语法模型随机生成的 `denied the <word>` 必定是上述之一。换言之，假设测试集中有下列短语：

<pre>
    denied the offer
    denied the loan
</pre>

那么三元语法模型计算上述文本的概率为 0，进而无法计算困惑度。事实上，`denied the` 的下一个词可以是很多名词。这显示了 n 元语法模型完全缺乏语言的创造力。

### 未知词

上面讨论了条件概率为 0 时会出现的问题。但这里我们先退一步，考虑绝对概率为 0，即测试集出现了训练集中没有出现的词的情形。

在有些任务中我们不必担心上述问题，因为所有可能出现的词都是已知的。在这样的一个**封闭词汇表（closed vocabulary）**系统中，测试集的词只能来自词汇表，因此不会出现未知词。例如在语音识别或机器翻译任务中，我们有一个固定的（fixed）语音词典或短语表，因此语言模型只能使用该词典或表中的词。

但在另一些情况下我们不得不处理没有出现过的词，称为**未知词（unknown word）**或 **OOV（out of vocabulary）**词，测试集中 OOV 词所占的比例称为 OOV 率（OOV rate）。在一个**开放词汇表（open vocabulary）**系统中，我们在词汇表中增加一个 `<UNK>` 标记，来匹配测试集中所有潜在的未知词。

通常有两种方法划定未知词并计算概率，其一为：

1. 选择固定的词汇表
2. 在文本归一化过程中将训练集中的所有 OOV 词替换为 `<UNK>` 标记
3. 将 `<UNK>` 视作一个普通的词并计算概率，就像训练集中的其他词一样

其二为：

1. 统计训练集中所有词的词频，选择词频 top V（或词频 > n）的词做成词汇表，将剩余词替换为 `<UNK>`
2. 将 `<UNK>` 视作一个普通的词并计算概率，就像训练集中的其他词一样

对于 `<UNK>` 替换词汇的具体选择对于困惑度等指标有一定的影响。一个语言模型可以通过选择小规模的词汇表并为未知词赋予高概率来达成低困惑度。因此在比较不同模型的困惑度时应当确保它们有相同的词汇表。

### 平滑

回到词的绝对概率不为 0 但条件概率为 0 的问题。为了避免模型为这些未出现过的序列赋予零概率，我们必须从其他出现过的序列匀一些概率过来分配给未出现过的序列。这种操作称为**平滑（smoothing）**或**折减（discounting）**。这里将介绍一些平滑的方法。

#### 拉普拉斯平滑

最简单的方法是在归一化之前将频率矩阵的所有计数加一，这种算法称为**拉普拉斯平滑（Laplace smoothing）**（也称为加一平滑（add-one smoothing））。在现代的 n 元语法模型中拉普拉斯平滑的效果已经不佳，但它作为一种基础方法依然能够帮助我们理解概念和作为基线，并且在文本分类等任务中仍被使用。

平滑可以视作是**折减（discount）**一些非零计数来让概率被分配给一些零计数。因此，我们可以以相对折减系数来描述平滑算法，即折减后的计数与折减前的计数的比值：

$$
d_c=\frac{c^*}{c}
$$

还是以之前的 Berkeley Restaurant Project 二元语法为例，与图 3.1、3.2 对比，以下是拉普拉斯平滑后的结果：

![](https://i.loli.net/2020/12/23/VHahDOWkqZo5umF.png)

![](https://i.loli.net/2020/12/23/ZiE8FGlSkXWtv9e.png)

重构频率矩阵以观察平滑算法对原始频率改变了多少：

![](https://s2.loli.net/2022/11/22/VpiBanSsouUyct6.png)

比较平滑前后的结果，我们发现频率和条件概率发生了非常大的改变：$C({\rm want to})$ 从 609 下降到 238，$P({\rm to}|{\rm want})$ 从 .66 下降到 .26，$P({\rm food}|{\rm chinese})$ 更是从 .52 下降到 .052。如此巨大的变化的原因是，V 大于，甚至远大于这一行的 N。

#### 加 k 平滑

由于加一平滑对概率分布的影响太大，一种替代方法是将 1 修改为一个较小的数 $k$，如 0.5、0.1、0.01 等。这里 $k$ 是一个超参数，可以通过在验证集上优化来选择 $k$ 的值。

**加 k 平滑（add-k smoothing）**对于一些任务（如文本分类）有用，但在语言模型上依然表现不好（Gale and Church，1994）。

#### 回退和插值

另一种解决零频率 n 元序列问题的方法是使用更短的上文以获得信息。当高阶 n 元序列出现零频率时，**回退（backoff）**法依次使用更低阶的 n 元序列，直到得到非零条件概率；而**插值（interpolation）**法则总是混合从 1 到 n 阶的各阶 n 元序列的条件概率，然后加权平均取最终值。

简单线性插值法为：

$$
\displaylines{
\hat{P}(w_n|w_{n-2}w_{n-1})=\lambda_1 P(w_n|w_{n-2}w_{n-1})+\lambda_2 P(w_n|w_{n-1})+\lambda_3P(w_n)\\
\lambda_1+\lambda_2+\lambda_3=1
}
$$

在更复杂一点的线性插值法中，参数 $\lambda$ 与上文有关。在这种方法中，如果我们对于一个特定的二元序列有特别准确的计数，那么我们可以假定基于该二元序列的三元序列的计数也更加可信，因此在插值的过程中可以给予三元序列（高阶序列）更高的权重。

$$
\hat{P}(w_n|w_{n-2}w_{n-1})=\lambda_1(w_{n-2:n-1}) P(w_n|w_{n-2}w_{n-1})+\lambda_2(w_{n-2:n-1}) P(w_n|w_{n-1})+\lambda_3(w_{n-2:n-1})P(w_n)
$$

上述两种插值法的参数 $\lambda$ 均为超参数，从 **held-out** 语料库学习得到。held-out 语料库是不同于训练语料库的，专门用于学习超参数的训练集。选择的 $\lambda$ 值应使得 held-out 语料库有最大的似然。寻找 $\lambda$ 的一组最优解有多种方法，其中之一是最大期望算法（EM algorithm）。

在回退法中，如果某一个 n 元序列有零频率，那么回退一步到 n-1 元序列，直到得到非零条件概率。此时应对高阶序列的概率做一些折减，以分配给低阶序列，否则将零频率的 n 元序列的概率替换为低阶序列的概率后，所有 n 元序列的概率之和将大于 1。此外，不同的得到非零条件概率的阶数也应该对应不同的权重。**Katz 回退（Katz backoff）**法如此计算

$$
P_{bo}(w_n|w_{n-N+1:n-1})=\begin{cases}P^*(w_n|w_{n-N+1:n-1}),\quad C(w_{n-N+1:n}>0)\\
\alpha(w_{n-N+1:n-1})P_{bo}(w_n|w_{n-N+2:n-1}),\quad {\rm otherwise}
\end{cases}
$$

其中 $P^*$ 是折减的概率，折减方法依然是归一化。

Katz 回退法经常和一种称为 Good-Turing 的平滑方法结合使用，称为 Good-Turing 回退法。该算法详细描述了计算 Good-Turing 平滑以及 $P^*$ 和 $\alpha$ 值的方法。

### Kneser-Ney 平滑

一种最常用并且效果最好的 n 元序列平滑方法就是插值 Knerser-Ney 算法（Kneser and Ney，1995）。该方法源于一种称为**绝对折减（absolute discounting）**的方法。

我们先从 Church and Gale（1991）这里借用一种思路。他们在 AP newswire 的一个 2200 万词的训练语料库上统计了所有二元序列的计数，然后检查计数为某个值 $c$ 的所有二元序列在另一个 2200 万词（held-out 语料库）上的平均计数是多少，下面是 $c$ 取 0 到 9 的结果：

![](https://i.loli.net/2020/12/24/LstYo2a9WwRGVUX.png)

我们发现当 $c$ 大于等于 2 时，held-out 语料库上的计数近似为 $c-0.75$。绝对折减方法就是基于这个发现，将每个计数减去固定值 $d$。插值绝对折减方法应用在二元序列的计算公式为：

$$
P_{AD}(w_i|w_{i-1})=\frac{C(w_{i-1}w_i)-d}{\sum C(w_{i-1}*)}+\lambda(w_{i-1})P(w_i)
$$

其中第一项是折减的二元序列，第二项是一元序列的插值项。我们可以令 $d=0.75$，或者当二元序列计数为 1 时单独令 $d=0.5$。

Kneser-Ney 折减（Kneser and Ney，1995）增强了绝对折减，通过改进对低阶序列分布的处理。考虑预测以下句子的下一个词，使用二元序列和一元序列的插值方法：

<pre>
    I can’t see without my reading _______________
</pre>

词 `glasses` 的概率应该比其他词，例如 `Kong`，要大得多，但实际上由于 `Hong Kong` 是一个高频词（取决于语料库），一个标准的一元序列模型中 `Kong` 的概率会比 `glasses` 高得多。我们这里想要的效果是 `Kong` 基本上只跟在 `Hong` 之后，而 `glasses` 则有更加广泛的分布。

换言之，我们希望一元序列模型能够预测词 $w$ 可以作为新的接续的概率，而不仅是出现概率。因此 Kneser-Ney 的想法是基于 $w$ 接续在多少个不同的词之后计算接续概率 $P_{\rm CONTINUATION}$，我们假定如果在过去的统计中 $w$ 接续在多个词之后，那么它就更有可能接续在另外的词之后：

$$
P_{\rm CONTINUATION}(w)\propto |\{v:C(vw)>0 \}|
$$

要将上面的计数转换为概率，我们用二元序列的总类型数来归一化：

$$
P_{\rm CONTINUATION}(w)=\frac{|\{v:C(vw)>0 \}|}{|\{(*,*):C(**)>0 \}|}
$$

这样，`Kong` 就会因为仅接续在 `Hong` 之后而使得接续概率非常小。

最终，插值 Kneser-Ney 平滑法应用在二元序列的公式即为：

$$
P_{KN}(w_i|w_{i-1})=\frac{\max(C(w_{i-1}w_i)-d,0)}{\sum C(w_{i-1}*)}+\lambda(w_{i-1})P_{\rm CONTINUATION}(w_i)
$$

其中 $\lambda(w_{i-1})$ 为归一化参数，用于分配折减的概率：

$$
\lambda(w_{i-1})=\frac{d}{\sum C(w_{i-1}*)}|\{w:C(w_{i-1}w)>0\}|
$$

该参数刻画了 $w_{i-1}$ 的下一个词的多样性。设想，如果词 $w_{i-1}$ 出现了 100 次，但接续在之后的词都是同一个，那么参数等于 $d/100$，即分配的概率很小。

!!! note "说明"
    这里既衡量了二元序列中前一个词所能接续后一个词的数量，又衡量了后一个词所能接续前一个词的数量。前者越大则分配的概率越大，后者越大则分配给它的比例越大。

扩展到 n 元序列的递归公式如下：

$$
P_{KN}(w_i|w_{i-n+1:i-1})=\frac{\max(C(w_{i-n+1:i-1}w_i)-d,0)}{\sum C(w_{i-n+1:i-1}*)}+\lambda(w_{i-n+1:i-1})P_{KN}(w_i|w_{i-n+2:i-1})
$$

其中计数 $C$ 的定义取决于是否是对最高阶计数：

$$
C=\begin{cases}{\rm count}(),\quad {\rm for the highest order}\\
{\rm continuationcount}(),\quad {\rm for lower order}
\end{cases}
$$

continuation count（接续计数）表示序列所能接续前一个词的类型数量。如同前面所分析的，越大表示其可能的应用范围越广。

在递归结束时，一元序列使用均匀分布插值，其中参数 $\epsilon$ 表示空字符串：

$$
P_{KN}(w)=\frac{\max(C(w)-d,0)}{\sum C(*)}+\lambda(\epsilon)\frac{1}{V}
$$

Kneser-Ney 平滑效果最好的版本是**修正的 Kneser-Ney 平滑（modified Kneser-Ney smoothing）**，来自于 Chen and Goodman（1998）。该方法为计数为 1、2、3 或更多的 n 元序列分别使用了不同的折减值 $d$。

## 傻瓜回退

通过使用互联网上的文本，我们可以建立极其巨大的语言模型。2006 年 Google 发布了 The Web 1 Trillion 5-gram 语料库（互联网一万亿五元序列语料库），其包含来自可以公开访问的英文网页的 1,024,908,267,229 词的文本当中出现至少 40 次的所有长度为 5 的序列中提取的一元序列到五元序列。2012 年 Google 又发布了 Google Books Ngrams 语料库，其包含从中文、英文、法文、德文、希伯来文、意大利文、俄文和西文藏书中提取的 n 元序列，共 8000 亿个 token。更小但更精心构建的英文 n 元序列语料库包含 the million most frequent n-grams（最高频的百万个 n 元序列），提取自 COCA（Corpus of Contemporary American English）十亿词的美式英语语料库。COCA 是一个均衡的语料库，其包含的来自不同体裁的词的数量大致均等：网络、报纸、会话稿、小说等等，年代为 1990-2019，有每一个 n 元序列的上下文，有体裁和出处的标签。

下面是一些来自 Google Web 语料库的四元序列的示例：

![](https://s2.loli.net/2022/12/01/bGzmk5IiYgMv9Ke.png)

当构建使用如此大规模的 n 元序列数据集的语言模型时，效率就显得十分关键。每一个词通常在内存中以一个 64 位哈希数表示，而词本身存储在硬盘上。概率通常使用 4-8 位进行量化（而不是 8 字节浮点数）。

n 元序列语言模型也可以通过剪枝来缩减大小，例如仅存储计数大于某个阈值的 n 元序列（例如对于 Google 的发布，计数阈值为 40），或使用熵来剪枝更加不重要的 n 元序列（Stolcke，1998）。另一种选项是使用诸如 **Bloom filters** 之类的技巧构建近似的语言模型（Talbot and Osborne 2007，Church et al. 2007）。最后，高效的语言模型工具包，例如 KenLM（Heafield 2011，Heafield et al. 2013)），使用有序数组，从而高效地结合概率和回退为单个值，并且使用归并排序以最少的迭代大型语料库的次数来高效地构建概率表。

尽管这些工具包的出现使得完全使用 Kneser-Ney 平滑构建互联网规模的语言模型成为可能，但 Brants 等人（2007）指出对于非常大的语言模型，一个更简单的回退法就已经足够，该算法称为**傻瓜回退（stupid backoff）**。傻瓜回退放弃了尽量使语言模型成为一个真正的概率分布的想法。它不折减高阶概率，而如果高阶概率为 0，则简单地回退到低阶序列，乘以一个固定（与上下文无关）的权重。此算法不产生一个概率分布，因此我们用 $S$ 来表示：

$$
S(w_i|w_{i-N+1:i-1})=\begin{cases}\frac{{\rm count}(w_{i-N+1:i})}{{\rm count}(w_{i-N+1:i-1})},\quad {\rm if count}(w_{i-N+1:i})>0\\
\lambda S(w_i|w_{i-N+2:i-1}),\quad {\rm otherwise}
\end{cases}
$$

回退结束于一元序列，其有 $S(w)=\frac{{\rm count}(w)}{N}$。Brants 等人（2007）发现 $\lambda$ 的值取 0.4 的效果不错。

## 困惑度与熵的关系

（参考信息论）
