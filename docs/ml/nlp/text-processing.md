# 文本处理

## 词

在讨论处理词之前，我们需要定义什么算作是一个词。让我们从一个具体的**语料库（corpus）**开始，Brown 语料库由 Brown 大学于 1963-64 年编纂，包括 500 篇不同体裁的英文文本（新闻，小说，纪实文学，学术文章）等。来看来自 Brown 语料库的一个句子：

<pre>
    ​He stepped out into the hall，was delighted to encounter a water
    brother.
</pre>

这个句子包含 13 个单词（不包含标点在内），或 15 个单词（包含标点在内），<u>是否包含标点取决于具体任务</u>。<u>标点对于寻找边界（逗号、句号、冒号）和识别含义（问号、感叹号、引号）都至关重要</u>。

再来看另一个语料库：Switchboard 语料库包括美国 1990 年代的 2430 通陌生人之间的电话对话，平均长度为 6 分钟。这样的口语语料库没有标点，但引入了其他的特殊标记。来看来自 Switchboard 语料库的一句**话（utterance）**（一句话是一个句子的口头表达形式）：

<pre>    ​I do uh main- mainly business data processing</pre>

这句话包含两种**不流利（disfluency）**，一种是断开的词例如 main-，称为**碎片（fragment）**，另一种是类似 uh 和 um 的语气词，称为**填充（filler, filled pause）**。是否将这些视作词同样取决于具体应用，例如将讲话转换为文字稿时应当忽略，但在语音识别任务中则应该视作一般的词。不流利实际上可以帮助预测接下来的词，因为它们可能预示着讲话者正在重新组织表达；不流利还可以作为识别讲话者身份的线索。

They 和 they 是否是相同的词？这依然取决于具体应用。对于语音识别任务，它们之间可以不加区分；而对于词性标注和命名实体检测任务，首字母大写是有用的特征而得到保留。

cats 和 cat 是否是相同的词？它们是同一**词元（lemma）**的不同**词形（word form）**。词形是词的完整的屈折或推导形式。对于形态复杂的黏着语，例如日语和阿拉伯语，我们一般处理词元；而对于大部分英文的任务，采用词形就已经足够。

语料库中有多少个词？这个问题有两种回答：语料库的总<u>词类</u>数（types），即互不相同的词的数量，也是词汇表的规模 $|V|$；语料库的总词数（长度）（tokens）$N$。如果我们忽略标点，下面这个来自 Brown 语料库的句子有 16 个词（tokens）和 14 个 词类（types）。

<pre>
    They picnicked by the pool, then lay back on the grass and looked
    at the stars.
</pre>

下表展示了一些常用的英文语料库的大致总词数和总词类数。我们看到越大的语料库就有越多的总词类数，总词类数 $|V|$ 和总词数 $N$ 的关系称为 Herdan's Law 或 Heaps' Law：

$$
|V|=kN^{\beta}
$$

参数 $\beta$ 的值取决于语料库的大小和体裁，但至少对于下表，$\beta$ 取 0.67-0.75。

| Corpus                              | Tokens = N   | Types = \|V\| |
| ----------------------------------- | ------------ | ------------- |
| Shakespeare                         | 884 thousand | 31 thousand   |
| Brown corpus                        | 1 million    | 38 thousand   |
| Switchboard telephone conversations | 2.4 million  | 20 thousand   |
| COCA                                | 440 million  | 2 million     |
| Google N-grams                      | 1 trillion   | 13 million    |

另一种衡量词数量的方法是取<u>词元</u>而非词形（词类）的数量。英文词典可以帮助计数词元：词典的词条数（entries）就是词元数的一个粗略的上界（考虑到一些词元可能有多个词条）。

## 语料库

语料库是一个计算机可以存储或读取的文本或语音集合。

我们所研究的任何特定的一段文本或语音，都是由一个或多个讲话者或写作者，以一种特定语言的特定方言，在一个特定的时间和地点，为一个特定的目的所产生的。

这当中最重要的变化维度就是语言，NLP 算法在能够应用于多种语言时是最有用的。将算法在多种语言，尤其是不同类型的多种语言中进行测试是十分重要的。

此外，大多数语言中也存在多种变体，称为**方言（dialect）**，通常在不同区域或不同社会群体中使用。

讲话者和写作者在一次单独的交流行为中使用多种语言也是相当常见的，这种现象称为**编码转换（code switching）**。

此外，文章体裁、写作者（或讲话者）的人口学特征、时间等因素都会产生影响。

## 文本归一化

几乎在处理任何自然语言文本之前都需要将文本**归一化（normalization）**。文本归一化过程通常包括以下步骤：

1. 分词（tokenizing，segmenting word）
2. 归一化词格式
3. 分句

### 分词

**分词（tokenization）**任务负责将文本划分为一个一个的词。

对于大部分 NLP 应用，其分词器应该满足：

* 将标点视作单独的 token，因为它们包含了句子边界的信息。

* 将包含标点或特殊符号的特殊字符串视作单独的 token，例如 `'Ph.D.','$45.55','http://www.example.com','someone@example.com'` 等。

* 将包含多个词的特定表达视作单独的 token，例如 `'New York',"rock'n'roll",'state-of-the-art'` 等。因此分词与**命名实体检测（named entity detection）**紧密关联。

!!! info "命名实体检测"
    **命名实体检测（Named Entity Detection，NED）**，或**命名实体识别（Named Entity Recognition，NER）**，是指识别文本中具有特定意义的实体，主要包括人名、地名、机构名、专有名词等，以及时间、数量、货币、比例数值等文字。

* 正确解析数字，例如在英文中数字表示为 `'123,456.78'`，其中使用了逗号和点号；而在欧陆语言如西语、法语、德语中，该数字表示为 `'123 456,78'`，其中使用了点号和空格。

* 正确展开缩略词，例如将 `"we’re"` 展开为 `'we are'`，这在法语中十分常见，例如`"j’ai","j’y","c’est"`等；在此基础上，正确区分缩略词、所有格（例如 `the book’s cover`）和引用（例如 `‘The other class’,she said`）

一个常用的分词标准为 Penn Treebank tokenization 标准，[Linguistic Data Consortium（LDC）](https://www.ldc.upenn.edu/)发布的语料库使用此标准。此标准划分缩略词，保留合成词，划分标点，以下为一例：

![](https://i.loli.net/2020/12/21/gflyrBPecUT3OsY.png)

在实践中，分词操作作为文本处理的第一步需要快速完成，因此标准方法是采用基于正则表达式的算法编译成的高效有穷自动机。下面展示了使用简单的正则表达式进行分词的两个例子：

```python
>>> s = "If you don't want me to go, I won't."
>>> s = s.lower().strip()                # 全部小写 
>>> s
"if you don't want me to go, i won't."
>>> s = re.sub(r"([.!?])", r" \1", s)    # 在.!?前面插入一个space
>>> s
"if you don't want me to go, i won't ."
>>> s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)  # 将a-zA-Z.!?之外的字符替换为space
>>> s
'if you don t want me to go i won t .'
```

```python
>>> text = 'That U.S.A. poster-print costs $12.40...'
>>> pattern = r'''(?x) # set flag to allow verbose regexps
... ([A-Z]\.)+ # abbreviations, e.g. U.S.A.
... | \w+(-\w+)* # words with optional internal hyphens
... | \$?\d+(\.\d+)?%? # currency and percentages, e.g. $12.40, 82%
... | \.\.\. # ellipsis
... | [][.,;"’?():-_‘] # these are separate tokens; includes ], [
... '''
>>> nltk.regexp_tokenize(text, pattern)
['That', 'U.S.A.', 'poster-print', 'costs', '$12.40', '...']
```

分词在中文、日文等语言中变得更加复杂，因为在这些语言中不使用空格作为词的分界，并且基本组成单位是汉字（character）、假名等文字。考虑以下句子：

<pre>    ​姚明进入总决赛</pre>

被“Chinese Treebank”分词分为 3 个词：

<pre>    ​姚明 进入 总决赛</pre>

被“北京大学”分词分为 5 个词：

<pre>    ​姚 明 进入 总 决赛</pre>

还可以不分词，即将单个字作为词：

<pre>    ​姚 明 进 入 总 决 赛</pre>

实际上，对于大部分中文 NLP 任务，结果都显示使用字作为输入比词作为输入效果更好，因为对于大多数应用字是合理的语义级别，而且大多数分词规则会导致词汇表中出现大量的罕见词汇，使用词典又难以涵盖所有领域或新出现的词汇。在少数应用中，中文的分词仍然是有益的。

!!! info "中文分词"
    关于中文分词的必要性至今仍存在争议。一些任务显示不分词的结果好于分词，一些人认为词级别的低级特征完全可以由卷积网络或循环网络提取，分词反而引入了额外的误差。我当前认为那些语言高度规范化的文本（如法律文书、学术文章等）需要分词，而其他情形则不需要分词。

与之相对地，日文的汉字或假名作为一个单独的单位又太小，因此依然需要分词；德语的名词复合词不用空格分隔，因此亦需要分词，例如 `'Lebensversicherungsgesellschaftsangestellter'`，意为 life insurance company employee。对于这些语言，标准的分词算法使用**序列模型（sequence model）**，该序列模型在人工分词的训练集上进行监督学习。 

### 用于分词的字节对编码

除了定义 token 为词或字之外，我们还有第三种选择：使用数据来自动告诉我们选择何种大小的 token。有些时候我们需要比词更大的 token（如 `'New York Times'`），而有些时候又需要比词更小的 token（如 `'un-','-er'`）。

为处理未知词问题，现代分词器经常自动引入一组小于词的 token，称为**子词（subword）**。子词可以是任意的子串，也可以是例如**语素（morpheme）** `-est` 或 `-er` 的意义承载单元。在现代分词方案中，大部分的 token 是词，一些 token 是频繁出现的语素或其他子词例如 `-er`。每一个像 `lower` 这样未见过的词都可以表示为像 `low + er` 这样已知的 token 的序列，必要时甚至可以表示为单个字母的序列。

!!! info "语素"
    语素是一种语言的最小意义承载单元，例如词 `unlikeliest` 包含语素 `un-`、`likely` 和 `-est`。

大部分分词方案都包含两部分：一个分词学习器和一个分词器。分词学习器接受一个未处理的训练语料库并推导出一个词汇表（一组 token），分词器接受一个未处理的测试句子并将其划分为词汇表中的 token。三种算法被广泛使用：**字节对编码（byte-pair encoding，BPE）**（Sennrich et al.，2016）、**一元语言建模（unigram language modeling）**（Kudo，2018）和**词片（wordpiece）**（Schuster and Nakajima，2012）。

这当中最简单的算法是字节对编码，其基于一种文本压缩方法（Gage，1994），但这里用来分词。算法的基本思想是合并出现频率最高的相邻字符作为新字符。分词学习器从一个仅包含所有的单个字符的词汇表开始，然后检查训练语料库，选择两个最频繁邻接的符号（设为“A”和“B”），添加合并后的新符号“AB”到词汇表中，并将语料库中的每个邻接的“A”“B”都替换为“AB”。它循环地计数和合并，创建越来越长的字符串，直到 $k$ 次合并完成创建了 $k$ 个新 token——$k$ 是这个算法的超参数。最终的词汇表包含原有的字符外加 $k$ 个新 token。

此算法通常在词的内部运行，因此输入语料库要首先用空白符划分来给出一组字符串，每个字符串对应一个词的所有字母，外加一个特殊的词结束符号 `_`，以及它的计数。让我们来看在一个仅包含 18 个 token 的迷你输入语料库上进行操作的过程：

![](https://i.loli.net/2020/12/21/wIRKlk1am8VvMxz.png)

BPE 算法计数所有邻接字符对，发现频次最高的是 `r _`（出现 9 次），于是合并这两个字符，将 `r_` 作为新字符加入词汇表，同时修改语料库：

![](https://i.loli.net/2020/12/21/7mAKwWvoJsbpfjS.png)

重复上述步骤：

![](https://i.loli.net/2020/12/21/KnX8ScJ75jkWL2E.png)

![](https://i.loli.net/2020/12/21/QJDrSqtM69AZY8I.png)

![](https://i.loli.net/2020/12/21/UhWqNFdCfViP1s2.png)

一旦我们学习到了一个词汇表，就可以用分词器对测试语料库中的句子进行分词：分词器只是在测试数据上执行我们从训练数据中学习到的合并，以学习到的顺序贪婪地依次执行。因此我们首先将测试句子划分为字母，然后应用第一条规则：替换每一处 `r _` 为 `r_`，然后第二条规则：替换每一处 `e r_` 为 `er_`，以此类推。如果测试语料库中包含 `n e w e r _`，那么其将被分词为一个完整的词 `newer_`；如果包含 `l o w e r _`，那么其将被分词为 `low er_`。

实际应用中对于较大的输入语料库，$k$ 通常取几千，这时大多数常用词的完整形式都被加入到词汇表，仅有少数罕见词（和训练语料库中未出现的词）会被表示为几部分。

#### 词片和贪心分词

（需要完善）

也有一些类似于 BPE 算法的生成 token 的方法。词片（wordpiece）算法与 BPE 算法类似，只不过将词边界符 `_` 从词尾挪到词首，并且合并使得训练数据的语言模型的似然最小的相邻字符，而非出现频率最高的相邻字符。具体内容将在语言模型一章中讨论，但简单来说，词片模型选择的合并方法使得训练语料库有最大的出现概率（Wu et al.，2016）。

在 BERT（Devlin et al.，2019）模型使用的词片分词器中，就像 BPE 算法一样，首先将输入文本用简单的分词器分成粗 token，然后不使用词边界 token，而是将所有非词首的子词用特殊符号标记，例如将 affordable 拆分为 `['un','##aff','##able']`，然后每一个词使用贪心的长度优先匹配算法（或称为最大匹配算法）进行分词。最大匹配算法根据给定的 token 表（通过学习得到），从字符串的起始位置开始，每次选择最长的匹配 token。（需要详细信息：如何从训练集得到 token 表）

例如给定字符串 intention 和 token 表 `['in','tent','intent',''##tent','##tention','##tion','##ion']`，那么 BERT 分词器将首先选择 `'intent'`，然后选择 `'##ion'`，分词结果即为 `['intent','##ion']`。

另一种分词算法为**句片（sentencepiece）**。BPE 和词片都假定我们已经有某种初始的分词，但句片模型直接处理原始文本，将空格也处理为一个字符，因此它不需要初始的分词或词表，可以用于中文或日文这种没有空格的语言。

### 词归一化、词元化和词干提取

**词归一化（word normalization）**任务负责将词/token 统一为标准形式，即为具有多种形式的词选择一个标准形式，例如为 USA 和 US，选择 US 为标准形式。对于信息检索（information retrieval）和信息提取（information extraction）任务，如果关键词是 `US`，那么不管文件提到的是 `US` 还是 `USA`，我们都想要看到来自它的信息。

**大小写折叠（case folding）**也是一类归一化方法，其将所有字母映射为小写字母。大小写折叠在许多任务中有利于泛化，例如信息检索、语音识别等；但一般不用于语义分析、文本分类、信息提取和机器翻译任务，在这些任务中大小写依然提供了有用信息，例如对于国家 `US` 和代词 `us` 等词对进行区分的好处大于泛化带来的好处。

在许多 NLP 任务中我们也希望同一个词的几种不同形态有相似的表现，例如在搜索引擎中搜索 `woodchucks` 也会返回 `woodchuck` 相关的页面。这在词法复杂的语言中尤为常见，例如俄文中 Москва（Moscow）有不同形态 Москвы（of Moscow）、Москву（to Moscow）等。**词元化**任务负责确定两个词拥有同一词元，例如 am、is、are 拥有同一词元 be，woodchucks、woodchuck 拥有同一词元 woodchuck，Москва、Москвы、Москву 拥有同一词元 Москва。将句子词元化的一个例子是 `He is reading detective stories` → `He be read detective story`。

如何进行词元化？最精细的方法是对词做完全的词法分析（morphological parsing）。**词法学（morphology）**是研究词的内部结构和形成方式的学科。词由更小的意义承载单元，即**词素（morpheme）**构成。词素可以分为两大类：**词干（stem）**（中心词素，提供主要含义）和**词缀（affix）**（提供附加含义）。例如 `cats` 包括词素 `cat` 和 `-s`，西文词 `amaren`（if in the future they would love）包含了语素 `amar`（to love），复数主语和将来虚拟语气。

!!! note "词元和词干的区别"
    词元和词干并不完全等同，例如 `distributing` 的词元是 `distribute`，但词干是 `distribut`（词缀是 `-ing`）。顺带一提，名词形式 `distribution` 的词干也是 `distribut`（词缀是 `-ion`）。词元化相当于词干提取+词干还原两步。

词元化算法往往十分复杂，因此我们有时使用更简单但也更粗糙的方法，即砍掉词末尾的后缀。这种简化的词法分析方法称为**词干提取（stemming）**。一种最广泛使用的词干提取算法是 **Porter 词干提取器（Porter stemmer）**（Porter，1980），它将以下文本

<pre>
​This was not the map we found in Billy Bones's chest, but an accurate
copy, complete in all things-names and heights and soundings-with the
single exception of the red crosses and the written notes.
</pre>

词干提取为

<pre>
​Thi wa not the map we found in Billi Bone s chest but an accur
copi complet in all thing name and height and sound with the
singl except of the red cross and the written note
</pre>

此算法将一系列的重写规则应用于输入文本，例如下列规则：

* ​ATIONAL → ATE（e.g. relational → relate）
* ​ING → $\epsilon$，如果词干包含元音（e.g. motoring → motor）
* ​SSES → SS（e.g. grasses → grass）

具体的规则列表和代码可以在 Martin Porter 的主页找到。

简单的词干提取器可能会犯下过泛化（不应该做却做）和欠泛化（应该做却未做）错误。

### 分句

**分句（sentence segmentation）**是文本处理的另一重要步骤。将段落划分为句子的最重要线索是标点，例如逗号、句号/点号、问号、感叹号等。问号和感叹号是相对无歧义的句子分界线，而点号则可能是句子分界线或者缩略词的标记（例如 `Mr.`）（甚至两者兼是）。因此分词和分句往往会联合处理。

一般而言，分句方法首先确定点号是句子分界线还是词的一部分，缩略词词典可以提供帮助（词典可以是手工编写的或学习得到的）。在 Stanford CoreNLP 工具包中，句子划分是基于规则的，如果一个句子结束符（点号、问号、感叹号）没有与邻近的字符组成一个 token，则认为句子结束。
