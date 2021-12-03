> 本文介绍大体概念和方法，具体算法见[此处](./process-algorithm.md)。

[toc]

# 词

在讨论词之前，我们需要定义什么算作是一个词。让我们从一个具体的**语料库(corpus)**开始，Brown语料库由Brown大学于1963-64年编纂，包括500篇不同体裁的英文文本（新闻，小说，纪实文学，学术文章）等。来看来自Brown语料库的下面一个句子：

​        He stepped out into the hall, was delighted to encounter a water brother.

这个句子包括13个单词（不包括标点在内），或15个单词（包括标点在内），是否包括标点在内取决于具体任务。标点起到了分隔文本以及表达含义（语气）的作用。

再来看另一个语料库：Switchboard语料库包括美国1990年代的2430通陌生人之间的电话对话，平均长度为6分钟。这样的语料库没有标点，但有其它的特殊标记。来看来自Switchboard语料库的下面一句**话(utterance)**：

​        I do uh main- mainly business data processing

这句话包含两种disfluency，一种是打断的词例如main-，称为**fragment**，另一种是类似uh或um的语气词，称为**filler**或**filled pause**。是否将这些视作词同样取决于具体应用，例如将讲话转换为文字稿时应当忽略，但在语音识别任务中则应该视作一般的词。

They和they是否是相同的词？这依然取决于具体应用。

cats和cat是否是相同的词？我们知道它们是同一**词元(lemma, 又称词典形, citation form)**的不同**词形(word form)**，对于英文的任务，我们一般采用词形；而对于日文这样的黏着语，我们一般采用词元。

语料库中有多少个词？这个问题有两种回答：语料库中不同的<u>词形</u>的数量，或者说词汇表的规模称为**词类(word type)**；**token**则是语料库的总词数（长度）。

| Corpus                              | Tokens = N   | Types = \|V\| |
| ----------------------------------- | ------------ | ------------- |
| Shakespeare                         | 884 thousand | 31 thousand   |
| Brown corpus                        | 1 million    | 38 thousand   |
| Switchboard telephone conversations | 2.4 million  | 20 thousand   |
| COCA                                | 440 million  | 2 million     |
| Google N-grams                      | 1 trillion   | 13 million    |

上表展示了常用的英文语料库的大致type数和token数。我们看到越大的语料库就有越大的type数，type数 $|V|$ 和token数 $N$ 的关系称为Herdan's Law
$$
|V|=kN^{\beta}
$$
参数 $\beta$ 取决于语料库大小和体裁，但至少对于上表， $\beta$ 取0.67-0.75。

另一种衡量词数量的方法是取<u>词元</u>而非词形，英文词典上的**词条(entry)**即对应这种方法。





# 语料库

语料库中变化最大的维度是语言，以及一种语言中的不同方言。

此外，文章体裁、作者特质、时间等因素都会产生影响。





# 文本标准化

几乎在处理任何自然语言文本之前都需要将文本标准化。文本标准化过程通常包括以下步骤：

1. 分词(tokenizing, segmenting word)
2. 标准化词格式
3. 分句



## 分词

大部分NLP应用的分词器应该满足：

+ 将标点视作单独的token，因为它们包含了句子划分的信息。

+ 将特殊字符串视作单独的token，例如`'Ph.D.', '$45.55', 'http://www.example.com', 'someone@example.com' `等；将多个词的特定表达视作单独的token，例如`'New York', "rock 'n' roll", 'state-of-the-art'`等。因此分词算法应该是一个**命名实体检测(named entity detection)**算法。

  > **命名实体识别(Named Entity Recognition, NER)**，或**命名实体检测(Named Entity Detection, NED)**，是指识别文本中具有特定意义的实体，主要包括人名、地名、机构名、专有名词等，以及时间、数量、货币、比例数值等文字。

+ 正确解析数字，例如在英文中数字表示为`'123,456.78'`，其中使用了comma和dot；而在欧陆语言如西语、法语、德语中，该数字表示为`'123 456,78'`，其中使用了dot和space。

+ 正确展开缩略词，例如将`"we're"`展开为`'we are'`，这在法语中十分常见，例如`"j’ai", "j'y", "c'est"`等；在此基础上，正确区分缩略词、所有格（例如`'the book's cover`）和引用（例如`‘The other class’, she said`）



一个常用的分词标准为Penn Treebank tokenization标准，[Linguistic Data Consortium(LDC)](https://www.ldc.upenn.edu/)的语料库使用此标准。此标准划分缩略词，保留合成词，划分标点，以下为一例：

![](https://i.loli.net/2020/12/21/gflyrBPecUT3OsY.png)



在实践中，分词操作作为文本处理的第一步需要快速完成。因此标准方法是采用基于正则表达式的算法编译成的高效有穷自动机。以下展示了使用简单的正则表达式进行分词的一个例子

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



分词在中文、日文等语言中变得更加复杂，因为在这些语言中不使用空格作为词的分界，并且基本组成单位是汉字(character)、假名等文字。考虑以下句子：

​        姚明进入总决赛

被某分词工具分为3个词：

​        姚明 进入 总决赛

被另一分词工具分为5个词：

​        姚 明 进入 总 决赛

还可以不分词，即将单个字作为词：

​        姚 明 进 入 总 决 赛

实际上，对于大部分中文NLP任务，结果都显示使用字输入比词输入效果更好，因为在大多数应用条件下字是合理的语义级别，而且大多数分词规则会导致词汇表中出现大量的罕见词汇，使用词典又难以涵盖所有领域或新出现的词汇。在少数应用中，中文的分词仍然是有益的。

> 关于中文分词的必要性至今仍存在争议。一些任务显示不分词的结果好于分词，一些人认为词级别的低级特征完全可以由卷积网络或循环网络提取，分词反而引入了额外的误差。我当前认为那些语言高度规范化的文本（如法律文书、学术文章等）需要分词，而其它情形则不需要分词。

相反，日文的汉字或假名作为一个单独的单位又太小，因此依然需要分词；德语的名词复合词不用空格分隔，因此亦需要分词，例如`'Lebensversicherungsgesellschaftsangestellter'`，意为life insurance company employee。对于这些语言，标准的分词算法使用序列模型，该序列模型在人工分词的训练集上进行监督学习。



> | 文字     | 英文                               | 中文                            | 日文   |
> | -------- | ---------------------------------- | ------------------------------- | ------ |
> | 原生分词 | 是                                 | 否                              | 否     |
> | 需要分词 | -                                  | ？                              | 是     |
> | 词元数   | 常用词<br />总共：牛津高阶词典 42k | 常用字<br />总共：新华字典 13k+ | 常用词 |
> |          |                                    |                                 |        |
> |          |                                    |                                 |        |
>
> 



## Byte-Pair Encoding for Tokenization

除了定义词或字作为token之外，我们还有第三种选择：使用数据来自动告诉我们选择何种大小的token。有些时候我们需要比词更大的token（如`'New York Times'`），而有些时候又需要比词更小的token（如`'un-','-er'`）。

使用亚词(subword)token的一个原因是其可以处理未知的词。例如训练集的词汇表包含low, lowest而不包含lower，那么当lower出现在测试集时模型将无法处理这个词。一种解决方法是在token中加入一些前后缀。

最简单的此种算法是**byte-pair encoding**，或称为**BPE**(Sennrich et al., 2016)，其基于一种文本压缩方法(Gage, 1994)，但这里用来分词。算法的基本思想是合并出现频率最高的相邻字符作为新字符。

看一个简单的例子：统计一段文本的所有词和词频，得到的词汇表（词拆散，结尾加上词边界token`_`）和相应的token表为

![](https://i.loli.net/2020/12/21/wIRKlk1am8VvMxz.png)

统计出现频率最高的相邻字符，发现是`r _`（出现9次），那么将`r_`作为新字符加入token表，同时修改词汇表为

![](https://i.loli.net/2020/12/21/7mAKwWvoJsbpfjS.png)

重复上述步骤，总共执行步骤的次数为超参数 $k$，

![](https://i.loli.net/2020/12/21/KnX8ScJ75jkWL2E.png)

![](https://i.loli.net/2020/12/21/QJDrSqtM69AZY8I.png)

![](https://i.loli.net/2020/12/21/UhWqNFdCfViP1s2.png)

在测试过程中我们使用同样的分词方法。如果测试文本中包含`n e w e r _`，那么其将按照上述规则分词为`newer_`；如果包含`l o w e r _`，那么其将按照上述规则分词为`low er_`。

实际应用中对于较大的词汇表， $k$ 通常取几千，这时大多数常用词的完整形式都被加入到token表，而少数罕见词和训练集未出现的词被表示为几部分。



### 词片和贪心分词

也有一些类似于BPE算法的生成token的方法。词片(wordpiece)算法与BPE算法类似，只不过将词边界token`_`从词尾挪到词首，并且合并使得训练数据的语言模型的似然最小的相邻字符，而非出现频率最高的相邻字符。具体内容将在语言模型一章中讨论，但简单来说，词片模型选择的合并方法使得训练语料库有最大的出现概率(Wu et al., 2016)。

> 词片是否就是亚词？



在BERT(Devlin et al., 2019)模型使用的词片分词器~~中，就像BPE算法一样，首先将输入文本用简单的分词器分成粗token，然后不使用词边界token，而是将所有非词首的亚词用特殊符号标记，例如将affordable拆分为`['un', '##aff', '##able']`，然后每一个词~~使用贪心的长度优先匹配算法（或称为最大匹配算法）进行分词。最大匹配算法根据给定的token表（通过学习得到），从字符串的起始位置开始，每次选择最长的匹配token。（需要详细信息：如何从训练集得到token表）

例如给定字符串intention和token表`['in', 'tent', 'intent', ''##tent', '##tention', '##tion', '##ion']`，那么BERT分词器将首先选择`'intent'`，然后选择`'##ion'`，分词结果即为`['intent', '##ion']`。



另一种分词算法为**句片(sentencepiece)**。BPE和词片都假定我们已经有某种初始的分词，但句片模型直接处理原始文本，将空格也处理为一个字符，因此它不需要初始的分词或词表，可以用于中文或日文这种没有空格的语言。



## 词标准化，词元化和词干提取

**词标准化**是将词或token统一为标准形式的操作，即为具有多种形式的词选择一个标准形式，例如为USA和US，选择US为标准形式。

**大小写折叠**也是词标准化方法，将所有字母映射到小写字母。大小写折叠在信息检索(retrieval)、语音识别任务中有利于泛化；但一般不用于语义分析、文本分类、信息提取(extraction)和机器翻译任务，这些任务中大小写依然提供了有用信息，例如对于国家US和代词us等词对的区分的好处大于泛化带来的好处。



在许多NLP任务中我们也希望同一个词的不同形态有相似的表现，例如在搜索引擎中搜索woodchucks也会返回woodchuck相关的页面。这在词法复杂的语言中尤为常见，例如俄文中Москва有不同形态Москвы, Москву。**词元化**是确定两个词具有同一词元的操作，例如am, is, are具有同一词元be，woodchucks, woodchuck具有同一词元woodchuck，Москва, Москвы, Москву具有同一词元Москва。将句子词元化的一个例子是He is reading detective stories→He be read detective story。

如何完成词元化？最精细的方法是对词做完全的词法分析。**词法学(morphology)**是研究词的内部结构和形成方式的学科，词由更小的意义承载单元，即**词素(morpheme)**构成。词素可以分为两大类：**词干(stem)**（中心词素，提供主要含义）和**词缀(affix)**（提供附加含义）。例如cats包括词素cat和-s，西文词amaren(if in the future they would love)包含了语素amar(to love)，复数主语和将来虚拟语气。

> 词元和词干并不完全等同，例如distributing的词元是distribute，但词干是distribut，因为还有名词形式distribution等。词元化相当于词干提取+词干还原两步。

词元化算法往往十分复杂，因此我们有时使用更简单也更粗糙的方法，即从词的末尾提取后缀。这种简化的词法分析方法称为词干提取。使用最广泛的词干提取算法是**Porter词干提取算法**(Porter, 1980)，它将以下文本

​        This was not the map we found in Billy Bones’s chest, but an accurate copy, complete in all things-names and heights and soundings-with the single exception of the red crosses and the written notes.

词干提取为

​        Thi wa not the map we found in Billi Bone s chest but an accur copi complet in all thing name and height and sound with the singl except of the red cross and the written note

> 可以看到这一方法存在诸多错误操作

该算法将一系列的重写规则应用于输入文本，例如下列规则：

​        ATIONAL → ATE (e.g. relational→relate)

​                 ING → $\epsilon$，如果词干包含元音 (e.g., motoring → motor)

​               SSES → SS (e.g., grasses → grass)

具体的规则列表和代码可以在Martin Porter的主页找到。



## 分句

**分句(sentence segmentation)**是文本处理的另一重要步骤。将段落划分为句子的最重要线索为标点，例如dot、comma、问号、感叹号等。问号和感叹号一般是句子的明确分界线，而dot则可能是句子分界线或者缩略词的标记（例如Mr.）。因此分词和分句往往会同时进行处理。

一般而言，分句方法首先确定dot是句子分界线还是词的一部分，缩略词词典可以帮助这一过程（词典可以是现有的或学习得到的）。在Stanford CoreNLP工具包中，如果一个句子结束符(dot, 问号, 感叹号)没有与邻近的字符构成一个token，那么认为句子结束。

