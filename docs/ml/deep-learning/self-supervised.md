# 自监督学习模型

## 讨论

各个大型语言模型的参数量比较如下：

| 模型               | 参数量            |
| ------------------ | ----------------- |
| ELMO               | 94,000,000        |
| BERT               | 340,000,000       |
| GPT-2              | 1,542,000,000     |
| Megatron           | 80,000,000,000    |
| T5                 | 110,000,000,000   |
| GPT-3              | 175,000,000,000   |
| Switch Transformer | 1,600,000,000,000 |

## BERT (Bidirectional Encoder Representations from Transformers)

### 参考

* [BERT 论文逐段精读【论文精读】](https://www.bilibili.com/video/BV1PL411M7eQ)
* [【機器學習2021】自督導式學習 (Self-supervised Learning) (二) – BERT簡介](https://www.youtube.com/watch?v=gmsMY5kc-zw&list=PLJV_el3uVTsMhtt7_Y6sgTHGHp1Vb2P2J&index=19)
* [【機器學習2021】自督導式學習 (Self-supervised Learning) (三) – BERT的奇聞軼事](https://www.youtube.com/watch?v=gmsMY5kc-zw&list=PLJV_el3uVTsMhtt7_Y6sgTHGHp1Vb2P2J&index=20)

### 论文

* [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)：
    * 提出 BERT 模型，基于 Transformer 的编码器，双向编码
    * 提出预训练的方法；将预训练+微调引入 NLP，可用于解决众多问题
    * 将具有亿数量级参数的模型引入 NLP，引发一轮暴力竞赛
    * 由于舍弃解码器，不适合机器翻译、摘要提取等生成式任务
* [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942)

### 细节

#### 模型架构

BERT 模型就是一个 Transformer 的编码器。

#### 训练

通过以下 2 个任务预训练 BERT 模型：

1. Masked LM：从语料库中选取一段文字，随机遮罩（mask）部分 token 后输入到模型，遮罩位置的输出向量再输入到一个线性层并输出一个 token 的概率分布，将其与原 token 计算交叉熵损失。如下图所示：

![](https://s2.loli.net/2023/02/07/8v1bVkeqcDtan7N.png)

其中 BERT 模型和线性模型是共同训练的。

2. Next Sentence Prediction(NSP)：从语料库中选取两个句子（第二个句子选取第一个句子的后一句，或任意选取一句），拼接后输入到模型，输出一个向量，其再输入到一个线性层并输出一个二分类结果（指示这两个句子是否有前后连接关系）。如下图所示：

![](https://s2.loli.net/2023/02/07/LZjf5nlPUugHXmI.png)

许多研究认为这个任务对于训练没有帮助，因为难度太小，模型没有学到太多东西。

[ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942) 提出任务 Sentence Order Prediction(SOP) 来替换该任务。SOP 中选取的两个句子一定是相邻的，只是可能将它们颠倒顺序。

预训练完成后，BERT 模型继续被用于下游任务，在这些任务中使用较少的标注数据微调（fine-tune）。如下图所示：

![](https://s2.loli.net/2023/02/07/wClHBUOVYqRXSvy.png)

BERT 模型可以用于下列下游任务：

![](https://s2.loli.net/2023/02/07/lg1BPHq5zNMSkEL.png)

![](https://s2.loli.net/2023/02/07/iqLe5yBTvSZJ4bP.png)

![](https://s2.loli.net/2023/02/07/Z6bUw2Wyrt9KvpQ.png)

![](https://s2.loli.net/2023/02/07/tS7BLQcrIbxAJRw.png)

![](https://s2.loli.net/2023/02/07/vOxRZAS9pEKdbmn.png)

![](https://s2.loli.net/2023/02/07/GRijL8OWne9tAcP.png)

### 扩展

#### 预训练 Transformer 模型

**论文**

* [MASS: Masked Sequence to Sequence Pre-training for Language Generation](https://arxiv.org/abs/1905.02450)
* [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461)

#### 多语言 BERT

### 讨论

为什么 BERT 有效？可以考虑以下几个因素：

* 在预训练过程中，模型通过 Masked LM 任务获取从上下文的 token 推断当前位置的 token 的能力，这实质上是一种语言能力。预训练之后的模型在每个位置的输出表示，既包含当前 token 的信息，又抽取了前后 token 的信息，可以视作是考虑上下文的词嵌入（contextualized word embedding），因而提供了更强的基础语言理解能力。
* 巨大规模的训练语料库使得模型学习到各种上下文，相当于获取丰富的语言经验。

### 应用


## GPT (Generative Pretrained Transformer)

### 参考

* [【機器學習2021】自督導式學習 (Self-supervised Learning) (四) – GPT的野望](https://www.youtube.com/watch?v=WY_E0Sd4K80&list=PLJV_el3uVTsMhtt7_Y6sgTHGHp1Vb2P2J&index=21)

### 论文

* [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)

### 细节

#### 模型架构

GPT 模型类似于一个 Transformer 的解码器，但是没有跨注意力层（因为没有编码器）。

#### 训练

通过 Predict Next Token 任训练 GPT 模型：从语料库中选取一段文字输入到模型（自注意力层为 masked），每个位置的输出向量再输入到一个线性层并输出一个 token 的概率分布，将其与下一个位置的 token 计算交叉熵损失。如下图所示：

![](https://s2.loli.net/2023/02/09/2yzPNaVR5OMd3ip.png)

### 讨论

与 BERT 模型相比，GPT：

* 专注于文本生成任务，并试图用文本生成解决一切 NLP 问题。
* 因模型规模过大而难以作进一步微调。

GPT 模型的有效性主要来自于巨大规模的模型和训练语料库，使得模型学习到 token 之间复杂/多样/高级的关系/模式/组合。

### 应用

* [ChatGPT](https://chat.openai.com/chat)
    * 推出仅两个月月活用户突破 1 亿
    * 介绍视频：[ChatGPT (可能)是怎麼煉成的 - GPT 社會化的過程](https://www.youtube.com/watch?v=e0aKI2GGZNg)

## GLUE

[General Language Understanding Evaluation(GLUE)](https://gluebenchmark.com/) 是一套综合评估自然语言理解系统的 benchmark。其有对应的中文 benchmark [中文语言理解测评基准(CLUE)](https://cluebenchmarks.com/)。

[Super GLUE](https://super.gluebenchmark.com/) 是 GLUE 的新版本，其更新了任务，并提高了难度。

可以在 leaderboard 中找到目前先进的自监督学习模型（大型语言模型）。

## CV 领域的自监督学习模型

### 论文

* [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)
* [Bootstrap your own latent: A new approach to self-supervised Learning](https://arxiv.org/abs/2006.07733)

## 语音领域的自监督学习模型

### 参考

* [s3prl speech toolkit](https://github.com/s3prl/s3prl)
