# 自监督学习模型

## 讨论

各个大型语言模型的参数量比较如下：

| 模型               | 参数量 |
| ------------------ | ------ |
| ELMO               | 94M    |
| BERT               | 340M   |
| GPT-2              | 1.5B   |
| Megatron           | 80B    |
| Turing NLG         | 17B    |
| T5                 | 110B   |
| LamDA              | 137B   |
| GPT-3              | 175B   |
| PaLM               | 540B   |
| Switch Transformer | 1.6T   |

## ELMo

### 论文

* [Deep contextualized word representations (Peters, 2018)](https://arxiv.org/abs/1802.05365)
* [Universal Language Model Fine-tuning for Text Classification (Howard, 2018)](https://arxiv.org/abs/1801.06146)

### 讨论

ELMo 和 ULMFiT 是最后一代使用 LSTM 层的自监督预训练模型。之后的大型语言模型全部基于自注意力层（Transformer 架构）。

## BERT (Bidirectional Encoder Representations from Transformers)

### 参考

* [BERT 论文逐段精读【论文精读】](https://www.bilibili.com/video/BV1PL411M7eQ)
* [【機器學習2021】自督導式學習 (Self-supervised Learning) (二) – BERT簡介](https://www.youtube.com/watch?v=gmsMY5kc-zw&list=PLJV_el3uVTsMhtt7_Y6sgTHGHp1Vb2P2J&index=19)
* [【機器學習2021】自督導式學習 (Self-supervised Learning) (三) – BERT的奇聞軼事](https://www.youtube.com/watch?v=gmsMY5kc-zw&list=PLJV_el3uVTsMhtt7_Y6sgTHGHp1Vb2P2J&index=20)

### 论文

* [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (Devlin, 2018)](https://arxiv.org/abs/1810.04805)：
    * 提出 BERT 模型，基于 Transformer 的编码器，双向编码
    * 提出预训练的方法；将预训练+微调引入 NLP，可用于解决众多问题
    * 将具有亿数量级参数的模型引入 NLP，引发一轮暴力竞赛
    * 由于舍弃解码器，不适合机器翻译、摘要提取等生成式任务
* [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations (Lan, 2019)](https://arxiv.org/abs/1909.11942)

### 细节

#### 模型架构

BERT 模型就是一个 Transformer 的编码器。

#### 训练

通过以下 2 个任务预训练 BERT 模型：

1. Masked LM：从语料库中选取一段文字，随机遮罩（mask）部分 token 后输入到模型，遮罩位置的输出向量再输入到一个线性层并输出一个 token 的概率分布，将其与原 token 计算交叉熵损失。如下图所示：

    ![](https://s2.loli.net/2023/02/07/8v1bVkeqcDtan7N.png)

    其中 BERT 模型和线性模型是共同训练的。

2. Next Sentence Prediction (NSP)：从语料库中选取两个句子（第二个句子选取第一个句子的后一句，或任意选取一句），拼接后输入到模型，输出一个向量，其再输入到一个线性层并输出一个二分类结果（指示这两个句子是否有前后连接关系）。如下图所示：

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

微调时，可以选择冻结 BERT 模型的参数（仅训练下游模型），也可以选择不冻结（训练下游模型和 BERT 模型）。一些研究指出，使用后一种做法，整个模型的表现更好。

### 扩展

#### 遮罩方法

**论文**

* [Pre-Training with Whole Word Masking for Chinese BERT (Cui, 2019)](https://arxiv.org/abs/1906.08101)
* [ERNIE: Enhanced Representation through Knowledge Integration (Sun, 2019)](https://arxiv.org/abs/1904.09223)
* [SpanBERT: Improving Pre-training by Representing and Predicting Spans (Joshi, 2019)](https://arxiv.org/abs/1907.10529)

在 BERT 的原始论文中，遮罩的 token 是随机选取的。Cui（2019）和 Sun（2019）分别提出以（中文的）整词为单位和以（英文的）短语和命名实体为单位进行遮罩，发现训练效果更好。Joshi（2019）提出遮罩的长度是一个随机变量（在 1-10 之间随机取值）。

#### 多语言 BERT（Multi-BERT）

#### 多任务学习

**论文**

* [Parameter-Efficient Transfer Learning for NLP (Houlsby, 2019)](https://arxiv.org/abs/1902.00751)
* [BERT and PALs: Projected Attention Layers for Efficient Adaptation in Multi-Task Learning (Stickland, 2019)](https://arxiv.org/abs/1902.02671)

前面已经提到，对于下游任务微调整个模型（下游模型 + BERT 模型）得到的表现更好，但如果为每个下游任务都微调一次 BERT 模型，则需要分别存储每一次微调后的 BERT 模型参数，导致占用大量存储空间，参数利用率低，如下图所示：

![](https://s2.loli.net/2023/02/14/DZjYFAcgb9vmt1R.png)

一些研究提出为 BERT 模型嵌入改装模块（adaptation module），微调时只更新改装模块的参数，从而复用预训练参数，如下图所示：

![](https://s2.loli.net/2023/02/14/vNFab2KyGo1wBiW.png)

### 讨论

为什么 BERT 有效？可以考虑以下几个因素：

* 在预训练过程中，模型通过 Masked LM 任务获取从上下文的 token 推断当前位置的 token 的能力，这实质上是一种语言能力。预训练之后的模型在每个位置的输出表示，既包含当前 token 的信息，又抽取了前后 token 的信息，可以视作是考虑上下文的词嵌入（contextualized word embedding），因而提供了更强的基础语言理解能力。
* 巨大规模的训练语料库使得模型学习到各种上下文，相当于获取丰富的语言经验。

更加深入的讨论请参阅以下论文：

* [Visualizing and Understanding the Effectiveness of BERT (Hao, 2019)](https://arxiv.org/abs/1908.05620)

### 应用


## GPT (Generative Pretrained Transformer)

### 参考

* [GPT，GPT-2，GPT-3 论文精读【论文精读】](https://www.youtube.com/watch?v=t70Bl3w7bxY)
* [【機器學習2021】自督導式學習 (Self-supervised Learning) (四) – GPT的野望](https://www.youtube.com/watch?v=WY_E0Sd4K80&list=PLJV_el3uVTsMhtt7_Y6sgTHGHp1Vb2P2J&index=21)

### 论文

* [Improving Language Understanding by Generative Pre-Training (Alec, 2018)](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
* [Language Models are Unsupervised Multitask Learners (Alec, 2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
* [Language Models are Few-Shot Learners (Brown, 2020)](https://arxiv.org/abs/2005.14165)
* [Training language models to follow instructions with human feedback (Ouyang, 2022)](https://arxiv.org/abs/2203.02155)

### 细节

#### 模型架构

GPT 模型类似于一个 Transformer 的解码器，但是没有跨注意力层（因为没有编码器）。

#### 训练

通过 Predict Next Token 任务训练 GPT 模型：从语料库中选取一段文字输入到模型（自注意力层为 masked），每个位置的输出向量再输入到一个线性层并输出一个 token 的概率分布，将其与下一个位置的 token 计算交叉熵损失。如下图所示：

![](https://s2.loli.net/2023/02/09/2yzPNaVR5OMd3ip.png)

#### In-context Learning

GPT3 的目标是用文本生成的方式解决各种 NLP 任务（GPT 也只能进行文本生成，没有对于下游任务微调的步骤）。任务描述和少量示例被直接放到输入中，模型根据该任务描述和示例来完成相应的任务，称为上下文中学习（In-context Learning），如下图所示：

![](https://s2.loli.net/2023/02/14/7bcQCfKGS2z4rlk.png)

GPT3 在 SuperGLUE 上的表现已经超过了微调的 BERT Large：

![](https://s2.loli.net/2023/02/14/7EcgGu3DLYWzIfr.png)

例如，GPT3 生成的新闻报道已经能够以假乱真：

![](https://s2.loli.net/2023/02/14/hTCMaGkpj195KlJ.png)

### 扩展

#### Image GPT

**论文**

* [Generative Pretraining from Pixels (Chen, 2020)](https://cdn.openai.com/papers/Generative_Pretraining_from_Pixels_V2.pdf)

OpenAI 的研究人员发现，生成 token 序列的 GPT 模型同样可以用于生成像素序列——从而生成图片。

### 讨论

GPT 的发展历程如下：

* GPT (2018)：使用大规模的未标注的文本语料库对语言模型进行生成式的预训练，然后在下游任务中使用较少的标注的语料库进行微调（预训练+微调的方法在 CV 领域已行之有年，但直到 GPT 和 BERT 提出自监督预训练任务并开展实践，这一方法才被广泛应用于 NLP 领域）。与以往的方法不同，GPT 在微调时对输入文本作相应的 adaptation，而尽量不对模型架构作修改，以达到高效的迁移。基于 Transformer 解码器构建的文本生成模型在研究的 12 个 NLP 任务中的 9 个取得了 SOTA 的效果。
* GPT-2 (2019)：模型规模是 BERT 的 5 倍，获得更好的表现。尝试用语言模型做 In-context Learning，但效果不佳。
* GPT-3 (2020)：模型规模和训练语料库是 GPT-2 的 100+ 倍，在各种 NLP 任务上展现出好的效果。
* InstructGPT (2022)：
* GPT-4 (2023?)

从 GPT 到 GPT-3，模型的规模和训练语料库高速膨胀，这是模型在文本生成乃至各种 NLP 任务上有效性的主要来源（“大力出奇迹”）。模型的巨大规模使得模型拥有学习到 token 之间复杂/多样/高级的关系/模式/组合的 capacity。

但从 GPT-4 开始，模型的规模不再增长，而是将重心放在优化内部结构上。 

与 BERT 模型相比，GPT：

* 专注于文本生成任务，并试图通过文本生成完成各种 NLP 任务（In-context Learning，Few-shot Learning）。
* 因模型规模过大而难以作进一步微调。

据小道消息，BERT 的原始论文是在一两个月之内完成的，很有可能一作是受到 GPT 论文的启发。



### 应用

* [ChatGPT](https://chat.openai.com/chat)
    * 推出仅两个月月活用户突破 1 亿
    * 介绍视频：[ChatGPT (可能)是怎麼煉成的 - GPT 社會化的過程](https://www.youtube.com/watch?v=e0aKI2GGZNg)

## XLNet

## MASS & BART

### 论文

* [MASS: Masked Sequence to Sequence Pre-training for Language Generation (Song, 2019)](https://arxiv.org/abs/1905.02450)
* [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation,  Translation, and Comprehension (Lewis, 2019)](https://arxiv.org/abs/1910.13461)

预训练 Transformer 模型

### 讨论

BERT 模型被认为 不适用于生成类任务

## PaLM

## GLUE

[General Language Understanding Evaluation (GLUE)](https://gluebenchmark.com/) 是一套综合评估自然语言理解系统的 benchmark。其有对应的中文 benchmark [中文语言理解测评基准（CLUE）](https://cluebenchmarks.com/)。

[Super GLUE](https://super.gluebenchmark.com/) 是 GLUE 的新版本，其更新了任务，并提高了难度。

可以在 leaderboard 中找到目前先进的自监督学习模型（大型语言模型）。

## CV 领域的自监督学习模型

### 论文

* [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)
* [Bootstrap your own latent: A new approach to self-supervised Learning](https://arxiv.org/abs/2006.07733)

## 语音领域的自监督学习模型

### 参考

* [s3prl speech toolkit](https://github.com/s3prl/s3prl)
