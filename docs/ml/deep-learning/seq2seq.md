# 序列到序列模型

## 参考

* [【機器學習2021】Transformer (上)](https://www.youtube.com/watch?v=gmsMY5kc-zw&list=PLJV_el3uVTsMhtt7_Y6sgTHGHp1Vb2P2J&index=12)
* [《神经网络与深度学习》](https://nndl.github.io/) 15.6 P375, 8.2 P194

## 项目

* [基于注意力的机器翻译](https://github.com/t9k/sample-docs/blob/master/docs/text/attention-based-machine-translation.md)

## 讨论

NLP 的许多任务（阅读理解、机器翻译、文章摘要、情感分析等等）都可以在形式上转换为问答/对话，从而使用 seq2seq 模型进行训练。这一模型在各种任务上的表现都不如这些任务各自的最适模型，但优势在于泛用（可以参照 ChatGPT 在多种 NLP 任务上的表现）。

例如，[Grammar as a Foreign Language (2014)](https://arxiv.org/abs/1412.7449) 将语法解析的结果用序列表示，从而可以使用 seq2seq 模型训练，达到当时的 state-of-the-art 的结果。

## 编码器-解码器结构

### 论文

* [Sequence to Sequence Learning with Neural Networks (Sutskever, 2014)](https://arxiv.org/abs/1409.3215)
* [Listen, Attend and Spell (2015)](https://arxiv.org/abs/1508.01211)

### 扩展

#### 束搜索

解码器在每一个时间步都选择最有可能的 token（贪心搜索），最终得到的不一定是最有可能的序列。确保得到最有可能的序列需要穷举搜索，但其指数时间复杂度无法承受。束搜索是一种折中方法，其在每一个时间步选取常数个最有可能的序列。

束搜索并非对于所有任务都适用，因为其牺牲计算效率是为了换取优化效果。一般来说，其适用于有标准答案的任务（例如语音识别等），而不适用于生成类的、创造性的任务（例如文本生成等）。

**参考**

* Dive into Deep Learning - [10.8. Beam Search](https://d2l.ai/chapter_recurrent-modern/beam-search.html)

**论文**

* [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751)
    * 指出当前解码策略（取最大概率、随机采样等）存在的问题，并提出新的采样方法——Nucleus Sampling

#### Scheduled Sampling

**论文**

* [Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks (2015)](https://arxiv.org/abs/1506.03099)

### 讨论

编码器-解码器是深度学习模型的经典结构之一，许多模型，从进行机器翻译的 LSTM 模型到现在广泛应用 Transformer 模型，都属于编码器-解码器结构。

## 注意力机制（Attention Mechanism）

### 参考

* Dive into Deep Learning
    * [11.1. Queries, Keys, and Values](https://d2l.ai/chapter_attention-mechanisms-and-transformers/queries-keys-values.html)
    * [11.3. Attention Scoring Functions](https://d2l.ai/chapter_attention-mechanisms-and-transformers/attention-scoring-functions.html)
    * [11.4. The Bahdanau Attention Mechanism](https://d2l.ai/chapter_attention-mechanisms-and-transformers/bahdanau-attention.html)

### 论文

* [Neural Machine Translation by Jointly Learning to Align and Translate (Bahdanau, 2014)](https://arxiv.org/abs/1409.0473)

### 细节

#### 机制架构

大部分注意力机制的研究中，机制遵循如下架构：

![](https://s2.loli.net/2023/02/10/DYyFmXh69VcRkOz.png)

其中：

* 注意力权重（attention weight）的计算方法为：

    $$
    \alpha(\pmb q,\pmb k_i)=\frac{\exp(f(\pmb q,\pmb k_i))}{\sum_{j}\exp(f(\pmb q,\pmb k_j))}
    $$

    输出的计算方法为（被定义为注意力（attention））：

    $$
    {\rm Attention}(\pmb q,\{\pmb k,\pmb v\})=\sum_{i=1}^m\alpha(\pmb q,\pmb k_i)\pmb v_i
    $$

* 缩放点积注意力（scaled dot-product attention）的打分函数为：

    $$
    f(\pmb q,\pmb k_i)=\pmb q^{\rm T}\pmb k_i/\sqrt{d}
    $$

    其中 $\pmb q\in\mathbb{R}^d,\pmb k_i\in\mathbb{R}^d$。假定 $\pmb q$ 和 $\pmb k_i$ 的每个元素都服从期望为 0、方差为 1 的独立同分布，那么点积每一项的期望为 0，方差为 1，点积本身期望为 0，方差为 $d$。为使点积的方差保持为 1（而与向量长度 $d$ 无关），这里取缩放系数 $1/\sqrt{d}$。点积如有较大方差，会导致 softmax 操作中各指数项的大小差距过大，从而梯度过小。

* 双线性注意力的打分函数为：

    $$
    f(\pmb q,\pmb k_i)=\pmb q^{\rm T}W\pmb k_i
    $$

* 加性注意力（[Bahdanau，2014](https://arxiv.org/abs/1409.0473)）的打分函数为：

    $$
    f(\pmb q,\pmb k_i)=\pmb w_v^{\rm T}\tanh(W_q\pmb q+W_k\pmb k_i)
    $$

#### 可视化

### 扩展

#### 引导注意力机制（Guided Attention Mechanism）

在一些任务中，如语音识别、TTS 等，输入和输出是单调对齐的（monotonically aligned），因此可以人为设定注意力的位置和移动规则。

**论文**

#### 自注意力机制（Self-Attention Mechanism）

参阅[自注意力机制](./transformer.md#自注意力机制self-attention-mechanism)。

### 讨论
