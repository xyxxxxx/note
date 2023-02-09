# 序列到序列模型

## 参考

* [【機器學習2021】Transformer (上)](https://www.youtube.com/watch?v=gmsMY5kc-zw&list=PLJV_el3uVTsMhtt7_Y6sgTHGHp1Vb2P2J&index=12)
* [《神经网络与深度学习》](https://nndl.github.io/) 15.6 P375, 8.2 P194

## 项目

* [基于注意力的机器翻译](https://github.com/t9k/sample-docs/blob/master/docs/text/attention-based-machine-translation.md)

## 讨论

NLP 的许多任务（阅读理解、机器翻译、文章摘要、情感分析等等）都可以在形式上转换为问答/对话，从而使用 seq2seq 模型进行训练。这一模型在各种任务上的表现都不如这些任务各自的最适模型，但优势在于泛用（可以参照 ChatGPT 在多种 NLP 任务上的表现）。

例如，[Grammar as a Foreign Language](https://arxiv.org/abs/1412.7449) 将语法解析的结果用序列表示，从而可以使用 seq2seq 模型训练，达到当时的 state-of-the-art 的结果。

## 编码器-解码器结构

### 论文

* [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)
* [Listen, Attend and Spell](https://arxiv.org/abs/1508.01211)

### 讨论

编码器-解码器是深度学习模型的经典结构之一，许多模型，从进行机器翻译的 LSTM 模型到 Transform 模型，都属于编码器-解码器结构。

## 注意力机制

### 扩展

#### 引导注意力（guided attention）机制

在一些任务中，如语音识别、TTS 等，输入和输出是单调对齐的（monotonically aligned），因此可以人为设定注意力的位置和移动规则。

**论文**

#### 束搜索

解码器在每一个时间步都选择最有可能的 token（贪心搜索），最终得到的不一定是最有可能的序列。确保得到最有可能的序列需要穷举搜索，但其指数时间复杂度无法承受。束搜索是一种折中方法，其在每一个时间步选取常数个最有可能的序列。

束搜索并非对于所有任务都适用，因为其牺牲计算效率是为了换取优化效果。一般来说，其适用于有标准答案的任务（例如语音识别等），而不适用于生成类的、创造性的任务（例如文本生成等）。

**参考**

* [Beam Search - Dive into Deep Learning](https://d2l.ai/chapter_recurrent-modern/beam-search.html)

**论文**

* [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751)
    * 指出当前解码策略（取最大概率、随机采样等）存在的问题，并提出新的采样方法——Nucleus Sampling

#### Scheduled Sampling

**论文**

* [Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks](https://arxiv.org/abs/1506.03099)
