# Transformer

## 参考

* [【機器學習2021】自注意力機制 (Self-attention) (上)](https://www.youtube.com/watch?v=hYdO9CscNes&list=PLJV_el3uVTsMhtt7_Y6sgTHGHp1Vb2P2J&index=10)
* [【機器學習2021】自注意力機制 (Self-attention) (下)](https://www.youtube.com/watch?v=gmsMY5kc-zw&list=PLJV_el3uVTsMhtt7_Y6sgTHGHp1Vb2P2J&index=11)
* [【機器學習2021】Transformer (下)](https://www.youtube.com/watch?v=gmsMY5kc-zw&list=PLJV_el3uVTsMhtt7_Y6sgTHGHp1Vb2P2J&index=13)
* [Transformer论文逐段精读【论文精读】](https://www.bilibili.com/video/BV1pu411o7BE/)

## 论文

* [Attention Is All You Need](https://arxiv.org/abs/1706.03762)：
    * 提出新的基础网络架构 Transformer，仅基于自注意力机制，并行度好

## 细节

### 自注意力（self-attention）机制

自注意力层输入一个向量序列（长度不定），输出一个等长的向量序列。它通过计算每个向量与所有向量的注意力分数（关联度），来构建这一向量新的表示，如下图所示：

![](https://s2.loli.net/2023/02/05/MAZgEsKcuDXo8QW.png)

自注意力层可以并行计算，如下图所示：

![](https://s2.loli.net/2023/02/05/7z6PonJA4DUs9Ny.png)

![](https://s2.loli.net/2023/02/05/OlHMd69Rx3hDLGS.png)

![](https://s2.loli.net/2023/02/05/w8y7vAocb2XsLfQ.png)

自注意力层的扩展：

* 引入多组 $W^q,W^k,W^v$，分组计算，最后拼接结果再作线性变换，称为多头自注意力（multi-head self-attention）。
* 引入位置编码，加到输入向量中，以补充位置信息。位置编码是人为设定的。
* 若序列过长，仅计算每个向量与给定距离内所有向量的注意力分数，以限制注意力矩阵的计算复杂度，称为截断自注意力（truncated self-attention）。（需要问题本身具有依赖距离较短的性质，例如语音识别。）

相比 CNN，自注意力层：

* 具有动态计算的感受野和权重（注意力分数）

[On the Relationship between Self-Attention and Convolutional Layers](https://arxiv.org/abs/1911.03584) 认为，CNN 就是一种受限制的自注意层，自注意层就是更加灵活的 CNN；若为自注意层设定合适的条件和参数，则可以完全实现 CNN 的功能。 

相比 RNN，自注意力层：

* 可以并行计算
* 没有长程依赖问题
* 需要额外进行位置编码

### Transformer 模型架构

模型架构图如下：

![](https://s2.loli.net/2023/02/05/sWcST7DgrbIfUiw.png)

其中：

* 多头注意力下的三个箭头分别为 $k,v,q$，前两个向量来自编码器，后一个向量来自解码器，称为跨注意力（cross-attention）机制。
* “masked”多头注意力指仅计算每个向量与其左侧所有向量（包括其自身）的注意力分数，这样每个向量的输出表示都与“仅有该向量与其左侧所有向量输入自注意力层”情况下该向量的输出表示相同，因此在（teacher forcing）训练过程中解码器只需要一趟计算，就可以对最终输出的所有向量计算交叉熵损失。

编码器的每个 block 的计算流程如下：

![](https://s2.loli.net/2023/02/05/3MZIPGKck1boChv.png)

解码器的每个 block 的跨注意力机制的计算流程如下：

![](https://s2.loli.net/2023/02/06/9LDtVEjfqSxKY7N.png)

可以理解为解码器产生一个查询向量，向编码器的输出向量进行查询并抽取信息，最后综合得到当前位置的输出向量表示。

### 训练

teacher forcing 训练使用 ground truth 作为输入，如下图所示：

![](https://s2.loli.net/2023/02/06/jg9Z1YWraPehTiU.png)

适当地在输入的 ground truth 中增加一些噪音，可以提高模型的自我纠错能力以及健壮性。

## 扩展

### 自回归解码器和非自回归解码器

传统解码器采用的 RNN（LSTM 等）层只有自回归计算这一种选择，Transformer 模型解码器采用的自注意力层为非自回归计算提供了条件。

![](https://s2.loli.net/2023/02/06/vy4rHlDAhE1Cwtb.png)

自回归解码器和非自回归解码器的比较如上图所示，其中：

* “Another predictor” 指训练另一个模型，其输入编码器的输入数据，输出解码器的输出长度。
* “controllable output length” 指可以控制输出长度。例如对于 TTS 模型，减小输出长度可以让生成的语音语速更快；对于机器翻译模型，减小输出长度可以让翻译出的文本更简略。

### Transformer 的变体

* [Reformer: The Efficient Transformer](https://arxiv.org/abs/2001.04451)
* [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150)
