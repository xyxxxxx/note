











# 序列到序列模型

在序列生成任务中，有一类任务是序列到序列生成任务，即输入一个序列，生成另一个序列，比如机器翻译、语音识别、文本摘要、对话系统、图像标题生成等。

**序列到序列(Sequence-to-Sequence , Seq2Seq)**是一种条件的序列生成问题，给定一个序列$$\pmb x_{1∶S}$$，生成另一个序列$$\pmb y_{1∶T}$$。输入序列的长度$$S$$和输出序列的长度$$T$$可以不同，比如在机器翻译中，输入为源语言，输出为目标语言。下图给出了基于循环神经网络的序列到序列机器翻译示例，其中$$⟨EOS⟩$$表示输入序列的结束，虚线表示用上一步的输出作为下一步的输入。

![](https://i.loli.net/2020/11/10/jHZJGxo6Wmc7aSn.png)

序列到序列模型的目标是估计条件概率
$$
p_\theta(\pmb y_{1∶T}|\pmb x_{1∶S})=\prod_{t=1}^T p_{\theta}(\pmb y_t|\pmb y_{1∶(t-1)},\pmb x_{1∶S})
$$
其中$$\pmb y_t ∈\mathcal{V}$$为词表$$\mathcal{V}$$中的某个词。

给定一组训练数据$$\{(\pmb x_{S_n}, \pmb y_{T_n})\}^N_{n=1}$$，我们可以使用最大似然估计来训练模型参数
$$
\hat{\theta}=\arg\max_\theta\sum_{n=1}^N\log p_\theta(\pmb y_{1:T_n}|\pmb x_{1:S_n})
$$
一旦训练完成，模型就可以根据一个输入序列$$x$$来生成最可能的目标序列
$$
\hat{\pmb y} =\arg\max_y p_{\hat{\theta}}(\pmb y|\pmb x)
$$
具体的生成过程可以通过贪婪方法或束搜索来完成。

和一般的序列生成模型类似，条件概率$$p_θ (y_t|\pmb y_{1∶(t−1)},\pmb x_{1∶S})$$可以使用各种不同的神经网络来实现。这里我们介绍三种主要的序列到序列模型：基于循环神经网络的序列到序列模型、基于注意力的序列到序列模型、基于自注意力的序列到序列模型。



## 基于循环神经网络的序列到序列模型

实现序列到序列的最直接方法是使用两个循环神经网络来分别进行编码和解码，也称为**编码器 - 解码器(encoder-decoder)**模型。

**编码器**

首先使用一个循环神经网络$$f_{\rm enc}$$来编码输入序列$$\pmb x_{1∶S}$$得到一个固定维数的向量$$\pmb u$$，$$\pmb u$$一般为<u>编码循环神经网络最后时刻的隐状态</u>。
$$
\overline{\pmb h}_s=f_{\rm enc}(\overline{\pmb h}_s-1,\pmb x_{s-1},\theta_{\rm enc}),\ s=1,\cdots,S\\
\pmb c=\overline{\pmb h}_S
$$
其中$$f_{\rm enc}(⋅)$$为<u>编码循环神经网络</u>，可以是 LSTM 或 GRU ，其参数为$$\theta_{\rm enc}$$，$$\pmb x_{s-1}$$为词$$x$$的词向量；$$\pmb c$$称为上下文向量(context vector)。

**解码器**

在生成目标序列时，使用另外一个循环神经网络$$f_{\rm dec}$$来进行解码。在解码过程的第$$t$$步时，已生成前缀序列为$$\pmb y_{1:T_n}$$ . 令$$\overline{\pmb h}_t$$表示在网络$$f_{\rm dec}$$的隐状态，$$\pmb o_t ∈ (0, 1)^{|\mathcal{V}|}$$为词表中所有词的后验概率，则
$$
\pmb h_0=\pmb c=\overline{\pmb h}_S\\
\pmb h_t=f_{\rm dec}(\pmb h_{t-1},\pmb y_{t-1},\theta_{\rm dec})\\
\pmb o_t=g(\overline{\pmb h}_t,\theta_o),\ t=1,\cdots,T
$$
其中$$f_{\rm dec}(\cdot)$$为<u>解码循环神经网络</u>，$$g(⋅)$$为最后一层为 Softmax 函数的前馈神经网络，$$\theta_{\rm dec}$$和$$θ_o$$为网络参数，$$\pmb y_{t-1}$$为词$$y$$的词向量，$$\pmb y_0$$为一个特殊符号，比如$$⟨EOS⟩$$。

基于循环神经网络的序列到序列模型的缺点是：(1) 编码向量$$\pmb u$$的容量问题，输入序列的信息很难全部保存在一个固定维度的向量中； (2) 当序列很长时， 由于循环神经网络的长程依赖问题，容易丢失输入序列的信息。



## 基于注意力的序列到序列模型

> 推荐阅读：
>
> [常见注意力机制原理介绍与对比](https://blog.csdn.net/linchuhai/article/details/87099930)

为了获取更丰富的输入序列信息，我们可以在每一步中通过注意力机制来从输入序列中选取有用的信息。

在解码过程的第$$t$$步中，先用上一步的隐状态$$\pmb h_{t-1}$$作为查询向量，利用注意力机制从所有输入序列的隐状态$$H_{\rm enc}=[\overline{\pmb h}_1,\cdots,\overline{\pmb h}_S]$$中选择相关信息
$$
\pmb c_t={\rm att}(H_{\rm enc},\pmb h_{t-1})\\
=\sum_{i=1}^S \alpha_{i}\overline{\pmb h}_i \\
=\sum_{i=1}^S {\rm softmax}(s(\overline{\pmb h}_i,\pmb h_{t-1}))\overline{\pmb h}_i\\
$$
其中$$\pmb c_t$$称为上下文向量，$$s(\cdot)$$为注意力打分函数。

然后将从输入序列中选择的信息$$\pmb c_t$$也作为解码器$$f_{\rm dec}(\cdot)$$在第$$t$$步时的输入，得到第$$t$$步的隐状态
$$
\pmb h_t=f_{\rm dec}(\pmb h_{t-1},[\pmb y_{t-1}, \pmb c_t],\theta_{\rm dec})\\
$$
最后将$$\overline{\pmb h}_t$$输入到分类器$$g(\cdot)$$中来预测词表中每个词出现的概率。



### BahdanauAttention

[Bahdanau Attention](https://arxiv.org/pdf/1409.0473.pdf)是由Bahdanau等人在2015年提出来的一种注意力机制，其结构仍然采用encoder-decoder形式，如下图所示。对于编码器，Bahdanau Attention采用的是双向循环神经网络结构，其中RNN采用GRU单元。

![](https://i.loli.net/2020/11/11/CWIYdh8l5ZH3Awi.png)

首先定义条件概率
$$
p(y_t|y_{1∶(t-1)},x_{1∶S})=g(y_{i-1},s_i,c_i)
$$
其中$$s_i$$是解码循环神经网络的隐状态，由下式计算
$$
s_i=f(s_{i-1},y_{i-1},c_i)
$$
上下文向量$$c_i$$由编码循环神经网络的隐状态加权求和得到
$$
c_i=\sum_{j=1}^T\alpha_{ij}h_j\\
=\sum_{j=1}^T{\rm softmax}({\rm score}(s_{i-1},h_j))h_j
$$
其中$$h_j$$由双向循环神经网络分别计算的隐状态拼接得到，即$$h_j=\vec{h_j}\oplus\overleftarrow{h_j}$$。



### LuongAttention

[Luong Attention](https://arxiv.org/pdf/1508.04025v5.pdf) 也是在2015年由Luong提出来的一种注意力机制。Luong在论文中提出了两种类型的注意力机制：一种是全局注意力模型，即每次计算上下文向量时都考虑输入序列的所有隐藏状态；另一种是局部注意力模型，即每次计算上下文向量时只考虑输入序列隐藏状态中的一个子集。

Luong Attention的模型结构也是采用encoder-decoder的形式，只是在encoder和decoder中，均采用多层LSTM的形式，如下图所示。对于全局注意力模型和局部注意力模型，在计算上下文向量时，均使用encoder和decoder最顶层的LSTM的隐藏状态。

![](https://i.loli.net/2020/11/11/WBnJPoQX8Tmdy9V.png)

**全局注意力模型**

全局注意力模型在计算decoder的每个时间步的上下文向量$$\pmb c_t$$时，均考虑encoder的所有隐藏状态，记每个时间步对应的权重向量为$$\pmb a_t$$，其计算公式如下：
$$
\pmb a_t(s)={\rm align}(\pmb h_t,\overline{\pmb h}_s)\\
={\rm softmax}({\rm score}(\pmb h_t,\overline{\pmb h}_s))
$$
其中，$$\pmb h_t$$表示当前decoder第$$t$$个时间步的隐藏状态，$$\overline{\pmb h}_s$$表示encoder第$$s$$个时间步的隐藏状态，这里与Bahdanau Attention不同的是在计算权重时，采用的是decoder当前时刻的隐藏状态，而不是上一时刻的隐藏状态，即attention是在decoder中LSTM层之后的，而Bahdanau Attention的attention则是在decoder的RNN层之前，换句话说，全局注意力模型的计算路径是$$\pmb h_t\to \pmb a_t\to \pmb c_t\to \tilde{\pmb h}_t$$，而Bahdanau的计算路径是$$\pmb h_{t-1}\to \pmb a_t\to \pmb c_t\to \pmb h_t$$。另外，Luong Attention在计算权重时提供了三种计算方式，并且发现<u>对于全局注意力模型，采用dot的权重计算方式效果要更好</u>：
$$
{\rm score}(\pmb h_t,\overline{\pmb h}_s)=

\left\{ 
    \begin{array}{l}
        \pmb h_t^{\rm T} \overline{\pmb h}_s,\quad\quad {\rm dot} \\ 
        \pmb h_t^{\rm T}W_a \overline{\pmb h}_s,\ \ {\rm general} \\ 
        \pmb v_a^{\rm T}\tanh(W_a[\pmb h_t;\overline{\pmb h}_s])\\
    \end{array}
\right.
$$
其中，concat模式跟Bahdanau Attention的计算方式一致，而dot和general则直接采用矩阵乘积的形式。在计算完权重向量$$\pmb a_t$$后，将其对encoder的隐藏状态进行加权平均得到此刻的上下文向量$$\pmb c_t$$，
$$
\pmb c_t=\sum_{s=1}^S \pmb a_t(s)\overline{\pmb h}_s
$$
然后Luong Attention将其与decoder此刻的隐藏状态$$\pmb h_t$$进行拼接，并通过一个带有tanh的全连接层得到$$\tilde{\pmb h}_t$$：
$$
\tilde{\pmb h}_t=\tanh(W_c[\pmb c_t;\pmb h_t])
$$
最后，将$$\tilde{\pmb h}_t$$传入带有softmax的输出层即可得到此刻目标词汇的概率分布：
$$
p(\pmb y_t|\pmb y_{1∶(t-1)},\pmb x)={\rm softmax}(W_s\tilde{\pmb h}_t)
$$


**局部注意力模型**

然而，全局注意力模型由于在每次decoder时，均考虑encoder所有的隐藏状态，因此其计算成本是非常昂贵的，特别是对于一些长句子或长篇文档，其计算就变得不切实际。因此作者又提出了另一种注意力模式，即局部注意力模型，即每次decoder时不再考虑encoder的全部隐藏状态了，只考虑局部的隐藏状态。

在局部注意力模型中，在decoder的每个时间步$$t$$，需要先确定输入序列中与该时刻对齐的一个位置$$p_t$$，然后以该位置为中心设定一个窗口大小，即$$[p_t-D,p_t+D]$$，其中$$D$$是表示窗口大小的整数，具体的取值需要凭经验设定，作者在论文中设定的是10。接着在计算权重向量时，只考虑encoder中在该窗口内的隐藏状态，当窗口的范围超过输入序列的范围时，则对超出的部分直接舍弃。局部注意力模型的计算逻辑如下图所示。

![](https://i.loli.net/2020/11/11/CUfWaAhH13qvc6m.png)

 在确定位置$$p_t$$时，作者也提出了两种对齐方式，一种是单调对齐，一种是预测对齐，分别定义如下：

+ 单调对齐(local-m)：即直接设定$$p_t=t$$，该对齐方式假设输入序列与输出序列的按时间顺序对齐的，接着计算$$\pmb a_t$$的方式与全局注意力模型相同。

+ 预测对齐(local-p)：预测对齐在每个时间步时会对$$p_t$$进行预测，其计算公式如下：
  $$
  p_t=S\cdot {\rm sigmoid}(\pmb v_p^{\rm T}\tanh(W_p \pmb h_t))
  $$
  其中，$$W_p,\pmb v_p$$为参数，$$S$$为输入序列的长度，这样一来，$$p_t\in [0,S]$$。另外在计算$$\pmb a_t$$时，作者对$$\pmb a_t$$还采用了一个高斯分布进行修正，其计算公式如下：
  $$
  \pmb a_t(s)={\rm align}(\pmb h_t,\overline{\pmb h}_s)\exp(-\frac{(s-p_t)^2}{2\sigma^2})
  $$
  其中，$${\rm align}(\pmb h_t,\overline{\pmb h}_s)$$与全局注意力模型的计算公式相同，$$s$$表示输入序列的位置，$$\sigma=D/2$$。

计算完权重向量后，后面$$\pmb c_t,\ \tilde{\pmb h}_t$$以及概率分布的计算都与全局注意力模型的计算方式相同，这里不再赘述。作者在实验中发现局部注意力模型采用local-p的对齐方式往往效果更好，因为在真实场景中输入序列和输出序列往往没有严格单调对齐的，比如在翻译任务中，往往两种语言在表述同样一种意思时，其语序是不一致的。另外，计算权重向量的方式采用general的方式效果比较好。



**Input-feeding**

在论文中，作者还提及了一个技巧——Input-feeding，即在decoder时，前面的词汇的对齐过程往往对后续词汇的对齐和预测是有帮助的。那么如何将前面的对齐信息传递给后面的预测？作者提出了一个很简单的技巧，即把上一个时刻的$$\tilde{\pmb h}_t$$与下一个时刻的输入进行拼接，一起作为下一个时刻的输入，作者在实验时发现这样一个技巧可以显著提高decoder的效果。但是这样的操作对encoder和decoder第一层的LSTM就有一个要求，假设顶层LSTM的隐藏单元数量都设置为$$n$$，那么，采用Input-feeding后，就要求encoder和decoder第一层的LSTM的隐藏单元数量都为$$2n$$，因为拼接操作会使得输入向量的长度翻倍，如下图所示。

![](https://i.loli.net/2020/11/11/JZr9MzaD5lcVmhS.png)



### 比较

<img src="https://i.stack.imgur.com/yqJpG.png" style="zoom:150%;" />





## 基于自注意力的序列到序列模型

除长程依赖问题外，基于循环神经网络的序列到序列模型的另一个缺点是无法并行计算。为了提高并行计算效率以及捕捉长距离的依赖关系，我们可以使用**自注意力模型(self-attention model)**来建立一个全连接的网络结构. 本节介绍一个目前非常成功的基于自注意力的序列到序列模型：Transformer [Vaswani et al., 2017]。

……