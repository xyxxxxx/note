在前馈神经网络中，信息的传递是单向的，这种限制虽然使得网络变得更容易学习，但在一定程度上也减弱了神经网络模型的能力。在生物神经网络中神经元之间的连接关系要复杂得多。前馈神经网络可以看作一个复杂的函数，每次输入都是独立的，即网络的输出只依赖于当前的输入。但是在很多现实任务中，网络的输出不仅和当前时刻的输入相关，也和其过去一段时间的输出相关，比如一个有穷自动机，其下一个时刻的状态(输出)不仅仅和当前输入相关，也和当前状态(上一个时刻的输出)相关。此外，前馈网络难以处理时序数据，比如视频、 语音、 文本等，时序数据的长度一般是不固定的，而前馈神经网络要求输入和输出的维数都是固定的。因此，当处理这一类和时序数据相关的问题时就需要一种能力更强的模型。

循环神经网络(Recurrent Neural Network , RNN)是一类具有短期记忆能力的神经网络。在循环神经网络中, 神经元不但可以接受其他神经元的信息，也可以接受自身的信息，形成具有环路的网络结构。和前馈神经网络相比，循环神经网络更加符合生物神经网络的结构。循环神经网络已经被广泛应用在语音识别、语言模型以及自然语言生成等任务上。循环神经网络的参数学习可以通过随时间反向传播算法 [Werbos, 1990] 来学习，随时间反向传播算法即按照时间的逆序将错误信息一步步地往前传递，但当输入序列比较长时, 会存在梯度爆炸和消失问题 [Bengio et al., 1994; Hochreiter et al., 1997, 2001] ，也称为长程依赖问题。为了解决这个问题，人们对循环神经网络进行了很多的改进，其中最有效的改进方式引入门控机制(gating mechanism)。

此外，循环神经网络可以很容易地扩展到两种更广义的记忆网络模型：递归神经网络和图网络。





# 为网络增加记忆能力

为了处理这些时序数据并利用其历史信息，我们需要让网络具有短期记忆能力。以下三种方法可以用于为网络增加短期记忆能力。



## 延时神经网络

一种简单的利用历史信息的方法是建立一个额外的延时单元，用来存储网络的历史信息(可以包括输入、 输出、 隐状态等)。比较有代表性的模型是**延时神经网络(Time Delay Neural Network , TDNN)** [Lang et al., 1990; Waibel et al.,1989]。

延时神经网络在前馈网络中的每个非输出层的神经元上都添加一个延时器，记录神经元的最近几次活性值。在第$$t$$个时刻，第$$l$$层神经元的活性值依赖于第$$l − 1$$层神经元的最近$$K$$个时刻的活性值，即
$$
\pmb h_t^{(l)}=f(\pmb h_t^{(l-1)},\pmb h_{t-1}^{(l-1)},\cdots,\pmb h_{t-K}^{(l-1)})
$$
其中$$\pmb h_{t}^{(l)}\in \mathbb{R}^{M_l}$$表示第$$l$$层神经元在$$t$$时刻的活性值，$$M_l$$为第$$l$$层的神经元数量。



## 有外部输入的非线性自回归模型

**自回归(Auto Regressive, AR)**模型是统计学上常用的一类时间序列模型，预测变量$$\pmb y_t$$时使用它的历史信息
$$
\pmb y_t=w_0+\sum_{k=1}^Kw_k\pmb y_{t-k}+\varepsilon_t
$$
其中$$K$$为超参数，$$w_0 , ⋯ , w_K$$为可学习参数，$$ε_t ∼ N(0, σ^2 )$$为第 t 个时刻的噪声，方差$$σ^2$$和时间无关。

**有外部输入的非线性自回归模型(Nonlinear Auto Regressive with Exogenous Inputs Model , NARX)** [Leontaritis et al., 1985] 是自回归模型的扩展，在每个时刻$$\pmb t$$都有一个外部输入$$\pmb x_t$$，产生一个输出$$\pmb y_t$$。NARX 通过一个延时器记录最近$$K_x$$次的外部输入和最近$$K_y$$次的输出，第$$t$$个时刻的输出$$\pmb y_t$$为
$$
\pmb y_t=f(\pmb x_t,\pmb x_{t-1},\cdots,\pmb x_{t-K_x},\pmb y_{t-1},\pmb y_{t-2},\cdots,\pmb y_{t-K_y})
$$
其中$$f(⋅)$$表示非线性函数，可以是一个前馈网络，$$K_x$$和$$K_y$$为超参数。



## 循环神经网络

**循环神经网络(Recurrent Neural Network, RNN)**通过使用带自反馈的神经元，能够处理任意长度的时序数据。

给定一个输入序列$$\pmb x_{1:T}=(\pmb x_1,\pmb x_2,\cdots,\pmb x_t,\cdots,\pmb x_T)$$，循环神经网络通过以下公式更新带反馈边的隐藏层的活性值$$\pmb h_t$$：
$$
\pmb h_t=f(\pmb h_{t-1},\pmb x_t)
$$
其中$$\pmb h_0=0$$，$$f(\cdot)$$为一个非线性函数，可以是一个前馈网络。

下图给出了循环神经网络的示例，其中“延时器”为一个虚拟单元，用于记录神经元的最近一次（或几次）活性值。

![](https://i.loli.net/2020/09/22/gQEepNbvfxJu6j7.png)

从数学上讲，公式$$\pmb h_t=f(\pmb h_{t-1},\pmb x_t)$$可以看作一个动力系统，因此隐藏层的活性值$$\pmb h_t$$在很多文献中也称为**状态(state)**或**隐状态(hidden state)**。

由于循环神经网络具有短期记忆能力，相当于存储装置， 因此其计算能力十分强大。理论上，循环神经网络可以近似任意的非线性动力系统。前馈神经网络可以模拟任何连续函数，而循环神经网络可以模拟任何程序。

> **动力系统(dynamical system)**是一个数学上的概念，指系统状态按照一定的规律随时间变化的系统。具体地讲，动力系统是使用一个函数来描述一个给定空间（如某个物理系统的状态空间）中所有点随时间的变化情况。生活中很多现象（比如钟摆晃动、 台球轨迹等）都可以动力系统来描述。





# 简单循环网络

**简单循环网络(Simple Recurrent Network, SRN)** [Elman, 1990] 是一个非常简单的循环神经网络，只有一个隐藏层。

令向量$$\pmb x_t ∈ \mathbb{R}^M$$表示在时刻$$t$$时网络的输入，$$\pmb h_t ∈\mathbb{R}^D$$表示隐藏层状态（即隐藏层神经元活性值），则简单循环网络在时刻$$t$$的更新公式为
$$
\pmb z_t = \pmb U \pmb h_{t-1}+\pmb W\pmb x_t+\pmb b\\
\pmb h_t=f(\pmb z_t)
$$
其中$$\pmb z_t$$为隐藏层的净输入，$$\pmb U\in \mathbb{R}^{D\times D}$$为状态-状态权重矩阵，$$\pmb W\in \mathbb{R}^{D\times M}$$为状态-输入权重矩阵，$$\pmb b\in \mathbb{R}^D$$为偏置向量，$$f(\cdot)$$为非线性激活函数，通常为 Logistic 函数或 Tanh 函数。上式也经常合并为
$$
\pmb h_t = f(\pmb U \pmb h_{t-1}+\pmb W\pmb x_t+\pmb b)
$$
下图给出了按时间展开的循环神经网络

<a name="img1">![](/home/xyx/Pictures/Screenshot from 2020-09-22 18-30-07.png)</a>



**完全连接**的循环神经网络定义为：网络输入$$\pmb x_t$$，输出$$\pmb y_t$$，动力系统为
$$
\pmb h_t = f(\pmb U \pmb h_{t-1}+\pmb W\pmb x_t+\pmb b)\\
\pmb y_t=\pmb V\pmb h_t
$$
其中$$\pmb h$$为状态，$$f(\cdot)$$为非线性激活函数，$$\pmb U,\pmb W,\pmb b,\pmb V$$为网络参数。



一个完全连接的循环网络是任何非线性动力系统的近似器：

**循环神经网络的通用近似定理** [Haykin, 2009] : 如果一个完全连接的循环神经网络有足够数量的 Sigmoid 型隐藏神经元，它可以以任意的准确率去近似任何一个非线性动力系统
$$
\pmb s_t=g(\pmb s_{t-1},\pmb x_t)\\
\pmb y_t=o(\pmb s_t)
$$
其中$$\pmb s_t$$为每个时刻的隐状态，$$\pmb x_t$$是外部输入，$$g(⋅)$$是可测的状态转换函数，$$o(⋅)$$是连续输出函数，并且对状态空间的紧致性没有限制。

> 证明略



一个完全连接的循环神经网络可以近似解决所有的可计算问题：

**循环神经网络的图灵完备定理**[Siegelmann et al., 1991] : 所有的图灵机都可以被一个由使用 Sigmoid 型激活函数的神经元构成的全连接循环网络来进行模拟。

> 证明略
>
> 图灵完备(Turing completeness)是指一种数据操作规则，可以实现图灵机(Turing machine)的所有功能，解决所有的可计算问题。目前主流的编程语言(比如 C++ 、Java 、Python 等)都是图灵完备的。





# 应用到机器学习

## 序列到类别模式

序列到类别模式中，输入为序列，输出为类别，比如在文本分类中，输入数据为单词的序列，输出为该文本的类别。

输入一个长度为$$T$$的序列$$\pmb x_{1∶T}=(\pmb x_1 , ⋯ ,\pmb x_T)$$，输出一个类别$$y ∈ \{1, ⋯ , C\}$$。我们将$$\pmb x_{1∶T}$$按时间顺序输入到循环神经网络中，得到对应时刻的状态序列$$(\pmb h_1 , ⋯ ,\pmb h_T)$$，然后将$$\pmb h_T$$作为输入序列的最终表示（或特征）输入给分类器$$g(⋅)$$进行分类，即
$$
\hat{y}=g(\pmb h_T)
$$
其中$$g(⋅)$$可以是简单的线性分类器（比如 Logistic 回归）或复杂的分类器（比如多层前馈神经网络）。

另一种方法是对状态序列中的所有状态的平均作为输入序列的表示，即
$$
\hat{y}=g(\frac{1}{T}\sum_{t=1}^T\pmb h_t)
$$
![](https://i.loli.net/2020/09/22/cSkCjL2N9Xzb8RV.png)



## 同步的序列到序列模式

同步的序列到序列模式主要用于序列标注(sequence labeling)任务，即每一时刻都有输入和输出，输入序列和输出序列的长度相同，比如在词性标注(part-of-speech tagging)中，每输入一个单词都输出其对应的词性标签。

输入一个长度为$$T$$的序列$$\pmb x_{1∶T}=(\pmb x_1 , ⋯ ,\pmb x_T)$$，输出一个序列$$y_{1∶T}=(y_1 , ⋯ ,y_T)$$。将每个状态$$\pmb h_t$$输入给分类器$$g(⋅)$$进行分类，即
$$
\hat{y}_t=g(\pmb h_t),\quad t=1,2,\cdots,T
$$
![](https://i.loli.net/2020/09/22/9DRjqSx65gI4JL1.png)



## 异步的序列到序列模式

异步的序列到序列模式也称为编码器 - 解码器(encoder-decoder)模型，即输入序列和输出序列不需要有严格的对应关系，也不需要保持相同的长度，比如在机器翻译中，输入为源语言的单词序列，输出为目标语言的单词序列。

输入一个长度为$$T$$的序列$$\pmb x_{1∶T}=(\pmb x_1 , ⋯ ,\pmb x_T)$$，输出一个长度为$$M$$的序列$$y_{1∶M}=(y_1 , ⋯ , y_M)$$。异步的序列到序列模式一般通过先编码后解码的方式来实现：先将$$\pmb x_{1∶T}$$按时间顺序输入到循环神经网络（编码器）中得到状态$$\pmb h_T$$，然后再使用另一个循环神经网络（解码器）得到输出序列$$\hat y_{1:M}$$。为了建立输出序列之间的依赖关系，在解码器中通常使用非线性的自回归模型。令$$f_1(\cdot)$$和$$f_2(\cdot)$$分别为用作编码器和解码器的循环神经网络，则编码器 - 解码器模型可以写为
$$
\pmb h_t=f_1(\pmb h_{t-1},\pmb x_t),\quad \forall t\in [1,T]\\
\pmb h_{T+t}=f_2(\pmb h_{T+t-1},\hat{\pmb y}_{t-1}),\quad \forall t\in [1,M]\\
\hat{y}_t=g(\pmb h_{T+t}),\quad \forall t\in [1,M]
$$
其中$$g(\cdot)$$为分类器，$$\hat{\pmb y}_t$$是$$\hat y_t$$的向量表示。

下图给出了异步的序列到序列模式示例，其中$$⟨EOS⟩$$表示输入序列的结束，虚线表示将上一个时刻的输出作为下一个时刻的输入。

![](https://i.loli.net/2020/09/23/uX5kSw9tVdrfUQW.png)





# 参数学习

循环神经网络的参数可以通过梯度下降方法来进行学习。以<u>同步的序列到序列模式</u>和随机梯度下降法为例，给定一个训练样本$$(\pmb x,y)$$，其中$$\pmb x_{1∶T}=(\pmb x_1 , ⋯ ,\pmb x_T),y_{1:T}=(y_1,\cdots,y_T)$$。我们定义$$t$$时刻的损失函数为
$$
\mathcal{L}_t=\mathcal{L}(y_t,g(\pmb h_t))
$$
其中$$g(\pmb h_t)$$为$$t$$时刻的输出，$$\mathcal{L}$$为可微的损失函数。那么整个序列的损失函数为
$$
\mathcal{L}=\sum_{t=1}^T\mathcal{L}_t
$$
整个序列的损失函数$$\mathcal{L}$$关于参数$$\pmb U$$的梯度为
$$
\frac{\partial \mathcal{L}}{\partial \pmb U}=\sum_{t=1}^T\frac{\partial \mathcal{L}_t}{\partial \pmb U}
$$
即每个时刻的损失$$\mathcal{L}_t$$对参数$$\pmb U$$的偏导数之和。



循环神经网络中存在一个递归调用的函数$$f(⋅)$$，因此其计算参数梯度的方式和前馈神经网络不太相同。在循环神经网络中主要有两种计算梯度的方式：随时间反向传播(BPTT)算法和实时循环学习(RTRL)算法。



## 随时间反向传播算法

**随时间反向传播(BackPropagation Through Time , BPTT)**算法的主要思想是通过类似前馈神经网络的错误反向传播算法 [Werbos, 1990] 来计算梯度。

BPTT 算法将循环神经网络看作一个展开的多层前馈网络，其中 “每一层” 对应循环网络中的 “每个时刻“([如图](#img1))，这样循环神经网络就可以按照前馈网络中的反向传播算法计算参数梯度。？在 “展开” 的前馈网络中，所有层的参数是共享的，因此参数的真实梯度是所有 “展开层” 的参数梯度之和。

**计算偏导数$$\frac{\partial \mathcal{L}_t}{\partial \pmb U}$$**
$$
\frac{\partial \mathcal{L}_t}{\partial u_{ij}}=\frac{\partial \pmb z_{t}}{\partial u_{ij}}\frac{\partial \mathcal{L}_t}{\partial\pmb z_{t}}=(\pmb h_{t-1}^{\rm T}\frac{\partial U}{\partial u_{ij}}+\frac{\partial\pmb  h_{t-1}}{\partial u_{ij}}U^{\rm T})\frac{\partial \mathcal{L}_t}{\partial\pmb  z_{t}}\\
=(\mathbb{I}_i(\pmb h_{t-1,j})+\frac{\partial\pmb  z_{t-1}}{\partial u_{ij}}\frac{\partial \pmb h_{t-1}}{\partial\pmb  z_{t-1}}\frac{\partial\pmb  z_{t}}{\partial\pmb  h_{t-1}})\frac{\partial \mathcal{L}_t}{\partial\pmb  z_{t}}\\
=\mathbb{I}_i(\pmb h_{t-1,j})\frac{\partial \mathcal{L}_t}{\partial \pmb z_{t}}+\frac{\partial\pmb  z_{t-1}}{\partial u_{ij}}\frac{\partial \mathcal{L}_t}{\partial\pmb  z_{t-1}}\\
= \cdots\\
=\sum_{k=1}^t \mathbb{I}_i((\pmb h_{k-1})_j)\frac{\partial \mathcal{L}_t}{\partial \pmb z_{k}}\\
\triangleq \sum_{k=1}^t \mathbb{I}_i((\pmb h_{k-1})_j)\pmb \delta_{t,k}\\
=\sum_{k=1}^t (\pmb \delta_{t,k})_i(\pmb h_{k-1})_j
$$
其中$$(\pmb h_{k-1})_j$$为$$k-1$$时刻的状态的第$$j$$维，$$\mathbb{I}_i(x)$$表示第$$i$$个值为$$x$$，其它值为0的行向量，$$\pmb \delta_{t,k}=\frac{\partial \mathcal{L}_t}{\partial \pmb z_{k}}$$为定义的误差项，根据上面的计算，有
$$
\pmb \delta_{t,k}=\frac{\partial \pmb h_{k}}{\partial\pmb  z_{k}}\frac{\partial\pmb  z_{k+1}}{\partial\pmb  h_{k}})\delta_{t,k+1}\\
={\rm diag}(f'(\pmb z_k))U^{\rm T}\pmb \delta_{t,k+1}
$$
即为误差项反向传播的表达式。

将$$\frac{\partial \mathcal{L}_t}{\partial u_{ij}}$$写成矩阵形式，即
$$
\frac{\partial \mathcal{L}_t}{\partial U}=\sum_{k=1}^t \pmb \delta_{t,k}\pmb h^{\rm T}_{k-1}
$$

下图给出了误差项随时间反向传播的示例

![](https://i.loli.net/2020/09/23/6FIYRaW4OncSlri.png)

**计算梯度$$\frac{\partial \mathcal{L}}{\partial \pmb U}$$**
$$
\frac{\partial \mathcal{L}}{\partial \pmb U}=\sum_{t=1}^T\frac{\partial \mathcal{L}_t}{\partial \pmb U}=\sum_{t=1}^T\sum_{k=1}^t \pmb \delta_{t,k}\pmb h^{\rm T}_{k-1}
$$
同理可得，$$\mathcal{L}$$关于权重$$\pmb W$$和偏置$$\pmb b$$的梯度为
$$
\frac{\partial \mathcal{L}}{\partial \pmb W}=\sum_{t=1}^T\sum_{k=1}^t \pmb \delta_{t,k}\pmb x^{\rm T}_{k}\\
\frac{\partial \mathcal{L}}{\partial \pmb b}=\sum_{t=1}^T\sum_{k=1}^t \pmb \delta_{t,k}
$$


## 实时循环学习算法

与反向传播的 BPTT 算法不同，**实时循环学习(Real-Time Recurrent Learning , RTRL)**通过前向传播的方式来计算梯度 [Williams et al., 1995] 。

假设循环神经网络中$$t + 1$$时刻的状态$$\pmb h_{t+1}$$为
$$
\pmb h_{t+1} = f(\pmb z_{t+1}) = f(\pmb U \pmb h_{t}+\pmb W\pmb x_{t+1}+\pmb b)
$$
其关于参数$$u_{ij}$$的偏导数为
$$
\frac{\partial \pmb h_{t+1}}{\partial u_{ij}}=(\pmb h_{t}^{\rm T}\frac{\partial U}{\partial u_{ij}}+\frac{\partial\pmb  h_{t}}{\partial u_{ij}}U^{\rm T})\frac{\partial \pmb h_{t+1}}{\partial\pmb z_{t+1}}\\
=(\mathbb{I}_i((\pmb h_t)_j)+\frac{\partial\pmb  h_{t}}{\partial u_{ij}}U^{\rm T}){\rm diag}(f'(\pmb z_{t+1}))\\
=(\mathbb{I}_i((\pmb h_t)_j)+\frac{\partial\pmb  h_{t}}{\partial u_{ij}}U^{\rm T})\odot (f'(\pmb z_{t+1}))^{\rm T}
$$
RTRL 算法从1时刻开始，除了计算循环神经网络的状态之外，还利用上式依次前向计算偏导数$$\frac{\partial \pmb h_{1}}{\partial u_{ij}},\frac{\partial \pmb h_{2}}{\partial u_{ij}},\cdots$$

这样，假设第$$t$$个时刻存在一个输出$$\hat y_t$$，其损失函数为$$\mathcal{L}_t$$，那么就可以在计算损失函数的同时计算其对$$u_{ij}$$的偏导数
$$
\frac{\partial \mathcal{L}_t}{\partial u_{ij}}=\frac{\partial \pmb h_t}{\partial u_{ij}}\frac{\partial \mathcal{L}_t}{\partial \pmb h_t}
$$
这样就可以实时计算$$\mathcal{L}_t$$关于参数的梯度并更新参数。



RTRL 算法和 BPTT 算法都是基于梯度下降的算法，分别通过前向模式和反向模式应用链式法则来计算梯度。在循环神经网络中，一般网络输出维度远低于输入维度，因此 BPTT 算法的计算量会更小，但是 BPTT 算法需要保存所有时刻的中间梯度，空间复杂度较高。RTRL 算法不需要梯度回传，因此非常适合用于需要在线学习或无限序列的任务中。

> RTRL 和 BPTT 类似于算法的迭代和递归。





# 长程依赖问题

循环神经网络在学习过程中的主要问题是由于梯度消失或爆炸问题，而难以建模长时间间隔的状态之间的依赖关系。

在 BPTT 算法中，将以下公式迭代若干次
$$
\pmb \delta_{t,k}
={\rm diag}(f'(\pmb z_k))U^{\rm T}\pmb \delta_{t,k+1}=\cdots=\prod_{\tau=k}^{t-1}({\rm diag}(f'(\pmb z_{\tau}))U^{\rm T})\pmb \delta_{t,t}
$$
假设$${\rm diag}(f'(\pmb z_\tau))U^{\rm T}\in \mathbb{R}^{D\times D},\ \tau=k,\cdots,t-1$$彼此近似相等，记为$$A$$，再假设$$A$$的特征值的绝对值均大于1且特征方程有$$D$$个互不相等的根，那么当n较大时，$$A^n=P{\rm diag}(\lambda_1^n,\lambda_2^n,\cdots,\lambda_D^n)P^{-1}$$中的元素（绝对值）很大，导致$$\pmb \delta_{t,k}$$的元素很大，进而导致梯度的元素很大，造成系统不稳定，称为**梯度爆炸问题(gradient exploding problem)**。

相反，若$$A$$的特征值的绝对值均小于1，则会导致梯度的元素很小，会出现和深层前馈神经网络类似的**梯度消失问题(vanishing gradient problem)**。

由于循环神经网络经常使用非线性激活函数为 Logistic 函数或 Tanh 函数作为非线性激活函数，其导数值都小于 1 ，并且权重矩阵$$U$$的元素也不会太大，因此如果时间间隔$$t − k$$过大，$$\pmb δ_{t,k}$$会趋向于$$\pmb0$$，因而经常会出现梯度消失问题。



虽然简单循环网络理论上可以建立长时间间隔的状态之间的依赖关系，但是由于梯度爆炸或消失问题，实际上只能学习到短期的依赖关系。简单循环网络难以建模长距离的依赖关系，称为**长程依赖问题(long-term dependencies problem)**。

> 参照梯度计算公式
> $$
> \frac{\partial \mathcal{L}}{\partial \pmb U}=\sum_{t=1}^T\sum_{k=1}^t \pmb \delta_{t,k}\pmb h^{\rm T}_{k-1}
> $$
> 梯度爆炸问题可以理解为很久之前的状态占据了绝大部分权重，而梯度消失问题可以理解为很久之前的状态几乎没有权重。



## 改进方案

为了避免梯度爆炸或消失问题，一种最直接的方式就是选取合适的参数，同时使用非饱和的激活函数，尽量使得$${\rm diag}(f'(\pmb z_k))U^{\rm T}$$接近正交矩阵，但这种方式需要足够的人工调参经验，限制了模型的广泛应用。更佳有效的方法是通过改进模型或优化方法来缓解循环网络的梯度爆炸和梯度消失问题。

**梯度爆炸** 一般而言，循环网络的梯度爆炸问题比较容易解决，一般通过权重衰减或梯度截断来避免。

权重衰减是通过给参数增加$$l_1$$或$$l_2$$范数的正则化项来限制参数的取值范围，从而使各特征值$$\lambda_i \le 1$$。梯度截断则是当梯度的模大于一定阈值时，就将它截断成为一个较小的数。

**梯度消失** 梯度消失是循环网络的主要问题。除了使用一些优化技巧外，更有效的方式是改变模型，简单循环网络的模型为
$$
\pmb h_t = f(\pmb U \pmb h_{t-1}+\pmb W\pmb x_t+\pmb b)
$$
令$$U=I,\ \frac{\partial \pmb h_t}{\partial \pmb h_{t-1}}=I$$，得到
$$
\pmb h_t=\pmb h_{t-1}+g(\pmb x_t;\theta)
$$
其中$$g(\cdot)$$是一个非线性函数，$$\pmb \theta$$为参数。

上式中$$\pmb h_t$$和$$\pmb h_{t-1}$$之间为线性关系，且系数为1，这样就不存在梯度爆炸或消失问题；然而与此同时也丢失了部分非线性激活的性质，因而降低了模型的表示能力。为了避免这一缺点，我们进一步改进为
$$
\pmb h_t=\pmb h_{t-1}+g(\pmb x_t,\pmb h_{t-1};\theta)
$$
上式中$$\pmb h_t$$和$$\pmb h_{t-1}$$之间既有线性关系又有非线性关系，可以缓解梯度消失问题；然而依旧存在两个问题：

+ 梯度爆炸问题
+ **记忆容量(memory capacity)**问题：假设$$g(\cdot)\in (0,1)$$为 Logistic 函数，则随着时间$$t$$的增长$$\pmb h_t$$会累积得越来越大，从而导致$$\pmb h$$变得饱和。换言之，状态$$\pmb h$$可以存储的信息受到限制，如状态不能回滚。





# 基于门控的循环神经网络

为了改善循环神经网络的长程依赖问题，一种非常好的解决方案是在模型$$\pmb h_t=\pmb h_{t-1}+g(\pmb x_t,\pmb h_{t-1};\theta)$$的基础上引入门控机制来控制信息的累积速度，包括有选择地加入新的信息，并有选择地遗忘之前累积的信息。这一类网络可以称为**基于门控的循环神经网络(Gated RNN)**。本节中，主要介绍两种基于门控的循环神经网络：长短期记忆网络和门控循环单元网络。



## 长短期记忆网络

**长短期记忆网络(Long Short-Term Memory Network, LSTM)** [Gers et al.,2000; Hochreiter et al., 1997] 是循环神经网络的一个变体，可以有效地解决简单循环神经网络的梯度爆炸或消失问题。

LSTM 基于以下模型进行两点改进：
$$
\pmb h_t=\pmb h_{t-1}+g(\pmb x_t,\pmb h_{t-1};\theta)
$$
**内部状态** LSTM网络引入一个新的**内部状态(internal state)** $$\pmb c_t\in  \mathbb{R}^D$$专门进行线性的循环信息传递，同时非线性地输出信息给隐藏层的外部状态$$\pmb h_t\in \mathbb{R}^D$$。内部状态$$\pmb c_t$$通过下面的公式计算：
$$
\pmb c_t=\pmb f_t\odot \pmb c_{t-1}+\pmb i_t\odot \tilde{\pmb c}_t\\
\pmb h_t=\pmb o_t\odot \tanh(\pmb c_t)
$$
其中$$\pmb f_t,\pmb i_t,\pmb o_t\in\{0,1\}^D$$是三个门(gate)，用于控制信息传递的路径；$$\odot$$为逐元素乘法；$$\pmb c_{t}$$为$$t$$时刻的记忆单元；$$\tilde {\pmb c}_{t}\in \mathbb{R}^D$$是通过非线性函数得到的**候选状态**
$$
\tilde{\pmb c}_t=\tanh(\pmb W_c\pmb x_t+\pmb U_c\pmb h_{t-1}+\pmb b_c)
$$
在每个时刻$$t$$，LSTM 网络的内部状态$$\pmb c_t$$记录了到当前时刻为止的历史信息。

**门控机制** 在数字电路中，门(gate)为一个二值变量 {0, 1}， 0 代表关闭状态，不许信息通过；1 代表开放状态，允许信息通过。

LSTM 网络引入**门控机制(gating mechanism)**来控制信息传递的路径，三个门的作用分别为

1. 遗忘门$$\pmb f_t$$控制上一个时刻的内部状态$$\pmb c_{t-1}$$需要遗忘多少信息
2. 输入门$$\pmb i_t$$控制当前时刻的候选状态$$\tilde{\pmb c}_t$$有多少信息需要保存
3. 输出门$$\pmb o_t$$控制当前时刻的内部状态$$\pmb c_t$$有多少信息需要输出给外部状态$$\pmb h_t$$

当$$\pmb f_t=\pmb 0,\pmb i_t=\pmb 1$$时，记忆单元将历史信息清空，并将候选状态向量$$\tilde{\pmb c}_t$$写入。但此时记忆单元$$\pmb c_t$$依然和上一时刻的历史信息相关。当$$\pmb f_t=\pmb 1,\pmb i_t=\pmb 0$$时，记忆单元将复制上一时刻的内容，不写入新的信息。

下图给出了 LSTM 网络的循环单元结构，其计算过程为：

1. 利用上一时刻的外部状态$$\pmb h_{t−1}$$和当前时刻的输入$$\pmb x_t$$计算出三个门，以及候选状态$$\tilde{\pmb c}_t$$ 
2. 结合遗忘门$$\pmb f_t$$和输入门$$\pmb i_t$$来更新记忆单元$$\pmb c_t$$
3. 结合输出门$$\pmb o_t$$，将内部状态的信息传递给外部状态$$\pmb h_t$$

![](https://i.loli.net/2020/09/23/ySW7f9hqeHUi1YO.png)

通过 LSTM 循环单元，整个网络可以建立较长距离的时序依赖关系。模型可以简洁地描述为
$$
\begin{bmatrix}
\tilde{\pmb c}_t\\
\pmb o_t\\
\pmb i_t\\
\pmb f_t
\end{bmatrix}=\begin{bmatrix}
\tanh\\
\sigma\\
\sigma\\
\sigma
\end{bmatrix}\begin{pmatrix}W\begin{bmatrix}
\pmb x_t\\
\pmb h_{t-1}
\end{bmatrix}+\pmb b
\end{pmatrix}\\
\pmb c_t=\pmb f_t\odot \pmb c_{t-1}+ \pmb i_t\odot \tilde{\pmb c}_t\\
\pmb h_{t}=\pmb o_t\odot \tanh(\pmb c_t)
$$
其中$$\pmb x_t\in \mathbb{R}^M$$为当前时刻的输入，$$\pmb W\in \mathbb{R}^{4D\times (D+M)}$$和$$\pmb b\in \mathbb{R}^{4D}$$为网络参数。

下图给出了 LSTM 网络的结构，相当于在$$\pmb x_t$$和$$\pmb h_t$$之间增加了一层：

![](https://img-blog.csdn.net/20180903162759532?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L20wXzM3OTE3Mjcx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)



循环神经网络中的状态$$h$$存储了历史信息，可以看作一种**记忆(memory)**。在简单循环网络中，状态每个时刻都会被重写，因此可以看作一种**短期记忆(short-term memory)**。在神经网络中，**长期记忆(long-term memory)**可以看作网络参数，隐含了从训练数据中学到的经验，其更新周期要远远慢于短期记忆。而在 LSTM 网络中，记忆单元$$\pmb c$$可以在某个时刻捕捉到某个关键信息，并有能力将此关键信息保存一定的时间间隔。记忆单元$$\pmb c$$中保存信息的生命周期要长于短期记忆$$\pmb h$$，但又远远短于长期记忆，因此称为**长短期记忆(long short-term memory)**。





## 门控循环单元网络

**门控循环单元(Gated Recurrent Unit, GRU)**网络 [Cho et al., 2014; Chung et al., 2014] 是一种比 LSTM 网络更加简单的循环神经网络。







