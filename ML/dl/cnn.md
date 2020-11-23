**卷积神经网络(Convolutional Neural Network , CNN or ConvNet)**是一种具有局部连接、权重共享等特性的深层前馈神经网络，最早主要是用来处理图像信息。在用全连接前馈网络来处理图像时，会存在以下两个问题：

1. 参数太多：如果输入图像大小为 100 × 100 × 3（即图像高度为 100 , 宽度为 100 以及 RGB 3 个颜色通道）, 在全连接前馈网络中，第一个隐藏层的每个神经元到输入层都有 100 × 100 × 3 = 30 000 个互相独立的连接，每个连接都对应一个权重参数。随着隐藏层神经元数量的增多，参数的规模也会急剧增加，这会导致整个神经网络的训练效率非常低，也很容易出现过拟合。
2. 局部不变性特征：自然图像中的物体都具有局部不变性特征，比如尺度缩放、平移、旋转等操作不影响其语义信息，而全连接前馈网络很难提取这些局部不变性特征，一般需要进行数据增强来提高性能。

卷积神经网络是受生物学上感受野机制的启发而提出的。**感受野(receptive field)**机制主要是指听觉、视觉等神经系统中一些神经元的特性，即神经元只接受其所支配的刺激区域内的信号。在视觉神经系统中，视网膜上的光感受器受刺激兴奋时将神经冲动信号传到视觉皮层，但只有视觉皮层中特定区域的神经元才会接受这些信号。

目前的卷积神经网络一般是由<u>卷积层、汇聚层和全连接层</u>交叉堆叠而成的前馈神经网络。卷积神经网络有三个结构上的特性：<u>局部连接、权重共享以及汇聚</u>，这些特性使得卷积神经网络具有一定程度上的平移、缩放和旋转不变性，和前馈神经网络相比，卷积神经网络的参数更少。

卷积神经网络主要使用在图像和视频分析的各种任务（比如图像分类、人脸识别、物体识别、图像分割等）上，其准确率一般也远远超出了其他的神经网络模型。近年来卷积神经网络也广泛地应用到自然语言处理、推荐系统等领域。





# 卷积

**卷积(convolution)**是分析数学中一种重要的运算，在信号处理或图像处理中，经常使用一维或二维卷积。



## 一维卷积

一维卷积经常用在信号处理中，用于计算信号的延迟累积。假设一个信号发生器每$$t$$时间产生一个信号$$x_t$$，其信息的衰减率为$$w_k$$，即在$$k$$个时间步长后，信息为原来的$$w_k$$倍。那么在时刻$$t$$收到的信号$$y_t$$为当前时刻产生的信息和以前时刻延迟信息的叠加，
$$
y_t=\sum_{k=0}^{\infty}w_k x_{t-k}
$$
我们把$$w_1 , w_2 , ⋯$$称为**滤波器(filter)**或**卷积核(convolution kernel)**。假设滤波器长度为$$K$$，它和一个信号序列$$x_1 , x_2 , ⋯,x_K$$的卷积为
$$
Y=W*X=\sum_{k=1}^Kw_kx_{K-k+1}
$$
其中$$*$$表示卷积运算。一般情况下滤波器的长度$$K$$远小于信号序列$$X$$的长度。

我们可以设计不同的滤波器来提取信号序列的不同特征。例如令滤波器$$W=[1/K,\cdots,1/K]$$时，卷积相当于信号序列的**简单移动平均(simple moving average)**；

> 移动平均是在分析时间序列数据时的一种简单平滑技术，能有效地消除数据中的随机波动。

令滤波器$$W=[1,-2,1]$$时，可以近似实现对信号序列的二阶微分，即
$$
x''(t)=x(t+1)+x(t-1)-2x(t)
$$

> 泰勒展开即可证明。

下图给出了两个滤波器的一维卷积示例，可以看到两个滤波器分别提取了输入序列的不同特征。滤波器$$w = [1/3, 1/3, 1/3]$$可以检测信号序列中的低频信息，而滤波器$$w = [1, −2, 1]$$可以检测信号序列中的高频信息。

![](https://i.loli.net/2020/09/15/WmaVfL49ctXY62b.png)



## 二维卷积

卷积也经常用在图像处理中，因为图像是一个二维结构，所以需要将一维卷积进行扩展。给定一个图像$$X\in \mathbb{R}^{M\times N}$$和一个滤波器$$W\in \mathbb{R}^{U\times V}$$，一般$$U<<M,V<<N$$，其二维卷积定义为
$$
Y=W* X\\
y_{ij}=\sum_{u=1}^U\sum_{v=1}^V w_{uv}x_{i+U-u,j+V-v}
$$
下图给出了二维卷积示例

![](https://i.loli.net/2020/09/15/t4OxsUq1R2vnNiA.png)

在图像处理中常用的**均值滤波(mean filter)**就是一种二维卷积，将当前位置的像素值设为滤波器窗口中所有像素的平均值，即$$w_{uv}=\frac{1}{UV}$$。

在图像处理中，卷积经常作为特征提取的有效方法。一幅图像在经过卷积操作后得到结果称为**特征映射(feature map)**。下图给出在图像处理中几种常用的滤波器，以及其对应的特征映射。图中最上面的滤波器是常用的高斯滤波器，可以用来对图像进行平滑去噪； 中间和最下面的滤波器可以用来提取边缘特征。

![](https://i.loli.net/2020/09/15/6C51TUiQjodLqwt.png)

![](https://geektutu.com/post/tensorflow2-mnist-cnn/cnn_image_sample.gif)



## 互相关

在机器学习和图像处理领域，卷积的主要功能是在一个图像（或某种特征）上滑动一个卷积核(即滤波器)，通过卷积操作得到一组新的特征。在计算卷积的过程中，需要进行卷积核翻转（旋转180°），而在具体实现上一般会以互相关操作来代替卷积，从而会减少一些不必要的操作或开销。**互相关(cross-correlation)**是一个衡量两个序列相关性的函数，通常是用滑动窗口的点积计算来实现。给定一个图像$$X\in \mathbb{R}^{M\times N}$$和卷积核$$W\in \mathbb{R}^{U\times V}$$，它们的互相关为
$$
Y=W\otimes X={\rm rot180}(W)* X \\
y_{ij}=\sum_{u=1}^U\sum_{v=1}^V w_{uv}x_{i+u-1,j+v-1}
$$
互相关和卷积的区别仅仅在于卷积核是否进行翻转，因此互相关也可以称为**不翻转卷积**。

在神经网络中使用卷积是为了进行特征抽取，卷积核是否翻转不影响特征抽取的能力。为了实现上的方便，我们用互相关来代替卷积。实际上，很多深度学习工具中的卷积操作其实都是互相关操作。



## 卷积的变种

在卷积的标准定义基础上，还可以引入卷积核的滑动步长和零填充来增加卷积的多样性，可以更灵活地进行特征抽取。

**步长(stride)**是指卷积核在滑动时的时间间隔。下图给出了步长为2的卷积示例。

**零填充(zero padding)**是在输入向量两端进行补零。下图给出了输入两端补零的卷积示例。

![](https://i.loli.net/2020/09/15/lg1yThCoXNamjFI.png)

假设卷积层的输入神经元个数为$$M$$，卷积核大小为$$K$$，步长为$$S$$，在输入两端各填补$$P$$个 0，那么该卷积层的神经元数量为$$(M − K + 2P)/S + 1$$。

一般常用的卷积有以下三类：

1. **窄卷积(narrow convolution)**：步长$$S = 1$$，两端不补零$$P = 0$$， 卷积后输出长度为$$M − K + 1$$。
2. **宽卷积(wide convolution)**：步长$$S = 1$$ ，两端补零$$P = K − 1$$， 卷积后输出长度$$M + K − 1$$ 。
3. **等宽卷积(equal-width convolution)**：步长$$S = 1$$， 两端补零$$P =(K − 1)/2$$，卷积后输出长度$$M$$ 。



## 卷积的数学性质

### 交换性

卷积运算具有交换性，互相关也具有同样的性质
$$
X*Y=Y*X\\
X\otimes Y=Y\otimes X\\
$$

### 导数

假设$$Y=W\otimes X$$，其中$$X\in\mathbb{R}^{M\times N},W\in\mathbb{R}^{U\times V},Y\in\mathbb{R}^{(M-U+1)\times (N-V+1)}$$，函数$$f(Y)\in\mathbb{R}$$是一个标量函数，则
$$
\frac{\partial f(Y)}{\partial W}=\frac{\partial f(Y)}{\partial Y}\otimes X\\
\frac{\partial f(Y)}{\partial X}={\rm rot180}(X)\tilde\otimes \frac{\partial f(Y)}{\partial Y}
$$
> 证明$$\frac{\partial f(Y)}{\partial W}=\frac{\partial f(Y)}{\partial Y}\otimes X$$
> $$
> \frac{\partial f(Y)}{\partial w_{uv}}=\sum_{i=1}^{M-U+1} \sum_{j=1}^{N-V+1} \frac{\partial f(Y)}{\partial y_{ij}}\frac{\partial y_{ij}}{\partial w_{uv}}\quad(链式法则)\\
> =\sum_{i=1}^{M-U+1} \sum_{j=1}^{N-V+1} \frac{\partial f(Y)}{\partial y_{ij}}x_{i+u-1,j+v-1}\quad(互相关定义y_{ij}=\sum_{u=1}^U\sum_{v=1}^Vw_{uv}x_{i+u-1,j+v-1})\\
> =(\frac{\partial f(Y)}{\partial Y}\otimes X)_{uv}\quad(互相关定义)\\
> \therefore \frac{\partial f(Y)}{\partial W}=\frac{\partial f(Y)}{\partial Y}\otimes X
> $$

其中**宽卷积(wide convolution)**定义为
$$
W \tilde\otimes X = W \otimes \tilde{X}
$$
其中$$\tilde{X}\in \mathbb{R}^{(M+2U-2)\times (N+2V-2)}$$是图像$$X$$的上下各补$$U-1$$个0，左右各补$$V-1$$个0得到的**全填充(full padding)**图像。





# 卷积神经网络

## 用卷积代替全连接

在全连接前馈神经网络中，如果第$$l$$层有$$M_l$$个神经元，第$$l − 1$$层有$$M_{l−1}$$个神经元，连接边有$$M_l × M_{l−1}$$个，也就是权重矩阵有$$M_l × M_{l−1}$$个参数。当$$M_l$$和$$M_{l−1}$$都很大时，权重矩阵的参数非常多，训练的效率会非常低。

采用卷积来代替全连接
$$
Z^{(l)}=W^{(l)}\otimes\pmb a^{(l-1)}+b^{(l)}
$$

其中卷积核$$W^{(l)}\in \mathbb{R}^K$$为可学习的权重向量，$$b^{(l)}\in \mathbb{R}$$为可学习的偏置。

根据卷积的定义，卷积层有两个很重要的性质：

**局部连接** 卷积层（假设是第$$l$$层）中的每一个神经元都只和前一层（第$$l − 1$$层）中某个局部窗口内的神经元相连，构成一个局部连接网络（如下图所示），卷积层和下一层之间的连接数大大减少，由原来的$$M_l × M_{l−1}$$个连接变为$$M_l × K$$个连接，$$K$$为卷积核大小。

**权重共享** 作为参数的卷积核$$W^{(l)}$$对于第$$l$$层的所有的神经元都是相同的。下图中所有的同颜色连接上的权重是相同的。权重共享可以理解为一个卷积核只捕捉输入数据中的一种特定的局部特征，因此如果要提取多种特征就需要使用多个不同的卷积核。

![](https://i.loli.net/2020/09/15/f9G4lEeVAPKHqQS.png)

由于局部连接和权重共享，卷积层的参数只有一个$$K$$维的权重$$W^{(l)}$$和 1 维的偏置$$b^{(l)}$$，共$$K + 1$$个参数。参数个数和神经元的数量无关。



## 卷积层

卷积层的作用是提取一个局部区域的特征，不同的卷积核相当于不同的特征提取器。上一节中描述的卷积层的神经元和全连接网络一样都是一维结构，由于卷积网络主要应用在图像处理上，而图像为三维结构（对于RGB三通道图像），因此为了更充分地利用图像的局部信息，通常将神经元组织为三维结构的神经层，其大小为高度 $$M$$× 宽度 $$N$$× 深度 $$D$$。

为了提高卷积网络的表示能力，可以在每一层使用多个不同的特征映射，以更好地表示图像的特征。在输入层，特征映射就是图像本身，如果是灰度图像，就有一个特征映射，输入层的深度$$D = 1$$；如果是彩色图像，分别有 RGB 三个颜色通道的特征映射，输入层的深度$$D = 3$$。

不失一般性，假设一个卷积层的结构如下：

1. 输入特征映射组：$$\mathcal{X}\in\mathbb{R}^{M\times N\times D}$$为三维**张量(tensor)**，其中每个**切片(slice)**矩阵$$X^d\in \mathbb{R}^{M\times N}$$为一个输入特征映射，$$1\le d\le D$$
2. 输出特征映射组：$$\mathcal{Y}\in\mathbb{R}^{M'\times N'\times P}$$为三维张量，其中每个切片矩阵$$X^p\in \mathbb{R}^{M'\times N'}$$为一个输出特征映射，$$1\le p\le P$$
3. 卷积核：$$\mathcal{W}\in \mathbb{R}^{U\times V\times P\times D}$$为四维张量，其中每个切片矩阵$$W^{p,d}\in \mathbb{R}^{U \times V}$$为一个二维卷积核，$$1\le d\le D,1\le p\le P$$。

下图给出了卷积层的三维结构表示

![](https://i.loli.net/2020/09/15/Q2OKcoED9wfF86l.png)

为了计算输出特征映射$$Y_p$$ , 用卷积核$$W^{p,1},W^{p,2}, ⋯ ,W^{p,D}$$分别对输入特征映射$$X^1, X^2, ⋯ , X^D$$进行卷积,，然后将卷积结果相加，并加上一个标量偏置$$b^p$$得到卷积层的净输入$$Z^p$$，再经过非线性激活函数后得到输出特征映射 $$Y^p$$。
$$
Z^p=W^p \otimes X +b^p=\sum_{d=1}^D W^{p,d} \otimes X^d +b^p\\
Y^p=f(Z^p)
$$
其中$$W^p\in \mathbb{R}^{U\times V\times D}$$是三维卷积核，$$f(\cdot)$$为非线性激活函数，一般为 ReLU 函数。

整个计算过程如下图所示。如果希望卷积层输出$$P$$个特征映射，将计算过程重复$$P$$次即可，得到$$Y^1,Y^2,\cdots,Y^P$$。

![](https://i.loli.net/2020/09/15/rAqMc7I1E9wNj6m.png)

在上述卷积层中，每一次$$W^{p,d} \otimes X^d$$运算需要卷积核的$$U\times V$$个参数，因此总共需要$$P\times D\times U\times V+P$$个参数。



## 汇聚层

**汇聚层(池化层，pooling layer)**也叫**子采样层(subsampling layer)**，其作用是进行特征选择，降低特征数量，从而减少参数数量。

卷积层虽然可以显著减少网络中连接的数量，但特征映射组中的神经元个数并没有显著减少。如果后面接一个分类器，分类器的输入维数依然很高，很容易出现过拟合。为了解决这个问题，可以在卷积层之后加上一个汇聚层，从而降低特征维数，避免过拟合。

假设汇聚层的输入特征映射组为$$\mathcal{X}\in\mathbb{R}^{M\times N\times D}$$，对于其中每一个特征映射$$X^d ∈ \mathbb{R}^{M×N}, 1 ≤ d ≤ D$$ ，将其划分为很多区域$$R^d_{m,n} , 1 ≤ m ≤ M ′ , 1 ≤ n ≤ N ′$$，这些区域可以重叠，也可以不重叠。**汇聚(pooling)**指对每个区域进行**下采样(down sampling)**得到一个值，作为这个区域的概括。

常用的汇聚函数有两种：

1. **最大汇聚(max pooling)**：对于一个区域$$R^d_{m,n}$$，选择这个区域内所有神经元的最大活性值作为这个区域的表示
2. **平均汇聚(mean pooling)**：一般是取区域内所有神经元活性值的平均值

对每一个输入特征映射$$X^d$$的$$M ′ × N ′$$个区域进行子采样，得到汇聚层的输出特征映射$$Y^d =\{y_{m,n}^d
\}, 1 ≤ m ≤ M ′ , 1 ≤ n ≤ N ′$$。

下图给出了采样最大汇聚进行子采样操作的示例。可以看出汇聚层不但可以有效地减少神经元的数量还可以使得网络对一些小的局部形态改变保持不变性，并拥有更大的感受野。

![](https://i.loli.net/2020/09/15/2a83pO1WzkRCKvG.png)

目前主流的卷积网络中，汇聚层仅包含下采样操作。但在早期的一些卷积网络（比如 LeNet-5）中，有时也会在汇聚层使用非线性激活函数，比如
$$
Y'^d=f(w^dY^d+b^d)
$$
典型的汇聚层是将每个特征映射划分为 2 × 2 大小的不重叠区域，然后使用最大汇聚的方式进行下采样。<u>汇聚层也可以看作一个特殊的卷积层</u>，卷积核大小为$$K × K$$，步长为$$S × S$$，卷积核为 max 函数或 mean 函数。过大的采样区域会急剧减少神经元的数量，也会造成过多的信息损失。



## 卷积神经网络的整体结构

一个典型的卷积网络由卷积层、汇聚层、全连接层交叉堆叠而成。目前常用的卷积网络整体结构如下图所示：一个卷积块为连续$$M$$个卷积层和$$b$$个汇聚层（$$M$$通常设置为 2 ∼ 5，$$b$$为 0 或 1 )，一个卷积网络中可以堆叠$$N$$个连续的卷积块，然后在后面接着$$K$$个全连接层（$$N$$的取值区间比较大，比如 1 ∼ 100 或者更大；$$K$$一般为 0 ∼ 2 )。

![](https://i.loli.net/2020/09/15/fAsRQGLXeuctYdj.png)

目前，卷积网络的整体结构趋向于使用更小的卷积核（比如 1 × 1 和 3 × 3 ）以及更深的结构（比如层数大于 50）。此外，由于卷积的操作性越来越灵活（比如使用不同的步长），汇聚层的作用也变得越来越小，因此目前比较流行的卷积网络中，汇聚层的比例正在逐渐降低，趋向于全卷积网络。





# 参数学习

在卷积网络中，参数为卷积核中权重以及偏置。和全连接前馈网络类似，卷积网络也可以通过误差反向传播算法来进行参数学习。

不失一般性，对第$$l$$层为卷积层，第$$l-1$$层的输入特征映射为$$\mathcal{X}^{(l-1)}\in\mathbb{R}^{M\times N\times D}$$，通过卷积计算得到的第$$l$$层特征映射净输入$$\mathcal{Z}^{(l)}\in\mathbb{R}^{M'\times N'\times P}$$。第$$l$$层特征映射净输入中的第$$p$$个为

$$
Z^{(l,p)}=W^{(l,p)} \otimes X^{(l-1)} +b^{(l,p)}=\sum_{d=1}^D W^{(l,p,d)} \otimes X^{(l-1,d)} +b^{(l,p)}\\
$$
其中$$W^{(l,p,d)}$$和$$b^{(l,p)}$$为卷积核和偏置。第$$l$$层共有$$P\times D$$个卷积核和$$P$$个偏置。计算损失函数对卷积核$$W^{(l,p,d)}$$的偏导数为
$$
\frac{\partial \mathcal{L}}{\partial W^{(l,p,d)}}=\frac{\partial \mathcal{L}}{\partial Z^{(l,p)}}\otimes X^{(l-1,d)}\\
=\pmb \delta^{(l,p)}\otimes X^{(l-1,d)}
$$
其中$$\pmb \delta^{(l,p)}=\frac{\partial \mathcal{L}}{\partial Z^{(l,p)}}$$为损失函数关于第$$l$$层的第$$p$$个特征映射净输入$$Z^{(l,p)}$$的偏导数，即第$$l$$层的误差项。

> 回想卷积的导数计算，若$$Y=W\otimes X$$，则
> $$
> \frac{\partial f(Y)}{\partial W}=\frac{\partial f(Y)}{\partial Y}\otimes X\\
> $$

同理计算损失函数对偏置$$b^{(l,p)}$$的偏导数为
$$
\frac{\partial \mathcal{L}}{\partial b^{(l,p)}}=\sum_{i=1}^{M'}\sum_{j=1}^{N'}\frac{\partial \mathcal{L}}{\partial z_{ij}^{(l,p)}}\frac{\partial z_{ij}^{(l,p)}}{\partial b^{(l,p)}}\\
=\sum_{i=1}^{M'} \sum_{j=1}^{N'} \frac{\partial \mathcal{L}}{\partial z_{ij}^{(l,p)}}\quad(z_{ij}^{(l,p)}=\sum_{m=1}^{M'}\sum_{n=1}^{N'}w_{mn}^{(l,p)}x_{i+m-1,j+n-1}^{(l-1)}+b^{(l,p)})\\
=\frac{\partial \mathcal{L}}{\partial Z^{(l,p)}}的所有元素和\\
=\pmb \delta^{(l,p)} 的所有元素和
$$



## 反向传播算法

### 汇聚层

当第$$l + 1$$层为汇聚层时，因为汇聚层是下采样操作，$$l + 1$$层的每个神经元的误差项$$δ$$对应于第$$l$$ 层的相应特征映射的一个区域。$$l$$层的第$$p$$个特征映射中的每个神经元都有一条边和$$l + 1$$层的第$$p$$个特征映射中的一个神经元相连。

第$$l$$层的第$$p$$个特征映射的误差项$$\pmb \delta^{(l,p)}$$的具体推导过程如下：
$$
\pmb \delta^{(l,p)}\triangleq\frac{\partial \mathcal{L}}{\partial Z^{(l,p)}}\\
=\frac{\partial \mathcal{L}}{\partial Z^{(l+1,p)}}\frac{\partial Z^{(l+1,p)}}{\partial X^{(l,p)}}\frac{\partial X^{(l,p)}}{\partial Z^{(l,p)}}\\
=\pmb \delta^{(l+1,p)} f_l'(Z^{(l,p)})
$$






# 典型的卷积神经网络

## LeNet-5

LeNet-5 [LeCun et al., 1998] 虽然提出的时间比较早，但它是一个非常成功的神经网络模型。基于 LeNet-5 的手写数字识别系统在 20 世纪 90 年代被美国很多银行使用，用来识别支票上面的手写数字。LeNet-5 的网络结构如下图所示。

![](https://i.loli.net/2020/09/15/YoEJpLhvgsuR8kH.png)

LeNet-5 共有 7 层，接受输入图像大小为 32 × 32 = 1024，输出对应 10 个类别的得分。LeNet-5 中的每一层结构如下：

1. C1 层是卷积层，使用 6 个 5×5 的卷积核，得到 6 组大小为 28×28 = 784的特征映射。因此 C1 层的神经元数量为 6 × 784 = 4704，可训练参数数量为 6 × 25 + 6 = 156，连接数为 156 × 784 = 122304（包括偏置在内， 下同）
2. S2 层为汇聚层，采样窗口为 2 × 2 ，使用平均汇聚，并使用一个非线性函数。神经元个数为 6 × 14 × 14 = 1 176 ，可训练参数数量为 6 × (1 + 1) = 12 ，连接数为 6 × 196 × (4 + 1) = 5880。
3. C3 层为卷积层。LeNet-5 中用一个连接表来定义输入和输出特征映射之间的依赖关系，如下图所示，共使用 60 个 5 × 5 的卷积核，得到 16 组大小为 10 × 10 的特征映射。神经元数量为 16 × 100 = 1600，可训练参数数量为 (60 × 25) + 16 = 1516 ，连接数为 100 × 1516 = 151600。
4. S4 层是一个汇聚层，采样窗口为 2 × 2，得到 16 个 5 × 5 大小的特征映射，可训练参数数量为 16 × 2 = 32 ，连接数为 16 × 25 × (4 + 1) = 2000。
5. C5 层是一个卷积层，使用 120 × 16 = 1920 个 5 × 5 的卷积核，得到120 组大小为 1 × 1 的特征映射。C5 层的神经元数量为 120，可训练参数数量为1920 × 25 + 120 = 48120，连接数为 120 × (16 × 25 + 1) = 48120。
6. F6 层是一个全连接层，有 84 个神经元，可训练参数数量为 84 × (120 +1) = 10164 ，连接数同样为 10164。
7. 输出层：输出层由 10 个径向基函数(Radial Basis Function , RBF)组成。这里不再详述。







## AlexNet

AlexNet [Krizhevsky et al., 2012] 是第一个现代深度卷积网络模型，其首次使用了很多现代深度卷积网络的技术方法，比如使用 GPU 进行并行训练，采用了 ReLU 作为非线性激活函数，使用 Dropout 防止过拟合，使用数据增强来提高模型准确率等。AlexNet 赢得了 2012 年 ImageNet 图像分类竞赛的冠军。

AlexNet 的结构如下图所示，包括 5 个卷积层、3 个汇聚层和 3 个全连接层（其中最后一层是使用 Softmax 函数的输出层）。因为网络规模超出了当时的单个 GPU 的内存限制，AlexNet 将网络拆为两半，分别放在两个 GPU 上，GPU 间只在某些层（比如第 3 层）进行通信。

![](https://i.loli.net/2020/09/15/ezxTEyKmCk2oS9D.png)











# 其它卷积方式

## 转置卷积

## 空洞卷积