# 异常检测

![](https://s2.loli.net/2023/03/07/YptArbn1xheKQo8.png)

![](https://s2.loli.net/2023/03/07/1lJ8dvmpuBUh5sg.png)

![](https://s2.loli.net/2023/03/07/N14fSmXyngh6c39.png)

## 参考

* [Anomaly Detection (1/7)](https://www.youtube.com/watch?v=gDp2LXGnVLQ) ...

## 分类器和信心分数

让分类器在输出类型的同时输出信心分数，根据信心分数进行异常检测。

![](https://s2.loli.net/2023/03/07/ct7GN42pie85BjC.png)

可以根据输出概率分布计算信心分数；信心分数可以是概率最大值，概率分布的熵，等等。有研究指出使用这些信心分数的效果差别不大。

![](https://s2.loli.net/2023/03/07/fDkPMr6vRdnEz9x.png)

分类器用于异常检测的效果并没有特别差，因此常被用作 baseline。

也可以训练一个网络直接输出信心分数。

![](https://s2.loli.net/2023/03/07/XfE6ASQdpolUnrv.png)

验证集用于确定阈值 $\lambda$ 的值，通过计算 $\lambda$ 取不同值时二分类器 $f(x)$ 的表现（precision、recall、f1 score 等）。

![](https://s2.loli.net/2023/03/08/cBeQIkjf329zWp6.png)

一个潜在的问题是若异常样本具有正常样本的一些特征（不论是天然相似还是人为添加），则分类器容易为其赋予高信心分数从而判断为正常样本。

![](https://s2.loli.net/2023/03/08/LE1DkcKx6S4BOhJ.png)

![](https://s2.loli.net/2023/03/08/oEn3rKTajWF8ZbP.png)

一个解决方法是在训练数据中加入异常样本，并为它们设定低目标信心分数，作监督学习。异常样本可以通过生成模型（例如 GAN）来获得，生成的异常样本应与正常样本比较相似但又有所不同。

![](https://s2.loli.net/2023/03/08/5ZmFMOCKqpPDvib.png)

## 概率密度估计

![](https://s2.loli.net/2023/03/08/J8f1mRhB9NK3uAw.png)

![](https://s2.loli.net/2023/03/08/MQsPeoRY4tSOmCx.png)

![](https://s2.loli.net/2023/03/08/3QrVFos8TiS92HI.png)

![](https://s2.loli.net/2023/03/08/mCiBDvJXVRPtw94.png)

## 自编码器

![](https://s2.loli.net/2023/03/07/x6I59AjcOWRCv3m.png)

![](https://s2.loli.net/2023/03/07/WPCfoAJxw78lahv.png)
