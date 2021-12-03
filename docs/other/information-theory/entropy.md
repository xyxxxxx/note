**熵(Entropy)**最早是物理学的概念,用于表示一个热力学系统的无序程度.在信息论中,熵用来衡量一个随机事件的不确定性.





# 自信息和熵

**自信息(Self Information)**表示一个随机事件所包含的信息量.一个随机事件发生的概率越高,其自信息越低.如果一个事件必然发生,其自信息为0.

对于一个随机变量 $X$ (取值集合为 $\mathcal{X}$ ,概率分布为 $p(x),x∈\mathcal{X}$ ),当 $X=x$ 时的自信息 $I(x)$ 定义为
$$
I(x) = − \log p(x)
$$
在自信息的定义中,对数的底可以使用2,自然常数e或是10.当底为2时,自信息的单位为bit;当底为e时,自信息的单位为nat.

对于分布为 $p(x)$ 的随机变量 $X$ ,其自信息的数学期望,即熵 $H(X)$ 定义为
$$
H(X)=E(I(x))=E(−\log p(x))
=−\sum_{x\in \mathcal{X}} p(x)\log p(x)
$$

其中当 $p(x_i)=0$ 时,我们定义 $0\log0=0$ ,这与极限一致,即 $\lim_{p→0+}p\log p=0$ .

熵越高,则随机变量的信息越多;熵越低,则随机变量的信息越少.如果变量 $X$ 当且仅当在 $x$ 时 $p(x)=1$ ,则熵为0.也就是说,对于一个确定的信息,其熵为0,信息量也为0.如果其概率分布为一个均匀分布,则熵最大.

<img src="http://photos1.blogger.com/blogger/5682/4111/1600/EntropyVersusProbability.0.png" style="zoom:67%;" />





# 熵编码

信息论的研究目标之一是如何用最少的编码表示传递信息.假设我们要传递一段文本信息,这段文本中包含的符号都来自于一个字母表A,我们就需要对字母表A中的每个符号进行编码.以二进制编码为例,我们常用的ASCII码就是用固定的8bits来编码每个字母.但这种固定长度的编码方案不是最优的.一种高效的编码原则是字母的出现概率越高,其编码长度越短.比如对字母a,b,c分别编码为0,10,110.
给定一串要传输的文本信息,其中字母 $x$ 的出现概率为 $p(x)$ ,其最佳编码长度为 $−\log p(x)$ ,整段文本的平均编码长度为 $−\sum_{x\in\mathcal{X}}p(x)\log p(x)$ ,即熵.
在对分布 $p(x)$ 的符号进行编码时,熵 $H(p)$ 也是理论上最优的平均编码长度,这种编码方式称为熵编码(Entropy Encoding).
由于每个符号的自信息通常都不是整数,因此在实际编码中很难达到理论上的最优值.霍夫曼编码(Huffman Coding)和算术编码(Arithmetic Coding)是两种最常见的熵编码技术.





# 联合熵和条件熵

对于两个离散随机变量 $X$ 和 $Y$ ,假设 $X$ 取值集合为 $\mathcal{X}$ ; $Y$ 取值集合为 $\mathcal{Y}$ ,其联合概率分布满足为 $p(x,y)$ ,则 $X$ 和 $Y$ 的**联合熵(Joint Entropy)**为
$$
H(X, Y ) = − \sum_{x\in\mathcal{X}} \sum_{y\in\mathcal{Y}} p(x, y) \log p(x, y)
$$
$X$ 和 $Y$ 的**条件熵(Conditional Entropy)**为
$$
H(X| Y ) = − \sum_{x\in\mathcal{X}} \sum_{y\in\mathcal{Y}} p(x, y) \log p(x| y)\\
=− \sum_{x\in\mathcal{X}} \sum_{y\in\mathcal{Y}} p(x, y) \log \frac{p(x,y)}{p(y)}
$$
根据其定义,条件熵也可以写为
$$
H(X|Y ) = H(X, Y ) − H(Y )
$$






# 互信息

**互信息(Mutual Information)**是衡量已知一个变量时,另一个变量不确定性的减少程度.两个离散随机变量X和Y的互信息定义为
$$
I(X; Y ) =\sum_{x\in\mathcal{X}} \sum_{y\in\mathcal{Y}} p(x, y) \log \frac{p(x, y)}{p(x)p(y)}
$$
互信息的一个性质为
$$
I(X; Y ) = H(X) − H(X|Y )= H(Y ) − H(Y |X)
$$

如果变量 X 和 Y 互相独立,它们的互信息为零.

![](https://lh3.googleusercontent.com/proxy/YVe7uTW_NamRMeTUndmA5qwFXMEpll65gCwVVYjE-me52OuyeYqOoy51ck55741_Arx08of9vXh_Sxu8JZudoR45)





# 交叉熵和散度

## 交叉熵

对于分布为 $p(x)$ 的随机变量,熵 $H(p)$ 表示其最优编码长度.**交叉熵(Cross Entropy)**是按照概率分布 $q$ 的最优编码对真实分布为 $p$ 的信息进行编码的长度,定义为
$$
H(p, q) = E (− \log q(x))= − \sum_x p(x) \log q(x)
$$
在给定 $p$ 的情况下,如果 $q$ 和 $p$ 越接近,交叉熵越小;如果 $q$ 和 $p$ 越远,交叉熵就越大.



## KL散度

**KL散度(Kullback-Leibler Divergence)**,也叫KL距离或**相对熵(Relative Entropy)**,是用概率分布q来近似p时所造成的信息损失量.KL散度是按照概率分布q的最优编码对真实分布为p的信息进行编码,其平均编码长度(即交叉熵)H(p,q)和p的最优平均编码长度(即熵)H(p)之间的差异.对于离散概率分布p和q,从q到p的KL散度定义为
$$
KL(p, q) = H(p, q) − H(p)=\sum_x p(x \log \frac{p(x)}{q(x)})
$$
其中为了保证连续性,定义 $0\log \frac{0}{0}=0, 0\log \frac{0}{q}=0$ .

KL散度总是非负的, $KL(p,q)≥0$ ,可以衡量两个概率分布之间的距离.KL散度只有当 $p=q$ 时, $KL(p,q)=0$ .如果两个分布越接近,KL散度越小;如果两个分布越远,KL散度就越大.但KL散度并不是一个真正的度量或距离,一是KL散度不满足距离的对称性,二是KL散度不满足距离的三角不等式性质.





## JS散度



## Wasserstein距离