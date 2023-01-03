# 熵、相对熵与互信息

## 熵

熵是随机变量不确定性的度量。设 $X$ 是一个<u>离散型</u>随机变量，其取值空间为 $\mathcal{X}$，概率密度函数（概率分布）$p(x)={\rm Pr}(X=x),x\in\mathcal{X}$。

一个离散型随机变量 $X$ 的**熵（entropy）**$H(X)$ 定义为

$$
H(X)=-\sum_{x\in\mathcal{X}}p(x)\log_2 p(x)
$$

熵的单位用比特（位）表示。例如，抛掷均匀硬币这一事件的熵为 1 比特。由于当 $x\to 0$ 时，$x\log x\to 0$，我们约定 $0\log 0=0$，这意味着加上零概率的项不改变熵的值（因为它没有增加不确定性）。

!!! note "说明"
    熵实际上是随机变量 $X$ 的分布的泛函（和期望、方差一样），并不依赖于 $X$ 的实际取值，而仅依赖于其概率分布。

$X$ 的熵又解释为随机变量 $\log \frac{1}{p(X)}$ 的期望，于是

$$
H(X)=E\log\frac{1}{p(x)}
$$

引理 1：

$$
H(X)\ge 0
$$

证明：

由 $0\le p(x)\le 1$ 有 $-\log{p(x)}\ge 0$，有 $-p(x)\log{p(x)}\ge 0$

@设 $P(X=1)=p$，$P(X=0)=1-p$，则

$$
H(X)=-p\log p-(1-p)\log (1-p)\stackrel{\text{def}}{=}H(p)
$$

函数 $H(p)$ 的图形如下，它说明了熵的一些基本性质：

* $H(p)$ 为上凸函数
* 当 $p=0$ 或 $1$ 时，$H(p)=0$，表示变量不再是随机的，因而不再具有不确定性
* 当 $p=\frac{1}{2}$ 时，变量的不确定性达到最大，此时对应于熵也取最大值

![](https://s2.loli.net/2022/12/22/KTx4QYPwHmdan8G.png)

@假定有 8 匹马参加一场赛马比赛，这 8 匹马的获胜概率分布为 $(\frac{1}{2},\frac{1}{4},\frac{1}{8},\frac{1}{16},\frac{1}{64},\frac{1}{64},\frac{1}{64},\frac{1}{64})$，则这场赛马的熵为

$$
H(X)=-\frac{1}{2}\log\frac{1}{2}-\frac{1}{4}\log\frac{1}{4}-\frac{1}{8}\log\frac{1}{8}-\frac{1}{16}\log\frac{1}{16}-4\frac{1}{64}\log\frac{1}{64}=2比特
$$

假定我们要把哪匹马会获胜的消息发送出去，其中一个策略是发送胜出马的编号，这样对任意一匹马，描述需要固定 3 比特。但由于获胜的概率不是均等的，因此明智的方法是对获胜概率较大的马使用较短的描述，而对获胜概率较小的马使用较长的描述。这样做，我们会获得一个更短的平均描述长度。例如，使用以下的一组二元字符串来表示 8 匹马：0、10、110、1110、111100、111101、111110、111111。此时平均描述长度为 2 比特（正好等于熵），小于等长编码的 3 比特。

## 联合熵和条件熵

现在我们将定义推广到两个随机变量的情形。由于可将 $(X,Y)$ 视为一个随机向量，所以定义其实并无新鲜之处。

对于服从联合分布为 $p(x,y)$ 的一对离散型随机变量 $(X,Y)$，其**联合熵（joint entropy）**$H(X,Y)$ 定义为

$$
H(X,Y)=-\sum_{x\in\mathcal{X}}\sum_{y\in\mathcal{Y}}p(x,y)\log p(x,y)
$$

上式亦可表示为

$$
H(X,Y)=E\log \frac{1}{p(x,y)}
$$

也可以定义一个随机变量在给定另一个随机变量下的条件熵，它是条件分布熵关于起条件作用的那个随机变量取平均之后的期望值。

对于服从联合分布为 $p(x,y)$ 的一对离散型随机变量 $(X,Y)$，其**条件熵（joint entropy）**$H(Y|X)$ 定义为

$$
\begin{align}
H(Y|X)&=\sum_{x\in\mathcal{X}}p(x)H(Y|X=x)\\
&=-\sum_{x\in\mathcal{X}}p(x)\sum_{y\in\mathcal{Y}}p(y|x)\log p(y|x)\\
&=-\sum_{x\in\mathcal{X}}\sum_{y\in\mathcal{Y}}p(x,y)\log p(y|x)\\
&=-E\log p(y|x)
\end{align}
$$

联合熵和条件熵的定义的这种自然性克由一个事实得到体现，即一对随机变量的联合熵等于其中一个随机变量的熵加上另一个随机变量的条件熵。

定理 1：

$$
H(X,Y)=H(X)+H(Y|X)
$$

证明：

$$
\begin{align}
H(X,Y)&=-\sum_{x\in\mathcal{X}}\sum_{y\in\mathcal{Y}}p(x,y)\log p(x,y)\\
&=-\sum_{x\in\mathcal{X}}\sum_{y\in\mathcal{Y}}p(x,y)\log p(x)p(y|x)\\
&=-\sum_{x\in\mathcal{X}}\sum_{y\in\mathcal{Y}}p(x,y)\log p(x)-\sum_{x\in\mathcal{X}}\sum_{y\in\mathcal{Y}}p(x,y)\log p(y|x)\\
&=-\sum_{x\in\mathcal{X}}p(x)\log p(x)-\sum_{x\in\mathcal{X}}\sum_{y\in\mathcal{Y}}p(x,y)\log p(y|x)\\
&=H(X)+H(Y|X)
\end{align}
$$

@设 $(X, Y)$ 服从如下的联合分布：

![](https://s2.loli.net/2022/12/23/9gUrslYT78NIXiM.png)

$X$ 的边际分布为 $(\frac{1}{2},\frac{1}{4},\frac{1}{8},\frac{1}{8})$，Y 的边际分布为 $(\frac{1}{4},\frac{1}{4},\frac{1}{4},\frac{1}{4})$，因而 $H(X)=\frac{7}{4}$ 比特，$H(Y)=2$ 比特，并且

$$
\begin{align}
H(X|Y)&=\frac{1}{4}H(\frac{1}{2},\frac{1}{4},\frac{1}{8},\frac{1}{8})+\frac{1}{4}H(\frac{1}{4},\frac{1}{2},\frac{1}{8},\frac{1}{8})+\frac{1}{4}H(\frac{1}{4},\frac{1}{4},\frac{1}{4},\frac{1}{4})+\frac{1}{4}H(1,0,0,0)\\
&=\frac{1}{4}×\frac{7}{4}+\frac{1}{4}×\frac{7}{4}+\frac{1}{4}×2+\frac{1}{4}×0\\
&=\frac{11}{8} 比特
\end{align}
$$

同样地，有 $H(Y|X)=\frac{13}{8}$ 比特，$H(X,Y)=\frac{27}{8}$ 比特。

## 相对熵和互信息

熵是随机变量不确定性的度量：它也是平均意义上描述随机变量所需的信息量的度量。本节介绍两个相关的额念：相对熵和互信息。

相对熵是两个随机分布之间距离的度量。在统计学中，它对应的是似然比的对数期望。相对熵 $D(p\parallel q)$ 度量当真实分布为 $p$，而假定分布为 $q$ 时的无效性。例如，巳知随机变量的真实分布为 $p$，那么可以构造平均描述长度为 $H(p)$ 的编码；但是，如果使用的是基于分布 $q$ 构造的编码，那么在平均意义上就需要 $H(p) +D(p\parallel q)$ 比特来描述这个随机变量。

两个概率分布 $p(x)$ 和 $q(x)$ 之间的**相对熵（relative entropy）**或 **Kullback-Leibler 距离**定义为：

$$
\begin{align}
D(p \parallel q)&=\sum_{x\in\mathcal{X}}p(x)\log\frac{p(x)}{q(x)}\\
&=E_p\log\frac{p(x)}{q(x)}\\
\end{align}
$$

在上述定义中，我们约定 $0\log\frac{0}{0}=0$，$0\log\frac{0}{q}=0$，这意味着对于分布 $p$ 加上零概率的项不改变相对熵的值（因为它不需要被编码）。约定 $p\log\frac{p}{0}=\infty$，这意味着如果分布 $q$ 不能取到分布 $p$ 可以取到的特定值，则相对熵为无穷（因为它无法被编码）。

之后我们将证明相对熵总是非负的，并且当且仅当 $p=q$ 时为零。我们经常将相对熵视作分布之间的“距离”，尽管它并不对称，也不满足三角不等式。

现在来介绍互信息，它是一个随机变量包含另一个随机变量信息量的度量。互信息也是在给定另一随机变量分布的系件下，原随机变量不确定性的缩减量。

考虑两个随机变量 $X$ 和 $Y$，它们的联合概率密度函数为 $p(x,y)$，边际概率密度函数分别是 $p(x)$ 和 $p(y)$。**互信息（mutual information）**$I(X;Y)$ 是联合分布 $p(x,y)$ 和乘积分布 $p(x)p(y)$ 之间的相对熵，即：

$$
\begin{align}
I(X;Y)&=\sum_{x\in\mathcal{X}}\sum_{y\in\mathcal{Y}}p(x,y)\log\frac{p(x,y)}{p(x)p(y)}\\
&=D(p(x,y)\parallel p(x)p(y))\\
&=E_{p(x,y)}\log\frac{p(x,y)}{p(x)p(y)}
\end{align}
$$

@设 $\mathcal{X}=\{0,1\}$，考虑 $\mathcal{X}$ 上的两个分布 $p$ 和 $q$。设 $p(0)=1-r,p(1)=r$ 以及 $q(0)=1-s,q(1)=s$，则：

$$
D(p\parallel q)=(1-r)\log\frac{1-r}{1-s}+r\log\frac{r}{s}
$$

以及

$$
D(q\parallel p)=(1-s)\log\frac{1-s}{1-r}+s\log\frac{s}{r}
$$

如果 $r=s$，则 $D(p\parallel q)=D(q\parallel p)=0$。若 $r=\frac{1}{2},s=\frac{1}{4}$，可以计算得到：

$$
D(p\parallel q)=0.2075比特
$$

和

$$
D(q\parallel p)=0.1887比特
$$

## 熵和互信息的关系

可将互信息 $I(X;Y)$ 重新写为：

$$
\begin{align}
I(X;Y)&=\sum_{x,y}p(x,y)\log\frac{p(x,y)}{p(x)p(y)}\\
&=\sum_{x,y}p(x,y)\log\frac{p(x|y)}{p(x)}\\
&=-\sum_{x,y}p(x,y)\log p(x)+\sum_{x,y}p(x,y)\log p(x|y)\\
&=-\sum_{x}p(x)\log p(x)-(-\sum_{x,y}p(x,y)\log p(x|y))\\
&=H(X)-H(X|Y)\\
\end{align}
$$

由此，互信息 $I(X;Y)$ 是在给定 $Y$ 知识的条件下 $X$ 的不确定性的缩减量。

对称地，亦可得到：

$$
I(X;Y)=H(Y)-H(Y|X)
$$

因此，$X$ 含有 $Y$ 的信息量等于 $Y$ 含有 $X$ 的信息量。

再由 $H(X,Y)=H(X)+H(Y|X)$，可以得到：

$$
I(X;Y)=H(X)+H(Y)-H(X,Y)
$$

最后，注意到：

$$
I(X;X)=H(X)-H(X|X)=H(X)
$$

随机变量与其自身的互信息为该随机变量的熵，因此熵有时称为**自信息（self-information）**。

综合以上结论，有下面的定理：

定理 2：

$$
\displaylines{
I(X;Y)=H(X)-H(X|Y)=H(Y)-H(Y|X)\\
I(X;Y)=H(X)+H(Y)-H(X,Y)\\
I(X;Y)=I(Y;X)\\
I(X;X)=H(X)
}
$$

上面各项之间的关系可用如下文氏图表示：

![](https://s2.loli.net/2022/12/26/nskQZaoe67pUIjy.png)

## 熵、相对熵和互信息的链式法则

定理 3：设随机变量 $X_1,X_2,\cdots,X_n$ 服从 $p(x_1,x_2,\cdots,x_n)$，则：

$$
H(X_1,X_2,\cdots,X_n)=\sum_{i=1}^nH(X_i|X_{i-1},\cdots,X_1)
$$

证明：

$$
\begin{align}
H(X_1,X_2,\cdots,X_n)&=-\sum_{x_1,x_2,\cdots,x_n}p(x_1,x_2,\cdots,x_n)\log p(x_1,x_2,\cdots,x_n)\\
&=-\sum_{x_1,x_2,\cdots,x_n}p(x_1,x_2,\cdots,x_n)\log\prod_{i=1}^np(x_i|x_{i-1},\cdots,x_1)\\
&=-\sum_{x_1,x_2,\cdots,x_n}\sum_{i=1}^np(x_1,x_2,\cdots,x_n)\log p(x_i|x_{i-1},\cdots,x_1)\\
&=-\sum_{i=1}^n\sum_{x_1,x_2,\cdots,x_n}p(x_1,x_2,\cdots,x_n)\log p(x_i|x_{i-1},\cdots,x_1)\\
&=-\sum_{i=1}^n\sum_{x_1,x_2,\cdots,x_i}p(x_1,x_2,\cdots,x_n)\log p(x_i|x_{i-1},\cdots,x_1)\\
&=\sum_{i=1}^nH(X_i|X_{i-1},\cdots,X_1)\\
\end{align}
$$

下面定义条件互信息，它是在给定 $Z$ 时由于 $Y$ 的知识而引起关于 $X$ 的不确定性的缩减量。

随机变量 $X$ 和 $Y$ 在给定随机变量 $Z$ 时的**条件互信息（conditional mutual information）**定义为：

$$
\begin{align}
I(X;Y|Z)&=H(X|Z)-H(X|Y,Z)\\
&=E_p(x,y,z)\log\frac{p(x,y|z)}{p(x|z)p(y|z)}
\end{align}
$$

互信息亦满足链式法则。

定理 4：

$$
I(X_1,X_2,\cdots,X_n;Y)=\sum_{i=1}^nI(X_i;Y|X_{i-1},X_{i-2},\cdots,X_1)
$$

证明：

$$
\begin{align}
I(X_1,X_2,\cdots,X_n;Y)&=H(X_1,X_2,\cdots,X_n)-H(X_1,X_2,\cdots,X_n|Y)\\
&=\sum_{i=1}^nH(X_i|X_{i-1},\cdots,X_1)-\sum_{i=1}^nH(X_i|X_{i-1},\cdots,X_1,Y)\\
&=\sum_{i=1}^nI(X_i;Y|X_{i-1},\cdots,X_1)
\end{align}
$$

下面定义相对熵的条件形式。

对于联合概率密度函数 $p(x,y)$ 和 $q(x,y)$，**条件相对熵（conditional relative entropy）**$D(p(y|x)\parallel q(y|x))$ 定义为条件概率密度函数 $p(y|x)$ 和 $q(y|x)$ 之间的平均相对熵，其中取平均是关于概率密度函数 $p(x)$ 而言的。更确切地，

$$
\begin{align}
D(p(y|x)\parallel q(y|x))&=\sum_xp(x)\sum_yp(y|x)\log\frac{p(y|x)}{q(y|x)}\\
&=\sum_{x,y}p(x,y)\log{\frac{p(y|x)}{q(y|x)}}\\
&=E_{p(x,y)}\log\frac{p(y|x)}{q(y|x)}
\end{align}
$$

一对随机变量的两个联合分布之间的相对熵可以展开为相对熵和条件相对熵之和。

定理 5：

$$
D(p(x,y)\parallel q(x,y))=D(p(x)\parallel q(x))+D(p(y|x)\parallel q(y|x))
$$

证明：

$$
\begin{align}
D(p(x,y)\parallel q(x,y))&=\sum_{x,y}p(x,y)\log\frac{p(x,y)}{q(x,y)}\\
&=\sum_{x,y}p(x,y)\log\frac{p(x)p(y|x)}{q(x)q(y|x)}\\
&=\sum_{x,y}p(x,y)\log\frac{p(x)}{q(x)}+\sum_{x,y}p(x,y)\log\frac{p(y|x)}{q(y|x)}\\
&=D(p(x)\parallel q(x))+D(p(y|x)\parallel q(y|x))
\end{align}
$$

## 熵、相对熵和互信息的性质

定理 6（Jensen 不等式）：若给定凸函数 $f$ 和一个随机变量 $X$，则

$$
Ef(X)\ge f(EX)
$$

进一步地，若 $f$ 是严格凸函数，则上式中的等式蕴含 $X=EX$ 的概率为 1（即 $X$ 是个常量）。

证明略。

定理 7（信息不等式）：设 $p(x),q(x)(x\in\mathcal{X})$ 是两个概率密度函数。则

$$
D(p\parallel q)\ge 0
$$

当且仅当 $\forall x,p(x)=q(x)$ 时等号成立。

证明：

设 $A=\{x:p(x)>0\}$ 是 $p(x)$ 的支撑集，则：

$$
\begin{align}
-D(p\parallel q)&=-\sum_{x\in A}p(x)\log\frac{p(x)}{q(x)}\\
&=\sum_{x\in A}p(x)\log\frac{q(x)}{p(x)}\\
&\le\log\sum_{x\in A}p(x)\frac{q(x)}{p(x)}\quad{\rm (1)}\\
&=\log\sum_{x\in A}q(x)\\
&\le\log\sum_{x\in\mathcal{X}}q(x)\quad{\rm (2)}\\
&=\log 1\\
&=0
\end{align}
$$

式 (1) 由 Jensen 不等式得到。由于 $\log t$ 是关于 $t$ 的严格上凸函数，当且仅当 $q(x)/p(x)$ 恒为常数（即 $\forall x\in A,q(x)=cp(x)$）时式 (1) 取等号，即 $\sum_{x\in A}q(x)=c\sum_{x\in A}p(x)=c$。另外，式 (2) 取等号即 $\sum_{x\in A}q(x)=\sum_{x\in\mathcal{X}}q(x)=1$。因此式 (1) 和式 (2) 同时取等号，即 $D(p\parallel q)=0$，当且仅当 $\forall x,p(x)=q(x)$。

推论（互信息的非负性）：对任意两个随机变量 $X$ 和 $Y$，

$$
I(X;Y)\ge 0
$$

当且仅当 $X$ 与 $Y$ 相互独立时，等号成立。

证明：$I(X;Y)=D(p(x,y)\parallel p(x)p(y))\ge 0$，当且仅当 $p(x,y)=p(x)p(y)$ （即 $X$ 与 $Y$ 相互独立）时，等号成立。

推论：

$$
D(p(y|x)\parallel q(y|x))\ge 0
$$

当且仅当对任意的 $y$ 以及满足 $p(x)>0$ 的 $x$，有 $p(y|x)=q(y|x)$ 时，等号成立。

推论：

$$
I(X;Y|Z)\ge 0
$$

当且仅当对给定随机变量 $Z$，$X$ 和 $Y$ 条件独立时，等号成立。

下面证明字母表 $\mathcal{X}$ 上的均匀分布是 $\mathcal{X}$ 上的最大熵分布。由此可知，$\mathcal{X}$ 上的任何随机变量的熵都不超过 $\log|\mathcal{X}|$。

定理 8：$H(X)\le\log|\mathcal{X}|$，其中 $|\mathcal{X}|$ 表示 $X$ 的字母表 $\mathcal{X}$ 中元素的个数，当且仅当 $X$ 服从 $\mathcal{X}$ 上的均匀分布时，等号成立。

证明：设 $u(x)=\frac{1}{|\mathcal{X}|}$ 为 $\mathcal{X}$ 上均匀分布的概率密度函数，$p(x)$ 是随机变量 $X$ 的概率密度函数，于是：

$$
D(p\parallel u)=\sum p(x)\log\frac{p(x)}{u(x)}=\log|\mathcal{X}|-H(X)
$$

然后由相对熵的非负性：

$$
0\le D(p\parallel u)=\log|\mathcal{X}|-H(X)
$$

定理 9（条件作用使熵减小）（信息不会有负面影响）：

$$
H(X|Y)\le H(X)
$$

当且仅当 $X$ 与 $Y$ 相互独立时，等号成立。

证明：$0\le I(X;Y)=H(X)-H(X|Y)$

从直观上讲，此定理说明知道另一随机变量 $Y$ 的信息只会降低 $X$ 的不确定度。注意，这仅对平均意义成立。具体来讲，$H(X|Y=y)$ 可能比 $H(X)$ 大或者小，或两者相等，但在平均意义上。$H(X|Y)= \sum_{y}p(y)H(X|Y=y)\le H(X)$。例如，在法庭上。特定的新证据可能会增加不确定性，但在通常情况下，证据是降低不确定性的。

@设 $(X,Y)$ 服从下图的联合分布：

![](https://s2.loli.net/2022/12/27/yYrIvLHpi1nqztB.png)

则 $H(X)=H(\frac{1}{8},\frac{7}{8})=0.544 比特$，$H(X|Y=1)=0$ 比特，$H(X|Y=2)=1$ 比特，故 $H(X|Y)=\frac{3}{4}H(X|Y=1)+\frac{1}{4}H(X|Y=2)=0.25 比特$。因此，当观察到 $Y=2$ 时，$X$ 的不确定性增加；而观察到 $Y=1$ 时，$X$ 的不确定性减少，但是在平均意义下 $X$ 的不确定性是减少的。

定理 10（熵的独立界）设 $X_1,X_2,\cdots,X_n$ 服从 $p(x_1,x_2,\cdots,x_n)$ 则

$$
H(X_1,X_2,\cdots,X_n)\le\sum_{i=1}^nH(X_i)
$$

当且仅当 $X_i$ 相互独立时，等号成立。

证明：由熵的链式法则，

$$
H(X_1,X_2,\cdots,X_n)=\sum_{i=1}^nH(X_i|X_{i-1},\cdots,X_1)\le\sum_{i=1}^nH(X_i)
$$

其中不等式由定理 9 得到，当且仅当 $\forall i$，$X_i$ 与 $X_{i-1},\cdots,X_1$ 独立时（即当且仅当 $X_i$ 相互独立时），等号成立。

定理 11（对数和不等式）：
