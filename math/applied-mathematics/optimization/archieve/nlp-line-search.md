# 线搜索

求目标函数在直线上的极小点称为**线搜索**，或称为**线搜索**。

设目标函数为$$f(\pmb x)$$，过点$$\pmb x^{(k)}$$沿方向$$\pmb p^{(k)}$$的直线可用点集表示，记作
$$
L=\{\pmb x|\pmb x=\pmb x^{(k)}+\alpha \pmb p^{(k)},-\infty<\alpha<\infty \}
$$
求$$f(\pmb x)$$在直线$$L$$上的极小点转化为求一元函数
$$
\varphi(\alpha)=f(\pmb x^{(k)}+\alpha \pmb p^{(k)})
$$
的极小点。

线搜索的方法归纳起来，大体可分为两类：

+ **试探法** 这类方法需要按某种方式寻找试探点，通过一系列试探点来确定极小点
+ **函数逼近法/插值法** 这类方法用某种较简单的曲线逼近本来的函数曲线，通过求逼近函数的极小点来估计目标函数的极小点

这两类方法一般只能求得极小点的近似值。

**线搜索的算法映射** 算法映射$$M:\mathbb{R}^n\times \mathbb{R}^n\to \mathbb{R}^n$$定义为
$$
M(\pmb x,\pmb p)=\{\pmb y|\pmb y= \pmb x+\overline{\alpha}\pmb p, \overline{\alpha}=\arg\min_{0\le \alpha <\infty}f(\pmb x+\alpha \pmb p) \}
$$
**定理** 设$$f$$是定义在$$\mathbb{R}^n$$上的连续函数，$$\pmb p\neq \pmb 0$$，则线搜索的算法映射$$M$$在$$(\pmb x,\pmb p)$$处是闭的。



## 试探法

**单峰函数** 设$$f$$是定义在闭区间$$[a,b]$$上的一元实函数，$$\overline{x}$$是$$f$$在$$[a,b]$$上的极小点，并且$$\forall x^{(1)},x^{(2)}\in [a,b],x^{(1)}<x^{(2)}$$，都有$$x^{(2)}\le \overline{x}\Rightarrow f(x^{(1)})>f(x^{(2)})$$，$$\overline{x}\le x^{(1)} \Rightarrow f(x^{(1)})<f(x^{(2)})$$，则称$$f$$是闭区间$$[a,b]$$上的**单峰函数**。下图展示了一些示例：

![Screenshot from 2020-10-14 10-24-49.png](https://i.loli.net/2020/10/14/HOtb6RcsJ2xkhTC.png)

**定理** 设$$f$$是区间$$[a,b]$$上的单峰函数，$$a\le x^{(1)}< x^{(2)}\le b$$，则有$$f(x^{(1)})>f(x^{(2)}) \Rightarrow \forall x\in [a,x^{(1)}],f(x)>f(x^{(2)})$$，$$f(x^{(1)})\le f(x^{(2)}) \Rightarrow \forall x\in [x^{(2)},b],f(x)\ge f(x^{(1)})$$。



下面给出试探法的算法：

设$$f$$在区间$$[a_1,b_1]$$上单峰，极小点$$\overline{x}\in [a_1,b_1]$$，进行第$$k$$次迭代时有$$\overline{x}\in [a_k,b_k]$$。为缩小包含$$\overline{x}$$的区间，取两个试探点$$a_k\le \alpha_k<\beta_k\le b_k$$，计算函数值$$f(\alpha_k)$$和$$f(\beta_k)$$，

+ 若$$f(\alpha_k)>f(\beta_k)$$，根据上述定理，有$$\overline{x}\in [\alpha_k,b_k]$$，因此令
  $$
  a_{k+1}=\alpha_k,\ b_{k+1}=b_k
  $$

+ 若$$f(\alpha_k)\le f(\beta_k)$$，根据上述定理，有$$\overline{x}\in [a_k,\beta_k]$$，因此令

$$
  a_{k+1}=a_k,\ b_{k+1}=\beta_k
$$

如下图所示

![Screenshot from 2020-10-14 11-02-02.png](https://i.loli.net/2020/10/14/HA2YVsdej6FgmT3.png)

然后确定$$\alpha_k,\beta_k$$，为它们设定以下条件：
$$
b_k-\alpha_k=\beta_k-a_k\\
b_{k+1}-a_{k+1}=\lambda(b_k-a_k)
$$
解上述条件得
$$
\alpha_k=a_k+(1-\lambda)(b_k-a_k)\\
\beta_k=a_k+\lambda(b_k-a_k)
$$


### 0.618法

在上述条件下$$\alpha$$的值可以任意选择，选取合适的值可以使得每次迭代未使用的试探点可以直接作为下一次迭代的试探点之一。假设在第$$k$$次迭代有$$f(\alpha_k)> f(\beta_k)$$，那么$$a_{k+1}=\alpha_k,\ b_{k+1}=b_k$$，令
$$
\beta_k=\alpha_{k+1}\\
即\ a_k+\alpha(b_k-a_k)=b_k-\alpha(b_k-(b_k-\alpha(b_k-a_k)))\\
(\alpha^2+\alpha-1)(a_k-b_k)=0\\
\alpha = \frac{\sqrt{5}-1}{2}\approx 0.618
$$
类似地，当$$f(\alpha_k)\le f(\beta_k)$$时，令$$\alpha_k=\beta_{k+1}$$，同样求得$$\alpha=0.618$$。

因此0.618法计算试探点的公式为：
$$
\alpha_k=a_k+0.382(b_k-a_k)\\
\beta_k=a_k+0.618(b_k-a_k)
$$


@用0.618法解下列非线性规划问题：
$$
\min\ f(x)\triangleq 2x^2-x-1
$$
初始区间$$[a_1,b_1]=[-1,1]$$，精度$$L\le 0.16$$。

计算过程如下表：

![Screenshot from 2020-10-14 10-34-47.png](https://i.loli.net/2020/10/14/OuKgdcUNDypjVCM.png)

迭代6次后达到精度$$b_7-a_7=0.111<0.16$$，极小点$$\overline{x}\in[0.168,0.279]$$，可取$$\overline{x}=(0.168+0.279)/2\approx 0.23$$作为近似解。实际上的最优解$$x^*=0.25$$。



### Fibonacci法

Fibonacci法计算试探点的公式为：
$$
\alpha_k=a_k+\frac{F_{n-k-1}}{F_{n-k+1}} (b_k-a_k),\quad k=1,\cdots,n-1 \\
\beta_k=a_k+\frac{F_{n-k}}{F_{n-k+1}} (b_k-a_k),\quad k=1,\cdots,n-1
$$
其中$$F_i$$表示Fibonacci数列的第$$i$$项，$$F_0=F_1=1$$。

0.618法可以看作Fibonacci法的极限形式，因为
$$
\lim_{i\to \infty}\frac{F_{i-1}}{F_i}\approx0.618
$$
Fibonacci法的缺点是需要事先知道计算函数值的次数，因此实际应用中一般采用0.618法。



### 进退法

前面的方法都需要事先给定一个包含极小点的区间。**进退法**用于输入初点和步长，返回一个包含极小点的区间。

进退法的计算步骤略。



## 函数逼近法

### 牛顿法

牛顿法的基本思想是，在极小点附近用二阶Taylor多项式近似目标函数$$f(x)$$，进而求出极小点的估计值。

考虑问题
$$
\min\quad f(x),\ x\in\mathbb{R}
$$
令
$$
\varphi(x)=f(x^{(k)})+f'(x^{(k)})(x-x^{(k)})+\frac{1}{2}f''(x^{(k)})(x-x^{(k)})^2\\
$$
得到$$f(x)$$的近似函数，再令
$$
\varphi'(x)=f'(x^{(k)})+f''(x^{(k)})(x-x^{(k)})=0\\
\Rightarrow x^{(k+1)}=x^{(k)}-\frac{f'(x^{(k)})}{f''(x^{(k)})}
$$
得到$$\varphi(x)$$的驻点，亦即$$f(x)$$的极小点的估计。可以证明，在一定条件下，序列$$\{x^{(k)}\}$$收敛于上述问题的最优解。

**定理** 设$$f(x)$$存在连续三阶导数，$$\overline{x}$$满足
$$
f'(\overline{x})=0,\ f''(\overline{x})\neq 0
$$
初点$$x^{(1)}$$充分接近$$\overline{x}$$，则牛顿法产生的序列$$\{x^{(k)}\}$$至少以2级收敛速率收敛于$$\overline{x}$$。

牛顿法的初点选择十分重要，如果初始点接近极小点，则可能很快收敛；如果初始点远离极小点，迭代产生的点列可能不收敛于极小点。



牛顿法的计算步骤如下：

1. 给定初点$$x^{(0)}$$，允许误差$$\varepsilon>0$$，置$$k=0$$

2. 若$$|f'(x^{(k)}|<\varepsilon$$，则停止迭代，得到点$$x^{(k)}$$

3. 计算点
   $$
   x^{(k+1)}=x^k-\frac{f'(x^{(k)})}{f''(x^{(k)})}
   $$
   置$$k=k+1$$，goto 2。



@用牛顿法计算$$\sqrt 2$$的值。

设$$f(x)=x+\frac{2}{x},x>0$$，由集合不等式知$$f(x)\ge 2\sqrt 2$$，当且仅当$$x=\sqrt 2$$时取等。因此计算$$\sqrt 2$$转化为以下优化问题
$$
\begin{align}
\min &\quad f(x)=x+\frac{2}{x}\\
{\rm s.t.}&\quad x>0
\end{align}
$$
故
$$
f'(x)=1-\frac{2}{x^2},\ f''(x)=\frac{4}{x^3}
$$
给定初点1，允许误差$$0.0001$$，迭代过程为
$$
x^{(1)}=1-\frac{-1}{4}=1.25,\quad x^{(2)}=1.25-\frac{-0.28}{2.048}=1.38672,\\
x^{(3)}= 1.41342,\ x^{(4)}= 1.41421,\ x^{(5)}= 1.41421
$$
$$|x^{(5)}-x^{(4)}|<0.0001$$，停止迭代，计算结果即$$x^{(5)}= 1.41421$$。



### 割线法

割线法的基本思想是，在极小点附近用割线逼近目标函数的导函数的曲线$$y=f'(x)$$，把割线的零点作为目标函数的驻点的估计，如下图所示。

![Screenshot from 2020-10-14 14-30-58.png](https://i.loli.net/2020/10/14/2eJaXuOjWfrkRFx.png)

令
$$
\varphi(x)=f'(x^{(k)})+\frac{f'(x^{(k)})-f'(x^{(k-1)})}{x^{(k)}-x^{(k-1)}}(x-x^{(k)})=0
$$
解得
$$
x^{(k+1)}=x^{(k)}-\frac{x^{(k)}-x^{(k-1)}}{f'(x^{(k)})-f'(x^{(k-1)})}  f'(x^{(k)})
$$
即为$$f'(x)$$的零点的估计。可以证明，在一定条件下，序列$$\{x^{(k)}\}$$收敛于问题的最优解。

**定理** 设$$f(x)$$存在连续三阶导数，$$\overline{x}$$满足
$$
f'(\overline{x})=0,\ f''(\overline{x})\neq 0
$$
初点$$x^{(1)},x^{(2)}$$充分接近$$\overline{x}$$，则割线法产生的序列$$\{x^{(k)}\}$$至少1.618级收敛速率收敛于$$\overline{x}$$。

割线法与牛顿法相比，<u>收敛速率较慢</u>，但是<u>不需要计算二阶导数</u>。它的缺点与牛顿法相似，即不具有全局收敛性，如果初点选得不好则可能不收敛。



### 抛物线法

抛物线法的基本思想是，在极小点附近用二次多项式$$\varphi(x)$$近似目标函数$$f(x)$$，进而求出极小点的估计值。

具体步骤略。



### 三次插值法

抛物线法的基本思想是，在极小点附近用三次多项式$$\varphi(x)$$近似目标函数$$f(x)$$，进而求出极小点的估计值。

具体步骤略。

三次插值法的收敛级为2，一般认为三次插值法优于抛物线法。



### 有理插值法

具体步骤略。