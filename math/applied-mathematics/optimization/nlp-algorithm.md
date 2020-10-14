# 算法

前面讨论了最优性条件，理论上讲可以用这些条件求非线性规划的最优解，但在实际中往往并不可行，因为一般需要求解非线性方程组，这本来就是一个困难问题。因此求解非线性规划一般采用数值计算方法。

最常见的解非线性规划的算法是**迭代下降算法**。所谓**迭代**，就是从$$\pmb x^{(k)}$$出发，按照某种规则$$A$$求出后继点$$\pmb x^{(k+1)}$$，用$$k+1$$替代$$k$$，重复以上过程便产生点列$$\{\pmb x^{(k)}\}$$；所谓**下降**，就是每次迭代后函数值都有所减小。在一定条件下，迭代下降算法产生的点列收敛于原问题的解。

**算法映射** **算法**$$A$$是定义在空间$$X$$上的点到集映射，即对任意点$$\pmb x\in X$$，返回一个子集$$A(\pmb x)\sub X$$。

**解集合** 满足最终要求的点的集合称为**解集合**，当迭代点属于这个集合时就停止迭代。

**下降函数** 设解集合$$\Omega\sub X$$，算法$$A$$，$$\alpha(\pmb x)$$是定义在$$X$$上的连续实函数，若满足以下条件：

1. 当$$\pmb x\notin \Omega$$且$$\pmb y\in A(\pmb x)$$时，$$\alpha(\pmb y)<\alpha(\pmb x)$$
2. 当$$\pmb x\in \Omega$$且$$\pmb y\in A(\pmb x)$$时，$$\alpha(\pmb y)\le\alpha(\pmb x)$$

则称$$\alpha$$是关于解集合$$\Omega$$和算法$$A$$的**下降函数**。

一般地，当我们求解非线性规划问题时，通常取$$||\nabla f(\pmb x)||$$或$$f(\pmb x)$$作为下降函数。

**闭映射** 设$$X,Y$$分别是$$\mathbb{R}^p,\mathbb{R}^q$$中的非空闭集，$$A:X\to Y$$为点到集映射，如果
$$
\pmb x^{(k)}\in X,\ \pmb x^{(k)}\to \pmb x\\
\Rightarrow A(\pmb x^{(k)})\ni \pmb y^{(k)}\to \pmb y\in A(\pmb x)\\
$$
则称映射$$A$$在$$\pmb x\in X$$处是**闭的**。如果映射$$A$$在集合$$Z\sub X$$上的每一点是闭的，则称$$A$$在$$Z$$上是**闭的**。



## 算法收敛问题

### 收敛定理

**收敛** 设解集合$$\Omega$$，算法$$A:X\to X$$，若以任一初点$$\pmb x^{(1)}\in X$$开始，算法产生的序列的任一收敛子序列的极限属于$$\Omega$$，则称算法映射$$A$$在$$X$$上**收敛**。

**收敛定理** 设解集合$$\Omega$$，$$A$$是$$X$$上的算法，给定初点$$\pmb x^{(1)}\in X$$，进行如下迭代：

+ 若$$\pmb x^{(k)}\in \Omega$$，则停止迭代；否则令$$\pmb x^{(k+1)}\in A(\pmb x^{(k)})$$，将$$k+1$$替换为$$k$$，重复本操作

由此产生序列$$\{\pmb x^{(k)}\}$$，又设

1. 序列$$\{\pmb x^{(k)}\}$$包含于$$X$$的紧子集中
2. 存在一个连续函数$$\alpha$$，它是关于$$\Omega$$和$$A$$的下降函数
3. 映射$$A$$在$$\Omega$$的补集上是闭的

则序列$$\{\pmb x^{(k)}\}$$的任一收敛子序列的极限属于$$\Omega$$。



### 实用收敛准则

迭代下降算法中，当$$\pmb x^{(k)}\in \Omega$$才停止迭代，然而在实践过程中的很多情况下迭代永远不会停止。因此为了解决实际问题，需要规定一些实用的终止迭代的准则，称为**收敛准则**或**停步准则**。

常用的收敛准则有以下几种：

1. 当自变量的改变量充分小时，停止计算
   $$
   ||\pmb x^{(k+1)}-\pmb x^{(k)} ||<\varepsilon\\
   或\ \frac{||\pmb x^{(k+1)}-\pmb x^{(k)} ||}{||\pmb x^{(k)}||}<\varepsilon
   $$

2. 当函数值的下降量充分小时，停止计算
   $$
   |f(\pmb x^{(k+1)})-f(\pmb x^{(k)})|<\varepsilon\\
   或\ \frac{|f(\pmb x^{(k+1)})-f(\pmb x^{(k)})|}{|f(\pmb x^{(k)})|}<\varepsilon
   $$

3. 在无约束最优化中，当梯度充分接近零时，停止计算
   $$
   ||\nabla f(\pmb x^{(k)}) ||<\varepsilon
   $$
   

### 收敛速率

**收敛速率** 设序列$$\{\pmb \gamma^{(k)}\}$$收敛于$$\pmb \gamma^*$$，定义满足
$$
0\le \lim_{k\to \infty}\frac{||\pmb \gamma^{(k+1)}-\pmb \gamma^* ||}{||\pmb \gamma^{(k)}-\pmb \gamma^* ||^p}=\beta <\infty
$$
的非负数$$p$$的上确界为序列$$\{\pmb \gamma^{(k)}\}$$的**收敛级**，称序列是$$p$$级收敛的。若$$p=1,\beta<1$$则称序列是**以收敛比$$\beta$$线性收敛的**。



@考虑序列
$$
\{a^k\},\quad 0<a<1
$$
由于$$a^k\to 0$$以及
$$
\lim_{k\to \infty}\frac{a^{k+1}}{a^k}=a<1
$$
因此，序列$$\{a^k\}$$以收敛比$$a$$线性收敛于零。



@考虑序列
$$
\{a^{2^k}\},\quad 0<a<1
$$
由于$$a^{2^k}\to 0$$以及
$$
\lim_{k\to \infty}\frac{a^{2^{k+1}}}{(a^{2^k})^2}=1
$$
因此，序列$$\{a^{2^k}\}$$2级收敛。





# 一维搜索

求目标函数在直线上的极小点称为**一维搜索**，或称为**线搜索**。

设目标函数为$$f(\pmb x)$$，过点$$\pmb x^{(k)}$$沿方向$$\pmb d^{(k)}$$的直线可用点集表示，记作
$$
L=\{\pmb x|\pmb x=\pmb x^{(k)}+\lambda \pmb d^{(k)},-\infty<\lambda<\infty \}
$$
求$$f(\pmb x)$$在直线$$L$$上的极小点转化为求一元函数
$$
\varphi(\lambda)=f(\pmb x^{(k)}+\lambda \pmb d^{(k)})
$$
的极小点。

一维搜索的方法归纳起来，大体可分为两类：

+ **试探法** 这类方法需要按某种方式寻找试探点，通过一系列试探点来确定极小点
+ **函数逼近法/插值法** 这类方法用某种较简单的曲线逼近本来的函数曲线，通过求逼近函数的极小点来估计目标函数的极小点

这两类方法一般只能求得极小点的近似值。

**一维搜索的算法映射** 算法映射$$M:\mathbb{R}^n\times \mathbb{R}^n\to \mathbb{R}^n$$定义为
$$
M(\pmb x,\pmb d)=\{\pmb y|\pmb y= \pmb x+\overline{\lambda}\pmb d, \overline{\lambda}=\arg\min_{0\le \lambda <\infty}f(\pmb x+\lambda \pmb d) \}
$$
**定理** 设$$f$$是定义在$$\mathbb{R}^n$$上的连续函数，$$\pmb d\neq \pmb 0$$，则一维搜索的算法映射$$M$$在$$(\pmb x,\pmb d)$$处是闭的。



## 试探法

**单峰函数** 设$$f$$是定义在闭区间$$[a,b]$$上的一元实函数，$$\overline{x}$$是$$f$$在$$[a,b]$$上的极小点，并且$$\forall x^{(1)},x^{(2)}\in [a,b],x^{(1)}<x^{(2)}$$，都有$$x^{(2)}\le \overline{x}\Rightarrow f(x^{(1)})>f(x^{(2)})$$，$$\overline{x}\le x^{(1)} \Rightarrow f(x^{(1)})<f(x^{(2)})$$，则称$$f$$是闭区间$$[a,b]$$上的**单峰函数**。下图展示了一些示例：

![Screenshot from 2020-10-14 10-24-49.png](https://i.loli.net/2020/10/14/HOtb6RcsJ2xkhTC.png)

**定理** 设$$f$$是区间$$[a,b]$$上的单峰函数，$$a\le x^{(1)}< x^{(2)}\le b$$，则有$$f(x^{(1)})>f(x^{(2)}) \Rightarrow \forall x\in [a,x^{(1)}],f(x)>f(x^{(2)})$$，$$f(x^{(1)})\le f(x^{(2)}) \Rightarrow \forall x\in [x^{(2)},b],f(x)\ge f(x^{(1)})$$。



下面给出试探法的算法：

设$$f$$在区间$$[a_1,b_1]$$上单峰，极小点$$\overline{x}\in [a_1,b_1]$$，进行第$$k$$次迭代时有$$\overline{x}\in [a_k,b_k]$$。为缩小包含$$\overline{x}$$的区间，取两个试探点$$a_k\le \lambda_k<\mu_k\le b_k$$，计算函数值$$f(\lambda_k)$$和$$f(\mu_k)$$，

+ 若$$f(\lambda_k)>f(\mu_k)$$，根据上述定理，有$$\overline{x}\in [\lambda_k,b_k]$$，因此令
  $$
  a_{k+1}=\lambda_k,\ b_{k+1}=b_k
  $$
  
+ 若$$f(\lambda_k)\le f(\mu_k)$$，根据上述定理，有$$\overline{x}\in [a_k,\mu_k]$$，因此令
$$
  a_{k+1}=a_k,\ b_{k+1}=\mu_k
$$
如下图所示

![Screenshot from 2020-10-14 11-02-02.png](https://i.loli.net/2020/10/14/HA2YVsdej6FgmT3.png)

然后确定$$\lambda_k,\mu_k$$，为它们设定以下条件：
$$
b_k-\lambda_k=\mu_k-a_k\\
b_{k+1}-a_{k+1}=\alpha(b_k-a_k)
$$
解上述条件得
$$
\lambda_k=a_k+(1-\alpha)(b_k-a_k)\\
\mu_k=a_k+\alpha(b_k-a_k)
$$


### 0.618法

在上述条件下$$\alpha$$的值可以任意选择，选取合适的值可以使得每次迭代未使用的试探点可以直接作为下一次迭代的试探点之一。假设在第$$k$$次迭代有$$f(\lambda_k)> f(\mu_k)$$，那么$$a_{k+1}=\lambda_k,\ b_{k+1}=b_k$$，令
$$
\mu_k=\lambda_{k+1}\\
即\ a_k+\alpha(b_k-a_k)=b_k-\alpha(b_k-(b_k-\alpha(b_k-a_k)))\\
(\alpha^2+\alpha-1)(a_k-b_k)=0\\
\alpha = \frac{\sqrt{5}-1}{2}\approx 0.618
$$
类似地，当$$f(\lambda_k)\le f(\mu_k)$$时，令$$\lambda_k=\mu_{k+1}$$，同样求得$$\alpha=0.618$$。

因此0.618法计算试探点的公式为：
$$
\lambda_k=a_k+0.382(b_k-a_k)\\
\mu_k=a_k+0.618(b_k-a_k)
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
\lambda_k=a_k+\frac{F_{n-k-1}}{F_{n-k+1}} (b_k-a_k),\quad k=1,\cdots,n-1 \\
\mu_k=a_k+\frac{F_{n-k}}{F_{n-k+1}} (b_k-a_k),\quad k=1,\cdots,n-1
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





# 使用导数的最优化方法

考虑无约束问题
$$
\min\quad f(\pmb x),\quad \pmb x\in\mathbb{R}^n
$$
其中函数$$f(\pmb x)$$具有一阶连续偏导数。



## 梯度下降法

**梯度下降法**（也称**最速下降法**）的目标是从任意一点出发，总是选择一个使目标函数值下降最快的方法，以利于尽快到达极小点。



### 最速下降方向

可微函数$$f(\pmb x)$$在点$$\pmb x$$处沿方向$$\pmb d$$的变化率可用方向导数表示：
$$
{\rm D}f(\pmb x;\pmb d)=\pmb d^{\rm T} \nabla f(\pmb x)
$$
因此，求函数$$f(\pmb x)$$在点$$\pmb x$$处下降最快的方向，即求解下列非线性规划问题：
$$
\begin{align}
\min &\quad \pmb d^{\rm T} \nabla f(\pmb x)\\
{\rm s.t.}&\quad ||\pmb d||\le 1
\end{align}
$$
根据柯西不等式，有
$$
|\pmb d^{\rm T} \nabla f(\pmb x)|\le ||\nabla f(\pmb x)||\ ||\pmb d||\le ||\nabla f(\pmb x)||\\
\Rightarrow \pmb d^{\rm T} \nabla f(\pmb x)\ge -||\nabla f(\pmb x)||
$$
因此最优解为
$$
\pmb d=-\frac{\nabla f(\pmb x)}{||\nabla f(\pmb x)||}
$$
即在点$$\pmb x$$处<u>**负梯度方向**为**最速下降方向**</u>。

需要指出的是，上面定义的最速下降方向是在$$\ell_2$$范数$$||\pmb d||$$不大于1的限制下得到的，即欧氏度量意义下的最速下降方向。若改用其它度量，得到的最速下降方向与会有所不同。通常使用的最速下降法均指欧氏度量意义下的最速下降法。



### 梯度下降算法

梯度下降法的迭代公式是
$$
\pmb x^{(k+1)}=\pmb x^{(k)}+\lambda_k\pmb d^{(k)}
$$
其中$$\pmb d^{(k)}$$是从$$\pmb x^{(k)}$$出发的搜索方向，这里取最速下降方向，即
$$
\pmb d^{(k)}=-\nabla f(\pmb x^{(k)})
$$
$$\lambda_k$$是从$$\pmb x^{(k)}$$出发沿方向$$\pmb d^{(k)}$$进行一维搜索的步长，即$$\lambda_k$$满足
$$
f(\pmb x^{(k)}+\lambda_k\pmb d^{(k)})=\min_{\lambda \ge 0}f(\pmb x^{(k)}+\lambda\pmb d^{(k)})
$$
计算步骤如下：

1. 给定初点$$\pmb x^{(1)}\in\mathbb{R}^n$$，允许误差$$\varepsilon >0$$，置$$k=1$$

2. 计算搜索方向$$\pmb d^{(k)}=-\nabla f(\pmb x^{(k)})$$

3. 若$$||\pmb d^{(k)}||\le \varepsilon$$，则停止计算；否则从$$\pmb x^{(k)}$$出发，沿$$\pmb d^{(k)}$$进行一维搜索，求$$\lambda_k$$使得
   $$
   f(\pmb x^{(k)}+\lambda_k\pmb d^{(k)})=\min_{\lambda \ge 0}f(\pmb x^{(k)}+\lambda\pmb d^{(k)})
   $$

4. 令$$\pmb x^{(k+1)}=\pmb x^{(k)}+\lambda_k\pmb d^{(k)}$$，置$$k=k+1$$，goto 2



@用梯度下降法解下列非线性规划问题：
$$
\min\quad f(\pmb x)=2x_1^2+x_2^2
$$
初点$$\pmb x^{(1)}=(1,1)^{\rm T},\varepsilon = 0.1$$。

计算梯度
$$
\nabla f(\pmb x)=\begin{bmatrix}4x_1\\2x_2 \end{bmatrix}
$$
第1次迭代，
$$
\pmb d^{(1)}=-\nabla f(\pmb x^{(1)})=\begin{bmatrix}-4\\-2 \end{bmatrix},\ ||\pmb d^{(1)}||=2\sqrt{5}>0.1\\
\min_{\lambda\ge 0}\quad\varphi (\lambda)\triangleq f(\pmb x^{(1)}+\lambda\pmb d^{(1)})=f(\begin{bmatrix}1-4\lambda\\1-2\lambda \end{bmatrix})=2(1-4\lambda)^2+(1-2\lambda)^2\\
令\varphi'(\lambda)=-16(1-4\lambda)-4(1-2\lambda)=0\Rightarrow
\lambda_1=\frac{5}{18}\\
\pmb x^{(2)}=\pmb x^{(1)}+\lambda_1\pmb d^{(1)}=\begin{bmatrix}-1/9 \\4/9 \end{bmatrix}
$$

> 这里由于$$\varphi(\lambda)$$的极小值可以直接求得，因此不需要试探法或插值法。

类似地，第2, 3次迭代，
$$
\pmb x^{(3)}=\pmb x^{(2)}+\lambda_2\pmb d^{(2)}=\frac{2}{27}\begin{bmatrix}1\\1 \end{bmatrix}\\
\pmb x^{(4)}=\pmb x^{(3)}+\lambda_3\pmb d^{(3)}=\frac{2}{243}\begin{bmatrix}-1\\4 \end{bmatrix}\\
$$
达到精度要求$$||\nabla f(\pmb x^{(4)})||=\frac{8}{243}\sqrt{5}<0.1$$，于是近似解$$\overline{\pmb x}=\frac{2}{243}(-1,4)^{\rm T}$$。实际上，问题的最优解为$$\pmb x^*=(0,0)^{\rm T}$$。



**定理** 设$$f(\pmb x)$$是连续可微实函数，解集合$$\Omega=\{\overline{\pmb x}|\nabla f(\overline{\pmb x})=\pmb 0 \}$$，最速下降算法产生的序列$$\{\pmb x^{(k)}\}$$包含于某个紧集，则序列$$\{\pmb x^{(k)}\}$$的每个聚点$$\hat{\pmb x}\in \Omega$$。

**定理** 设$$f(\pmb x)$$存在连续二阶偏导数，$$\overline{\pmb x}$$是局部极小点，Hessian矩阵$$\nabla^2 f(\overline{\pmb x})$$的最小特征值$$a>0$$，最大特征值为$$A$$，算法产生的序列$$\{\pmb x^{(k)}\}$$收敛于点$$\overline{\pmb x}$$，则目标函数值的序列$$\{f(\pmb x^{(k)})\}$$以不大于
$$
(\frac{A-a}{A+a})^2
$$
的收敛比线性地收敛于$$f(\overline{\pmb x})$$。

**条件数** 在以上定理中，令$$r=A/a$$，则
$$
(\frac{A-a}{A+a})^2=(\frac{r-1}{r+1})^2<1
$$
$$r$$称为对称正定矩阵$$\nabla^2 f(\overline{\pmb x})$$的**条件数**。以上定理表明，<u>条件数越小，收敛越快；条件数越大，收敛越慢</u>。

**锯齿现象** 用梯度下降法极小化目标函数时，相邻的两个搜索方向是正交的，因为
$$
\varphi(\lambda)= f(\pmb x^{(k)}+\lambda\pmb d^{(k)}), \\
\pmb d^{(k)}=-\nabla f(\pmb x^{(k)}), \\
令\ \varphi'(\lambda)=\pmb d^{(k){\rm T}} \nabla f(\pmb x^{(k)}+\lambda\pmb d^{(k)})=0\Rightarrow \lambda=\lambda_k \\
\Rightarrow \pmb d^{(k){\rm T}}\nabla f(\pmb x^{(k+1)})=0\Rightarrow -\pmb d^{(k){\rm T}}\pmb d^{(k+1)}=0
$$
即方向$$\pmb d^{(k)},\pmb d^{(k+1)}$$正交，这表明迭代产生的序列$$\{\pmb x^{(k)}\}$$所循路径是“之”字形的，如下图所示。特别是当$$\pmb x^{(k)}$$接近极小点$$\overline{\pmb x}$$时，每次迭代移动的步长很小，于是出现了**锯齿现象**，影响了收敛速率。

![Screenshot from 2020-10-14 18-28-01.png](https://i.loli.net/2020/10/14/9JyA4CnPRMmvq5p.png)

当条件数比较大时，锯齿现象的影响尤为严重，原因略。



从局部看，最速下降方向是函数值下降最快的方向，选择这样的方向进行搜索是有利的；但从全局看，由于锯齿现象的影响，收敛速率大为减慢。梯度下降法并不是收敛最快的方法，相反，从全局看，它的收敛是比较慢的。因此梯度下降法一般适用于计算过程的前期迭代，而不适用于后期接近极小点的情形。



## 牛顿法

这里将一维搜索中的牛顿法推广到求解一般无约束问题的牛顿法。

设$$f(\pmb x)$$是二次可微实函数，$$\pmb x\in\mathbb{R}^n$$，又设$$\pmb x^{(k)}$$是$$f(\pmb x)$$的极小点的一个估计，把$$f(\pmb x)$$在$$\pmb x^{(k)}$$展成 Taylor 级数，并取二阶近似
$$
f(\pmb x)\approx \phi(\pmb x)=f(\pmb x^{(k)})+\nabla f(\pmb x^{(k)})^{\rm T}(\pmb x-\pmb x^{(k)})+\frac{1}{2}(\pmb x-\pmb x^{(k)})^{\rm T}\nabla^2 f(\pmb x^{(k)})(\pmb x-\pmb x^{(k)})
$$
其中$$\nabla^2 f(\pmb x^{(k)})$$是 Hessian 矩阵。为求$$\phi(\pmb x)$$的驻点，令
$$
\nabla\phi(\pmb x)=\pmb 0\\
即\ \nabla f(\pmb x^{(k)})+\nabla^2 f(\pmb x^{(k)})(\pmb x-\pmb x^{(k)})=\pmb 0
$$
设$$\nabla^2 f(\pmb x^{(k)})$$可逆，则得到牛顿法的迭代公式
$$
\pmb x^{(k+1)}=\pmb x^{(k)}-(\nabla^2 f(\pmb x^{(k)}))^{-1}\nabla f(\pmb x^{(k)})
$$
**定理** 设$$f(\pmb x)$$为二次连续可微函数，$$\pmb x\in\mathbb{R}^n$$，







## 共轭梯度法



