## 点估计point estimation

构造适当的统计量$$\hat\theta=\hat\theta(X_1,X_2,\cdots,X_n)$$，估计总体分布的参数$$\theta$$，就称$$\hat\theta$$为参数$$\theta$$的点估计。



### 矩估计法,method of moment

矩估计法的基本思想是，推导总体的矩（原点矩或中心矩），然后取出样本并计算样本的矩，由样本矩近似等于原点矩建立等式，解出需要估计的参数。



@设$$x_1,\cdots,x_n$$是从正态总体$$N(\mu,\sigma^2)$$抽出的样本，估计$$\mu,\sigma^2,\sigma$$。
$$
一阶原点矩\ \mu=\overline{x}\\
二阶中心矩\ \sigma^2=s^2\\
\sigma=s
$$


@设$$x_1,\cdots,x_n$$是从均匀总体$$U(a,b)$$抽出的样本，估计$$a,b$$。
$$
一阶原点矩\ \frac{a+b}{2}=\overline{x}\\
二阶中心矩\ \frac{(b-a)^2}{12}=s^2\\
\Rightarrow a=\overline{x}-\sqrt{3}s,\ b=\overline{x}+\sqrt{3}s
$$


@设$$x_1,\cdots,x_n$$是从泊松总体$$\pi(\lambda)$$抽出的样本，估计$$\lambda$$。
$$
一阶原点矩\ \lambda=\overline{x}\\
二阶原点矩\ \lambda=s^2\\
$$
两种方法都可以估计$$\lambda$$，但是哪种更好？一般而言优先使用低阶矩，具体原因将在后面介绍。



![padogji2ntj4rvbne](C:\Users\Xiao Yuxuan\Documents\pic\padogji2ntj4rvbne.PNG)

> 尽可能使用低阶矩

![ur13to4g2rjbhvwqde](C:\Users\Xiao Yuxuan\Documents\pic\ur13to4g2rjbhvwqde.PNG)

![acndvjeoyu6y52ktmrg](C:\Users\Xiao Yuxuan\Documents\pic\acndvjeoyu6y52ktmrg.PNG)



### 最大似然估计法maximum likelihood estimation,MLE

设总体分布$$f(X;\theta_1,\cdots,\theta_k)$$，抽取样本$$X_1,\cdots,X_n$$，那么样本$$(X_1,\cdots,X_n)$$的分布函数记作
$$
L(X_1,\cdots,X_n;\theta_1,\cdots,\theta_k)=f(X_1;\theta_1,\cdots,\theta_k)\cdots f(X_n;\theta_1,\cdots,\theta_k)
$$
$$L$$称为**似然函数**。如果我们把$$\theta_i$$看作参数，$$X_i$$看作变量，那么$$L$$是概率密度；但如果我们反过来把$$X_i$$看作参数，$$\theta_i$$看作变量，那么既然抽取到样本$$(X_1,\cdots,X_n)$$，我们认为$$L(X_1,\cdots,X_n;\theta_1,\cdots,\theta_k)$$的值是比较大的，于是转化为优化问题：$$\theta_1,\cdots,\theta_k$$取何值时$$L$$有最大值，这样的$$\theta_i^*$$称为$$\theta_i$$的**最大似然估计**，即最有可能的真参数值。这种推理类似于贝叶斯公式的思想，即倒果为因。

为求$$L$$的最大值，建立**似然方程组**
$$
\frac{\partial \log L}{\partial \theta_i}=0,\ i=1,\cdots,k
$$
如果该方程组有唯一解，并且解为极大值点，那么它必定是使$$L$$达到最大的点。有时该方程会出现几组解、难解或不能对$$\theta_i$$求偏导数，这时只能寻求另外的方法解此优化问题。



@设$$x_1,\cdots,x_n$$是从正态总体$$N(\mu,\sigma^2)$$抽出的样本，估计$$\mu,\sigma^2,\sigma$$。
$$
似然函数\ L=\prod_{i=1}^n((\sqrt{2\pi\sigma^2})^{-1}\exp(-\frac{1}{2\sigma^2}(x_i-\mu)^2))\\
\log L=-\frac{n}{2}\log(2\pi )-\frac{n}{2}\log(\sigma^2)-\frac{1}{2\sigma^2}\sum_{i=1}^n(x_i-\mu)^2\\
\frac{\partial \log L}{\partial \mu}=\frac{1}{\sigma^2}\sum_{i=1}^n(x_i-\mu)=0\\
\frac{\partial \log L}{\partial \sigma^2}=\frac{n}{2\sigma^2}+\frac{1}{2\sigma^4}\sum_{i=1}^n(x_i-\mu)^2=0\\
\Rightarrow \mu^*=\sum_{i=1}^n x_i/n=\overline{x}\\
\sigma^{2*}=\sum_{i=1}^n (x_i-\overline{x})/n=b_2\\
$$






> 以最大概率解释样本数据

![h359y0jihwgnbofjfqe](C:\Users\Xiao Yuxuan\Documents\pic\h359y0jihwgnbofjfqe.PNG)

![31rgrojthy3tnbe](C:\Users\Xiao Yuxuan\Documents\pic\31rgrojthy3tnbe.PNG)



![o2y40jiernvjdbqef](C:\Users\Xiao Yuxuan\Documents\pic\o2y40jiernvjdbqef.PNG)

![ijy396oihtnwjvfgtbw](C:\Users\Xiao Yuxuan\Documents\pic\ijy396oihtnwjvfgtbw.PNG)统计推断statistical inference
参数点估计,point estimation
构造适当的统计量，估计总体分布的参数，就称为参数的点估计

矩估计法,method of moment
![padogji2ntj4rvbne](C:\Users\Xiao Yuxuan\Documents\pic\padogji2ntj4rvbne.PNG)

尽可能使用低阶矩

![ur13to4g2rjbhvwqde](C:\Users\Xiao Yuxuan\Documents\pic\ur13to4g2rjbhvwqde.PNG)

![acndvjeoyu6y52ktmrg](C:\Users\Xiao Yuxuan\Documents\pic\acndvjeoyu6y52ktmrg.PNG)

 ![82ugijrwnvfsjpkbtwh](C:\Users\Xiao Yuxuan\Documents\pic\82ugijrwnvfsjpkbtwh.PNG)



![31rtgrmtkhnhrlgmt](C:\Users\Xiao Yuxuan\Documents\pic\31rtgrmtkhnhrlgmt.PNG)

![5i09yijhtnkwfeoprght](C:\Users\Xiao Yuxuan\Documents\pic\5i09yijhtnkwfeoprght.PNG)



### 点估计的无偏性和有效性

**无偏性**

![jy4hejtnjb52tgbt](C:\Users\Xiao Yuxuan\Documents\pic\jy4hejtnjb52tgbt.PNG)

**无偏校正**

![cdvnjbthwt4rgjbeyjdc](C:\Users\Xiao Yuxuan\Documents\pic\cdvnjbthwt4rgjbeyjdc.PNG)

**有效性**

![r31t48jigtnjbfvdnjb](C:\Users\Xiao Yuxuan\Documents\pic\r31t48jigtnjbfvdnjb.PNG)



![erqghji5u3nyhgbrfe](C:\Users\Xiao Yuxuan\Documents\pic\erqghji5u3nyhgbrfe.PNG)



### 实例

![940t1gjir2nbfvsdqef](C:\Users\Xiao Yuxuan\Documents\pic\940t1gjir2nbfvsdqef.PNG)



![vsfjbt3ijyngwrvbf](C:\Users\Xiao Yuxuan\Documents\pic\vsfjbt3ijyngwrvbf.PNG)

![i53hojtn3jrfrbvfgrw](C:\Users\Xiao Yuxuan\Documents\pic\i53hojtn3jrfrbvfgrw.PNG)

![h35860ji5tnrfgbuwgdca](C:\Users\Xiao Yuxuan\Documents\pic\h35860ji5tnrfgbuwgdca.PNG)

![98gr2hwvbfhgrfeqdac](C:\Users\Xiao Yuxuan\Documents\pic\98gr2hwvbfhgrfeqdac.PNG)



## 区间估计

![vfdbmgkrhy5t4fvfsdas](C:\Users\Xiao Yuxuan\Documents\pic\vfdbmgkrhy5t4fvfsdas.PNG)

**置信水平confidence level, 置信区间confidence interval,信頼区間**

![cvbgmrhpkuy53gwr](C:\Users\Xiao Yuxuan\Documents\pic\cvbgmrhpkuy53gwr.PNG)

![spdovjifogehy5t4nr](C:\Users\Xiao Yuxuan\Documents\pic\spdovjifogehy5t4nr.PNG)



### 单正态总体

$$\sigma$$已知$$\mu$$的置信区间
$$
枢轴量G=\frac{\overline{X}-\mu}{\sigma/\sqrt{n}}\sim N(0,1)
$$

![cavwfgehjiy35t4qrgw](C:\Users\Xiao Yuxuan\Documents\pic\cavwfgehjiy35t4qrgw.PNG)

![249t0gjirwvdnugrdw](C:\Users\Xiao Yuxuan\Documents\pic\249t0gjirwvdnugrdw.PNG)



$$\sigma$$未知$$\mu$$的置信区间
$$
\frac{\overline{X}-\mu}{S/\sqrt{n}}\sim t(n-1)
$$



$$\sigma$$未知$$\sigma^2$$的置信区间
$$
\frac{(n-1)S^2}{\sigma^2}\sim \chi^2(n-1)
$$

![t8rgihvubtvndeqfv](C:\Users\Xiao Yuxuan\Documents\pic\t8rgihvubtvndeqfv.PNG)



![t2y5gprjibntrfeqm](C:\Users\Xiao Yuxuan\Documents\pic\t2y5gprjibntrfeqm.PNG)

![t24ij5htnjbfvfeqcdvf](C:\Users\Xiao Yuxuan\Documents\pic\t24ij5htnjbfvfeqcdvf.PNG)



### 双正态总体

Behrens-Fisher问题

![vdbhg3htkj4yoyt4rfv](C:\Users\Xiao Yuxuan\Documents\pic\vdbhg3htkj4yoyt4rfv.PNG)



$$\mu_2-\mu_1:\sigma_1,\sigma_2$$已知情形

![czefojr13knjghtbgnyh](C:\Users\Xiao Yuxuan\Documents\pic\czefojr13knjghtbgnyh.PNG)



$$\mu_2-\mu_1:\sigma_1=\sigma_2$$情形

![zcxfqeojrwgithny351](C:\Users\Xiao Yuxuan\Documents\pic\zcxfqeojrwgithny351.PNG)

![hj4jp63i524ntrbjffeq](C:\Users\Xiao Yuxuan\Documents\pic\hj4jp63i524ntrbjffeq.PNG)



$$\sigma_2^2/\sigma_1^2$$

![zbfthkoj6nk3y5jhbt](C:\Users\Xiao Yuxuan\Documents\pic\zbfthkoj6nk3y5jhbt.PNG)



### 大样本置信区间

样本容量足够大（>30）时，根据中心极限定理，可以利用<u>渐进正态分布</u>构造置信区间

Behrens-Fisher问题

![zvxnkhty5t4kgrnkbbg](C:\Users\Xiao Yuxuan\Documents\pic\zvxnkhty5t4kgrnkbbg.PNG)



![y3k5ogmrwvnjfbthgrt](C:\Users\Xiao Yuxuan\Documents\pic\y3k5ogmrwvnjfbthgrt.PNG)

![2093rqifrnbgrfedvf](C:\Users\Xiao Yuxuan\Documents\pic\2093rqifrnbgrfedvf.PNG)

![lamfg35yjp24fnvfs](C:\Users\Xiao Yuxuan\Documents\pic\lamfg35yjp24fnvfs.PNG)

