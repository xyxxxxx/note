# 统计学基础statistics

> 概率论vs统计学：
>
> 概率论是数学，基本特征是 **从法则到结果**
>
> 统计学是单独学科，其基本特征是 **从结果到法则**
>
> ![t427ubvftgkjbnlwr](C:\Users\Xiao Yuxuan\Documents\pic\t427ubvftgkjbnlwr.PNG)
>
> ![902tigonvfjbfqeef](C:\Users\Xiao Yuxuan\Documents\pic\902tigonvfjbfqeef.PNG)
>
> ![9130rjfgrijnfedbfvhr](C:\Users\Xiao Yuxuan\Documents\pic\9130rjfgrijnfedbfvhr.PNG)



## 总体和样本

> 统计学研究中，总是假定总体服从某种分布

**总体population，个体individuality，样本sample標本，简单随机样本simple random sample無作為標本，样品，样本容量**

样本为随机变量，用$X_1,X_2,\cdots,X_n$表示；样本在抽取后的确定观测值用$x_1,x_2,\cdots,x_n$



## 常用统计量

**统计量** $X_1,X_2,\cdots,X_n$为取自某总体的样本，若样本函数$T=T(X_1,X_2,\cdots,X_n)$中不含有任何位置参数，则T称为统计量. 统计量的分布称为抽样分布

**样本均值sample mean標本平均 $\overline{X}$**

+ 若总体$X\sim N(\mu,\sigma^2)$，则$\overline{X}\sim N(\mu,\frac{\sigma^2}{n})$
+ 若总体不服从正态分布，$E(X)=\mu,Var(X)=\sigma^2$，则由中心极限定理，n较大时$\overline{X}$渐进分布为$N(\mu,\frac{\sigma^2}{n})$

**样本方差sample variance $S^2$**
$$
S^2=\frac{1}{n-1}\sum_{i=1}^n(X_i-\overline{X})^2
$$
**定理** 设总体X具有二阶中心矩，$E(X)=\mu$，$Var(X)=\sigma^2<+\infty$，$X_1,X_2,\cdots,X_n$为样本，$\overline{X}和S^2$分别是样本均值和样本方差，则$E(S^2)=\sigma^2$. 样本方差是总体方差的无偏估计，样本均值是总体均值的无偏估计.

**样本k阶原点矩sample moment of order k about the origin $a_k$**
$$
a_k=\frac{1}{n}\sum_{i=1}^nX_i^k
$$
**样本k阶中心矩sample central moment of order k $b_k$**
$$
b_k=\frac{1}{n}\sum_{i=1}^n(X_i-\overline{X})^k
$$



## 常用统计分布

**$\chi^2$分布,カイ二乗分布** $X_1,X_2,\cdots,X_n$相互独立且服从标准正态分布，则随机变量
$$
Y=X_1^2+\cdots+X_n^2
$$
的分布称为自由度为n的$\chi^2$分布，记作$Y\sim\chi^2(n)$

$X\sim\chi^2(n),则E(X)=n,Var(X)=2n$

$X_1\sim\chi^2(m),X_2\sim\chi^2(n)，则X_1+X_2\sim\chi^2(m+n)$

**定理** $X_1,X_2,\cdots,X_n$是来自正态总体$N(\mu,\sigma^2)$的样本，则

1. $\overline{X}和S^2$相互独立
2. $\overline{X}\sim N(\mu,\sigma^2/n)$
3. $\frac{(n-1)S^2}{\sigma^2}\sim\chi^2(n-1)$



**t分布** $X_1\sim N(0,1),X_2\sim \chi^2(n),X_1,X_2$相互独立，则随机变量
$$
Y=\frac{X_1}{\sqrt{X_2/n}}
$$
的分布称为自由度为n的t分布，记作$Y\sim t(n)$

> t>30时，t分布基本等同于正态分布

**F分布** $X_1\sim \chi^2(m),X_2\sim \chi^2(n),X_1,X_2$相互独立，则随机变量
$$
Y=\frac{X_1/m}{X_2/n}
$$
的分布称为自由度为m与n的F分布，记作$Y\sim F(m,n)$



**分位点** 连续型随机变量的分布函数$F(x)$，$F(a)=P(X\le a)=\alpha$，称a为该分布的$\alpha$分位点





# 参数点估计,point estimation

构造适当的统计量$\hat\theta=\hat\theta(X_1,X_2,\cdots,X_n)$，估计总体分布的参数$\theta$，就称$\hat\theta$为参数$\theta$的点估计



## 矩估计法,method of moment

![padogji2ntj4rvbne](C:\Users\Xiao Yuxuan\Documents\pic\padogji2ntj4rvbne.PNG)

> 尽可能使用低阶矩

![ur13to4g2rjbhvwqde](C:\Users\Xiao Yuxuan\Documents\pic\ur13to4g2rjbhvwqde.PNG)

![acndvjeoyu6y52ktmrg](C:\Users\Xiao Yuxuan\Documents\pic\acndvjeoyu6y52ktmrg.PNG)



## 极大似然估计法maximum likelihood estimation,MLE

> 以最大概率解释样本数据

极大似然估计量,maximum-likelihood estimator,最尤推定量

![h359y0jihwgnbofjfqe](C:\Users\Xiao Yuxuan\Documents\pic\h359y0jihwgnbofjfqe.PNG)

![31rgrojthy3tnbe](C:\Users\Xiao Yuxuan\Documents\pic\31rgrojthy3tnbe.PNG)



![o2y40jiernvjdbqef](C:\Users\Xiao Yuxuan\Documents\pic\o2y40jiernvjdbqef.PNG)

![ijy396oihtnwjvfgtbw](C:\Users\Xiao Yuxuan\Documents\pic\ijy396oihtnwjvfgtbw.PNG)

 ![82ugijrwnvfsjpkbtwh](C:\Users\Xiao Yuxuan\Documents\pic\82ugijrwnvfsjpkbtwh.PNG)



![31rtgrmtkhnhrlgmt](C:\Users\Xiao Yuxuan\Documents\pic\31rtgrmtkhnhrlgmt.PNG)

![5i09yijhtnkwfeoprght](C:\Users\Xiao Yuxuan\Documents\pic\5i09yijhtnkwfeoprght.PNG)



## 点估计的无偏性和有效性

**无偏性**

![jy4hejtnjb52tgbt](C:\Users\Xiao Yuxuan\Documents\pic\jy4hejtnjb52tgbt.PNG)

**无偏校正**

![cdvnjbthwt4rgjbeyjdc](C:\Users\Xiao Yuxuan\Documents\pic\cdvnjbthwt4rgjbeyjdc.PNG)

**有效性**

![r31t48jigtnjbfvdnjb](C:\Users\Xiao Yuxuan\Documents\pic\r31t48jigtnjbfvdnjb.PNG)



![erqghji5u3nyhgbrfe](C:\Users\Xiao Yuxuan\Documents\pic\erqghji5u3nyhgbrfe.PNG)



## 实例

![940t1gjir2nbfvsdqef](C:\Users\Xiao Yuxuan\Documents\pic\940t1gjir2nbfvsdqef.PNG)



![vsfjbt3ijyngwrvbf](C:\Users\Xiao Yuxuan\Documents\pic\vsfjbt3ijyngwrvbf.PNG)

![i53hojtn3jrfrbvfgrw](C:\Users\Xiao Yuxuan\Documents\pic\i53hojtn3jrfrbvfgrw.PNG)

![h35860ji5tnrfgbuwgdca](C:\Users\Xiao Yuxuan\Documents\pic\h35860ji5tnrfgbuwgdca.PNG)

![98gr2hwvbfhgrfeqdac](C:\Users\Xiao Yuxuan\Documents\pic\98gr2hwvbfhgrfeqdac.PNG)





# 参数区间估计method of interval

![vfdbmgkrhy5t4fvfsdas](C:\Users\Xiao Yuxuan\Documents\pic\vfdbmgkrhy5t4fvfsdas.PNG)

**置信水平confidence level, 置信区间confidence interval,信頼区間**

![cvbgmrhpkuy53gwr](C:\Users\Xiao Yuxuan\Documents\pic\cvbgmrhpkuy53gwr.PNG)

![spdovjifogehy5t4nr](C:\Users\Xiao Yuxuan\Documents\pic\spdovjifogehy5t4nr.PNG)



## 单正态总体

### $\sigma$已知$\mu$的置信区间

$$
枢轴量G=\frac{\overline{X}-\mu}{\sigma/\sqrt{n}}\sim N(0,1)
$$

![cavwfgehjiy35t4qrgw](C:\Users\Xiao Yuxuan\Documents\pic\cavwfgehjiy35t4qrgw.PNG)

![249t0gjirwvdnugrdw](C:\Users\Xiao Yuxuan\Documents\pic\249t0gjirwvdnugrdw.PNG)



### $\sigma$未知$\mu$的置信区间

$$
\frac{\overline{X}-\mu}{S/\sqrt{n}}\sim t(n-1)
$$



### $\sigma$未知$\sigma^2$的置信区间

$$
\frac{(n-1)S^2}{\sigma^2}\sim \chi^2(n-1)
$$

![t8rgihvubtvndeqfv](C:\Users\Xiao Yuxuan\Documents\pic\t8rgihvubtvndeqfv.PNG)



![t2y5gprjibntrfeqm](C:\Users\Xiao Yuxuan\Documents\pic\t2y5gprjibntrfeqm.PNG)

![t24ij5htnjbfvfeqcdvf](C:\Users\Xiao Yuxuan\Documents\pic\t24ij5htnjbfvfeqcdvf.PNG)



## 双正态总体

### Behrens-Fisher问题

![vdbhg3htkj4yoyt4rfv](C:\Users\Xiao Yuxuan\Documents\pic\vdbhg3htkj4yoyt4rfv.PNG)



### $\mu_2-\mu_1:\sigma_1,\sigma_2$已知情形

![czefojr13knjghtbgnyh](C:\Users\Xiao Yuxuan\Documents\pic\czefojr13knjghtbgnyh.PNG)



### $\mu_2-\mu_1:\sigma_1=\sigma_2$情形

![zcxfqeojrwgithny351](C:\Users\Xiao Yuxuan\Documents\pic\zcxfqeojrwgithny351.PNG)

![hj4jp63i524ntrbjffeq](C:\Users\Xiao Yuxuan\Documents\pic\hj4jp63i524ntrbjffeq.PNG)



### $\sigma_2^2/\sigma_1^2$

![zbfthkoj6nk3y5jhbt](C:\Users\Xiao Yuxuan\Documents\pic\zbfthkoj6nk3y5jhbt.PNG)



## 大样本置信区间

样本容量足够大（>30）时，根据中心极限定理，可以利用<u>渐进正态分布</u>构造置信区间

### Behrens-Fisher问题

![zvxnkhty5t4kgrnkbbg](C:\Users\Xiao Yuxuan\Documents\pic\zvxnkhty5t4kgrnkbbg.PNG)



![y3k5ogmrwvnjfbthgrt](C:\Users\Xiao Yuxuan\Documents\pic\y3k5ogmrwvnjfbthgrt.PNG)

![2093rqifrnbgrfedvf](C:\Users\Xiao Yuxuan\Documents\pic\2093rqifrnbgrfedvf.PNG)

![lamfg35yjp24fnvfs](C:\Users\Xiao Yuxuan\Documents\pic\lamfg35yjp24fnvfs.PNG)

 



# 假设检验

**零假设,null hypothesis,帰無仮説，备择假设，显著性水平significance level有意水準，检验统计量**

基本步骤

1. 建立假设$H_0:\theta\in \Theta_0,H_1:\theta\in \Theta_1$
2. 选择检验统计量，给出拒绝域W的形式（接受域$\overline{W}$
3. 选择显著性水平$\alpha$，$\alpha=P(拒绝H_0|H_0为真)=P_\theta(X\in W)$

![difee3rgfvbgnjyrhd](C:\Users\Xiao Yuxuan\Documents\pic\difee3rgfvbgnjyrhd.PNG)

![mbgrwtbeojtt4rgebfvdc](C:\Users\Xiao Yuxuan\Documents\pic\mbgrwtbeojtt4rgebfvdc.PNG)



**单边检验，双边检验**
$$
双边检验\quad H_0:\mu=\mu_0,H_1:\mu\neq\mu_0\\
单边检验\quad H_0:\mu\le\mu_0,H_1:\mu>\mu_0\\
$$




## 两类错误

![cxdopfjwrig4ntrjvf](C:\Users\Xiao Yuxuan\Documents\pic\cxdopfjwrig4ntrjvf.PNG)

|            | 合格产品          | 不合格产品 |
| ---------- | ----------------- | ---------- |
| 检验合格   |                   | 第二类错误 |
| 检验不合格 | <u>第一类错误</u> |            |

+ 假设检验中两类错误的概率<u>不能同时减小</u>
+ 原假设是经验上认为正常的假设
+ 理想的检验应该是，在控制犯第一类错误的基础上尽量少犯第二类错误
+ 犯第一类错误的概率不高于显著性水平$\alpha$
+ 显著性检验具有“保护原假设”的特点
+ 增加样本量可以降低犯第二类错误的概率

![zxcsfktym5t4rgbhet](C:\Users\Xiao Yuxuan\Documents\pic\zxcsfktym5t4rgbhet.PNG)

![acifeot4gbngofefwgrb](C:\Users\Xiao Yuxuan\Documents\pic\acifeot4gbngofefwgrb.PNG)



**p值,probability value** 在<u>原假设</u>下，出现<u>观测值或比观测值更极端</u>的概率

**p值检验法**

1. 若p值$\le \alpha$，则在显著性水平$\alpha$下拒绝$H_0$
2. 若p值$> \alpha$，则在显著性水平$\alpha$下接受$H_0$



## 单个正态总体的假设检验

![dvsfdwr6u4hymngre](C:\Users\Xiao Yuxuan\Documents\pic\dvsfdwr6u4hymngre.PNG)



### $\sigma$已知：Z检验

$$
Z=\frac{\overline{X}-\mu_0}{\sigma/\sqrt{n}}
$$



![cznofwejrgthjnej2t](C:\Users\Xiao Yuxuan\Documents\pic\cznofwejrgthjnej2t.PNG)





![cjiovetyn53gvfdscv](C:\Users\Xiao Yuxuan\Documents\pic\cjiovetyn53gvfdscv.PNG)

![gijbejt4j2igrbfwefv](C:\Users\Xiao Yuxuan\Documents\pic\gijbejt4j2igrbfwefv.PNG)

![zdczvnkgt4njborhtb](C:\Users\Xiao Yuxuan\Documents\pic\zdczvnkgt4njborhtb.PNG)



### $\sigma$未知：t检验

![zxeqjirwgon524itjnjgb](C:\Users\Xiao Yuxuan\Documents\pic\zxeqjirwgon524itjnjgb.PNG)



## 拟合优度检验

### 卡方检验

![dfvjngt54jigrnbr](C:\Users\Xiao Yuxuan\Documents\pic\dfvjngt54jigrnbr.PNG)

 

![e24tigrjthnj63yhtnb](C:\Users\Xiao Yuxuan\Documents\pic\e24tigrjthnj63yhtnb.PNG)

![sfgthmyk523uh5tbgwrf](C:\Users\Xiao Yuxuan\Documents\pic\sfgthmyk523uh5tbgwrf.PNG)



![09ijouytresxcyuf](C:\Users\Xiao Yuxuan\Documents\pic\09ijouytresxcyuf.PNG)

![ohuyfr7s486ct9ugb](C:\Users\Xiao Yuxuan\Documents\pic\ohuyfr7s486ct9ugb.PNG)

![nojcryd57v9yctu](C:\Users\Xiao Yuxuan\Documents\pic\nojcryd57v9yctu.PNG)



### 基于卡方分布的独立性检验

**列联表**

![jifdvnjwr4tgbewqfevd](C:\Users\Xiao Yuxuan\Documents\pic\jifdvnjwr4tgbewqfevd.PNG)



![y8ig7vy8ze5s6d85r](C:\Users\Xiao Yuxuan\Documents\pic\y8ig7vy8ze5s6d85r.PNG)

![dfjigyt4irgtjehbntgrwf](C:\Users\Xiao Yuxuan\Documents\pic\dfjigyt4irgtjehbntgrwf.PNG)



![vnojgt5grwbnjvqer](C:\Users\Xiao Yuxuan\Documents\pic\vnojgt5grwbnjvqer.PNG)



网球比赛胜率计算

![gy5t2j4ionjh35yt42](C:\Users\Xiao Yuxuan\Documents\pic\gy5t2j4ionjh35yt42.PNG)







