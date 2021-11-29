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



 



# 假设检验

**零假设,null hypothesis,帰無仮説，备择假设，显著性水平significance level有意水準，检验统计量**

基本步骤

1. 建立假设 $H_0:\theta\in \Theta_0,H_1:\theta\in \Theta_1$ 
2. 选择检验统计量，给出拒绝域W的形式（接受域 $\overline{W}$ 
3. 选择显著性水平 $\alpha$， $\alpha=P(拒绝H_0|H_0为真)=P_\theta(X\in W)$ 

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
+ 犯第一类错误的概率不高于显著性水平 $\alpha$ 
+ 显著性检验具有“保护原假设”的特点
+ 增加样本量可以降低犯第二类错误的概率

![zxcsfktym5t4rgbhet](C:\Users\Xiao Yuxuan\Documents\pic\zxcsfktym5t4rgbhet.PNG)

![acifeot4gbngofefwgrb](C:\Users\Xiao Yuxuan\Documents\pic\acifeot4gbngofefwgrb.PNG)



**p值,probability value** 在<u>原假设</u>下，出现<u>观测值或比观测值更极端</u>的概率

**p值检验法**

1. 若p值 $\le \alpha$，则在显著性水平 $\alpha$ 下拒绝 $H_0$ 
2. 若p值 $> \alpha$，则在显著性水平 $\alpha$ 下接受 $H_0$ 



## 单个正态总体的假设检验

![dvsfdwr6u4hymngre](C:\Users\Xiao Yuxuan\Documents\pic\dvsfdwr6u4hymngre.PNG)



### $\sigma$ 已知：Z检验

$$
Z=\frac{\overline{X}-\mu_0}{\sigma/\sqrt{n}}
$$



![cznofwejrgthjnej2t](C:\Users\Xiao Yuxuan\Documents\pic\cznofwejrgthjnej2t.PNG)





![cjiovetyn53gvfdscv](C:\Users\Xiao Yuxuan\Documents\pic\cjiovetyn53gvfdscv.PNG)

![gijbejt4j2igrbfwefv](C:\Users\Xiao Yuxuan\Documents\pic\gijbejt4j2igrbfwefv.PNG)

![zdczvnkgt4njborhtb](C:\Users\Xiao Yuxuan\Documents\pic\zdczvnkgt4njborhtb.PNG)



### $\sigma$ 未知：t检验

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





# 相关性和回归分析

## 相关性



## 线性回归



| 年份 | 国内私人汽车保有量(万辆) | 国内人均家庭支出(美元) |
| ---- | ------------------------ | ---------------------- |
| 2011 | 7326.7938                | 2406.523               |
| 2012 | 8838.6014                | 2711.693               |
| 2013 | 10501.6827               | 3007.066               |
| 2014 | 12339.3597               | 3241.226               |
| 2015 | 14099.1037               | 3404.792               |
| 2016 | 16330.2248               | 3474.861               |
| 2017 | 18515.1085               | 3618.053               |
| 2018 | 20574.9339               | 3949.071               |
| 2019 | 22513.40                 | 4062.960               |

> 数据来源https://www.ceicdata.com/zh-hans/china/no-of-motor-vehicle-private-owned/motor-vehicle-owned-private-total
>
> https://data.stats.gov.cn/search.htm?s=%E7%A7%81%E4%BA%BA%E6%B1%BD%E8%BD%A6%E6%8B%A5%E6%9C%89%E9%87%8F
>
> https://www.ceicdata.com/zh-hans/indicator/china/annual-household-expenditure-per-capita





## 非线性回归