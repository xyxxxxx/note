极限定理是概率论的基本理论，在理论研究和应用中有着重要作用。

## 大数定律LLN

**弱大数定律** 设 $X_1,X_2,\cdots$ 是相互独立，服从同一分布的随机变量序列，且具有数学期望 $E(X_i)=\mu(i=1,2,\cdots)$，作前n个变量的算数平均 $\frac{1}{n}\sum_{i=1}^{n}X_i$，则对于任意 $\varepsilon>0$，有
$$
\lim_{n\to \infty}P(\left | \frac{1}{n}\sum_{i=1}^nX_i-\mu \right |<\varepsilon)=1
$$

> 证明：使用切比雪夫不等式

**切比雪夫不等式**
$$
\forall \varepsilon>0,\ P(|X-\mu)|\ge \varepsilon)\le \frac{Var(X)}{\varepsilon^2}
$$

> 证明：以连续型随机变量 $X$ 为例
> $$
> Var(X)=\int_{-\infty}^{+\infty}(t-\mu)^2f_X(t){\rm d}t\\
> \ge \int_{-\infty}^{\mu-\varepsilon}(t-\mu)^2f_X(t){\rm d}t+\int_{\mu+\varepsilon}^{+\infty}(t-\mu)^2f_X(t){\rm d}t\\
> \ge \int_{-\infty}^{\mu-\varepsilon}\varepsilon^2f_X(t){\rm d}t+\int_{\mu+\varepsilon}^{+\infty}\varepsilon^2f_X(t){\rm d}t\\
> = \varepsilon^2(\int_{-\infty}^{\mu-\varepsilon}f_X(t){\rm d}t+\int_{\mu+\varepsilon}^{+\infty}f_X(t){\rm d}t)\\
> = \varepsilon^2P(X\le\mu-\varepsilon {\rm~or~} X\ge\mu+\varepsilon)\\
> = \varepsilon^2 P(|X-\mu|\ge \varepsilon)\\
> \therefore P(|X-\mu|\ge \varepsilon) \le \frac{Var(X)}{\varepsilon^2}
> $$

弱大数定律的含义是：对于独立同分布且具有均值 $\mu$ 的随机变量 $X_1,\cdots,X_n$，当 $n$ 很大时，它们的算术平均很可能接近 $\mu$。

## 中心极限定理CLT

**独立同分布的中心极限定理** 设随机变量 $X_1,X_2,\cdots,X_n,\cdots$ 独立同分布，且具有有限的数学期望和方差 $\mu,\sigma^2>0$，记 $\overline{X}=\frac{1}{n}\sum_{i=1}^{n}X_i$， $Y_n=\frac{\overline{X}-\mu}{\sigma/\sqrt{n}}$，则 $Y_n$ 的分布函数满足
$$
\lim_{n\to \infty}F_n(x)=\Phi(x)
$$
亦即近似有
$$
\overline{X} \sim N(\mu,\sigma^2/n)
$$

**李雅普诺夫中心极限定理** 设随机变量 $X_1,X_2,\cdots,X_n,\cdots$ 独立，且具有有限的数学期望和方差 $\mu_i,\sigma_i^2>0,\ i=1,2,\cdots,n,\cdots$，定义
$$
s_n^2=\sum_{i=1}^n\sigma_i^2
$$
若存在 $\delta>0$，使得李雅普诺夫条件
$$
\lim_{n\to \infty}\frac{1}{s_n^{2+\delta}}\sum_{i=1}^nE(|X_i-\mu_i|^{2+\delta})=0
$$

成立，记 $Y_n=\frac{1}{s_n}\sum_{i=1}^n(X_i-\mu_i)$，则 $Y_n$ 的分布函数满足

$$
\lim_{n\to \infty}F_n(x)=\Phi(x)
$$
亦即近似有
$$
\frac{1}{s_n}\sum_{i=1}^n(X_i-\mu_i) \sim N(0,1)
$$

