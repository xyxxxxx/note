# 函数

$$
\lim_{n\to+\infty}(1+\frac{1}{n})^n=e\\
\lim_{n\to+\infty}(1+\frac{x}{n})^n=e^x\\
\sum_{k=0}^\infty\frac{x^k}{k!}=e^x
$$



# 数列

## 特殊数列

**等差数列**
$$
a_1=a,\;a_n=a_1+(n-1)d,\; S_n=na_1+\frac{n(n-1)}{2}d
$$
**等比数列**
$$
a_1=a,\;a_n=aq^{n-1}， S_n=a_1\frac{1-q^n}{1-q}
$$

**其它**
$$
1^2+2^2+3^2+\cdots+n^2=\frac{n(n+1)(2n+1)}{6}\\
1^3+2^3+3^3+\cdots+n^3=(\frac{n(n+1)}{2})^2
$$


## 通项公式求解

### **数学归纳法**(证明)

### **逐差**

### **逐商**

### **不动点法**

$$
Aa_{n+1}+Ba_n+C=0 \Rightarrow^{} A(a_{n+1}+k)=-B(a_n+k)\\
Aa_{n+1}+Ba_n+Ca_{n-1}+D=0 \Rightarrow Aa_{n+1}+Ea_n+F=k(Aa_{n}+Ea_{n-1}+F)
$$

**一阶常系数线性递推数列**
$$
a_{n+1}=pa_n+h \Rightarrow a_{n+1}+\frac{h}{p-1}=p(a_n+\frac{h}{p-1})\\
a_{n+1}=pa_n+hq^n \Rightarrow (两边同除p^{n+1}/q^{n+1}/不动点)
$$
**二阶常系数齐次线性递推数列**
$$
a_{n+2}=c_1a_{n+1}+c_2a_n\\
$$
其特征方程 $x^2=c_1x+c_2$ 的根为 $\lambda_1,\lambda_2$，如果 $\lambda_1 \neq \lambda_2$，
$$
\left\{ 
\begin{array}{c}
a_{n+2}-\lambda_1a_{n+1}=\lambda_2(a_{n+1}-\lambda_1a_n)\\
a_{n+2}-\lambda_2a_{n+1}=\lambda_1(a_{n+1}-\lambda_2a_n)
\end{array}
\right.
\\\Rightarrow
\left\{ 
\begin{array}{c}
a_{n+1}-\lambda_1a_{n}=\lambda_2^n(a_{1}-\lambda_1a_0)\\
a_{n+1}-\lambda_2a_{n}=\lambda_1^n(a_{1}-\lambda_2a_0)
\end{array}
\right.,消去a_n即得
$$
如果 $\lambda_1 = \lambda_2$，
$$
a_{n+2}-\lambda a_{n+1}=\lambda (a_{n+1}-\lambda a_n)
\\\Rightarrow a_{n+1}-\lambda a_{n}=\lambda^{n} (a_{1}-\lambda a_0),同除\lambda^{n}即得
$$
**分式线性递推数列**
$$
a_{n+1}=\frac{Aa_n+B}{Ca_n+D}
\\求特征方程\lambda=\frac{A\lambda+B}{C\lambda+D}的根，在递推两边减去，异根联立，重根进一步化简
$$



# 不等式

## 三角不等式

$$
\vert \vert a \vert -\vert b \vert \vert \le \vert a \pm b \vert \le\vert a \vert +\vert b \vert  
$$

## 平均数不等式

$$
\forall x_i>0,H_n \le G_n \le A_n \le Q_n
\\H_n=\frac{n}{\frac{1}{x_1}+\frac{1}{x_2}+\cdots+\frac{1}{x_n}}
\\G_n=\sqrt[n]{x_1x_2\cdots x_n}
\\A_n=\frac{x_1+x_2+\cdots+x_n}{n}
\\Q_n=\sqrt{\frac{x_1^2+x_2^2+\cdots+x_n^2}{n}}
$$

## 柯西不等式

$$
\sum x_i^2\sum y_i^2 \ge (\sum(x_iy_i))^2
$$


