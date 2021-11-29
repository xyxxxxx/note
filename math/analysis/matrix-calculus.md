> 所有证明省略，仅供应用参考
>
> 所有矩阵用非粗体的大写字母表示

> [在线计算矩阵导数工具](http://www.matrixcalculus.org/)




# 导数

## 标量、向量与矩阵之间的导数

**标量关于向量的偏导数** 对于M维向量 $\pmb x\in \mathbb{R}^M$ 和函数 $y=f(\pmb x)\in \mathbb{R}$， $y$ 关于 $\pmb x$ 的偏导数为
$$
\frac{\partial y}{\partial \pmb x}=[\frac{\partial y}{\partial  x_1},\cdots,\frac{\partial y}{\partial x_M}]^{\rm T}
$$
> 多元微积分中的梯度即是标量对向量求偏导数

$y$ 关于 $\pmb x$ 的二阶偏导数为
$$
H=\frac{\partial^2 y}{\partial \pmb x^2}=\begin{bmatrix} \frac{\partial^2 y}{\partial x_1^2} & \cdots & \frac{\partial^2 y}{\partial  x_1 \partial x_M}\\
\vdots & \ddots & \vdots \\
\frac{\partial^2 y}{\partial x_M \partial x_1} & \cdots & \frac{\partial^2 y}{\partial  x_M^2}
\end{bmatrix}
\in \mathbb{R}^{M\times M}
$$
称为函数 $f(\pmb x)$ 的**Hessian矩阵**，也写作 $\nabla^2 f(\pmb x)$ 

**向量关于标量的偏导数** 对于标量 $x\in \mathbb{R}$ 和函数 $\pmb y=f(x)\in \mathbb{R}^N$， $\pmb y$ 关于 $x$ 的偏导数为
$$
\frac{\partial \pmb y}{\partial x}=[\frac{\partial y_1}{\partial  x},\cdots,\frac{\partial y_N}{\partial x}]
$$
**向量关于向量的偏导数** 对于M维向量 $\pmb x\in \mathbb{R}^M$ 和函数 $\pmb y=f(\pmb x)\in \mathbb{R}^N$， $\pmb y$ 关于 $\pmb x$ 的偏导数为
$$
\frac{\partial \pmb y}{\partial \pmb x}=\begin{bmatrix} \frac{\partial y_1}{\partial  x_1} & \cdots & \frac{\partial y_N}{\partial  x_1}\\
\vdots & \ddots & \vdots \\
\frac{\partial y_1}{\partial  x_M} & \cdots & \frac{\partial y_N}{\partial  x_M}
\end{bmatrix}
\in \mathbb{R}^{M\times N}
$$
称为函数 $f(\pmb x)$ 的**雅可比矩阵（Jacobian Matrix）**的转置。

**标量关于矩阵的偏导数** 对于 $M\times N$ 维矩阵 $X\in \mathbb{R}^{M\times N}$ 和函数 $y=f(X)\in \mathbb{R}$， $y$ 关于 $X$ 的偏导数为
$$
\frac{\partial y}{\partial X}=\begin{bmatrix}
\frac{\partial y}{\partial  x_{11}}&\cdots&\frac{\partial y}{\partial x_{1N}}\\
\vdots & \ddots & \vdots\\
\frac{\partial y}{\partial  x_{M1}}&\cdots&\frac{\partial y}{\partial x_{MN}}\\
\end{bmatrix}\in\mathbb{R}^{M\times N}
$$

**矩阵关于标量的偏导数**  对于标量 $x\in \mathbb{R}$ 和函数 $Y=f(x)\in \mathbb{R}^{M\times N}$， $Y$ 关于 $x$ 的偏导数为
$$
\frac{\partial Y}{\partial x}=\begin{bmatrix}
\frac{\partial y_{11}}{\partial  x}&\cdots&\frac{\partial y_{M1}}{\partial x}\\
\vdots & \ddots & \vdots\\
\frac{\partial y_{1N}}{\partial  x}&\cdots&\frac{\partial y_{MN}}{\partial x}\\
\end{bmatrix}\in \mathbb{R}^{N\times M}
$$




@ $y=\pmb x^{\rm T}A\pmb x$，其中 $\pmb x\in \mathbb{R}^n,A\in\mathbb{R}^{n\times n}$，计算 $\frac{\partial y}{\partial \pmb x}$。
$$
y=\pmb x^{\rm T}A\pmb x=\sum_{i=1}^n \sum_{j=1}^n a_{ij}x_ix_j\quad(二次型)\\
\frac{\partial y}{\partial x_1}=\sum_{i=1}^na_{i1}x_i+\sum_{j=1}^na_{1j}x_j=(A^{\rm T}\pmb x)_1+(A\pmb x)_1\\
\therefore \frac{\partial y}{\partial \pmb x}=A\pmb x+A^{\rm T}\pmb x
$$


@ $y={\rm tr}(A)$ 其中 $A\in \mathbb{R}^{n\times n}$，计算 $\frac{\partial y}{\partial A}$。
$$
\frac{\partial y}{\partial a_{ij}}=\frac{\partial\sum_{k=1}^na_{kk}}{\partial a_{ij}}=\begin{cases}1,&i=j\\
0,&i\neq j
\end{cases}\\
\therefore \frac{\partial y}{\partial A}=I
$$



## 导数计算法则

**加法法则**

若 $\pmb x\in \mathbb{R}^M$， $\pmb y=f(\pmb x)\in \mathbb{R}^N$， $\pmb z=g(\pmb x)\in \mathbb{R}^N$，则
$$
\frac{\partial (\pmb y+ \pmb z)}{\partial \pmb x}=\frac{\partial \pmb y}{\partial \pmb x}+\frac{\partial \pmb z}{\partial \pmb x} \in \mathbb{R}^{M\times N}
$$
**乘法法则**

1. 若 $\pmb x\in \mathbb{R}^M$， $\pmb y=f(\pmb x)\in \mathbb{R}^N$， $\pmb z=g(\pmb x)\in \mathbb{R}^N$，则
   $$
   \frac{\partial \pmb y^{\rm T} \pmb z}{\partial \pmb x}=\frac{\partial \pmb y}{\partial \pmb x}\pmb z+\frac{\partial \pmb z}{\partial \pmb x}\pmb y \in \mathbb{R}^{M}
   $$

2. 若 $\pmb x\in \mathbb{R}^M$， $\pmb y=f(\pmb x)\in \mathbb{R}^S$， $\pmb z=g(\pmb x)\in \mathbb{R}^T$， $A \in \mathbb{R}^{S\times T}$ 和 $\pmb x$ 无关，则
   $$
   \frac{\partial \pmb y^{\rm T} A \pmb z}{\partial \pmb x}=\frac{\partial \pmb y}{\partial \pmb x}A\pmb z+\frac{\partial \pmb z}{\partial \pmb x} A^{\rm T} \pmb y \in \mathbb{R}^{M}
   $$

3. 若 $\pmb x\in \mathbb{R}^M$， $y=f(\pmb x)\in \mathbb{R}$， $\pmb z=g(\pmb x)\in \mathbb{R}^N$，则
   $$
   \frac{\partial y \pmb z}{\partial \pmb x}=y\frac{\partial \pmb z}{\partial \pmb x}+\frac{\partial y}{\partial \pmb x}\pmb z^{\rm T} \in \mathbb{R}^{M\times N}
   $$
   
4. 若 $x\in \mathbb{R},Y\in \mathbb{R}^{M\times N},Z\in \mathbb{R}^{N\times P}$，则
   $$
   \frac{\partial YZ}{\partial x}=Z^{\rm T}\frac{\partial Y}{\partial x}+\frac{\partial Z}{\partial x}Y^{\rm T}
   $$
   



**链式法则（Chain Rule）**

1. 若 $x\in \mathbb{R}$， $\pmb y=f(x)\in \mathbb{R}^M$， $\pmb z=g(\pmb y)\in \mathbb{R}^N$，则
   $$
   \frac{\partial \pmb z}{\partial x}=\frac{\partial \pmb y}{\partial x}\frac{\partial \pmb z}{\partial \pmb y} \in \mathbb{R}^{1\times N}
   $$

2. 若 $\pmb x\in \mathbb{R}^M$， $\pmb y=f(\pmb x)\in \mathbb{R}^K$， $\pmb z=g(\pmb y)\in \mathbb{R}^N$，则
   $$
   \frac{\partial \pmb z}{\partial \pmb x}=\frac{\partial \pmb y}{\partial \pmb x}\frac{\partial \pmb z}{\partial \pmb y} \in \mathbb{R}^{M\times N}
   $$

3. 若 $X\in \mathbb{R}^{M\times N}$， $\pmb y=f(X)\in \mathbb{R}^K$， $z=g(\pmb y)\in \mathbb{R}$，则
   $$
   \frac{\partial z}{\partial x_{ij}}=\frac{\partial \pmb y}{\partial x_{ij}}\frac{\partial z}{\partial \pmb y} \in \mathbb{R}
   $$
   
4. 若 $X\in \mathbb{R}^{M\times N}$， $Y=f(X)\in \mathbb{R}^{M\times N}$， $z=g(Y)\in \mathbb{R}$，则
   $$
   \frac{\partial z}{\partial x_{ij}}=\sum_{p=1}^{M}\sum_{q=1}^{N}\frac{\partial y_{pq}}{\partial x_{ij}}\frac{\partial z}{\partial y_{pq}} \in \mathbb{R}
   $$



## 导数计算的微分方法

微分方法通过推导出微分与导数的关系式得到导数
$$
{\rm d}y=(\frac{\partial y}{\partial \pmb x})^{\rm T}{\rm d}\pmb x\\
{\rm d}y={\rm tr}((\frac{\partial y}{\partial X})^{\rm T}{\rm d}X)
$$
计算对矩阵的偏导数时，一些迹技巧（trace trick）非常有用：
$$
a={\rm tr}(a)\\
{\rm tr}(A^{\rm T})={\rm tr}(A)\\
{\rm tr}(A\pm B)={\rm tr}(A)\pm {\rm tr}(B)\\
{\rm tr}(AB)={\rm tr}(BA)\\
{\rm tr}(A^{\rm T}(B\odot C))={\rm tr}((A\odot B)^{\rm T} C)
$$


@ $y=\pmb x^{\rm T}A\pmb x$，其中 $\pmb x\in \mathbb{R}^n,A\in\mathbb{R}^{n\times n}$，计算 $\frac{\partial y}{\partial \pmb x}$。
$$
{\rm d}y={\rm d}(\pmb x^{\rm T})A\pmb x+\pmb x^{\rm T}{\rm d}A\pmb x+\pmb x^{\rm T}A{\rm d}\pmb x\\
=({\rm d}\pmb x)^{\rm T}A\pmb x+\pmb x^{\rm T}A{\rm d}\pmb x\\
=\pmb x^{\rm T}A^{\rm T}{\rm d}\pmb x+\pmb x^{\rm T}A{\rm d}\pmb x\\
=(A\pmb x+A^{\rm T}\pmb x)^{\rm T}{\rm d}\pmb x\\
\therefore \frac{\partial y}{\partial \pmb x}=A\pmb x+A^{\rm T}\pmb x
$$


@ $W\in\mathbb{R}^{R\times S}$， $X=g(W)=AWB\in\mathbb{R}^{M\times N}$， $y=f(X)\in \mathbb{R}$， $\frac{\partial y}{\partial X}$ 已知，求 $\frac{\partial y}{\partial W}$。
$$
\because {\rm d}y={\rm tr}((\frac{\partial y}{\partial X})^{\rm T}{\rm d}X)={\rm tr}((\frac{\partial y}{\partial X})^{\rm T}A{\rm d}WB)={\rm tr}(B(\frac{\partial y}{\partial X})^{\rm T}A{\rm d}W)={\rm tr}((A^{\rm T}(\frac{\partial y}{\partial X})B^{\rm T})^{\rm T}{\rm d}W)\\
\therefore \frac{\partial y}{\partial W}=A^{\rm T}(\frac{\partial y}{\partial X})B^{\rm T}
$$




## 常用导数

$$
\frac{\partial}{\partial \pmb x}\pmb x=I\\
\frac{\partial}{\partial \pmb x}A\pmb x=A^{\rm T},\ 
\frac{\partial}{\partial \pmb x}A^{\rm T}\pmb x=\frac{\partial}{\partial \pmb x}\pmb x^{\rm T}A =A\quad A换成\pmb a同样成立\\
\frac{\partial}{\partial \pmb x}\pmb x^{\rm T} A \pmb x=(A+A^{\rm T})\pmb x\\
\frac{\partial}{\partial \pmb x}||\pmb x||^2=2\pmb x\\
$$

$$
\frac{\partial}{\partial X}X=I\otimes I\\
\frac{\partial}{\partial X}AX=I\otimes A^{\rm T}\\
\frac{\partial}{\partial X}\pmb a^{\rm T}X\pmb b=\pmb a\pmb b^{\rm T},\frac{\partial}{\partial X}\pmb a^{\rm T}X^{\rm T}\pmb b=\pmb b\pmb a^{\rm T}\\
\frac{\partial}{\partial X}\pmb a^{\rm T}X^{\rm T}X\pmb b=X(\pmb a\pmb b^{\rm T}+\pmb b\pmb a^{\rm T})\\
\frac{\partial}{\partial X}{\rm tr}(X)=I\\
\frac{\partial}{\partial X}{\rm tr}(XA)=A^{\rm T},\frac{\partial}{\partial X}{\rm tr}(X^{\rm T}A)=\frac{\partial}{\partial X}{\rm tr}(AX^{\rm T})=A\\
\frac{\partial}{\partial X}{\rm tr}(AXB)=A^{\rm T}B^{\rm T},\frac{\partial}{\partial X}{\rm tr}(AX^{\rm T}B)=BA\\
\frac{\partial}{\partial X}{\rm det}(X)=(X^*)^{\rm T},\quad X^*为伴随矩阵\\


\frac{\partial}{\partial X}||X||_F^2=2X\\
$$

> $$
> I\otimes I=\begin{bmatrix}I & \cdots & \pmb 0\\
> \vdots & \ddots & \vdots\\
> \pmb 0 &\cdots & I
> \end{bmatrix}
> $$





# 微分

## 矩阵微分

回顾一元和多元微积分中的微分与导数的关系
$$
y=f(x):{\rm d}y=y'{\rm d}x\\
y=f(x_1,x_2,\cdots,x_n)=f(\pmb x):{\rm d}y=\sum_{i=1}^n\frac{\partial y}{\partial x_i}{\rm d}x_i=(\frac{\partial y}{\partial \pmb x})^{\rm T}{\rm d}\pmb x
$$
类似地，我们建立矩阵微分与导数的关系
$$
y=f(X):{\rm d}y=\sum_{i=1}^m\sum_{j=1}^n \frac{\partial y}{\partial x_{ij}}{\rm d}x_{ij}={\rm tr}((\frac{\partial y}{\partial X})^{\rm T}{\rm d}X)
$$

> ${\rm tr}(A^{\rm T}B)=\sum_{i,j} A_{ij}B_{ij}$ 称为矩阵 $A,B$ 的内积



## 常用微分

|            |                                                              |
| ---------- | ------------------------------------------------------------ |
| 加减法     | ${\rm d}(X\pm Y)={\rm d}X\pm {\rm d}Y$                     |
| 数乘       | ${\rm d}(\alpha X)=\alpha {\rm d} X$                       |
| 乘法       | ${\rm d}(XY)={\rm d}X\ Y+X{\rm d}Y$                        |
| 幂         | ${\rm d}X^n=\sum_{i=0}^{n-1} X^i {\rm d}X X^{n-1-i}$       |
| 转置       | ${\rm d}(X^{\rm T})=({\rm d}X)^{T}$                        |
| 迹         | ${\rm d}{\rm tr}(X)={\rm tr}({\rm d}X)$                    |
| 逆         | ${\rm d}X^{-1}=-X^{-1}{\rm d}XX^{-1}$                      |
| 行列式     | ${\rm d}\vert X\vert={\rm tr}(X^*{\rm d}X)$ , $X^*$ 为伴随矩阵 |
|            | ${\rm d}\vert X\vert=\vert X\vert{\rm tr}(X^{-1}{\rm d}X)$ , 如果 $X$ 可逆 |
| 逐元素乘法 | ${\rm d}(X\odot Y)={\rm d}X\odot Y+X\odot {\rm d}Y $       |
| 逐元素函数 | ${\rm d}\sigma(X)=\sigma'(X)\odot {\rm d}X$ , $\sigma$ 为逐元素函数运算 |
|            |                                                              |



