> 所有证明省略，仅供应用参考
>
> 所有矩阵用非粗体的大写字母表示

# 超越运算

对于$$M\times N$$维矩阵$$A\in \mathbb{R}^{M\times N}$$，以$$A$$为参数的超越函数使用无穷级数的方法计算，例如
$$
\exp(A)=1+A+\frac{A^2}{2!}+\cdots\\
\sin(A)=A-\frac{A^3}{3!}+\frac{A^5}{5!}-\cdots
$$




# 导数

## 标量、向量与矩阵之间的导数

**标量关于向量的偏导数** 对于M维向量$$\pmb x\in \mathbb{R}^M$$和函数$$y=f(\pmb x)\in \mathbb{R}$$，$$y$$关于$$\pmb x$$的偏导数为
$$
\frac{\partial y}{\partial \pmb x}=[\frac{\partial y}{\partial  x_1},\cdots,\frac{\partial y}{\partial x_M}]^{\rm T}
$$
> 多元微积分中的梯度即是标量对向量求偏导数

$$y$$关于$$\pmb x$$的二阶偏导数为
$$
H=\frac{\partial^2 y}{\partial \pmb x^2}=\begin{bmatrix} \frac{\partial^2 y}{\partial x_1^2} & \cdots & \frac{\partial^2 y}{\partial  x_1 \partial x_M}\\
\vdots & \ddots & \vdots \\
\frac{\partial^2 y}{\partial x_M \partial x_1} & \cdots & \frac{\partial^2 y}{\partial  x_M^2}
\end{bmatrix}
\in \mathbb{R}^{M\times M}
$$
称为函数$$f(\pmb x)$$的**Hessian矩阵**，也写作$$\nabla^2 f(\pmb x)$$

**向量关于标量的偏导数** 对于标量$$x\in \mathbb{R}$$和函数$$\pmb y=f(x)\in \mathbb{R}^N$$，$$\pmb y$$关于$$x$$的偏导数为
$$
\frac{\partial \pmb y}{\partial x}=[\frac{\partial y_1}{\partial  x},\cdots,\frac{\partial y_N}{\partial x}]
$$
**向量关于向量的偏导数** 对于M维向量$$\pmb x\in \mathbb{R}^M$$和函数$$\pmb y=f(\pmb x)\in \mathbb{R}^N$$，$$\pmb y$$关于$$\pmb x$$的偏导数为
$$
\frac{\partial \pmb y}{\partial \pmb x}=\begin{bmatrix} \frac{\partial y_1}{\partial  x_1} & \cdots & \frac{\partial y_N}{\partial  x_1}\\
\vdots & \ddots & \vdots \\
\frac{\partial y_1}{\partial  x_M} & \cdots & \frac{\partial y_N}{\partial  x_M}
\end{bmatrix}
\in \mathbb{R}^{M\times N}
$$
称为函数$$f(\pmb x)$$的**雅可比矩阵（Jacobian Matrix）**的转置。



**标量关于矩阵的偏导数** 对于$$M\times N$$维矩阵$$X\in \mathbb{R}^{M\times N}$$和函数$$y=f(X)\in \mathbb{R}$$，$$y$$关于$$X$$的偏导数为
$$
\frac{\partial y}{\partial X}=\begin{bmatrix}
\frac{\partial y}{\partial  x_{11}}&\cdots&\frac{\partial y}{\partial x_{1N}}\\
\vdots & \ddots & \vdots\\
\frac{\partial y}{\partial  x_{M1}}&\cdots&\frac{\partial y}{\partial x_{MN}}\\
\end{bmatrix}
$$






## 导数计算法则

**加法法则**

若$$\pmb x\in \mathbb{R}^M$$，$$\pmb y=f(\pmb x)\in \mathbb{R}^N$$，$$\pmb z=g(\pmb x)\in \mathbb{R}^N$$，则
$$
\frac{\partial (\pmb y+ \pmb z)}{\partial \pmb x}=\frac{\partial \pmb y}{\partial \pmb x}+\frac{\partial \pmb z}{\partial \pmb x} \in \mathbb{R}^{M\times N}
$$
**乘法法则**

1. 若$$\pmb x\in \mathbb{R}^M$$，$$\pmb y=f(\pmb x)\in \mathbb{R}^N$$，$$\pmb z=g(\pmb x)\in \mathbb{R}^N$$，则
   $$
   \frac{\partial \pmb y^{\rm T} \pmb z}{\partial \pmb x}=\frac{\partial \pmb y}{\partial \pmb x}\pmb z+\frac{\partial \pmb z}{\partial \pmb x}\pmb y \in \mathbb{R}^{M}
   $$

2. 若$$\pmb x\in \mathbb{R}^M$$，$$\pmb y=f(\pmb x)\in \mathbb{R}^S$$，$$\pmb z=g(\pmb x)\in \mathbb{R}^T$$，$$A \in \mathbb{R}^{S\times T}$$和$$\pmb x$$无关，则
   $$
   \frac{\partial \pmb y^{\rm T} A \pmb z}{\partial \pmb x}=\frac{\partial \pmb y}{\partial \pmb x}A\pmb z+\frac{\partial \pmb z}{\partial \pmb x} A^{\rm T} \pmb y \in \mathbb{R}^{M}
   $$

3. 若$$\pmb x\in \mathbb{R}^M$$，$$y=f(\pmb x)\in \mathbb{R}$$，$$\pmb z=g(\pmb x)\in \mathbb{R}^N$$，则
   $$
   \frac{\partial y \pmb z}{\partial \pmb x}=y\frac{\partial \pmb z}{\partial \pmb x}+\frac{\partial y}{\partial \pmb x}\pmb z^{\rm T} \in \mathbb{R}^{M\times N}
   $$

**链式法则（Chain Rule）**

1. 若$$x\in \mathbb{R}$$，$$\pmb y=f(x)\in \mathbb{R}^M$$，$$\pmb z=g(\pmb y)\in \mathbb{R}^N$$，则
   $$
   \frac{\partial \pmb z}{\partial x}=\frac{\partial \pmb y}{\partial x}\frac{\partial \pmb z}{\partial \pmb y} \in \mathbb{R}^{1\times N}
   $$

2. 若$$\pmb x\in \mathbb{R}^M$$，$$\pmb y=f(\pmb x)\in \mathbb{R}^K$$，$$\pmb z=g(\pmb y)\in \mathbb{R}^N$$，则
   $$
   \frac{\partial \pmb z}{\partial \pmb x}=\frac{\partial \pmb y}{\partial \pmb x}\frac{\partial \pmb z}{\partial \pmb y} \in \mathbb{R}^{M\times N}
   $$

3. 若$$X\in \mathbb{R}^{M\times N}$$，$$\pmb y=f(X)\in \mathbb{R}^K$$，$$z=g(\pmb y)\in \mathbb{R}$$，则
   $$
   \frac{\partial z}{\partial x_{ij}}=\frac{\partial \pmb y}{\partial x_{ij}}\frac{\partial z}{\partial \pmb y} \in \mathbb{R}
   $$



## 导数计算的微分方法

例如，$$W\in\R^{R\times S}$$，$$X=g(W)=AWB\in\R^{M\times N}$$，$$y=f(X)\in \R$$，$$\frac{\partial y}{\partial X}$$已知，求$$\frac{\partial y}{\partial W}$$。
$$
\because {\rm d}y={\rm tr}((\frac{\partial y}{\partial X})^{\rm T}{\rm d}X)={\rm tr}((\frac{\partial y}{\partial X})^{\rm T}A{\rm d}WB)={\rm tr}(B(\frac{\partial y}{\partial X})^{\rm T}A{\rm d}W)={\rm tr}((A^{\rm T}(\frac{\partial y}{\partial X})B^{\rm T})^{\rm T}{\rm d}W)\\
\therefore \frac{\partial y}{\partial W}=A^{\rm T}(\frac{\partial y}{\partial X})B^{\rm T}
$$




## 常用导数

$$
\frac{\partial \pmb x}{\partial \pmb x}=\pmb I\\
\frac{\partial ||\pmb x||^2}{\partial \pmb x}=2\pmb x\\
\frac{\partial A \pmb x}{\partial \pmb x}=A^{\rm T}\\
\frac{\partial A \pmb x}{\partial \pmb x^{\rm T}}=\frac{\partial \pmb x^{\rm T} A}{\partial \pmb x}=A\\
\frac{\partial \pmb x^{\rm T} A \pmb x}{\partial \pmb x}=(A+A^{\rm T})\pmb x\\
$$





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

> $${\rm tr}(A^{\rm T}B)=\sum_{i,j} A_{ij}B_{ij}$$称为矩阵$$A,B$$的内积



## 运算法则

| 加减法     | $${\rm d}(X\pm Y)={\rm d}X\pm {\rm d}Y$$                     |
| ---------- | ------------------------------------------------------------ |
| 乘法       | $${\rm d}(XY)=Y{\rm d}X+X{\rm d}Y$$                          |
| 转置       | $${\rm d}(X^{\rm T})=({\rm d}X)^{T}$$                        |
| 迹         | $${\rm d}{\rm tr}(X)={\rm tr}({\rm d}X)$$                    |
| 逆         | $${\rm d}X^{-1}=-X^{-1}{\rm d}XX^{-1}$$                      |
| 行列式     | $${\rm d}|X|={\rm tr}(X^*{\rm d}X)$$, $$X^*$$为伴随矩阵      |
|            | $${\rm d}|X|=|X|{\rm tr}(X^{-1}{\rm d}X)$$, 如果$$X$$可逆    |
| 逐元素乘法 | $${\rm d}(X\odot Y)={\rm d}X\odot Y+X\odot {\rm d}Y $$       |
| 逐元素函数 | $${\rm d}\sigma(X)=\sigma'(X)\odot {\rm d}X$$, $$\sigma$$为逐元素函数运算 |



