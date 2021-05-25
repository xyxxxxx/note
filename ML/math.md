# 函数

## Logistic函数

Logistic函数是一种常用的S型函数, 是比利时数学家Pierre François Ver-hulst在1844年~1845年研究种群数量的增长模型时提出命名的, 最初作为一种生态学模型.

Logistic函数定义为
$$
{\rm logistic}(x)=\frac{L}{1+\exp(-K(x-x_0))}
$$
当参数为（$$k=1, x_0=0,L=1$$）时，Logistic函数称为标准Logistic函数，记为$$\sigma(x)$$：
$$
\sigma(x)=\frac{1}{1+\exp(-x)}
$$
标准 Logistic 函数在机器学习中使用得非常广泛，经常用来<u>将一个实数空间的数映射到(0, 1)区间</u>。

标准 Logistic 函数的导数为
$$
\sigma'(x)=\sigma(x)(1-\sigma(x))
$$
当输入为$$K$$维向量$$x = [x_1,⋯,x_K]^{\rm T}$$时，其导数为
$$
σ′(\boldsymbol x) ={\rm diag}(σ(\boldsymbol x) ⊙ (1 − σ(\boldsymbol x)))
$$



## Softmax函数

Softmax函数可以将多个标量映射为一个概率分布。对于$$K$$个标量$$x_1,\cdots,x_K$$，Softmax函数定义为
$$
z_k={\rm softmax}(x_k)=\frac{\exp(x_k)}{\sum_{i=1}^K \exp(x_i)}
$$
这样，我们可以将$$K$$个标量$$x_1, ⋯, x_K$$转换为一个分布：$$z_1, ⋯, z_K$$，满足
$$
\sum_{k=1}^Kz_k=1,\quad z_k \in (0,1)
$$
为了简便起见，将Softmax函数简写为
$$
\hat{\boldsymbol z}={\rm softmax}(\boldsymbol x)=\frac{\exp(\boldsymbol x)}{\boldsymbol 1_K^{\rm T}\exp(\boldsymbol x)}
$$
其中$$\boldsymbol 1_K$$是$$K$$维的全1向量

Softmax函数的导数为
$$
\frac{\partial {\rm softmax}(\boldsymbol x)}{\partial \boldsymbol x}={\rm diag}({\rm softmax}(\boldsymbol x))-\rm softmax(\boldsymbol x)\rm softmax(\boldsymbol x)^{\rm T}
$$

