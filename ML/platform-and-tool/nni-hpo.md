# 穷举搜索Exhaustive search

## 网格搜索Grid Search

穷举搜索空间中定义的每一种超参组合。

### 配置示例

```yaml
# config.yml
tuner:
  builtinTunerName: GridSearch
```



### 使用建议

建议在搜索空间较小，可以穷尽时使用。



## 随机搜索Random Search

在搜索空间中随机取值。

### 配置示例

```yaml
# config.yml
tuner:
  builtinTunerName: Random
```



### 使用建议

+ 建议在每一次trial花费时间不长，并且计算资源充足的情形下使用。

+ 可以帮助你均匀地探索整个搜索空间。

+ 可以作为基线搜索算法，尤其是当你对超参数的先验分布没有任何信息时。



## 批量调参器Batch Tuner

仅运行用户提供的几组超参组合。

### 配置示例

```yaml
# config.yml
tuner:
  builtinTunerName: BatchTuner
```

搜索空间定义如下：

```json
{
    "combine_params":
    {
        "_type" : "choice",
        "_value" : [{"optimizer": "Adam", "learning_rate": 0.00001},
                    {"optimizer": "Adam", "learning_rate": 0.0001},
                    {"optimizer": "Adam", "learning_rate": 0.001},
                    {"optimizer": "SGD", "learning_rate": 0.01},
                    {"optimizer": "SGD", "learning_rate": 0.005},
                    {"optimizer": "SGD", "learning_rate": 0.0002}]
    }
}
```



### 使用建议

使用的超参组合已提前确定，将它们罗列到搜索空间中运行即可。



## 网格搜索与随机搜索

实践证明，搜索同样数量的点，随机搜索的效果好于网格搜索。

其原因解释为：有些超参数比其余超参数重要得多（亦即，有些超参数的变化对训练结果的影响大，其余超参数影响不大），即使你在事前并不知道哪些参数更重要。网格搜索会在那些不重要的参数上花费过多的计算资源，而对于重要的参数取值又太少。随机搜索会为每个参数取到非常多数量的值，对于不重要的参数而言，取哪些值影响都不大，但对于重要的参数而言，这些值可以为找到全局最小/最大值提供更多的信息。如下图所示。
![](https://lh5.googleusercontent.com/cXgNEuYsOeaGVNflSM-1Pl9_qG30ybNUzoq1nPtlwJZHcgo1MgTYElKu7XCJ7rsR7vxViCXpMBDEBmGWoJ_I_EzzZLjvSqDFa8gOsFzLE4F5Kw2upl9uHK5zLbJgzIfEQ9h9cB-q)





# 启发式搜索Heuristic search

## 朴素进化Naive Evolution

来自于论文[Large-Scale Evolution of Image Classifiers](https://arxiv.org/pdf/1703.01041.pdf)。该算法会基于搜索空间随机生成一个指定规模的*种群*（模型集合），并让这个种群不断进化。

具体方法如下：

+ 将*个体*（模型）在单独的验证集上的准确率作为其*体质*的衡量标准；
+ 在每一个进化步中，工作节点从种群随机抽取两个个体，比较它们的体质，体质更差的那一个会被立即去除（即被*杀死*），而体质更好的那一个会被保留，并且*繁殖*一个*子代*；
+ 子代是*亲代*的一个副本，但会应用一个称为*变异*的修改，修改的具体操作从预定义了一组变异（例如修改一个超参数，增加或减少一层网络等）的集合中随机抽取；
+ 工作节点会继续训练子代，将其在验证集上测试，并放回种群



### 配置示例

```yaml
# config.yml
tuner:
  builtinTunerName: Evolution
  classArgs:
    optimize_mode: maximize
    population_size: 100
```

**参数**

+ `optimize_mode`：若为maximize，调参器会最大化指标；若为minimize，调参器会最小化指标。
+ `population_size`：种群的规模。建议`population_size`取值大于`concurrency`……



### 使用建议

+ 此算法对计算资源的要求较高。它需要设定较大的种群规模，以达到更好的局部最优解；也需要设定较大的训练步数，以使得每个模型得到较为充分的训练。在此基础上，还需要非常多次的trial才能得到表现较好的模型。
+ 变异的行为由人工设定，一般包括保持不变、更改超参数（如学习率，网络层规模）、重置参数、增加网络层、去除网络层等。
+ 更改超参数的突变会使得搜索空间更大，甚至没有边界。
+ 在定义了一组变异之后，只需要构造一组简单的初始模型，并赋予搜索空间中的随机超参数。随着训练过程的推进，好的结构和超参数会自发地进化出来。
+ 建议使用权重继承，即子代会继承亲代的模型参数（变异的情况除外），这将大大提升训练速度，并使得模型充分训练。
+ 训练结束后，可以根据验证集准确率挑选出一个最佳模型，也可以挑选出多个最佳模型，再通过多数表决等方式进行集成。



## 模拟退火Simulated Annealing

模拟退火算法的介绍请参考[英文维基页面](https://en.wikipedia.org/wiki/Simulated_annealing)。



### 配置示例

```yaml
# config.yml
tuner:
  builtinTunerName: Anneal
  classArgs:
    optimize_mode: maximize
```

**参数**

+ `optimize_mode`：若为maximize，调参器会最大化指标；若为minimize，调参器会最小化指标。



### 使用建议

+ 此算法可以视作随机搜索的变体，区别在于：在超参数空间中以当前状态为中心的范围内随机生成新状态（范围大小由温度$$T$$控制），并且在新状态更优时接受新状态，新状态更劣时以一定概率接受新状态。
+ 建议在每一次trial花费时间不长，并且计算资源充足的情形下使用。



## Hyperband





## PBT

来自于论文[Large-Scale Evolution of Image Classifiers](https://arxiv.org/pdf/1703.01041.pdf)。该算法是一种简单的异步优化算法，在固定的计算资源下，它能有效地联合优化一组模型及其超参来最大化性能。



### 配置示例

```yaml
# config.yml
tuner:
  builtinTunerName: Anneal
  classArgs:
    optimize_mode: maximize
```

**参数**

+ `optimize_mode`：若为maximize，调参器会最大化指标；若为minimize，调参器会最小化指标。



### 使用建议





# 贝叶斯优化Bayesian optimization

## BOHB



## TPE



### 配置示例

```yaml
# config.yml
tuner:
  builtinTunerName: TPE
  classArgs:
    optimize_mode: maximize
```



### 使用建议

+ TPE是一种黑盒优化方法，可以使用在各种场景中，通常情况下都能得到较好的结果。特别是在计算资源有限，只能运行少量Trial的情况。
+ 大量实验表明，TPE的性能远远优于随机搜索。







## SMAC



## Metis Tuner



## GP Tuner

