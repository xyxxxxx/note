

# 通用配置示例

## NNI配置示例

## AutoTune配置示例

# Exhaustive search穷举搜索

## Grid Search网格搜索

每个超参数的搜索空间都是离散值，总的搜索空间即为这些搜索空间的笛卡尔积。算法穷举搜索空间中的每一组超参数配置，最终选取最好的一个。

### NNI配置示例

> 参考NNI官方[文档](https://github.com/microsoft/nni/blob/master/docs/zh_CN/Tuner/BuiltinTuner.rst)。

```json
// search_space.json
{
    "batch_size": {"_type": "choice", "_value": [16, 32, 64, 128]},
    "hidden_size": {"_type": "choice", "_value": [128, 256, 512, 1024]},
    "lr": {"_type": "choice", "_value": [0.0001, 0.001, 0.01, 0.1]}
}
```

搜索空间接受的参数类型包括 `choice`，`quniform`，`randint`。

```yaml
# config.yml
tuner:
  builtinTunerName: GridSearch
```

### AutoTune配置示例

```yaml
spec:
  searchSpace: |-
    {
        "batch_size": {"_type": "choice", "_value": [16, 32, 64, 128]},
        "hidden_size": {"_type": "choice", "_value": [128, 256, 512, 1024]},
        "lr": {"_type": "choice", "_value": [0.0001, 0.001, 0.01, 0.1]}
    }
  tuner:
    builtinTunerName: GridSearch
```

### 使用建议

不建议使用。因为随着超参数的数量增加，超参数组合数会呈指数增长；与此同时，由于完全没有寻找（全局或局部）最大/最小值或者它们的近似值的机制，最终结果也无法保证。总的来说，此算法的效率极低。

## Random Search随机搜索

算法在搜索空间中随机选取超参数的值。由于此算法没有终点，因此需要指定计算资源（trial 次数或最大运行时间）。

### NNI配置示例

```json
// search_space.json
{
    "batch_size": {"_type": "choice", "_value": [16, 32, 64, 128]},
    "hidden_size": {"_type": "choice", "_value": [128, 256, 512, 1024]},
    "lr": {"_type": "choice", "_value": [0.0001, 0.001, 0.01, 0.1]},
    "momentum": {"_type": "uniform", "_value": [0, 1]}
}
```

```yaml
# config.yml
tuner:
  builtinTunerName: Random
```

### AutoTune配置示例

```yml
spec:
  searchSpace: |-
    {
        "batch_size": {"_type": "choice", "_value": [16, 32, 64, 128]},
        "hidden_size": {"_type": "choice", "_value": [128, 256, 512, 1024]},
        "lr": {"_type": "choice", "_value": [0.0001, 0.001, 0.01, 0.1]},
        "momentum": {"_type": "uniform", "_value": [0, 1]}
    }
  tuner:
    builtinTunerName: Random
```

### 使用建议

* 建议在每一次 trial 花费时间不长，计算资源充足的情形下使用。
* 常用作基线搜索算法，尤其建议在不知道超参数的先验分布时使用。
* 此算法支持并行。

## 网格搜索与随机搜索

论文 [Random search for hyper-parameter optimization](https://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a) 从经验和理论两方面证明了，随机搜索相比传统的网格搜索，能在更短的时间内找到一样或更好的模型参数。

其原因可以简单解释为：有些超参数比其余超参数重要得多（亦即，有些超参数的变化对训练结果的影响大，其余超参数影响不大），即使你在事前并不知道哪些参数更重要。网格搜索会在那些不重要的参数上花费过多的计算资源，而对于重要的参数取值又太少。随机搜索会为每个参数取到非常多数量的值，对于不重要的参数而言，取哪些值影响都不大，但对于重要的参数而言，这些值可以为找到全局最小/最大值提供更多的信息。如下图所示。

![](https://pic1.zhimg.com/80/v2-40bb19a5d4da47c36bf549dd3f71aa44_720w.jpg)

网格搜索和随机搜索是最简单的两种优化方法，它们容易实现，可以并行运行，本身也没有额外的参数，但缺点在于没有任何机制能够保证它们能找到（全局或局部）最大/最小值或者它们的近似值。当模型运行时间较长、消耗计算资源较多时，这两种算法将十分低效，因为它们无法利用之前的 trial 的信息。

## Batch Tuner批量调参器

用户自行写入想要尝试的超参数配置，算法将依次运行这些配置。此算法相当于一个批处理工具。

### NNI配置示例

```json
// search_space.json
{
    "combine_params":
    {
        "_type": "choice",
        "_value": [{"batch_size": 32, "hidden_size": 128, "lr": 0.001},
                    {"batch_size": 32, "hidden_size": 128, "lr": 0.01},
                    {"batch_size": 32, "hidden_size": 256, "lr": 0.001},
                    {"batch_size": 32, "hidden_size": 256, "lr": 0.01},
                    {"batch_size": 64, "hidden_size": 128, "lr": 0.001},
                    {"batch_size": 64, "hidden_size": 128, "lr": 0.01},
                    {"batch_size": 64, "hidden_size": 256, "lr": 0.001},
                    {"batch_size": 64, "hidden_size": 256, "lr": 0.01}]
    }
}
```

```yaml
# config.yml
tuner:
  builtinTunerName: BatchTuner
```

### AutoTune配置示例

```yaml
spec:
  searchSpace: |-
    {
        "combine_params":
        {
            "_type": "choice",
            "_value": [{"batch_size": 32, "hidden_size": 128, "lr": 0.001},
                        {"batch_size": 32, "hidden_size": 128, "lr": 0.01},
                        {"batch_size": 32, "hidden_size": 256, "lr": 0.001},
                        {"batch_size": 32, "hidden_size": 256, "lr": 0.01},
                        {"batch_size": 64, "hidden_size": 128, "lr": 0.001},
                        {"batch_size": 64, "hidden_size": 128, "lr": 0.01},
                        {"batch_size": 64, "hidden_size": 256, "lr": 0.001},
                        {"batch_size": 64, "hidden_size": 256, "lr": 0.01}]
        }
    }
  tuner:
    builtinTunerName: BatchTuner
```

### 使用建议

若使用的超参数配置已预先确定，使用此算法并将这些配置罗列到搜索空间中即可。

# Heuristic search启发式搜索

## Naive Evolution朴素进化

来自于论文 [Large-Scale Evolution of Image Classifiers](https://arxiv.org/pdf/1703.01041.pdf)。此算法会基于搜索空间随机生成一个指定规模的*种群*（模型集合），并让这个种群不断进化。

具体方法如下：

* 将*个体*（模型）在单独的验证集上的准确率作为其*体质*的衡量标准；
* 在每一个进化步中，工作节点从种群随机抽取两个个体，比较它们的体质，体质更差的那一个会被立即去除（即被*杀死*），而体质更好的那一个会被保留，并且*繁殖*一个*子代*；
* 子代是*亲代*的一个副本，但会应用一个称为*变异*的修改，修改的具体操作从预定义的一组变异（例如修改一个超参数，增加或减少一层网络等）集合中随机抽取；
* 工作节点会继续训练子代，将其在验证集上测试，并放回种群。

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

* `optimize_mode`：若为 `maximize`，调参器会最大化指标；若为 `minimize`，调参器会最小化指标。
* `population_size`：种群的规模。建议 `population_size` 取值大于 `concurrency`……

### 使用方法与建议

* 此算法对计算资源的要求较高。它需要设定较大的种群规模，以达到更好的局部最优解；也需要设定较大的训练步数，以使得每个模型得到较为充分的训练。在此基础上，还需要非常多次的 trial 才能得到表现较好的模型。
* 变异的行为由人工设定，一般包括保持不变、更改超参数（如学习率，网络层规模）、重置参数、增加网络层、去除网络层等。
* 更改超参数的突变会使得搜索空间更大，甚至没有边界。
* 在定义了一组变异之后，只需要构造一组简单的初始模型，并赋予搜索空间中的随机超参数。随着训练过程的推进，好的结构和超参数会自发地进化出来。
* 建议使用权重继承，即子代会继承亲代的模型参数（变异的情况除外），这将大大提升训练速度，并使得模型充分训练。
* 训练结束后，可以根据验证集准确率挑选出一个最佳模型，也可以挑选出多个最佳模型，再通过多数表决等方式进行集成。

## Anneal退火

来自于模拟退火算法（Simulated Annealing，SA）。此算法是一种通用概率算法，常用来在一定时间内寻找在一个很大搜寻空间中的近似最优解。模拟退火算法类似于贪心算法和遗传算法的结合，其先对上一步中尝试的超参数组合进行随机扰动产生新解，若该新解有更好的结果则接受新解，若结果变差则按 Metropolis 准则以一定概率接受新解。

可以证明，模拟退火算法所得解依概率收敛到全局最优解。

### NNI配置示例

```json
// search_space.json
{
    "batch_size": {"_type": "choice", "_value": [16, 32, 64, 128]},
    "hidden_size": {"_type": "choice", "_value": [128, 256, 512, 1024]},
    "lr": {"_type": "choice", "_value": [0.0001, 0.001, 0.01, 0.1]},
    "momentum": {"_type": "uniform", "_value": [0, 1]}
}
```

```yaml
# config.yml
tuner:
  builtinTunerName: Anneal
  classArgs:
    optimize_mode: maximize
```

**参数**

* `optimize_mode`：若为 `maximize`，调参器会最大化指标；若为 `minimize`，调参器会最小化指标。

### AutoTune配置示例

```yaml
spec:
  searchSpace: |-
    {
        "batch_size": {"_type": "choice", "_value": [16, 32, 64, 128]},
        "hidden_size": {"_type": "choice", "_value": [128, 256, 512, 1024]},
        "lr": {"_type": "choice", "_value": [0.0001, 0.001, 0.01, 0.1]},
        "momentum": {"_type": "uniform", "_value": [0, 1]}
    }
  tuner:
    builtinTunerName: Anneal
    classArgs:
      optimize_mode: maximize
```

### 使用建议

* 此算法可以视作随机搜索的变体，区别在于：在超参数空间中以当前状态为中心的范围内随机生成新状态（范围大小由温度控制），并且在新状态更优时接受新状态，新状态更劣时以一定概率接受新状态。
* 建议在每一次 trial 花费时间不长，并且计算资源充足的情形下使用。
* 效果可能仅与随机搜索相当。

## Hyperband

## PBT

### 配置示例

```yaml
# config.yml
tuner:
  builtinTunerName: Anneal
  classArgs:
    optimize_mode: maximize
```

**参数**

* `optimize_mode`：若为 `maximize`，调参器会最大化指标；若为 `minimize`，调参器会最小化指标。

### 使用建议

# 贝叶斯优化Bayesian optimization

贝叶斯优化方法基于过去的评价结果构建一个概率模型，将一组超参数映射为目标函数的指标的概率分布，这一模型称为目标函数的“代理（surrogate）”函数。代理函数的优化比目标函数容易得多，因此我们将代理函数上最好的一组超参数配置作为目标函数下一次尝试的对象。换言之，用较少的时间优化代理函数得到（可能）更好的超参数配置，以避免目标函数在盲目尝试中花费更多的时间。由于贝叶斯优化可以利用过去的 trial 信息，其相比随机搜索更加有效，能在更短的时间内寻找到更好的参数。

## GP Tuner

GP（Gaussian Process）调参器实现了使用高斯过程回归的贝叶斯优化算法，是一种基于顺序模型的优化（Sequential Model-Based Optimization，SMBO）算法，参考论文 [Algorithms for Hyper-Parameter Optimization](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf)。此算法根据已有的观测样本点，通过高斯过程回归计算函数的后验分布，得到函数在每一个超参数取值点的期望和方差，期望越大表示该点的函数取值期望越大，方差越大表示该点可以探索的空间越大，算法会采取策略（使用采集函数）以权衡开发和探索。随着观测次数的增加，后验分布将不断得到改善，趋近于目标函数。更详细的介绍可以参考这篇[博客](https://zhuanlan.zhihu.com/p/29779000)。

> SMBO算法是贝叶斯优化的一种形式，其中顺序(sequential)表示trial一个接一个地运行，每一次trial尝试由应用贝叶斯推断并更新代理模型得到的一组更好的超参数。SMBO算法可描述为以下步骤：
>
> 1. 定义搜索空间、目标函数(target function)和目标函数的代理函数(surrogate function)；
> 2. 使用选择函数(selection function)根据当前代理函数选出下一组评估的超参数；
> 3. 使用这一组超参数计算目标函数，并根据已有的超参数与指标信息更新代理函数，回到2。
>
> SMBO算法的不同变体的差别主要在于代理函数以及选择函数的定义。代理函数通常使用高斯过程回归，随机森林回归和TPE(Tree Parzen Estimator)，选择函数通常使用期望改善、置信上限等。

### NNI配置示例

```json
// search_space.json
{
    "batch_size": {"_type": "choice", "_value": [16, 32, 64, 128]},
    "hidden_size": {"_type": "choice", "_value": [128, 256, 512, 1024]},
    "lr": {"_type": "choice", "_value": [0.0001, 0.001, 0.01, 0.1]},
    "momentum": {"_type": "uniform", "_value": [0, 1]}
}
```

搜索空间接受的参数类型包括 `randint`，`uniform`，`quniform`，`loguniform`，`qloguniform`，以及数值的 `choice`。

```yaml
# config.yml
tuner:
  builtinTunerName: GPTuner
  classArgs:
    optimize_mode: maximize
    utility: 'ei'
    kappa: 5.0
    xi: 0.0
    nu: 2.5
    alpha: 1e-6
    cold_start_num: 10
    selection_num_warm_up: 100000
    selection_num_starting_points: 250
```

**参数**

* `optimize_mode`：若为 `maximize`，调参器会最大化指标；若为 `minimize`，调参器会最小化指标。
* `utility`：工具函数（采集函数）的类型，`ei`，`ucb`，`poi` 分别为期望改进（Expected Improvement），置信上限（Upper Confidence Bound）和改进概率（Probability of Improvement）。默认为 `ei`。
* `kappa`：用于 `ucb` 函数，越大则调参器的探索性越强。默认为 5。
* `xi`：用于 `ei` 和 `poi` 函数，越大则调参器的探索性越强。默认为 0。
* `nu`：用于指定 Matern 核，越小则近似函数越不平滑。默认为 2.5。
* `alpha`：用于高斯过程回归器，越大则观察中的噪声水平越高。默认为 1e-6。
* `cold_start_num`：在高斯过程前（冷启动）执行随机探索的数量。随机探索可帮助提高探索空间的广泛性。默认为 10。
* `selection_num_warm_up`：为了获得采集函数的最大值而评估的随机点数量。默认为 1e5。
* `selection_num_starting_points`：……

### 使用方法与建议

* 搜索空间必须是连续区间或者离散的数值，不能是离散的类别，因为需要计算样本点之间的距离。
* 此算法的时间复杂度为 $O(n^3)$ ,其中 $n$ 为已观测的样本点数量，因此建议在运行少量 trial（几十到几百）的情形下使用。
* 在低维空间中，此算法的表现远远优于随机搜索；但在高维（几十维）空间中，此算法近乎于随机搜索，因为想要观测样本点布满整个搜索空间就需要指数数量的样本，但我们有的样本远远没有这么多，样本点之间距离比较远，几乎不能提供有用信息。因此此算法也不适用于超参数非常多的大规模系统。
* 此算法的朴素形式不支持并行。

## TPE

TPE（Tree-structured Parzen Estimator）算法来自于论文 [Algorithms for Hyper-Parameter Optimization](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf)，也是一种 SMBO 算法，但与使用高斯过程的贝叶斯回归不同的是，它使用贝叶斯公式为 $p(x|y)$ 而非 $p(y|x)$ 建立概率模型(这里 $x$ 指超参数配置, $y$ 指目标函数值),并根据阈值 $y^*$ 将 $p(x|y)$ 划分为两个分布 $l(x)$ 和 $g(x)$，取它们的商作为代理函数的优化目标。具体细节请参考原论文。

此算法处理树状结构的超参数空间，也就是参数间存在依赖关系，例如必须在指定神经网络的层数之后才能指定某一层的参数。

### NNI配置示例

```json
// search_space.json
{
    "batch_size": {"_type": "choice", "_value": [16, 32, 64, 128]},
    "hidden_size": {"_type": "choice", "_value": [128, 256, 512, 1024]},
    "lr": {"_type": "choice", "_value": [0.0001, 0.001, 0.01, 0.1]},
    "momentum": {"_type": "uniform", "_value": [0, 1]}
}
```

```yaml
# config.yml
tuner:
  builtinTunerName: TPE
  classArgs:
    optimize_mode: maximize
```

**参数**

* `optimize_mode`：若为 maximize，调参器会最大化指标；若为 minimize，调参器会最小化指标。

### AutoTune配置示例

```yaml
spec:
  searchSpace: |-
    {
        "batch_size": {"_type": "choice", "_value": [16, 32, 64, 128]},
        "hidden_size": {"_type": "choice", "_value": [128, 256, 512, 1024]},
        "lr": {"_type": "choice", "_value": [0.0001, 0.001, 0.01, 0.1]},
        "momentum": {"_type": "uniform", "_value": [0, 1]}
    }
  tuner:
    builtinTunerName: TPE
    classArgs:
      optimize_mode: maximize
```

### 使用建议

* 此算法适用于各种情形下，并且通常都能得到比较好的结果。特别是在计算资源有限，只能运行少量 trial 的情形下建议使用此算法。
* 此算法的并行化研究请参考 [NNI 文档](https://github.com/microsoft/nni/blob/master/docs/zh_CN/CommunitySharings/ParallelizingTpeSearch.rst)。

## SMAC

SMAC（Sequential Model-based Algorithm Configuration）算法来自于论文 [Sequential model-based optimization for general algorithm configuration](https://ml.informatik.uni-freiburg.de/papers/11-LION5-SMAC.pdf)，也是一种 SMBO 算法，其特征是可以处理类别超参数。此算法使用随机森林

## Metis Tuner

## BOHB

> 像遗传算法和PSO这些经典黑盒优化算法，我归类为群体优化算法，也不是特别适合模型超参数调优场景，因为需要有足够多的初始样本点，并且优化效率不是特别高，本文也不再详细叙述。
>
> 实际上Grid search和Random search都是非常普通和效果一般的方法，在计算资源有限的情况下不一定比建模工程师的个人经验要好，接下来介绍的Bayesian Optimization就是“很可能”比普通开发者或者建模工程师调参能力更好的算法。