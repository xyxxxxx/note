# 强化学习

## 参考

* [【機器學習2021】概述增強式學習 (Reinforcement Learning, RL) (一) – 增強式學習跟機器學習一樣都是三個步驟](https://www.youtube.com/watch?v=gmsMY5kc-zw&list=PLJV_el3uVTsMhtt7_Y6sgTHGHp1Vb2P2J&index=29)
* [【機器學習2021】概述增強式學習 (Reinforcement Learning, RL) (二) – Policy Gradient 與修課心情](https://www.youtube.com/watch?v=gmsMY5kc-zw&list=PLJV_el3uVTsMhtt7_Y6sgTHGHp1Vb2P2J&index=30)
* [【機器學習2021】概述增強式學習 (Reinforcement Learning, RL) (三) - Actor-Critic](https://www.youtube.com/watch?v=gmsMY5kc-zw&list=PLJV_el3uVTsMhtt7_Y6sgTHGHp1Vb2P2J&index=31)
* [【機器學習2021】概述增強式學習 (Reinforcement Learning, RL) (四) - 回饋非常罕見的時候怎麼辦？機器的望梅止渴](https://www.youtube.com/watch?v=gmsMY5kc-zw&list=PLJV_el3uVTsMhtt7_Y6sgTHGHp1Vb2P2J&index=32)
* [【機器學習2021】概述增強式學習 (Reinforcement Learning, RL) (五) - 如何從示範中學習？逆向增強式學習 (Inverse RL)](https://www.youtube.com/watch?v=gmsMY5kc-zw&list=PLJV_el3uVTsMhtt7_Y6sgTHGHp1Vb2P2J&index=33)

## 概论

![](https://s2.loli.net/2023/02/28/tLWgSkf2wcyKFoJ.png)

强化学习是一种通过与环境交互学习如何做出正确决策的机器学习方法，其核心概念包括：

* Actor：一个智能体或代理程序，用于在给定的环境中选择一个最佳的行动策略。Actor 可以是任何一种算法，如深度神经网络、决策树、线性模型等，其目标是找到一个最优的策略，使其能够在给定的环境下最大化预期的奖励。
* Environment（环境）：一个与 Actor 交互的外部系统或场景，它负责提供状态和奖励信号，同时接受 Actor 执行的动作并更新状态。环境可以是一个物理环境、一个虚拟仿真场景或一个游戏。
* Observation（观测）：Observation 是环境状态的一部分，也是 Actor 的输入。它是 Actor 通过感知环境来获取的信息，通常是由一组连续或离散的数值组成的向量。Actor 的目标是通过 Observation 来学习环境的动态特性并选择最佳的行动策略。
* Action（动作，行动）：Action 是 Actor 为了最大化奖励而选择的行动。它是由 Actor 根据当前环境状态和其内部策略计算出来的，并且会影响环境状态的转移。Action 可以是离散或连续的，取决于所解决的问题。
* Reward（奖励）：Reward 是环境向 Actor 提供的反馈信号，用于指导 Actor 选择正确的动作。Reward 通常是一个标量值，表示 Actor 在某个状态下执行某个动作的好坏程度。Actor 的目标是通过不断尝试最大化奖励值。

在每个时间步上，Actor 根据当前环境状态选择一个动作并执行，环境接受动作并将状态转换为新状态，并根据新状态提供一个奖励信号。

下图给出了玩电子游戏和下棋的示例：

![](https://s2.loli.net/2023/02/28/inwO5PNzp2Uv9Zd.png)

![](https://s2.loli.net/2023/02/28/vVybp5ts4ihTKHZ.png)

强化学习和机器学习一样都是三个步骤：

![](https://s2.loli.net/2023/02/28/9iNe2cg5jvfLTMm.png)

其中：

* 神经网络可以采用任何架构，如 CNN、Transformer 等。
* 通常基于分数采样而非直接取最大值，以提供动作的随机性（采用混合策略）。

![](https://s2.loli.net/2023/02/28/1jt956yQHcIsM4a.png)

![](https://s2.loli.net/2023/02/28/vBu4rdtc2iwqyA8.png)

其中：

* 一局游戏称为一个 episode；一个 episode 中得到的所有奖励求和即得到总奖励（total reward），也称为返回（return），优化目标即为最大化返回。

![](https://s2.loli.net/2023/02/28/GRfio2yDXjeCnlF.png)

![](https://s2.loli.net/2023/02/28/eDO7xcPiBL4qF3o.png)

![](https://s2.loli.net/2023/02/28/nAropikdVmRMK6Z.png)

![](https://s2.loli.net/2023/02/28/etnohDHpQux2Szw.png)

接下来的问题就是如何获取训练数据。我们随机初始化 Actor 网络的参数，用它去和环境互动，获得的数据作为训练数据去计算 $A_t$ 和 $L$，然后用梯度下降的方法更新参数，如此循环迭代。这里和非强化学习最大的不同是，每一次迭代都要重新搜集训练数据，因为一个（具有特定参数的）模型的训练数据不一定适用于另一个（具有特定参数的）模型。

![](https://s2.loli.net/2023/03/01/YhuWvFqHpwQtUl6.png)

![](https://s2.loli.net/2023/03/01/3iHUR8VXMcINfah.png)

搜集训练数据时应鼓励 Actor 多去尝试不同的动作：

![](https://s2.loli.net/2023/03/02/XoYiDOuBMR6pgEG.png)

训练的关键在于如何计算 $A_t$。A2C（Advantage Actor-Critic）方法给出的计算公式如下：

![](https://s2.loli.net/2023/03/02/WDTfIX5u9rnGOZc.png)

其中 $V^{\theta}(s_t)$ 称为价值函数（value function），其针对给定的 Actor，根据观测给出折减累计奖励的期望值。价值函数扮演的角色称为 Critic。

![](https://s2.loli.net/2023/03/02/ilK7Dvk69BSfymb.png)

价值函数的计算方法包括蒙特卡洛法和时序差分法：

![](https://s2.loli.net/2023/03/02/V8Xbqd5ZFt29SPl.png)

![](https://s2.loli.net/2023/03/02/tOaPj7rbERlK6F4.png)

## A3C

## Reward Shaping

### 论文

* [Curiosity-driven Exploration by Self-supervised Prediction (Pathak, 2017)](https://arxiv.org/abs/1705.05363)

## DQN

## PPO

## IRL
