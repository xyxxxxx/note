[toc]

# torch.optim

`torch.optim` 包实现了多种优化算法。最常用的优化方法已经得到支持，并且接口足够泛用，使得更加复杂的方法在未来也能够容易地集成进去。

## Adam

实现 Adam 算法。

```python
class torch.optim.Adam(
    params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, 
    amsgrad=False
)
# params        要优化的参数的可迭代对象,或定义了参数组的字典
# lr            学习率
# betas         用于计算梯度的移动平均和其平方的系数
# eps           添加到分母的项,用于提升数值稳定性
# weight_decay  权重衰退(L2惩罚)
# amsgrad       是否使用此算法的AMSGrad变体
```

## Optimizer

所有优化器的基类。

### add_param_group()

向优化器的 `param_groups` 添加一个参数组。这在精调一个预训练网络时十分有用。

```python
optimizer.add_param_group({'params': model.layer.parameters(), 'lr': 1e-3})
```

### load_state_dict()

加载优化器状态字典。

### param_groups

返回优化器的所有参数组。

```python
>>> w = torch.tensor([1.], requires_grad=True)
>>> b = torch.tensor([1.], requires_grad=True)
>>> x = torch.tensor([2.])
>>> y = torch.tensor([4.])
>>> z = w @ x + b
>>> l = (y - z)**2
>>> l.backward()
>>> w.grad
tensor([-4.])
>>> b.grad
tensor([-2.])
>>> optimizer = torch.optim.SGD([
    {'params': w},
    {'params': b, 'lr': 1e-3},
], lr=1e-2)
>>> optimizer.step()
>>> w
tensor([1.0400], requires_grad=True)
>>> b
tensor([1.0020], requires_grad=True)
>>> pprint(optimizer.param_groups)    # 两个参数组
[{'dampening': 0,
  'lr': 0.01,
  'momentum': 0,
  'nesterov': False,
  'params': [tensor([1.0400], requires_grad=True)],
  'weight_decay': 0},
 {'dampening': 0,
  'lr': 0.001,
  'momentum': 0,
  'nesterov': False,
  'params': [tensor([1.0020], requires_grad=True)],
  'weight_decay': 0}]
```

### state_dict()

返回优化器的状态为一个字典，其中包含两项：

+ `state`：包含当前优化状态的字典
+ `param_groups`：包含所有参数组的字典

### step()

执行单步优化。

### zero_grad()

将所有参数的梯度置零。

## SGD

实现随机梯度下降算法。

```python
class torch.optim.SGD(
    params, lr=<required parameter>, momentum=0, dampening=0, weight_decay=0, 
    nesterov=False
)
# params        要优化的参数的可迭代对象,或定义了参数组的字典
# lr            学习率
# momentum      动量系数
# weight_decay  权重衰退(L2惩罚)
# dampening     
# nesterov      启用Nesterov动量
```

```python
>>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
>>> optimizer.zero_grad()
>>> loss_fn(model(input), target).backward()
>>> optimizer.step()
```

## lr_scheduler

学习率规划器。

### _LRScheduler

所有学习率规划器的基类。

#### get_last_lr()

返回规划器计算的最后一个学习率。

#### load_state_dict()

加载规划器状态字典。

#### print_lr()

打印规划器的当前学习率。

#### state_dict()

返回规划器的状态为一个字典。

#### step()

更新学习率，具体操作取决于规划器的实现以及当前回合数。

### ChainedScheduler

```python
class torch.optim.lr_scheduler.ChainedScheduler(schedulers)
# schedulers  组合调用的规划器列表
```

将多个规划器组合为一个规划器，调用其 `step()` 方法相当于顺序调用其成员的 `step()` 方法。

```python
# Assuming optimizer uses lr = 1. for all groups
# lr = 0.09     if epoch == 0
# lr = 0.081    if epoch == 1
# lr = 0.729    if epoch == 2
# lr = 0.6561   if epoch == 3
# lr = 0.59049  if epoch >= 4
>>> scheduler1 = ConstantLR(self.opt, factor=0.1, total_iters=2)
>>> scheduler2 = ExponentialLR(self.opt, gamma=0.9)
>>> scheduler = ChainedScheduler([scheduler1, scheduler2])
>>> for epoch in range(100):
    train(...)
    validate(...)
    scheduler.step()
```

### ConstantLR

```python
class torch.optim.lr_scheduler.ConstantLR(
    optimizer, factor=0.3333333333333333, total_iters=5, last_epoch=-1, 
    verbose=False
)
# optimzer     包装的优化器
# factor       到达里程碑之前的学习率折减乘数
# total_iters  学习率折减的回合数
# last_epoch   最后一个回合的索引.若为`-1`,则此参数没有作用
# verbose      若为`True`,则每次更新学习率时向标准输出打印一条消息
```

前 `total_iters` 回合的学习率使用固定乘数进行折减。注意此折减可以与其它规划器引起的学习率变化同时发生。

```python
# Assuming optimizer uses lr = 0.05 for all groups
# lr = 0.025   if epoch == 0
# lr = 0.025   if epoch == 1
# lr = 0.025   if epoch == 2
# lr = 0.025   if epoch == 3
# lr = 0.05    if epoch >= 4
>>> scheduler = ConstantLR(optimizer, factor=0.5, total_iters=4)
>>> for epoch in range(100):
    train(...)
    validate(...)
    scheduler.step()
```

### CosineAnnealingLR

使用余弦退火算法设定学习率，……

### CosineAnnealingWarmRestarts

### CyclicLR

根据循环学习率策略设定学习率，此策略下学习率在两个边界之间以固定频率变化。

循环学习率策略在每个批次结束时都要改变学习率，因此 `step()` 应在每个批次训练完毕后调用。

```python
class torch.optim.lr_scheduler.CyclicLR(
    optimizer, base_lr, max_lr, step_size_up=2000, step_size_down=None, 
    mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle', 
    cycle_momentum=True, base_momentum=0.8, max_momentum=0.9, last_epoch=-1, 
    verbose=False
)
# optimzer    包装的优化器
# base_lr     初始学习率,同时是学习率的下界
# max_lr      学习率的上界
# step_size_up    循环上升期的训练批次数
# step_size_down  循环下降期的训练批次数
# mode        `'triangular'`,`'triangular2'`或`'exp_range'`,对应的三种模式见论文
#             https://arxiv.org/pdf/1506.01186.pdf
# gamma       `'exp_range'`模式下的学习率上下界的衰减乘数
# scale_fn    自定义缩放策略,由接收单个参数的匿名函数定义.若指定了此参数,则`mode`参数将被忽略
#             (到底缩放的是上界还是下界?)
# scale_mode  若为`'cycle'`,则`scale_fn`接收的参数视为回合数;若为`'iterations'`,
#             则`scale_fn`接收的参数视为批次数
# cycle_momentum  若为`True`,则动量与学习率反相循环
# base_momentum   动量的下界
# max_momentum    动量的上界
# last_epoch  最后一个回合的索引.若为`-1`,则此参数没有作用
# verbose     若为`True`,则每次更新学习率时向标准输出打印一条消息
```

### ExponentialLR

```python
class torch.optim.lr_scheduler.ExponentialLR(
    optimizer, gamma, last_epoch=-1, verbose=False
)
# optimzer    包装的优化器
# gamma       学习率衰减乘数
# last_epoch  最后一个回合的索引.若为`-1`,则此参数没有作用
# verbose     若为`True`,则每次更新学习率时向标准输出打印一条消息
```

每回合学习率衰减为原来的 `gamma` 倍。

### LambdaLR

```python
class torch.optim.lr_scheduler.LambdaLR(
    optimizer, lr_lambda, last_epoch=-1, verbose=False
)
# optimzer    包装的优化器
# lr_lambda   接收一个整数参数(回合数)并返回一个乘数的自定义函数,或为每组参数分别指定的自定义函数列表
# last_epoch  最后一个回合的索引.若为`-1`,则此参数没有作用
# verbose     若为`True`,则每次更新学习率时向标准输出打印一条消息
```

每回合学习率设定为原来的自定义函数返回值的倍数。

```python
>>> lambda1 = lambda epoch: epoch // 30
>>> lambda2 = lambda epoch: 0.95 ** epoch
>>> scheduler = LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])
                # 此优化器有两组参数,两个函数分别对应一组
>>> for epoch in range(100):
    train(...)
    validate(...)
    scheduler.step()
```

### LinearLR

```python
class torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.3333333333333333, end_factor=1.0, total_iters=5, 
    last_epoch=- 1, verbose=False
)
# optimzer      包装的优化器
# start_factor  第一个回合的学习率折减乘数,该乘数会在接下来的回合中向着`end_factor`线性变化
# end_factor    最终的学习率折减乘数
# total_iters   学习率折减的回合数
# last_epoch  最后一个回合的索引.若为`-1`,则此参数没有作用
# verbose     若为`True`,则每次更新学习率时向标准输出打印一条消息
```

前 `total_iters` 回合的学习率使用线性变化的乘数进行折减。注意此折减可以与其它规划器引起的学习率变化同时发生。

```python
# Assuming optimizer uses lr = 0.1 for all groups
# lr = 0.05     if epoch == 0
# lr = 0.0625   if epoch == 1
# lr = 0.075    if epoch == 2
# lr = 0.0875   if epoch == 3
# lr = 0.1      if epoch >= 4
>>> scheduler = LinearLR(self.opt, start_factor=0.5, total_iters=4)
>>> for epoch in range(100):
    train(...)
    validate(...)
    scheduler.step()
```

### MultiplicativeLR

```python
class torch.optim.lr_scheduler.MultiplicativeLR(
    optimizer, lr_lambda, last_epoch=-1, verbose=False
)
# optimzer    包装的优化器
# lr_lambda   接收一个整数参数(回合数)并返回一个乘数的自定义函数,或为每组参数分别指定的自定义函数列表
# last_epoch  最后一个回合的索引.若为`-1`,则此参数没有作用
# verbose     若为`True`,则每次更新学习率时向标准输出打印一条消息
```

每回合学习率设定为原来的自定义函数返回值的倍数。

```python
>>> lambda1 = lambda epoch: epoch // 30
>>> lambda2 = lambda epoch: 0.95 ** epoch
>>> scheduler = MultiplicativeLR(optimizer, lr_lambda=[lambda1, lambda2])  
                # 此优化器有两组参数,两个函数分别对应一组
>>> for epoch in range(100):
    train(...)
    validate(...)
    scheduler.step()
```

### MultiStepLR

```python
class torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones, gamma=0.1, last_epoch=-1, verbose=False
)
# optimzer    包装的优化器
# milestones  作为里程碑的回合索引列表,必须是递增的
# gamma       学习率衰减乘数
# last_epoch  最后一个回合的索引.若为`-1`,则此参数没有作用
# verbose     若为`True`,则每次更新学习率时向标准输出打印一条消息
```

每当回合数到达里程碑之一时学习率衰减为原来的 `gamma` 倍。注意此衰减可以与其它规划器引起的学习率变化同时发生。

```python
# Assuming optimizer uses lr = 0.05 for all groups
# lr = 0.05     if epoch < 30
# lr = 0.005    if 30 <= epoch < 80
# lr = 0.0005   if epoch >= 80
>>> scheduler = MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)
>>> for epoch in range(100):
    train(...)
    validate(...)
    scheduler.step()
```

### OneCycleLR

### SequentialLR

```python
class torch.optim.lr_scheduler.SequentialLR(
    optimizer, schedulers, milestones, last_epoch=-1, verbose=False
)
# optimzer    包装的优化器
# schedulers  顺序调用的规划器列表
# milestones  作为里程碑的回合索引列表,必须是递增的
# last_epoch  最后一个回合的索引.若为`-1`,则此参数没有作用
# verbose     若为`True`,则每次更新学习率时向标准输出打印一条消息
```

顺序调用多个规划器，每当回合数到达里程碑之一时切换为下一个规划器。

```python
# Assuming optimizer uses lr = 1. for all groups
# lr = 0.1     if epoch == 0
# lr = 0.1     if epoch == 1
# lr = 0.9     if epoch == 2
# lr = 0.81    if epoch == 3
# lr = 0.729   if epoch == 4
>>> scheduler1 = ConstantLR(optimizer, factor=0.1, total_iters=2)
>>> scheduler2 = ExponentialLR(optimizer, gamma=0.9)
>>> scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[2])
>>> for epoch in range(100):
    train(...)
    validate(...)
    scheduler.step()
```

### StepLR

```python
class torch.optim.lr_scheduler.StepLR(
    optimizer, step_size, gamma=0.1, last_epoch=-1, verbose=False
)
# optimzer    包装的优化器
# step_size   学习率衰减周期
# gamma       学习率衰减乘数
# last_epoch  最后一个回合的索引.若为`-1`,则此参数没有作用
# verbose     若为`True`,则每次更新学习率时向标准输出打印一条消息
```

每 `step_size` 回合学习率衰减为原来的 `gamma` 倍。注意此衰减可以与其它规划器引起的学习率变化同时发生。

```python
# Assuming optimizer uses lr = 0.05 for all groups
# lr = 0.05     if epoch < 30
# lr = 0.005    if 30 <= epoch < 60
# lr = 0.0005   if 60 <= epoch < 90
# ...
>>> scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
>>> for epoch in range(100):
    train(...)
    validate(...)
    scheduler.step()
```

### ReduceLROnPlateau

当指标不再改善时降低学习率。每当学习停滞时，降低学习率为原来的二到十分之一一般都能够改善模型。此规划器读取一个指标的值，并在若干个回合内没有看到改善时降低学习率。

```python
class torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001, 
    threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False
)
# optimzer    包装的优化器
# mode        若为`min`,则在监视的量不再减小时降低学习率;若为`max`,则在监视的量不再增加时降低学习率
# factor      学习率衰减乘数
# patience    等待的没有改善的回合数.例如此参数为2,则会忽略前2次指标没有改善,而在第3次指标仍没有改善时降低学习率
# threshold   可以被视作指标改善的阈值
# threshold_mode  若为`rel`,则动态阈值为`best*(1+threshold)`(对于`mode=max`)或`best*(1-threshold)`(对于`mode=min`)
#                 若为`abs`,则动态阈值为`best+threshold`(对于`mode=max`)或`best-threshold`(对于`mode=min`)
# cooldown    降低学习率之后再次恢复工作的冷却回合数
# min_lr      学习率的下限,或为每组参数分别指定的列表
# eps         应用于学习率的最小衰减,若新旧学习率之间的差值小于此参数,则忽略此次更新
# verbose     若为`True`,则每次更新学习率时向标准输出打印一条消息
```
