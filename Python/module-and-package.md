[toc]

# 模块

一个 `.py` 源文件就是一个模块（module）。

## 导入模块

```python
import math	     # 导入math包，即执行该文件
import math as m # 导入math包并在本地赋名
from math import cos, sin # 导入math包并将cos, sin添加到本地命名空间

x = math.sin(math.pi/2)
```

> Python标准库https://docs.python.org/zh-cn/3/library/index.html



### 导入\__future__模块

`__future__` 模块包含了一些将在未来 Python 版本中成为标准特性的语法，让我们可以在之前的版本中提前使用该特性。它的目的是使得向引入了不兼容改变的新版本迁移更加容易。

```python
# in python 2.x
from __future__ import division
print 8 / 7    # 除法
print 8 // 7   # 整数除法取商
```

导入 `__future__` 模块的语句必须位于模块的顶部，在模块的 `docstring` 和注释之后，其它 `import` 语句之前。

`__future__` 模块包含以下特性：

| 特性名称         | 引入版本 | 强制版本 | 效果简介          | 详细的 PEP 文章                                       |
| :--------------- | :------- | :------- | :---------------- | :---------------------------------------------------- |
| nested_scopes    | 2.1.0b1  | 2.2      | 嵌套作用域        | [PEP-0227](https://www.python.org/dev/peps/pep-0227/) |
| generators       | 2.2.0a1  | 2.3      | 生成器语法        | [PEP-0255](https://www.python.org/dev/peps/pep-0255/) |
| division         | 2.2.0a2  | 3.0      | 强制浮点数除法    | [PEP-0238](https://www.python.org/dev/peps/pep-0238/) |
| absolute_import  | 2.5.0a1  | 3.0      | 绝对引入          | [PEP-0328](https://www.python.org/dev/peps/pep-0328/) |
| with_statement   | 2.5.0.a1 | 2.6      | with 声明         | [PEP-0343](https://www.python.org/dev/peps/pep-343/)  |
| print_function   | 2.6.0a2  | 3.0      | 强制 print 为函数 | [PEP-3105](https://www.python.org/dev/peps/pep-3105/) |
| unicode_literals | 2.6.0a2  | 3.0      | 默认为 unicode    | [PEP-3112](https://www.python.org/dev/peps/pep-3112/) |
| generator_stop   | 3.5.0b1  | 3.7      | 终止生成器        | [PEP-0479](https://www.python.org/dev/peps/pep-0479/) |
| annotations      | 3.7.0b1  | 3.10     | 注解              | [PEP-0563](https://www.python.org/dev/peps/pep-0563/) |



## 定义模块

```python
#!/usr/bin/env python3			# 标准注释:py3文件
# -*- coding: utf-8 -*-			# 标准注释:使用UTF-8编码

"""a test module."""			  # 文档注释

__author__ = 'Michael Liao'	 # 作者名

import sys						      # 正文

def test():
    args = sys.argv
    if len(args)==1:
        print('Hello, world!')
    elif len(args)==2:
        print('Hello, %s!' % args[1])
    else:
        print('Too many arguments!')

if __name__=='__main__':        # 执行该模块时运行
    test()
```



## 作用域

```python
# abc		public变量
# _abc		public变量，但惯例不直接引用
# __abc		private变量，不可直接引用
# __abc__	特殊变量，可以直接引用

def _private_1(name):		# 内部函数
    return 'Hello, %s' % name

def _private_2(name):		
    return 'Hi, %s' % name

def greeting(name):			# 外部接口
    if len(name) > 3:
        return _private_1(name)
    else:
        return _private_2(name)
```





# 包

对于更大规模的库文件，通常的做法是将模块组织成包（package）。

```python
# from this
util1.py
util2.py
util3.py

# to this
util/
    __init__.py
    util1.py
    util2.py
    util3.py
    
# use package
import util
util.util1.func1()

from util import util1
util1.func1()

from util.util1 import func1
func1()
```

同一包中各个模块的互相导入方法需要改变：

```python
# from this
# util1.py
import util2

# to this
# util1.py
from . import util2
```

运行包中这些模块的命令也需要改变：

```shell
# from this
$ python3 util/util1.py

# to this
$ python3 -m util.util1
```

```python
# __init__.py
from .util1 import func1

# 上述声明使func1成为util命名空间下的顶级名称
import util
util.func1()

from util import func1
func1()
```





# 文档字符串示例

```python
class WandbLogger(LightningLoggerBase):
    r"""
    Log using `Weights and Biases <https://www.wandb.com/>`_.     # 功能
    
    Install it with pip:
    
    .. code-block:: bash
    
        pip install wandb
        
    Args:                     # 参数
        name: Display name for the run.
        save_dir: Path where data is saved (wandb dir by default).
        offline: Run offline (data can be streamed later to wandb servers).
        id: Sets the version, mainly used to resume a previous run.
        version: Same as id.
        anonymous: Enables or explicitly disables anonymous logging.
        project: The name of the project to which this run will belong.
        log_model: Save checkpoints in wandb dir to upload on W&B servers.
        prefix: A string to put at the beginning of metric keys.
        experiment: WandB experiment object. Automatically set when creating a run.
        \**kwargs: Additional arguments like `entity`, `group`, `tags`, etc. used by
            :func:`wandb.init` can be passed as keyword arguments in this logger.
            
    Raises:                   # 可能引起的错误及其原因
        ImportError:
            If required WandB package is not installed on the device.
        MisconfigurationException:
            If both ``log_model`` and ``offline``is set to ``True``.
            
    Example::                 # 使用示例
        from pytorch_lightning.loggers import WandbLogger
        from pytorch_lightning import Trainer
        wandb_logger = WandbLogger()
        trainer = Trainer(logger=wandb_logger)

    Note:                     # 注意事项
    When logging manually through `wandb.log` or `trainer.logger.experiment.log`,
    make sure to use `commit=False` so the logging step does not increase.
    
    See Also:                 # 参见
        - `Tutorial <https://colab.research.google.com/drive/16d1uctGaw2y9KhGBlINNTsWpmlXdJwRW?usp=sharing>`__
          on how to use W&B with PyTorch Lightning
        - `W&B Documentation <https://docs.wandb.ai/integrations/lightning>`__
    """
```

```python
class Adam(optimizer_v2.OptimizerV2):
  r"""Optimizer that implements the Adam algorithm.              # 功能
  
  Adam optimization is a stochastic gradient descent method that is based on   # 算法介绍
  adaptive estimation of first-order and second-order moments.
  
  According to [Kingma et al., 2014](http://arxiv.org/abs/1412.6980),
  the method is "*computationally efficient, has little memory requirement, 
  invariant to diagonal rescaling of gradients, and is well suited for 
  problems that are large in terms of data/parameters*".
  
  Args:                     # 参数
    learning_rate: A `Tensor`, floating point value, or a schedule that is a
      `tf.keras.optimizers.schedules.LearningRateSchedule`, or a callable
      that takes no arguments and returns the actual value to use, The
      learning rate. Defaults to 0.001.
    beta_1: A float value or a constant float tensor, or a callable
      that takes no arguments and returns the actual value to use. The
      exponential decay rate for the 1st moment estimates. Defaults to 0.9.
    beta_2: A float value or a constant float tensor, or a callable
      that takes no arguments and returns the actual value to use, The
      exponential decay rate for the 2nd moment estimates. Defaults to 0.999.
    epsilon: A small constant for numerical stability. This epsilon is
      "epsilon hat" in the Kingma and Ba paper (in the formula just before
      Section 2.1), not the epsilon in Algorithm 1 of the paper. Defaults to
      1e-7.
    amsgrad: Boolean. Whether to apply AMSGrad variant of this algorithm from
      the paper "On the Convergence of Adam and beyond". Defaults to `False`.
    name: Optional name for the operations created when applying gradients.
      Defaults to `"Adam"`.
    **kwargs: Keyword arguments. Allowed to be one of
      `"clipnorm"` or `"clipvalue"`.
      `"clipnorm"` (float) clips gradients by norm; `"clipvalue"` (float) clips
      gradients by value.
      
  Usage:
  >>> opt = tf.keras.optimizers.Adam(learning_rate=0.1)
  >>> var1 = tf.Variable(10.0)
  >>> loss = lambda: (var1 ** 2)/2.0       # d(loss)/d(var1) == var1
  >>> step_count = opt.minimize(loss, [var1]).numpy()
  >>> # The first step is `-learning_rate*sign(grad)`
  >>> var1.numpy()
  9.9
  Reference:
    - [Kingma et al., 2014](http://arxiv.org/abs/1412.6980)
    - [Reddi et al., 2018](
        https://openreview.net/pdf?id=ryQu7f-RZ) for `amsgrad`.
  Notes:
  The default value of 1e-7 for epsilon might not be a good default in
  general. For example, when training an Inception network on ImageNet a
  current good choice is 1.0 or 0.1. Note that since Adam uses the
  formulation just before Section 2.1 of the Kingma and Ba paper rather than
  the formulation in Algorithm 1, the "epsilon" referred to here is "epsilon
  hat" in the paper.
  The sparse implementation of this algorithm (used when the gradient is an
  IndexedSlices object, typically because of `tf.gather` or an embedding
  lookup in the forward pass) does apply momentum to variable slices even if
  they were not used in the forward pass (meaning they have a gradient equal
  to zero). Momentum decay (beta1) is also applied to the entire momentum
  accumulator. This means that the sparse behavior is equivalent to the dense
  behavior (in contrast to some momentum implementations which ignore momentum
  unless a variable slice was actually used).
  """
```

