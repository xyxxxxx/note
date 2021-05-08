[toc]

PyTorch Lightning 是基于 PyTorch 的一个研究框架，旨在：

* 保留 PyTorch 代码的灵活性，而去掉大量的模板内容
* 通过将代码的研究部分和工程部分解耦，使得代码更易读
* 更容易复现
* 通过自动化复杂的工程细节使得更不容易出错
* 可以在任何硬件上缩放而无需修改模型



# 入门示例

```python
import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
```



## 1.定义LightningModule

```python
# Define a LightningModule
class LitAutoEncoder(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 28*28)
        )

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        # Logging to TensorBoard by default
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
```

一个 LightningModule 定义的不仅是一个 <u> 模型 </u>，而是一个 <u> 系统 </u>：

<img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/model_system.png"style="zoom：50%;"/>

LightningModule 的底层依然只是一个 `torch.nn.Module`，它汇集了所有研究代码到一个类中，使得该类各部分齐全：

* 训练循环
* 验证循环
* 测试循环
* 模型或模型结构
* 优化器



## 2.使用Lightning Trainer训练

```python
dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
train_loader = DataLoader(dataset)

# init model
autoencoder = LitAutoEncoder()

# most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
# trainer = pl.Trainer(gpus=8) (if you have GPUs)
trainer = pl.Trainer()
trainer.fit(autoencoder, train_loader)
```

实例化模型和 `trainer`，然后调用 `fit` 方法传入模型和数据。数据只需要传入一个 `Dataloader` 对象。

`trainer` 自动化下列步骤：

* epoch 和 batch 迭代
* 调用 `optimizer.step(),backward,zero_grad()`
* 调用 `.eval()`
* 启用/禁用梯度计算
* 保存和加载模型参数
* TensorBoard
* 多 GPU 训练支持
* TPU 训练支持
* 16 位训练支持





# 基本特性

`LightningModule` 和 `Trainer` 是你唯二需要知道的两个概念，下面的所有内容都是 `LightningModule` 或 `Trainer` 的特性。



## 日志

TensorBoard 需要日志，我们使用 `log()` 方法作为日志器和进度条，其可以在 `LightningModule` 的任何方法中，对需要记录的指标调用：

```python
def training_step(self, batch, batch_idx):
    self.log('my_metric', x)
```

`log()` 方法有一些选项：

* `on_step`（logs the metric at that step in training）
* `on_epoch`（automatically accumulates and logs at the end of the epoch）
* `prog_bar`（logs to the progress bar）
* `logger`（logs to the logger like TensorBoard）

取决于 `log` 被调用的位置，Lightning 会自动决定纠正模式，当然你也可以人工设置 flag 以重载默认行为：

```python
def training_step(self, batch, batch_idx):
    self.log('my_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
```

> 进度条中显式的损失值根据最近几次的值进行了平滑，因此其与训练/验证step的真实损失有差异。

一旦训练开始，你就可以查看日志：

```shell
tensorboard --logdir ./lightning_logs
```



## 验证

如同 `training_step` 一样，我们可以定义 `validation_step` 来检查参数，并将其添加到日志中：

```python
def validation_step(self, batch, batch_idx):
    x, y = batch
    logits = self(x)
    loss = F.nll_loss(logits, y)
    self.log('val_loss', loss)        
```

现在训练中就加入了验证循环：

```python
from pytorch_lightning import Trainer

model = LitMNIST()
trainer = Trainer(tpu_cores=8)
trainer.fit(model, train_loader, val_loader)
# trainer.fit(model)      # 如果LightningModule包含了数据处理的方法
# trainer.fit(model, dm)  # 如果使用LightningDataModule
```

你可能注意到日志中记录了 `Validation Sanity Check`，这时因为 Lightning 在训练开始之前就运行了验证集的 2 个 batch，以确定验证过程没有 bug，否则你可能要等到 1 个完整的 epoch 训练完才能发现。

验证过程的底层实现相当于：

```python
model = Model()

for epoch in epochs:
    # train
    model.train()
	torch.set_grad_enabled(True)
    
    for batch in data:
        # ...

    # validate
    model.eval()
    torch.set_grad_enabled(False)

    outputs = []
    for batch in val_data:
        x, y = batch                        # validation_step
        y_hat = model(x)                    # validation_step
        loss = loss(y_hat, x)               # validation_step
        outputs.append({'val_loss': loss})  # validation_step

    total_loss = outputs.mean()             # validation_epoch_end
```

如果想要更精细地控制验证过程，可以使用[数据流](#数据流)部分的钩子。



## 测试

如同 `validation_step` 一样，定义 `test_step`：

```python
def test_step(self, batch, batch_idx):
    x, y = batch
    logits = self(x)
    loss = F.nll_loss(logits, y)
    self.log('test_loss', loss)
```

Lightning 使用了另一个 API 用于测试。如果在 `.fit()` 之后调用 `.test()`，则无需再传入 model，并且它会选择最好的检查点存档（根据 `val_loss`）：

```python
from pytorch_lightning import Trainer

model = LitMNIST()
trainer = Trainer(tpu_cores=8)
trainer.fit(model, train_loader, val_loader)

trainer.test(test_dataloaders=test_loader)
# trainer.test()               # 如果LightningModule包含了数据处理的方法
# trainer.test(datamodule=dm)  # 如果使用LightningDataModule
```



## 预测或部署

训练好的模型可以用于预测：

```python
model = LitMNIST.load_from_checkpoint('path/to/checkpoint_file.ckpt')
x = torch.randn(1, 1, 28, 28)
out = model(x)
```

与 PyTorch 相同，`model(x)` 调用的是 `forward()` 方法。

`forward` 和 `training_step` 的区别在于，Lightning 在设计时将训练和预测分离，其中 `training_step` 定义了完整的训练循环，`forward` 定义了预测过程的操作。`training_step` 经常会调用 `forward`（通过 `self(x)`），但我们依然推荐分离这两种目标。



对于生产环境，使用 `onnx` 或 `torchscript` 会快很多。确认你添加了 `forward` 方法或使用了需要的子模型。

```python
# ----------------------------------
# torchscript
# ----------------------------------
autoencoder = LitAutoEncoder()
torch.jit.save(autoencoder.to_torchscript(), "model.pt")
os.path.isfile("model.pt")
```

```python
# ----------------------------------
# onnx
# ----------------------------------
with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmpfile:
     autoencoder = LitAutoEncoder()
     input_sample = torch.randn((1, 28 * 28))
     autoencoder.to_onnx(tmpfile.name, input_sample, export_params=True)
     os.path.isfile(tmpfile.name)
```



## 使用CPUs/GPUs/TPUs

Lightning 中使用 CPU、GPU、TPU 非常简单。你无需修改任何代码，只需要修改 `Trainer` 的选项：

```python
# train on CPU
trainer = pl.Trainer()

# train on 8 CPUs
trainer = pl.Trainer(num_processes=8)

# train on 1024 CPUs across 128 machines
trainer = pl.Trainer(
    num_processes=8,
    num_nodes=128
)
```

```python
# train on 1 GPU
trainer = pl.Trainer(gpus=1)

# train on multiple GPUs across nodes (32 gpus here)
trainer = pl.Trainer(gpus=4, num_nodes=8)

# train on gpu 1, 3, 5 (3 gpus total)
trainer = pl.Trainer(gpus=[1, 3, 5])

# Multi GPU with mixed precision
trainer = pl.Trainer(gpus=2, precision=16)
```

```python
# Train on TPUs
trainer = pl.Trainer(tpu_cores=8)
```



## 检查点

Lightning 自动保存模型的检查点。你可以这样加载检查点：

```python
model = LitAutoEncoder.load_from_checkpoint(path)
```



## 数据流

每个（训练、验证、测试）循环有 3 个钩子（回调）可以实现：

* `[training/validation/test]_step`
* `[training/validation/test]_step_end`
* `[training/validation/test]_epoch_end`

对于训练循环：

```python
outs = []
for batch in data:
    out = training_step(batch)
    outs.append(out)
training_epoch_end(outs)
```

其在 Lightning 中等价于：

```python
def training_step(self, batch, batch_idx):
    loss = ...
    return loss

def training_epoch_end(self, training_step_outputs):
    for loss in training_step_outputs:
        # do something with these
```

当你使用 DP 或 DDP 分布式模式时（即将一个 batch 划分给多个 GPU），可以用 `training_step_end` 手动合并（也可以不实现此方法，Lightning 会自动合并）：

```python
for batch in data:
    model_copies = copy_model_per_gpu(model, num_gpus)
    batch_split = split_batch_per_gpu(batch, num_gpus)

    gpu_outs = []
    for model, batch_part in zip(model_copies, batch_split):
        # LightningModule hook
        gpu_out = model.training_step(batch_part)
        gpu_outs.append(gpu_out)

    # LightningModule hook
    out = training_step_end(gpu_outs)
```

其在 Lightning 中等价于：

```python
def training_step(self, batch, batch_idx):
    loss = ...
    return loss

def training_step_end(self, losses):
    gpu_0_loss = losses[0]
    gpu_1_loss = losses[1]
    return (gpu_0_loss + gpu_1_loss) * 1/2
```





## 回调

回调是一个 self-contained 的程序，可以执行在训练循环中的任何位置。

下面是一个学习率衰减的规则：

```python
class DecayLearningRate(pl.callbacks.Callback):

    def __init__(self):
        self.old_lrs = []

    def on_train_start(self, trainer, pl_module):
        # track the initial learning rates
        for opt_idx, optimizer in enumerate(trainer.optimizers):
            group = [param_group['lr'] for param_group in optimizer.param_groups]
            self.old_lrs.append(group)

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        for opt_idx, optimizer in enumerate(trainer.optimizers):
            old_lr_group = self.old_lrs[opt_idx]
            new_lr_group = []
            for p_idx, param_group in enumerate(optimizer.param_groups):
                old_lr = old_lr_group[p_idx]
                new_lr = old_lr * 0.98
                new_lr_group.append(new_lr)
                param_group['lr'] = new_lr
            self.old_lrs[opt_idx] = new_lr_group
```

将其作为回调传递给 `trainer`：

```
decay_callback = DecayLearningRate()
trainer = Trainer(callbacks=[decay_callback])
```

回调的所有钩子见 [Callback](https://pytorch-lightning.readthedocs.io/en/stable/callbacks.html#callbacks)。

回调可以帮助你做这些事情：

* 在训练的某个节点发送 email
* 更新学习率
* 可视化梯度
* ……（任何事情）



## LightningDataModules

`Dataloader` 和数据处理代码也可能分散在各处，可以将它们组织在一个 `LightningDataModules` 中，让代码复用性更好：

```python
class MNISTDataModule(pl.LightningDataModule):

      def __init__(self, batch_size=32):
          super().__init__()
          self.batch_size = batch_size

      # When doing distributed training, Datamodules have two optional arguments for
      # granular control over download/prepare/splitting data:

      # OPTIONAL, called only on 1 GPU/machine
      def prepare_data(self):
          MNIST(os.getcwd(), train=True, download=True)
          MNIST(os.getcwd(), train=False, download=True)

      # OPTIONAL, called for every GPU/machine (assigning state is OK)
      def setup(self, stage):
          # transforms
          transform=transforms.Compose([
              transforms.ToTensor(),
              transforms.Normalize((0.1307,), (0.3081,))
          ])
          # split dataset
          if stage == 'fit':
              mnist_train = MNIST(os.getcwd(), train=True, transform=transform)
              self.mnist_train, self.mnist_val = random_split(mnist_train, [55000, 5000])
          if stage == 'test':
              self.mnist_test = MNIST(os.getcwd(), train=False, transform=transform)

      # return the dataloader for each split
      def train_dataloader(self):
          mnist_train = DataLoader(self.mnist_train, batch_size=self.batch_size)
          return mnist_train

      def val_dataloader(self):
          mnist_val = DataLoader(self.mnist_val, batch_size=self.batch_size)
          return mnist_val

      def test_dataloader(self):
          mnist_test = DataLoader(self.mnist_test, batch_size=self.batch_size)
          return mnist_test
```

`LightningDataModules` 被设计用于在不同的项目之间分享或重用数据划分和处理的代码。它封装了数据处理的所有步骤：下载、分词、提取词干等等。

现在你可以直接将 `LightningDataModules` 传入 `Trainer`：

```python
# init model
model = LitAutoEncoder()

# init data
dm = MNISTDataModule()

# train
trainer = pl.Trainer()
trainer.fit(model, dm)

# test
trainer.test(datamodule=dm)
```

`LightningDataModules` 对于构建基于数据的模型十分有用。



## 手动 vs 自动优化

在 Lightning 中，你无需操心何时启用/禁用梯度计算、做反向传播或优化器更新，只要你在 `training_step` 返回一个带有计算图的损失项，Lightning 会自动化优化过程：

```python
def training_step(self, batch, batch_idx):
    loss = self.encoder(batch[0])
    return loss
```

尽管如此，对于特定的研究如 GAN、强化学习，或者模型具有多个优化器或具有内部循环，你可以关闭自动优化并完全控制循环：

```python
trainer = Trainer(automatic_optimization=False)

def training_step(self, batch, batch_idx, opt_idx):
    # access your optimizers with use_pl_optimizer=False. Default is True
    (opt_a, opt_b, opt_c) = self.optimizers(use_pl_optimizer=True)

    loss_a = self.generator(batch[0])

    # use this instead of loss.backward so we can automate half precision, etc...
    self.manual_backward(loss_a, opt_a, retain_graph=True)
    self.manual_backward(loss_a, opt_a)
    opt_a.step()
    opt_a.zero_grad()

    loss_b = self.discriminator(batch[0])
    self.manual_backward(loss_b, opt_b)
    ...
```

……





# 将PyTorch模型转换为LightningModule

参考 [How to organize PyTorch into Lightning](https://pytorch-lightning.readthedocs.io/en/latest/starter/converting.html)





# 最佳实践

## 代码风格

Lightning 的一个主要目标是提高可读性和复现能力。这一部分意在鼓励 Lightning 代码有相似的结构。



### self-contained

一个 `LightningModule` 对象应该是 self-contained 的。换言之，用户可以将 `LightningModule` 对象传入一个 `Trainer` 而无需知晓其内部构造。



### init

对于 `init` 方法，我们应该显式地定义所有参数，打消用户对于这些重要参数的疑问，例如：

```python
class LitModel(pl.LightningModule):
    def __init__(self, encoder: nn.Module, coeff_x: float = 0.2, lr: float = 1e-3)
```

这样用户就一目了然参数的类型，以及可以用作参考的默认值。



### 方法顺序

对于一个 `LightningModule` 的完整实现，各方法的推荐顺序是：

```python
class LitModel(pl.LightningModule):

    def __init__(...):

    def forward(...):

    def training_step(...)

    def training_step_end(...)

    def training_epoch_end(...)

    def validation_step(...)

    def validation_step_end(...)

    def validation_epoch_end(...)

    def test_step(...)

    def test_step_end(...)

    def test_epoch_end(...)

    def configure_optimizers(...)

    def any_extra_hook(...)
```



### forward vs training_step





### 数据

Lightning 使用 dataloader 处理所有的数据流。





# MNIST示例

参考 ml/lightning/tutorial_mnist.ipynb





# LightningModule

> https://pytorch-lightning.readthedocs.io/en/stable/lightning_module.html





# Trainer

> https://pytorch-lightning.readthedocs.io/en/stable/trainer.html





# 日志

Lightning 支持最流行的日志框架（TensorBoard，Comet 等）。使用一个日志器，只需要将其传入 Trianer 的构造函数。Lightning 默认使用 TensorBoard。

使用 TensorBoard 作为日志器：

```python
from pytorch_lightning.loggers import TensorBoardLogger

tb_logger = TensorBoardLogger('logs/')
trainer = Trainer(logger=tb_logger)
```

> 事实上trainer使用的默认日志器就是：
>
> ```python
> from pytorch_lightning.loggers import TensorBoardLogger
> 
> # default logger used by trainer
> logger = TensorBoardLogger(
>     save_dir=os.getcwd(),
>     version=0,
>     name='lightning_logs'
> )
> Trainer(logger=logger)
> ```

Lightning 同样支持 MLflow，Comet，Neptune，WandB 等主流日志器：

```python
from pytorch_lightning.loggers import CometLogger

comet_logger = CometLogger(save_dir='logs/')
trainer = Trainer(logger=comet_logger)
```

使用多个日志器，只需要传入一个日志器的列表或元组：

```python
trainer = Trainer(logger=[tb_logger, comet_logger])
```

> 可以通过设置trainer的`default_root_dir`和`logger`修改日志的保存路径：
>
> + 如果你既不设置`default_root_dir`，也不设置`logger`，trainer会使用上面的默认日志器，保存路径为`./lightning_logs/version0/`
> + 如果你设置了`default_root_dir='mydir/'`，trainer依然会使用上面的默认日志器，保存路径为`./mydir/lightning_logs/version0/`
>
> + 如果你设置日志器`logger = TensorBoardLogger(save_dir='savedir/', version=1, name='mylogs')`，保存路径为`./savedir/mylogs/version1/`
> + 注意设置`logger`会重载`default_root_dir`，即`default_root_dir`无效



## 在LightningModule中记录

Lightning 提供了日志功能，可以自动记录标量、手动记录其它数据。

### 自动记录

调用 `log()` 方法可以在 LightningModule 和回调的任何位置（名称包含 `batch_start` 的函数除外）记录一个标量：

```python
def training_step(self, batch, batch_idx):
    self.log('my_metric', x)
```

根据方法调用的位置，Lightning 会自动决定记录模式。当然你也可以通过手动设置 `log()` 参数来重载默认行为：

```python
def training_step(self, batch, batch_idx):
    self.log('my_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
```

`log()` 的四个选项分别代表：

* `on_step`：记录当前 step 的指标。对于 `training_step()` 和 `training_step_end()` 默认为真
* `on_epoch`：自动累积并在 epoch 结束时记录。对于 `training_epoch_end()`，以及验证和测试循环默认为真
* `prog_bar`：记录到进度条
* `logger`：记录到日志器，即 trainer 设置的 `logger`

> 设置`on_epoch=True`将缓存一个epoch中所有记录的值，并在epoch结束时执行一个折减。



### 手动记录

如果你想要记录一个图片、文本、直方图等，你可以直接使用日志器对象。

```python
def training_step(...):
    ...
    # the logger you used (in this case tensorboard)
    tensorboard = self.logger.experiment
    tensorboard.add_image()
    tensorboard.add_histogram(...)
    tensorboard.add_figure(...)
```



### 访问日志

使用日志器访问日志，例如 TensorBoard：

```shell
tensorboard --logdir ./lightning_logs
```



## 实现自定义日志器



## 控制记录频率

记录每一个 batch 会降低训练速度。默认情况下，Lightning 每 50 个训练 step 记录一次。可以通过设置 trainer 的 `log_every_n_steps` 改变此行为：

```python
k = 10
trainer = Trainer(log_every_n_steps=k)
```



写入日志器的操作十分昂贵，因此默认情况下 Lightning 每 100 个训练 step 将日志写入日志器或磁盘。可以通过设置 trainer 的 `flush_logs_every_n_steps` 改变此行为：

```python
k = 100
trainer = Trainer(log_every_n_steps=k)
```

此参数仅适用 TensorBoard 日志器。



## 进度条

你可以通过调用 `log()` 方法并设置 `prog_bar=True`，来向进度条添加任何指标：

```python
def training_step(self, batch, batch_idx):
    self.log('my_loss', loss, prog_bar=True)
```



进度条默认已经包含了训练损失和实验的版本号（根据设置的日志器）。可以在 LightningModule 中重载 `get_progress_bar_dict()` 钩子来修改默认行为：

```python
def get_progress_bar_dict(self):
    # don't show the version number
    items = super().get_progress_bar_dict()
    items.pop("v_num", None)
    return items
```

> 默认的训练损失是一个运行中平均值。



## 配置控制台日志

Lightning 向控制台记录了关于训练过程和用户警告的有用信息，你可以取回 Lightning 日志器并根据自己的需要进行修改。例如，调整记录等级或者重定向输出到日志文件。

```python
import logging

# configure logging at the root level of lightning
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

# configure logging on module level, redirect to file
logger = logging.getLogger("pytorch_lightning.core")
logger.addHandler(logging.FileHandler("core.log"))
```



## 记录超参数

当训练模型时，了解模型使用了何种超参数是十分有用的。当 Lightning 创建了一个检查点时，它存储了键 `'hyper_parameters'` 和所有超参数值。

一些日志器也会记录实验中使用的超参数。例如，当使用 `TensorBoardLogger` 时，所有的超参数都会在 HPARAMS 标签页展示。





# 回调



## 持久化状态



## 最佳实践





# 保存和加载权重

> https://pytorch-lightning.readthedocs.io/en/stable/weights_loading.html

Lightning 自动保存和加载检查点，检查点包含了模型使用的所有参数值。

在训练过程中保存检查点允许你继续训练，不管是因为训练中断（包括 Ctrl+C 终止），想要精调模型，或使用预训练模型。



## 检查点保存

Lightning 检查点包含了恢复一个训练会话所需要的所有内容：

* 16 位缩放因子
* 当前 epoch
* 全局 step
* 模型的 state_dict
* 所有优化器的状态
* 所有学习率调度器的状态
* 所有回调的状态
* 作为 hparams 传入的模型超参数



## 自动保存

Lightning 自动保存包含最后一个训练 epoch 状态的检查点。换言之，每完成一个训练 epoch，存档就被替换一次，例如从 `epoch=0-step=1718.ckpt` 到 `epoch=1-step=3437.ckpt`。

> 可以通过设置trainer的`default_root_dir`, `weights_save_path`和`ModelCheckpoint`回调修改检查点的保存路径：
>
> + 如果你既不设置`weights_save_path`，也不设置`ModelCheckpoint`回调，检查点会和日志保存在同一位置，路径取决于`default_root_dir`
>
> + 如果你设置了`weights_save_path='wdir/'`，保存路径为
>
> + 如果你设置`ModelCheckpoint`回调
>
> + 注意设置`ModelCheckpoint`回调会重载`weights_save_path`，即`weights_save_path`无效，例如下面的例子：
>
>   ```python
>   # NOTE: this saves weights to some/path NOT my/path
>   checkpoint = ModelCheckpoint(dirpath='some/path')
>   trainer = Trainer(
>       callbacks=[checkpoint],
>       weights_save_path='my/path'
>   )
>   ```

你可以自定义保存检查点的行为，使其监视训练或验证 step 中的任何变量。例如，如果你想要根据验证损失更新检查点：

1. 计算你想要监视的任何指标，例如验证损失
2. 调用 `log()` 方法记录该变量，使用键例如 `'val_loss'`
3. 初始化 `ModelCheckpoint` 回调，设置 `monitor` 为该变量的键
4. 将该回调传入 Trainer 的 `callbacks` 参数

```python
from pytorch_lightning.callbacks import ModelCheckpoint

class LitAutoEncoder(pl.LightningModule):
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)

        # 1. calculate loss
        loss = F.cross_entropy(y_hat, y)

        # 2. log `val_loss`
        self.log('val_loss', loss)

# 3. Init ModelCheckpoint callback, monitoring 'val_loss'
checkpoint_callback = ModelCheckpoint(monitor='val_loss')

# 4. Add your callback to the callbacks list
trainer = Trainer(callbacks=[checkpoint_callback])
```

你也可以控制更多的高级选项，例如 `save_top_k` 保存最佳的 $$k$$ 个检查点，`mode` 指定监视变量取最大值还是最小值等等。

```python
from pytorch_lightning.callbacks import ModelCheckpoint

class LitAutoEncoder(pl.LightningModule):
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss)

# saves a file like: my/path/sample-mnist-epoch=02-val_loss=0.32.ckpt
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='my/path/',
    filename='sample-mnist-{epoch:02d}-{val_loss:.2f}',
    save_top_k=3,
    mode='min',
)

trainer = Trainer(callbacks=[checkpoint_callback])
```

你可以在训练结束后取回检查点：

```python
checkpoint_callback = ModelCheckpoint(dirpath='my/path/')
trainer = Trainer(callbacks=[checkpoint_callback])
trainer.fit(model)
checkpoint_callback.best_model_path     # path to best checkpoint
```



### 禁用检查点

你可以禁用保存检查点，通过传入

```python
trainer = Trainer(checkpoint_callback=False)
```



### 保存超参数

Lightning 检查点同时保存了传入 LightningModule 的参数，保存在检查点的 `hyper_parameters` 键下。

```python
class MyLightningModule(LightningModule):

   def __init__(self, learning_rate, *args, **kwargs):  # hparam
        super().__init__()
        self.save_hyperparameters()                     # save hparam

# all init args were saved to the checkpoint
checkpoint = torch.load(CKPT_PATH)
print(checkpoint['hyper_parameters'])                   # access hparam
# {'learning_rate': the_value}
```



## 手动保存

你可以手动保存和恢复检查点。

```python
model = MyModel(hparams)
trainer.fit(model)
trainer.save_checkpoint("example.ckpt")
new_model = MyModel.load_from_checkpoint(checkpoint_path="example.ckpt")
```



## 使用加速器的手动保存

当使用 DDP 加速器时我们的训练脚本跨多个设备同时运行，此时 Lightning 将自动确保模型仅保存在主进程中，其它进程不干扰保存检查点。这一功能不需要改变任何代码：

```python
trainer = Trainer(accelerator="ddp")
model = MyLightningModule(hparams)
trainer.fit(model)
# Saves only on the main process
trainer.save_checkpoint("example.ckpt")
```

不使用 `trainer.save_checkpoint` 保存而使用其它保存函数会导致所有设备尝试保存检查点，因而会造成意想不到的行为和潜在的死锁。因此我们推荐使用 trainer 的保存功能。



## 加载检查点

加载一个模型，包括它的权重、偏置和超参数，使用以下方法：

```python
model = MyLightingModule.load_from_checkpoint(PATH)

print(model.learning_rate)
# prints the learning_rate you used in this checkpoint

model.eval()
y_hat = model(x)
```



如果你想在加载时覆写一些超参数值，可以在加载方法中传入：

```python
class LitModel(LightningModule):

    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        # ...
        
# use lr=1e-3
LitModel(lr=1e-3)

# save lr=1e-3
model = LitModel.load_from_checkpoint(PATH)

# overwrite lr=2e-3 when loading
model = LitModel.load_from_checkpoint(PATH, lr=2e-3)
```



## 直接恢复训练

如果你加载模型是为了恢复训练，可以直接用 Trainer 加载检查点：

```python
model = LitModel()
trainer = Trainer(resume_from_checkpoint='some/path/to/my_checkpoint.ckpt')

# automatically restores model, epoch, step, LR schedulers, apex, etc...
trainer.fit(model)
```





# 超参数

> https://pytorch-lightning.readthedocs.io/en/stable/hyperparameters.html

Lightning 具有功能，可以和命令行参数解析器 `ArgumentParser` 无缝衔接，并且和你选择的超参数优化框架可以很好地配合。



## ArgumentParser

Lightning 被设计为增强内置 Python ArgumentParser 的许多功能。`ArgumentParser` 的用法参考 Python 标准库 `argparse`。



## argparse的最佳实践

最佳实践将所有参数分为三部分：

1.Trainer 参数（`gpus`，`num_nodes`，etc）
2. 模型参数（`layer_dim`，`learning_rate`，etc）
3. 程序参数（`data_path`，`cluster_email`，etc）

我们进行如下操作：首先，在 `LightningModule` 中定义模型的参数：

```python
class LitModel(LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--encoder_layers', type=int, default=12)
        parser.add_argument('--data_path', type=str, default='/some/path')
        return parser
```

然后在 trainer 文件中，为 parser 添加 Trainer 参数、程序参数和模型参数：

```python
# ----------------
# trainer_main.py
# ----------------
from argparse import ArgumentParser
parser = ArgumentParser()

# add program level args
parser.add_argument('--conda_env', type=str, default='some_name')
parser.add_argument('--notification_email', type=str, default='will@email.com')

# add model specific args
parser = LitModel.add_model_specific_args(parser)

# add all the available trainer options to argparse
# ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
parser = Trainer.add_argparse_args(parser)

args = parser.parse_args()
```

最后，按如下方法初始化模型和 trainer：

```python
# init the trainer like this
trainer = Trainer.from_argparse_args(args, early_stopping_callback=...)

# NOT like this
trainer = Trainer(gpus=hparams.gpus, ...)

# init the model with Namespace directly
model = LitModel(args)

# or init the model with all the key-value pairs
dict_args = vars(args)
model = LitModel(**dict_args)
```

现在，你就可以在命令行运行：

```python
python trainer_main.py --gpus 2 --num_nodes 2 --conda_env 'my_env' --encoder_layers 12
```



## LightningModule的超参数

我们经常训练一个模型的很多不同的版本。当你回过头去看几个月前训练的模型时，你可能已经不记得模型是如何训练得到的（例如学习率是多少，何种网络结构等等）。

Lightning 有一些方法为你保存这些信息到检查点或 yaml 文件中。此处的目标是提升代码的可读性和复用性。

1. 第一种方法是保存构造函数中的所有参数值到检查点。这些参数也可以通过 `self.hparams` 访问。

   ```python
   class LitMNIST(LightningModule):
   
       def __init__(self, layer_1_dim=128, learning_rate=1e-2, **kwargs):
           super().__init__()
           # call this to save (layer_1_dim=128, learning_rate=1e-2) to the checkpoint
           self.save_hyperparameters()
   
           # equivalent
           self.save_hyperparameters('layer_1_dim', 'learning_rate')
   
           # Now possible to access layer_1_dim from hparams
           self.hparams.layer_1_dim
   ```

2. 有些时候你可能不想保存所有的参数，此时可以选择部分参数：

   ```python
   class LitMNIST(LightningModule):
   
       def __init__(self, loss_fx, generator_network, layer_1_dim=128 **kwargs):
           super().__init__()
           self.layer_1_dim = layer_1_dim
           self.loss_fx = loss_fx
   
           # save only (layer_1_dim=128) to the checkpoint
           self.save_hyperparameters('layer_1_dim')
   
   # when loading specify the other args
   model = LitMNIST.load_from_checkpoint(PATH, loss_fx=torch.nn.SomeOtherLoss, generator_network=MyGenerator())
   ```

3. 你还可以将完整的对象（例如词典）保存到检查点：

   ```python
   
   ```



## Trainer参数

简言之，将所有可能的 trainer 参数添加到 argparser 并按如下方法初始化 `Trainer`：

```python
parser = ArgumentParser()
parser = Trainer.add_argparse_args(parser)
hparams = parser.parse_args()

trainer = Trainer.from_argparse_args(hparams)

# 如果使用回调,那么还需要传入回调对象
trainer = Trainer.from_argparse_args(hparams, checkpoint_callback=..., callbacks=[...])
```



## 多个LightningModule

我们经常使用多个 `LightningModule`，每一个都有不同的参数。你可以在每个 `LightningModule` 中定义参数：

```python
class LitMNIST(LightningModule):

    def __init__(self, layer_1_dim, **kwargs):
        super().__init__()
        self.layer_1 = torch.nn.Linear(28 * 28, layer_1_dim)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--layer_1_dim', type=int, default=128)
        return parser
```

```python
class GoodGAN(LightningModule):

    def __init__(self, encoder_layers, **kwargs):
        super().__init__()
        self.encoder = Encoder(layers=encoder_layers)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--encoder_layers', type=int, default=12)
        return parser
```

此时我们就可以在两种模型中选择，被选择的模型会注入它的参数：

```python
def main(args):
    dict_args = vars(args)

    # pick model
    if args.model_name == 'gan':
        model = GoodGAN(**dict_args)
    elif args.model_name == 'mnist':
        model = LitMNIST(**dict_args)

    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)

    # figure out which model to use
    parser.add_argument('--model_name', type=str, default='gan', help='gan or mnist')

    # 解析已知的部分参数(例如18行增加的--model_name)
    # 获取模型名称的关键步骤
    temp_args, _ = parser.parse_known_args()

    # let the model add what it wants
    if temp_args.model_name == 'gan':        # 不同模型注入不同参数
        parser = GoodGAN.add_model_specific_args(parser)
    elif temp_args.model_name == 'mnist':
        parser = LitMNIST.add_model_specific_args(parser)

    args = parser.parse_args()

    # train
    main(args)
```

现在你就可以使用命令行选择训练的模型：

```shell
$ python main.py --model_name gan --encoder_layers 24
$ python main.py --model_name mnist --layer_1_dim 128
```





# 提前停止

> https://pytorch-lightning.readthedocs.io/en/stable/early_stopping.html

`EarlyStopping` 回调用于监视一个验证指标，并在观察到该指标没有改善时停止训练。

使用步骤：

* 导入 `EarlyStopping` 回调
* 使用 `log()` 方法记录你想要监视的指标
* 初始化 `EarlyStopping` 回调，设定 `monitor` 为该指标
* 将该回调传入 `Trainer` 的 `callbacks`

```python
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

def validation_step(...):
    self.log('val_loss', loss)

trainer = Trainer(callbacks=[EarlyStopping(monitor='val_loss')])
```

你可以通过修改参数自定义回调的行为：

```python
early_stop_callback = EarlyStopping(
   monitor='val_accuracy',
   min_delta=0.00,
   patience=3,
   verbose=False,
   mode='max'
)
trainer = Trainer(callbacks=[early_stop_callback])
```

`EarlyStopping` 回调在每个验证 epoch 结束时运行，而在默认配置下，每个验证 epoch 发生在每个训练 epoch 之后。验证的频率可以通过设置 `Trainer` 的参数进行修改，例如 `check_val_every_n_epoch` 和 `val_check_interval`。注意 `patience` 参数计数的是验证 epoch 没有改善的次数而非训练 epoch，例如设置 `check_val_every_n_epoch=10` 和 `patience=3` 时，trainer 在至少 40 个训练 epoch 后才会停止。





# 性能和瓶颈信息

> https://pytorch-lightning.readthedocs.io/en/stable/profiler.html

分析（profile）训练过程可以帮助你理解代码中是否存在任何瓶颈。



## 简单分析(profiling)

如果只想要分析标准动作，可以在构造 Trainer 对象时设置 `profiler="simple"`：

```python
trainer = Trainer(..., profiler="simple")
```

训练完成后，分析器（profiler）会打印如下结果：

```python
Profiler Report

Action                  |  Mean duration (s)    |  Total time (s)
-----------------------------------------------------------------
on_epoch_start          |  5.993e-06            |  5.993e-06
get_train_batch         |  0.0087412            |  16.398
on_batch_start          |  5.0865e-06           |  0.0095372
model_forward           |  0.0017818            |  3.3408
model_backward          |  0.0018283            |  3.4282
on_after_backward       |  4.2862e-06           |  0.0080366
optimizer_step          |  0.0011072            |  2.0759
on_batch_end            |  4.5202e-06           |  0.0084753
on_epoch_end            |  3.919e-06            |  3.919e-06
on_train_end            |  5.449e-06            |  5.449e-06
```



## 高级分析

如果你想获取每个事件中的函数调用信息，那么可以使用 `AdvancedProfiler`。该选项使用 Python 的 cProfiler 提供每一个调用的函数的时间花费报告。

```python
trainer = Trainer(..., profiler="advanced")
```

训练完成后，分析器会打印其结果。报告可能会非常长，因此你可以将其打印到指定文件中。

……







# 分布式训练

## 多GPU训练

