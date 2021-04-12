[toc]

# pytorch_lightning.callbacks

> 注意keras.callbacks同时对应pytorch_lightning.callbacks和pytorch_lightning.loggers。

## Callback

用于创建新回调的抽象类。若要创建回调，继承此类并任意重载以下方法。

```python
class CustomCallback(pytorch_lightning.callbacks.Callback):
    def on_before_accelerator_backend_setup(self, trainer, pl_module: LightningModule) -> None:
        """Called before accelerator is being setup"""
        pass

    def setup(self, trainer, pl_module: LightningModule, stage: Optional[str] = None) -> None:
        """Called when fit, validate, test, predict, or tune begins"""
        # 当训练/测试开始时调用
        pass

    def teardown(self, trainer, pl_module: LightningModule, stage: Optional[str] = None) -> None:
        """Called when fit, validate, test, predict, or tune ends"""
        # 当训练/测试结束时调用,全部结束时亦调用一次
        pass

    def on_init_start(self, trainer) -> None:
        """Called when the trainer initialization begins, model has not yet been set."""
        # 当训练器初始化开始时调用,此时模型尚未设置
        pass

    def on_init_end(self, trainer) -> None:
        """Called when the trainer initialization ends, model has not yet been set."""
        # 当训练器初始化结束时调用,此时模型尚未设置
        pass

    def on_fit_start(self, trainer, pl_module: LightningModule) -> None:
        """Called when fit begins"""
        pass

    def on_fit_end(self, trainer, pl_module: LightningModule) -> None:
        """Called when fit ends"""
        pass

    def on_sanity_check_start(self, trainer, pl_module: LightningModule) -> None:
        """Called when the validation sanity check starts."""
        # 当验证集可用性检查开始时调用
        pass

    def on_sanity_check_end(self, trainer, pl_module: LightningModule) -> None:
        """Called when the validation sanity check ends."""
        # 当验证集可用性检查结束时调用
        pass

    def on_train_batch_start(
        self, trainer, pl_module: LightningModule, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        """Called when the train batch begins."""
        # 当训练batch开始时调用
        pass

    def on_train_batch_end(
        self, trainer, pl_module: LightningModule, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        """Called when the train batch ends."""
        # 当训练batch结束时调用
        pass

    def on_train_epoch_start(self, trainer, pl_module: LightningModule) -> None:
        """Called when the train epoch begins."""
        # 当训练epoch开始时调用
        pass

    def on_train_epoch_end(self, trainer, pl_module: LightningModule, outputs: List[Any]) -> None:
        """Called when the train epoch ends."""
        # 当训练epoch结束时调用
        # 调用trainer.callback_metrics得到当前训练epoch的train_loss和前一个验证epoch的val_loss和val_acc
        pass

    def on_validation_epoch_start(self, trainer, pl_module: LightningModule) -> None:
        """Called when the val epoch begins."""
        # 当验证epoch开始时调用
        pass

    def on_validation_epoch_end(self, trainer, pl_module: LightningModule, outputs: List[Any]) -> None:
        """Called when the val epoch ends."""
        # 当验证epoch结束时调用
        # outputs参数可能接受不到任何值,建议写为outputs=None
        # 调用trainer.callback_metrics得到当前训练epoch的train_loss和当前验证epoch的val_loss和val_acc
        pass

    def on_test_epoch_start(self, trainer, pl_module: LightningModule) -> None:
        """Called when the test epoch begins."""
        # 当测试epoch开始时调用
        pass

    def on_test_epoch_end(self, trainer, pl_module: LightningModule, outputs: List[Any]) -> None:
        """Called when the test epoch ends."""
        # 当测试epoch结束时调用
        # 调用trainer.callback_metrics得到测试epoch的test_loss和test_acc以及最后一个训练+验证epoch的
        #     train_loss, val_loss和val_acc
        pass

    def on_epoch_start(self, trainer, pl_module: LightningModule) -> None:
        """Called when the epoch begins."""
        # 当(训练/测试)epoch开始时调用
        pass

    def on_epoch_end(self, trainer, pl_module: LightningModule) -> None:
        """Called when the epoch ends."""
        # 当(训练/验证/测试)epoch结束时调用
        # 调用trainer.callback_metrics的结果与其之前的on_train_epoch_end, on_validation_epoch_end, 
        #     on_test_epoch_end相同
        pass

    def on_batch_start(self, trainer, pl_module: LightningModule) -> None:
        """Called when the training batch begins."""
        pass

    def on_validation_batch_start(
        self, trainer, pl_module: LightningModule, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        """Called when the validation batch begins."""
        pass

    def on_validation_batch_end(
        self, trainer, pl_module: LightningModule, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        """Called when the validation batch ends."""
        pass

    def on_test_batch_start(
        self, trainer, pl_module: LightningModule, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        """Called when the test batch begins."""
        pass

    def on_test_batch_end(
        self, trainer, pl_module: LightningModule, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        """Called when the test batch ends."""
        pass

    def on_batch_end(self, trainer, pl_module: LightningModule) -> None:
        """Called when the training batch ends."""
        pass

    def on_train_start(self, trainer, pl_module: LightningModule) -> None:
        """Called when the train begins."""
        # 当训练开始时调用
        pass

    def on_train_end(self, trainer, pl_module: LightningModule) -> None:
        """Called when the train ends."""
        # 当训练结束时调用
        pass

    def on_pretrain_routine_start(self, trainer, pl_module: LightningModule) -> None:
        """Called when the pretrain routine begins."""
        pass

    def on_pretrain_routine_end(self, trainer, pl_module: LightningModule) -> None:
        """Called when the pretrain routine ends."""
        pass

    def on_validation_start(self, trainer, pl_module: LightningModule) -> None:
        """Called when the validation loop begins."""
        # 当验证开始时调用
        pass

    def on_validation_end(self, trainer, pl_module: LightningModule) -> None:
        """Called when the validation loop ends."""
        # 当验证结束时调用
        pass

    def on_test_start(self, trainer, pl_module: LightningModule) -> None:
        """Called when the test begins."""
        # 当测试开始时调用
        pass

    def on_test_end(self, trainer, pl_module: LightningModule) -> None:
        """Called when the test ends."""
        # 当测试结束时调用
        pass

    def on_keyboard_interrupt(self, trainer, pl_module: LightningModule) -> None:
        """Called when the training is interrupted by ``KeyboardInterrupt``."""
        pass

    def on_save_checkpoint(self, trainer, pl_module: LightningModule, checkpoint: Dict[str, Any]) -> dict:
        """
        Called when saving a model checkpoint, use to persist state.
        Args:
            trainer: the current Trainer instance.
            pl_module: the current LightningModule instance.
            checkpoint: the checkpoint dictionary that will be saved.
        Returns:
            The callback state.
        """
        pass

    def on_load_checkpoint(self, callback_state: Dict[str, Any]) -> None:
        """Called when loading a model checkpoint, use to reload state.
        Args:
            callback_state: the callback state returned by ``on_save_checkpoint``.
        """
        pass

    def on_after_backward(self, trainer, pl_module: LightningModule) -> None:
        """Called after ``loss.backward()`` and before optimizers do anything."""
        pass

    def on_before_zero_grad(self, trainer, pl_module: LightningModule, optimizer) -> None:
        """Called after ``optimizer.step()`` and before ``optimizer.zero_grad()``."""
        pass
```

调用的顺序如下：

```python
$ python mnist_lambdacallback.py
GPU available: False, used: False
TPU available: None, using: 0 TPU cores
    
init start ...                   # 训练器初始化
init end ...

setup ...                        # setup
fit start ...

  | Name  | Type       | Params
-------------------------------------
0 | model | Sequential | 55.1 K
-------------------------------------
55.1 K    Trainable params
0         Non-trainable params
55.1 K    Total params
0.220     Total estimated model params size (MB)

validation start ...              # 验证集测试
validation epoch start ...
validation epoch end ...
epoch end ...
validation end ...

train start ...                   # 训练开始

epoch start ...                   # 一个训练+验证epoch
train epoch start ...
train epoch end ...
epoch end ...
validation start ...
validation epoch start ...
validation epoch end ...
epoch end ...
validation end ...

epoch start ...
train epoch start ...
train epoch end ...
epoch end ...
validation start ...
validation epoch start ...
validation epoch end ...
epoch end ...
validation end ...

epoch start ...
train epoch start ...
train epoch end ...
epoch end ...
validation start ...
validation epoch start ...
validation epoch end ...
epoch end ...
validation end ...

train end ...                     # 训练结束

fit end ...
teardown ...                      # teardown

setup ...                         # setup
fit start ...

test start ...                    # 测试开始

test epoch start ...              # 一个测试epoch
test epoch end ...
epoch end ...

test end ...                      # 测试结束

--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'test_acc': 0.9527999758720398, 'test_loss': 0.14726394414901733}
--------------------------------------------------------------------------------

fit end ...
teardown ...                      # teardown

teardown ...
```





## EarlyStopping

当监视的参数不再改善时提前停止训练。

```python
pytorch_lightning.callbacks.EarlyStopping(monitor='early_stop_on', min_delta=0.0, patience=3, verbose=False, mode='min', strict=True)
# monitor               监视的指标
# min_delta             可以视为改善的最小绝对变化量,换言之,小于该值的指标绝对变化量视为没有改善
# patience              若最近patience次epoch的指标都没有改善(即最后patience次的指标都比倒数第patience+1次差),则停止训练
# mode                  若为'min',则指标减小视为改善;若为'max',则指标增加视为改善;若为'auto',则方向根据指标的名称自动推断
# strict                若为True,则当监视的指标不存在时抛出异常
```



## LambdaCallback

创建简单的自定义回调。

```python
pytorch_lightning.callbacks.LambdaCallback(on_before_accelerator_backend_setup=None, setup=None, teardown=None, on_init_start=None, on_init_end=None, on_fit_start=None, on_fit_end=None, on_sanity_check_start=None, on_sanity_check_end=None, on_train_batch_start=None, on_train_batch_end=None, on_train_epoch_start=None, on_train_epoch_end=None, on_validation_epoch_start=None, on_validation_epoch_end=None, on_test_epoch_start=None, on_test_epoch_end=None, on_epoch_start=None, on_epoch_end=None, on_batch_start=None, on_validation_batch_start=None, on_validation_batch_end=None, on_test_batch_start=None, on_test_batch_end=None, on_batch_end=None, on_train_start=None, on_train_end=None, on_pretrain_routine_start=None, on_pretrain_routine_end=None, on_validation_start=None, on_validation_end=None, on_test_start=None, on_test_end=None, on_keyboard_interrupt=None, on_save_checkpoint=None, on_load_checkpoint=None, on_after_backward=None, on_before_zero_grad=None)
# on_before_accelerator_backend_setup...   参见`Callback`
```

```python
>>> from pytorch_lightning import Trainer
>>> from pytorch_lightning.callbacks import LambdaCallback
>>> trainer = Trainer(callbacks=[LambdaCallback(setup=lambda *args: print('setup'))])
```





## LearningRateMonitor



## ModelCheckpoint





# pytorch_lightning.LightningModule

## methods

### freeze()

冻结所有参数以推断。



### load_from_checkpoint()

从检查点加载模型的主要方法。当Lightning保存检查点时，其将构造函数的参数保存在检查点的`hyper_parameters`键下。

任何通过`**kwargs`传入的参数都将重载保存在`hyper_parameters`下的参数。

```python
classmethod LightningModule.load_from_checkpoint(checkpoint_path, map_location=None, hparams_file=None, strict=True, **kwargs)
# checkpoint_path  检查点的路径,可以是url或类似文件的对象
# map_location     如果你保存了一个GPU模型并在CPU或者不同数量的GPU上加载,使用此参数...
# hparam_file      一个yaml文件的路径,其包含了模型超参数值
#                  一般用不到,因为超参数会自动保存到检查点中
# strict           ...
# kwargs           初始化模型的额外参数,用于重载已保存的超参数值
```

```python
# 直接加载
MyLightningModule.load_from_checkpoint('path/to/checkpoint.ckpt')

# 将所有权重从GPU1映射到GPU0
map_location = {'cuda:1':'cuda:0'}
MyLightningModule.load_from_checkpoint(
    'path/to/checkpoint.ckpt',
    map_location=map_location
)

# 从不同文件分别加载权重和超参数
MyLightningModule.load_from_checkpoint(
    'path/to/checkpoint.ckpt',
    hparams_file='/path/to/hparams_file.yaml'
)

# 重载部分超参数
MyLightningModule.load_from_checkpoint(
    PATH,
    num_layers=128,
    pretrained_ckpt_path: NEW_PATH,
)

# 预测
pretrained_model.eval()
pretrained_model.freeze()
y_hat = pretrained_model(x)
```



### save_hyperparameters()

将模型超参数保存至`hparams`属性。

```python
>>> class ManuallyArgsModel(LightningModule):
...     def __init__(self, arg1, arg2, arg3):
...         super().__init__()
...         self.save_hyperparameters('arg1', 'arg3')   # 保存指定参数
...     def forward(self, *args, **kwargs):
...         ...
>>> model = ManuallyArgsModel(1, 'abc', 3.14)
>>> model.hparams
"arg1": 1
"arg3": 3.14
```

```python
>>> class AutomaticArgsModel(LightningModule):
...     def __init__(self, arg1, arg2, arg3):
...         super().__init__()
...         self.save_hyperparameters()                 # 保存构造函数的所有参数
...     def forward(self, *args, **kwargs):
...         ...
>>> model = AutomaticArgsModel(1, 'abc', 3.14)
>>> model.hparams
"arg1": 1
"arg2": abc
"arg3": 3.14
```

```python
>>> class ManuallyArgsModel(LightningModule):
...     def __init__(self, arg1, arg2, arg3):
...         super().__init__()
...         self.save_hyperparameters(ignore='arg2')    # 忽略指定参数
...     def forward(self, *args, **kwargs):
...         ...
>>> model = ManuallyArgsModel(1, 'abc', 3.14)
>>> model.hparams
"arg1": 1
"arg3": 3.14
```



### unfreeze()

解冻所有参数以训练。



## properties

### current_epoch

当前epoch的序号。

```python
def training_step(...):
    if self.current_epoch == 0:
```



### device

此模块位于的设备。

```python
def training_step(...):
    z = torch.rand(2, 3, device=self.device)
```



### global_rank

此模块的全局rank。Lightning仅从global_rank=0保存日志、权重等。

全局rank表示所有GPU之中的GPU索引。例如，使用10台机器，每台有4个GPU，那么第10台机器的第4个GPU有global_rank=39。



### global_step

当前step的序号（每个epoch不重置）。



### hparams

通过调用`save_hyperparameters()`保存的模型超参数。

```python
def __init__(self, learning_rate):
    self.save_hyperparameters()

def configure_optimizers(self):
    return Adam(self.parameters(), lr=self.hparams.learning_rate)
```



### logger

当前使用的日志器。



### local_rank

此模块的局部rank。Lightning仅从global_rank=0保存日志、权重等。

局部rank表示当前机器的GPU索引。例如，使用10台机器，每台有4个GPU，那么第10台机器的第4个GPU有local_rank=3。



### trainer

当前使用的训练器。





## hooks





# pytorch_lightning.loggers

> 注意keras.callbacks同时对应pytorch_lightning.callbacks和pytorch_lightning.loggers。



# pytorch_lightning.metrics

## functional

### accuracy()

计算准确率。

```python
>>> from pytorch_lightning.metrics.functional import accuracy
>>> x = torch.tensor([0, 2, 1, 3])
>>> y = torch.tensor([0, 1, 2, 3])
>>> accuracy(x, y)
tensor(0.5000)
```



### confusion_matrix()

计算混淆矩阵。

见precision(), recall(), precision_recall(), f1()。



### precision(), recall(), precision_recall(), f1()

计算精确率、召回率、F1值。对比sklearn包的相关函数。

```python
>>> import torch
>>> from pytorch_lightning.metrics.functional import precision
>>> from pytorch_lightning.metrics.functional import recall
>>> from pytorch_lightning.metrics.functional import precision_recall
>>> from pytorch_lightning.metrics.functional import f1
>>> from pytorch_lightning.metrics.functional import confusion_matrix
>>> y_true = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])  # 二分类
>>> y_pred = torch.tensor([0, 0, 1, 1, 0, 1, 1, 1])
>>> confusion_matrix(y_pred, y_true, num_classes=2)  # 必须传入num_classes参数
# pred   0   1
tensor([[2., 2.],   # true  0
        [1., 3.]])  #       1
>>> precision(y_pred, y_true, class_reduction='none')  # 不分辨阴阳性,对每个类型计算
tensor([0.6667, 0.6000])
#       0       1
>>> recall(y_pred, y_true, class_reduction='none')
tensor([0.5000, 0.7500])
>>> precision_recall(y_pred, y_true, class_reduction='none')
(tensor([0.6667, 0.6000]), tensor([0.5000, 0.7500]))   # (precision, recall)
>>> f1(y_pred, y_true, num_classes=2, average='none')
tensor([0.5714, 0.6667])

```

```python
>>> import torch
>>> from pytorch_lightning.metrics.functional import precision
>>> from pytorch_lightning.metrics.functional import recall
>>> from pytorch_lightning.metrics.functional import precision_recall
>>> from pytorch_lightning.metrics.functional import f1
>>> from pytorch_lightning.metrics.functional import confusion_matrix
>>> y_true = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])  # 多分类
>>> y_pred = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2])
>>> confusion_matrix(y_pred, y_true, num_classes=3)
# pred   0   1   2
tensor([[3., 0., 0.],   # true  0
        [1., 2., 0.],   #       1
        [0., 2., 1.]])  #       2
>>> precision(y_pred, y_true, class_reduction='none')
tensor([0.7500, 0.5000, 1.0000])   # 分别对于pred(列) 0,1,2
>>> precision(y_pred, y_true, class_reduction='macro')
tensor(0.7500)                     # 宏平均
>>> precision(y_pred, y_true, class_reduction='micro')
tensor(0.6667)                     # 微平均
>>> recall(y_pred, y_true, class_reduction='none')
tensor([1.0000, 0.6667, 0.3333])   # 分别对于true(行) 0,1,2
>>> recall(y_pred, y_true, class_reduction='macro')
tensor(0.6667)
>>> recall(y_pred, y_true, class_reduction='micro')
tensor(0.6667)
>>> precision_recall(y_pred, y_true, class_reduction='none')
(tensor([0.7500, 0.5000, 1.0000]), tensor([1.0000, 0.6667, 0.3333])) 
>>> precision_recall(y_pred, y_true, class_reduction='macro')
(tensor(0.7500), tensor(0.6667))   # (precision, recall)
>>> precision_recall(y_pred, y_true, class_reduction='micro')
(tensor(0.6667), tensor(0.6667))
>>> f1(y_pred, y_true, num_classes=3, average='none')
tensor([0.8571, 0.5714, 0.5000])
>>> f1(y_pred, y_true, num_classes=3, average='macro')
tensor(0.6429)
>>> f1(y_pred, y_true, num_classes=3, average='micro')
tensor(0.6667)

```



### mean_absolute_error()

```python
>>> from pytorch_lightning.metrics.functional import mean_absolute_error
>>> a1 = torch.arange(10.0)
>>> a2 = a1+2
>>> mean_absolute_error(a1, a2)
tensor(2.)
```



### mean_squared_error()

```python
>>> from pytorch_lightning.metrics.functional import mean_squared_error
>>> a1 = torch.arange(10.0)
>>> a2 = a1+2
>>> mean_squared_error(a1, a2)
tensor(4.)
```





# pytorch_lightning.Trainer

## methods

### init()

```python
Trainer.__init__(logger=True, checkpoint_callback=True, callbacks=None, default_root_dir=None, gradient_clip_val=0, process_position=0, num_nodes=1, num_processes=1, gpus=None, auto_select_gpus=False, tpu_cores=None, log_gpu_memory=None, progress_bar_refresh_rate=1, overfit_batches=0.0, track_grad_norm=-1, check_val_every_n_epoch=1, fast_dev_run=False, accumulate_grad_batches=1, max_epochs=1000, min_epochs=1, max_steps=None, min_steps=None, limit_train_batches=1.0, limit_val_batches=1.0, limit_test_batches=1.0, val_check_interval=1.0, flush_logs_every_n_steps=100, log_every_n_steps=50, accelerator=None, sync_batchnorm=False, precision=32, weights_summary='top', weights_save_path=None, num_sanity_val_steps=2, truncated_bptt_steps=None, resume_from_checkpoint=None, profiler=None, benchmark=False, deterministic=False, reload_dataloaders_every_epoch=False, auto_lr_find=False, replace_sampler_ddp=True, terminate_on_nan=False, auto_scale_batch_size=False, prepare_data_per_node=True, plugins=None, amp_backend='native', amp_level='O2', distributed_backend=None, automatic_optimization=None, move_metrics_to_cpu=False, enable_pl_optimizer=None)
```



#### accelerator

使用的加速器后端。

+ `dp`：DataParallel，将batch划分给一台机器上的多个GPU
+ `ddp`：DistributedDataParallel，每台节点上的每个GPU进行训练，并同步梯度
+ `ddp_cpu`：CPU上的DistributedDataParallel，与`ddp`相同，但不使用GPU，用于多节点CPU训练。注意这对于单个节点没有加速作用，因为Torch已经能够充分运用单机的多个CPU（核）。
+ `ddp2`：在节点上使用dp，在节点间使用ddp

```python
# default used by the Trainer
trainer = Trainer(accelerator=None)

# dp = DataParallel
trainer = Trainer(gpus=2, accelerator='dp')

# ddp = DistributedDataParallel
trainer = Trainer(gpus=2, num_nodes=2, accelerator='ddp')

# ddp2 = DistributedDataParallel + dp
trainer = Trainer(gpus=2, num_nodes=2, accelerator='ddp2')
```



#### accumulate_grad_batches

每k个batch累积梯度，trainer会在最后一个step调用`optimizer.step()`。

```python
# default used by the Trainer (no accumulation)
trainer = Trainer(accumulate_grad_batches=1)

# accumulate every 4 batches (effective batch size is batch*4)
trainer = Trainer(accumulate_grad_batches=4)

# accumulate 3 from epoch 5, accumulate 20 from epoch 10
trainer = Trainer(accumulate_grad_batches={5: 3, 10: 20})
```

`accumulate_grad_batches=4`等效于4倍的batch size，但区别在于前者占用更小的显存，后者花费更少的时间。例如你的显存不足以训练batch size=64，但足以训练batch size=32，就可以通过设置`accumulate_grad_batches=2`等效实现batch size=64。



#### automatic_optimization

若设为`False`，则禁用Lightning的自动优化过程，此时你需要负责你自己的优化器行为。

当只使用一个优化器时不推荐禁用，但当使用两个优化器并且你是一个专业用户时推荐禁用。通常用于强化学习、稀疏编码和GAN研究。



#### auto_scale_batch_size

自动尝试显存所能接受的最大batch size。

```python
# default used by the Trainer (no scaling of batch size)
trainer = Trainer(auto_scale_batch_size=None)

# run batch size scaling, result overrides hparams.batch_size
trainer = Trainer(auto_scale_batch_size='binsearch')

# call tune to find the batch size
trainer.tune(model)
```

> 注意使用batch的初衷是让优化算法的效果更好，而非充分利用GPU的计算资源。可以假定GPU的运行速度受batch size的影响不大。



#### auto_select_gpus

若设为`True`且`gpus`是一个整数，自动选择可用的GPU。当GPU都被设置为独占模式(exclusive mode)时（即同一时刻只有一个进程能访问它），这尤其有用。

```python
# no auto selection (picks first 2 gpus on system, may fail if other process is occupying)
trainer = Trainer(gpus=2, auto_select_gpus=False)

# enable auto selection (will find two available gpus on system)
trainer = Trainer(gpus=2, auto_select_gpus=True)

# specifies all GPUs regardless of its availability
Trainer(gpus=-1, auto_select_gpus=False)

# specifies all available GPUs (if only one GPU is not occupied, uses one gpu)
Trainer(gpus=-1, auto_select_gpus=True)
```



#### auto_lr_find

当调用`trainer.tune()`时运行一个学习率查找算法（参考[论文](https://arxiv.org/abs/1506.01186)），以寻找最优初始学习率。

```python
# default used by the Trainer (no learning rate finder)
trainer = Trainer(auto_lr_find=False)

# run learning rate finder, results override hparams.learning_rate
trainer = Trainer(auto_lr_find=True)
# call tune to find the lr
trainer.tune(model)

# run learning rate finder, results override hparams.my_lr_arg
trainer = Trainer(auto_lr_find='my_lr_arg')
# call tune to find the lr
trainer.tune(model)
```



#### callbacks

添加一个回调列表。

```python
from pytorch_lightning.callbacks import Callback

class PrintCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training is started!")
    def on_train_end(self, trainer, pl_module):
        print("Training is done.")
        
callbacks = [PrintCallback()]
trainer = Trainer(callbacks=callbacks)        
```



#### check_val_every_n_epoch

每n个训练epoch运行一次验证epoch。

```python
# default used by the Trainer
trainer = Trainer(check_val_every_n_epoch=1)

# run val loop every 10 training epochs
trainer = Trainer(check_val_every_n_epoch=10)
```

参见提前停止。



#### checkpoint_callback

默认情况下Lightning保存最后一个训练epoch的状态为检查点，在当前工作目录下。设为`False`以禁用自动保存检查点。

```python
# default used by Trainer
trainer = Trainer(checkpoint_callback=True)

# turn off automatic checkpointing
trainer = Trainer(checkpoint_callback=False)
```

你可以使用`ModelCheckpoint`回调重载默认行为，参考保存和加载权重。



#### default_root_dir

日志和检查点的默认存放路径，如果不传入日志器、`ModelCheckpoint`回调或`weights_save_path`参数。有些集群需要分别存储日志和检查点，如果你不需要，则可以简单地使用此参数。

```python
# default used by the Trainer
trainer = Trainer(default_root_dir=os.getcwd())
```

> 详细的路径规则参考日志和保存和加载权重。



#### fast_dev_run

快速测试训练、验证和测试是否有bug，是一种单元测试。设为`True`时运行1个训练、验证、测试batch，设为n（正整数）时运行n个训练、验证、测试batch，设为`False`时不运行。

```python
# default used by the Trainer
trainer = Trainer(fast_dev_run=False)

# runs 1 train, val, test batch and program ends
trainer = Trainer(fast_dev_run=True)

# runs 7 train, val, test batches and program ends
trainer = Trainer(fast_dev_run=7)
```



#### flush_logs_every_n_steps

将日志写入磁盘的频率。

```python
# default used by the Trainer
trainer = Trainer(flush_logs_every_n_steps=100)
```



#### gpus

使用的GPU。参考多GPU训练。

```python
# default used by the Trainer (ie: train on CPU)
trainer = Trainer(gpus=None)

# equivalent
trainer = Trainer(gpus=0)

# int: train on 2 gpus
trainer = Trainer(gpus=2)

# list: train on GPUs 1, 4 (by bus ordering)
trainer = Trainer(gpus=[1, 4])
trainer = Trainer(gpus='1, 4') # equivalent

# -1: train on all gpus
trainer = Trainer(gpus=-1)
trainer = Trainer(gpus='-1') # equivalent

# combine with num_nodes to train on multiple GPUs across nodes
# uses 8 gpus in total
trainer = Trainer(gpus=2, num_nodes=4)

# train only on GPUs 1 and 4 across nodes
trainer = Trainer(gpus=[1, 4], num_nodes=4)
```



#### limit_train_batches

每个训练epoch仅运行指定比例的训练集或指定数量的batch，用于debug或测试。

```python
# default used by the Trainer
trainer = Trainer(limit_train_batches=1.0)

# run through only 25% of the training set each epoch
trainer = Trainer(limit_train_batches=0.25)

# run through only 10 batches of the training set each epoch
trainer = Trainer(limit_train_batches=10)
```



#### limit_test_batches

同上。



#### limit_val_batches

同上。



#### log_every_n_steps

增加日志行的频率。

```python
# default used by the Trainer
trainer = Trainer(log_every_n_steps=50)
```



#### log_gpu_memory

记录显存占用（使用`nvidia-smi`的输出）。

```python
# default used by the Trainer
trainer = Trainer(log_gpu_memory=None)

# log all the GPUs (on master node only)
trainer = Trainer(log_gpu_memory='all')

# log only the min and max memory on the master node
trainer = Trainer(log_gpu_memory='min_max')
```



#### logger

日志器。

```python
from pytorch_lightning.loggers import TensorBoardLogger

# default logger used by trainer
logger = TensorBoardLogger(
    save_dir=os.getcwd(),
    version=0,
    name='lightning_logs'
)
Trainer(logger=logger)
```



#### max_epochs

最大训练epoch数。

```python
# default used by the Trainer
trainer = Trainer(max_epochs=1000)
```



#### min_epochs

最小训练epoch数。

```python
# default used by the Trainer
trainer = Trainer(min_epochs=1)
```



#### max_steps

最大训练step数。

```python
# Default (disabled)
trainer = Trainer(max_steps=None)

# Stop after 100 steps
trainer = Trainer(max_steps=100)
```

当`max_epochs`或`max_steps`之一达到时，训练即终止。



#### min_steps

最小训练step数。

```python
# Default (disabled)
trainer = Trainer(min_steps=None)

# Run at least for 100 steps (disable min_epochs)
trainer = Trainer(min_steps=100, min_epochs=0)
```

训练数需要同时达到`min_epochs`和`min_steps`。



#### num_nodes

分布式训练使用的GPU节点。

```python
# default used by the Trainer
trainer = Trainer(num_nodes=1)

# to train on 8 nodes
trainer = Trainer(num_nodes=8)
```



#### num_processes

训练的进程数。当使用`accelerator="ddp"`时自动设置为GPU的数量。当使用`accelerator="ddp_cpu"`时，设置大于1的数可以在一台没有GPU的机器上模拟分布式训练。这对于debug有用，但不会有任何加速效果，因为PyTorch的单进程已经可以充分利用多核CPU。

```python
# Simulate DDP for debugging on your GPU-less laptop
trainer = Trainer(accelerator="ddp_cpu", num_processes=2)
```



#### num_sanity_val_steps

在训练开始前对验证集的n个batch进行合法性检查，用于快速检查验证过程的bug而不用等待第一个验证epoch。默认使用2个step。

```python
# default used by the Trainer
trainer = Trainer(num_sanity_val_steps=2)

# turn it off
trainer = Trainer(num_sanity_val_steps=0)

# check all validation data
trainer = Trainer(num_sanity_val_steps=-1)
```

合法性检查完成后会重置验证dataloader。



#### overfit_batches

仅使用训练集的指定比例或指定数量的样本，并且用其进行验证和测试。如果训练dataloader有`shuffle=True`，Lightning会自动禁用之。

用于快速debug或故意实现过拟合。

```python
# default used by the Trainer
trainer = Trainer(overfit_batches=0.0)

# use only 1% of the train set (and use the train set for val and test)
trainer = Trainer(overfit_batches=0.01)

# overfit on 10 of the same batches
trainer = Trainer(overfit_batches=10)
```



#### cluster_environment



#### prepare_data_per_node

若为`True`则在每个节点的`LOCAL_RANK=0`上调用`prepare_data()`，若为`False`则仅在`NODE_RANK=0, LOCAL_RANK=0`上调用。

```python
# default
Trainer(prepare_data_per_node=True)

# use only NODE_RANK=0, LOCAL_RANK=0
Trainer(prepare_data_per_node=False)
```



#### precision

使用全精度（32）或半精度（16），可以用在CPU, GPU或TPU上。

```python
# default used by the Trainer
trainer = Trainer(precision=32)

# 16-bit precision
trainer = Trainer(precision=16)
```



#### profiler

打印训练事件和统计信息，用于帮助寻找性能上的瓶颈。

```python
from pytorch_lightning.profiler import SimpleProfiler, AdvancedProfiler

# default used by the Trainer
trainer = Trainer(profiler=None)

# to profile standard training events, equivalent to `profiler=SimpleProfiler()`
trainer = Trainer(profiler="simple")

# advanced profiler for function-level stats, equivalent to `profiler=AdvancedProfiler()`
trainer = Trainer(profiler="advanced")
```



#### progress_bar_refresh_rate

刷新进度条的频率。已知notebook由于屏幕刷新率的问题，使用高刷新率（低数值）会造成崩溃，建议赋值50以上。

```python
# default used by the Trainer
trainer = Trainer(progress_bar_refresh_rate=1)

# disable progress bar
trainer = Trainer(progress_bar_refresh_rate=0)
```



#### reload_dataloaders_every_epoch

若为`True`，则每个epoch都重新加载一个dataloader。

```python
# if False (default)
train_loader = model.train_dataloader()
for epoch in epochs:
    for batch in train_loader:
        ...

# if True
for epoch in epochs:
    train_loader = model.train_dataloader()
    for batch in train_loader:
```



#### replace_sampler_ddp





#### resume_from_checkpoint

从路径指定的检查点恢复训练。如果恢复的检查点训练到一个epoch半途，则训练从下一个epoch开始。

```python
# default used by the Trainer
trainer = Trainer(resume_from_checkpoint=None)

# resume from a specific checkpoint
trainer = Trainer(resume_from_checkpoint='some/path/to/my_checkpoint.ckpt')
```



#### sync_batchnorm



#### track_grad_norm



#### val_check_interval

一个训练epoch中插入验证epoch的次数，可以指定浮点数或整数。

```python
# default used by the Trainer
trainer = Trainer(val_check_interval=1.0)

# check validation set 4 times during a training epoch
trainer = Trainer(val_check_interval=0.25)

# check validation set every 1000 training batches
# use this when using iterableDataset and your dataset has no length
# (ie: production cases with streaming data)
trainer = Trainer(val_check_interval=1000)
```



#### weights_save_path

保存权重的目录。

```python
# default used by the Trainer
trainer = Trainer(weights_save_path=os.getcwd())

# save to your custom path
trainer = Trainer(weights_save_path='my/path')
```

> 详细的路径规则参考日志和保存和加载权重。



#### weights_summary

训练开始前打印模型参数的摘要，选项包含`'full', 'top', None` 。

```python
# default used by the Trainer (ie: print summary of top level modules)
trainer = Trainer(weights_summary='top')

# print full summary of all modules and submodules
trainer = Trainer(weights_summary='full')

# don't print a summary
trainer = Trainer(weights_summary=None)
```



### fit()

运行完整的优化步骤。

使用方法参考pytorch-lightning/基本特性/验证。



### test()

运行测试步骤。与优化步骤分离。

使用方法参考pytorch-lightning/基本特性/测试。



### tune()

在训练之前运行调整超参数的步骤，例如`auto_lr_find, auto_scale_batch_size`。



## properties

### callback_metrics

回调可用的指标，是一个字典对象。

每个训练epoch结束时更新`train_loss`指标，每个验证epoch结束时更新`val_loss`和`val_acc`指标，每个测试epoch结束时更新`test_loss`和`test_acc`指标；也可以调用`self.log`自动设置指标。

```python
def training_step(self, batch, batcbh_idx):
    self.log('a_val', 2)

trainer.callback_metrics['a_val']  # 2
```



### current_epoch

当前epoch序数。

```python
def training_step(self, batch, batch_idx):
    current_epoch = self.trainer.current_epoch
    if current_epoch > 100:
        # do something
        pass
```



### logger

当前使用的日志器。



### logged_metrics

发送到日志器的指标。

```python
def training_step(self, batch, batch_idx):
    self.log('a_val', 2, log=True)


trainer.logged_metrics['a_val']  # 2
```



### log_dir

日志的保存路径，用于保存图片等。

```python
def training_step(self, batch, batch_idx):
    img = ...
    save_img(img, self.trainer.log_dir)
```



### is_global_zero

确认当前进程是否为多节点训练的全局零。

```python
def training_step(self, batch, batch_idx):
    if self.trainer.is_global_zero:
        print('in node 0, accelerator 0')
```



### progress_bar_metrics

发送到进度条的指标。

```python
def training_step(self, batch, batch_idx):
    self.log('a_val', 2, prog_bar=True)

trainer.progress_bar_metrics['a_val']  # 2
```







