[toc]

[Horovod](https://horovod.ai/) 是一套面向 TensorFlow、Keras、PyTorch 和 Apache MXNet 的分布式深度学习训练框架。Horovod 的目标是让分布式深度学习快速且易用。



# 安装

参见 [Horovod Installation Guide](https://github.com/horovod/horovod/blob/master/docs/install.rst)。

建议使用官方镜像 [Dockerfile.cpu](https://github.com/horovod/horovod/blob/master/Dockerfile.cpu) 和 [Dockerfile.gpu](https://github.com/horovod/horovod/blob/master/Dockerfile.gpu)。





# 基本概念

Horovod 基于下列 MPI 概念，这里结合实例进行解释。假设我们有 4 台机器，各有 4 个 GPU，在每个 GPU 上执行训练脚本的一个副本，那么：

+ **size**：进程数，此处为 16
+ **rank**：进程的唯一 ID，这里为 0-15
+ **local rank**：进程在本机的唯一 ID，这里为 0-3
+ **allreduce**, **allgather**, **broadcast**：参见 [MPI 集体通信模式](../distribute/strategy.md)





# 脚本示例

## Keras

```python
import tensorflow as tf
from tensorflow import keras
import horovod.tensorflow.keras as hvd

# Horovod: initialize Horovod.
hvd.init()

# Horovod: pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

(mnist_images, mnist_labels), _ = \
    keras.datasets.mnist.load_data(path='mnist-%d.npz' % hvd.rank())

dataset = tf.data.Dataset.from_tensor_slices(
    (tf.cast(mnist_images[..., tf.newaxis] / 255.0, tf.float32),
             tf.cast(mnist_labels, tf.int64))
)
dataset = dataset.repeat().shuffle(10000).batch(128)

mnist_model = keras.Sequential([
    keras.layers.Conv2D(32, [3, 3], activation='relu'),
    keras.layers.Conv2D(64, [3, 3], activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.25),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')
])

# Horovod: adjust learning rate based on number of GPUs.
scaled_lr = 0.001 * hvd.size()
opt = keras.optimizers.Adam(learning_rate=scaled_lr)

# Horovod: add Horovod DistributedOptimizer.
opt = hvd.DistributedOptimizer(
    opt, backward_passes_per_step=1, average_aggregated_gradients=True)

# Horovod: Specify `experimental_run_tf_function=False` to ensure TensorFlow
# uses hvd.DistributedOptimizer() to compute gradients.
mnist_model.compile(loss='sparse_categorical_crossentropy',
                    optimizer=opt,
                    metrics=['accuracy'],
                    experimental_run_tf_function=False)

callbacks = [
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),

    # Horovod: average metrics among workers at the end of every epoch.
    #
    # Note: This callback must be in the list before the ReduceLROnPlateau,
    # TensorBoard or other metrics-based callbacks.
    hvd.callbacks.MetricAverageCallback(),

    # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
    # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
    # the first three epochs. See https://arxiv.org/abs/1706.02677 for details.
    hvd.callbacks.LearningRateWarmupCallback(initial_lr=scaled_lr, warmup_epochs=3, verbose=1),
]

# Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
if hvd.rank() == 0:
    callbacks.append(keras.callbacks.ModelCheckpoint('./checkpoint-{epoch}.h5'))

# Horovod: write logs on worker 0.
verbose = 1 if hvd.rank() == 0 else 0

# Train the model.
# Horovod: adjust number of steps based on number of GPUs.
mnist_model.fit(dataset, steps_per_epoch=500 // hvd.size(), callbacks=callbacks, epochs=24, verbose=verbose)
```



## PyTorch

```python
import argparse
import os
from filelock import FileLock

import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.utils.data.distributed
import horovod.torch as hvd

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
parser.add_argument('--use-adasum', action='store_true', default=False,
                    help='use adasum algorithm to do reduction')
parser.add_argument('--gradient-predivide-factor', type=float, default=1.0,
                    help='apply gradient predivide factor in optimizer (default: 1.0)')
parser.add_argument('--data-dir',
                    help='location of the training dataset in the local filesystem (will be downloaded if needed)')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


def train(epoch):
    model.train()
    # Horovod: set epoch to sampler for shuffling.
    train_sampler.set_epoch(epoch)
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            # Horovod: use train_sampler to determine the number of examples in
            # this worker's partition.
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_sampler),
                100. * batch_idx / len(train_loader), loss.item()))


def metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()


def test():
    model.eval()
    test_loss = 0.
    test_accuracy = 0.
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target, size_average=False).item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        test_accuracy += pred.eq(target.data.view_as(pred)).cpu().float().sum()

    # Horovod: use test_sampler to determine the number of examples in
    # this worker's partition.
    test_loss /= len(test_sampler)
    test_accuracy /= len(test_sampler)

    # Horovod: average metric values across workers.
    test_loss = metric_average(test_loss, 'avg_loss')
    test_accuracy = metric_average(test_accuracy, 'avg_accuracy')

    # Horovod: print output only on first rank.
    if hvd.rank() == 0:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
            test_loss, 100. * test_accuracy))


if __name__ == '__main__':
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # Horovod: initialize library.
    hvd.init()
    torch.manual_seed(args.seed)

    if args.cuda:
        # Horovod: pin GPU to local rank.
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(args.seed)


    # Horovod: limit # of CPU threads to be used per worker.
    torch.set_num_threads(1)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
    # issues with Infiniband implementations that are not fork-safe
    if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
            mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
        kwargs['multiprocessing_context'] = 'forkserver'

    data_dir = args.data_dir or './data'
    with FileLock(os.path.expanduser("~/.horovod_lock")):
        train_dataset = \
            datasets.MNIST(data_dir, train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ]))

    # Horovod: use DistributedSampler to partition the training data.
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler, **kwargs)

    test_dataset = \
        datasets.MNIST(data_dir, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))
    # Horovod: use DistributedSampler to partition the test data.
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size,
                                              sampler=test_sampler, **kwargs)

    model = Net()

    # By default, Adasum doesn't need scaling up learning rate.
    lr_scaler = hvd.size() if not args.use_adasum else 1

    if args.cuda:
        # Move model to GPU.
        model.cuda()
        # If using GPU Adasum allreduce, scale learning rate by local_size.
        if args.use_adasum and hvd.nccl_built():
            lr_scaler = hvd.local_size()

    # Horovod: scale learning rate by lr_scaler.
    optimizer = optim.SGD(model.parameters(), lr=args.lr * lr_scaler,
                          momentum=args.momentum)

    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # Horovod: (optional) compression algorithm.
    compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(optimizer,
                                         named_parameters=model.named_parameters(),
                                         compression=compression,
                                         op=hvd.Adasum if args.use_adasum else hvd.Average,
                                         gradient_predivide_factor=args.gradient_predivide_factor)

    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test()
```



## Lightning

```python
import argparse
import os
from filelock import FileLock
import tempfile

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
# import torch.utils.data.distributed

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import horovod.torch as hvd

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
parser.add_argument('--use-adasum', action='store_true', default=False,
                    help='use adasum algorithm to do reduction')
parser.add_argument('--gradient-predivide-factor', type=float, default=1.0,
                    help='apply gradient predivide factor in optimizer (default: 1.0)')
parser.add_argument('--data-dir',
                    help='location of the training dataset in the local filesystem (will be downloaded if needed)')


# Define the PyTorch model without any Horovod-specific parameters
class Net(LightningModule):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.float()
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, -1)

    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=0.01, momentum=0.5)

    def training_step(self, batch, batch_nb):
        x, y = batch[0], batch[1]
        y_hat = self(x)
        loss = F.nll_loss(y_hat, y.long())
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        x, y = batch[0], batch[1]
        y_hat = self(x)
        return {'val_loss': F.nll_loss(y_hat, y.long())}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}


def metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()


def test():
    model.eval()
    test_loss = 0.
    test_accuracy = 0.
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target, size_average=False).item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        test_accuracy += pred.eq(target.data.view_as(pred)).cpu().float().sum()

    # Horovod: use test_sampler to determine the number of examples in
    # this worker's partition.
    test_loss /= len(test_sampler)
    test_accuracy /= len(test_sampler)

    # Horovod: average metric values across workers.
    test_loss = metric_average(test_loss, 'avg_loss')
    test_accuracy = metric_average(test_accuracy, 'avg_accuracy')

    # Horovod: print output only on first rank.
    if hvd.rank() == 0:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
            test_loss, 100. * test_accuracy))


if __name__ == '__main__':
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    hvd.init()

    kwargs = {'num_workers': 2}
    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
    # issues with Infiniband implementations that are not fork-safe
    if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
            mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
        kwargs['multiprocessing_context'] = 'forkserver'

    # get data
    data_dir = args.data_dir or './data'
    with FileLock(os.path.expanduser("~/.horovod_lock")):
        train_dataset = \
            datasets.MNIST(data_dir, train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ]))

    # set training data loader
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler, **kwargs)

    test_dataset = \
        datasets.MNIST(data_dir, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))

    # set validation data loader
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size,
                                              sampler=test_sampler, **kwargs)

    epochs = args.epochs
    with tempfile.TemporaryDirectory() as run_output_dir:
        ckpt_path = os.path.join(run_output_dir, "checkpoint")
        os.makedirs(ckpt_path, exist_ok=True)

        logs_path = os.path.join(run_output_dir, "logger")
        os.makedirs(logs_path, exist_ok=True)
        logger = TensorBoardLogger(logs_path)

        train_percent = 1.0
        val_percent = 1.0

        model = Net()
        setattr(model, 'train_dataloader', lambda: train_loader)
        setattr(model, 'val_dataloader', lambda: test_loader)

        from pytorch_lightning.callbacks import Callback

        class MyDummyCallback(Callback):
            def __init__(self):
                self.epcoh_end_counter = 0
                self.train_epcoh_end_counter = 0

            def on_init_start(self, trainer):
                print('Starting to init trainer!')

            def on_init_end(self, trainer):
                print('Trainer is initialized.')

            def on_epoch_end(self, trainer, model):
                print('A epoch ended.')
                self.epcoh_end_counter += 1

            def on_train_epoch_end(self, trainer, model, unused=None):
                print('A train epoch ended.')
                self.train_epcoh_end_counter += 1

            def on_train_end(self, trainer, model):
                print('Training ends')
                assert self.epcoh_end_counter == 2 * epochs
                assert self.train_epcoh_end_counter == epochs

        callbacks = [MyDummyCallback(), ModelCheckpoint(dirpath=ckpt_path)]

        trainer = Trainer(accelerator='horovod',
                          gpus=(1 if torch.cuda.is_available() else 0),
                          callbacks=callbacks,
                          max_epochs=epochs,
                          limit_train_batches=train_percent,
                          limit_val_batches=val_percent,
                          logger=logger,
                          num_sanity_val_steps=0)

        trainer.fit(model)

        test()
```





# 运行

下面的示例命令展示了如何启动分布式训练：

1. 单机多卡，例如在有 4 个 GPU 的单台机器上运行：

   ```shell
   $ horovodrun -np 4 -H localhost:4 python train.py
   ```

2. 多机多卡，例如在各有 4 个 GPU 的 4 台机器上运行：

   ```shell
   $ horovodrun -np 16 -H server1:4,server2:4,server3:4,server4:4 python train.py
   ```





> 参考 https://towardsdatascience.com/distributed-deep-learning-training-with-horovod-on-kubernetes-6b28ac1d6b5d





# 弹性训练

Horovod 的弹性训练允许在运行过程中动态地增加或减少工作器的数量，而无需重启训练或从保存到持久存储的检查点继续。

需要：Gloo 支持；运行过程中发现可用主机的方法。



## 使用状态同步修改训练脚本

弹性训练需要在工作器加入或移除时追踪和同步各工作器的状态，为此对训练脚本进行如下修改：

1. 将主要的训练过程（初始化之后的所有操作）包装到一个被 `hvd.elastic.run` 装饰的函数中。

   被装饰的函数的第一个参数是一个 `hvd.elastic.State` 实例，在执行被装饰的函数前，该状态实例会被同步到各工作器中。这将确保新增加的工作器以及状态不一致的工作器在训练开始前都同步到相同的状态。

   在此函数之前不可调用 Horovod 集体通信操作（Broadcast, All-Reduce, All-Gather 等）。

2. 将所有需要在各工作器间保持同步的变量（模型参数、优化器状态、训练参数等）放置到 `hvd.elastic.State` 实例中。

   Horovod 提供了 TensorFlow、Keras 和 PyTorch 的标准的状态类型实现，但在有些情况下也需要重载基类 `hvd.elastic.State` 以处理自定义广播操作。

3. 周期性地调用 `state.commit()` 以在内存中备份当前状态。

   这有助于防止当工作器意外出错时状态被破坏。例如，如果训练在参数更新的过程中出错，部分参数的梯度更新被应用而部分参数的梯度仍在进行 All-Reduce 操作，那么此时将引发一个 `HorovodInternalError`，所有的参数都将恢复到上一次提交的值。

   提交的代价可能十分昂贵（对于大型模型），因此你需要在提交的频率与回滚的距离之间寻求平衡。

   Horovod 可以通过（我们称为）工作器的优雅移除来防止此类回滚。当主进程发现一个工作器被标记为移除时，其向所有工作器推送一个通知，在下次 `state.commit()` 被调用时引发一个 `HostsUpdatedInterrupt`，参数不会恢复到上一次提交的值。

   通常情况下，如果你的硬件是比较可靠的，并且你的调度系统会在计划移除工作器时给予主进程足够的警告，那么你可以安全地以比较低的频率调用 `state.commit()`，并在每个 batch 结束时调用 `state.check_host_updates()`。

4. 为 `hvd.elastic.State` 实例注册回调以因应训练过程中工作器成员的变化。

   例如根据新的全局批次规模重新调整学习率，或者重新划分数据集等操作通常在这些回调中完成。
   
   回调在 Horovod 重新初始化之后、状态在各工作器之间同步之前调用。



`HorovodInternalError`（出错）或 `HostsUpdatedInterrupt`（增加/移除请求）之后的重启过程如下：

1. 抓取 `hvd.elastic.run` 装饰器内的异常，若为 `HorovodInternalError`，恢复到上一次提交的状态。
2. 通过一轮新的协调组织重新初始化 Horovod 上下文。
3. 通过广播新的 0 号工作器的状态同步各工作器的状态。上一个步骤中，越老的工作器被指定为 0 号工作器的优先级越高，以确保广播的状态是最新的。
4. 继续训练，执行底层的训练函数。



## 脚本示例

[Keras 示例](https://horovod.readthedocs.io/en/stable/elastic_include.html#elastic-keras)

[PyTorch 示例](https://horovod.readthedocs.io/en/stable/elastic_include.html#elastic-pytorch)



## 使用 horovodrun 运行

弹性训练通过 `horovodrun` 命令行工具启动，启动时最大的不同是不再显式地指定主机，而是在运行过程中动态地发现主机。最通常的使  Horovod 发现可用主机的方法是在 `--host-discovery-script` 选项下提供一个脚本：

```shell
$ horovodrun -np 8 --host-discovery-script discover_hosts.sh python train.py
```

该主机发现脚本需要有用户执行权限，并且以 `<hostname>:<slots>` 的格式每行返回一个主机和它的可用槽位，例如：

```shell
$ ./discover_hosts.sh
host-1:4
host-2:4
host-3:4
```

……



## 实践过程中的思考







# API

## 共有

### allgather

All-Gather 操作。所有收集的张量在第一个维度进行拼接，因此各进程提供的张量必须具有相同的形状，除了第一个维度的规模可以不同。

```python
horovod.tensorflow.allgather(tensor, name=None, ignore_name_scope=False)
# tensor         收集的数据,是`tf.Tensor`,`tf.Variable`或`tf.IndexedSlices`类型
```

```python
horovod.tensorflow.keras.allgather(value, name=None)
# value          收集的数据,是`tf.Tensor`,`tf.Variable`或`tf.IndexedSlices`类型
```

```python
horovod.keras.allgather(value, name=None)
# value          收集的数据,是`tf.Tensor`,`tf.Variable`或`tf.IndexedSlices`类型
```

```python
horovod.torch.allgather(tensor, name=None)
# tensor         收集的数据,是`torch.Tensor`类型
```



### allreduce

All-Reduce 操作。

```python
horovod.tensorflow.allreduce(tensor, average=None, device_dense='', device_sparse='', compression=<class 'horovod.tensorflow.compression.NoneCompressor'>, op=None, prescale_factor=1.0, postscale_factor=1.0, name=None)
# tensor         归约的数据,是`tf.Tensor`,`tf.Variable`或`tf.IndexedSlices`类型
# compression    用于减少数据通信量的压缩算法
```

```python
horovod.tensorflow.keras.allreduce(value, name=None, average=None, prescale_factor=1.0, postscale_factor=1.0, op=None, compression=<class 'horovod.tensorflow.compression.NoneCompressor'>)
# value          归约的数据,是`tf.Tensor`,`tf.Variable`或`tf.IndexedSlices`类型
# compression    用于减少数据通信量的压缩算法
```

```python
horovod.keras.allreduce(value, name=None, average=True, prescale_factor=1.0, postscale_factor=1.0)
# value          归约的数据,是`tf.Tensor`,`tf.Variable`或`tf.IndexedSlices`类型
```

```python
horovod.torch.allreduce(tensor, average=None, name=None, compression=<class 'horovod.torch.compression.NoneCompressor'>, op=None, prescale_factor=1.0, postscale_factor=1.0)
# tensor         归约的数据,是`torch.Tensor`类型
# compression    用于减少数据通信量的压缩算法
```



### broadcast

Broadcast 操作。

```python
horovod.tensorflow.broadcast(tensor, root_rank, name=None, ignore_name_scope=False)
# tensor        广播的数据,是`tf.Tensor`,`tf.Variable`或`tf.IndexedSlices`类型
# root_rank     发送数据的进程的秩
```

```python
horovod.tensorflow.keras.broadcast(value, root_rank, name=None)
# value         广播的数据,是`tf.Tensor`,`tf.Variable`或`tf.IndexedSlices`类型
# root_rank     发送数据的进程的秩
```

```python
horovod.keras.broadcast(value, root_rank, name=None)
# value         广播的数据,是`tf.Tensor`,`tf.Variable`或`tf.IndexedSlices`类型
# root_rank     发送数据的进程的秩
```

```python
horovod.torch.broadcast(tensor, root_rank, name=None)
# tensor        广播的数据,是`torch.Tensor`类型
# root_rank     发送数据的进程的秩
```



### Compression

可选的 All-Reduce 操作中用于减少数据通信量的压缩算法。



#### NoneCompressor



#### FP16Compressor



### cuda_built()

若 Horovod 编译时包含了 CUDA 支持，返回 `True`。



### DistributedOptimizer

返回一个包装了原优化器的分布式优化器，其负责各进程间的通信，计算梯度值和应用参数更新则委托原优化器完成。

```python
horovod.tensorflow.DistributedOptimizer(optimizer, name=None, use_locking=False, device_dense='', device_sparse='', compression=<class 'horovod.tensorflow.compression.NoneCompressor'>, sparse_as_dense=False, backward_passes_per_step=1, op=<MagicMock name='mock().horovod_reduce_op_average()' id='140316232634960'>, gradient_predivide_factor=1.0, average_aggregated_gradients=False, num_groups=0, groups=None)
# optimizer      用于计算梯度和应用参数更新的优化器
# compression    All-Reduce操作中用于减少数据通信量的压缩算法
```

```python
horovod.tensorflow.keras.DistributedOptimizer(optimizer, name=None, device_dense='', device_sparse='', compression=<class 'horovod.tensorflow.compression.NoneCompressor'>, sparse_as_dense=False, gradient_predivide_factor=1.0, op=<MagicMock name='mock().horovod_reduce_op_average()' id='140316232634960'>, backward_passes_per_step=1, average_aggregated_gradients=False, num_groups=0, groups=None)
# optimizer      用于计算梯度和应用参数更新的优化器
# compression    All-Reduce操作中用于减少数据通信量的压缩算法
# backward_passes_per_step        
# average_aggregated_gradients   
```

```python
horovod.keras.DistributedOptimizer(optimizer, name=None, device_dense='', device_sparse='', compression=<class 'horovod.tensorflow.compression.NoneCompressor'>, sparse_as_dense=False, gradient_predivide_factor=1.0, op=<MagicMock name='mock().horovod_reduce_op_average()' id='140316232634960'>, num_groups=0, groups=None)
# optimizer      用于计算梯度和应用参数更新的优化器
# compression    All-Reduce操作中用于减少数据通信量的压缩算法
```

```python
horovod.torch.DistributedOptimizer(optimizer, named_parameters=None, compression=<class 'horovod.torch.compression.NoneCompressor'>, backward_passes_per_step=1, op=<MagicMock name='mock().horovod_reduce_op_average()' id='140316224808592'>, gradient_predivide_factor=1.0, num_groups=0, groups=None, sparse_as_dense=False)
# optimizer          用于计算梯度和应用参数更新的优化器
# named_parameters   参数名称到值的映射,用于allreduce操作的命名.一般就是`model.named_parameters()`
# compression        All-Reduce操作中用于减少数据通信量的压缩算法
```



### elastic.run()

用于运行弹性训练过程的装饰器。参见[弹性训练](#弹性训练)。



### gloo_enabled()

若 Gloo 在当前运行时可用，返回 `True`。



### gloo_built()

若 Horovod 编译时包含了 Gloo 支持，返回 `True`。



### init()

初始化 Horovod。

```python
horovod.tensorflow.init(comm=None)
# comm     通讯器,给定的通讯器将被复制并使用副本,默认使用`MPI_COMM_WORLD`通讯器
```



### is_initialized()

若 Horovod 已经初始化，返回 `True`。



### local_rank()

返回当前进程的本地 Horovod rank。



### local_size()

返回当前进程所在节点上的 Horovod 进程数。



### mpi_threads_supported()

若支持 MPI 多线程，返回 `True`。



### mpi_enabled()

若 MPI 在当前运行时可用，返回 `True`。



### mpi_built()

若 Horovod 编译时包含了 MPI 支持，返回 `True`。



### nccl_built()

若 Horovod 编译时包含了 NCCL 支持，返回 `True`。



### rank()

返回当前进程的 Horovod rank。



### shutdown()

关闭 Horovod。



### size()

返回 Horovod 进程数。



### start_timeline()

创建时间线（日志）文件并开始记录。

```python
horovod.tensorflow.start_timeline(file_path, mark_cycles=False)
# file_path    时间线文件的路径
# mark_cycles  若为`True`,时间线中将标记循环
```



### stop_timeline()

停止记录时间线并关闭文件。



## horovod.tensorflow

### alltoall

All-to-all 操作。所有发送和接收的张量在第一个维度进行切分和拼接，因此各进程提供的张量必须具有相同的形状，除了第一个维度的规模可以不同。

```python
horovod.tensorflow.alltoall(tensor, splits=None, name=None, ignore_name_scope=False)
# tensor       分发的数据,是`tf.Tensor`,`tf.Variable`或`tf.IndexedSlices`类型
# splits       指示数据分发的数组,索引为i的整数n表示`tensor`接下来的n个元素向秩为i的进程发送.若为`None`,则`tensor`
#              的所有元素将被均分并发送到每个进程
```



### cross_rank()

返回当前进程所在节点的 rank。



### cross_size()

返回与当前进程具有相同本地 rank 的进程数。



### is_homogeneous()

若集群的所有节点上的进程数相同，返回 `True`。



## horovod.tensorflow.keras

### broadcast_global_variables()

根进程向所有（其它）进程广播所有全局变量。

```python
horovod.tensorflow.keras.broadcast_global_variables(root_rank)
# root_rank    发送数据的进程的秩
```



### callbacks.BroadcastGlobalVariablesCallback

根进程向所有（其它）进程广播所有全局变量，以确保所有的进程的模型初始化是一致的。

```python
horovod.tensorflow.keras.callbacks.BroadcastGlobalVariablesCallback(root_rank, device)
# root_rank    发送数据的进程的秩
```



### callbacks.MetricAverageCallback

在 epoch 结束后对所有进程的指标求平均，常配合 `ReduceLROnPlateau`, `TensorBoard` 和其它指标相关的回调使用（必须在回调列表中位于这些回调之前）。



### callbacks.LearningRateScheduleCallback

> 建议使用 Keras 的相关回调而非此回调。

计划学习率。



### callbacks.LearningRateWarmupCallback

> 建议使用 Keras 的相关回调而非此回调。

学习率 warmup。



### load_model

使用 Horovod 分布式优化器加载保存的 Keras 模型。分布式优化器将包装原优化器，使用其计算梯度值和应用参数更新。

```python
horovod.tensorflow.keras.load_model(filepath, custom_optimizers=None, custom_objects=None, compression=<class 'horovod.tensorflow.compression.NoneCompressor'>)
# filepath    模型的保存路径或h5格式的文件对象
```



## horovod.keras

### broadcast_global_variables()

根进程向所有（其它）进程广播所有全局变量。

```python
horovod.keras.broadcast_global_variables(root_rank)
# root_rank    发送数据的进程的秩
```



### load_model

见 `horovod.tensorflow.keras.load_model`。



### callbacks.BroadcastGlobalVariablesCallback, callbacks.MetricAverageCallback, callbacks.LearningRateScheduleCallback, callbacks.LearningRateWarmupCallback

见 `horovod.tensorflow.keras.callbacks.BroadcastGlobalVariablesCallback`, `horovod.tensorflow.keras.callbacks.MetricAverageCallback`, `horovod.tensorflow.keras.callbacks.LearningRateScheduleCallback`, `horovod.tensorflow.keras.callbacks.LearningRateWarmupCallback`。



## horovod.torch

### allgather_async()

All-Gather 操作的异步版本，返回此操作的用于 `poll()` 和 `synchronize()` 调用的柄。



### allreduce_async()

All-Reduce 操作的异步版本，返回此操作的用于 `poll()` 和 `synchronize()` 调用的柄。



### alltoall()

All-to-all 操作。所有发送和接收的张量在第一个维度进行切分和拼接，因此各进程提供的张量必须具有相同的形状，除了第一个维度的规模可以不同。

```python
horovod.torch.alltoall(tensor, splits=None, name=None)
# tensor       分发的数据,是`torch.Tensor`类型
# splits       指示数据分发的数组,索引为i的整数n表示`tensor`接下来的n个元素向秩为i的进程发送.若为`None`,则`tensor`
#              的所有元素将被均分并发送到每个进程
```



### alltoall_async()

All-to-all 操作的异步版本，返回此操作的用于 `poll()` 和 `synchronize()` 调用的柄。



### broadcast_async()

Broadcast 操作的异步版本，返回此操作的用于 `poll()` 和 `synchronize()` 调用的柄。



### broadcast_object()





### broadcast_optimizer_state()

从根进程广播优化器状态到所有其它进程。

```python
horovod.torch.broadcast_optimizer_state(optimizer, root_rank)
# optimizer      优化器
# root_rank      进程的rank,该进程的优化器将被广播到所有其它进程
```



### broadcast_parameters()

从根进程广播参数状态到所有其它进程，主要用于广播 `model.state_dict()`, `model.named_parameters()` 和 `model.parameters()`。

```python
horovod.torch.broadcast_parameters(params, root_rank)
# params         模型参数
# root_rank      进程的rank,该进程的优化器将被广播到所有其它进程
```



### cross_rank()

返回当前进程所在节点的 rank。



### cross_size()

返回与当前进程具有相同本地 rank 的进程数。



### join()

阻塞直到所有进程调用此方法，返回最后调用此方法的进程的 rank。



### poll()

若异步操作完成，返回 `True`，此时调用 `synchronize()` 将不再阻塞。

```python
horovod.torch.poll(handle)
# handle      异步操作返回的柄
```



### synchronize()

同步异步操作直到其完成，返回该操作的结果。

```python
horovod.torch.synchronize(handle)
# handle      异步操作返回的柄
```



