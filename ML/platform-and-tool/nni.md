> nni的Github页面https://github.com/microsoft/nni

**NNI(Neural Network Intelligence)**是一个工具包，帮助用户设计机器学习模型的神经网络架构，调优模型的超参数等。



## 主要概念

+ *Experiment*：一个具体任务，例如寻找给定模型的最优超参或最佳神经网络架构。experiment 由多个 trial 和自动机器学习算法组成。
+ *Search Space*：调优模型的搜索域，例如每一个参数的取值范围。
+ *Configuration*：search space 的一个实例，即一组参数，每个参数都有一个特定值。
+ *Trial*：使用给定的 configutation 运行模型的单次尝试。
+ *Tuner*：一个 AutoML 算法，接受上一次 trial 的结果并为此次 trial 生成 configuration。
+ *Assessor*：分析一次 trial 的中间结果（例如每个epoch的训练或验证损失）以决定是否提前停止。
+ *Training Platform*：trials 的执行环境，取决于 experiment 的 configuration，它可以是本机、远程服务器或大规模训练平台

一个 experiment 的运行过程大体上为：tuner 接受 search space 并生成 configuration，这些 configuration 将被提交到训练平台，如本机、远程服务器或训练集群。模型的表现会被返回给 tuner，然后再生成并提交新的 configuration。

对于每次 experiment，用户只需要定义 search space，改动几行代码，就能利用 NNI 内置的 tuner/assessor 和训练平台来搜索最佳的超参和神经网络架构。 基本上分为三步：

![](https://user-images.githubusercontent.com/23273522/51816627-5d13db80-2302-11e9-8f3e-627e260203d5.jpg)



## 核心功能

### 超参调优

见上。



### 通用 NAS 框架

此 NAS 框架可供用户轻松指定候选的神经网络架构，例如，可以为单个层指定多个候选操作（例如，可分离的 conv、扩张 conv），并指定可能的跳过连接。 NNI 将自动找到最佳候选。 另一方面，NAS 框架为其他类型的用户（如 NAS 算法研究人员）提供了简单的接口，以实现新的 NAS 算法。 NAS 详情及用法参考[这里](https://github.com/microsoft/nni/blob/master/docs/zh_CN/NAS/Overview.md)。

NNI 通过 Trial SDK 支持多种 one-shot（一次性） NAS 算法，如ENAS、DARTS。 使用这些算法时，无需启动 NNI Experiment，在 trial 代码中加入算法并直接运行即可。 如果要调整算法中的超参数，或运行多个实例，可以选择一个 Tuner 并启用 NNI Experiment。

除了 one-shot NAS 外，NAS 还能以 NNI 模式运行，其中每个候选的网络结构都作为独立 Trial 任务运行。 在此模式下，与超参调优类似，必须启用 NNI Experiment 并为 NAS 选择 Tuner。



### 模型压缩





### 自动特征工程





## 快速入门

以MNIST分类模型的pytorch实现为例，完整代码请参考https://github.com/microsoft/nni/tree/master/examples/trials/mnist-pytorch。

### 三步启用 Experiment

**第一步**：编写  JSON 格式的 `search space` 文件，包括所有需要搜索的超参的 `名称` 和 `分布` （离散和连续值均可）。

```json
{
    "batch_size": {"_type":"choice", "_value": [16, 32, 64, 128]},
    "hidden_size":{"_type":"choice","_value":[128, 256, 512, 1024]},
    "lr":{"_type":"choice","_value":[0.0001, 0.001, 0.01, 0.1]},
    "momentum":{"_type":"uniform","_value":[0, 1]}
}

```

**第二步**：修改 `trial` 代码来从 NNI 获取超参，并返回 NNI 最终结果。

```python
import nni

def main(args):

    for epoch in range(1, args['epochs'] + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test_acc = test(args, model, device, test_loader)

        # report intermediate result
        nni.report_intermediate_result(test_acc)

    # report final result
    nni.report_final_result(test_acc)


def get_params():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument("--data_dir", type=str,
                        default='./data', help="data directory")
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument("--batch_num", type=int, default=None)
    parser.add_argument("--hidden_size", type=int, default=512, metavar='N',
                        help='hidden layer size (default: 512)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log_interval', type=int, default=1000, metavar='N',
                        help='how many batches to wait before logging training status')


    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    try:
        # get parameters form tuner
        tuner_params = nni.get_next_parameter()
        logger.debug(tuner_params)
        params = vars(merge_parameter(get_params(), tuner_params))
        print(params)
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise

```

**第三步**：定义 YAML 格式的`configuration`文件，其中声明了搜索空间和 Trial 文件的`路径`。它还包含了一些其他信息，例如调整算法，最大 Trial 运行次数和最大持续时间的参数。

```yml
authorName: default
experimentName: example_mnist_pytorch
trialConcurrency: 1
maxExecDuration: 1h
maxTrialNum: 10
#choice: local, remote, pai
trainingServicePlatform: local
searchSpacePath: search_space.json
#choice: true, false
useAnnotation: false
tuner:
  #choice: TPE, Random, Anneal, Evolution, BatchTuner, MetisTuner, GPTuner
  #SMAC (SMAC should be installed through nnictl)
  builtinTunerName: TPE
  classArgs:
    #choice: maximize, minimize
    optimize_mode: maximize
trial:
  command: python3 mnist.py
  codeDir: .
  gpuNum: 0

```



然后就可以在命令行通过配置文件 `config.yml`文件启动 Experiment。

```shell
nnictl create --config nni/examples/trials/mnist-tfv1/config.yml
```

> ``nnictl`` 是一个命令行工具，用来控制 NNI Experiment，如启动、停止、继续 Experiment，启动、停止 NNIBoard 等等。访问[文档](https://nni.readthedocs.io/en/latest/Tutorial/Nnictl.html)以查看 ``nnictl`` 的更多用法。

在命令行中等待输出`INFO:  Successfully started experiment!`，此消息表示 experiment 已成功启动。 期望的输出如下：

```
INFO:  Starting restful server...
INFO:  Successfully started Restful server!
INFO:  Setting local config...
INFO:  Successfully set local config!
INFO:  Starting experiment...
INFO:  Successfully started experiment!
------------------------------------------------------------------------------------
The experiment id is DHn70CaE
The Web UI urls are: http://127.0.0.1:8080   http://192.168.14.73:8080   http://172.17.0.1:8080
------------------------------------------------------------------------------------

You can use these commands to get more information about the experiment
------------------------------------------------------------------------------------
         commands                       description
1. nnictl experiment show        show the information of experiments
2. nnictl trial ls               list all of trial jobs
3. nnictl top                    monitor the status of running experiments
4. nnictl log stderr             show stderr log content
5. nnictl log stdout             show stdout log content
6. nnictl stop                   stop an experiment
7. nnictl trial kill             kill a trial job by id
8. nnictl --help                 get help information about nnictl
------------------------------------------------------------------------------------
```

如果根据上述步骤准备好了相应的 `trial`, `search space`和`configuration`，并成功创建了 NNI 任务。NNI 会根据 `search space` 自动地在每次 `trial` 使用不同的超参组合。 通过 Web 界面可看到 NNI 的进度。



## Web界面

在浏览器中打开命令行给出的`Web UI url`，就可以看到 experiment 的详细信息。