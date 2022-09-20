`nvidia-smi` 是一个跨平台工具，提供监控 GPU 使用情况和更改 GPU 状态的功能。N 卡驱动附带此工具，只要安装好驱动就可以使用。下面介绍 `nvidia-smi` 命令系列。

## `nvidia-smi`

显示所有 GPU 的当前信息状态。

```shell
$ nvidia-smi
Wed Jan 27 09:25:50 2021       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 440.100      Driver Version: 440.100      CUDA Version: 10.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  TITAN X (Pascal)    Off  | 00000000:41:00.0 Off |                  N/A |
| 23%   32C    P8     9W / 250W |  11969MiB / 12194MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
+-----------------------------------------------------------------------------+
```

各参数为：

| `NVIDIA-SMI`        | `nvidia-smi`版本                                             |
| ------------------- | ------------------------------------------------------------ |
| `Driver Version`    | Nvidia驱动版本                                               |
| `CUDA Version`      | CUDA版本                                                     |
| `GPU`               | GPU编号                                                      |
| `Fan`               | 风扇转速shiyongGPU                                           |
| `Name`              | GPU型号                                                      |
| `Temp`              | 温度                                                         |
| `Perf`              | 性能状态，从最小（P0）到最大（P12）                          |
| `Persistence-M`     | 持续模式的状态（持续模式功耗大，但在新的GPU应用启动时花费时间更少） |
| `Pwr`               | 功耗                                                         |
| `Bus-Id`            | GPU总线，`domain:bus:device.function`                        |
| `Disp.A`            | Display Active，表示GPU的显示是否初始化                      |
| `Memory-Usage`      | 显存使用                                                     |
| `Volatile GPU-Util` | GPU使用率                                                    |
| `ECC`               | 是否开启错误检查和纠正技术，0/DISABLED, 1/ENABLED            |
| `Compute M.`        | 计算模式，0/DEFAULT,1/EXCLUSIVE_PROCESS,2/PROHIBITED         |
| `Processes`         | 正在使用GPU的进程                                            |

## `nvidia-smi -q`

显示所有 GPU 的当前详细信息状态。

```shell
$ nvidia-smi -q

==============NVSMI LOG==============

Timestamp                           : Wed Jan 27 10:09:09 2021
Driver Version                      : 440.100
CUDA Version                        : 10.2

Attached GPUs                       : 1
GPU 00000000:41:00.0
    Product Name                    : TITAN X (Pascal)
    Product Brand                   : GeForce
    Display Mode                    : Disabled
    Display Active                  : Disabled
    Persistence Mode                : Disabled
    Accounting Mode                 : Disabled
    Accounting Mode Buffer Size     : 4000
    Driver Model
        Current                     : N/A
        Pending                     : N/A
# ...
```

## 命令附加选项

```shell
$ nvidia-smi –i 0
# 指定编号为0的GPU

$ nvidia-smi –l 10
# 动态监视信息,每10s刷新一次(默认为5s)

$ nvidia-smi –f <file>
# 将信息写入指定文件中,不在终端显示
```

## 设备状态更改选项

```shell
$ nvidia-smi –r
# GPU复位
```

## `nvidia-smi dmon`

滚动显示 GPU 设备统计信息。

GPU 统计信息以一行的滚动格式显示，要监控的指标可以基于终端窗口的宽度进行调整。监控最多 4 个 GPU，如果没有指定任何 GPU，则默认监控 GPU0-GPU3（GPU 索引从 0 开始）。

```shell
$ nvidia-smi dmon
# gpu   pwr gtemp mtemp    sm   mem   enc   dec  mclk  pclk
# Idx     W     C     C     %     %     %     %   MHz   MHz
    0     9    32     -     0     0     0     0   405   139
    0     9    32     -     0     0     0     0   405   139
    0     9    32     -     0     0     0     0   405   139
# ...    
```

## `nvidia-smi pmon`

滚动显示 GPU 进程状态信息。

GPU 进程统计信息以一行的滚动格式显示，此工具列出了 GPU 所有进程的统计信息。要监控的指标可以基于终端窗口的宽度进行调整。监控最多 4 个 GPU，如果没有指定任何 GPU，则默认监控 GPU0-GPU3（GPU 索引从 0 开始）。
