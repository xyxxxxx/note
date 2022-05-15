参考 MIT 6.824 Distributed Systems Spring 2020

# 分布式系统

## 绪论

使用分布式系统的原因：

* 并行（以提高性能）
* 容错
* 系统本身在物理上是分布式的
* 安全/隔离

面临挑战：

* 并发运行
* 部分故障
* 高性能

基于分布式系统的基础设施：

* 分布式存储
* 分布式计算
* 通信系统

实现：

* RPC（远程过程调用）
* 线程
* 并发编程

性能：

* 理想情形：伸缩性（双倍的计算资源带来双倍的性能或吞吐量）

容错：

* 可用性：在部分故障下仍可继续运行
* 可恢复性
* 手段：非易失性存储（保存检查点与系统日志，但时间花销大），复制（在不同位置保存多个副本）

一致性：

* 强一致性：可以保证 GET 到刚刚 PUT 的数据（要求高，实现困难且开销大）
* 弱一致性：不能保证 GET 到刚刚 PUT 的数据，一段时间之后才能 GET 到（更加实际，现实中广泛使用）

## MapReduce

> [MapReduce: Simplified Data Processing on Large Clusters](https://pdos.csail.mit.edu/6.824/papers/mapreduce.pdf)

考虑一个简单的单词统计任务：

```
   (read from local         Vocabulary
    disk or network)       ┌──────┐┌──────┐┌──────┐
INPUT 1 ──────> Map(Count) │ a, 1 ││ b, 1 ││      │  (by Map worker)
INPUT 2 ──────> Map        │      ││ b, 1 ││      │
INPUT 3 ──────> Map        │ a, 2 ││      ││ c, 1 │
                           └──────┘└──────┘└──────┘
                              └───────┼───────┼─────> Reduce(Sum)  a, 3   (by Reduce worker)
                                      └───────┼─────> Reduce       b, 2
                       (shuffle by network)   └─────> Reduce       c, 1
```             

使用 MapReduce 框架如下：

* `Map(k, v)`：拆分文本 `v` 为单词，统计每一个单词的出现次数
    * `emit(k, v)`：发送单词 `k` 的出现次数 `v`
* `Reduce(k, v)`：对单词 `k` 的所有出现次数 `v`（是一个数组）求和
    * `emit(k, v)`：发送单词 `k` 的出现总次数 `v`

要点：

* MapReduce 任务可以组成流水线，来进行复杂的多阶段分析或实现迭代算法。
* 适用于能够转换为上述形式的算法。
* `Map()` 必须是独立的操作，即可以作用在任意的输入片段上。
* Map 工作器从**网络文件系统**（论文中为 GFS）获取输入数据，网络文件系统应为分布式存储以实现**并行**读取。或者干脆就将存储了部分输入数据的网络文件系统（GFS）服务器本身作为这一部分数据的 Map 工作器（在当时网络条件受限的情况下）。Map 工作器的 `emit()` 是将中间结果写在本地磁盘上。
* Reduce 工作器必须和每一个 Map 工作器进行网络通信以搜集必要的中间结果，这种行存储转换为列存储的过程（通过网络）称为 shuffle。Reduce 工作器的 `emit()` 是将部分的最终结果写在集群存储（网络文件系统）的指定文件上。
* 论文中（2004 年当时）所有的机器由一个以太网交换机连接。如今的数据中心的以太网交换机有多个副本，因而有远高于过去的网络吞吐量，现在使用 MapReduce 框架将不再像过去那样受制于网络速度。尽管如此，随着越来越多高性能框架的出现，MapReduce 已经不再被使用。

## 线程和 RPC

使用线程的原因：

* IO 并发：为每个 RPC 创建一个线程（goroutine），以实现 IO 并发（由于历史原因称为 IO 并发），实质上就是不同进度的重叠（部分活动在等待而部分活动在进行）。
* 多核并行：充分利用机器的多个 CPU（核心）。
* 便利：定期运行一段指定程序。

与线程相对的模式（或风格）：事件驱动编程（异步编程）。单个线程循环，等待任何输入或事件并执行相应操作。这种模式实现的 IO 并发不能利用多核并行；编程更加复杂，要将各种活动切成小块并设定激活条件。

在同时服务非常多（例如一百万个）客户的情况下，启动同样数量的线程将占用大量内存，调度的效率变低，bookkeeping 的花销变大。这时花费程序员的一些时间写一个事件驱动程序反而能做到效率更高，开销更低。

使用线程的挑战：

* 竞险（使用共享变量时），使用锁（`sync.Mutex`）解决
* 协调多个线程的互动，使用通道（`channel`）、条件变量（`sync.Cond`）和等待组（`sync.WaitGroup`）解决
* 死锁

## 分布式存储系统

> [The Google File System](https://pdos.csail.mit.edu/6.824/papers/gfs.pdf)

难点：

* 性能 → 数据分片（sharding）
* 出错 → 容错 → 维护多个副本 → 一致性（高或低） → 降低性能

强一致性可以理解为就像是在向一个单线程服务器发送请求，每个请求都能看到最新的数据，或者说数据反映了先前的所有操作。

不好的副本设计：多个服务器各自维护自己的一个副本，客户端向每个服务器发送相同的请求。由于网络通信的不可靠性，可能会出现部分服务器未收到请求（或很晚收到请求）、不同的服务器以不同次序收到多个请求等情况。

建立 GFS 的动机：分布式存储巨型或海量文件（例如爬取的网络内容、Youtube 视频、建立搜索索引的中间文件、Web 服务器的日志文件、大型数据集等），高速并行访问（分片存储以提高总吞吐量），全局访问和权限控制，自动恢复。

GFS 部署在单个数据中心上，供谷歌内部使用（尽管谷歌也会出售基于 GFS 的其他服务）。GFS 为大型顺序（而非随机）文件读写以多种方式进行了专门定制。论文提出了在当时不被广泛接受的观点，即弱一致性的分布式存储系统是可以接受的，GFS 不保证返回数据的正确性，其目标在于获得更好的性能。GFS 只使用了一个 master（而非多个 master 进行分工），并且也工作得很好。

GFS master（参见论文图 1）主要维护两个表，其一将文件名映射到块柄（chunk handle）的数组，其二将块柄映射到