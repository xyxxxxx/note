# MPI 集体通信模式

## All-to-All

所有进程向所有进程发送任意数据，是最通用的模式。



## Broadcast（广播）

指定进程向所有进程发送相同的数据：

![broadcastvsscatter.png](https://i.loli.net/2021/06/18/r6ivAJhD3kfnqIx.png)



## Gather（收集）

所有进程向指定进程发送数据的等长片段并拼接：

![gather](https://mpitutorial.com/tutorials/mpi-scatter-gather-and-allgather/gather.png)



### All-Gather

所有进程各向所有进程发送相同的数据的等长片段并拼接，相当于 Gather+Broadcast：

![allgather](https://mpitutorial.com/tutorials/mpi-scatter-gather-and-allgather/allgather.png)



## Reduce（归约）

所有进程向指定进程发送数据的等长片段并归约：

![mpi_reduce_1](https://mpitutorial.com/tutorials/mpi-reduce-and-allreduce/mpi_reduce_1.png)

![mpi_reduce_2](https://mpitutorial.com/tutorials/mpi-reduce-and-allreduce/mpi_reduce_2.png)



### All-Reduce

所有进程各向所有进程发送相同的数据的等长片段并归约，相当于 Reduce+Broadcast：

![mpi_allreduce_1](https://mpitutorial.com/tutorials/mpi-reduce-and-allreduce/mpi_allreduce_1.png)



### Ring-All-Reduce







## Scatter

指定进程向所有进程分发数据的等长片段：

![scatter](https://i.loli.net/2021/06/18/KHQpmUqAhLk1y5B.png)







# 分布式策略

## Parameter Server







