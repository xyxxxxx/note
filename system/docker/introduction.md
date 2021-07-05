> 参考[Docker —— 从入门到实践](https://yeasy.gitbook.io/docker_practice/)
>
> 参考[Docker 入门教程](https://www.ruanyifeng.com/blog/2018/02/docker-tutorial.html)，[Docker 微服务教程](http://www.ruanyifeng.com/blog/2018/02/docker-wordpress-tutorial.html)，[Docker简介](https://www.qikqiak.com/k8s-book/docs/2.Docker%20%E7%AE%80%E4%BB%8B.html)



# 什么是Docker?

软件开发最大的麻烦事之一就是环境配置。想要正确运行软件，用户必须保证操作系统的设置和各种库和组件的安装都要正确。因此我们想要从根本上解决问题，即安装应用程序的时候把原始环境一模一样地复制过来。

虚拟机（virtual machine）就是带环境安装的一种解决方案。它可以在一种操作系统里面运行另一种操作系统，比如在 Windows 系统里面运行 Linux 系统。应用程序对此毫无感知，因为虚拟机看上去跟真实系统一模一样，而对于底层系统来说，虚拟机就是一个普通文件，不需要了就删掉，对其他部分毫无影响。

然而虚拟机存在一些缺点：

1. 资源占用多
2. 冗余步骤多
3. 启动慢

由于虚拟机存在这些缺点，Linux 发展出了另一种虚拟化技术：Linux 容器（Linux Containers，缩写为 LXC）。**Linux 容器不是模拟一个完整的操作系统，而是对进程进行隔离。**或者说，在正常进程的外面套了一个[保护层](https://opensource.com/article/18/1/history-low-level-container-runtimes)。对于容器里面的进程来说，它接触到的各种资源都是虚拟的，从而实现与底层系统的隔离。

进程级别的容器像是轻量级的虚拟机，能够提供虚拟化的环境，但是成本开销小得多。

**Docker 属于 Linux 容器的一种封装，提供简单易用的容器使用接口。**它是目前最流行的 Linux 容器解决方案。Docker 将应用程序与该程序的依赖，打包在一个文件里面。运行这个文件，就会生成一个虚拟容器。程序在这个虚拟容器里运行，就好像在真实的物理机上运行一样。有了 Docker，就不用担心环境问题。

Docker 的用途主要分为三类：

1.**提供一次性的环境**。比如，本地测试他人的软件、持续集成的时候提供单元测试和构建的环境。

2.**提供弹性的云服务**。因为 Docker 容器可以随开随关，很适合动态扩容和缩容。

3.**组建微服务架构**。通过多个容器，一台机器可以跑多个服务，因此在本机就可以模拟出微服务架构。


