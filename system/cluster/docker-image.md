> 参考[Docker —— 从入门到实践](https://yeasy.gitbook.io/docker_practice/)

我们知道操作系统分为内核和用户空间。对于 Linux 而言，内核启动后会挂载 `root` 文件系统为其提供用户空间支持。而 Docker 镜像（image），就相当于是一个 `root` 文件系统。比如官方镜像 `ubuntu:18.04` 就包含了完整的一套 Ubuntu 18.04 最小系统的 `root` 文件系统。

Docker 镜像是一个特殊的文件系统，除了提供容器运行时所需的程序、库、资源、配置等文件外，还包含了一些为运行时准备的一些配置参数（如匿名卷、环境变量、用户等）。镜像不包含任何动态数据，其内容在构建之后也不会被改变。





# 镜像管理

## 获取镜像

Docker 运行容器前需要本地存在对应的镜像，如果本地不存在该镜像，Docker 会从镜像仓库下载该镜像。

从 Docker 镜像仓库获取镜像文件的命令为

```shell
$ docker pull [option] [Docker_Registry_addr[:port]/][username/]app_name[:tag]
# Docker镜像仓库地址的格式一般是<域名/IP>[:端口号]，默认地址是Docker Hub(docker.io)
# 对于Docker Hub, username为library, 即官方镜像
```

示例

```shell
$ docker pull ubuntu:18.04
```



## 列出镜像

列出已经下载到本地的所有镜像文件的命令为

```shell
$ docker image ls
# REPOSITORY           TAG                 IMAGE ID            CREATED             SIZE
# redis                latest              5f515359c7f8        5 days ago          183 MB
# nginx                latest              05a60462f8ba        5 days ago          181 MB
# mongo                3.2                 fe9198c04d62        5 days ago          342 MB
# <none>               <none>              00285df0df87        5 days ago          342 MB
# ubuntu               18.04               f753707788c5        4 weeks ago         127 MB
# ubuntu               latest              f753707788c5        4 weeks ago         127 MB
```

其中镜像 ID 是镜像的唯一标识，一个镜像可以对应多个标签。

也可以根据仓库名和标签列出指定镜像

```shell
$ docker image ls ubuntu
# REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE
# ubuntu              18.04               f753707788c5        4 weeks ago         127 MB
# ubuntu              latest              f753707788c5        4 weeks ago         127 MB

$ docker image ls ubuntu:18.04
# REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE
# ubuntu              18.04               f753707788c5        4 weeks ago         127 MB
```



### 镜像体积

这里给出的镜像大小和在 Docker Hub 上看到的镜像大小不同，例如`ubuntu:18.04` 镜像大小在本地是 `127 MB`，但是在 [Docker Hub](https://hub.docker.com/_/ubuntu?tab=tags) 显示的却是 `50 MB`。这是因为 Docker Hub 中显示的体积是压缩后的体积，镜像在下载和上传过程中保持着压缩状态；而`docker image ls` 给出的是镜像下载到本地后展开的各层所占空间的总和，是磁盘空间占用的大小。

另一个需要注意的地方在于，实际磁盘空间占用并非`docker image ls`给出的镜像大小之和，因为 Docker 镜像是多层存储结构，并且可以继承、复用，不同镜像可能使用相同的基础镜像，即拥有共同的层，共同的层仅需保存一份，因此实际磁盘空间占用很可能远小于镜像大小之和。



### 虚悬镜像

仓库名和标签均为`<none>`的镜像称为**虚悬镜像(dangling image)**。当镜像的仓库名和标签被赋予到另一个镜像时，镜像即变为虚悬镜像。

虚悬镜像失去了作用，使用以下命令以删除

```shell
$ docker image prune
```



### 中间层镜像

为了加速镜像构建、重复利用资源，Docker 会利用**中间层镜像**。以下命令列出包含中间层镜像的所有镜像

```shell
$ docker image ls -a
```

中间层镜像也是无标签的，但是它们不应该被删除，否则会导致上层镜像因为依赖丢失而出错。当删除某些上层镜像后，不再被依赖的中间层镜像也会被删除。



## 删除本地镜像

删除本地镜像的命令为

```shell
$ docker image rm [option] <image_name> [<image1_name> ...]
```

其中`<image_name>`可以是`镜像完整ID`，`镜像短ID`，`镜像名`或`镜像摘要`，例如

```shell
# redis                latest              5f515359c7f8        5 days ago          183 MB
$ docker image rm 5f5
$ docker image rm redis:latest
$ docker image rm $(docker image ls -q redis)
# 删除所有仓库名为redis的镜像
```

删除某个镜像的操作实际上是先取消镜像的标签(`Untagged`)，若镜像仍有另外的标签，则得到保留，否则删除(`Deleted`)此镜像。



# `commit`理解镜像构成



`docker image ls` 列表中的镜像体积总和并非是所有镜像实际硬盘消耗。由于 Docker 镜像是多层存储结构，并且可以继承、复用，因此不同镜像可能会因为使用相同的基础镜像，从而拥有共同的层。由于 Docker 使用 Union FS，相同的层只需要保存一份即可，因此实际镜像硬盘占用空间很可能要比这个列表镜像大小的总和要小的多。

可以通过以下命令查看镜像、容器、数据卷所占用的空间

```shell
$ docker system df
```



# `Dockerfile`定制镜像