> 参考[Docker —— 从入门到实践](https://yeasy.gitbook.io/docker_practice/)

镜像（`Image`）和容器（`Container`）的关系，就像是面向对象程序设计中的 `类` 和 `实例` 一样，镜像是静态的定义，容器是镜像运行时的实体。容器可以被创建、启动、停止、删除、暂停等。

容器的实质是进程，但与直接在宿主执行的进程不同，容器进程运行于属于自己的独立的 [命名空间](https://en.wikipedia.org/wiki/Linux_namespaces)。因此容器可以拥有自己的 `root` 文件系统、自己的网络配置、自己的进程空间，甚至自己的用户 ID 空间。容器内的进程是运行在一个隔离的环境里，使用起来，就好像是在一个独立于宿主的系统下操作一样。这种特性使得容器封装的应用比直接在宿主运行更加安全。也因为这种隔离的特性，很多人初学 Docker 时常常会混淆容器和虚拟机。

前面讲过镜像使用的是分层存储，容器也是如此。每一个容器运行时，是以镜像为基础层，在其上创建一个当前容器的存储层，我们可以称这个为容器运行时读写而准备的存储层为 **容器存储层**。

容器存储层的生存周期和容器一样，容器消亡时，容器存储层也随之消亡。因此，任何保存于容器存储层的信息都会随容器删除而丢失。

按照 Docker 最佳实践的要求，容器不应该向其存储层内写入任何数据，容器存储层要保持无状态化。所有的文件写入操作，都应该使用 [数据卷（Volume）]()、或者 [绑定宿主目录]()，在这些位置的读写会跳过容器存储层，直接对宿主（或网络存储）发生读写，其性能和稳定性更高。数据卷的生存周期独立于容器，容器消亡，数据卷不会消亡。因此，使用数据卷后，容器删除或者重新运行之后，数据却不会丢失。





# 容器操作

## 运行容器

以下命令创建一个新的容器并运行

```shell
$ docker run [option] [image_name]
# -d 容器启动后在后台运行
# -it 容器的shell映射到当前的shell
# -p [IP_addr:port] 容器的指定端口映射到指定地址或本机的指定端口
# --env [key]=[value] 向容器传入一个环境变量
# --link [container] 容器连接到指定容器
# --name [name] 容器命名
# --rm 停止运行后自动删除容器文件
# --volume [dir]:[container_dir] 将指定目录映射到容器的指定目录，因此对当前目录的任何修改都会反映到容器里面
```

示例

```shell
$ docker run ubuntu:16.04 /bin/echo 'Hello world'
# 启动容器, 执行一个文件, 终止
$ docker run -it ubuntu:16.04 /bin/bash
# 启动一个bash终端
```

当利用`docker run`来创建容器时，Docker 在后台运行的标准操作包括：

+ 检查本地是否存在指定的镜像，不存在就从公有仓库下载
+ 利用镜像创建并启动一个容器
+ 分配一个文件系统，并在只读的镜像层外面挂载一层可读写层
+ 从宿主主机配置的网桥接口中桥接一个虚拟接口到容器中去
+ 从地址池配置一个 ip 地址给容器
+ 执行用户指定的应用程序
+ 执行完毕后容器被终止



其它常用的容器运行命令如下

```shell
# 启动已终止容器
$ docker start <container ID or name>

# 终止容器运行
$ docker stop <container ID or name>

# 重启运行的容器
$ docker restart <container ID or name>

# 终止容器运行
$ docker kill <container ID or name>
```



## 管理容器

```shell
# 列出本机正在运行的容器
$ docker container ls

# 列出本机的所有容器
$ docker container ls -a
$ docker ps -a

# 获取容器的输出信息
$ docker container logs <container ID or name>

# 删除容器
$ docker rm <container ID or name>
# -r 删除正在运行的容器

# 清除所有处于终止状态的容器
$ docker container prune
```



## 进入容器

```shell
# 进入一个正在运行的docker容器
$ docker exec -it <container ID or name> /bin/bash
# exit 退出时不会导致容器的停止
```



## 导入导出容器

```shell
# 导出容器
$ docker export 7691a814370e > ubuntu.tar

# 导入镜像
$ cat ubuntu.tar | docker import - test/ubuntu:v1.0
```

