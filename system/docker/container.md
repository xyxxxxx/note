[toc]

镜像（`Image`）和容器（`Container`）的关系，就像是面向对象程序设计中的 `类` 和 `实例` 一样，镜像是静态的定义，容器是镜像运行时的实体。容器可以被创建、启动、停止、删除、暂停等。

容器的实质是进程，但与直接在宿主执行的进程不同，容器进程运行于属于自己的独立的命名空间，因此容器可以拥有自己的 `root` 文件系统、自己的网络配置、自己的进程空间，甚至自己的用户 ID 空间。容器内的进程是运行在一个隔离的环境里，使用起来就好像是在一个独立于宿主的系统下操作一样。这种特性使得容器封装的应用比直接在宿主运行更加安全。也因为这种隔离的特性，很多人初学 Docker 时常常会混淆容器和虚拟机。



# 运行容器

根据指定镜像创建一个新的容器并运行的命令为：

```shell
$ docker run [option] <image_name> <command>
# -d              后台运行容器并打印容器ID
# --dns           设置自定义DNS服务器
# -e              设置环境变量
# -expose         暴露一个或多个端口
# --gpus          添加GPU设备到容器
# -it             启动交互式终端
# --link          容器连接到指定容器
# --mount         挂载主机目录或数据卷
# -p              容器的指定端口映射到本机的指定端口
# -P              容器的所有暴露的端口全部映射到本机的随机端口
# --name          为容器命名
# --privileged    给予容器额外的特权
# --rm            容器退出后自动移除
# --tmpfs         挂载tmpfs目录
# -v              挂载主机目录或数据卷,建议使用`--mount`
# --volumes-from  挂载指定容器挂载的数据卷
```

示例：

```shell
# 启动容器,执行命令,终止容器
$ docker run ubuntu:18.04 /bin/echo 'Hello world'

# 启动容器,启动bash终端以及交互式终端,退出时终止容器
$ docker run -it ubuntu:18.04 [/bin/bash]

# 后台启动容器,启动bash终端以及交互式终端
$ docker run -dit ubuntu:18.04 [/bin/bash]
```

当使用 `docker run` 命令创建容器时，Docker 在后台的运行步骤包括：

* 检查本地是否存在指定的镜像，不存在就从公有仓库下载
* 利用镜像创建并启动一个容器
* 分配一个文件系统，并在只读的镜像层外面挂载一层可读写层
* 从宿主主机配置的网桥接口中桥接一个虚拟接口到容器中去
* 从地址池分配一个 ip 地址给容器
* 执行用户指定的应用程序
* 执行完毕后终止容器



其它常用的容器运行命令如下：

```shell
# 启动已终止容器
$ docker container start <container ID or name>

# 终止容器运行
$ docker container stop <container ID or name>

# 重启运行的容器
$ docker container restart <container ID or name>

# 终止容器运行
$ docker kill <container ID or name>
```





# 管理容器

使用 `docker container ls` 命令列出本机的容器：

```shell
$ docker container ls      # 列出正在运行的容器
$ docker container ls -l   # 列出详细信息
$ docker container ls -a   # 列出所有容器,包括终止的容器
$ docker ps -a

# 获取容器的输出信息
$ docker container logs <container ID or name>

# 删除终止的容器
$ docker container rm <container ID or name>
#                     -f 删除正在运行的容器

# 清除所有处于终止状态的容器
$ docker container prune
```



使用 `docker exec` 命令进入正在运行的容器：

```shell
$ docker exec -it <container ID or name> /bin/bash
# 使用`exit`命令退出时不会导致容器的停止
```



使用 `docker export` 命令导出本地容器为快照文件：

```shell
$ docker container ls                     
CONTAINER ID   IMAGE                          COMMAND       CREATED         STATUS         PORTS     NAMES
8f84b6fe9178   ubuntu:18.04                   "/bin/bash"   6 seconds ago   Up 5 seconds             test
$ docker export 8f84b6fe9178 > ubuntu.tar
```



使用 `docker import` 命令

```shell
$ cat ubuntu.tar | docker import - test/ubuntu:v1.0
```



> `docker export` 





# 网络配置

## 外部访问容器

容器中可以运行一些网络应用，要让外部也可以访问这些应用，可以通过 `-P` 或 `-p` 参数来指定端口映射。

当使用 `-P` 时，Docker 会随机映射一个端口到内部容器开放的网络端口。下面这个例子中，本地主机的 32768 被映射到容器的 80 端口，访问本机的 32768 端口即可访问容器内 nginx 的默认页面。

```shell
$ docker run -d -P nginx:alpine

$ docker container ls -l
CONTAINER ID        IMAGE               COMMAND                  CREATED             STATUS              PORTS                   NAMES
fae320d08268        nginx:alpine        "/docker-entrypoint.…"   24 seconds ago      Up 20 seconds       0.0.0.0:32768->80/tcp   bold_mcnulty
```

`-p` 则可以指定要映射的端口，一个指定端口上只可以绑定一个容器。支持的格式有 `ip:hostPort:containerPort|ip::containerPort|hostPort:containerPort`。

```shell
# 本地主机所有地址的80端口映射到容器的80端口
$ docker run -d -p 80:80 nginx:alpine

# 本地主机特定地址的80端口映射到容器的80端口
$ docker run -d -p 127.0.0.1:80:80 nginx:alpine

# 本地主机特定地址的随机端口映射到容器的80端口
$ docker run -d -p 127.0.0.1::80 nginx:alpine
# 本地主机特定地址的随机UDP端口映射到容器的80端口
$ docker run -d -p 127.0.0.1::80/udp nginx:alpine

# 绑定多个端口
$ docker run -d \
    -p 80:80 \
    -p 443:443 \
    nginx:alpine
```

使用 `docker port` 查看当前映射的端口配置

```shell
$ docker port fa 80
0.0.0.0:32768
```



## 容器互联

创建一个新的 Docker 网络。

```shell
$ docker network create -d bridge my-net
```

运行一个容器并连接到新建的 `my-net` 网络

```shell
$ docker run -it --rm --name busybox1 --network my-net busybox sh
```

打开新的终端，再运行一个容器并加入到 `my-net` 网络

```shell
$ docker run -it --rm --name busybox2 --network my-net busybox sh
```

再打开一个新的终端查看容器信息

```shell
$ docker container ls
CONTAINER ID        IMAGE               COMMAND             CREATED             STATUS              PORTS               NAMES
b47060aca56b        busybox             "sh"                11 minutes ago      Up 11 minutes                           busybox2
8720575823ec        busybox             "sh"                16 minutes ago      Up 16 minutes                           busybox1
```

下面通过 `ping` 来证明 `busybox1` 容器和 `busybox2` 容器建立了互联关系。在 `busybox1` 容器输入以下命令

```shell
/ # ping busybox2
PING busybox2 (172.19.0.3): 56 data bytes
64 bytes from 172.19.0.3: seq=0 ttl=64 time=0.072 ms
64 bytes from 172.19.0.3: seq=1 ttl=64 time=0.118 ms
```

用 ping 来测试连接 `busybox2` 容器，它会解析成 `172.19.0.3`。

同理在 `busybox2` 容器执行 `ping busybox1`，也会成功连接到。

```shell
/ # ping busybox1PING busybox1 (172.19.0.2): 56 data bytes64 bytes from 172.19.0.2: seq=0 ttl=64 time=0.064 ms64 bytes from 172.19.0.2: seq=1 ttl=64 time=0.143 ms
```

这样 `busybox1` 容器和 `busybox2` 容器建立了互联关系。





## 配置DNS







# 数据管理

> 参考：
>
> [Manage data in Docker](https://docs.docker.com/storage/)
>
> [Docker——从入门到实践 数据管理](https://yeasy.gitbook.io/docker_practice/data_management)







## 数据卷

### 创建和管理数据卷

创建数据卷：

```shell
$ docker volume create my-vol
```

查看所有数据卷：

```shell
$ docker volume ls
DRIVER              VOLUME NAME
local               my-vol
```

检查数据卷：

```shell
$ docker volume inspect my-vol
[
    {
        "Driver": "local",
        "Labels": {},
        "Mountpoint": "/var/lib/docker/volumes/my-vol/_data",
        "Name": "my-vol",
        "Options": {},
        "Scope": "local"
    }
]
```

删除数据卷：

```shell
$ docker volume rm my-vol
```

删除无主的数据卷：

```shell
$ docker volume prune
```



### 启动容器并挂载数据卷

下面的例子挂载数据卷 `my-vol` 到容器的 `/usr/share/nginx/html` 目录下：

```shell
# --mount
$ docker run -d -P \
    --name web \
    --mount source=my-vol,target=/usr/share/nginx/html \
    # type             挂载的类型,对于数据卷为`volume`,这里根据`source`传入的数据卷名称推断为`volume`
    # source,src       数据卷的名称.对于匿名数据卷,该参数将被省略
    # target,destination,dst   在容器中挂载的路径
    nginx:latest

# -v
$ docker run -d -P \
    --name web \
    -v my-vol:/usr/share/nginx/html \
    # 第一个参数  数据卷的名称.对于匿名数据卷,该参数将被省略
    # 第二个参数  在容器中挂载的路径
    nginx:latest
```

使用 `docker inspect` 命令查看 `web` 容器的信息，其中包括挂载信息：

```shell
$ docker inspect web
# ...
"Mounts": [
    {
        "Type": "volume",
        "Name": "my-vol",
        "Source": "/var/lib/docker/volumes/my-vol/_data",
        "Destination": "/usr/share/nginx/html",
        "Driver": "local",
        "Mode": "",
        "RW": true,
        "Propagation": ""
    }
],
# ...
```



若要将数据卷挂载为只读目录，在选项列表中添加 `ro` 或 `readonly`：

```shell
# --mount
$ docker run -d -P \
    --name web \
    --mount source=my-vol,target=/usr/share/nginx/html,ro \
    nginx:latest

# -v
$ docker run -d -P \
    --name web \
    -v my-vol:/usr/share/nginx/html:ro \
    nginx:latest
```

使用 `docker inspect` 命令验证正确创建了只读挂载：

```shell
$ docker inspect web
# ...
"Mounts": [
    {
        "Type": "volume",
        "Name": "my-vol",
        "Source": "/var/lib/docker/volumes/my-vol/_data",
        "Destination": "/usr/share/nginx/html",
        "Driver": "local",
        "Mode": "",
        "RW": false,           # 只读
        "Propagation": ""
    }
],
# ...
```



### 备份、恢复和迁移数据卷

例如我们创建名为 `dbstore` 的容器并挂载（命名或匿名）数据卷：

```shell
$ docker run -v /dbdata --name dbstore ubuntu /bin/bash
```

使用以下命令备份此数据卷：

```shell
$ docker run --rm --volumes-from dbstore -v $(pwd):/backup ubuntu tar cvf /backup/backup.tar /dbdata
# 此命令:
# * 启动一个新的容器并挂载`dbstore`容器挂载的数据卷
# * 挂载主机目录到容器的`/backup`目录下
# * 备份`/dbdata`目录(即数据卷)并保存到`/backup`目录下
```

然后创建名为 `dbstore2` 的容器并挂载数据卷：

```shell
$ docker run -v /dbdata --name dbstore2 ubuntu /bin/bash
```

使用以下命令即可恢复备份的数据卷（到当前挂载的数据卷中）：

```shell
$ docker run --rm --volumes-from dbstore2 -v $(pwd):/backup ubuntu bash -c "cd /dbdata && tar xvf /backup/backup.tar --strip 1"
```

数据卷的备份文件 `backup.tar` 可以十分方便地迁移。



## 挂载主机目录

### 启动容器并挂载主机目录

下面的例子挂载本地主机的当前工作目录下的 `webapp` 目录到容器的 `/usr/share/nginx/html` 目录下：

```shell
# --mount
$ docker run -d -P \
    --name web \
    --mount type=bind,source="$(pwd)"/webapp,target=/usr/share/nginx/html \
    # type             挂载的类型,对于主机目录为`bind`
    # source,src       挂载的主机目录的路径,须为绝对路径,可以是不存在的路径
    # target,destination,dst   在容器中挂载的路径
    nginx:latest

# -v
$ docker run -d -P \
    --name web \
    -v "$(pwd)"/webapp:/usr/share/nginx/html \
    # 第一个参数  挂载的主机目录的路径,须为绝对路径,须为存在的路径
    # 第二个参数  在容器中挂载的路径
    nginx:latest
```



若要将主机目录挂载为只读目录，在选项列表中添加 `ro` 或 `readonly`：

```shell
# --mount
$ docker run -d -P \
    --name web \
    --mount type=bind,source="$(pwd)"/webapp,target=/usr/share/nginx/html,ro \
    nginx:latest

# -v
$ docker run -d -P \
    --name web \
    -v "$(pwd)"/webapp:/usr/share/nginx/html:ro \
    nginx:latest
```

使用 `docker inspect` 命令验证正确创建了只读挂载：

```shell
$ docker inspect web
# ...
"Mounts": [
    {
        "Type": "volume",
        "Name": "my-vol",
        "Source": "/var/lib/docker/volumes/my-vol/_data",
        "Destination": "/usr/share/nginx/html",
        "Driver": "local",
        "Mode": "",
        "RW": false,           # 只读
        "Propagation": ""
    }
],
# ...
```



上面的命令加载主机的 `/src/webapp` 目录到容器的 `/usr/share/nginx/html` 目录。这个功能在进行测试的时候十分方便，比如用户可以放置一些程序到本地目录中，来查看容器是否正常工作。本地目录的路径必须是绝对路径。

Docker 挂载主机目录的默认权限是 `读写`，用户也可以通过增加 `readonly` 指定为 `只读`。

```shell
$ docker run -d -P \
    --name web \
    # -v /src/webapp:/usr/share/nginx/html:ro \
    --mount type=bind,source=/src/webapp,target=/usr/share/nginx/html,readonly \
    nginx:alpine
```





