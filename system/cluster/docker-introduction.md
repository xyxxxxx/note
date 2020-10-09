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

1. **提供一次性的环境**。比如，本地测试他人的软件、持续集成的时候提供单元测试和构建的环境。

2. **提供弹性的云服务**。因为 Docker 容器可以随开随关，很适合动态扩容和缩容。

3. **组建微服务架构**。通过多个容器，一台机器可以跑多个服务，因此在本机就可以模拟出微服务架构。





# 镜像和容器的基本操作

> 如果出现权限问题：
>
> ```
> Got permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock: Get http://%2Fvar%2Frun%2Fdocker.sock/v1.40/version: dial unix /var/run/docker.sock: connect: permission denied
> ```
>
> 使用`sudo docker`命令

## 镜像管理

```shell
# 获取镜像文件
$ docker pull [option] [Docker_Registry_addr[:port]/][username/]app_name[:tag]
# Docker镜像仓库地址的格式一般是<域名/IP>[:端口号]，默认地址是Docker Hub
# 用户名默认为library,即官方镜像
$ docker pull ubuntu:16.04

# 列出本机的所有镜像文件
$ docker image ls

# 删除镜像文件
$ docker image rm [image_name]
```



## 运行

```shell
# 从指定image文件生成一个新的容器并运行
$ docker [option] run [image_name]
# -d 容器启动后在后台运行
# -it 容器的shell映射到当前的shell
# -p [IP_addr:port] 容器的指定端口映射到指定地址或本机的指定端口
# --env [key]=[value] 向容器传入一个环境变量
# --link [container] 容器连接到指定容器
# --name [name] 容器命名
# --rm 停止运行后自动删除容器文件
# --volume [dir]:[container_dir] 将指定目录映射到容器的指定目录，因此对当前目录的任何修改都会反映到容器里面
$ docker run ubuntu:16.04 /bin/echo 'Hello world'
# 执行一个命令
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



```shell
# 列出本机正在运行的容器
$ docker container ls

# 运行已终止容器
$ docker start [containerID]

# 终止容器运行
$ docker stop [containerID]

# 重启运行的容器
$ docker restart [containerID]

# 终止容器运行
$ docker kill [containerID]

# 进入一个正在运行的docker容器
$ docker exec -it [containerID] /bin/bash

```



## 容器管理

```shell
# 列出本机所有容器，包括终止运行的容器
$ docker ps -a

# 删除容器
$ docker rm [containerID]
# -r 删除正在运行的容器

# 清除所有处于终止状态的容器
$ docker container prune

```





# 定制镜像

## docker commit定制镜像

以定制一个 Web 服务器为例：

```shell
$ docker run --name webserver -d -p 80:80 nginx
```

这条命令会用 nginx 镜像启动一个容器，命名为 webserver，并且映射了 80 端口，这样我们可以用浏览器去访问这个 nginx 服务器。

假设我们非常不喜欢这个欢迎页面，我们希望改成欢迎 Docker 的文字，我们可以使用 docker exec命令进入容器，修改其内容。

```shell
$ docker exec -it webserver bash
root@3729b97e8226:/# echo '<h1>Hello, Docker!</h1>' > /usr/share/nginx/html/index.html
root@3729b97e8226:/# exit
```

我们修改了容器的文件，也就是改动了容器的存储层。我们可以通过`docker diff`命令看到具体的改动。

```shell
$ docker diff webserver
C /root
A /root/.bash_history
C /run
C /usr
C /usr/share
C /usr/share/nginx
C /usr/share/nginx/html
C /usr/share/nginx/html/index.html
C /var
C /var/cache
C /var/cache/nginx
A /var/cache/nginx/client_temp
A /var/cache/nginx/fastcgi_temp
A /var/cache/nginx/proxy_temp
A /var/cache/nginx/scgi_temp
A /var/cache/nginx/uwsgi_temp
```

现在我们定制好了变化，我们希望能将其保存下来形成镜像。Docker 提供了一个`docker commit`命令，可以将容器的存储层保存下来成为镜像。换句话说，就是在原有镜像的基础上，再叠加上容器的存储层，并构成新的镜像：

```shell
$ docker commit \
    --author "xyxxxxx" \
    --message "修改了默认首页" \
    webserver \
    nginx:v2
sha256:07e33465974800ce65751acc279adc6ed2dc5ed4e0838f8b86f0c87aa1795214
```



## Dockerfile定制镜像

Dockerfile 是一个文本文件，其内包含了一条条的指令(Instruction)，每一条指令构建一层。

还以之前定制 nginx 镜像为例，在一个空白目录中创建一个Dockerfile文件：

```dockerfile
FROM nginx
RUN echo '<h1>Hello, Docker!</h1>' > /usr/share/nginx/html/index.html
```

### FROM

`FROM`指定基础镜像，我们在这个镜像的基础上进行修改。在[Docker Store](https://store.docker.com/)上有非常多的高质量的官方镜像，有可以直接拿来使用的服务类的镜像，如 nginx、redis、mongo、mysql、httpd、php、tomcat 等；也有一些方便开发、构建、运行各种语言应用的镜像，如 node、openjdk、python、ruby、golang 等。可以在其中寻找一个最符合我们最终目标的镜像为基础镜像进行定制。

除了选择现有镜像为基础镜像外，Docker 还存在一个特殊的镜像，名为`scratch`。这个镜像是虚拟的概念，并不实际存在，它表示一个空白的镜像。直接`FROM scratch`会让镜像体积更加小巧。使用 Go 语言 开发的应用很多会使用这种方式来制作镜像，这也是为什么有人认为 Go 是特别适合容器微服务架构的语言的原因之一。

### RUN

`RUN`指令用来执行命令行命令。由于命令行的强大能力，`RUN`指令在定制镜像时是最常用的指令之一。

```dockerfile
FROM debian:jessie
RUN apt-get update
RUN apt-get install -y gcc libc6-dev make
RUN wget -O redis.tar.gz "http://download.redis.io/releases/redis-3.2.5.tar.gz"
RUN mkdir -p /usr/src/redis
RUN tar -xzf redis.tar.gz -C /usr/src/redis --strip-components=1
RUN make -C /usr/src/redis
RUN make -C /usr/src/redis install
```

Dockerfile 中每一个指令都会建立一层，上面的这种写法创建了 7 层镜像，这是完全没有意义的，而且很多运行时不需要的东西都被装进了镜像里，比如编译环境、更新的软件包等等。结果就是产生非常臃肿、非常多层的镜像，不仅仅增加了构建部署的时间，也很容易出错。 

```dockerfile
FROM debian:jessie
RUN buildDeps='gcc libc6-dev make' \
    && apt-get update \
    && apt-get install -y $buildDeps \
    && wget -O redis.tar.gz "http://download.redis.io/releases/redis-3.2.5.tar.gz" \
    && mkdir -p /usr/src/redis \
    && tar -xzf redis.tar.gz -C /usr/src/redis --strip-components=1 \
    && make -C /usr/src/redis \
    && make -C /usr/src/redis install \
    && rm -rf /var/lib/apt/lists/* \
    && rm redis.tar.gz \
    && rm -r /usr/src/redis \
    && apt-get purge -y --auto-remove $buildDeps
```

既然所有命令的目的就是编译、安装redis可执行文件，那么就只需要建立1层。这一组命令的最后还添加了清理工作的命令，删除了为了编译构建所需要的软件，清理了所有下载、展开的文件，并且还清理了 apt 缓存文件。这是很重要的一步，在镜像构建时，一定要确保每一层只添加真正需要添加的东西，任何无关的东西都应该清理掉。

### 构建

在Dockerfile文件所在目录执行：

```shell
$ docker build -t nginx:v3 .
Sending build context to Docker daemon 2.048 kB # 发送构建上下文
Step 1 : FROM nginx
 ---> e43d811ce2f4
Step 2 : RUN echo '<h1>Hello, Docker!</h1>' > /usr/share/nginx/html/index.html
 ---> Running in 9cdc27646c7b
 ---> 44aa4490ce2c
Removing intermediate container 9cdc27646c7b
Successfully built 44aa4490ce2c
```

从命令的输出结果中，我们可以清晰的看到镜像的构建过程。

### 构建上下文

我们注意到 docker build 命令最后有一个`.`，这个参数表示构建镜像的上下文路径。docker build 命令得知这个路径后，会将路径下的所有内容打包，然后上传给 Docker 引擎（由服务器完成构建）。这样 Docker 引擎收到这个上下文包后，展开就会获得构建镜像所需的一切文件。例如在 Dockerfile 中这么写：

```dockerfile
COPY ./package.json /app/
```

那么其含义就是 Docker 引擎将 docker build 命令传递过来的上下文路径下的所有内容的根路径下的`package.json`文件复制到镜像的`/app/`路径下。

对于 Dockerfile 文件，习惯的做法是使用默认的文件名 Dockerfile，以及将其置于镜像构建上下文目录中。





# Docker Compose

`Docker Compose`是`Docker`官方编排（Orchestration）项目之一，负责快速的部署分布式应用。其代码目前在https://github.com/docker/compose上开源。Compose 定位是“定义和运行多个 Docker 容器的应用（Defining and running multi-container Docker applications）”。

我们已经学习过使用一个`Dockerfile`模板文件，可以很方便的定义一个单独的应用容器。然而，在日常工作中，经常会碰到需要多个容器相互配合来完成某项任务的情况。例如要实现一个 Web 项目，除了 Web 服务容器本身，往往还需要再加上后端的数据库服务容器或者缓存服务容器，甚至还包括负载均衡容器等。Compose 恰好满足了这样的需求。它允许用户通过一个单独的 `docker-compose.yml`模板文件（YAML 格式）来定义一组相关联的应用容器为一个项目（project）。

Compose 中有两个重要的概念：

+ 服务 (service)：一个应用的容器，实际上可以包括若干运行相同镜像的容器实例。
+ 项目 (project)：由一组关联的应用容器组成的一个完整业务单元，在 docker-compose.yml 文件中定义。

Compose 的默认管理对象是项目，通过子命令对项目中的一组容器进行便捷地生命周期管理。



编写如下`docker-compose.yml`文件：

```yaml
mysql:
    image: mysql:5.7
    environment:
     - MYSQL_ROOT_PASSWORD=123456
     - MYSQL_DATABASE=wordpress
web:
    image: wordpress
    links:
     - mysql
    environment:
     - WORDPRESS_DB_PASSWORD=123456
    ports:
     - "127.0.0.3:8080:80"
    working_dir: /var/www/html
    volumes:
     - wordpress:/var/www/html
```

运行 compose 项目：

```shell
$ docker-compose up
```

此时就可以访问`127.0.0.3:8080`。



