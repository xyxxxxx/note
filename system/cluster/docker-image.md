> 参考[Docker —— 从入门到实践](https://yeasy.gitbook.io/docker_practice/)

> 如果出现权限问题：
>
> ```
> Got permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock: Get http://%2Fvar%2Frun%2Fdocker.sock/v1.40/version: dial unix /var/run/docker.sock: connect: permission denied
> ```
>
> 使用`sudo docker`命令

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

以定制一个 Web 服务器为例：

```shell
$ docker run --name webserver -d -p 8081:80 nginx
```

这条命令会用 nginx 镜像启动一个容器，命名为 webserver，并且映射了 80 端口，这样我们可以用浏览器去访问这个 nginx 服务器。

![](https://gblobscdn.gitbook.com/assets%2F-M5xTVjmK7ax94c8ZQcm%2F-M5xT_hHX2g5ldlyp9nm%2F-M5xTq0Ue52VtZnqYRkR%2Fimages-mac-example-nginx.png?alt=media)

假设我们非常不喜欢这个欢迎页面，我们希望改成欢迎 Docker 的文字，我们可以使用 `docker exec` 命令进入容器，修改其内容。

```shell
$ docker exec -it webserver bash
root@3729b97e8226:/echo '<h1>Hello, Docker!</h1>' > /usr/share/nginx/html/index.html
root@3729b97e8226:/exit
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

现在我们定制好了变化，接下来希望将其保存下来形成镜像。

需要注意的是，当我们运行一个容器而不使用卷时，任何文件修改都被记录于容器的存储层中。Docker 提供了一个`docker commit`命令，可以将容器的存储层保存下来成为镜像。换句话说，就是在原有镜像的基础上叠加容器的存储层，构成新的镜像：

```shell
$ docker commit \
    --author "xyxxxxx" \
    --message "修改了默认首页" \
    webserver \
    nginx:v2
sha256:07e33465974800ce65751acc279adc6ed2dc5ed4e0838f8b86f0c87aa1795214
```

> docker的`diff`,`commit`,`history`命令类似于git

运行这个新的镜像，再次访问这个 nginx 服务器，我们发现主页已经是修改后的版本。

上面这个例子帮助了我们理解镜像的分层存储概念（就如同git的版本控制），但实际环境中并不会使用`docker commit`定制镜像。





# `Dockerfile`定制镜像

Dockerfile 是一个脚本文件，其中包含了一条条的指令(Instruction)，每一条指令构建一层，即每一条指令的内容描述了该层应当如何构建。

还是以定制 nginx 镜像为例，创建 Dockerfile 文件：

```dockerfile
FROM nginx
RUN echo '<h1>Hello, Docker!</h1>' > /usr/share/nginx/html/index.html
```

### FROM

`FROM`指定基础镜像，我们在这个镜像的基础上进行修改。在[Docker Hub](https://hub.docker.com/)上有非常多的高质量的官方镜像，有可以直接拿来使用的服务类的镜像，如 [`nginx`](https://hub.docker.com/_/nginx/)、[`redis`](https://hub.docker.com/_/redis/)、[`mongo`](https://hub.docker.com/_/mongo/)、[`mysql`](https://hub.docker.com/_/mysql/)、[`httpd`](https://hub.docker.com/_/httpd/)、[`php`](https://hub.docker.com/_/php/)、[`tomcat`](https://hub.docker.com/_/tomcat/) 等；也有一些方便开发、构建、运行各种语言应用的镜像，如 [`node`](https://hub.docker.com/_/node)、[`openjdk`](https://hub.docker.com/_/openjdk/)、[`python`](https://hub.docker.com/_/python/)、[`ruby`](https://hub.docker.com/_/ruby/)、[`golang`](https://hub.docker.com/_/golang/) 等。可以在其中寻找一个最符合我们最终目标的镜像为基础镜像进行定制。

如果没有找到对应服务的镜像，官方镜像中还提供了一些更为基础的操作系统镜像，如 [`ubuntu`](https://hub.docker.com/_/ubuntu/)、[`debian`](https://hub.docker.com/_/debian/)、[`centos`](https://hub.docker.com/_/centos/)、[`fedora`](https://hub.docker.com/_/fedora/)、[`alpine`](https://hub.docker.com/_/alpine/) 等，这些操作系统的软件库为我们提供了更广阔的扩展空间。

除了选择现有镜像为基础镜像外，Docker 还存在一个特殊的镜像，名为`scratch`。这个镜像是虚拟的概念，并不实际存在，它表示一个空白的镜像。对于 Linux 下静态编译的程序来说，并不需要有操作系统提供运行时支持，所需的一切库都已经在可执行文件里了，因此直接 `FROM scratch` 会让镜像体积更加小巧。使用 Go 语言开发的应用很多会使用这种方式来制作镜像，这也是为什么有人认为 Go 是特别适合容器微服务架构的语言的原因之一。

### RUN

`RUN`指令用来执行命令行命令。由于命令行的强大能力，`RUN`指令在定制镜像时是最常用的指令之一。其格式有两种：

+ *shell* 格式：`RUN <命令>`，就像直接在命令行中输入的命令一样。

+ *exec* 格式：`RUN <"可执行文件", "参数1", "参数2">`，这更像是函数调用中的格式

```dockerfile
FROM debian:stretch
RUN apt-get update
RUN apt-get install -y gcc libc6-dev make
RUN wget -O redis.tar.gz "http://download.redis.io/releases/redis-3.2.5.tar.gz"
RUN mkdir -p /usr/src/redis
RUN tar -xzf redis.tar.gz -C /usr/src/redis --strip-components=1
RUN make -C /usr/src/redis
RUN make -C /usr/src/redis install
```

Dockerfile 中每一个指令都会建立一层，上面的例子中创建了 7 层镜像，这是完全没有意义的，而且很多运行时不需要的东西都被装进了镜像里，比如编译环境、更新的软件包等等。结果就是产生非常臃肿、非常多层的镜像，不仅仅增加了构建部署的时间，也很容易出错。 因此正确写法应该是：

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

既然所有命令的目的就是编译、安装 redis 可执行文件，那么就只需要建立1层。这一组命令的最后还添加了清理工作的命令，删除了为了编译构建所需要的软件，清理了所有下载、展开的文件，并且还清理了 `apt` 缓存文件。这是很重要的一步，在镜像构建时，一定要确保每一层只添加真正需要添加的东西，任何无关的东西都应该清理掉。

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

从命令的输出结果中，我们可以清晰地看到镜像的构建过程。在 `Step 2` 中，如同我们之前所说的那样，`RUN` 指令启动了一个容器 `9cdc27646c7b`，执行了所要求的命令，并最后提交了这一层 `44aa4490ce2c`，随后删除了所用到的这个容器 `9cdc27646c7b`。

### 构建上下文

我们注意到 `docker build` 命令最后有一个`.`，这个参数表示构建镜像的上下文路径。`docker build` 命令得知这个路径后，会将路径下的所有内容打包，然后上传给 Docker 引擎（由服务器完成构建）。这样 Docker 引擎收到这个上下文包后，展开就会获得构建镜像所需的一切文件。例如在 Dockerfile 中这么写：

```dockerfile
COPY ./package.json /app/
```

那么其含义就是 Docker 引擎将 `docker build` 命令传递过来的上下文路径下的所有内容的根路径下的`package.json`文件复制到镜像的`/app/`路径下。

一般来说，应该会将 `Dockerfile` 置于一个空目录下，或者项目根目录下。如果该目录下没有所需文件，那么应该把所需文件复制一份过来。如果目录下有些东西确实不希望构建时传给 Docker 引擎，那么可以用 `.gitignore` 一样的语法写一个 `.dockerignore`，该文件是用于剔除不需要作为上下文传递给 Docker 引擎的。

对于 Dockerfile 文件，习惯的做法是使用默认的文件名 `Dockerfile`，以及将其置于镜像构建上下文目录中。



## Dockerfile指令详解

> 参考[Dockerfile 指令详解](https://yeasy.gitbook.io/docker_practice/image/dockerfile)

### COPY

`COPY` 指令从构建上下文目录中 `<源路径>` 的文件或目录复制到新的一层的镜像内的 `<目标路径>` 位置。比如：

```dockerfile
COPY package.json /usr/src/app/
```



### ADD

`ADD` 指令在 `COPY` 基础上增加了一些功能。当 `<源路径>` 是一个 `tar` 压缩文件并且压缩格式为 `gzip`, `bzip2` 以及 `xz` 的情况下，`ADD` 指令将会自动解压缩这个压缩文件到 `<目标路径>` 去。

在 Docker 官方的 [Dockerfile 最佳实践文档]() 中要求，尽可能的使用 `COPY`，因为 `COPY` 的语义很明确，就是复制文件而已，而 `ADD` 则包含了更复杂的功能，其行为也不一定很清晰。最适合使用 `ADD` 的场合，就是所提及的需要自动解压缩的场合。



### CMD



### EXPOSE

格式为 `EXPOSE <端口1> [<端口2>...]`。

`EXPOSE` 指令是声明运行时容器提供服务端口，这只是一个声明，在运行时并不会因为这个声明应用就会开启这个端口的服务。在 Dockerfile 中写入这样的声明有两个好处，一个是帮助镜像使用者理解这个镜像服务的守护端口，以方便配置映射；另一个用处则是在运行时使用随机端口映射时，也就是 `docker run -P` 时，会自动随机映射 `EXPOSE` 的端口。

要将 `EXPOSE` 和在运行时使用 `-p <宿主端口>:<容器端口>` 区分开来。`-p`，是映射宿主端口和容器端口，换句话说，就是将容器的对应端口服务公开给外界访问，而 `EXPOSE` 仅仅是<u>声明</u>容器打算使用什么端口而已，并不会自动在宿主进行端口映射。