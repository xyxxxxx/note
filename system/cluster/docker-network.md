> 参考[Docker —— 从入门到实践](https://yeasy.gitbook.io/docker_practice/)

# 外部访问容器

容器中可以运行一些网络应用，要让外部也可以访问这些应用，可以通过 `-P` 或 `-p` 参数来指定端口映射。

当使用 `-P` 时，Docker 会随机映射一个端口到内部容器开放的网络端口。下面这个例子中，本地主机的 32768 被映射到容器的 80 端口，访问本机的 32768 端口即可访问容器内 nginx 的默认页面。

```shell
$ docker run -d -P nginx:alpine

$ docker container ls -l
CONTAINER ID        IMAGE               COMMAND                  CREATED             STATUS              PORTS                   NAMES
fae320d08268        nginx:alpine        "/docker-entrypoint.…"   24 seconds ago      Up 20 seconds       0.0.0.0:32768->80/tcp   bold_mcnulty
```

`-p` 则可以指定要映射的端口，一个指定端口上只可以绑定一个容器。支持的格式有 `ip:hostPort:containerPort | ip::containerPort | hostPort:containerPort`。

```shell
$ docker run -d -p 80:80 nginx:alpine # 映射容器的80端口到本地主机的80端口, 映射所有接口地址
$ docker run -d -p 127.0.0.1:80:80 nginx:alpine # 映射一个特定地址
$ docker run -d -p 127.0.0.1::80 nginx:alpine # 本地主机分配一个端口
$ docker run -d -p 127.0.0.1::80/udp nginx:alpine # 指定UDP端口
$ docker run -d \
    -p 80:80 \
    -p 443:443 \ # 绑定多个端口
    nginx:alpine



```





# 容器互联

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
/ # ping busybox1
PING busybox1 (172.19.0.2): 56 data bytes
64 bytes from 172.19.0.2: seq=0 ttl=64 time=0.064 ms
64 bytes from 172.19.0.2: seq=1 ttl=64 time=0.143 ms
```

这样`busybox1` 容器和 `busybox2` 容器建立了互联关系。





# 配置DNS