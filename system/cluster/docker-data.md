> 参考[Docker —— 从入门到实践](https://yeasy.gitbook.io/docker_practice/)

# 数据卷

`数据卷` 是一个可供一个或多个容器使用的特殊目录，它绕过 UFS，可以提供很多有用的特性：

+ `数据卷` 可以在容器之间共享和重用
+ 对 `数据卷` 的修改会立马生效
+ 对 `数据卷` 的更新，不会影响镜像
+ `数据卷` 默认会一直存在，即使容器被删除



## 创建数据卷

```shell
$ docker volume create my-vol
```

查看所有的数据卷

```shell
$ docker volume ls
DRIVER              VOLUME NAME
local               my-vol
```

在主机里使用以下命令可以查看指定数据卷的信息

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



## 启动一个挂载数据卷的容器

```shell
$ docker run -d -P \
    --name web \
    # 挂载数据卷 my-vol 到容器的 /usr/share/nginx/html 目录
    --mount source=my-vol,target=/usr/share/nginx/html \
    nginx:alpine
```

使用以下命令可以查看 `web` 容器的信息，其中包括挂载信息

```shell
$ docker inspect web
# ...
"Mounts": [  # mount info
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



## 删除数据卷

```shell
$ docker volume rm my-vol
```

数据卷被设计用于持久化数据，它的生命周期独立于容器，Docker 不会在容器被删除后自动删除数据卷，并且也不存在垃圾回收这样的机制来处理没有任何容器引用的数据卷。如果需要在删除容器的同时移除数据卷，可以使用 `docker rm -v` 命令。删除无主的数据卷可以使用命令

```shell
$ docker volume prune
```





# 挂载主机目录

`--mount`选项同样可以挂载一个本地主机的目录到容器中

```shell
$ docker run -d -P \
    --name web \
    # -v /src/webapp:/usr/share/nginx/html \
    --mount type=bind,source=/src/webapp,target=/usr/share/nginx/html \
    nginx:alpine
```

上面的命令加载主机的 `/src/webapp` 目录到容器的 `/usr/share/nginx/html`目录。这个功能在进行测试的时候十分方便，比如用户可以放置一些程序到本地目录中，来查看容器是否正常工作。本地目录的路径必须是绝对路径。

Docker 挂载主机目录的默认权限是 `读写`，用户也可以通过增加 `readonly` 指定为 `只读`。

```shell
$ docker run -d -P \
    --name web \
    # -v /src/webapp:/usr/share/nginx/html:ro \
    --mount type=bind,source=/src/webapp,target=/usr/share/nginx/html,readonly \
    nginx:alpine
```

