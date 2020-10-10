> 参考[Docker —— 从入门到实践](https://yeasy.gitbook.io/docker_practice/)

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

