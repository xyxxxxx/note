> 参考[Docker —— 从入门到实践](https://yeasy.gitbook.io/docker_practice/)

`Docker Compose`是`Docker` 官方编排（Orchestration）项目之一，负责快速的部署分布式应用。其代码目前在 https://github.com/docker/compose 上开源。Compose 定位是“定义和运行多个 Docker 容器的应用（Defining and running multi-container Docker applications）”。

我们已经学习过使用一个 `Dockerfile` 模板文件，可以很方便的定义一个单独的应用容器。然而，在日常工作中，经常会碰到需要多个容器相互配合来完成某项任务的情况。例如要实现一个 Web 项目，除了 Web 服务容器本身，往往还需要再加上后端的数据库服务容器或者缓存服务容器，甚至还包括负载均衡容器等。Compose 恰好满足了这样的需求。它允许用户通过一个单独的 `docker-compose.yml` 模板文件（YAML 格式）来定义一组相关联的应用容器为一个项目（project）。

Compose 中有两个重要的概念：

* 服务（service）：一个应用的容器，实际上可以包括若干运行相同镜像的容器实例。
* 项目（project）：<u> 由一组关联的应用容器组成的一个完整业务单元 </u>，在 docker-compose.yml 文件中定义。

Compose 的默认管理对象是项目，通过子命令对项目中的一组容器进行便捷地生命周期管理。



# Get Started

















# 实战

## WordPress

编写如下 `docker-compose.yml` 文件：

```yaml
version: "3"
services:

   db:
     image: mysql:8.0
     command:
      - --default_authentication_plugin=mysql_native_password
      - --character-set-server=utf8mb4
      - --collation-server=utf8mb4_unicode_ci     
     volumes:
       - db_data:/var/lib/mysql
     restart: always
     environment:
       MYSQL_ROOT_PASSWORD: somewordpress
       MYSQL_DATABASE: wordpress
       MYSQL_USER: wordpress
       MYSQL_PASSWORD: wordpress

   wordpress:
     depends_on:
       - db
     image: wordpress:latest
     ports:
       - "8000:80"
     restart: always
     environment:
       WORDPRESS_DB_HOST: db:3306
       WORDPRESS_DB_USER: wordpress
       WORDPRESS_DB_PASSWORD: wordpress
volumes:
  db_data:
```

运行 compose 项目：

```shell
$ docker-compose up
```

compose 就会拉取镜像再创建我们所需要的镜像，然后启动 `wordpress` 和数据库容器。接着浏览器访问 `127.0.0.1:8000` 端口就能看到 `WordPress` 安装界面了。

