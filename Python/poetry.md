Poetry 和 Pipenv 类似，是一个 Python 虚拟环境和依赖管理工具，它还提供了包管理功能，比如打包和发布。你可以把它看做是 Pipenv 和 Flit 这些工具的超集。它可以让你用 Poetry 来同时管理 Python 库和 Python 程序。

主页：https://poetry.eustace.io/

源码：https://github.com/sdispater/poetry

文档：https://poetry.eustace.io/docs



## 安装



## 使用

```shell
$ poetry init                 # 在已有项目中创建一个pyproject.toml文件

$ poetry install              # 创建虚拟环境 (需要pyproject.toml文件)
$ poetry shell                # 进入虚拟环境

$ poetry add <dependency>     # 增加一个依赖
$ poetry remove <dependency>  # 卸载删除一个依赖

$ poetry update               # 更新所有锁定版本的依赖
$ poetry update <dependency>  # 更新某个锁定版本的依赖
```



## tips

国内用户在安装虚拟环境时最好使用国内的PyPI镜像，方法是在`pyproject.toml`中加入

```toml
# 阿里云
[[tool.poetry.source]]
name = "aliyun"
url = "https://mirrors.aliyun.com/pypi/simple/"
default = true

# 豆瓣
[[tool.poetry.source]]
name = "douban"
url = "https://pypi.doubanio.com/simple/"
default = true
```

如果已经使用国内PyPI镜像，但`poetry install`的解析依赖依然很慢，可以使用`poetry install -vvv`查看解析依赖进度。