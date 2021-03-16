```shell
$ pip install package            # 安装包(的最新版本)
$ pip install package=1.0.4      # 安装包的指定版本

$ pip install --upgrade package  # 升级包到最新版本

$ pip install -r requirements.txt   # 从requirements文件中安装

$ pip install package -i <url>   # 指定pypi镜像地址,默认为https://pypi.org/simple
                                 # 豆瓣 https://pypi.douban.com/simple/
                                 # 阿里云 https://mirrors.aliyun.com/pypi/simple/
                                 # 清华 https://pypi.tuna.tsinghua.edu.cn/simple
                                 
$ pip install -e .                 # 以"可编辑"模式从VCS安装项目;安装当前目录下的项目
$ pip install -e path/to/project   # 安装指定目录下的项目
```



`requirements.txt`是

