## apt

**apt(Advancd Packaging Tool)**是Linux系统下的一款安装包管理工具，用于更新软件包列表索引、执行安装新软件包、升级现有软件包，还能够升级整个 Ubuntu 系统

```shell
# 安装包
$ sudo apt install nginx

# 安装本地包
$ sudo apt install name.deb

# 显示包
$ apt list --upgradeable  # 可以更新的包
$ apt list --installed    # 已安装的包
$ apt list docker         # 指定的包

# 更新包
$ sudo apt update       # 检查所有包的更新
$ sudo apt upgrade      # 更新所有包

# 删除包
$ sudo apt remove nmap  # 保留配置文件
$ sudo apt purge nmap   # 删除配置文件

# 搜索包
$ apt search docker     # 显示相关的包

```

> `apt`命令可以简单地视作`apt-get`, `apt-cache`和`apt-config`命令的集合。



## 源码安装



