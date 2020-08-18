

```shell
# 初始化新模块
$ go mod init <module-name>

# 查看当前模块的所有依赖
$ go list -m all

# 获取依赖
$ go get rsc.io/sampler

# 查看指定依赖的所有版本
$ go list -m -versions rsc.io/sampler

# 获取依赖的指定版本
$ go get rsc.io/sampler@v1.3.1
```

Go模块的每个不同的大版本都使用不同的路径：例如`rsc.io/quote`的`v3`版本使用路径`rsc.io/quote/v3`，这样做的目的是让不兼容的模块（不同大版本的模块）有不同的路径。同一路径下的新版本模块应该向后兼容老版本模块。

Go的构建允许每个不同路径下有至多一个模块，因此项目可以同时依赖不同大版本的模块