> 参考[Using Go Modules](https://blog.golang.org/using-go-modules)

Go 模块是 Go 包的集合，由根目录下的 `go.mod` 文件定义模块的路径以及模块的依赖，满足了依赖要求的模块才能被成功构建。每个依赖需要写为一个模块路径和相应的版本。

下面展示了一个 `go.mod` 文件示例：

```go
module github.com/tektoncd/dashboard

go 1.13

// Pin k8s deps to 1.16.5
replace (
	k8s.io/api => k8s.io/api v0.16.5
	k8s.io/apimachinery => k8s.io/apimachinery v0.16.5
	k8s.io/client-go => k8s.io/client-go v0.16.5
	k8s.io/code-generator => k8s.io/code-generator v0.16.5
	k8s.io/gengo => k8s.io/gengo v0.0.0-20190327210449-e17681d19d3a
)

require (
	github.com/emicklei/go-restful v2.12.0+incompatible
	github.com/google/go-cmp v0.5.0
	github.com/gorilla/csrf v1.7.0
	github.com/gorilla/websocket v1.4.2
	github.com/openshift/api v3.9.0+incompatible // indirect
	github.com/openshift/client-go v0.0.0-20191125132246-f6563a70e19a
	github.com/tektoncd/pipeline v0.15.2
	github.com/tektoncd/plumbing v0.0.0-20200430135134-e53521e1d887
	github.com/tektoncd/triggers v0.6.1
	go.uber.org/zap v1.15.0
	k8s.io/api v0.18.2
	k8s.io/apimachinery v0.18.2
	k8s.io/client-go v11.0.1-0.20190805182717-6502b5e7b1b5+incompatible
	k8s.io/code-generator v0.18.0
	knative.dev/pkg v0.0.0-20200702222342-ea4d6e985ba0
)

```

## 创建一个新模块

在一个新的空白目录（假设为 `/home/gopher/hello`）下创建一个新的源文件 `hello.go`：

```go
package hello

func Hello() string {
    return "Hello, world."
}
```

此时该目录包含一个包。再使用 `go mod init` 命令：

```shell
# 初始化新模块
$ go mod init example.com/hello
#             <path/name>
```

就创建了一个新模块。

`go mod init`命令将当前目录设为一个模块的根目录，并创建`go.mod` 文件。位于当前目录子目录中的包自动被识别为模块的一部分。

## 添加模块依赖

Go 模块的主要功能是改善依赖管理的体验。

编译 Go 模块时，`go` 命令首先使用 `go.mod` 中给出的依赖模块版本来解析导入；当遇到未由 `go.mod` 中的任何依赖提供的导入时，`go` 命令将在线查找该模块，使用其最新的稳定版本，将依赖下载到 `$GOPATH/pkg/mod` 并将依赖项添加到 `go.mod` 中。

现在我们更新 `hello.go` 以导入模块：

```go
package hello

import "rsc.io/quote"

func Hello() string {
    return quote.Hello()
}
```

运行该程序，再查看 `go.mod` 文件：

```shell
$ cat go.mod
module example.com/hello

go 1.12

require rsc.io/quote v1.5.2
```

只有直接依赖项会被记录在 `go.mod` 文件中，以下命令用于查看所有依赖项：

```shell
# 查看当前模块的所有依赖
$ go list -m all
example.com/hello # 当前模块总是出现在第一行
golang.org/x/text v0.0.0-20170915032832-14c0d48ead0c
rsc.io/quote v1.5.2
rsc.io/sampler v1.3.0
```

除了 `go.mod` 之外，`go` 命令还会维护一个名为 `go.sum` 的文件，其中包含依赖模块版本的加密哈希值：

```
golang.org/x/text v0.0.0-20170915032832-14c0d48ead0c h1:qgOY6WgZO...
golang.org/x/text v0.0.0-20170915032832-14c0d48ead0c/go.mod h1:Nq...
rsc.io/quote v1.5.2 h1:w5fcysjrx7yqtD/aO+QwRjYZOKnaM9Uh2b40tElTs3...
rsc.io/quote v1.5.2/go.mod h1:LzX7hefJvL54yjefDEDHNONDjII0t9xZLPX...
rsc.io/sampler v1.3.0 h1:7uVkIFmeBqHfdjD+gZwtXXI+RODJ2Wc4O7MPEh/Q...
rsc.io/sampler v1.3.0/go.mod h1:T1hPZKmBbMNahiBKFy5HrXp6adAjACjK9...
```

`go.sum` 文件用于确保这些模块的将来的下载与第一次下载相同，以确保项目所依赖的模块不会由于恶意，意外或其他原因而意外更改。`go.sum` 是一个构建状态跟踪文件，它记录当前模块所有的直接和间接依赖，以及这些依赖的校验和，从而提供一个可以 100% 复现的构建过程并对构建对象提供安全性的保证。所以**应该将 `go.mod` 和 `go.sum` 都添加到版本控制中**。

## 更新依赖

Go 模块的版本包括三部分：主要、次要和补丁，例如 `v0.1.2` 的主要版本为 0，次要版本为 1，补丁版本为 2。

从之前的输出可以看到依赖 `golang.org/x/text` 使用了未标记版本（`v0.0.0-20170915...`），现在我们将其升级到最新的标记版本：

```shell
$ go get golang.org/x/text
# 默认为                   @latest    
go: finding golang.org/x/text v0.3.0
go: downloading golang.org/x/text v0.3.0
go: extracting golang.org/x/text v0.3.0
```

运行程序，再查看 `go.mod` 文件和 `go list -m all` 的输出：

```shell
$ cat go.mod
module example.com/hello

go 1.12

require (
    golang.org/x/text v0.3.0 // indirect
    rsc.io/quote v1.5.2
)
$ go list -m all
example.com/hello
golang.org/x/text v0.3.0
rsc.io/quote v1.5.2
rsc.io/sampler v1.3.0
```

我们看到依赖项 `golang.org/x/text` 已经升级到 `v0.3.0` 版本。`indirect` 注释指明依赖项不被当前模块直接使用，而是由其依赖的模块所使用的。

再来尝试升级 `rsc.io/sampler`，首先列出它的可用版本：

```shell
$ go list -m -versions rsc.io/sampler
rsc.io/sampler v1.0.0 v1.2.0 v1.2.1 v1.3.0 v1.3.1 v1.99.99
```

我们将其升级到指定版本 `v1.3.1`

```shell
$ go get rsc.io/sampler@v1.3.1
```

## 增加依赖的主版本

我们再次更新 `hello.go`：

```go
package hello

import (
    "rsc.io/quote"
    quoteV3 "rsc.io/quote/v3" // assign a new name
)

func Hello() string {
    return quote.Hello()
}

func Proverb() string {
    return quoteV3.Concurrency()
}
```

现在模块同时依赖了 `rsc.io/quote` 和 `rsc.io/quote/v3`，即同一模块的两个不同的大版本。

Go 模块的每个不同的大版本都使用不同的路径：例如 `rsc.io/quote` 的 `v3` 版本使用路径 `rsc.io/quote/v3`，这样做的目的是让不兼容的模块（不同大版本的模块）有不同的路径。同一路径下的新版本模块应该向后兼容老版本模块。

Go 的构建允许每个不同路径下有至多一个模块，因此项目可以同时依赖不同大版本的模块。

将旧版本模块的 API 调用替换为新版本模块的 API 调用，即完成了向新版本模块的迁移：

```go
package hello

import "rsc.io/quote/v3"

func Hello() string {
    return quote.HelloV3()
}

func Proverb() string {
    return quote.Concurrency()
}
```

## 删除未使用的依赖项

现在我们已经删除了对 `rsc.io/quote` 的所有使用，但由于 `go build`命令无法确定每个依赖是否可以安全删除，它仍显示在`go list -m all`的输出和`go.mod`文件中。`go mod tidy` 命令用于清除这些未使用的依赖项：

```shell
$ go mod tidy
$ go list -m all
example.com/hello
golang.org/x/text v0.3.0
rsc.io/quote/v3 v3.1.0
rsc.io/sampler v1.3.1
$ cat go.mod
module example.com/hello

go 1.12

require (
    golang.org/x/text v0.3.0 // indirect
    rsc.io/quote/v3 v3.1.0
    rsc.io/sampler v1.3.1 // indirect
)
```

