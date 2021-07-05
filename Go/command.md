## go build

```
$ go build       //编译
$ ls
gobuild  lib.go  main.go
$ ./gobuild      //执行可执行文件
call pkgFunc
hello world
```

```
-o 指定可执行文件的文件名

```



## go clean

```
$ go clean
```

```
-x 打印出来执行的详细命令
```



## go run

`go run` 命令会编译源码，并且直接执行源码的 main（）函数，不会在当前目录留下可执行文件。

```
$ go run .\main.go
```

