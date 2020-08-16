# 读取用户输入





# 文件读写

```go
package main

import (
	"fmt"
	"io"
	"strings"
)

func main() {
	r := strings.NewReader("Hello, Reader!") //为字符串创建一个Reader

	b := make([]byte, 8)
	for {
		n, err := r.Read(b)                  //每次读取填充字节切片的量
		fmt.Printf("n = %v err = %v b = %v\n", n, err, b)
		fmt.Printf("b[:n] = %q\n", b[:n])
		if err == io.EOF {                   //读取完毕之后返回错误EOF
			break
		}
	}
}

/*
n = 8 err = <nil> b = [72 101 108 108 111 44 32 82]
b[:n] = "Hello, R"
n = 6 err = <nil> b = [101 97 100 101 114 33 32 82]
b[:n] = "eader!"
n = 0 err = EOF b = [101 97 100 101 114 33 32 82]
b[:n] = ""
*/
```





# 文件拷贝





# 从命令行读取参数





# 用buffer读取文件





# 用切片读写文件




# 用defer关闭文件





# JSON





# XML



