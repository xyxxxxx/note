# 文件读写

## 读文件

 Go 语言中，文件使用指向 `os.File` 类型的指针来表示，也叫做文件句柄。标准输入 `os.Stdin` 和标准输出 `os.Stdout`的类型都是 `*os.File`。

`io.Reader`接口用于包装基本的读取方法：

```go
type Reader interface {
    Read(p []byte) (n int, err error)
}
```

`Read`方法读取`len(p)`字节数据写入`p`，返回写入的字节数和遇到的错误。如果有部分可用数据，但不够len(p)字节，Read按惯例会返回可以读取到的数据，而不是等待更多数据。

当`Read`在读取n > 0个字节后遭遇错误或者到达文件结尾时，会返回读取的字节数。它可能会在该次调用返回一个非nil的错误，或者在下一次调用时返回0和该错误。一个常见的例子，Reader接口会在输入流的结尾返回非0的字节数，返回值err == EOF或err == nil，但不管怎样，下一次Read调用必然返回(0, EOF)。

`bufio.Reader`结构给一个`io.Reader`接口对象附加缓冲。



带缓冲区的读文件（读取到指定字符）：

```go
package main

import (
    "bufio"
    "fmt"
    "io"
    "os"
)

func main() {
    //打开文件，返回一个*os.File结构的对象(实现了io.Read和io.Write方法)
    inputFile, inputError := os.Open("input.dat")
    if inputError != nil {
        fmt.Printf("An error occurred on opening the inputfile\n" +
            "Does the file exist?\n" +
            "Have you got access to it?\n")
        return // exit the function on error
    }
    //使用defer关闭文件
    defer inputFile.Close()

    //传入io.Reader接口对象，返回bufio.Reader结构对象的指针
    inputReader := bufio.NewReader(inputFile)
    for {
        //读取直到第一次遇到delim字节，这里即逐行读取
        inputString, readerError := inputReader.ReadString('\n')
        fmt.Printf("The input was: %s", inputString)
        //读到文件末尾时，返回在错误之前读取的数据以及io.EOF
        //io.EOF的定义为 var EOF = errors.New("EOF")
        if readerError == io.EOF {
            return
        }      
    }
}
```

`bufio`包中的`Reader.ReadString`和`Reader.ReadBytes`用于读取直到指定字符的内容。



带缓冲区的读文件（读取到缓冲区）：

```go
buf := make([]byte, 1024)
...
for {
    //读取数据写入buf,返回写入的字节数
	n, err := inputReader.Read(buf)
    //读取到达结尾时返回 0, io.EOF
    if (n == 0) { break}
}
```



将整个文件的内容读到一个字符串中：

```go
package main
import (
    "fmt"
    "io/ioutil"
    "os"
)

func main() {
    inputFile := "products.txt"
    outputFile := "products_copy.txt"
    //读取数据直到EOF或遇到error,返回读取的数据([]byte类型)，成功的调用返回的err为nil
    buf, err := ioutil.ReadFile(inputFile)
    if err != nil {
        fmt.Fprintf(os.Stderr, "File Error: %s\n", err)
        // panic(err.Error())
    }
    fmt.Printf("%s\n", string(buf))
    err = ioutil.WriteFile(outputFile, buf, 0644) // oct, not hex
    if err != nil {
        panic(err.Error())
    }
}
```



按列读取文件：

```go
package main
import (
    "fmt"
    "os"
)

func main() {
    file, err := os.Open("products2.txt")
    if err != nil {
        panic(err)
    }
    defer file.Close()

    var col1, col2, col3 []string
    for {
        var v1, v2, v3 string
        _, err := fmt.Fscanln(file, &v1, &v2, &v3)
        // scans until newline
        if err != nil {
            break
        }
        col1 = append(col1, v1)
        col2 = append(col2, v2)
        col3 = append(col3, v3)
    }

    fmt.Println(col1)
    fmt.Println(col2)
    fmt.Println(col3)
}
```



为字符串创建Reader：

```go
package main

import (
	"fmt"
	"io"
	"strings"
)

func main() {
    //返回一个strings.Reader结构对象，实现了io.Reader等接口
	r := strings.NewReader("Hello, Reader!")

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



## 写文件

`io.Writer`接口用于包装基本的写入方法：

```go
type Writer interface {
	Write(p []byte) (n int, err error)
}
```

`Write`方法将`len(p)`字节数据从`p`写入底层的数据流。它会返回写入的字节数和遇到的任何导致写入提取结束的错误。



使用缓冲区的写入：

```go
package main

import (
	"os"
	"bufio"
	"fmt"
)

func main () {
	// var outputWriter *bufio.Writer
	// var outputFile *os.File
	// var outputError os.Error
	// var outputString string
	outputFile, outputError := os.OpenFile("output.dat", os.O_WRONLY|os.O_CREATE, 0666)
	if outputError != nil {
		fmt.Printf("An error occurred with file opening or creation\n")
		return  
	}
	defer outputFile.Close()

    //传入io.Writer接口对象，返回bufio.Writer结构对象的指针
	outputWriter := bufio.NewWriter(outputFile)
	outputString := "hello world!\n"

	for i:=0; i<10; i++ {
        //写入一个字符串，返回写入的字节数
		outputWriter.WriteString(outputString)
	}
	outputWriter.Flush()
}
```

`OpenFile` 函数是比`Open`更一般的文件打开函数，有三个参数：文件名、一个或多个标志（使用逻辑运算符“|”连接），使用的文件权限。

我们通常会用到以下标志：

- `os.O_RDONLY`：只读
- `os.O_WRONLY`：只写
- `os.O_CREATE`：创建：如果指定文件不存在，就创建该文件。
- `os.O_TRUNC`：截断：如果指定文件已存在，就舍弃该文件的原有内容。

不管是Unix还是Windows，文件权限都需要使用0666。



不使用缓冲区的写入：

```go
package main

import "os"

func main() {
	os.Stdout.WriteString("hello, world\n")
	f, _ := os.OpenFile("test", os.O_CREATE|os.O_WRONLY, 0666)
	defer f.Close()
	f.WriteString("hello, world in a file\n")
}
```



最简单的写入：

```go
fmt.Fprintf(outputFile, "Some test data.\n")
```





# 文件拷贝

```go
package main

import (
	"fmt"
	"io"
	"os"
)

func main() {
	CopyFile("target.txt", "source.txt")
	fmt.Println("Copy done!")
}

func CopyFile(dstName, srcName string) (written int64, err error) {
	src, err := os.Open(srcName)
	if err != nil {
		return
	}
	defer src.Close()

	dst, err := os.Create(dstName)
	if err != nil {
		return
	}
	defer dst.Close()

	return io.Copy(dst, src)
}
```





# 读取用户输入

从标准输入`os.Stdin`（键盘）读取输入最简单的方法是使用`fmt`包提供的 Scan 和 Sscan 开头的函数。

```go
package main
import "fmt"

var (
   firstName, lastName, s string
   i int
   f float32
   input = "56.12 / 5212 / Go"
   format = "%f / %d / %s"
)

func main() {
   fmt.Println("Please enter your full name: ")
   fmt.Scanln(&firstName, &lastName)
   // fmt.Scanf("%s %s", &firstName, &lastName)
   fmt.Printf("Hi %s %s!\n", firstName, lastName) // Hi Chris Naegels
   fmt.Sscanf(input, format, &f, &i, &s)
   fmt.Println("From the string we read: ", f, i, s)
    // 输出结果: From the string we read: 56.12 5212 Go
}
```

`Scanln` 扫描来自标准输入的文本，将空格分隔的值依次存放到后续的参数内，直到碰到换行。`Scanf` 与其类似，除了 `Scanf` 的第一个参数用作格式字符串，用来决定如何读取。`Sscan` 和以 `Sscan` 开头的函数则是从字符串读取，除此之外，与 `Scanf` 相同。

这些函数返回成功读入数据的个数和错误，可以用于检查。



也可以使用`bufio`包提供的缓冲读取来读取数据，示例如下：

```go
package main
import (
    "fmt"
    "bufio"
    "os"
)

func main() {
    //创建一个Reader并将其与标准输入绑定，返回指向该Reader的指针
    inputReader := bufio.NewReader(os.Stdin)
    fmt.Println("Please enter some input: ")
    //该方法从输入中读取内容，直到碰到指定的字符，然后将读取到的内容连同delim字符一起放到缓冲区
    input, err := inputReader.ReadString('\n')
    //
    if err == nil {
        fmt.Printf("The input was: %s\n", input)
    }
}
```



标准输出`os.Stdout`和错误输出`os.Stderr`是屏幕。





# 从命令行读取参数

## os包

os 包中有一个 string 类型的切片变量 `os.Args`，用来处理一些基本的命令行参数，它在程序启动后读取命令行输入的参数。

```go
package main

import (
	"fmt"
	"os"
	"strings"
)

func main() {
	who := "Alice "
	if len(os.Args) > 1 {
		who += strings.Join(os.Args[1:], " ")
	}
	fmt.Println("Good Morning", who)
}
```

命令行参数会放置在切片`os.Args[]`中，从索引1开始。



## flag包

flag 包有一个扩展功能用来解析命令行选项。





# JSON

数据结构要在网络中传输或保存到文件，就必须对其编码和解码；目前存在很多编码格式：JSON，XML，gob，Google 缓冲协议等等。Go 语言支持所有这些编码格式。

- 数据结构 --> 指定格式 = `序列化` 或 `编码`（传输之前）
- 指定格式 --> 数据格式 = `反序列化` 或 `解码`（传输之后）



尽管 XML 被广泛的应用，但是 JSON 更加简洁、轻量（占用更少的内存、磁盘及网络带宽）和更好的可读性，这也使它越来越受欢迎。

JSON 与 Go 类型对应如下：

- bool 对应 JSON 的 boolean
- float64 对应 JSON 的 number
- string 对应 JSON 的 string
- nil 对应 JSON 的 null



## 编码

Marshal函数返回v的json编码：

```go
func Marshal(v interface{}) ([]byte, error)
```



```go
package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
)

type Address struct {
	Type    string
	City    string
	Country string
}

type VCard struct {
	FirstName string
	LastName  string
	Addresses []*Address
	Remark    string
}

func main() {
	pa := &Address{"private", "Aartselaar", "Belgium"}
	wa := &Address{"work", "Boom", "Belgium"}
	vc := VCard{"Jan", "Kersschot", []*Address{pa, wa}, "none"}
	// fmt.Printf("%v: \n", vc) // {Jan Kersschot [0x126d2b80 0x126d2be0] none}:
	// Go struct object to JSON format:
	js, _ := json.Marshal(vc)
	fmt.Printf("JSON format: %s", js)
	// using an encoder:
	file, _ := os.OpenFile("vcard.json", os.O_CREATE|os.O_WRONLY, 0666)
	defer file.Close()
    //传入io.Writer接口对象，返回Encoder结构变量的指针
	enc := json.NewEncoder(file)
    //将vc的json编码写入输入流，并写入一个换行符
	err := enc.Encode(vc)
	if err != nil {
		log.Println("Error in encoding json")
	}
}
```





# XML



