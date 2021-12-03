# 并发、并行和 goroutine

goroutine 是由 Go 运行时管理的轻量级线程……

```go
package main

import (
	"fmt"
	"time"
)

func say(s string) {
	for i := 0; i < 5; i++ {
		time.Sleep(100 * time.Millisecond)
		fmt.Println(s)
	}
}

func main() {
	go say("world")  // 启动一个新的协程并执行
	say("hello")     // main协程中执行
}

```

# goroutine 间的信道

goroutine 之间需要通信，即互相发送和接收信息以协调/同步它们的工作。正如之前所提到的，Go 极不提倡 goroutine 使用共享内存进行通信，相应地，Go 提供了一种特殊的类型——通道（channel）——用于负责 goroutine 之间的通信。

通道就像是一个可以发送类型化数据的管道，数据本身及其所有权（即读写数据的能力）由通道进行传递，因此在任何给定的时刻，一个数据被设计为只有一个 goroutine 可以对其访问，因而也就避免了数据竞争。这种通过通道进行通信的方式保证了同步性。

通道就像工厂的传送带，一台机器（发送者 goroutine）在传送带上放置物品（数据），另一台机器（接收者 goroutine）从传送带拿到物品（数据），物品在某一时刻只有一台机器可以对其操作。

通常使用下列语句来声明通道：

```go
// var identifier chan datatype
//                     通道传输的数据类型
var ch1 chan string           // 字符串通道
ch1 = make(chan string)       // 通道是引用类型,因此使用`make()`为它分配内存
// or
ch1 := make(chan string)

funcCh := make(chan func())   // 函数通道
```

通道只能传输一种类型的数据，例如 `chan int` 或 `chan string`，但这里的类型可以是任意类型，包括空接口 `interface{}` 和通道类型自身。

通道实际上是类型化消息的队列，先进先出（FIFO）的结构保证了元素发送和接收的顺序一致（实际上，通道可以比作 Unix shells 中的双向管道（two-way pipe））。

通信操作符 `<-` 直观地标示了数据按照箭头的方向流动，使用方法如下：

+ `ch <- int1`：发送变量 `int1` 到通道 `ch`
+ `int2 := <- ch`：从通道 `ch` 获取一个值并赋给变量 `int2`
+ `<- ch`：从通道 `ch` 获取一个值（并返回）

通道的发送和接收都是原子操作，因此它们之间不会互相影响。来看下面的示例：

```go
package main

import (
	"fmt"
	"time"
)

func main() {
	ch := make(chan string)

	go sendData(ch)
	go getData(ch)

	time.Sleep(1e9)
}

func sendData(ch chan string) {
	ch <- "Washington"
	ch <- "Tripoli"
	ch <- "London"
	ch <- "Beijing"
	ch <- "Tokyo"
}

func getData(ch chan string) {
	var input string
	// time.Sleep(2e9)
	for {
		input = <-ch
		fmt.Printf("%s ", input)
	}
}

// output:
// Washington Tripoli London Beijing Tokyo
```

在这个示例中，注意以下几点：

+ `main()` 等待了 1 秒让两个 goroutine 完成，否则 `main()` 返回会导致程序结束，两个 goroutine 也会随之结束
+ 

* 没有缓冲的通道接收方会阻塞；带缓冲的通道仅当通道的缓冲区填满后，向其发送数据才会阻塞：

```go
package main

import "fmt"

func main() {
	ch := make(chan int, 2)        //缓冲区大小为2
	ch <- 1
	ch <- 2
	fmt.Println(<-ch)
	fmt.Println(<-ch)
}
```

* 发送者可通过 `close` 关闭一个通道来表示已经发送完毕

```go
package main

import (
	"fmt"
)

func fibonacci(n int, c chan int) {
	x, y := 0, 1
	for i := 0; i < n; i++ {
		c <- x
		x, y = y, x+y
	}
	close(c)
}

func main() {
	c := make(chan int, 10)
	go fibonacci(cap(c), c)
	for i := range c {   //从通道中循环取值,直到通道被关闭
		fmt.Println(i)
	}
}

```

* 通道关闭之后依然可以从中接收到该类型的零值

* `select` 语句可以使协程同时等待多个通信操作

```go
package main

import "fmt"

func fibonacci(c, quit chan int) {
	x, y := 0, 1
	for {
		select {
		case c <- x:        //当 <-c 准备时执行此分支
			x, y = y, x+y
		case <-quit:        //当 quit <-0 准备时执行此分支
			fmt.Println("quit")
			return
		}
	}
}

func main() {
	c := make(chan int)
	quit := make(chan int)
	go func() {
		for i := 0; i < 10; i++ {
			fmt.Println(<-c)
		}
		quit <- 0
	}()
	fibonacci(c, quit)
}

```

# 协程同步

# 使用select切换协程

# 通道，超时和计时器

# 协程和恢复

