# 并发、并行和协程

协程（goroutine）是由Go运行时管理的轻量级线程

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





# 使用通道进行协程间通信

通道是带有类型的管道，可以利用操作符`<-`发送或接收值：

```go
package main

import "fmt"

func sum(s []int, c chan int) {
	sum := 0
	for _, v := range s {
		sum += v
	}
	c <- sum // 将和送入 c
}

func main() {
	s := []int{7, 2, 8, -9, 4, 0}

	c := make(chan int)             //创建一个通道
	go sum(s[:len(s)/2], c)
	go sum(s[len(s)/2:], c)
    x, y := <-c, <-c                //等待:每当有一个值传入通道c,就传出一次

	fmt.Println(x, y, x+y)
}

```



+ 没有缓冲的通道接收方会阻塞；带缓冲的通道仅当通道的缓冲区填满后，向其发送数据才会阻塞：

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



+ 发送者可通过`close`关闭一个通道来表示已经发送完毕

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

+ 通道关闭之后依然可以从中接收到该类型的零值



+ `select`语句可以使协程同时等待多个通信操作

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



