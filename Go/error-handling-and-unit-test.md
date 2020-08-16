# 错误处理与单元测试

Go程序使用error值来表示错误状态

+ `error`类型是一个内建接口

  ```go
  type error interface {
      Error() string
  }
  ```

+ 部分函数会返回一个 `error` 值，调用的它的代码应当判断这个错误是否等于 `nil` 来进行错误处理

  ```go
  i, err := strconv.Atoi("42")
  if err != nil {
      fmt.Printf("couldn't convert number: %v\n", err)
      return
  }
  fmt.Println("Converted integer:", i)
  ```

  

  

  ```go
  package main
  
  import (
  	"fmt"
  	"time"
  )
  
  type MyError struct {
  	When time.Time
  	What string
  }
  
  func (e *MyError) Error() string {
  	return fmt.Sprintf("at %v, %s",
  		e.When, e.What)
  }
  
  func run() error {
  	return &MyError{
  		time.Now(),
  		"it didn't work",
  	}
  }
  
  func main() {
  	if err := run(); err != nil {
          fmt.Println(err)  //打印error实际上是调用Error()函数
  	}
  }
  
  ```

  

