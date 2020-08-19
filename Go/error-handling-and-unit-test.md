# 错误处理

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

  
## 定义错误

```go
//利用errors.New或fmt.Errorf函数创建错误类型
package main

import (
	"errors"
	"fmt"
)

func Sqrt(f float64) (float64, error) {
	if f < 0 {
		//return 0, errors.New ("math - square root of negative number")
        return 0, fmt.Errorf ("math - square root of negative number %g", f)
	}
   // implementation of Sqrt
}

func main(){
    if f, err := Sqrt(-1); err != nil {
		fmt.Printf("Error: %s\n", err) //Error: math - square root of negative number
    }
}
```



在大部分情况下自定义错误结构类型是很有意义的，可以包含错误信息以外的其它有用信息。


  ```go
//手动创建类型的error方法
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
    return fmt.Sprintf("at %v, %s", e.When, e.What)
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

  

如果不同错误条件可能发生，那么对实际的错误使用类型断言或类型判断是很有用的，并且可以根据错误场景做一些补救和恢复操作。

```go
//  err != nil
if e, ok := err.(*os.PathError); ok {
	// remedy situation
}
```

```go
switch err := err.(type) {
	case ParseError:
		PrintParseError(err)
	case PathError:
		PrintPathError(err)
	...
	default:
		fmt.Printf("Not a special error, just %s\n", err)
}
```



所有错误都遵循一种命名规范：错误类型以`Error`结尾，错误变量以`err`或`Err`结尾。