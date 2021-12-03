Go 没有像 Java 和.NET 那样的 `try/catch` 异常机制：不能执行抛异常操作。但是有一套 `defer-panic-and-recover` 机制。

Go 的设计者觉得 `try/catch` 机制的使用太泛滥了，而且从底层向更高的层级抛异常太耗费资源。他们给 Go 设计的机制也可以“捕捉”异常，但是更轻量，并且只应该作为处理错误的最后手段。

Go 通过在函数和方法中返回错误对象（如果返回 nil，则没有错误发生）的方法处理普通错误。主调函数总是应该检查收到的错误，处理错误并且在函数发生错误的地方给用户返回错误信息。

# 错误处理

Go 程序使用 error 值来表示错误状态

* `error` 类型是一个内建接口

  ```go
  type error interface {
      Error() string
  }
  ```

* 部分函数会返回一个 `error` 值，调用的它的代码应当判断这个错误是否等于 `nil` 来进行错误处理

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

所有错误都遵循一种命名规范：错误类型以 `Error` 结尾，错误变量以 `err` 或 `Err` 结尾。

# 运行时异常panic

## panic

当发生像数组下标越界或类型断言失败这样的运行错误时，Go 运行时会触发运行时 panic，伴随着程序的崩溃抛出一个 `runtime.Error` 接口类型的值，这个错误值有个 `RuntimeError()` 方法用于区别普通错误。

`panic` 可以直接从代码初始化：当错误很严苛且不可恢复，程序不能继续运行时，可以使用 `panic` 函数产生一个中止程序的运行时错误。`panic` 接收一个做任意类型的参数，通常是字符串，在程序死亡时被打印出来。Go 运行时负责中止程序并给出调试信息。

```go
package main

import "fmt"

func main() {
	fmt.Println("Starting the program")
	panic("A severe error occurred: stopping the program!")
	fmt.Println("Ending the program")
}
```

以上示例运行时输出如下：

```
Starting the program
panic: A severe error occurred: stopping the program!
panic PC=0x4f3038
runtime.panic+0x99 /go/src/pkg/runtime/proc.c:1032
       runtime.panic(0x442938, 0x4f08e8)
main.main+0xa5 E:/Go/GoBoek/code examples/chapter 13/panic.go:8
       main.main()
runtime.mainstart+0xf 386/asm.s:84
       runtime.mainstart()
runtime.goexit /go/src/pkg/runtime/proc.c:148
       runtime.goexit()
---- Error run E:/Go/GoBoek/code examples/chapter 13/panic.exe with code Crashed
---- Program exited with code -1073741783
```

当发生错误必须终止程序时，`panic` 可以用于错误处理：

```go
if err != nil {
	panic("ERROR occurred:" + err.Error())
}
```

在多层嵌套的函数调用中调用 panic，可以马上中止当前函数的执行，所有的 defer 语句都会保证执行并把控制权交还给接收到 panic 的主调函数。这样向上冒泡直到最顶层，执行每层的 defer，在栈顶处程序崩溃，并在命令行中用传给 panic 的值报告错误情况：这个终止过程就是 **panicking**。

不应当随意地用 panic 中止程序，必须尽力补救错误让程序能继续执行。

## recover

recover 被用于从 panic 或错误场景中恢复：让程序可以从 panicking 重新获得控制权，停止终止过程进而恢复正常执行。

`recover` 只能在 defer 修饰的函数中使用：用于取得 panic 调用中传递过来的错误值。如果是正常执行，调用 `recover` 会返回 nil，且没有其它效果。

下面例子中的 protect 函数调用函数参数 g 来保护主调函数防止从 g 中抛出的运行时 panic，并展示 panic 中的信息，这跟 Java 中的 catch 块类似：

```go
func protect(g func()) {
	defer func() {
		log.Println("done")
		// Println executes normally even if there is a panic
		if err := recover(); err != nil {
			log.Printf("run time panic: %v", err)
		}
	}()
	log.Println("start")
	g() //   possible runtime-error
}
```

以下示例完整展示了 panic，defer 和 recover 的结合使用：

```go
package main

import (
	"fmt"
)

func badCall() {
	panic("bad end")
}

func test() {
	defer func() {
		if e := recover(); e != nil {
			fmt.Printf("Panicing %s\r\n", e)
		}
	}()
	badCall()
	fmt.Printf("After bad call\r\n") // <-- wordt niet bereikt
}

func main() {
	fmt.Printf("Calling test\r\n")
	test()
	fmt.Printf("Test completed\r\n")
}

//output:
//Calling test
//Panicing bad end
//Test completed
```

