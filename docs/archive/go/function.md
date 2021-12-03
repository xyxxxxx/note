# 函数

Go 里面有三种类型的函数：

* 普通的带有名字的函数
* 匿名函数或者 lambda 函数
* 方法（Methods）

函数可以将其他函数调用作为它的参数，只要这个被调用函数的返回值个数、返回值类型和返回值的顺序与调用函数所需求的实参是一致的，例如：假设 f1 需要 3 个参数 `f1(a,b,c int)`，同时 f2 返回 3 个参数 `f2(a,b int)(int,int,int)`，就可以这样调用 f1：`f1(f2(a,b))`。

<u>Go 语言不支持函数重载，不支持泛型。</u>

# 函数参数与返回值

函数能够接收参数供自己使用，也可以返回零个或多个值（我们通常把返回多个值称为返回一组值）。相比与 C、C++、Java 和 C#，<u> 多值返回是 Go 的一大特性 </u>，为我们判断一个函数是否正常执行提供了方便。

我们通过 `return` 关键字返回一组值。任何一个有返回值（单个或多个）的函数都必须以 `return` 或 `panic` 结尾。

函数定义时，它的形参一般是有名字的，不过我们也可以定义没有形参名的函数，只有相应的形参类型，就像这样：`func f(int,int,float64)`。

## 值传递与引用传递

Go 默认使用按值传递来传递参数，也就是传递参数的副本。函数接收参数副本之后，在使用变量的过程中可能对副本的值进行更改，但不会影响到原来的变量，比如 `Function(arg1)`。

如果你希望函数可以直接修改参数的值，而不是对参数的副本进行操作，你需要将参数的地址（变量名前面添加&符号，比如&variable）传递给函数，这就是按引用传递，比如 `Function(&arg1)`，此时传递给函数的是一个指针。

几乎在任何情况下，<u> 传递指针（一个 32 位或者 64 位的值）的消耗都比传递副本来得少 </u>。

在函数调用时，像切片（slice）、字典（map）、接口（interface）、通道（channel）这样的引用类型都是默认使用引用传递（即使没有显式的指出指针）。

## 命名的返回值

```go
package main

import "fmt"

var num int = 10
var numx2, numx3 int

func main() {
    numx2, numx3 = getX2AndX3(num)
    PrintValues()
    numx2, numx3 = getX2AndX3_2(num)
    PrintValues()
}

func PrintValues() {
    fmt.Printf("num = %d, 2x num = %d, 3x num = %d\n", num, numx2, numx3)
}

func getX2AndX3(input int) (int, int) {
    return 2 * input, 3 * input
}

func getX2AndX3_2(input int) (x2 int, x3 int) {//给出返回的变量
    x2 = 2 * input
    x3 = 3 * input
    // return x2, x3
    return//直接return
}
```

## 空白符

空白符 `_` 用来匹配一些不需要的值，然后丢弃掉

# 多返回值的错误测试

```go
value, err := pack1.Function1(param1)

//结束并返回错误
if err != nil {
	fmt.Printf("An error occured in pack1.Function1 with parameter %v", param1)
	return err
}

//终止程序运行
if err != nil {
	fmt.Printf("Program stopping with error %v", err)
	os.Exit(1)
}
```

错误获取放置在 if 语句的初始化部分：

```go
if err := file.Chmod(0664); err != nil {
	fmt.Println(err)
	return err
}
```

# 传递变长参数

```go
func Greeting(prefix string, who ...string)
Greeting("hello:", "Joe", "Anna", "Eileen")
//变量 who 的值为 []string{"Joe", "Anna", "Eileen"}
```

```go
package main

import "fmt"

func main() {
	x := min(1, 3, 2, 0)
	fmt.Printf("The minimum is: %d\n", x)
	slice := []int{7,9,3,5,1}
	x = min(slice...)
	fmt.Printf("The minimum in the slice is: %d", x)
}

func min(s ...int) int {
	if len(s)==0 {
		return 0
	}
	min := s[0]
	for _, v := range s {
		if v < min {
			min = v
		}
	}
	return min
}

/*
The minimum is: 0
The minimum in the slice is: 1
*/
```

# defer和追踪

defer 关键字设定在外层函数返回之后才执行某个语句或函数（但语句的参数会立即求值）：

```go
func a() {
	i := 0
	defer fmt.Println(i)
    fmt.Println("test")
	i++
	return
}

//test
//0
```

当有多个 defer 行为被注册时，它们会以逆序执行（类似栈，即后进先出）：

```go
func f() {
	for i := 0; i < 5; i++ {
		defer fmt.Printf("%d ", i)
	}
}

//4 3 2 1 0
```

defer 允许我们进行一些函数执行完成后的收尾工作：

```go
//关闭文件流
f, err := os.Open(filename)
if err != nil {
    return "", err
}
defer f.Close()

//解锁一个上锁的资源
mu.Lock()  
defer mu.Unlock()

//打印尾报告
printHeader()  
defer printFooter()

//关闭数据库连接
// open a database connection  
defer disconnectFromDB()
```

使用 defer 实现代码追踪：

```go
package main

import "fmt"

func trace(s string)   { fmt.Println("entering:", s) }
func untrace(s string) { fmt.Println("leaving:", s) }

func a() {
	trace("a")
	defer untrace("a")
	fmt.Println("in a")
}

func b() {
	trace("b")
	defer untrace("b")
	fmt.Println("in b")
	a()
}

func main() {
	b()
}

/*
entering: b
in b
entering: a
in a
leaving: a
leaving: b
*/
```

使用 defer 语句来记录函数的参数与返回值：

```go
package main

import (
	"io"
	"log"
)

func func1(s string) (n int, err error) {
	defer func() {
		log.Printf("func1(%q) = %d, %v", s, n, err)
	}()
	return 7, io.EOF
}

func main() {
	func1("Go")
}

//2011/10/04 10:46:11 func1("Go") = 7, EOF
```

# 内置函数

| 名称                   | 说明                                                         |
| ---------------------- | ------------------------------------------------------------ |
| `close`                | 用于管道通信                                                 |
| `len`、`cap`           | `len` 用于返回某个类型的长度或数量（字符串、数组、切片、map 和管道）；`cap` 是容量的意思，用于返回某个类型的最大容量（只能用于切片和 map） |
| `new`、`make`          | `new` 和 `make` 均是用于分配内存：`new` 用于值类型和用户定义的类型，如自定义结构，`make` 用于内置引用类型（切片、map 和管道）。它们的用法就像是函数，但是将类型作为参数：`new(type)`、`make(type)`。`new(T)` 分配类型 T 的零值并返回其地址，也就是指向类型 T 的指针。它也可以被用于基本类型：`v := new(int)`。`make(T)` 返回类型 T 的初始化之后的值，因此它比 `new` 进行更多的工作。**`new()` 是一个函数，不要忘记它的括号** |
| `copy`、`append`       | 用于复制和连接切片                                           |
| `panic`、`recover`     | 两者均用于错误处理机制                                       |
| `print`、`println`     | 底层打印函数，在部署环境中建议使用 fmt 包                    |
| `complex`、`real imag` | 用于创建和操作复数                                           |

# 递归函数

当一个函数在其函数体内调用自身，则称之为递归。

使用递归函数时经常会遇到的一个重要问题就是**栈溢出**：一般出现在大量的递归调用导致的程序栈内存分配耗尽。

# 函数作为参数

函数作为其他函数的参数进行传递，然后在其他函数内调用执行，一般称为**回调**：

```go
package main

import (
	"fmt"
)

func main() {
	callback(1, Add)
}

func Add(a, b int) {
	fmt.Printf("The sum of %d and %d is: %d\n", a, b, a+b)
}

func callback(y int, f func(int, int)) {
	f(y, 2) // this becomes Add(1, 2)
}
```

# 匿名函数

当我们不希望给函数起名字的时候，可以使用**匿名函数**，例如：`func(x,y int)int{return x + y}`。这样的函数不能够独立存在，但可以被赋值于某个变量，即保存函数的地址到变量中：`fplus:= func(x,y int)int{return x + y}`，然后通过变量名对函数进行调用：`fplus(3,4)`。 

```go
func() {
	sum := 0
	for i := 1; i <= 1e6; i++ {
		sum += i
	}
}()  //匿名函数的调用

hypot := func(x, y float64) float64 {  //hypot的类型为 func(float64, float64) float64
		return math.Sqrt(x*x + y*y)
	}
```

```go
package main

import "fmt"

func f() (ret int) {
	defer func() {
		ret++
	}()
	return 1
}
func main() {
	fmt.Println(f())
}

//变量 ret 的值为 2，因为 ret++ 是在执行 return 1 语句后发生的。
```

# 闭包

```go
package main

import "fmt"

func main() {
	// make an Add2 function, give it a name p2, and call it:
	p2 := Add2()
	fmt.Printf("Call Add2 for 3 gives: %v\n", p2(3))
	// make a special Adder function, a gets value 2:
	TwoAdder := Adder(2)
	fmt.Printf("The result is: %v\n", TwoAdder(3))
}

func Add2() func(b int) int {
	return func(b int) int {
		return b + 2
	}
}

func Adder(a int) func(b int) int {
	return func(b int) int {
		return a + b
	}
}

//Call Add2 for 3 gives: 5
//The result is: 5
```

```go
package main

import "fmt"

func main() {
	var f = Adder()
	fmt.Print(f(1), " - ")
	fmt.Print(f(20), " - ")
	fmt.Print(f(300))
}

func Adder() func(int) int {
	var x int	//x为返回函数的静态变量
	return func(delta int) int {
		x += delta
		return x
	}
}

//1 - 21 - 321
```

我们可以看到，在多次调用中，变量 x 的值是被保留的，即 `0+1=1`，然后 `1+20=21`，最后 `21+300=321`：闭包函数保存并积累其中的变量的值，不管外部函数退出与否，它都能够继续操作外部函数中的局部变量。

```go
package main

import "fmt"

func main() {
    var g int
    fmt.Print(g)
    func(i int) {
    	s := 0
    	for j := 0; j < i; j++ { s += j }
    	g = s
        }(1000) // Passes argument 1000 to the function literal.
    fmt.Print(g)
}

//0
//499500
```

闭包函数可以使用外部函数声明的变量。

> 一个返回值为另一个函数的函数可以被称之为工厂函数，这在您需要创建一系列相似的函数的时候非常有用：书写一个工厂函数而不是针对每种情况都书写一个函数。下面的函数演示了如何动态返回追加后缀的函数：
>
> ```go
> func MakeAddSuffix(suffix string) func(string) string {
> 	return func(name string) string {
> 		if !strings.HasSuffix(name, suffix) {
> 			return name + suffix
> 		}
> 		return name
> 	}
> }
> 
> addBmp := MakeAddSuffix(".bmp")
> addJpeg := MakeAddSuffix(".jpeg")
> 
> addBmp("file") // returns: file.bmp
> addJpeg("file") // returns: file.jpeg
> ```

# 闭包调试

使用 where 闭包函数 <u> 打印函数执行的位置 </u>。

```go
where := func() {
	_, file, line, _ := runtime.Caller(1)
	log.Printf("%s:%d", file, line)
}
where()
// some code
where()
// some more code
where()

//使用log包的flag参数
log.SetFlags(log.Llongfile)
log.Print("")
```

# 计算函数执行时间

```go
start := time.Now()
longCalculation()
end := time.Now()
delta := end.Sub(start)
fmt.Printf("longCalculation took this amount of time: %s\n", delta)
```

