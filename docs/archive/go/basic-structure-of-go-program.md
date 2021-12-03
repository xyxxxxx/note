# 文件名、关键字与标识符

> 关键字=保留字，标识符=变量名

Go 的源文件以 `.go` 为后缀名存储在计算机中，这些文件名均由小写字母组成，如 `scanner.go`。如果文件名由多个部分组成，则使用下划线 `_` 对它们进行分隔，如 `scanner_test.go`。文件名不包含空格或其他特殊字符。

Go 语言区分大小写。有效的标识符必须以字母（可以使用任何 UTF-8 编码的字符或 `_`）开头，然后紧跟着 0 个或多个字符或 Unicode 数字，如：X56、group1、_x23、i、өԑ12。以下是无效的标识符：

* 1ab（以数字开头）
* case（Go 语言的关键字）
* a+b（运算符是不允许的）

程序的代码通过语句来实现结构化。每个语句不需要像 C 家族中的其它语言一样以分号 `;` 结尾，因为这些工作都将由 Go 编译器自动完成。

# Go程序的基本结构和要素

## 包

> 相当于jar包

包是结构化代码的一种方式：每个程序都由包（通常简称为 pkg）的概念组成，可以使用自身的包或者从其它包中导入内容。每个 Go 文件都属于且仅属于一个包。一个包可以由许多以 `.go` 为扩展名的源文件组成，因此文件名和包名一般来说都是不相同的。

必须在源文件中非注释的第一行指明这个文件属于哪个包，如：`package main`。`package main` 表示一个可独立执行的程序，每个 Go 应用程序都包含一个名为 `main` 的包。

所有的包名都应该使用小写字母。

Go 的安装文件中包含了**标准库**，包含了大量的包。

如果想要构建一个程序，则包和包内的文件都必须以正确的顺序进行编译。包的依赖关系决定了其构建顺序。**如果对一个包进行更改或重新编译，所有引用了这个包的客户端程序都必须全部重新编译。**

Go 中的包模型采用了显式依赖关系的机制来达到快速编译的目的，编译器会从后缀名为 `.o` 的对象文件（需要且只需要这个文件）中提取传递依赖类型的信息。

如果 `A.go` 依赖 `B.go`，而 `B.go` 又依赖 `C.go`：

* 编译 `C.go`，`B.go`，然后是 `A.go`.
* 为了编译 `A.go`，编译器读取的是 `B.o` 而不是 `C.o`.

`import"fmt"` 告诉 Go 编译器这个程序需要使用 `fmt` 包（的函数，或其他元素），`fmt` 包实现了格式化 IO（输入/输出）的函数。包名被封闭在半角双引号 `""` 中。

如果需要多个包，它们可以被分别导入：

```go
import "fmt"
import "os"
```

或：

```go
import "fmt"; import "os"
```

但是还有更短且更优雅的方法（被称为因式分解关键字，该方法同样适用于 const、var 和 type 的声明或定义）：

```go
import (
   "fmt"
   "os"
)
```

如果包名不是以 `.` 或 `/` 开头，如 `"fmt"` 或者 `"container/list"`，则 Go 会在全局文件进行查找；如果包名以 `./` 开头，则 Go 会在相对目录中查找；如果包名以 `/` 开头（在 Windows 下也可以这样使用），则会在系统的绝对路径中查找。

**可见性规则** 

当标识符（包括常量、变量、类型、函数名、结构字段等等）以一个大写字母开头，如：Group1，那么使用这种形式的标识符的对象就可以被外部包的代码所使用（客户端程序需要先导入这个包），这被称为导出（像面向对象语言中的 public）；标识符如果以小写字母开头，则对包外是不可见的，但是他们在整个包的内部是可见并且可用的（像面向对象语言中的 private）。

## 函数

定义一个函数：

```go
func functionName(param1 type1, param2 type2, …) (ret1 type1, ret2 type2, …) {
   …
}

//eg
func Sum(a, b int) int {
    return a + b 
}
```

只有当某个函数需要被外部包调用的时候才使用大写字母开头，并遵循 Pascal 命名法；否则就遵循骆驼命名法，即第一个单词的首字母小写，其余单词的首字母大写。

## 注释

```go
// Package superman implements methods for saving the world.
//
// Experience has shown that a small number of procedures can prove
// helpful when attempting to save the world.
package superman

// enterOrbit causes Superman to fly into low Earth orbit, a position
// that presents several possibilities for planet salvation.
func enterOrbit() error {
   ...
}
```

godoc 工具会收集文档注释并产生一个技术文档。

## 类型

变量（或常量）包含数据，这些数据可以有不同的数据类型，简称类型。使用 var 声明的变量的值会自动初始化为该类型的零值。

类型可以是基本类型，如：int、float、bool、string；结构化的（复合的），如：struct、array、slice、map、channel；只描述类型的行为的，如：interface。

结构化的类型没有真正的值，它使用 nil 作为默认值。

## 程序示例

```go
package main

import (
   "fmt"
)

const c = "C"

var v int = 5

type T struct{}

func init() { // initialization of package
}

func main() {
   var a int
   Func1()
   // ...
   fmt.Println(a)
}

func (t T) Method1() {
   //...
}

func Func1() { // exported function Func1
   //...
}
```

## 类型转换

```go
a := 5.0
b := int(a)
```

# 常量

常量使用关键字 `const` 定义，用于存储不会改变的数据。存储在常量中的数据类型只可以是布尔型、数字型（整数型、浮点型和复数）和字符串型。

```go
const b string = "abc"	//显式类型定义
const b = "abc"			//隐式类型定义,编译器根据变量的值推断其类型
```

一个没有指定类型的常量被使用时，会根据其使用环境而推断出它所需要具备的类型。

```go
var n int
f(n + 5) // 无类型的数字型常量 5 它的类型在这里变成了 int
```

常量的值必须是能够在编译时就能够确定的；你可以在其赋值表达式中涉及计算过程，但是所有用于计算的值必须在编译期间就能获得。**因为在编译期间自定义函数均属于未知，因此无法用于常量的赋值，但内置函数可以使用，如：len（）。**

* 正确的做法：`const c1=2/3`
* 正确的做法：`const c2= len()`
* 错误的做法：`const c3= getNumber()`//引发构建错误：`getNumber()used as value`

数字型的常量是没有大小和符号的，并且可以使用任何精度而不会导致溢出：

```go
const Ln2 = 0.693147180559945309417232121458\
			176568075500134360255254120680009
//反斜杠 \ 可以在常量表达式中作为多行的连接符使用
const Log2E = 1/Ln2 // this is a precise reciprocal
const Billion = 1e9 // float constant
const hardEight = (1 << 100) >> 97
```

与各种类型的数字型变量相比，你无需担心常量之间的类型转换问题，因为它们都是非常理想的数字。不过需要注意的是，当常量赋值给一个精度过小的数字型变量时，可能会因为无法正确表达常量所代表的数值而导致溢出，这会在编译期间就引发错误。

常量也允许使用并行赋值的形式：

```go
const beef, two, c = "eat", 2, "veg"
const Monday, Tuesday, Wednesday, Thursday, Friday, Saturday = 1, 2, 3, 4, 5, 6
const (
	Monday, Tuesday, Wednesday = 1, 2, 3
	Thursday, Friday, Saturday = 4, 5, 6
)
```

常量还可以用作枚举：

```go
const (
	Unknown = 0
	Female = 1
	Male = 2
)

//也可以使用某个类型作为枚举常量的类型
type Color int

const (
	RED Color = iota // 0
	ORANGE // 1
	YELLOW // 2
	GREEN // ..
	BLUE
	INDIGO
	VIOLET // 6
)
```

# 变量

声明变量的一般形式是使用 `var` 关键字：

```go
var identifier type

var a, b *int
```

> Go 和许多编程语言不同，它在声明变量时将变量的类型放在变量的名称之后。
>
> 首先，它是为了避免像 C 语言中那样含糊不清的声明形式，例如：`int* a, b;`。
>
> 其次，这种语法能够按照从左至右的顺序阅读，使得代码更加容易理解。

当一个变量被声明之后，<u> 系统自动赋予它该类型的零值 </u>：int 为 0，float 为 0.0，bool 为 false，string 为空字符串，指针为 nil。记住，所有的内存在 Go 中都是经过初始化的。

变量的命名规则遵循骆驼命名法，即首个单词小写，每个新单词的首字母大写，例如：`numShips` 和 `startDate`。但如果你的 <u> 全局变量希望能够被外部包所使用 </u>，则需要将首个单词的首字母也大写。

一个变量（常量、类型或函数）在程序中都有一定的作用范围，称之为**作用域**。如果一个变量在函数体外声明，则被认为是**全局变量**，可以在整个包甚至外部包（被导出后）使用，不管你声明在哪个源文件里或在哪个源文件里调用该变量。

在函数体内声明的变量称之为**局部变量**，它们的作用域只在函数体内，参数和返回值变量也是局部变量。

尽管变量的标识符必须是唯一的，但你可以在某个代码块的内层代码块中使用相同名称的变量，则此时 <u> 外部的同名变量将会暂时隐藏 </u>（结束内部代码块的执行后隐藏的外部同名变量又会出现，而内部同名变量则被释放），你任何的操作都只会影响内部代码块的局部变量。

声明与赋值（初始化）语句也可以组合起来：

```go
var identifier [type] = value
var a int = 15
var i = 5
var b bool = false
var str string = "Go says hello to the world!"
```

不过自动推断类型并不是任何时候都适用的，当你想要给变量的类型并不是自动推断出的某种类型时，你还是需要显式指定变量的类型，例如：

```go
var n int64 = 2
```

```go
package main

import (
	"fmt"
   "runtime"
	"os"
)

func main() {
	var goos string = runtime.GOOS
	fmt.Printf("The operating system is: %s\n", goos)
	path := os.Getenv("PATH")  //简短赋值
	fmt.Printf("Path is %s\n", path)
}
```

## 值类型和引用类型

所有像 int、float、bool 和 string 这些基本类型都属于**值类型**，使用这些类型的变量直接指向存在内存中的值：

<img src="https://github.com/unknwon/the-way-to-go_ZH_CN/raw/master/eBook/images/4.4.2_fig4.1.jpg?raw=true"style="zoom：67%;"/>

另外，像数组和结构这些复合类型也是值类型。

当使用等号 `=` 将一个变量的值赋值给另一个变量时，如：`j = i`，实际上是在内存中将 i 的值进行了拷贝：

<img src="https://github.com/unknwon/the-way-to-go_ZH_CN/raw/master/eBook/images/4.4.2_fig4.2.jpg?raw=true"style="zoom：67%;"/>

你可以通过 `&i` 来获取变量 `i` 的内存地址，例如：0xf840000040（每次的地址都可能不一样）。值类型的变量的值存储在栈中。

更复杂的数据通常会需要使用多个字，这些数据一般使用引用类型保存。一个引用类型的变量 r1 存储的是 r1 的值所在的内存地址（数字），或内存地址中第一个字所在的位置。

<img src="https://github.com/unknwon/the-way-to-go_ZH_CN/raw/master/eBook/images/4.4.2_fig4.3.jpg?raw=true"style="zoom：67%;"/>

这个内存地址被称之为指针，这个指针实际上也被存在另外的某一个字中。同一个引用类型的指针指向的多个字可以是在连续的内存地址中（内存布局是连续的），这也是计算效率最高的一种存储形式；也可以将这些字分散存放在内存中，每个字都指示了下一个字所在的内存地址。

当使用赋值语句 `r2= r1` 时，只有引用（地址）被复制。

在 Go 语言中，指针属于引用类型，其它的引用类型还包括 slices，maps 和 channel。被引用的变量会存储在堆中，以便进行垃圾回收，且比栈拥有更大的内存空间。

## 简短赋值

```go
a := 50
b := false
```

a 和 b 的类型（int 和 bool）将由编译器自动推断。

这是使用变量的 <u> 首选形式 </u>，但是它 <u> 只能被用在函数体内 </u>，而不可以用于全局变量的声明与赋值。使用操作符 `:=` 可以高效地创建一个新的变量，称之为**初始化声明**。

```go
a, b, c = 5, 7, "abc"
a, b, c := 5, 7, "abc"
a, b = b, a
```

这被称为**并行**或**同时**赋值。

空白标识符 `_` 也被用于抛弃值，如值 `5` 在：`_,b =5,7` 中被抛弃。

## init函数

init 函数是一类特殊的函数，它不能被认为调用，而是在每个包完成初始化后自动执行，并且执行优先级比 main 函数高。

每个源文件都只能包含一个 init 函数。初始化总是以单线程执行，并且按照包的依赖关系顺序执行。

```go
//init.go
package trans

import "math"

var Pi float64

func init() {
   Pi = 4 * math.Atan(1) // init() function computes Pi
}
```

```go
//user_init.go
package main

import (
   "fmt"
   "./trans"	//需要init.go目录为./trans/init.go
)

var twoPi = 2 * trans.Pi

func main() {
   fmt.Printf("2*Pi = %g\n", twoPi) // 2*Pi = 6.283185307179586
}
```

```go
package main

var a string

func main() {
   a = "G"
   print(a)
   f1()
}

func f1() {
   a := "O"
   print(a)
   f2()
}

func f2() {
   print(a)
}

//GOG
```

