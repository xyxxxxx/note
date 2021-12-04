# 布尔类型bool

```go
var b bool = true
```

布尔型的值只可以是常量 `true` 或者 `false`。

两个类型相同的值可以使用相等 `==` 或者不等 `!=` 运算符来进行比较并获得一个布尔型的值。

当相等运算符两边的值是完全相同的值的时候会返回 `true`，否则返回 `false`，并且只有在两个的值的类型相同的情况下才可以使用。

布尔型的常量和变量也可以通过和逻辑运算符（非 `!`、和 `&&`、或 `||`）结合来产生另外一个布尔值，这样的逻辑语句就其本身而言，并不是一个完整的 Go 语句。

逻辑值可以被用于条件结构中的条件语句，以便测试某个条件是否满足。另外，和 `&&`、或 `||` 与相等 `==` 或不等 `!=` 属于二元运算符，而非 `!` 属于一元运算符。在接下来的内容中，我们会使用 T 来代表条件符合的语句，用 F 来代表条件不符合的语句。

在 Go 语言中，&&和||是具有快捷性质的运算符，当运算符左边表达式的值已经能够决定整个表达式的值的时候（&&左边的值为 false，||左边的值为 true），运算符右边的表达式将不会被执行。利用这个性质，如果你有多个条件判断，应当将计算过程较为复杂的表达式放在运算符的右侧以减少不必要的运算。

对于布尔值的好的命名能够很好地提升代码的可读性，例如以 `is` 或者 `Is` 开头的 `isSorted`、`isFinished`、`isVisible`，使用这样的命名能够在阅读代码的获得阅读正常语句一样的良好体验，例如标准库中的 `unicode.IsDigit(ch)`。

格式化输出使用 `%t`。





# 数字类型

## 整数和浮点数

Go 语言支持整型和浮点型数字，并且原生支持复数，其中位的运算采用补码。

Go 语言中没有 float 类型和 double 类型。Go 语言中只有 float32 和 float64 类型。

与操作系统架构无关的类型都有固定的大小，并在类型的名称中就可以看出来：

整数：

* int8（-128 ->127）
* int16（-32768 ->32767）
* int32（-2，147，483，648 ->2，147，483，647）
* int64（-9，223，372，036，854，775，808 ->9，223，372，036，854，775，807）

无符号整数：

* uint8（0 ->255）
* uint16（0 ->65，535）
* uint32（0 ->4，294，967，295）
* uint64（0 ->18，446，744，073，709，551，615）

浮点型（IEEE-754 标准）：

* float32（+- 1e-45 ->+- 3.4*1e38）

* float64（+- 5*1e-324 ->107*1e308）

int 型是计算最快的一种类型。

整型的零值为 0，浮点型的零值为 0.0。

float32 精确到小数点后 7 位，float64 精确到小数点后 15 位。由于精确度的缘故，你在使用 `==` 或者 `!=` 来比较浮点数时应当非常小心。你最好在正式使用前测试对于精确度要求较高的运算。你应该尽可能地使用 float64，因为 `math` 包中所有有关数学运算的函数都会要求接收这个类型。

你可以通过增加前缀 0 来表示 8 进制数（如：077），增加前缀 0x 来表示 16 进制数（如：0xFF），以及使用 e 来表示 10 的连乘（如：1e3=1000，或者 6.022e23=6.022 x 1e23）。



Go 也有基于架构的类型，例如：int、uint 和 uintptr。

这些类型的长度都是根据运行程序所在的操作系统类型所决定的：

* `int` 和 `uint` 在 32 位操作系统上，它们均使用 32 位（4 个字节），在 64 位操作系统上，它们均使用 64 位（8 个字节）。
* `uintptr` 的长度被设定为足够存放一个指针即可。

**类型转换**

```go
package main

import "fmt"

func main() {
	var n int16 = 34
	var m int32
	// compiler error: cannot use n (type int16) as type int32 in assignment
	//m = n
	m = int32(n)

	fmt.Printf("32 bit int is: %d\n", m)
	fmt.Printf("16 bit int is: %d\n", n)
}
```

格式化输出中，`%d` 用于格式化整数（`%x` 和 `%X` 用于格式化 16 进制表示的数字），`%g` 用于格式化浮点型（`%f` 输出浮点数，`%e` 输出科学计数表示法），`%0nd` 用于规定输出长度为 n 的整数，其中开头的数字 0 是必须的。`%n.mg` 用于表示数字 n 并精确到小数点后 m 位，除了使用 g 之外，还可以使用 e 或者 f，例如：使用格式化字符串 `%5.2e` 来输出 3.4 的结果为 `3.40e+00`。



## 复数

Go 拥有复数类型 complex64（32 位实数和虚数），complex128（64 位实数和虚数）。

复数使用 `re+imI` 来表示，其中 `re` 代表实数部分，`im` 代表虚数部分，`I` 或 `i` 代表根号 -1。

```go
var c1 complex64 = 5 + 10i

c1 = complex(re, im)
```

格式化输出中，可以使用 `%v` 来表示复数，但当你希望只表示其中的一个部分的时候需要使用 `%f`。



## 位运算

二元运算符：按位与 `&`，按位或 `|`，按位异或 `^`

一元运算符：位左移 `<<`，位右移 `>>`

当希望把结果赋值给第一个操作数时，可以简写为 `a <<=2`或者`b^= a&0xffffffff`。

位左移的常见用例：

```go
type ByteSize float64
const (
	_ = iota // 通过赋值给空白标识符来忽略值
	KB ByteSize = 1<<(10*iota)
	MB
	GB
	TB
	PB
	EB
	ZB
	YB
)
```



## 逻辑运算符

Go 中拥有以下逻辑运算符：`==`、`!=`、`<`、`<=`、`>`、`>=`。



## 算术运算符

常见可用于整数和浮点数的二元运算符有 `+`、`-`、`*` 和 `/`。

`/` 对于整数运算而言，结果依旧为整数，例如：`9/4 ->2`。取余运算符只能作用于整数：`9%4 ->1`。

整数除以 0 可能导致程序崩溃，将会导致运行时的恐慌状态（如果除以 0 的行为在编译时就能被捕捉到，则会引发编译错误）；浮点数除以 0.0 会返回一个无穷尽的结果，使用 `+Inf` 表示。

对于整数和浮点数，你可以使用一元运算符 `++`（+1）和 `--`（-1），但只能用于后缀。

> 带有 `++` 和 `--` 的只能作为语句，而非表达式（ 虽然在C、C++ 和 Java 中允许）

在运算时**溢出**不会产生错误，Go 会简单地将超出位数抛弃。如果你需要范围无限大的整数或者有理数（意味着只被限制于计算机内存），你可以使用标准库中的 `big` 包，该包提供了类似 `big.Int` 和 `big.Rat` 这样的类型。



## 随机数

```go
package main
import (
	"fmt"
	"math/rand"
	"time"
)

func main() {
	for i := 0; i < 10; i++ {
		a := rand.Int()
		fmt.Printf("%d / ", a)
	}
	for i := 0; i < 5; i++ {
		r := rand.Intn(8)
		fmt.Printf("%d / ", r)
	}
	fmt.Println()
	timens := int64(time.Now().Nanosecond())
	rand.Seed(timens)	//提供随机数种子，一般使用当前时间的纳秒
	for i := 0; i < 10; i++ {
		fmt.Printf("%2.2f / ", 100*rand.Float32())	//返回0~1之间的伪随机数
	}
}
```



## 运算符的优先级

```
优先级 	运算符
 7 		^ !
 6 		* / % << >> & &^
 5 		+ - | ^
 4 		== != < <= >= >
 3 		<-
 2 		&&
 1 		||
```



## 类型别名

```go
package main
import "fmt"

type TZ int

func main() {
	var a, b TZ = 3, 4
	c := a + b
	fmt.Printf("c has the value: %d", c) // 输出：c has the value: 7
}
```



## 字符类型

字符类型严格来说不是 Go 语言的一个类型，而是整数的特殊用例，例如：

```go
var ch byte = 'A' 或 var ch byte = 65 或 var ch byte = '\x41'
//byte是uint8的别名
```

上述 3 种写法是等效的。



Go 同样支持 Unicode（UTF-8），因此字符同样称为 Unicode 代码点或者 runes，并在内存中使用 int 来表示。在文档中，一般使用格式 U+hhhh 来表示，其中 h 表示一个 16 进制数。其实 `rune` 也是 Go 当中的一个类型，并且是 `int32` 的别名。

在书写 Unicode 字符时，需要在 16 进制数之前加上前缀 `\u` 或者 `\U`。因为 Unicode 至少占用 2 个字节，所以我们使用 `int16` 或者 `int` 类型来表示。如果需要使用到 4 字节，则会加上 `\U` 前缀；前缀 `\u` 总是紧跟着长度为 4 的 16 进制数，前缀 `\U` 紧跟着长度为 8 的 16 进制数。

```go
var ch int = '\u0041'
var ch2 int = '\u03B2'
var ch3 int = '\U00101234'
fmt.Printf("%d - %d - %d\n", ch, ch2, ch3) // integer
fmt.Printf("%c - %c - %c\n", ch, ch2, ch3) // character
fmt.Printf("%X - %X - %X\n", ch, ch2, ch3) // UTF-8 bytes
fmt.Printf("%U - %U - %U", ch, ch2, ch3) // UTF-8 code point

/*print
65 - 946 - 1053236
A - β - r
41 - 3B2 - 101234
U+0041 - U+03B2 - U+101234
*/
```

格式化说明符 `%c` 用于表示字符；当和字符配合使用时，`%v` 或 `%d` 会输出用于表示该字符的整数；`%U` 输出格式为 U+hhhh 的字符串。

包 `unicode` 包含了一些针对测试字符的非常有用的函数（其中 `ch` 代表字符）：

* 判断是否为字母：`unicode.IsLetter(ch)`
* 判断是否为数字：`unicode.IsDigit(ch)`
* 判断是否为空白符号：`unicode.IsSpace(ch)`

这些函数返回一个布尔值。包 `utf8` 拥有更多与 rune 相关的函数。



# 字符串

字符串是字符的一个序列（当字符为 ASCII 码时则占用 1 个字节，为 UTF-8 字符时根据需要占用 2-4 个字节）。

UTF-8 是被广泛使用的编码格式，是文本文件的标准编码，其它包括 XML 和 JSON 在内，也都使用该编码。由于该编码对占用字节长度的不定性，Go 中的字符串里面的字符也可能根据需要占用 1 至 4 个字节，这与其它语言如 C++、Java 或者 Python 不同（Java 始终使用 2 个字节）。Go 这样做的好处是不仅减少了内存和硬盘空间占用，同时也 <u> 不用像其它语言那样需要对使用 UTF-8 字符集的文本进行编码和解码 </u>。

字符串是一种值类型，且值不可变；更深入地讲，字符串是字节的定长数组。

和 C/C++ 不一样，Go 中的字符串是根据长度限定，而非特殊字符 `\0`。

一般的比较运算符（`==`、`!=`、`<`、`<=`、`>=`、`>`）通过在内存中按字节比较来实现字符串的对比。你可以通过函数 `len()` 来获取字符串所占的字节长度，例如：`len(str)`。

字符串的内容（纯字节）可以通过标准索引法来获取：

* 字符串 str 的第 1 个字节：`str[0]`
* 第 i 个字节：`str[i - 1]`
* 最后 1 个字节：`str[len(str)-1]`

需要注意的是，这种转换方案只对纯 ASCII 码的字符串有效。

**字符串拼接符 `+`**：两个字符串 `s1` 和 `s2` 可以通过 `s:= s1+ s2` 拼接在一起。

```go
s := "hel" + "lo,"
s += "world!"
fmt.Println(s) //输出 “hello, world!”
```

> 在循环中使用加号 `+` 拼接字符串并不是最高效的做法，更好的办法是使用函数 `strings.Join()`。有没有更好的办法？有！使用字节缓冲（`bytes.Buffer`）拼接更加给力！

**长字符串引用**：包含多种字符的长字符串可以使用\`\` 引用。

```go
json := `{
			"id": 1,
            "name": "深入理解计算机系统",
            "author": {
            "firstName": "Randal",
            "lastName": "Bryant"
            },
            "isbn": "9787111544937",
            "tags": ["computer system", "primer"]
		}`
```





# strings和strconv包

## 包含

`HasPrefix` 判断字符串 `s` 是否以 `prefix` 开头：

```go
strings.HasPrefix(s, prefix string) bool
```

`HasSuffix` 判断字符串 `s` 是否以 `suffix` 结尾：

```go
strings.HasSuffix(s, suffix string) bool
```

`Contains` 判断字符串 `s` 是否包含 `substr`：

```go
strings.Contains(s, substr string) bool
```



## 位置

`Index` 返回字符串 `str` 在字符串 `s` 中的索引（`str` 的第一个字符的索引），-1 表示字符串 `s` 不包含字符串 `str`：

```go
strings.Index(s, str string) int
```

`LastIndex` 返回字符串 `str` 在字符串 `s` 中最后出现位置的索引（`str` 的第一个字符的索引），-1 表示字符串 `s` 不包含字符串 `str`：

```go
strings.LastIndex(s, str string) int
```

如果需要查询 <u> 非 ASCII 编码的字符在父字符串中的位置 </u>，建议使用以下函数来对字符进行定位：

```go
strings.IndexRune(s string, r rune) int
```



## 替换

`Replace` 用于将字符串 `str` 中的前 `n` 个字符串 `old` 替换为字符串 `new`，并返回一个新的字符串，如果 `n =-1`则替换所有字符串`old`为字符串`new`：

```go
strings.Replace(str, old, new, n) string
```



## 统计

`Count` 用于计算字符串 `str` 在字符串 `s` 中出现的非重叠次数：

```go
strings.Count(s, str string) int
```



## 重复

`Repeat` 用于重复 `count` 次字符串 `s` 并返回一个新的字符串：

```go
strings.Repeat(s, count int) string
```



## 大小写

`ToLower` 将字符串中的 Unicode 字符全部转换为相应的小写字符：

```go
strings.ToLower(s) string
```

`ToUpper` 将字符串中的 Unicode 字符全部转换为相应的大写字符：

```go
strings.ToUpper(s) string
```



## 裁剪

你可以使用 `strings.TrimSpace(s)` 来剔除字符串开头和结尾的空白符号；如果你想要剔除指定字符，则可以使用 `strings.Trim(s,"cut")` 来将开头和结尾的 `cut` 去除掉。该函数的第二个参数可以包含任何字符，如果你只想剔除开头或者结尾的字符串，则可以使用 `TrimLeft` 或者 `TrimRight` 来实现。



## 分割

`strings.Fields(s)` 将会利用 1 个或多个空白符号来作为动态长度的分隔符将字符串分割成若干小块，并返回一个 slice，如果字符串只包含空白符号，则返回一个长度为 0 的 slice。

`strings.Split(s,sep)` 用于自定义分割符号来对指定字符串进行分割，同样返回 slice。

因为这 2 个函数都会返回 slice，所以习惯使用 for-range 循环来对其进行处理。



## 拼接

`Join` 用于将元素类型为 string 的 slice 使用分割符号来拼接组成一个字符串：

```go
strings.Join(sl []string, sep string) string
             //必须是string类型的数组
```



## 读取

函数 `strings.NewReader(str)` 用于生成一个 `Reader` 并读取字符串中的内容，然后返回指向该 `Reader` 的指针，从其它类型读取内容的函数还有：

* `Read()` 从[]byte 中读取内容。
* `ReadByte()` 和 `ReadRune()` 从字符串中读取下一个 byte 或者 rune。



## 类型转换

从数字类型转换到字符串：

* `strconv.Itoa(i int)string` 返回数字 i 所表示的字符串类型的十进制数。
* `strconv.FormatFloat(f float64,fmt byte,prec int,bitSize int)string` 将 64 位浮点型的数字转换为字符串，其中 `fmt` 表示格式（其值可以是 `'b'`、`'e'`、`'f'` 或 `'g'`），`prec` 表示精度，`bitSize` 则使用 32 表示 float32，用 64 表示 float64。

从字符串类型转换为数字：

* `strconv.Atoi(s string)(i int,err error)` 将字符串转换为 int 型。
* `strconv.ParseFloat(s string,bitSize int)(f float64,err error)` 将字符串转换为 float64 型。



```go
package main

import (
	"fmt"
	"strconv"
)

func main() {
	var orig string = "666"
	var an int
	var newS string

	fmt.Printf("The size of ints is: %d\n", strconv.IntSize)	  

	an, _ = strconv.Atoi(orig)
	fmt.Printf("The integer is: %d\n", an) 
	an = an + 5
	newS = strconv.Itoa(an)
	fmt.Printf("The new string is: %s\n", newS)
}
```





# 时间和日期

`time` 包为我们提供了一个数据类型 `time.Time`（作为值使用）以及显示和测量时间和日期的功能函数。

当前时间可以使用 `time.Now()` 获取，或者使用 `t.Day()`、`t.Minute()` 等等来获取时间的一部分；你甚至可以自定义时间格式化字符串，例如：`fmt.Printf("%02d.%02d.%4d\n",t.Day(),t.Month(),t.Year())` 将会输出 `21.07.2011`。

`Duration` 类型表示两个连续时刻所相差的纳秒数，类型为 int64。`Location` 类型映射某个时区的时间，UTC 表示通用协调世界时间。



```go
package main
import (
	"fmt"
	"time"
)

var week time.Duration
func main() {
	t := time.Now()
	fmt.Println(t) // e.g. Wed Dec 21 09:52:14 +0100 RST 2011
	fmt.Printf("%02d.%02d.%4d\n", t.Day(), t.Month(), t.Year())
	// 21.12.2011
	t = time.Now().UTC()
	fmt.Println(t) // Wed Dec 21 08:52:14 +0000 UTC 2011
	fmt.Println(time.Now()) // Wed Dec 21 09:52:14 +0100 RST 2011
	// calculating times:
	week = 60 * 60 * 24 * 7 * 1e9 // must be in nanosec
	week_from_now := t.Add(time.Duration(week))
	fmt.Println(week_from_now) // Wed Dec 28 08:52:14 +0000 UTC 2011
	// formatting times:
	fmt.Println(t.Format(time.RFC822)) // 21 Dec 11 0852 UTC
	fmt.Println(t.Format(time.ANSIC)) // Wed Dec 21 08:56:34 2011
	// 自定义格式
	fmt.Println(t.Format("02 Jan 2006 15:04")) // 21 Dec 2011 08:52
	s := t.Format("20060102")
	fmt.Println(t, "=>", s)
	// Wed Dec 21 08:52:14 +0000 UTC 2011 => 20111221
}
```





# 指针

Go 语言为程序员提供了控制数据结构的指针的能力；但是，你不能进行指针运算。通过给予程序员基本内存布局，Go 语言允许你控制特定集合的数据结构、分配的数量以及内存访问模式，这些对构建运行良好的系统是非常重要的：指针对于性能的影响是不言而喻的，而如果你想要做的是系统编程、操作系统或者网络应用，指针更是不可或缺的一部分。

Go 语言的取地址符是 `&`，放到一个变量前使用就会返回相应变量的内存地址。

```go
var i1 = 5
fmt.Printf("An integer: %d, it's location in memory: %p\n", i1, &i1)
//An integer: 5, its location in memory: 0x6b0820
```

指针的格式化标识符为 `%p`



```go
var intP *int
intP = &i1
```

`intP` 存储了 `i1` 的内存地址；它指向了 `i1` 的位置，它引用了变量 `i1`。

**一个指针变量可以指向任何一个值的内存地址**。可以在指针类型前面加上\*号（前缀）来获取指针所指向的内容，这里的\*号是一个类型更改器。使用一个指针引用一个值被称为**间接引用**。

未初始化的指针的零值为 `nil`，未指向任何地址



```go
package main
import "fmt"
func main() {
	s := "good bye"
	var p *string = &s
	*p = "ciao"
	fmt.Printf("Here is the pointer p: %p\n", p) // prints address
	fmt.Printf("Here is the string *p: %s\n", *p) // prints string
	fmt.Printf("Here is the string s: %s\n", s) // prints same string
}

/*
Here is the pointer p: 0x2540820
Here is the string *p: ciao
Here is the string s: ciao
*/
```

<img src="https://github.com/unknwon/the-way-to-go_ZH_CN/raw/master/eBook/images/4.9_fig4.5.png?raw=true"style="zoom：67%;"/>

> 你不能获取字面量或常量的地址



Go 语言和 C、C++ 以及 D 语言这些低级（系统）语言一样，都有指针的概念。但是对于 <u> 经常导致 C 语言内存泄漏继而程序崩溃的指针运算 </u>（所谓的指针算法，如：`pointer+2`，移动指针指向字符串的字节数或数组的某个位置）<u> 是不被允许的 </u>。Go 语言中的 <u> 指针保证了内存安全 </u>，更像是 Java、C#和 VB.NET 中的引用。

指针的一个高级应用是你可以传递一个变量的引用（如函数的参数），这样不会传递变量的拷贝。指针传递是很廉价的，只占用 4 个或 8 个字节。当程序在工作中需要占用大量的内存，或很多变量，或者两者都有，使用指针会减少内存占用和提高效率。被指向的变量也保存在内存中，直到没有任何指针指向它们，所以从它们被创建开始就具有相互独立的生命周期。

指针也可以指向另一个指针，并且可以进行任意深度的嵌套，导致你可以有多级的间接引用，但在大多数情况这会使你的代码结构不清晰。

