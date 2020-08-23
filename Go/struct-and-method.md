# 结构体定义

```go
package main

import "fmt"

type Vertex struct {
	X int
	Y int
}

func main() {
	v := Vertex{1, 2}
	v.X = 4
    p := &v
    p.X = 1e9          // 隐式间接引用,相当于 (*p).X
	fmt.Println(v.X)
    
    v1 := Vertex{1, 2}  // 创建一个 Vertex 类型的结构体
    v2 := Vertex{X: 1}  // Y:0 被隐式地赋予
    v3 := Vertex{}      // X:0 Y:0
    p  := &Vertex{1, 2} // 创建一个 *Vertex 类型的结构体指针
    fmt.Println(v1, p, v2, v3)  // {1 2} &{1 2} {1 0} {0 0}
    fmt.Println(unsafe.Sizeof(Vertex{}))  // 返回Vertex结构的一个实例的大小
}

```





# 使用工厂方法创建结构体实例

Go语言不支持面向对象编程语言中的构造函数，但是可以很容易地在Go中实现“构造工厂”方法。按照惯例，工厂的名字以new或New开头。假设定义了如下结构：

```go
type File struct {
    fd      int     // 文件描述符
    name    string  // 文件名
}
```

结构对应的工厂方法就可以写成：

```go
func NewFile(fd int, name string) *File {
    if fd < 0 {
        return nil
    }

    return &File{fd, name}
    //或者
    //f := new(File)
    //f.fd = fd
    //f.name = name
    //return &f
}
```

然后这样调用之：

```go
f := NewFile(10, "./test.txt")
```



## 强制使用工厂方法

就像在面向对象语言那样，通过将类型变成私有的从而强制用户使用工厂方法。

```go
type matrix struct {
    ...
}

func NewMatrix(params) *matrix {
    m := new(matrix) // 初始化 m
    return m
}
```

在其他包使用工厂方法：

```go
package main
import "matrix"
...
wrong := new(matrix.matrix)     // 编译失败（matrix 是私有的）
right := matrix.NewMatrix(...)  // 实例化 matrix 的唯一方式
```





# 带标签的结构体

结构体中的字段除了有名字和类型外，还可以有一个可选的标签（tag）：它是一个附属于字段的字符串，可以是文档或其他的重要标记。标签的内容不可以在一般的编程中使用，只有包 `reflect` 能获取它。

```go
package main

import (
	"fmt"
	"reflect"
)

type TagType struct { // tags
	field1 bool   "An important answer"
	field2 string "The name of the thing"
	field3 int    "How much there are"
}

func main() {
	tt := TagType{true, "Barak Obama", 1}
	for i := 0; i < 3; i++ {
		refTag(tt, i)
	}
}

func refTag(tt TagType, ix int) {
	ttType := reflect.TypeOf(tt)
	ixField := ttType.Field(ix)
	fmt.Printf("%v\n", ixField.Tag)
}

//output
//An important answer
//The name of the thing
//How much there are
```





# 匿名字段和内嵌结构体

结构体可以包含一个或多个 **匿名（或内嵌）字段**，即这些字段没有显式的名字，只有字段的类型是必须的，此时类型就是字段的名字。匿名字段本身可以是一个结构体类型，即 **结构体可以包含内嵌结构体**。

可以粗略地将这个和面向对象语言中的继承概念相比较，Go 语言中的继承是通过内嵌或组合来实现的。

```go
package main

import "fmt"

type innerS struct {
	in1 int
	in2 int
}

type outerS struct {
	b    int
	c    float32
	int  // anonymous field
	innerS //anonymous field
}

func main() {
	outer := new(outerS)
	outer.b = 6
	outer.c = 7.5
	outer.int = 60
	outer.in1 = 5
	outer.in2 = 10

	fmt.Printf("outer.b is: %d\n", outer.b)
	fmt.Printf("outer.c is: %f\n", outer.c)
	fmt.Printf("outer.int is: %d\n", outer.int)
	fmt.Printf("outer.in1 is: %d\n", outer.in1)
	fmt.Printf("outer.in2 is: %d\n", outer.in2)

	// 使用结构体字面量
	outer2 := outerS{6, 7.5, 60, innerS{5, 10}}
	fmt.Println("outer2 is:", outer2)
}
```

通过类型 `outer.int` 的名字可以获取存储在匿名字段中的数据，在一个结构体中对于每一种数据类型只能有一个匿名字段。

结构体也可以作为一个匿名字段来使用，这个简单的“继承”机制提供了一种方式，使得可以从另外一个或一些类型继承部分或全部实现。



当两个字段拥有相同的名字时：

+ 外层名字会覆盖内层名字（但是两者的内存空间都保留）
+ 相同的名字在同一级别，那么必须由程序员自己修正

例如：

```go
type D struct {B; b float32}
var d D
```

使用 `d.b` 没有问题：它是 float32，而不是 `B` 的 `b`。如果想要内层的 `b` 可以通过 `d.B.b` 得到。

```go
type A struct {a int}
type B struct {a, b int}

type C struct {A; B}
var c C
```

使用 `c.a` 是错误的，会导致编译器错误：**ambiguous DOT reference c.a disambiguate with either c.A.a or c.B.a**。





# 方法

+ 最常用方法为结构体的方法

```go
package main

import (
	"fmt"
	"math"
)

type Vertex struct {
	X, Y float64
}

func (v Vertex) Abs() float64 {  // 接收者参数(v Vertex), 指明是结构体Vertex的方法 (相当于Java中类的方法)
	return math.Sqrt(v.X*v.X + v.Y*v.Y)
}

func main() {
	v := Vertex{3, 4}
	fmt.Println(v.Abs())
}

```



+ 接收者的类型定义和方法声明必须在同一包内；不能为内建类型声明方法

```go
package main

import (
	"fmt"
	"math"
)

type MyFloat float64

func (f MyFloat) Abs() float64 {  //为类型别名声明方法
	if f < 0 {
		return float64(-f)
	}
	return float64(f)
}

func main() {
	f := MyFloat(-math.Sqrt2)
	fmt.Println(f.Abs())
}
```



+ 指针接收者的方法可以修改接收者指向的值

```go
package main

import (
	"fmt"
	"math"
)

type Vertex struct {
	X, Y float64
}

func (v Vertex) Abs() float64 {      //传入Vertex变量的值(副本)
	return math.Sqrt(v.X*v.X + v.Y*v.Y)
}

/*
func (v *Vertex) Abs() float64 {     //不需要再为形参复制一遍结构体,更加高效
	return math.Sqrt(v.X*v.X + v.Y*v.Y)
}
*/

func (v *Vertex) Scale(f float64) {  //传入Vertex变量的指针
	v.X = v.X * f
	v.Y = v.Y * f
}

func main() {
	v := Vertex{3, 4}
    v.Scale(10)     //编译器自动解释为 (&v).Scale(10)
	fmt.Println(v.Abs())
}

```

> 函数的带指针参数必须接受一个指针，而方法的指针接收者既能接受值又能接受指针
>
> 函数的参数必须接受一个值，而方法的值接收者既能接受值又能接受指针



+ 指针接收者比值接收者更常用：避免在每次调用方法时复制该值（若值的类型为大型结构体时，这样做会更加高效）：





# 类型的String()方法和格式化标准符





# 垃圾回收和SetFinalizer