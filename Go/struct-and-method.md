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
}

```





# 使用工厂方法创建结构体实例





# 使用自定义包中的结构体





# 匿名字段和内嵌结构体





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