# 接口

> Go语言虽然不是传统的面向对象的编程语言（没有类和继承的概念），但是有非常灵活的接口的概念，可以实现很多面向对象的特性。

**接口**定义了一组抽象的方法：

```go
type Namer interface {
    Method1(param_list) return_type
    Method2(param_list) return_type
    ...
}
```

> 接口的名字约定俗成地由`方法名er`组成，此外也有一些形式如`~able`，`I~`

+ Go语言中的接口都很短，通常包含0~3个方法
+ Go语言中的接口类型的变量可以保存任何实现了该抽象方法的类型的值，其类型和值为底层变量的类型和值，调用方法为底层类型的同名方法

```go
package main

import (
	"fmt"
	"math"
)

type Abser interface {
	Abs() float64
}

func main() {
	var a Abser  //a并非Abser类型,而是赋值后的类型
	f := MyFloat(-math.Sqrt2)
	v := Vertex{3, 4}

	a = f  // a MyFloat 实现了 Abser
	a = &v // a *Vertex 实现了 Abser

	//a = v
    //err, 尽管v可以调用接收者为*Vertex的方法
    //但是接口认为Vertex类型未实现该接口而直接报错

	fmt.Println(a.Abs())
}

type MyFloat float64

func (f MyFloat) Abs() float64 { //为特定类型实现接口的抽象方法
	if f < 0 {
		return float64(-f)
	}
	return float64(f)
}

type Vertex struct {
	X, Y float64
}

func (v *Vertex) Abs() float64 {
	return math.Sqrt(v.X*v.X + v.Y*v.Y)
}

```



## 零值处理

接口值赋值某类型的零值：

```go
package main

import "fmt"

type I interface {
	M()
}

type T struct {
	S string
}

func (t *T) M() {
	if t == nil {             //手动防止空指针异常
		fmt.Println("<nil>")
		return
	}
	fmt.Println(t.S)
}

func main() {
	var i I

	var t *T
	i = t
	describe(i)        //(<nil>, *main.T)
	i.M()

	i = &T{"hello"}
	describe(i)        //(&{hello}, *main.T)
	i.M()
}

func describe(i I) {
	fmt.Printf("(%v, %T)\n", i, i)
}

```

接口值未被赋值（为接口值的零值nil）：

```go
package main

import "fmt"

type I interface {
	M()
}

func main() {
	var i I
	describe(i)        //(<nil>, <nil>)
	i.M()
}

func describe(i I) {
	fmt.Printf("(%v, %T)\n", i, i)
}

```





# 接口嵌套





# 空接口

定义了零个方法的接口称为空接口，空接口可以保存任何类型的值：

```go
package main

import "fmt"

func main() {
	var i interface{}
	describe(i)

    i = 42      //空接口变量可以保存任何类型的值,因为任何类型都实现了空接口中定义的方法(即没有方法)
	describe(i)

	i = "hello"
	describe(i)
}

func describe(i interface{}) {
	fmt.Printf("(%v, %T)\n", i, i)
}

```



## 类型断言

由于空接口变量可能保存了任何类型的值，可以使用类型断言访问其底层的具体值：

```go
package main

import "fmt"

func main() {
	var i interface{} = "hello"

	s := i.(string)      // hello
	fmt.Println(s)

	s, ok := i.(string)  // hello true
	fmt.Println(s, ok)

	f, ok := i.(float64) // 0 false
	fmt.Println(f, ok)

	f = i.(float64)      // 报错(panic)
	fmt.Println(f)
}

```



## 类型选择

```go
package main

import "fmt"

func do(i interface{}) {   //do()的参数可接受任意变量
	switch v := i.(type) {
	case int:
		fmt.Printf("Twice %v is %v\n", v, v*2)
	case string:
		fmt.Printf("%q is %v bytes long\n", v, len(v))
	default:
		fmt.Printf("I don't know about type %T!\n", v)
	}
}

func main() {
	do(21)
	do("hello")
	do(true)
}

```





# 方法重载

```go
/* fmt包中定义的 Stringer 是一个可以用字符串描述自己的类型
type Stringer interface {
    String() string
}
*/
package main

import "fmt"

type Person struct {
	Name string
	Age  int
}

func (p Person) String() string {
	return fmt.Sprintf("%v (%v years)", p.Name, p.Age)
}

func main() {
	a := Person{"Arthur Dent", 42}
	z := Person{"Zaphod Beeblebrox", 9001}
	fmt.Println(a, z)
}

```





# 反射包





# Printf和反射

​    2

1      3