# 数组和切片

## 数组

```go
var arr1 [5]int

var arrAge = [5]int{18, 20, 15, 22, 16}
var arrAge = [5]int{18, 20, 15}                   //前3个索引赋值
var arrAge = [...]int{18, 20, 15, 22, 16}         //编译器统计数量
var arrKeyValue = [5]string{3: "Chris", 4: "Ron"} //指定索引赋值
```

<img src="https://github.com/unknwon/the-way-to-go_ZH_CN/raw/master/eBook/images/7.1_fig7.1.png?raw=true" alt="img" style="zoom:67%;" />

+ 当声明数组时所有的元素都会被自动初始化为默认值 0
+ Go 语言中的数组是一种 **值类型**（不像 C/C++ 中是指向首元素的指针）



```go
var arr1 = new([5]int)	//arr1的类型是*[5]int
var arr2 [5]int			//arr2的类型是[5]int

arr2 := *arr1		//拷贝了一份数据
arr2[2] = 100		//修改arr2不影响arr1
```

所以在函数中数组作为参数传入时，如 `func1(arr2)`，会产生一次数组拷贝，func1 方法不会修改原始的数组 arr2。如果你想修改原数组，那么 arr2 必须通过&操作符以引用方式传过来，例如 `func1(&arr2)`

**遍历**

```go
package main
import "fmt"

func main() {
	var arr1 [5]int

	for i:=0; i < len(arr1); i++ {
		arr1[i] = i * 2
	}

	for i:=0; i < len(arr1); i++ {
		fmt.Printf("Array at index %d is %d\n", i, arr1[i])
	}
}
```

```go
//使用for range遍历
	for i,_:= range arr1 {
    ...
    }
```

**多维数组**

`[3][5]int`，`[2][2][2]float64`

```go
[3][5]int
[2][2][2]float64
```



## 切片

切片是对数组一个连续片段的引用，所以切片是一个引用类型。

+ 切片是可索引的
+ 切片的长度是它包含的元素个数，切片的长度可变。`len()`函数返回切片的长度
+ 切片的容量是<u>从它的第一个元素开始数，到其底层数组元素末尾的个数</u>。`cap()`函数返回切片的容量，切片的长度不会超过其容量，即`len(s) <= cap(s)`
+ 可以把切片传递给以数组为形参的函数

> Go程序中切片比数组更常用

```go
var slice1 []int               //声明切片,其零值为nil,指针为nil,长度和容量为0
var slice2 []int = arr1[0:10]  //声明并初始化切片,左闭右开
var slice3 []int = arr1[:]

var s1 = []int{5, 6, 7, 8, 22}   //创建一个数组,并返回一个引用之的切片

s1 := []int{1,2,3}
s2 := s1[:]
```

<img src="https://github.com/unknwon/the-way-to-go_ZH_CN/raw/master/eBook/images/7.2_fig7.2.png?raw=true" alt="img" style="zoom:67%;" />

**make()创建切片**

```go
v := make([]int, 50, 100)	//分配一个长度为100的元素为0的int数组,并且创建了一个长度为50,容量为100的切片v,该切片指向数组的前50个元素
new([100]int)[0:50]         //隐式间接引用,相当于 (*new([100]int))[0:50] 
```

<img src="https://github.com/unknwon/the-way-to-go_ZH_CN/raw/master/eBook/images/7.2_fig7.2.1.png?raw=true" alt="img" style="zoom: 50%;" />

```go
// new() 和 make() 的区别
p := new([]int)      //如图7.3上,即new了一个int切片(未初始化),返回一个指向该切片的指针
					 // *p == nil
p := make([]int, 0)  //如图7.3下,即创建了长度和容量为0一个切片,指向一个长度为0的未初始化的数组,返回该切片
                     // p != nil,因为指向了一个长度为0的数组
```

<img src="https://github.com/unknwon/the-way-to-go_ZH_CN/raw/master/eBook/images/7.2_fig7.3.png?raw=true" alt="img" style="zoom: 67%;" />



## for-range 遍历

当使用 `for` 循环遍历切片时，每次迭代都会返回两个值。第一个值为当前元素的下标，第二个值为该下标所对应元素的一份副本：

```go
package main

import "fmt"

func main() {
	pow := make([]int, 10)
	for i := range pow {      // 单个参数表示索引
		pow[i] = 1 << uint(i) // == 2**i
	}
	for _, value := range pow { //忽略索引
		fmt.Printf("%d\n", value)
	}
}

```



## 切片重组reslice



## 切片的复制和追加

增加切片的容量必须创建一个新的、更大容量的切片，然后将原有切片的内容复制到新的切片：

```go
package main

import "fmt"

func main() {
	var s []int
	printSlice(s)

	s = append(s, 0)
	printSlice(s)

	s = append(s, 1)
	printSlice(s)

    s = append(s, 2)  // t := make([]int,3,4); t[i]=s[i]; t[len(s)]=2; s=t
	printSlice(s)
	
}

func printSlice(s []int) {
	fmt.Printf("len=%d cap=%d %v\n", len(s), cap(s), s)
}

/*
len=0 cap=0 []
len=1 cap=1 [0]
len=2 cap=2 [0 1]
len=3 cap=4 [0 1 2]
*/
```

```go
s = append(s,t...)    //将t展开再拼接
```



## 字符串、数组和切片的应用





# 映射Map

## 初始化

```go
package main

import "fmt"

type Vertex struct {
	Lat, Long float64
}

var m map[string]Vertex    //将string类型映射到Vertex
						   //映射的零值为nil,没有键值对,不能添加键值对

func main() {
	m = make(map[string]Vertex)  //返回初始化完了的映射
	m["Bell Labs"] = Vertex{     //插入或修改元素
		40.68433, -74.39967,
	}
	fmt.Println(m["Bell Labs"])
}
```

```go
var m = map[string]Vertex{
	"Bell Labs": Vertex{
		40.68433, -74.39967,
	},
	"Google": Vertex{
		37.42202, -122.08408,
	},
}

var m = map[string]Vertex{
	"Bell Labs": {40.68433, -74.39967},
	"Google":    {37.42202, -122.08408},
}
```



## 基本操作

```go
package main

import "fmt"

func main() {
	m := make(map[string]int)

	m["Answer"] = 42  //增加键值对
	fmt.Println("The value:", m["Answer"])

	m["Answer"] = 48  //修改键值对
	fmt.Println("The value:", m["Answer"])

	delete(m, "Answer")  //删除键值对
	fmt.Println("The value:", m["Answer"])

	v, ok := m["Answer"] //返回值和 是否存在,若存在ok=true,
    									//若不存在ok=false,返回值为该类型的零值
	fmt.Println("The value:", v, "Present?", ok)
}

```



## for-range 遍历



## map类型的切片



## map的排序



## map的键值对调





# 其他数据结构

## 栈

```go
//使用slice手动实现栈
stack := make([]int,0,len)  //创建用作栈的slice

stack = append(stack, p)    //进栈语句

i = stack[len(stack)]       //出栈语句 
lps = lps[:len(lps)-1]
```

