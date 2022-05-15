

[toc]

# 条件语句

## if-else 语句

```go
if condition1 {
	// do something	
} else if condition2 {
	// do something else	
} else {
	// catch-all or default
}

// `if`和`else`关键字必须和之后的`{`在同一行;如果你使用了else-if结构,则前段代码块的`}`必须和
// `else (if)`关键字在同一行.这两条规则都是编译器强制规定的.
```

如果分支数量较多，建议使用后面将要介绍的 switch 语句；如果需要根据变量的取值分多种情况讨论，那么也建议使用 switch 语句。

当 if 结构内有 break、continue、goto 或者 return 语句时，Go 代码的常见写法是省略 else 部分，例如两个分支返回不同的结果时通常使用以下写法：

```go
if condition {
	return x
}
return y
```

if 语句可以包含一个初始化语句（例如给一个变量赋值），这种写法具有固定的格式：

```go
if initialization; condition {
	// do something
}
```

例如：

```go
if value := process(data); value > max {     // 获取函数的返回值并立即作为判定条件
	...
}

if err := file.Chmod(0664); err != nil {     // 调用函数并立即检查是否发生错误
	fmt.Println(err)
	return err
}

if value, ok := readData(); ok {             // 调用函数并立即检查是否成功
    ...
}
```

注意初始化语句中使用简短方式 `:=` 声明的变量的作用域只存在于 if 结构中（如果使用 if-else 语句则也存在于 else 结构中）。如果变量在 if 结构之前就已经存在，那么在 if 结构中该变量原来的值会被隐藏。



## switch 语句

switch 语句的第一种形式是提供一个变量以及多个可能的取值逐一进行测试：

```go
switch var1 {         // `switch`关键字必须和之后的`{`在同一行
	case val1:
		...
	case val2, val3:  // 同时测试多个可能的取值
		...
	case val4:
    	...
		fallthrough   // 直接进入下一分支
	case val5: f()    // 当`var1==val4`时函数也会被调用
                      // 当代码只有一行时,可以直接放置在`case`语句之后
	default:          // 剩余情况
		...
}
```

* `var1` 和 `val1` , `val2` , `val3` 必须是相同的类型。
* 每一个 case 分支都是唯一的，从上至下逐一测试，直到匹配为止。
* 一旦成功地匹配到某个分支，在执行完相应代码后就会退出整个 switch 代码块。如果在执行完某个分支的代码后，还希望继续执行后续分支的代码，则可以使用 `fallthrough` 关键字来达到目的。



switch 语句的第二种形式是不提供任何变量，然后在每个 case 分支中测试不同的条件：

```go
switch {
	case condition1:
		...
	case condition2:
		...
    	fallthrough   // 测试下一分支
    case condition3:
		...
	default:
		...
}
```

+ 从上至下逐一测试，直到测试结果为 `true` 。
+ 一旦某个分支的测试结果为 `true`，在执行完相应代码后就会退出整个 switch 代码块。如果在执行完某个分支的代码后，还希望继续测试后续分支的条件，则可以使用 `fallthrough` 关键字来达到目的。
+ 这种形式看起来非常像链式的 if-else 语句，但是在测试条件非常多的情况下，提供了可读性更好的书写方式。



switch 语句同样也可以包含一个初始化语句：

```go
switch initialization; (var1) {
	// do something
}
```

例如：

```go
switch value := process(data); value {   // 获取函数的返回值并立即测试可能的取值
	...
}

switch value := process(data); {         // 获取函数的返回值并立即作为各分支的判定条件
	...
}
```

同样地，初始化语句中使用简短方式 `:=` 声明的变量的作用域只存在于 switch 结构中。如果变量在 switch 结构之前就已经存在，那么在 switch 结构中该变量原来的值会被隐藏。



## select 语句





# 循环语句

Go 语言中的循环语句只有 for 语句，但它要比其它语言中的 for 语句更加灵活。



## 基于计数器的循环

```go
for i := 0; i < 5; i++ {   // 初始化语句; 条件语句; 修饰语句
                           // `for`关键字必须和之后的`{`在同一行
	fmt.Println(i)
}
```

```go
// 同时使用多个计数器
for i, j := 0, N; i < j; i, j = i+1, j-1 {
    ...
}

// 循环嵌套
for i:=0; i<5; i++ {
	for j:=0; j<10; j++ {
		println(j)
	}
}
```



## 基于条件判断的循环（相当于while）

```go
package main

import "fmt"

func main() {
	var i int = 5

	for i >= 0 {
		i = i - 1
		fmt.Printf("The variable i is now: %d\n", i)
	}
}
```



## 无限循环

```go
for {
    
}

//或者
for true {
    
}
```



## for-range 结构

> 循环迭代字符串
>
> ```go
> package main
> 
> import "fmt"
> 
> func main() {
> 	str := "Go is a beautiful language!"
> 	fmt.Printf("The length of str is: %d\n", len(str))
> 	for ix :=0; ix < len(str); ix++ {
> 		fmt.Printf("Character on position %d is: %c \n", ix, str[ix])
> 	}
> 	str2 := "日本語"
> 	fmt.Printf("The length of str2 is: %d\n", len(str2))
> 	for ix :=0; ix < len(str2); ix++ {
> 		fmt.Printf("Character on position %d is: %c \n", ix, str2[ix])
> 	}
> }
> 
> /*
> The length of str is: 27
> Character on position 0 is: G 
> Character on position 1 is: o 
> Character on position 2 is:   
> …
> The length of str2 is: 9
> Character on position 0 is: æ 
> Character on position 1 is: � 
> Character on position 2 is: ¥ 
> …
> */
> ```
>
> ASCII 编码的字符占用 1 个字节，即每个索引都指向不同的字符，而非 ASCII 编码的字符（占有 2 到 4 个字节）不能单纯地使用索引来判断是否为同一个字符。
>
> 而使用 for-range 结构的迭代则可以解决以上问题：
>
> ```go
> package main
> 
> import "fmt"
> 
> func main() {
> 	str := "Go is a beautiful language!"
> 	fmt.Printf("The length of str is: %d\n", len(str))
> 	for pos, char := range str {
> 		fmt.Printf("Character on position %d is: %c \n", pos, char)
> 	}
> 	fmt.Println()
> 	str2 := "Chinese: 日本語"
> 	fmt.Printf("The length of str2 is: %d\n", len(str2))
> 	for pos, char := range str2 {
>  	fmt.Printf("character %c starts at byte position %d\n", char, pos)
> 	}
> 	fmt.Println()
> 	fmt.Println("index int(rune) rune    char bytes")
> 	for index, rune := range str2 {
>  	fmt.Printf("%-2d      %d      %U '%c' % X\n", index, rune, rune, rune, []byte(string(rune)))
> 	}
> }
> 
> /*
> The length of str is: 27
> Character on position 0 is: G 
> Character on position 1 is: o 
> Character on position 2 is:   
> …
> The length of str2 is: 18
> character C starts at byte position 0
> character h starts at byte position 1
> character i starts at byte position 2
> …
> character 日 starts at byte position 9
> character 本 starts at byte position 12
> character 語 starts at byte position 15
> 
> index int(rune) rune    char bytes
> 0       67      U+0043 'C' 43
> 1       104      U+0068 'h' 68
> 2       105      U+0069 'i' 69
> 3       110      U+006E 'n' 6E
> 4       101      U+0065 'e' 65
> 5       115      U+0073 's' 73
> 6       101      U+0065 'e' 65
> 7       58      U+003A ':' 3A
> 8       32      U+0020 ' ' 20
> 9       26085      U+65E5 '日' E6 97 A5
> 12      26412      U+672C '本' E6 9C AC
> 15      35486      U+8A9E '語' E8 AA 9E
> */
> ```





# break和continue

`break` 关键字的作用是退出循环，用于任何形式的 for 循环（计数器、条件判断等）。在 switch 或 select 语句中，`break` 关键字的作用是跳过整个代码块，执行后续的代码。

`continue` 关键字忽略剩余的循环体而直接进入下一次循环的过程，但执行之前依旧需要满足循环的判断条件。





# goto 语句

```go
package main

import "fmt"

func main() {

LABEL1:
	for i := 0; i <= 5; i++ {
		for j := 0; j <= 5; j++ {
			if j == 4 {
				continue LABEL1
                //break
			}
			fmt.Printf("i is: %d, and j is: %d\n", i, j)
		}
	}

}

//continue LABEL1 表示外侧循环continue
//break 表示结束内侧循环
//两者起到相同作用
//break LABEL1 则会结束外侧循环
/*
i is: 0, and j is: 0
i is: 0, and j is: 1
i is: 0, and j is: 2
i is: 0, and j is: 3
i is: 1, and j is: 0
i is: 1, and j is: 1
i is: 1, and j is: 2
i is: 1, and j is: 3
…
*/
```

使用标签和 goto 语句是不被鼓励的：它们很容易导致非常糟糕的程序设计，而且总有更加可读的替代方案来实现相同的需求。

如果必须使用 goto，应当只使用正序的标签（标签位于 goto 语句之后），但注意标签和 goto 语句之间不能出现定义新变量的语句，否则会导致编译失败。
