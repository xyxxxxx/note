# fmt

```go
//普通输出
fmt.Print("Hello 711\n")

//格式化输出
fmt.Printf("Hello %d\n", 711)

//Println会在实参之间插入空格，并在结尾追加一个换行符
fmt.Println("Hello",711)

//Fprint可接受任何实现了io.Writer接口的对象作为第一个实参
fmt.Fprint(os.Stdout, "Hello ", 711, "\n")
           //os.Stdout os.Stderr
```

```go
//Stringer接口定义的String方法是Print对象的方法
//定义了该方法的类型的对象foo可以调用fmt.Print(foo)打印自己
type Stringer interface {
    String() string
}
```

