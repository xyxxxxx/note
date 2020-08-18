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

