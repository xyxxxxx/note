Go 语言规范

# 注释

## 包注释

每个包都应包含一段包注释。对于包含多个文件的包，包注释只需出现在其中的任一文件中即可。

```go
/*
Package regexp implements a simple library for regular expressions.

The syntax of the regular expressions accepted is:

    regexp:
        concatenation { '|' concatenation }
    concatenation:
        { closure }
    closure:
        term [ '*' | '+' | '?' ]
    term:
        '^'
        '$'
        '.'
        character
        '[' [ '^' ] character-ranges ']'
        '(' regexp ')'
*/
package regexp
```

```go
// Package path implements utility routines for
// manipulating slash-separated filename paths.
package path
```

## 文档注释

任何顶级声明前面的注释都将作为该声明的文档注释。程序中每个可导出（首字母大写）的名称都应该有文档注释。文档注释最好是完整的句子，第一句应当以被声明的东西开头，并且是单句的摘要。

```go
// Compile parses a regular expression and returns, if successful,
// a Regexp that can be used to match against text.
func Compile(str string) (*Regexp, error) {
```

# 命名

## 包名

包名应该简洁明了，易于理解。包名应以小写的单个单词来命名，且不应使用下划线或驼峰记法。

## 获取器

```go
owner := obj.Owner()
if owner != user {
	obj.SetOwner(user)
}
```

## 接口名

只包含一个方法的接口应当以该方法的名称加上后缀-er 命名，如 `Reader`、`Writer`、`Formatter`、`CloseNotifier` 等。将字符串转换方法命名为 `String` 而非 `ToString`。

## 驼峰记法

此外，Go 使用驼峰记法。

# 控制结构

`if` 语句中不必要的 `else` 会被省略。

将 `if-else-if-else` 链写成一个 `switch` 更符合 Go 的风格。

# 函数

Go 函数的返回值（或称为结果形参）可被命名，并作为常规变量使用。一旦函数开始执行，它们就会被初始化为与其类型相应的零值。此命名不是强制性的，但能使代码更加简短清晰。

