# make

[make](https://zh.wikipedia.org/zh-cn/Make) 是最常用的构建工具，诞生于 1977 年，主要用于 C 语言的项目。但是实际上，任何只要某个文件有变化就需要重新构建的项目，都可以用 make 构建。

## 引入

假设工作目录下有文件 `b` 和 `c`，想要拼接这两个文件为一个新文件，可以执行命令

```shell
$ cat b c > a
```

将上述规则写到一个名为 `makefile` 的文件中

```makefile
a: b c
	cat b c > a
```

再执行命令

```shell
$ make a
```

即可由 make 自动完成文件 `a` 的构建。

## makefile 文件格式

makefile 文件包含了所有构建规则。

### 规则

makefile 文件由一系列规则（rule）构成，每条规则的形式如下

```
<target>: <prerequisites>
[Tab]<commands>
```

注意命令之前是 tab 字符而非空格。

### 目标

目标（target）通常是文件名，指明 `make` 命令所要构建的对象。目标可以是多个文件名，之间用空格分隔。

目标也可以是某个操作的名字，称为伪目标（phony target）

```makefile
clean:
	rm *.o
```

`clean` 目标并非构建名为 `clean` 的文件，而是执行清理操作。如果当前目录中正好有一个文件叫做 `clean`，那么这个命令不会执行，因为 make 发现 `clean` 文件已经存在，就认为没有必要重新构建了，就不会执行指定的 `rm` 命令。为了避免这种情况，可以明确声明 clean 是伪目标

```makefile
.PHONY: clean
clean:
	rm *.o temp
```

如果执行 `make` 命令时没有指定目标，那么默认执行 makefile 文件的第一个目标。

### 前置条件

（描述存在错误）前置条件（prerequisite）通常是一组文件名，之间用空格分隔。它指定了目标是否重新构建的判断条件：只要有一个前置文件不存在，或者有过更新（即前置文件的最后一次修改时间戳比目标的时间戳新），目标就需要重新构建。

```makefile
result.txt: source.txt
	cp source.txt result.txt
```

上面的代码中，构建 `result.txt` 的前置条件是 `source.txt`。如果当前目录中 `source.txt` 已经存在，那么 `make result.txt` 可以正常运行，否则必须再写一条规则，来生成`source.txt`。

```makefile
source.txt:
	echo "new source" > source.txt
```

上面的代码中，`source.txt` 后面没有前置条件，意味着它跟其他文件都无关。只要这个文件不存在，调用 `make source.txt` 都会生成它。

```shell
$ make result.txt
$ make result.txt
```

连续执行两次 `make result.txt`。第一次执行会先新建`source.txt`，然后再新建`result.txt`。第二次执行 make 发现 `source.txt` 没有变动（时间戳早于 `result.txt`），就不会执行任何操作，`result.txt` 也不会重新生成。

如果需要一次生成多个文件，往往采用下面的写法：

```makefile
source: file1 file2 file3
```

其中 `source` 是一个伪目标，只有三个前置文件，这些前置文件的构建规则再分别定义。这样执行 `make source`命令，就会一次性生成`file1,file2,file3` 三个文件。

### 命令

命令（command）表示如何更新目标文件，由一行或多行的 Shell 命令组成。它是构建目标的具体指令，它的运行结果通常就是生成目标文件。

每行命令之前必须有一个 tab 键。如果想用其他键，可以用内置变量 `.RECIPEPREFIX` 声明。

```makefile
.RECIPEPREFIX = >
all:
>echo hello, world
```

上面 `.RECIPEPREFIX` 声明为 `>`，因此每一行命令的起首变成了 `>`。

需要注意的是，每行命令在一个单独的 shell 中执行。

```makefile
var-lost:
	export foo=bar
	echo "foo=[$$foo]"
```

执行 `make var-lost`发现取不到`foo` 的值，因为两行命令在两个不同的进程执行。解决办法是将两行命令写在一行，中间用分号分隔

```makefile
var-kept:
	export foo=bar; echo "foo=[$$foo]"
```

或者在换行符前加反斜杠转义

```makefile
var-kept:
    export foo=bar; \
    echo "foo=[$$foo]"	
```

或者加上 `.ONESHELL:` 命令

```makefile
.ONESHELL:
var-kept:
    export foo=bar; 
    echo "foo=[$$foo]"
```

## makefile 文件语法

### 注释

`#` 表示注释。

```
# comment

@# comment not printed

@echo "hello"
```

### 回声

正常情况下，make 在执行每条命令之前会先打印命令。

```makefile
test:
    # 这是测试
```

执行上面的规则

```bash
$ make test
# 这是测试
```

在命令或注释的前面加上 `@`，就可以关闭回声

```makefile
test:
    @# 这是测试
```

### 模式匹配

make 命令允许对文件名进行类似正则运算的匹配，使用匹配符 `%`。比如，假定当前目录下有 `f1.c` 和 `f2.c` 两个源码文件，需要将它们编译为对应的对象文件，那么

```makefile
%.o: %.c
```

等同于

```makefile
f1.o: f1.c
f2.o: f2.c
```

使用匹配符可以将大量同类型的文件只用一条规则就完成构建。

### 变量和赋值符

makefile 允许使用等号自定义变量

```makefile
txt = Hello  # str
test:
	@echo $(txt)
```

调用环境变量时需要使用两个 `$` 符号

```makefile
test:
	@echo $$HOME
```

也可以使用变量为变量赋值

```makefile
v1 = $(v2)
```

上面代码中，变量 `v1` 的值是另一个变量 `v2`。这时会产生一个问题，`v1` 的值到底在定义时扩展（静态扩展），还是在运行时扩展（动态扩展）？如果 `v2` 的值是动态的，这两种扩展方式的结果可能会差异很大。

为了解决类似问题，Makefile 一共提供了四个赋值运算符（`=、:=、？=、+=`）

```makefile
VARIABLE = value
# 在执行时扩展，允许递归扩展。

VARIABLE := value
# 在定义时扩展。

VARIABLE ?= value
# 只有在该变量为空时才设置值。

VARIABLE += value
# 将值追加到变量的尾端。
```

### 内置变量

make 命令提供一系列内置变量，比如，`$(CC)` 指向当前使用的编译器,`$(MAKE)` 指向当前使用的 Make 工具。这主要是为了跨平台的兼容性，详细的内置变量清单见[手册](https://www.gnu.org/software/make/manual/html_node/Implicit-Variables.html)。

```makefile
output:
    $(CC) -o output input.c
```

### 自动变量

`$@` 指代当前目标,即 make 命令当前构建的那个目标,比如 `make foo`的`$@`就指代`foo`。

```makefile
a.txt b.txt: 
    touch $@
```

等同于

```makefile
a.txt:
    touch a.txt
b.txt:
    touch b.txt
```

`$<` 指代第一个前置条件。比如,规则为 `t:p1 p2`,那么`$<`就指代`p1`。

```makefile
a.txt: b.txt c.txt
    cp $< $@ 
```

等同于

```makefile
a.txt: b.txt c.txt
    cp b.txt a.txt 
```

`$?` 指代比目标更新的所有前置条件，之间以空格分隔。

`$^` 指代所有前置条件，之间以空格分隔。

`$(@D)` 和 `$(@F)` 分别指向 `$@` 的目录名和文件名。

`$(<D)` 和 `$(<F)` 分别指向 `$<` 的目录名和文件名。

### 条件和循环结构

makefile 使用 Bash 语法，完成判断和循环。

```makefile
ifeq ($(CC),gcc)
	libs=$(libs_for_gcc)
else
	libs=$(normal_libs)
endif
```

```makefile
LIST = one two three
all:
	for i in $(LIST); do \
		echo $(i); \
	done
```

## 函数

makefile 还可以使用函数，格式如下。

```makefile
$(function arguments)
# 或
${function arguments}
```

```
srcfiles := $(shell echo src/{00..99}.txt)
```

### subst

`subst` 函数用来文本替换，格式如下。

```makefile
$(subst from,to,text)
```

下面的例子将字符串"feet on the street"替换成"fEEt on the strEEt"。

```makefile
$(subst ee,EE,feet on the street)
```

### patsubst

`patsubst` 函数用于模式匹配的替换，格式如下。

```makefile
$(patsubst pattern,replacement,text)
```

下面的例子将文件名 `x.c.c,bar.c` 替换成 `x.c.o,bar.o`。

```makefile
$(patsubst %.c, %.o, x.c.c bar.c)
```

## 实例

### 执行多个目标

```makefile
.PHONY: cleanall cleanobj cleandiff

cleanall : cleanobj cleandiff
        rm program

cleanobj :
        rm *.o

cleandiff :
        rm *.diff
```

上面代码可以调用不同目标，删除不同后缀名的文件，也可以调用一个目标（cleanall），删除所有指定类型的文件。
