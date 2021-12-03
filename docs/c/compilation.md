# 编译过程

**Preprocessing**

```c
gcc -E hello.c -o hello.i
```

1. 将`#define`语句宏进行替换
2. 处理`#if`,`#ifdef`等条件编译指令
3. 将`#include`语句替换为头文件内容
4. 删除所有注释
5. 添加行号和文件标识
6. 保留`#pragma`编译器指令

**compilation**

```
gcc -S hello.c -o hello.s
```

1. 词法分析
2. 语法分析
3. 语义分析
4. 生成汇编代码

**assembly**

```
gcc -c hello.c -o hello.o
```

**link**

```
gcc hello.c -o hello.exe
```

# 编译多个文件

```
gcc main.c module1.c module2.c
```

