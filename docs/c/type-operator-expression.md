# 运算符

## 算数运算符

```c
+  -  *  /  %  ++  --
```

+ `++i`表示`i+=1`再执行该语句，`i++`表示执行该语句再`i+=1`



## 关系运算符

```c
==  !=  >  <  >=  <=
```

+ 关系运算语句返回数值1如果结果为真，返回0如果为假，例如

  ```c
  d = c >= '0' && c <= '9'  // d = 1 if c is digit, 0 if not
  ```

  

## 逻辑运算符

```c
&&  ||  !    
```

+ 若逻辑语句的值已经确定，则立即结束该语句的运算
+ `!0 == 1`，`!n == 0`，其中`n != 0`



## 位运算符

```c
&  |  ^异或  ~取反  <<左移  >>右移
```

+ 将后6位置零的最好方法是：`x & ~077`，因为不必考虑`x`的位数



## 赋值运算符

```c
=  +=  -=  *=  /=  %=  <<=  >>=  &=  ^=  |=  
```



## 杂项运算符

```c
sizeof()  &  *  ?:
```



## 三元运算符

```c
a?b:c
```

+ `(n > 0) ? f : n`返回结果的类型被转换为`float`



## 运算顺序

| 类别       | 运算符                                       | 结合性   |
| :--------- | :------------------------------------------- | :------- |
| 后缀       | `()  []  ->  .  ++  --`                      | 从左到右 |
| 一元       | `!  ~  ++  --  (type)  *  &  sizeof`         | 从右到左 |
| 乘除       | `*  /  %`                                    | 从左到右 |
| 加减       | `+  -`                                       | 从左到右 |
| 移位       | `<<  >>`                                     | 从左到右 |
| 关系       | `<  <=  >  >=`                               | 从左到右 |
| 相等       | `==  !=`                                     | 从左到右 |
| 位与 AND   | `&`                                          | 从左到右 |
| 位异或 XOR | `^`                                          | 从左到右 |
| 位或 OR    | \|                                           | 从左到右 |
| 逻辑与 AND | `&&`                                         | 从左到右 |
| 逻辑或 OR  | \|\|                                         | 从左到右 |
| 条件       | `?:`                                         | 从右到左 |
| 赋值       | =  +=  -=  *=  /=  %=  >>=  <<=  &=  ^=  \|= | 从右到左 |
| 逗号       | `,`                                          | 从左到右 |

+ 从左到右运算

+ 按照优先级排序，同级别按照结合性

+ 当两次运算不能确定先后顺序时，结果取决于编译器，例如：

  ```c
  printf("%d %d\n", ++n, power(2,n)); //WRONG
  
  a[i] = i++; //WRONG
  ```





# 类型

## 数据类型

| 类型           | 存储大小    | 值范围                                               |
| :------------- | :---------- | :--------------------------------------------------- |
| char           | 1 字节      | -128 到 127 或 0 到 255（取决于机器）                |
| unsigned char  | 1 字节      | 0 到 255                                             |
| signed char    | 1 字节      | -128 到 127                                          |
| int            | 2 或 4 字节 | -32,768 到 32,767 或 -2,147,483,648 到 2,147,483,647 |
| unsigned int   | 2 或 4 字节 | 0 到 65,535 或 0 到 4,294,967,295                    |
| short          | 2 字节      | -32,768 到 32,767                                    |
| unsigned short | 2 字节      | 0 到 65,535                                          |
| long           | 4 或 8 字节 | -2,147,483,648 到 2,147,483,647 或 ...               |
| unsigned long  | 4 字节      | 0 到 4,294,967,295                                   |
| float          | 4 字节      | 1位符号, 8位指数, 23位小数                           |
| double         | 8 字节      | 1位符号, 11位指数, 52位小数                          |

+ 使用`float`的原因：减少大型数组的存储大小；单精度浮点运算节省时间



### 类型转换

+ 计算转换： `char/short` → `int` → `long` → `float` → `double` →`long double`
+ 赋值转换：
  + `int` ←→`char`，`int`在 0-127 范围外时 ？
  + `float` → `int`时丢失小数部分
  + `double` → `float`时丢失精度或截取部分，取决于具体实现

+ 强制转换：`sqrt((double) n)`

+ `char`本身是小整数类型，因此可以自由做算术运算，例如`'1'-'0'==1`，`'C'-'A'+'a'=='c'`

+ 缺少函数原型时，`char`和`short`变成`int`，`float`变成`double`，因此通常声明函数参数为`int`或`double`

  



## 自定义类型

```c
typedef float real;  //使用real表示float类型
typedef int num[100];//声明num为整数数组类型
typedef char *string;//使用string表示字符指针类型
```



## 枚举类型

```c
enum month{ Jan, Feb, Mar, ..., Dec};//定义month类型的变量仅能有12个取值,赋值为0,1,...,11
month < Jul //可以比较
month1=(enum month)3;//赋值4月    
```



## 其它类型

|        |                                                        |
| ------ | ------------------------------------------------------ |
| size_t | unsigned int 类型，用来表示数据结构大小，sizeof 返回值 |
| FILE   | 存储文件流信息的类型                                   |
|        |                                                        |





# 特殊语法

## #

```c
//file inclusion
#include <stdio.h>   //预处理时此语句被替换为头文件内容

//macro substitution
#define pi 3.1415926 //将token(pi)替换为text(3.1415926)
#define forever for(;;)
#define max(A,B) ((A)>(B)?(A):(B)) //带参数的宏
                     //和函数相比,宏只有替换的功能
                     //用宏替代函数可以避免函数运行的开支
#define dprint(expr) printf(#expr " = %g\n",expr) //#expr被替换为"x/y"
#define paste(front,back) front ## back //拼接front和back,即frontback
#undef getchar       //取消宏定义				

//conditional inclusion
#if SYSTEM == SYSV
	#define HDR "stsv.h"
#elif SYSTEM == BSD
	#define HDR "bsd.h"
#elif SYSTEM == MSDOS
	#define HDR "msdos.h"
#else
	#define HDR "default.h"
#endif
#include HDR

#ifndef HDR    //如果未定义宏
#define HDR
/*contents of hdr.h*/
#endif

```



## const

```c
//const变量只能在初始化时被赋值
const double e = 2.71828182845905;
const char msg[] = "warning:";

//参数声明const表示函数不会更改之
int strlen(const char[]);
```



## extern

> https://blog.csdn.net/xingjiarong/article/details/47656339

```c
//extern关键字声明外部变量
//如果外部变量在同一源文件的首部声明，则省略extern
//如果外部变量在另一源文件声明
```



## static & automatic

```c
#include <stdio.h>
 
void test()
{
    auto a = 0;               //定义自动存储类型变量
    static int b = 3;         //定义静态存储类型变量
    a++;
    b++;
    printf("%d\n", a);        //输出a
    printf("%d\n", b);        //输出b
}

int main()
{
    int i;
    for (i = 0; i < 3; i++)
    {
        test();              //调用test函数
    }
    return 0;
}
//输出 1 4 1 5 1 6
```

