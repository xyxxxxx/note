# 库

## <stdio.h>

参见IO



## <stdlib.h>

参见内存管理

```c
atof()     //str2double
atoi()     //str2int
atol()     //str2long
    
```

```c
rand()     //返回伪随机数0～RAND_MAX
srand(seed)//使用非负整数作为种子
```

```c
res=div(numer,denom)  //整数带余除法
printf("商=%d\n",res.quot);
printf("余数=%d\n",res.rem);

abs()       //返回绝对值
```



```c
system()   //将字符串交由环境(shell)去执行
abort()    //程序异常终止
exit(status)     //程序终止,status=0正常终止,=-1因错误而终止
    			 //exit亦会调用fclose关闭所有文件指针并flush
```



## <string.h>

```c
strcpy(str1,str2) /*str1用str2赋值*/
                  
strcat(str1,str2) //字符串拼接
strcmp(str1,str2) /*字符串比较*/
strchr(cs,c)      //查找cs中c首次出现的位置，返回指向该位置的指针
strlen(str)       /*字符串长度*/  
```

```c
memcpy(str1, str2, n) //复制str2(指针位置)的前n个字符到str1(指针位置)
memmove(str1, str2, n)    
memset(str, c, n)     //复制字符c到str指向的字符串的前n个字符,返回str
memcmp(str1, str2, n) //比较str1和str2的前n个字符
memchr(str, c, n)     //查找c在str的前n个字符中首次出现的位置
```



```

```



## <ctype.h>

```c
//判定变量类型
```



## <stdarg.h>

```c
//获取可变个数的参数

#include<stdarg.h>
#include<stdio.h>

int sum(int, ...);

int main(void)
{
   printf("10、20 和 30 的和 = %d\n",  sum(3, 10, 20, 30) );
   printf("4、20、25 和 30 的和 = %d\n",  sum(4, 4, 20, 25, 30) );

   return 0;
}

int sum(int num_args, ...)
{
   int val = 0;
   va_list ap;
   int i;

   va_start(ap, num_args);       //初始化指针ap
   for(i = 0; i < num_args; i++)  
   {
      val += va_arg(ap, int);    //返回一个实参并使指针移动一位
   }
   va_end(ap);                   //清理指针
 
   return val;
}
```





## <math.h>

