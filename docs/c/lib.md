# 通用规则

+ 以下划线开始的函数，结构体，类型，变量名表示仅在库中被使用，相当于Java的private
+ 





# stdio.h

参见IO





# stdlib.h

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





# string.h

```c
char *strcpy(char *dest, const char *src) //将src指向的字符串复制到dest指向的字符串，dest指向字符串的原有内容被丢弃，返回指针dest
//可能引起内存上溢
    
char *strcat(char *dest, const char *src) //将src指向的字符串追加到dest指向的字符串的结尾,返回指针dest
//可能引起栈溢出    
    
int strcmp(const char *str1, const char *str2)　//比较字符串, return negative, 0, positive if str1 <, =, > str2

char *strchr(const char *str, int ch) //查找str指向的字符串中字符ch首次出现的位置，返回指向该位置的指针，查找失败则返回NULL
    
size_t strlen(const char *str) //返回str指向的字符串的长度，不包括\0
```

```c
void *memcpy(void *dest, const void *src, size_t n) //将src指向的内存块的前n个字节复制到dest指向的内存块的前n个字节，dest指向的内存块的其它字节不变，返回指针dest

void *memmove(void *dest, const void *src, size_t n) //与memcpy功能相同，但如果内存块存在区域重叠时是比memcpy更安全的方法

void *memset(void *str, int c, size_t n) //复制字符c到str指向的内存块的前n个字节，dest指向的内存块的其它字节不变,返回指针str

int memcmp(const void *str1, const void *str2, size_t n) //比较str1指向的内存块和str2指向的内存块的前n个字节的大小
    
void *memchr(const void *str, int c, size_t n) //查找字符c在str指向的内存块的前n个字节中首次出现的位置
```





# ctype.h

```c
//判定变量类型


isdigit(c)        // 0-9
    
tolower(c)        // A-Z -> a-z    
```





# stdarg.h

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



# signal.h

```c
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <signal.h>

void sighandler(int);

int main()
{
   signal(SIGINT, sighandler); //接收信号SIGINT
                               //内核向进程发出信号，进程捕捉信号(发生软件中断)，并执行信号处理函数，之后进程继续执行

   while(1) 
   {
      printf("开始休眠一秒钟...\n");
      sleep(1);
   }

   return(0);
}

void sighandler(int signum) //信号处理函数
{
   printf("捕获信号 %d，跳出...\n", signum);
   exit(1);
}
```





# math.h





# unistd.h, sys/stat.h, sys/mman.h

参见[文件管理]()





# dirent.h

参见[目录管理]()