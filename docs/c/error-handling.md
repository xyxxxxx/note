# 错误处理

```c
//errno表示错误代码，int类型
//strerror通过参数errno返回错误信息
stderror(errno)
//perror输出参数字符串和当前错误信息到标准错误输出stderr
perror(const char *s) //等价于
fprintf(stderr,"%s: %s\n",s,err_msg)    
```

```c
    #define EPERM 1 /* Operation not permitted */
    #define ENOENT 2 /* No such file or directory */
    #define ESRCH 3 /* No such process */
    #define EINTR 4 /* Interrupted system call */
    #define EIO 5 /* I/O error */
    #define ENXIO 6 /* No such device or address */
    #define E2BIG 7 /* Argument list too long */
    #define ENOEXEC 8 /* Exec format error */
    #define EBADF 9 /* Bad file number */
    #define ECHILD 10 /* No child processes */
    #define EAGAIN 11 /* Try again */
    #define ENOMEM 12 /* Out of memory */
    #define EACCES 13 /* Permission denied */
    #define EFAULT 14 /* Bad address */
    #define ENOTBLK 15 /* Block device required */
    #define EBUSY 16 /* Device or resource busy */
    #define EEXIST 17 /* File exists */
    #define EXDEV 18 /* Cross-device link */
    #define ENODEV 19 /* No such device */
    #define ENOTDIR 20 /* Not a directory */
    #define EISDIR 21 /* Is a directory */
    #define EINVAL 22 /* Invalid argument */
    #define ENFILE 23 /* File table overflow */
    #define EMFILE 24 /* Too many open files */
    #define ENOTTY 25 /* Not a typewriter */
    #define ETXTBSY 26 /* Text file busy */
    #define EFBIG 27 /* File too large */
    #define ENOSPC 28 /* No space left on device */
    #define ESPIPE 29 /* Illegal seek */
    #define EROFS 30 /* Read-only file system */
```

```c
#include <stdio.h>
#include <errno.h>
#include <string.h>
 
extern int errno ;          //声明外部变量错误代码errno
 
int main ()
{
   FILE * pf;
   int errnum;
   pf = fopen ("unexist.txt", "rb");
   if (pf == NULL)
   {
      errnum = errno;
      fprintf(stderr, "错误号: %d\n", errno);
      perror("通过 perror 输出错误");
      fprintf(stderr, "打开文件错误: %s\n", strerror( errnum ));
   }
   else
   {
      fclose (pf);
   }
   return 0;
}

```
