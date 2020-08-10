UNIX系统通过一系列的系统调用为应用程序提供服务。



# 系统调用

## 文件描述符

在UNIX系统中，所有的I/O都通过读写文件完成。所有的外部设备，包括显示器，键盘，socket，etc，都被操作系统视为文件，这意味着有一个单独的接口负责处理程序和所有外部设备的通信。

文件描述符是一个非负小整数，用于确定一个文件，所有关于此文件的信息都由系统维护，应用程序只需要通过文件描述符来操作文件。





## 底层I/O——`Read, Write`

`unistd.h`的函数`read`和`write`通过系统调用`read`和`write`进行I/O。

```c
int n_read = read(int fd, void *buff, size_t n);　　　// 从文件描述符(流)fd读取n个字节写入字符数组buff的前n个索引
                                                     // return number of bytes transferred, 0 if EOF, -1 if err
int n_written = write(int fd, void *buff, size_t n); // 将字符数组buff的前n个索引(字节)写入文件描述符(流)fd
                                                     // return number of bytes written, -1 if err
```

```c
#include <string.h>
#include <unistd.h>

int main() // copy input to output
{
   char buf[100];
   int n;

   while ((n=read(0, buf, 100)) > 0){
      write(1, buf, strlen(buf));
   }

   return 0;
}
```





## `Open`, `Creat`, `Close`, `Unlink`

```c
int open(char *name, int flags, int perms); //系统调用open用于打开文件
/*  参数flags指文件打开类型
　　　　　O_RDONLY 只读
　　　　　O_WRONLY 只写
     　　O_RDWR 读写
    
    参数perms通常置0
*/

int fd;
fd = open("file.txt", O_RDONLY, 0); //返回文件描述符, -1 if err


int creat(char *name, int perms); //系统调用creat用于创建文件,会覆盖已有文件
/*  
    参数perms指文件权限
*/

fd = creat("file.txt", 0755);
```

一个程序能同时打开的文件数量上限在20个左右，因此必要时需要复用文件描述符。系统调用`close`用于切断文件描述符和文件之间的连接，并释放文件描述符。系统调用`exit`或者返回`main`函数也会关闭所有文件。

系统调用`unlink`用于删除文件。





## 随机访问——`lseek`

系统调用`lseek`设定文件的位置指针从指定位置偏移。

```c
long lseek(int fd, long offset, int origin);
/*  参数origin表示指定位置
　　　　　0 文件起始位置
　　　　　1 文件中间位置
　　　　　2 文件结束位置

*/

lseek(fd, 0L, 2);  //置为文件结束位置
lseek(fd, 0L, 0);  //置为文件起始位置,即rewind
```





# 实例

## `fopen`

标准库中的文件用文件指针`FILE *fp`而非文件标识符`int fd`表示。`FILE`结构体包含了一个文件的若干信息：指向缓冲区的指针，缓冲区的字符数，指向缓冲区中下一个字符位置的指针，文件描述符，读写模式、错误状态的标识等。



## `ls`

UNIX文件系统中，目录就是一个包含了文件名和i节点（？）的特殊文件。i节点包含了一个文件除文件名以外的所有信息。

```c
#include <string.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <dirent.h>

#define MAX_PATH 1024

void fsize(char *);
void dirwalk(char *, void (*fcn)(char *));


int main()
{
   fsize("/home/xyx/codes");

   return 0;
}

void fsize(char *name) //print size of file
{
   struct stat stbuf;

   if (stat(name, &stbuf) == -1)
   {
      fprintf(stderr, "fsize: cannot access %s\n", name);
      return;
   }
   if ((stbuf.st_mode & S_IFMT) == S_IFDIR)
   {
      dirwalk(name, fsize); //if dir, call dirwalk to fsize all files in it
   }
   printf("%8ld %s\n", stbuf.st_size, name);
}

void dirwalk(char *dir, void (*fcn)(char *)){
   char name[MAX_PATH];
   struct dirent *dp;
   DIR *dfd;

   if ((dfd = opendir(dir)) == NULL){
      fprintf(stderr, "dirwalk: cannot open %s\n", dir);
      return;
   }
   while ((dp = readdir(dfd)) != NULL){
      if (strcmp(dp->d_name, ".") == 0 || strcmp(dp->d_name, "..") == 0){
         continue;
      }
      if (strlen(dir)+strlen(dp->d_name)+2 > sizeof(name)){
         fprintf(stderr, "dirwalk: name %s/%s is too long\n", dir, dp->d_name);
      }
      else{
         sprintf(name, "%s/%s", dir, dp->d_name);
         (*fcn)(name);
      }
      closedir(dfd);
   }
}
```



## 内存分配





# 附录：系统调用与标准库函数的对应

| 系统调用        | 标准库函数 |
| --------------- | ---------- |
| `read`          | `fread`    |
| `write`         | `fwrite`   |
| `open`, `creat` | `fopen`    |
| `close`         | `fclose`   |
| `unlink`        | `remove`   |
| `lseek`         | `fseek`    |



