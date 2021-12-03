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
#include <fcntl.h>
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



`opendir`打开目录，通过系统调用`fstat`验证文件是目录，分配一个目录结构体并记录信息。

```c
int fstat(int fd, struct stat *);

DIR *opendir(char *dirname)
{
    int fd;
    struct stat stbuf;
    DIR *dp;
    
    if ((fd = open(dirname, O_RDONLY, 0)) == -1
      || fstat(fd, &stbuf) == -1
      || (stbuf.st_mode & S_IFMT) != S_IFDIR
      || (dp = (DIR *) malloc(sizeof(DIR)) == NULL)
        return NULL;
    dp->fd = fd;
    return dp;    
}
```



`closedir`关闭目录并释放空间。

```c
void closedir(DIR *dp)
{
    if (dp){
        close(dp->fd);
        free(dp);
    }
}
```



`readdir`使用`read`读取



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
      if (strcmp(dp->d_name, ".") == 0 || strcmp(dp->d_name, "..") == 0){ //
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

> 内存的分配和回收方式参见[内存]()

`malloc`负责管理一部分内存，其中的空闲块被维护成一个空闲块链表，其中每一块都有一个首部，包含指向下一块的指针以及本块大小，首部被定义为一个联合：

```c
typedef long Align;　// 4或8字节

union header{
    struct{
        union header *ptr; //指向下一块的指针
        unsigned size;     //块大小(unit)
    } s;
    Align x;        // 用于占用固定大小
}

typedef union header Header;
```



为了便于对齐，首部被定义为固定大小，所有块都是该大小的整数倍。

`malloc`为需求的内存大小分配刚好能够容纳之的首部大小单元个数，并增加一个首部单元（也计入块大小）。`malloc`返回一个指向空闲块的指针。

```c
static Header base; //空块的首部
static Header *freep = NULL; //freep指向当前空闲块的首部

void *malloc(unsigned nbytes)
{
    Header *p, *prevp;
    Header *morecore(unsigned); //向操作系统请求更多内存
    unsigned nunits;
    
    nunits = (nbytes+sizeof(Header)-1)/sizeof(Header)+1; // 4B -- 2unit, 5B -- 3unit
    if ((prevp = freep) == NULL){ //初始化freep和prevp
        base.s.ptr = freep = prevp = &base; //利用空块的首部
        base.s.size = 0;
    }
    for (p = prevp->s.ptr; ; prevp = p, p = p->s.ptr){ //(画图理解)
        if (p->s.size >= nunits){ //first fit,如果块p的大小足够
            if (p->s.size == nunits) //如果大小正好
                prevp->s.ptr = p->s.ptr; //直接从链表删除块p
            else {
                p->s.size -= nunits; //将块p的尾端分配
                p += p->s.size;
                p->s.size = nunits;
            }
            freep = prevp;
            return (void *)(p+1); //返回的空闲块依然有首部占据第一个unit
        }
        if (p == freep)
            if ((p = morecore(nunits)) == NULL)
                return NULL; //请求内存失败,返回NULL
    }
}
```



`morecore`向操作系统请求更多内存，并且调用`free`将该空闲块插入链表。

```c
#define NALLOC 1024

static Header *morecore(unsigned nu)
{
    char *cp, *sbrk(int);
    Header *up;
    
    if (nu < NALLOC) //至少请求1024个单元,因为向系统请求内存的操作花费较大
        nu = NALLOC;
    cp = sbrk(nu * sizeof(Header)); //返回指向额外nu单元内存的指针
                                    //系统调用sbrk返回指向额外n字节内存的指针
                                    //return -1 if no space (in fact NULL is better)
    if (cp == (char *)-1)           //如果给指针赋值-1,必须强制转型以进行比较
        return NULL;
    up = (Header *)cp;              //返回块的首部指针
    up->s.size = nu;                //写入块长度
    free((void *)(up+1));           //传入指向块的空闲位置的指针
    return freep;                   //返回更新后的freep
}
```



`free`从`freep`开始扫描空闲块链表，查找空闲块插入链表的位置（地址大小序）。如果插入链表的块与空闲块相邻，则合并。

```c
void free(void *ap) //释放ap指向的块，即将块放入空闲链表中
{
    Header *bp, *p;
    
    bp = (Header *)ap-1; //返回块的首部指针;为什么传入(void *)(up+1)？
    for (p = freep; !(bp > p && bp < p->s.ptr); p = p->s.ptr) //直到块bp在2空闲块p和p->s.ptr之间
        if (p >= p->s.ptr && (bp > p || bp < p->s.ptr)) //或者块bp在所有空闲块的前方或后方
            break;
    
    if (bp+bp->s.size == p->s.ptr){ //如果块bp的后方紧邻空闲块p->s.ptr
        bp->s.size += p->s.ptr->s.size; //将p->s.ptr并入bp
        bp->s.ptr = p->s.ptr->s.ptr;
    } else
        bp->s.ptr = p->s.ptr; //否则bp->s.ptr指向p->s.ptr
    
    if (p+p->s.size == bp){ //如果块bp的前方紧邻空闲块p
        p->s.size += bp->s.size; //将bp并入p
        p->s.ptr = bp->s.ptr;
    } else
        p->s.ptr = bp; //否则p->s.ptr指向bp
    
    freep = p; //将p设为当前空闲块
}
```









# 附录：系统调用与标准库函数的对应

| 系统调用        | 标准库函数 |
| --------------- | ---------- |
| `read`          | `fread`    |
| `write`         | `fwrite`   |
| `open`, `creat` | `fopen`    |
| `close`         | `fclose`   |
| `unlink`        | `remove`   |
| `lseek`         | `fseek`    |



