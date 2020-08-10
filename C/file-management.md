# 文件管理

`unistd.h`包含了Unix的系统调用的封装，`unistd`即unix std的含义。

```c
#include <unistd.h>
//for func that returns int, return 0 if ok, -1 if err


//file operation
ssize_t read(int fd, void *buff, size_t n);   // 从文件描述符(流)fd读取n个字节写入字符数组buff的前n个索引
                                              // return number of bytes transferred, 0 if EOF, -1 if err
ssize_t write(int fd, void *buff, size_t n);  // 将字符数组buff的前n个索引(字节)写入文件描述符(流)fd
                                              // return number of bytes written, -1 if err
int unlink(const char* path);                 // 删除文件


//dir
char *getcwd(char *buf, size_t size); // 获取当前目录的绝对路径并存储到buf中,参数size为buf的长度
int chdir(const char *path); // 将path设定为当前目录
int rmdir(const char *path); // 将path目录删除


//access
int access(const char *, int);          // 权限检测
						//R_OK, W_OK, X_OK, F_OK
//#include <sys/types.h>
//#include <sys/stat.h>
int chmod(const char *path, mode_t mode); // 权限修改,
							//S_IRUSR S_IWUSR S_IXUSR S_IRWXU
							//S_IRGRP S_IWGRP S_IXGRP S_IRWXG
							//S_IROTH S_IWOTH S_IXOTH S_IRWXO

//sys
exit(0)      //退出程序并结束进程,0 正常退出,1 异常退出
```





# 目录管理

`dirent.h`包含了文件系统相关的结构体和函数。

```c
struct dirent
  {
#ifndef __USE_FILE_OFFSET64
    __ino_t d_ino;               //i节点号
    __off_t d_off;               //在目录文件中的偏移
#else
    __ino64_t d_ino;
    __off64_t d_off;
#endif
    unsigned short int d_reclen; //文件长度
    unsigned char d_type;        //文件类型
    char d_name[256];		     //文件名+'\0'
  };

typedef struct {
    int fd;
    dirent d;
} DIR;

DIR *opendir(char *dirname);
dirent *readdir(DIR *dfd);
void closedir(DIR *dfd);
```

```c
#include <dirent.h>
#include <sys/types.h>
#include <unistd.h>

int main()
{
    DIR * dfd;
    struct dirent * dp;
    int i;
    dfd = opendir("/etc/rc");    //根据路径打开目录,返回指向DIR结构体变量的指针
    while((dp = readdir(dfd)) != NULL) //读取目录,返回dirent结构体变量
    {
        printf("d_name : %s\n", dp->d_name);
    }
    closedir(dfd);
    return 0;
}
```



系统调用`stat`返回文件i节点中的所有信息

```c
int stat(char *, struct stat *);

char *name;
struct stat stbuf;
stat(name, &stbuf);   //将i节点的所有信息存储到stbuf中, return -1 if err
```

