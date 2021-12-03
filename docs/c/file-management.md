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
int chmod(const char *path, mode_t mode); // 权限修改
							//S_IRUSR S_IWUSR S_IXUSR S_IRWXU
							//S_IRGRP S_IWGRP S_IXGRP S_IRWXG
							//S_IROTH S_IWOTH S_IXOTH S_IRWXO

//sys
exit(0)      //退出程序并结束进程,0 正常退出,1 异常退出
```



`sys/stat.h`的`stat`函数用于获取文件信息，并存储在`struct stat`中。

```c
#include <sys/types.h>
#include <sys/stat.h>

//file attributes
int stat (const char*, struct stat * buf)   //返回文件的属性并存储在结构体变量buf中, return 0 if ok, -1 if err
    
struct stat
{
    dev_t       st_dev;     /* ID of device containing file -文件所在设备的ID*/
    ino_t       st_ino;     /* inode number -inode节点号*/
    mode_t      st_mode;    /* mode bits -模式位，即文件类型*/
    nlink_t     st_nlink;   /* number of hard links -链向此文件的连接数(硬连接)*/
    uid_t       st_uid;     /* user ID of owner -user id*/
    gid_t       st_gid;     /* group ID of owner - group id*/
    dev_t       st_rdev;    /* device ID (if special file) -设备号，针对设备文件*/
    off_t       st_size;    /* total size, in bytes -文件大小，字节为单位*/
    blksize_t   st_blksize; /* blocksize for filesystem I/O -系统块的大小*/
    blkcnt_t    st_blocks;  /* number of blocks allocated -文件所占块数*/
    time_t      st_atime;   /* time of last access -最近存取时间*/
    time_t      st_mtime;   /* time of last modification -最近修改时间*/
    time_t      st_ctime;   /* time of last inode change -最近i节点更改时间*/
};

#define S_IFMT   0170000    文件类型的位遮罩
#define S_IFSOCK 0140000    socket
#define S_IFLNK  0120000    符号连接
#define S_IFREG  0100000    一般文件
#define S_IFBLK  0060000    区块装置
#define S_IFDIR  0040000    目录
#define S_IFCHR  0020000    字符装置
#define S_IFIFO  0010000    先进先出

#define S_ISUID    04000    文件的(set user-id on execution)位
#define S_ISGID    02000    文件的(set group-id on execution)位
#define S_ISVTX    01000    文件的sticky位

#define S_IRUSR    00400    文件所有者具可读取权限
#define S_IWUSR    00200    文件所有者具可写入权限
#define S_IXUSR    00100    文件所有者具可执行权限
#define S_IRWXU    00700    文件所有者具有所有权限
#define S_IRGRP    00040    用户组具可读取权限
#define S_IWGRP    00020    用户组具可写入权限
#define S_IXGRP    00010    用户组具可执行权限
#define S_IRWXG    00070    用户组具有所有权限
#define S_IROTH    00004    其他用户具可读取权限
#define S_IWOTH    00002    其他用户具可写入权限
#define S_IXOTH    00001    其他用户具有所有权限
#define S_IRWXO    00007    其他用户具有所有权限

S_ISLNK(st_mode)	// 是否是一个连接.
S_ISREG(st_mode)	// 是否是一个常规文件.
S_ISDIR(st_mode)	// 是否是一个目录
S_ISCHR(st_mode)	// 是否是一个字符设备.
S_ISBLK(st_mode)	// 是否是一个块设备
S_ISFIFO(st_mode)	// 是否是一个FIFO文件.
S_ISSOCK(st_mode)	// 是否是一个SOCKET文件 
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

