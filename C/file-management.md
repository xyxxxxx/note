# 文件管理

`unistd.h`包含了Unix的系统调用的封装，`unistd`即unix std的含义。

```c
#include <unistd.h>
//for func that returns int, return 0 if ok, -1 if err


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


//rw
ssize_t read(int fd, void *buff, size_t n);   // 从文件描述符(流)fd读取n个字节写入字符数组buff的前n个索引
ssize_t write(int fd, void *buff, size_t n);  // 将字符数组buff的前n个索引(字节)写入文件描述符(流)fd

```

`dirent.h`

```c
#include <dirent.h>
#include <sys/types.h>
#include <unistd.h>

int main()
{
    DIR * dir;                   //目录流结构体
    struct dirent * ptr;
    int i;
    dir = opendir("/etc/rc.d");  //根据路径打开目录,返回指向目录结构变量的指针
    while((ptr = readdir(dir)) != NULL) //读取目录流,返回dirent结构变量
    {
        printf("d_name : %s\n", ptr->d_name);
    }
    closedir(dir);
    return 0;
}

struct dirent                    //文件信息结构体
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
    char d_name[256];		     //文件名
  };
```

