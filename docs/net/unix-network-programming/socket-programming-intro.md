# Socket

Socket 是对 TCP/IP 协议族的一种封装，是应用层与TCP/IP协议族通信的中间软件抽象层。从设计模式的角度看来，Socket其实就是一个门面模式，它把复杂的TCP/IP协议族隐藏在Socket接口后面，对用户来说，一组简单的接口就是全部，让Socket去组织数据，以符合指定的协议。

Socket 还可以认为是一种网络间不同计算机上的进程通信的一种方法，利用三元组（ip地址，协议，端口）就可以唯一标识网络中的进程，网络中的进程通信可以利用这个标志与其它进程进行交互。

Socket 起源于 Unix ，Unix/Linux 基本哲学之一就是“一切皆文件”，都可以用“打开(open) –> 读写(write/read) –> 关闭(close)”模式来进行操作。因此 Socket 也被处理为一种特殊的文件。

# socket地址结构

## IPv4 socket地址结构——sockaddr_in

```c
struct sockaddr_in                 // POSIX定义
  {
//  uint8_t          sin_len       // 此结构体长度(16)     
    sa_family_t      sin_family    // 地址族
    in_port_t        sin_port;	   // Port number
    struct in_addr   sin_addr;     // Internet address
	char             sin_zero[8];  // unused, 总是将其置0
  };

typedef unsigned short int sa_family_t;  //地址族类型,即16位无符号整数

typedef uint16_t in_port_t;     //端口号类型,即16位无符号整数

struct in_addr                  //早期in_addr结构曾为多种结构的union
  {
    in_addr_t s_addr;
  };

typedef uint32_t in_addr_t;     //internet地址类型,即32位无符号整数

/*******************************************/
serv_addr.sin_family = AF_INET;
serv_addr.sin_addr.s_addr = inet_addr("127.0.0.1"); //inet_addr()将IP地址字符串转换为in_addr_t类型
serv_addr.sin_port = htons(1234); //htons()将主机字节序转换为网络字节序,即从小端模式转换为大端模式
                                  //例如端口18 在x86 CPU上实际存储为1200,而在报文中为0012

```

## 通用socket地址结构

```c
struct sockaddr
  {
//  uint8_t          sa_len    
    sa_family_t      sa_family       // 地址族
    char             sa_data[14];	 // Address data
  };
```

socket函数将指向该通用地址结构的指针作为参数之一，如

```c
int bind(int,struct sockaddr *, socklen_t)
```

因此指向特定协议族的socket地址结构都必须强制转型为通用地址结构的指针

## IPv6 socket地址结构——sockaddr_in6

```c
struct sockaddr_in6
{
//  uint8_t         sin6_len;          // 此结构体长度(28)
    sa_family_t     sin6_family        // 地址族
    in_port_t       sin6_port;		   // Port number
    uint32_t        sin6_flowinfo;     // 低位20 bit flow label, 高位12 bit保留
    struct in6_addr sin6_addr;         // Internet address
	uint32_t        sin6_scope_id;     // 标记地址的范围
}
```

## 新通用socket地址结构

```c
struct sockaddr_storage {
//  uint8_t         ss_len;
    sa_family_t     ss_family;
    char            ss_padding[118];
    uint32_t        ss_align;
}

//sockaddr_storage能满足最苛刻的对齐要求
//sockaddr_storage足够大以能够容纳任何地址结构
```

## socket地址结构比较

![111.png](https://i.loli.net/2020/08/07/4dqrwleMtP8J3EC.png)

由于Unix域结构和数据链路结构是可变长度的，因此除了传递地址结构的指针给套接字函数，也需要传递该地址结构的长度。

# 值－结果参数

`bind`,` connet`,` sendto`从进程到内核传递地址结构，如

```c
connect(sockfd, (struct sockaddr *)&serv, sizeof(serv));
```

`accept`,`recvfrom`,`getsockname`,`getpeername`从内核到进程传递地址结构，如

```c
struct sockaddr_un cli;
socklen_t len;

len-sizeof(sli);
getpeername(unixfd,(struct sockaddr *)&cli, &len);
```

传递地址结构长度的指针的原因是，对于可变长度的结构，内核会修改长度为实际长度

# 字节排序函数

考虑一个16位整数存储在2个字节中，内存中存储这两个字节有两种方法：小端字节序（little-endian）和大端（big-endian）字节序。

![Screenshot from 2020-08-14 10-13-25.png](https://i.loli.net/2020/08/14/CFsy2pXAxYKOvHT.png)

遗憾的是，这两种字节序之间没有标准可循。我们把某个给定系统所用的字节序称为主机字节序（host byte order），网络协议指定的字节序称为网络字节序（network byte order）。IP协议使用大端字节序来传送这些多字节整数。

两种字节序之间的转换使用以下函数：

```c
#include <netinet/in.h>

//返回网络字节序的值
uint16_t htons(uint16_t host16bitvalue);
uint32_t htonl(uint32_t host32bitvalue);
    
//返回主机字节序的值
uint16_t ntohs(uint16_t net16bitvalue);
uint32_t ntohl(uint32_t net32bitvalue);
```

# 字节操纵函数

```c
#include <strings.h>
void bzero(void *dest, size_t nbytes);
    //将目标字节串的前n个字节置0
    //通常用于把一个socket地址结构初始化为0
void bcopy(const void *src, void *dest, size_t nbytes);
    //将前n个字节从源字节串复制到目标字节串
int bcmp(const void *ptr1, const void *ptr2, size_t nbytes);
    //return -1, 0, 1 if <, =, >
```

> `const *`表示函数不会更改指针指向的变量，否则函数可能更改变量并作为返回值

```c
#include <string.h>
void memset(void *dest, int c, size_t nbytes);
    //将目标字节串的前n个字节置c
    //通常用于把一个socket地址结构初始化为0
void memcpy(void *dest, const void *src, size_t nbytes);
    //将前n个字节从源字节串复制到目标字节串
void memmove(void *dest, const void *src, size_t nbytes);
    //作用与memcpy相同,但是当源字符串与目标字符串重叠时此函数是安全的
int memcmp(const void *ptr1, const void *ptr2, size_t nbytes);
    //return -1, 0, 1 if <, =, >
```

# `inet_aton`, `inet_addr`和`inet_ntoa`函数

地址转换函数在ASCII字符串与网络字节序的二进制值之间转换网际地址。`inet_aton`, `inet_addr`和`inet_ntoa`函数在点分十进制串与它长度为32位的网络字节序二进制值之间转换IPv4地址。

```c
#include <arpa/inet.h>

int inet_aton(const char *strptr, struct in_addr *addrptr);
    //将strptr所指点分十进制串转换为32位网络字节序二进制值，并存储在addrptr指向的结构体变量
    //若字符串有效则返回1，否则返回0
in_addr_t inet_addr(const char *strptr);
    //若字符串有效则返回32位网络字节序二进制值，否则返回INADDR_NONE
    //不能处理255.255.255.255
char *inet_ntoa(struct in_addr inaddr);
    //将32位网络字节序二进制值转换成点分十进制串
    //返回指向一个点分十进制串的指针
```

# `inet_pton`, `inet_ntop`函数

`inet_pton`, `inet_ntop`函数对于IPv4地址和IPv6地址都适用。

```c
#include <arpa/inet.h>

int inet_pton(int family, const char *strptr, void *addrptr);
              //AF_INET, AF_INET6
    //从表达格式(strptr)转换到数值格式(addrptr)
    //return 1 if ok, 0 if illegal expression, -1 if err
const char *inet_ntop(int family, const void *addrptr, char *strptr, size_t len);
    //从数值格式(addrptr)转换到表达格式(strptr)
    //参数len是目标存储单元的大小;如果len不足以容纳表达格式结果,则返回一个空指针
    //return 
```

![Screenshot from 2020-08-14 10-10-15.png](https://i.loli.net/2020/08/14/V4bLcnKOxah9Fur.png)