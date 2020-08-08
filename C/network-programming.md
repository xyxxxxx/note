![](https://img-blog.csdn.net/20180205104507262?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvY2FpbmlhbzAwMDAwMQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)



# Socket编程简介

## socket地址结构

### IPv4 socket地址结构——sockaddr_in

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

### 通用socket地址结构

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

### IPv6 socket地址结构——sockaddr_in6

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

### 新通用socket地址结构

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

### socket地址结构比较

![111.png](https://i.loli.net/2020/08/07/4dqrwleMtP8J3EC.png)

由于Unix域结构和数据链路结构是可变长度的，因此除了传递地址结构的指针给套接字函数，也需要传递该地址结构的长度。



## socket函数

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



## 字节排序函数

考虑一个16位整数存储在2个字节中，内存中



**socket**

socket是应用层进入传输层的接口，我们使用socket编写使用TCP或UDP协议的网络应用程序。



```c
int socket (int __domain, int __type, int __protocol)
/*  参数domain指通信域/地址族(即IPv4,IPv6,...)
    	PF_INET/AF_INET IPv4协议
    	PF_UNIX/PF_LOCAL/AF_UNIX/AF_LOCAL 进程通信协议
    
    参数type指socket的类型
    	SOCK_STREAM 可靠的双向的有连接流,即TCP
    	SOCK_DGRAM  不可靠的无连接的报文,即UDP
    	SOCK_SEQPACKET 有序的可靠的双向的有连接的传输
    
    参数protocol通常置0,自动选择type支持的协议
    
    返回一个小整数描述符,失败则返回负数
*/
```

> 参考https://zhuanlan.zhihu.com/p/24916785



```c

```



