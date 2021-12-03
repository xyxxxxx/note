![](https://raw.githubusercontent.com/xyxxxxx/image/master/sdfgjiokm35kolmh35kl5h.PNG)

# `socket`函数

```c
#include <sys/socket.h>
int socket (int family, int type, int protocol)
/*  参数family指通信域/协议族/协议域(即IPv4,IPv6,...)
    	AF_INET  IPv4协议
    	AF_INET6 IPv6协议
    	AF_LOCAL 进程通信协议
    	AF_ROUTE 路由socket
    	AF_KEY   密钥socket
    
    参数type指socket类型
    	SOCK_STREAM    字节流socket,可靠的双向的有连接流,即TCP
    	SOCK_DGRAM     数据报socket,不可靠的无连接的报文,即UDP
    	SOCK_SEQPACKET 有序分组socket,有序的可靠的双向的有连接的传输
    	SOCK_RAW       原始socket
    
    参数protocol通常置0,自动选择type支持的协议
        IPPROTO_CP    TCP传输协议
        IPPROTO_UDP   UDP传输协议
        IPPROTO_SCTP  SCTP传输协议
    
    返回一个小整数描述符,失败则返回-1
*/
```



`AF_`前缀表示地址族（address family），`PF_`前缀表示协议族（protocol family）。实际上`<sys/socket.h>`中给定协议的`AF_`值总是与`PF_`值相等。



# `connect`函数

TCP客户用`connect`函数建立与TCP服务器的连接。

```c
#include <sys/socket.h>
int connect(int sockfd, const struct sockaddr *servaddr, socklen_t addrlen);
    //return 0 if ok, -1 if err
```



TCP socket调用`connect`函数将激发TCP的三路握手过程，在连接建立成功或出错时返回，其中出错返回有以下几种情况：

+ TCP客户没有受到SYN分节的响应，则返回`ETIMEDOUT`错误。例如调用`connect`函数时，内核发送一个SYN，若无响应则等待6s再发送一个，若仍无响应则等待24s再发送一个，总共等待75s后仍无响应则返回本错误。

+ 服务器对客户SYN的响应是RST，则表明服务器主机在指定的端口上没有及昵称在等待与之连接。客户一接收到RST就马上返回`ECONNREFUSED`错误。

  RST是TCP在发生错误时发送的一种TCP分节。产生RST的三个条件是：SYN的目的端口没有正在监听的服务器；TCP想取消一个已有连接；TCP接收到一个根本不存在的连接上的分节。

+ 客户发出的SYN在中间的某个路由器上引发了一个“destination unreachable”ICMP错误。客户主机内核保存该消息，并按第一种情况继续发送SYN。若在75s后仍未收到响应，则把保存的消息（ICMP错误）作为`EHOSTUNREACH`或`ENETUNREACH`错误返回给进程。亦有可能是以下两种情况：按照本地系统的转发表，没有到达远程系统的路径；调用`connect`不等待就返回。



运行时间获取客户程序，指定运行着服务器的本地主机：

```
$ daytimetcpcli 127.0.0.1
Sun Jul 27 22:01:51 2003
```

指定本地子网上不存在主机的一个IP地址：

```
$ daytimetcpcli 192.168.1.100
connect error: Connection timed out
```

指定一个没有运行服务器的主机：

```
$ daytimetcpcli 192.168.1.5
connect error: Connection refused
```

指定一个互联网中不可到达的IP地址：

```
$ daytimetcpcli 192.3.4.5
connect error: No route to host
```



`connect`函数使当前socket从CLOSED状态转移到SYS_SENT状态，若成功则再转移到ESTABLISHED状态。若`connect`失败则该socket不可用，必须关闭。





# `bind`函数

`bind`函数把一个本地协议地址赋予一个socket。对于IP协议，协议地址是32位的IPv4地址或128位的IPv6地址与16位的端口号的组合。

```c
#include <sys/socket.h>
int bind(int sockfd, const struct sockaddr *myaddr, socklen_t addrlen);
    //return 0 if ok, -1 if err
```



+ 如果TCP客户或服务器未调用`bind`捆绑一个端口，当调用`connect`或`listen`时，内核就为相应的socket选择一个临时端口。服务器通常捆绑其众所周知端口。
+ 进程可以把一个特定的IP地址捆绑到它的socket上，不过这个IP地址必须属性其所在主机的网络接口之一。对于TCP客户，这为在该socket上发送的IP数据包指派了源IP地址；对于TCP服务器，这限定了该socket只接受目的地为这个IP地址的客户连接。TCP客户通常不把IP地址捆绑到它的socket上；如果TCP服务器没有把IP地址捆绑到它的socket上，内核九八客户发送的SYN的目的IP地址作为服务器的源IP地址

![](https://raw.githubusercontent.com/xyxxxxx/image/master/jkvsgdfngnjk134lnt4y2jk24tyg.PNG)

+ 如果指定端口号为0，内核就在调用`bind`时选择一个临时端口；如果指定IP地址为通配地址，内核将等到socket已连接（TCP）或已在socket上发出数据包（UDP）时才选择一个本地IP地址。

  ```c
  //通配地址
  //IPv4
  struct sockaddr_in servaddr;
  servaddr.sin_addr.s_addr = htonl(INADDR_ANY); //INADDR_ANY值一般为0
  
  //IPv6
  struct sockaddr_in6 serv;
  serv.sin6_addr = in6addr_any;
  ```



进程捆绑非通配IP地址到socket上的常见例子是在为多个组织提供Web服务器的主机上。网络层接收所有目的地为任何一个地址的外来数据报，而每个HTTP服务器仅捆绑相应的IP地址。这样做的好处是操作由内核完成。

另一种方法是运行捆绑通配地址的单个服务器。当连接到达时，服务器调用`getsockname`函数获取来自客户的目的IP地址，然后根据这个目的IP地址来处理客户请求。





# `listen`函数

`listen`函数由TCP服务器调用，其作用为：

1. 当socket函数创建一个socket时，其被假设为一个主动socket，即调用`connect`发起连接的客户socket。`listen`函数将一个未连接的socket转换成一个被动socket，指示内核应接受指向该socket的连接请求。调用`listen`导致socket从CLOSED状态转换到LISTEN状态。
2. `listen`函数的第二个参数规定了内核维护的监听socket队列中的最大连接个数

```c
#include <sys/socket.h>
int listen(int sockfd, int backlog);
    //return 0 if ok, -1 if err
```

内核为任何一个给定的监听socket维护两个队列：

1. 未完成连接队列（incomplete connection queue），每个已收到其SYN分节但仍在等待TCP三路握手过程完成的客户对应其中一项。这些socket处于SYN_RCVD状态。
2. 已完成连接队列（completed connection queue），每个已完成TCP三路握手过程的客户对应其中一项。这些socket处于ESTABLISHED状态。

![](https://raw.githubusercontent.com/xyxxxxx/image/master/ajfioj1234niongfjk.PNG)



当来自客户的SYN到达时，TCP在未完成连接队列中创建一个新项，然后响应以三路握手的第二个分节（服务器的SYN响应和对客户SYN的ACK）。这一项保留在未完成连接队列中，直到三路握手的第三个分节（客户对服务器SYN的ACK）到达或者该项超时。如果三路握手正常完成，该项就从完成连接队列移动到已完成连接队列的队尾。当进程调用`accept`时，已完成连接队列的队首项将返回给进程；如果该队列为空，那么进程将被投入睡眠，直到TCP在该队列中放入一项。

+ 不要把`backlog`定义为0
+ 队列的实际连接数目通常大于`backlog`值
+ 在三路握手正常完成的情况下，未完成连接队列中任何一项在其中的存留时间就是一个RTT（典型值为187ms）
+ 现今的服务器同时处理很多个连接，因此必须指定一个较大的`backlog`值
+ 当一个客户SYN到达时，若这些队列是满的，TCP就忽略该分节，不发送RST。客户将重发SYN，期望不久之后就能在这些队列中找到可用空间。





# `accept`函数

`accept`函数由TCP服务器调用，用于从已完成连接队列队首返回一个连接。如果已完成连接队列为空，那么进程被投入睡眠。

```c
#include <sys/socket.h>
int accept(int sockfd, struct sockaddr *cliaddr, socklen_t *addrlen);
    //return fd if ok, -1 if err
```

参数`cliaddr`和`addrlen`用来返回已连接的客户的协议地址。调用前，我们将由`addrlen`所指向的整数值置为`cliaddr`所指的socket地址结构的长度，返回时该整数值置为内核存放在该地址结构内的确切字节数。

`accept`函数返回由内核自动生成的一个全新描述符，代表与所返回客户的TCP连接。`accept`函数的第一个参数称为监听（listening）socket描述符，返回值称为已连接（connected）socket描述符。一个服务器通常仅创建一个监听socket，其在服务器的生命周期内一直存在；内核为每个由服务器进程接受的客户连接创建一个已连接socket，当服务器完成对某个客户的服务时，相应的已连接socket就被关闭。



# `fork`和`exec`函数

`fork`函数是Unix派生新进程的唯一方法。

```c
#include <unistd.h>

pid_t fork(void);
    //return pid of child process in parent process, 0 in child process, -1 if err
```

`fork`函数在调用进程（父进程）和派生进程（子进程）中各返回一次，在父进程中返回子进程的pid，在子进程中返回0。

父进程中调用`fork`之前打开的所有fd在`fork`返回之后与子进程共享。



`fork`有两种典型用法：

1. 一个进程创建一个自身的副本，这样每个副本都可以执行各自的操作
2. 一个进程调用`fork`创建一个自身的副本，然后其中一个副本调用`exec`将自身替换为新的程序



存放在硬盘上的可执行程序文件能够被Unix执行的唯一方法是，由一个现有进程调用`exec`函数。`exec`把当前进程映像替换成新的程序文件，新程序通常从`main`函数开始执行，进程的pid不变。

```c
#include <unistd.h>

//execve是系统调用，其它5个都是调用execve的库函数
//return NULL if ok, -1 if err

int execl(const char *pathname, const char *arg0, ...);

int execv(const char *pathname, char *const argv[]);

int execle(const char *pathname, const char *arg0, ...);

int execve(const char *pathname, char *const *argv[], char *const envp[]);

int execlp(const char *filename, const char *arg0, ...);

int execvp(const char *filename, const *const argv[]);
```

![](https://raw.githubusercontent.com/xyxxxxx/image/master/df1456241t5361fdg123.PNG)





# 并发服务器

对于简单的服务器而言，迭代服务器（iterative server）就已经足够。但我们希望服务器同时服务多个客户，而不是被单个客户长期占用。Unix中编写并发服务器最简单的方法就是`fork`一个子进程来服务每个客户。

```c
pid_t pid;
int listenfd, connfd;
listenfd = socket(...);
bind(listenfd, ...);
listen(listenfd, LISTENQ);

for ( ; ; ) {
    connfd = accept(listenfd, ...);

    if ( (childpid = Fork()) == 0) {	/* child process */
        Close(listenfd);	/* close listening socket */
        doit(connfd);   	/* process the request */
        exit(0);
    }
    Close(connfd);			/* parent closes connected socket */
}
```

当一个连接建立时，`accept`返回，服务器接着调用`fork`，然后由子进程服务客户，父进程则等待另一个连接。子进程关闭监听socket，父进程关闭已连接socket。



每个fd都有一个引用计数，表示当前打开着的引用该fd的个数。引用计数在文件表项中维护。`socket`和`accept`返回后，与`listenfd`和`connfd`关联的文件表项的引用计数值为1；但`fork`返回后，这两个fd就在父进程和子进程之间共享，因此这两个fd关联的文件表项的引用计数值变成2。当父进程关闭`connfd`时，相应的引用计数值从2减为1；当子进程关闭`connfd`时，引用计数值减为0，socket才被清理和资源释放。



![Screenshot from 2020-08-13 12-54-41.png](https://i.loli.net/2020/08/13/rE49yiHtTqXkGV8.png)
![Screenshot from 2020-08-13 12-54-51.png](https://i.loli.net/2020/08/13/CdGWtgAuYHvwM7K.png)
![Screenshot from 2020-08-13 12-55-13.png](https://i.loli.net/2020/08/13/8dXPMQSkTtAHGoq.png)
![Screenshot from 2020-08-13 12-55-24.png](https://i.loli.net/2020/08/13/zIvUQHsx7nCY6AB.png)



# `close`函数

`close`函数也用来关闭socket，并终止TCP连接。

```c
#include <unistd.h>
int close(int sockfd);
    //return 0 if ok, -1 if err
```

`close`一个TCP socket的默认行为是把该socket标记成已关闭，然后立即返回到调用进程。该sockfd不能再由调用进程使用。



# `getsockname`和`getpeername`函数

`getsockname`和`getpeername`函数返回与某个socket关联的本地协议地址和外地协议地址。

```c
#include <sys/socket.h>
int getsockname(int sockfd, struct sockaddr *localaddr, socklen_t *addrlen);
int getpeername(int sockfd, struct sockaddr *peeraddr, socklen_t *addrlen);
                                                       //值-结果参数
    //return 0 if ok, -1 if err
```

这两个函数的作用是：

+ 在一个没有调用`bind`的TCP客户上，`connect`成功返回后，`getsockname`用于返回由内核赋予该连接的本地IP地址和本地端口号

+ 在以端口号0调用`bind`后，`getsockname`用于返回由内核赋予的本地端口号

+ `getsockname`可用于获取某个socket的地址族

  ```c
  #include	"unp.h"
  
  int
  sockfd_to_family(int sockfd)
  {
  	struct sockaddr_storage ss; //通用地址结构sockaddr_storage能够承载任何socket地址结构
  	socklen_t	len;
  
  	len = sizeof(ss);
  	if (getsockname(sockfd, (SA *) &ss, &len) < 0)
  		return(-1);
  	return(ss.ss_family);
  }
  ```

+ 在一个以通配地址调用`bind`的TCP服务器上，与某个客户的连接建立后，`getsockname`用于返回由内核赋予该连接的本地IP地址。这里sockfd必须是已连接socket的描述符，而不是监听socket的描述符

+ 当一个服务器是由调用过`accept`的某个进程再调用`exec`执行程序时，它能够获取客户身份的唯一途径就是调用`getpeername`



