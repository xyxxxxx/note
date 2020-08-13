



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



