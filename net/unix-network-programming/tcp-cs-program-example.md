

参照《UNIX网络编程 卷1》第五章进行实验。

> 源代码位于https://github.com/unpbook/unpv13e

# 程序

## 服务器程序

```c
#include	"unp.h"

int
main(int argc, char **argv)
{
	int					listenfd, connfd;
	pid_t				childpid;
	socklen_t			clilen;
	struct sockaddr_in	cliaddr, servaddr;

	listenfd = Socket(AF_INET, SOCK_STREAM, 0);

	bzero(&servaddr, sizeof(servaddr));
	servaddr.sin_family      = AF_INET;
	servaddr.sin_addr.s_addr = htonl(INADDR_ANY);
	servaddr.sin_port        = htons(SERV_PORT);

	Bind(listenfd, (SA *) &servaddr, sizeof(servaddr));

	Listen(listenfd, LISTENQ);

	for ( ; ; ) {
		clilen = sizeof(cliaddr);
		connfd = Accept(listenfd, (SA *) &cliaddr, &clilen);

		if ( (childpid = Fork()) == 0) {	/* child process */
			Close(listenfd);	/* close listening socket */
			str_echo(connfd);	/* process the request */
			exit(0);
		}
		Close(connfd);			/* parent closes connected socket */
	}
}

```

```c
#include	"unp.h"
#include	"sum.h"

void
str_echo(int sockfd)
{
	ssize_t			n;
	struct args		args;
	struct result	result;

	for ( ; ; ) {
		if ( (n = Readn(sockfd, &args, sizeof(args))) == 0)
			return;		/* connection closed by other end */

		result.sum = args.arg1 + args.arg2;
		Writen(sockfd, &result, sizeof(result));
	}
}

```



## 客户程序

```c
#include	"unp.h"

int
main(int argc, char **argv)
{
	int					sockfd;
	struct sockaddr_in	servaddr;

	if (argc != 2)
		err_quit("usage: tcpcli <IPaddress>");

	sockfd = Socket(AF_INET, SOCK_STREAM, 0);

	bzero(&servaddr, sizeof(servaddr));
	servaddr.sin_family = AF_INET;
	servaddr.sin_port = htons(SERV_PORT);
	Inet_pton(AF_INET, argv[1], &servaddr.sin_addr);

	Connect(sockfd, (SA *) &servaddr, sizeof(servaddr));

	str_cli(stdin, sockfd);		/* do it all */

	exit(0);
}
```

```c
#include	"unp.h"
#include	"sum.h"

void
str_cli(FILE *fp, int sockfd)
{
	char			sendline[MAXLINE];
	struct args		args;
	struct result	result;

	while (Fgets(sendline, MAXLINE, fp) != NULL) {

		if (sscanf(sendline, "%ld%ld", &args.arg1, &args.arg2) != 2) {
			printf("invalid input: %s", sendline);
			continue;
		}
		Writen(sockfd, &args, sizeof(args));

		if (Readn(sockfd, &result, sizeof(result)) == 0)
			err_quit("str_cli: server terminated prematurely");

		printf("%ld\n", result.sum);
	}
}
```



# 正常启动，运行与关闭

| 服务器                                                       | 客户端                                          |                                   |
| ------------------------------------------------------------ | ----------------------------------------------- | --------------------------------- |
| 服务器启动                                                   |                                                 |                                   |
| 服务器调用`socket`, `bind`, `listen`和`accept`，并阻塞于`accept` |                                                 |                                   |
|                                                              | 客户端启动                                      |                                   |
|                                                              | 客户端调用`socket`和`connect`                   |                                   |
|                                                              |                                                 | TCP三路握手过程                   |
|                                                              | `connect`返回                                   | 第二个分节返回                    |
| `accept`返回                                                 |                                                 | 第三个分节返回                    |
| 服务器调用`fork`                                             | 客户端调用`str_cli`函数                         |                                   |
| 父进程再次调用`accept`并阻塞；<br />子进程调用`str_echo`，`str_echo`调用`readline`并阻塞 | 阻塞于`fgets`调用                               |                                   |
|                                                              |                                                 | 键入`hello\n`                     |
|                                                              | `fgets`读取`hello\n`，并写入`sockfd`            |                                   |
| `str_echo`读取`hello\n`并写入`sockfd`，`str_echo`调用`readline`并阻塞 |                                                 |                                   |
|                                                              | `fgets`读取`hello\n`，并打印，阻塞于`fgets`调用 |                                   |
|                                                              |                                                 | 键入EOF                           |
|                                                              | `fgets`返回一个空指针，`str_cli`函数返回        |                                   |
|                                                              | 客户端的`main`函数调用`exit`终止（关闭所有fd）  |                                   |
|                                                              |                                                 | TCP连接终止的前两个分节           |
| `readline`返回0，`str_echo`返回，子进程调用`exit`终止（关闭所有fd，向父进程发送`SIGCHLD`信号） |                                                 |                                   |
|                                                              |                                                 | TCP连接终止的后两个分节，连接终止 |
|                                                              |                                                 |                                   |



# POSIX信号处理

信号（signal）就是告知某个进程发生了某个事件的通知，也称为软件中断（software interrupt）。信号通常是异步发生的。信号可以由一个进程发给另一个进程，或由内核发给进程。

每个信号都有一个与之关联的处置（disposition），也称为行为（action）。我们通过调用`sigaction`函数来设定一个信号的处置，并有三种选择：

1. 信号处理函数（signal handler）：只要有特定信号就被调用。