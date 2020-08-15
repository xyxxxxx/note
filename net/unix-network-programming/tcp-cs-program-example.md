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
	void				sig_chld(int);

	listenfd = Socket(AF_INET, SOCK_STREAM, 0);

	bzero(&servaddr, sizeof(servaddr));
	servaddr.sin_family      = AF_INET;
	servaddr.sin_addr.s_addr = htonl(INADDR_ANY);
	servaddr.sin_port        = htons(SERV_PORT);

	Bind(listenfd, (SA *) &servaddr, sizeof(servaddr));

	Listen(listenfd, LISTENQ);

	Signal(SIGCHLD, sig_chld);	/* must call waitpid() */

	for ( ; ; ) {
		clilen = sizeof(cliaddr);
		if ( (connfd = accept(listenfd, (SA *) &cliaddr, &clilen)) < 0) {
			if (errno == EINTR)
				continue;		/* back to for() */
			else
				err_sys("accept error");
		}

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

```c
#include	"unp.h"
void
sig_chld(int signo)
{
	pid_t	pid;
	int		stat;

    while ( (pid = waitpid(-1, &stat, WNOHANG)) > 0)
		printf("child %d terminated\n", pid);
	return;
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

| 服务器                                                       | 客户端                                          | 用户                              |
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
| 收到FIN的服务器递送一个EOF给`readline`                       |                                                 | TCP连接终止的前两个分节           |
| `readline`返回0，`str_echo`返回，子进程调用`exit`终止（关闭所有fd，向父进程发送`SIGCHLD`信号） |                                                 |                                   |
|                                                              |                                                 | TCP连接终止的后两个分节，连接终止 |
|                                                              |                                                 |                                   |



# 信号处理

信号（signal）就是告知某个进程发生了某个事件的通知，也称为软件中断（software interrupt）。信号通常是异步发生的。信号可以由一个进程发给另一个进程，或由内核发给进程。

每个信号都有一个与之关联的处置（disposition），也称为行为（action）。我们通过调用`sigaction`函数来设定一个信号的处置，并有三种选择：

1. 提供一个信号处理函数（signal handler），该函数只要有特定信号就被调用。相应的处置称为捕获（catching）信号。信号调用函数由`int`类型参数信号值调用，且没有返回类型：

   ```c
   void handler(int signo);
   ```

2. 将某个信号的处置设定为`SIG_IGN`来忽略（ignore）它。

3. 将某个信号的处置设定为`SIG_DFL`来启用它的默认（default）处置。

> `SIGKILL`和`SIGSTOP`不能被捕获或忽略



POSIX信号处置方法就是调用`sigaction`函数，我们用其自定义`signal`函数。

```c
#include	"unp.h"

Sigfunc * signal(int signo, Sigfunc *func)
{
	struct sigaction	act, oact;

	act.sa_handler = func;
	sigemptyset(&act.sa_mask);
	act.sa_flags = 0;
	if (signo == SIGALRM) {
#ifdef	SA_INTERRUPT
		act.sa_flags |= SA_INTERRUPT;	/* SunOS 4.x */
#endif
	} else {
#ifdef	SA_RESTART
		act.sa_flags |= SA_RESTART;		/* SVR4, 44BSD */
#endif
	}
	if (sigaction(signo, &act, &oact) < 0)
		return(SIG_ERR);
	return(oact.sa_handler);
}
```

+ 一旦安装了信号处理函数，它便一直安装着
+ 在一个信号处理函数运行期间，正被递交的信号是阻塞的（即阻止递交）
+ 如果一个信号在被阻塞期间产生了多次，那么该信号在被解阻塞之后通常只递交一次



僵死状态的进程占用内核空间，最终可能导致耗尽进程资源。无论何时我们`fork`子进程都需要`wait`它们，以防它们变成僵死进程。为此创建一个捕获`SIGCHLD`信号的信号处理函数，并在服务器调用`listen`之后调用它。

```c
#include	"unp.h"
void
sig_chld(int signo)
{
	pid_t	pid;
	int		stat;

	pid = wait(&stat);
	printf("child %d terminated\n", pid);
	return;
}
```



信号处理的步骤如下：

| 服务器                                                       | 客户端                                         | 用户                              |
| ------------------------------------------------------------ | ---------------------------------------------- | --------------------------------- |
|                                                              |                                                | 键入EOF                           |
|                                                              | `fgets`返回一个空指针，`str_cli`函数返回       |                                   |
|                                                              | 客户端的`main`函数调用`exit`终止（关闭所有fd） |                                   |
| 收到FIN的服务器递送一个EOF给`readline`                       |                                                | TCP连接终止的前两个分节           |
| `readline`返回0，`str_echo`返回，子进程调用`exit`终止（关闭所有fd，向父进程发送`SIGCHLD`信号） |                                                |                                   |
| `SIGCHLD`信号递交，父进程执行`sig_chld`函数，调用`wait`取得子进程的PID和终止状态<br />与此同时父进程阻塞于`accept`调用被中断，内核使`accept`返回一个`EINTR`错误，但父进程不处理该错误 |                                                | TCP连接终止的后两个分节，连接终止 |
|                                                              |                                                |                                   |



慢系统调用指那些可能永远阻塞的系统调用。当阻塞于某个慢系统调用的一个进程捕获某个信号且相应的信号处理函数返回时，该系统调用可能返回一个`EINTR`错误。被中断的系统调用并非所有都可以自动重启，因此我们需要自己重启被中断的系统调用。



# `wait`和`waitpid`函数

`wait`函数用于处理已终止的子进程。

```c
#include <sys/wait.h>
pid_t wait(int *statloc);
pid_t waitpid(pid_t pid, int *statloc, int options);
            //等待的子进程pid,-1表示等待第一个终止的子进程
                                       //WNOHANG 没有已终止子进程时不要阻塞
    //return pid if ok, 0 or -1 if err
```

`wait`和`waitpid`返回已终止子进程的进程ID号，同时通过`statloc`指针返回子进程的终止状态（一个整数）。可以调用宏检查终止状态，判别子进程是正常终止、某个信号杀死或由作业控制停止。

如果调用`wait`的进程没有已终止的子进程，但有一个或多个子进程仍在执行，那么`wait`将阻塞直到有一个子进程终止。



假设现有一客户与服务器同时建立了5个连接，在并发服务器上派生了5个子进程。当客户终止时，所有打开的描述符由内核自动关闭。基本在同一时刻，5个连接终止，5个子进程终止，5个`SIGCHLD`信号递交给父进程。

![Screenshot from 2020-08-13 18-27-14.png](https://i.loli.net/2020/08/13/a8PYT6zgCDxjBrU.png)



![Screenshot from 2020-08-13 18-28-29](/home/xyx/Pictures/Screenshot from 2020-08-13 18-28-29.png)

我们发现仍然留下了3个僵死进程：第1个产生的信号引起信号处理函数执行，执行过程中另外4个信号发生，信号处理函数仅再被调用一次。正确的解决方法是在循环内调用`waitpid`，以获取所有已终止子进程的状态。必须指定`WNOHANG`选项以告知`waitpid`在有尚未终止的子进程在运行时不要阻塞。

```c
#include	"unp.h"
void
sig_chld(int signo)
{
	pid_t	pid;
	int		stat;

    while ( (pid = waitpid(-1, &stat, WNOHANG)) > 0)
		printf("child %d terminated\n", pid);
	return;
}
```



# 服务器进程终止

|      |      |      |
| ---- | ---- | ---- |
|      |      |      |
|      |      |      |
|      |      |      |



# 服务器主机关机





# 服务器崩溃后重启





# 数据格式

一般情况下，我们还要关心在客户和服务器之间进行交换的数据的格式。

当我们在客户与服务器

```c

```

