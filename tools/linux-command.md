[toc]

#  帮助

## info

查看命令的 `info` 格式的帮助文档。

```shell
$ info ls
10.1 ‘ls’: List directory contents
==================================

The ‘ls’ program lists information about files (of any type, including
directories).  Options and file arguments can be intermixed arbitrarily,
as usual.

...
```



## man

查看命令的使用说明。

```shell
$ man ls
NAME
       ls - list directory contents

SYNOPSIS
       ls [OPTION]... [FILE]...

DESCRIPTION
       List information about the FILEs (the current directory by default).  Sort entries
       alphabetically if none of -cftuvSUX nor --sort is specified.

       Mandatory arguments to long options are mandatory for short options too.

       -a, --all
              do not ignore entries starting with .
              
       ...
```



## whatis

在一些特定的包含系统命令的简短描述的数据库文件中查找关键字，并把结果发送到标准输出。whatis 数据库文件是由 `/usr/sbin/makewhatis` 命令建立的。

```shell
$ whatis ls
ls(1)                    - list directory contents
```



## which

在环境变量 `PATH` 指定的路径中查找命令的位置，并返回第一个搜索结果。

```shell
$ which cp
/usr/local/opt/coreutils/libexec/gnubin/cp
$ which python
/Users/xyx/.pyenv/shims/python
```





# 文件

## cat

连接文件并打印到标准输出设备上。

```shell
$ cat <file>              # 打印文件内容
$ cat <file> > <file>     # 将文件内容写入另一文件
$ cat <file1> >> <file2>  # 将文件内容添加到另一文件末尾
$ cat /dev/null > <file>  # 清空文件内容,等同于`:> <file>`
#     -n                    打印行号
#     -s                    连续两行以上的空白行替换为一行空白行
```



## chmod

修改文件或目录的权限。

```shell
# u=user, g=group, o=other group, a=ugo
$ chmod u=rwx,g=rx,o=x <file>    # rwxr-x--x
$ chmod 777 <file>               # rwxrwxrwx
$ chmod a+r <file>               # ugo 增加权限 r
$ chmod ug+w, o-w <file>         # ug 增加权限 w, o 减少权限 w
```



## cp

复制文件或目录。

```shell
$ cp <file> <file>      # 复制文件到指定文件
$ cp -r <dir> <dir>	    # 复制目录及其下所有文件到指定目录
#    -f                   直接覆盖同名文件而不询问
#    -i                   覆盖同名文件前询问用户
#    -p                   保留原文件的修改时间和访问权限
```



## diff

逐行比较文本文件的差异。

```shell
$ diff <file1> <file2>
#      -B                 忽略空白行
#      -c                 显示全文并标出不同之处
#      -q                 仅显示是否有差异,不显式详细信息
#      -w                 忽略whitespace
#      -W                 指定栏宽
#      -y                 并排格式输出
```



## ln

为文件创建链接。

```shell
$ ln -s <file1> <file2>	 # 软链接(符号链接),相当于Windows的快捷方式.lnk
$ ln <file1> <file2>	   # 硬链接,相当于文件有多个名称
```



## more

分页显示文本文件内容，按空格键到下一页，b 键到上一页，q 键退出。

```shell
$ more <file>
$ more -5 <file>      # 单次显示5行
$ more +5 <file>      # 从第5行开始显示
```



##  mv

移动文件或目录，也可用于文件和目录的重命名。

```shell
$ mv <file> <dir>       # 移动文件到目录下
$ mv <file> <file>	    # 移动文件并重命名
$ mv <dir> <dir>		    # 移动目录并重命名
#    -f                   直接覆盖同名文件而不询问
#    -i                   覆盖同名文件前询问用户
#    -n                   不覆盖同名文件
```



## rm

删除文件或目录。

```shell
$ rm <file>
$ rm -r <dir>			# 删除目录及其下所有文件
#    -f             直接删除只读文件
```



## sort

对文本文件内容进行排序。

```shell
$ sort <file>			# 将文件各行按ASCII的字典序进行排序
#      -d           忽略英文字母、数字、whitespace之外的所有字符
#      -f           将小写字母视作大写字母
#      -i           忽略040至176之间的ASCII字符之外的所有字符
#      -r           排逆序
#      -n           按数值大小排序
```



## tail

查看文本文件内容。

```shell
$ tail -n 5 <file>   # 显示最后5行
$ tail -n +5 <file>  # 从第5行开始显示
$ tail -f <file>		 # 显示更新内容
```



## tar

备份文件。

```shell
$ tar -cvf <file.tar> <dir>	            # 备份
$ tar -zcvf <file.tar.gz> <dir>	        # 备份并压缩
$ tar -xvf <file.tar> [-C <dir>]        # 还原              
$ tar -zxvf <file.tar.gz> [-C <dir>]    # 解压缩并还原
#     -c                                  新建备份文件
#     -C                                  指定目标目录
#     -f                                  指定备份文件
#     -v                                  显示执行过程
#     -x                                  从备份文件还原文件
#     -z                                  通过gzip指令压缩或解压缩文件
```



## touch

新建空文件，或修改文件或目录的时间属性。

```shell
$ touch <file>            # 新建空文件,或修改文件的时间属性为当前系统时间
```



## unzip

解压缩文件。

```shell
$ unzip file.zip -d <dir>  # 将压缩文件解压到目录下 
#       -c                   显示解压过程
#       -l                   显示压缩文件内包含的文件
```



## zip

压缩文件。

```shell
$ zip -r file.zip <dir>   # 将目录及其下所有文件打包为压缩文件
```





# 目录

## cd

切换当前工作目录。

```shell
$ cd              	# home
$ cd ~			        # home
$ cd /			        # 根目录
$ cd ..			        # 上级目录
$ cd -				      # 上一个目录
$ cd <dir>			    # 指定目录
```



## find

在目录下查找文件。

```shell
$ find . -name "*.c"    # 在当前目录下查找名称匹配`*.c`的文件
$ find . -mtime -20     #               20天内修改过的文件
$ find . -ctime +20     #               20天前创建的文件
$ find . -type f        #               类型为f(normal)的文件
$ find . -size +100c    #               大小大于100字节的文件(c,k,M分别表示B,kB,MB)
$ find . -size 0 -exec ls -l {} \;	  # 空文件,并对这些文件执行指定命令
```



## ls

显示目录下所有文件和子目录。

```shell
$ ls
$ ls -lrt               # 显示文件和目录的详细信息,按从旧到新的顺序排列
$ ls s*                 # 显示文件名匹配`s*`的文件
$ ls -a                 # 显示`.`开头的隐藏文件
#    -l                   显示文件权限、所有者、大小、创建时间等详细信息
#    -r                   逆序显示
#    -t                   按文件创建时间的先后顺序排列
```



## mkdir

新建目录。

```shell
$ mkdir <dir>
$ mkdir -p <dir1>/<dir2>  # 若dir1不存在,创建之
```



## pwd

显示当前工作目录。

```shell
$ pwd
```



## rmdir

删除空目录。删除非空目录请使用命令 `rm`。

```shell
$ rmdir <dir>
#       -p       若空目录被删除后其上级目录成为空目录,则一并删除
```





#  文本

## echo

打印字符串。

```shell
$ echo abc123            # 打印一般字符串
$ echo $PATH             # 打印环境变量
$ echo \"How are you\?\" \"I\'m fine.\"     # 使用转义符号打印特殊符号
$ echo abc123 > <file>   # 将打印内容写入文件
```

```shell
$ echo `date`            # 打印当前日期时间
```



## grep

查找文本文件中符合条件的内容。

```shell
$ grep <str> <file>      # 查找并打印文本文件中包含该字符串的所有行
#      -e                  使用正则表达式匹配字符串
#      -i                  忽略字符的大小写
#      -r                  递归地查找目录下的所有文件
#      -v                  查找不包括该字符串的所有行
```



## egrep

查找文本文件中符合条件的内容。与 `grep` 的区别在于解读字符串的方法：`egrep` 使用 extended regular expression 语法进行解读，而 `grep` 则使用 basic regular expression 语法进行解读，前者比后者的表达更规范。

```shell
# 查找文本
$ egrep <str>/<re> <file>      # search in file by str/regular expression
$ egrep -r <str>/<re> <dir>    # search in dir by str/regular expression
```





#  Shell

## clear

清空命令行。

```shell
$ clear
```



## env

显示环境变量。

```shell
$ env
```



## exit

退出命令行。

```shell
$ exit
```



## export

设置或显示环境变量。

```shell
$ export -p           # 显示Shell提供的环境变量
$ export MYENV=1      # 定义环境变量,仅在当前Shell中有效
$ export -n MYENV     # 删除环境变量,仅在当前Shell中有效
```



## set

设置 Shell 的运行参数。

```shell
$ set                 # 显示环境变量和Shell函数
$ set MYENV=1         # 定义环境变量,仅在当前Shell中有效
```



## unset

删除 Shell 的运行参数。

```shell
$ unset MYENV         # 删除环境变量,仅在当前Shell中有效
```



## xargs

`xargs` 是给命令传递参数的一个过滤器，通常和管道一起使用，相当于将前一命令的输出解析为后一命令的参数。

```shell
$ cat test.txt
a b c d e f g
h i j k
$ cat test.txt | xargs	    # `xargs`
$ cat test.txt | xargs -n3	# 多行输出,每行3个参数
$ echo "nameXnameXnameXname" | xargs -dX	# 用X分割并输出

# -I指定占位符,占位符指定参数传入的位置
$ ls *.jpg | xargs -n1 -I {} cp {} /data/images
# 将当前目录下所有.jpg文件备份并压缩
$ find . -type f -name "*.jpg" -print | xargs tar -czvf images.tar.gz

$ cat url-list.txt | xargs wget -i      # 批量下载`url-list.txt`中的链接
```



```shell
$ ls
$ ls -lrt               # 显示文件和目录的详细信息,按从旧到新的顺序排列
$ ls s*                 # 显示文件名匹配`s*`的文件
$ ls -a                 # 显示`.`开头的隐藏文件
#    -l                   显示文件权限、所有者、大小、创建时间等详细信息
#    -r                   逆序显示
#    -t                   按文件创建时间的先后顺序排列
```







#  网络

## curl

curl是一个常用的命令行工具，用于数据传输，支持多种协议，功能十分强大。

```shell
$ curl www.example.com         # 自动选择协议,默认为`GET`方法
<!doctype html>
<html>
<head>
    <title>Example Domain</title>    
...
</body>
</html>
```



### -b

向服务器发送Cookie。

```shell
$ curl -b 'foo=bar' https://www.google.com                # 生成请求头`Cookie: foo=bar`
$ curl -b 'foo1=bar1; foo2=bar2' https://www.google.com   # 发送多个Cookie
$ curl -b cookies.txt https://www.google.com              # 读取本地Cookie文件并发送
```



### -c

将服务器返回的Cookie写入文件。

```shell
$ curl -c cookies.txt https://www.google.com          # 将响应头的Cookie设置写入`cookie.txt`文件
```



### -d

发送`POST`请求的数据体。

```shell
# 请求自动转为POST方法,请求头设置 `Content-Type : application/x-www-form-urlencoded`
$ curl -d 'login=emma＆password=123' [-X POST] https://www.google.com/login
$ curl -d 'login=emma' -d 'password=123' [-X POST] https://www.google.com/login   # 同上
$ curl -d '@data.txt' https://www.google.com/login        # 读取本地`data.txt`文件并作为数据体发送
```



### -F

向服务器上传二进制文件。

```shell
# 请求头设置 `Content-Type: multipart/form-data`
$ curl -F 'file=@photo.png' https://www.google.com/profile                   # 上传`photo.png`文件
$ curl -F 'file=@photo.png;filename=me.png' https://www.google.com/profile   # 指定服务器接收到的文件名 
```



### -G

构造URL的查询字符串。

```shell
$ curl -G -d 'q=kitties' -d 'count=20' https://www.google.com/search
# 相当于发送GET请求,实际请求的URL为`https://www.google.com/search?q=kitties&count=20`
# 若去掉`-G`选项,则变为发送`POST`请求的数据体
```



### -H

增加请求头。

```shell
$ curl -H 'accept-language: en-US' https://www.google.com
```



### -i

打印响应头和响应内容。

```shell
$ curl -i https://www.example.com
HTTP/2 200
accept-ranges: bytes
age: 451289
cache-control: max-age=604800
content-type: text/html; charset=UTF-8
date: Fri, 26 Mar 2021 05:48:02 GMT
etag: "3147526947"
expires: Fri, 02 Apr 2021 05:48:02 GMT
last-modified: Thu, 17 Oct 2019 07:18:26 GMT
server: ECS (oxr/830C)
vary: Accept-Encoding
x-cache: HIT
content-length: 1256

<!doctype html>
<html>
<head>
    <title>Example Domain</title>
...
</body>
</html>
```



### -I, --head

向服务器发出`HEAD`请求，打印响应头。

```shell
$ curl -I https://www.example.com
HTTP/2 200 
content-encoding: gzip
accept-ranges: bytes
age: 563101
cache-control: max-age=604800
content-type: text/html; charset=UTF-8
date: Fri, 26 Mar 2021 05:59:42 GMT
etag: "3147526947"
expires: Fri, 02 Apr 2021 05:59:42 GMT
last-modified: Thu, 17 Oct 2019 07:18:26 GMT
server: ECS (oxr/8325)
x-cache: HIT
content-length: 648
```



### -L

使请求跟随服务器的重定向。默认不跟随重定向。

```shell
$ curl -L https://ff.sdo.com
```



### --limit-rate

限制请求和响应的带宽，用于模拟网速慢的环境。

```shell
$ curl --limit-rate 200k https://www.google.com    # 限速200kB/s
```



### -o, -O

将服务器的响应保存成文件，等同于`wget`命令。

```bash
$ curl -o example.html https://www.example.com     # 保存到`example.html`文件中
$ curl -O https://www.example.com/foo/bar.html     # 保存到`bar.html`文件中
```



### -u

设置服务器认证的用户名和密码。

```shell
$ curl -u 'bob:12345' https://google.com/login     # 设置用户名为`bob`,密码为`12345`
                                                   # 将被转换为请求头`Authorization: Basic Ym9iOjEyMzQ1`
```



### -v

输出通信的整个过程，用于调试。

```shell
$ curl -v https://www.example.com
* Uses proxy env variable NO_PROXY == '127.0.0.0/8,localhost,192.168.0.0/16,100.64.0.0/16,10.147.0.0/16,.tensorstack.net,.tsz.io,tsz.io'
*   Trying 93.184.216.34...
* TCP_NODELAY set
* Connected to www.example.com (93.184.216.34) port 443 (#0)
* ALPN, offering h2
* ALPN, offering http/1.1
* successfully set certificate verify locations:
*   CAfile: /etc/ssl/cert.pem
  CApath: none
* TLSv1.2 (OUT), TLS handshake, Client hello (1):
* TLSv1.2 (IN), TLS handshake, Server hello (2):
* TLSv1.2 (IN), TLS handshake, Certificate (11):
* TLSv1.2 (IN), TLS handshake, Server key exchange (12):
* TLSv1.2 (IN), TLS handshake, Server finished (14):
* TLSv1.2 (OUT), TLS handshake, Client key exchange (16):
* TLSv1.2 (OUT), TLS change cipher, Change cipher spec (1):
* TLSv1.2 (OUT), TLS handshake, Finished (20):
* TLSv1.2 (IN), TLS change cipher, Change cipher spec (1):
* TLSv1.2 (IN), TLS handshake, Finished (20):
* SSL connection using TLSv1.2 / ECDHE-RSA-AES128-GCM-SHA256
* ALPN, server accepted to use h2
* Server certificate:
*  subject: C=US; ST=California; L=Los Angeles; O=Internet Corporation for Assigned Names and Numbers; CN=www.example.org
*  start date: Nov 24 00:00:00 2020 GMT
*  expire date: Dec 25 23:59:59 2021 GMT
*  subjectAltName: host "www.example.com" matched cert's "www.example.com"
*  issuer: C=US; O=DigiCert Inc; CN=DigiCert TLS RSA SHA256 2020 CA1
*  SSL certificate verify ok.
* Using HTTP2, server supports multi-use
* Connection state changed (HTTP/2 confirmed)
* Copying HTTP/2 data in stream buffer to connection buffer after upgrade: len=0
* Using Stream ID: 1 (easy handle 0x7fb64a00d600)
> GET / HTTP/2
> Host: www.example.com
> User-Agent: curl/7.64.1
> Accept: */*
> 
* Connection state changed (MAX_CONCURRENT_STREAMS == 100)!
< HTTP/2 200 
< age: 595764
< cache-control: max-age=604800
< content-type: text/html; charset=UTF-8
< date: Fri, 26 Mar 2021 06:01:37 GMT
< etag: "3147526947+ident"
< expires: Fri, 02 Apr 2021 06:01:37 GMT
< last-modified: Thu, 17 Oct 2019 07:18:26 GMT
< server: ECS (oxr/830D)
< vary: Accept-Encoding
< x-cache: HIT
< content-length: 1256
< 
<!doctype html>
<html>
<head>
    <title>Example Domain</title>
</body>
</html>
* Connection #0 to host www.example.com left intact
* Closing connection 0
```



### -x

指定HTTP请求的代理。

```shell
$ curl -x socks5://james:cats@myproxy.com:8080 https://www.example.com
```



### -X

指定HTTP请求的方法。

```shell
$ curl -X POST https://www.example.com
```



### 常见应用





## hostname

显示或设置（仅限超级用户）主机名称。

```shell
$ hostname
192.168.0.100
# 在 /etc/sysconfig/network 位置修改
```



## ifconfig

显示或设置网络设备。

```shell
$ ifconfig                    # 显示网络设备
$ ifconfig eth0 down          # 关闭网络设备
$ ifconfig eth0 up            # 启动网络设备
$ ifconfig eth0 hw ether 00:AA:BB:CC:DD:EE              # 为eth0网卡修改MAC地址
$ ifconfig eth0 192.168.1.56 netmask 255.255.255.0      # 为eth0网卡配置IP地址,并加上子掩码
# 在 /etc/sysconfig/network-scripts/ifcfg-<name> 位置修改
```



## lsof

检查端口状态。

```shell
$ lsof -i:8080                # 检查8080端口的使用状态
```



## nc

设置路由器。



## netstat

显示网络状态。

```shell
$ netstat -an                 # 显示详细网络状态
$ netstat -i                  # 显示网络设备信息
$ netstat -rn                 # 显示本地路由表
$ netstat -s                  # 显示网络统计信息
$ netstat -l                  # 显示监听的服务器的套接字
$ netstat -tunlp | grep 8080  # [Linux]显示进程8080占用端口状态
```



## ping

检测与主机的连接。

```shell
$ ping www.bilibili.com
#      -c <num>                 发送信息的次数
#      -i <secs>                发送的间隔秒数
```



## wget

wget是一个常用的命令行工具，用于下载文件。wget非常稳定，其对带宽很窄、网络不稳定等情况均有很强的适应性。

```shell
$ wget <url>                  # 下载url位置的文件
$ wget -O <filename> <url>    # 下载文件并命名,默认名称为url最后一个`/`之后的字符串
$ wget --limit-rate=1M <url>  # 限速下载
$ wget -c <url>               # 断点续传,即继续下载中断的文件
$ wget -b <url>               # 后台下载
$ wget -i <urllistfile>       # 批量下载:从文本文件中读取所有需要下载的url,每个url占一行
$ wget -o download.log <url>  # 将下载信息保存到日志文件,而不显示在终端
```

```shell
$ wget                        # 镜像网站
```





#  系统

## date

显示或设置系统的日期时间。

```shell
$ date
Mon Jun 28 10:31:54 CST 2021
```



## kill

杀死进程。

```shell
$ kill -l             # 列出所有可用信号
HUP INT QUIT ILL TRAP ABRT EMT FPE KILL BUS SEGV SYS PIPE ALRM TERM URG STOP TSTP CONT CHLD TTIN TTOU IO XCPU XFSZ VTALRM PROF WINCH INFO USR1 USR2
$ kill 12345          # 正常终止进程12345
$ kill -9 12345       # 强制杀死进程12345
$ kill -KILL 12345    # 强制杀死进程12345
```



## reboot

重新启动计算机。

```shell
$ reboot
```



## rpm

管理套件。RPM（Redhat Package Manager）原本是 Red Hat Linux 发行版专门用来管理 Linux 各项套件的程序，由于它遵循 GPL 规则且功能强大方便，因而广受欢迎，逐渐受到其他发行版的采用。

```shell
$ rpm -a				          # 显示套件
$ rpm -e --nodeps <pkg>   # 卸载套件
```



## ps

显示当前的进程状态。

```shell
$ ps -ef             # 显示进程详细信息: USER, PID, PPID, START, CMD, ...
$ ps -aux            # [Linux]显示进程详细信息: USER, PID, %CPU, %MEM, START, CMD, ...
$ ps -A              # 显示所有进程
```



## shutdown

关闭计算机。

```shell
$ shutdown now       # 立即关闭计算机
#          -t <secs>   设定`secs`秒后运行关机程序
#          -r          重新启动计算机
```



## sudo

以系统超级用户（管理员）的身份执行命令。

```shell
$ sudo <command>
#      -u <username>/<uid>    以用户的身份执行命令
```



## top

实时显示进程的状态。

```shell
$ top
#     -p <pid>         [Linux]显示指定进程
#     -pid <pid>       [MacOS]显示指定进程
#     -d <secs>        [Linux]每`secs`秒更新
```





#  逻辑关系

```shell
# &&: 前一命令成功,则执行后一命令(command1 succeed, then command2)
$ cp sql.txt sql.bak.txt && cat sql.bak.txt

# ||: 前一命令失败,则执行后一命令(command1 failed, then command2)
$ ls /proc && echo success || echo failed

#  (;) 

# |: 前一命令的输出作为后一命令的输入(command1 output as command2 input)
$ ls -l /etc | more
$ echo "Hello World" | cat > hello.txt
$ ps -ef | egrep python
```





#  gcc

GCC（GNU Compiler Collection，GNU 编译器套装） 是一套编译器，起初为 C 语言编译器（GNU C Compiler），之后得到扩展，支持处理 C++、Fortran、Pascal、Java、Go 等语言。许多类 Unix 系统（包括 Linux、MacOS）都采用 GCC 作为标准编译器。

GCC 支持多种硬件平台，如 ARM、x86 等，并且还支持跨平台交叉编译（即 x86 平台上编译的程序可以在 ARM 平台上运行）。

GCC 通过 `gcc` 命令使用：

```shell
$ gcc -E hello.c -o hello.i    # 预处理(Preprocess)
$ gcc -S hello.c -o hello.s    # 编译(Compile)
$ gcc -c hello.c -o hello.o    # 汇编(Assemble)
$ gcc hello.c -o hello.exe     # 链接(Link)
```

