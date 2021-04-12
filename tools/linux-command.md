[toc]

#  info

```shell
$ whatis <command>      # 命令info
$ info <command>		# 命令详细info
$ man <command>		    # 命令manual

$ which <command>		# 查找命令,返回可执行程序的绝对路径
$ which mv
/bin/mv
```





# file

##  files management

```shell
# 移动文件或目录
$ mv <file> <dir>
$ mv <file> <file>	    # 重命名
$ mv <dir> <dir>		# 移动或重命名

# 删除
$ rm <file>
$ rm -r <dir>			# 删除目录及其目录下的所有文件
#    -f: 直接删除只读文件

# 复制
$ cp <file> <dir+file>	# 复制文件
$ cp -r <dir> <dir>	    # 复制目录

# 读
$ more <file>           # 打印文件内容
$ more -5 <file>	    # 单次打印5行, space 翻页, q 退出
$ more +5 <file>		# 从第5行开始打印
$ tail -5 <file>	    # 打印最后5行
$ tail +5 <file>		# 从第5行开始打印
$ tail -f <file>		# 打印更新内容

# 写
$ cat <file>		         # 打印文件内容
$ cat <file1> > <file2>	     # 将打印内容写入文件
$ cat <file1> >> <file2>     # 将打印内容添加到文件末尾
#     -n: 打印行号
#     -s: 连续两行以上的空白行替换为一行空白行
$ :> <file>			         # 清除文件内容

# 创建
$ touch <file>           # 新建空文件

# 权限, u=user,g=group,o=other group,a=ugo
$ chmod u=rwx,g=rx,o=x <file>    # rwxr-x--x
$ chmod 777 <file>               # rwxrwxrwx
$ chmod a+r <file>               # ugo + r
$ chmod ug+w,o-w <file>          # ug + w, o - w

# 比较
$ diff <file1> <file2>	     # 比较文件
#      -y: 并排格式输出

```



##  files operation

```shell
# 排序
$ sort <file>			# 将文件各行按ASCII序排序
#      -r: 排逆序
#      -n: 按数值大小排序

# 链接
$ ln -s <file1> <file2>	 # 软链接(符号链接),相当于Windows的快捷方式.lnk
$ ln <file1> <file2>	 # 硬链接,相当于文件有多个名称

# 压缩
$ tar -cvf <file.tar> <dir>	            # 备份
$ tar -zcvf <file.tar.gz> <dir>	        # 备份并压缩
$ tar -xvf <file.tar> [-C <dir>]        # 还原              
$ tar -zxvf <file.tar.gz> [-C <dir>]    # 解压缩并还原
#     -c: 新建备份文件
#     -C: 指定目的目录
#     -f: 指定备份文件(名)
#     -v: 显示执行过程
#     -x: 从备份文件还原文件
#     -z: 通过gzip指令压缩或解压缩文件

$ xz -z <file.tar>                      # 备份
$ xz -d <file.tar.xz>                   # 备份并压缩
#    -k: keep origin file
```



##  dir

```shell
# 查看目录
$ pwd					# current dir

# 切换目录
$ cd					# home
$ cd /			        # root
$ cd ~			        # home
$ cd -				    # last dir
$ cd <dir>			    # specified dir
$ cd ..			        # parent dir 

# 新建与删除
$ mkdir <dir>			# 创建目录
$ mkdir -p <dir1>/<dir2>  # 即使dir1不存在,依然创建
$ rmdir <dir>			# 删除目录
$ rm -r <dir>			# 删除目录及其目录下的所有文件

# 展示目录下文件
$ ls                    # list every file
$ ls -lrt               # list with info
$ ls -lrt s*            # list matching certain name
$ ls -a                 # list hidden file

# 文件系统下查找文件
$ find . -name "*.c"    # by name in current dir
$ find . -mtime -20     # revised in 20d
$ find . -ctime +20     # created before 20d
$ find . -type f        # type f:normal
$ find . -size +100c    # size > 100 Bytes, c,k,M
$ find / -type f -size 0 -exec ls -l {} \;	# find every size 0 normal file in \, show path
```





#  text

```shell
# 查找文本
$ egrep <str>/<re> <file>      # search in file by str/regular expression
$ egrep -r <str>/<re> <dir>    # search in dir by str/regular expression
```





#  logic

```shell
#  &&: command1 succeed, then command2
$ cp sql.txt sql.bak.txt && cat sql.bak.txt

#  ||: command1 failed, then command2
$ ls /proc && echo success || echo failed

#  (;) 

#  |: command1 output as command2 input
$ ls -l /etc | more
$ echo "Hello World" | cat > hello.txt
$ ps -ef | grep <str>

```





#  Shell

```shell
$ echo <str>           # 打印字符串 str

$ clear                # 清空命令行
$ exit                 # 退出命令行

# xargs 是给命令传递参数的一个过滤器,相当于把前面传来的文本作为参数解析
# 一般用法为 command1 |xargs -item command2
$ cat test.txt | xargs	    # 单行输出
$ cat test.txt | xargs -n3	# 多行输出,每行3个参数
$ echo "nameXnameXnameXname" | xargs -dX	# 用X分割并输出

# -I指定占位符,占位符指定参数传入的位置
$ ls *.jpg | xargs -n1 -I {} cp {} /data/images
# 将当前目录下所有.jpg文件备份并压缩
$ find . -type f -name "*.jpg" -print | xargs tar -czvf images.tar.gz
# 按照url-list.txt内容批量下载
$ cat url-list.txt | xargs wget -i
```





#  net

```shell
# 显示hostname
$ hostname
# revise at /etc/sysconfig/network

# ipconfig
$ ifconfig
# revise at /etc/sysconfig/network-scripts/ifcfg-<name>

# 检查端口状态
$ lsof -i:8080

# 显示网络状态
$ netstat -a                    # 显示详细的网络状态
$ netstat -tunlp | grep 8080    # 显示进程8080占用端口状态

# network restart
$ nmcli c reload

```



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



## wget

wget是一个常用的命令行工具，用于下载文件。wget非常稳定，其对带宽很窄、网络不稳定等情况均有很强的适应性。

```shell
$ wget [url]                  # 下载url的资源
$ wget -O [filename] [url]    # 下载并命名,默认名称为url最后一个`/`之后的字符串
$ wget --limit-rate=1M [url]  # 限速下载
$ wget -c [url]               # 断点续传,即继续下载中断的文件
$ wget -b [url]               # 后台下载
$ wget -i [urllistfile]       # 批量下载:从文本文件中读取所有需要下载的url,每个url占一行
$ wget -o download.log [url]  # 将下载信息保存到日志文件,而不显示在终端
```

```shell
$ wget                        # 镜像网站
```





#  system management

```shell
$ sudo                # run as root
$ sudo -u             # run as user

$ ps -ef			  # 查看所有进程: USER, PID, PPID, START, CMD, ...
$ ps -aux             # 查看所有进程: USER, PID, %CPU, %MEM, START, CMD, ...

$ top                 # 显示实时进程,相当于Windows的任务管理器
$ top -d 2            # 每2s更新
$ top -p <num>        # 显示指定进程

$ kill <num>		  # 杀死进程
$ kill -9 <num>	      # 强制杀死进程

$ date                # 打印当前时间

$ shutdown
$ reboot

```





#  system setting

```shell
# redhat package manager
$ rpm -a				          # show packages
$ rpm -qa | grep <name>	  # search package with name
$ rpm -e --nodeps <name>  # uninstall package

# 环境变量
$ env                     # 显示所有环境变量
$ export                  # 显示所有环境变量

$ echo $ENV               # 使用环境变量的值

$ export VAR=10           # 新增或修改环境变量
$ unset VAR               # 删除环境变量
                          # 新增,修改和删除操作仅在当前终端有效
```





#  gcc

```shell
$ gcc -E hello.c -o hello.i    # Preprocess
$ gcc -S hello.c -o hello.s    # Compile
$ gcc -c hello.c -o hello.o    # Assemble
$ gcc hello.c -o hello.exe     # Link
```

