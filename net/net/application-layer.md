[toc]



# 域名系统DNS

**DNS(Domain Name System)**是互联网使用的命名系统，用于将人们使用的主机名字转换为IP地址。DNS被设计为联机分布式数据库系统，DNS使大多数名字都在本地进行解析，仅少量解析需要在互联网上通信，因此效率很高。

域名到IP地址的解析由分布在互联网上的许多**域名服务器**共同完成，解析过程要点如下：当某个应用进程需要把主机名解析为IP地址时，该进程就调用解析程序(resolver)，并称为DNS的一个客户，把待解析的域名放在DNS请求报文中，以UDP用户数据报方式发送至本地域名服务器。本地域名服务器在查找域名后，把对应的IP地址放在回答报文中返回。应用进程获得目的主机的IP地址后开始通信。

若本地域名服务器不能回答该请求，则其向其他域名服务器发送查询请求，直到找到能够回答该请求的域名服务器。

DNS规定域名(domain name)中的标号都由英文字母和数字组成，每个标号不超过63个字符，也不区分大小写。级别最低的域名写在最左边，顶级域名则写在最右边，完整域名总共不超过255个字符。各级域名由上一级的域名管理机构管理，顶级域名则由ICANN管理。

```
mail.cctv.com
三级  二级  顶级域名
```



**顶级域名TLD(Top Level Domain)** 已有326个，分为三类：

(1) 国家顶级域名nTLD，如中国：cn，美国：us，英国：uk，日本：jp

(2) 通用顶级域名gTLD，如公司：com，网络服务机构：net，非营利性组织：org，国际组织：int，美国教育机构：edu，美国政府部门：gov，美国军事部门：mil

(3) 基础结构域名，如arpa

国家顶级域名下注册的二级域名由国家自行确定，例如日本的教育和企业机构的二级域名定为ac和co

![](https://raw.githubusercontent.com/xyxxxxx/image/master/dphom46kumlk3htnj.PNG)



## 域名服务器

一个DNS服务器负责管辖的范围称为**区(zone)**：

![](https://raw.githubusercontent.com/xyxxxxx/image/master/mgpomu4i6ohnejly35.PNG)

域名服务器分为四种类型：

(1) **根域名服务器(root name server)** 是最高层次的域名服务器，所有根域名服务器都知道所有的顶级域名服务器的域名和IP地址。目前世界上有13组根域名服务器

![](https://raw.githubusercontent.com/xyxxxxx/image/master/fgpmomu4iohnejiy35.PNG)

根域名服务器并非直接把待查询的域名转换为IP地址，而是告诉本地域名服务器下一步应该询问的顶级域名服务器

(2) **顶级域名服务器**负责管理在该顶级域名服务器注册的二级域名. 当收到DNS查询请求时就给出回答（IP地址，或下一步询问的域名服务器的IP地址）

(3) **权限域名服务器**负责一个区，如abc.com和y.abc.com各有一个权限域名服务器

(4) **本地域名服务器** 当一台主机发出DNS查询请求时，这个查询请求报文就发送到本地域名服务器，每一个ISP或单位都可以拥有一个本地域名服务器

为提高域名服务器的可靠性，DNS域名服务器把数据复制到几个域名服务器保存

![](https://raw.githubusercontent.com/xyxxxxx/image/master/903ykhgohihjeiongj.PNG)



为了提高DNS查询效率，减轻根域名服务器的负荷，减少互联网上DNS查询报文的数量，在域名服务器中广泛采用了**高速缓存**，用来存放最近查询过的域名以及从何处获得域名映射信息的记录。





# 文件传送协议FTP

**FTP(File Transfer Protocol)**是互联网上使用得最广泛的文件传送协议. FTP提供交互式的访问，允许客户知名文件的类型与格式，并允许文件具有存取权限

## 基本工作原理

FTP只提供文件传送的一些基本服务，其使用TCP可靠的运输服务，减少或消除在不同操作系统下处理文件的不兼容性

FTP使用客户服务器方式，一个FTP服务器进程可同时为多个客户进程提供服务。FTP服务器进程由两部分组成：主进程，负责接受新的请求；从属进程，负责处理单个请求. 其工作步骤如下：

![](https://raw.githubusercontent.com/xyxxxxx/image/master/mkfolwrnhtjonjwrgnjhirwg.PNG)

![](https://raw.githubusercontent.com/xyxxxxx/image/master/i4590hteoi3joi35y.PNG)

![](https://raw.githubusercontent.com/xyxxxxx/image/master/iofdnho3j5ntj5oyujooewf.PNG)





## 网络文件系统NFS

NFS允许应用进程打开一个远地文件，并在该文件的某一个位置上读写数据。



## 简单文件传送协议TFTP

TFTP也是用客户服务器方式，但使用UDP数据报，其主要优点是：

+ 可用于UDP环境
+ TFTP程序占用内存较小

TFTP的主要特点是：

![](https://raw.githubusercontent.com/xyxxxxx/image/master/jy829jgr2ionhtogrnoqj.PNG)





# 远程终端协议TELNET

TELNET提供远程服务，即用户通过TELNET即可使用主机通过TCP连接登录到远地的另一主机上，用户的击键传到远地主机，而远地主机的输出通过TCP连接返回到用户屏幕。





# 万维网WWW

**万维网WWW(World Wide Web)** 是一个**大规模的、联机式的信息储藏所**，其用链接的方法能非常方便地从互联网上的一个站点访问另一个站点。

万维网上的站点必须都连接在互联网上，每个站点都存放了许多文档。

万维网以客户服务器方式工作，浏览器就是用户主机上的万维网客户程序，万维网文档所在的主机运行服务器程序. 客户程序向服务器程序发出请求，服务器程序向客户程序送回客户需要的万维网文档。



## 统一资源定位符URL

URL（Uniform Resource Locator，统一资源定位符）是互联网上的资源的地址；互联网上的所有资源都有唯一确定的URL。URL的一般形式由以下四部分组成：

```
<协议>://<主机>:<端口>/<路径>

//目前最常用的协议是https, http, ftp
```

访问万维网的网点需要使用HTTP/HTTPS协议，其URL一般形式为：

```
http://<主机>:<端口>/<路径>
https://<主机>:<端口>/<路径>
```

HTTP的默认端口号是80，HTTPS的默认端口号是443，通常可省略. 若省略路径，则URL指向互联网上的某个**主页(home page)**

![](https://raw.githubusercontent.com/xyxxxxx/image/master/dfnklnhjklgnjkh5yq.PNG)



## 超文本传送协议HTTP

HTTP是**面向事务的**应用层协议，是万维网上能够可靠地交换文件的重要基础。

![](https://raw.githubusercontent.com/xyxxxxx/image/master/gdmikou4n6okly35nuouhjgn.PNG)

HTTP使用了面向连接的TCP作为运输层协议，保证了数据的可靠传输，但是HTTP协议本身是<u>无连接的</u>。HTTP协议是<u>无状态的(stateless)</u>，即每次访问同一个服务器上的页面时，服务器的响应相同。

HTTP协议默认端口号是80。

![](https://raw.githubusercontent.com/xyxxxxx/image/master/etjiojy5oirgjkg25yg.PNG)



### HTTP报文

HTTP报文分为请求报文和响应报文：

![](https://raw.githubusercontent.com/xyxxxxx/image/master/dgolkhmklgfnjkbgrkhwg.PNG)

HTTP是**面向文本的(text-oriented)**，报文中每个字段都是ASCII码串，字段长度也不固定。



请求报文中的方法有4种，分别是`GET`，`POST`，`PUT`，`DELETE`：

+ `GET`用于信息获取，而且应该是安全的和幂等的

  ```
  GET /books/?sex=female&name=Alice HTTP/1.1
  Host: www.example.com
  User-Agent: Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.7.6)
  Gecko/20050225 Firefox/1.0.1
  Connection: Keep-Alive
  ```

+ `POST`表示可能修改服务器上资源的请求

  ```
  POST / HTTP/1.1
  Host: www.example.com
  User-Agent: Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.7.6)
  Gecko/20050225 Firefox/1.0.1
  Content-Type: application/x-www-form-urlencoded
  Content-Length: 40
  Connection: Keep-Alive
  
  sex=female&name=Alice
  ```

  

响应报文示例

```
HTTP/1.1 200 OK

Server:Apache Tomcat/5.0.12
Date:Mon,6Oct2003 13:23:42 GMT
Content-Length:112

<html>...
```





**状态码(Status-Code)**是三位数字，分为5大类，分别以不同数字开头：

+ 1xx表示通知信息
+ 2xx表示成功
+ 3xx表示重定向
+ 4xx表示客户的差错
+ 5xx表示服务器的差错

常见的状态码有

| `200 OK`                    | 客户端请求成功                                               |
| --------------------------- | ------------------------------------------------------------ |
| `301 Moved Permanently`     | 请求永久重定向                                               |
| `302 Moved Temporarily`     | 请求临时重定向                                               |
| `304 Not Modified`          | 文件未修改，可以直接使用缓存的文件。                         |
| `400 Bad Request`           | 由于客户端请求有语法错误，不能被服务器所理解。               |
| `401 Unauthorized`          | 请求未经授权。这个状态代码必须和WWW-Authenticate报头域一起使用。 |
| `403 Forbidden`             | 服务器收到请求，但是拒绝提供服务。服务器通常会在响应正文中给出不提供服务的原因 |
| `404 Not Found`             | 请求的资源不存在，例如输入了错误的URL                        |
| `500 Internal Server Error` | 服务器发生不可预期的错误，导致无法完成客户端的请求。         |
| `503 Service Unavailable`   | 服务器当前不能够处理客户端的请求，在一段时间之后，服务器可能会恢复正常。 |



### 持续连接

HTTP/1.0的主要缺点是，每请求一个文档就需要两倍的RTT开销。HTTP/1.1协议则使用了持续连接(persistent connection)，有两种工作方式：非流水线方式(without pipelining)和流水线方式(with pipelining)。



### 会话跟踪

客户端打开与服务器的连接发出请求到服务器响应客户端请求的全过程称为会话。会话跟踪指的是对同一个用户对服务器的连续的请求和接受响应的监视。浏览器与服务器之间的通信是通过HTTP协议进行通信的，而HTTP协议是”无状态”的协议，它不能保存客户的信息，即一次响应完成之后连接就断开了，下一次的请求需要重新连接，这样就需要判断是否是同一个用户，所以才有会话跟踪技术来实现这种要求。

会话跟踪常用以下方法：

**URL 重写**

URL重写的技术就是在URL结尾添加一个附加数据以标识该会话，把会话ID通过URL的信息传递过去，以便在服务器端进行识别不同的用户。



**隐藏表单域**

将会话ID添加到HTML表单元素中提交到服务器，此表单元素并不在客户端显示



**Cookie**

Cookie是在HTTP服务器和客户之间传递的状态信息。当用户A浏览某个使用Cookie的网站时，该网站的服务器就为A产生一个<u>唯一的识别码</u>，并以此为索引在服务器的<u>后端数据库中产生一个项目</u>，然后在给A的HTTP响应报文中添加首部行Set-cookie，例如

```
Set-cookie:31d4d96e407aad42
```

A收到响应报文后，浏览器在其管理的特定Cookie文件中添加一行，包括服务器主机名和Set-cookie后的识别码，并放到HTTP请求报文的Cookie首部行中

```
Cookie:31d4d96e407aad42
```

于是网站就能跟踪用户31d4d96e407aad42在该网站的活动。

Cookie主要应用于：

+ 会话状态管理（如用户登录状态、购物车、游戏分数或其它需要记录的信息）
+ 个性化设置（如用户自定义设置、主题等）
+ 浏览器行为跟踪（如跟踪分析用户行为等）

Cookie具有过期时间，到期后会自动删除。

Cookie可以被客户端禁用。

使用Cookie可能造成用户隐私的保护问题，网站服务器可能把用户的信息出卖给第三方。



**Session**





### 代理服务器

**代理服务器(proxy server)**是一种网络实体，又称为万维网高速缓存(Web cache)，其把最近的一些请求和响应暂存在本地磁盘中，当新请求到达时，若代理服务器发现这个请求与暂时存放的请求相同，则返回暂存的响应. 代理服务器可以在客户端或服务器端工作，也可在中间系统工作。

![](https://raw.githubusercontent.com/xyxxxxx/image/master/fdiony35mojgrkhwt.PNG)

![](https://raw.githubusercontent.com/xyxxxxx/image/master/jy358iogrmwjlwgjnfewrg.PNG)

![](https://raw.githubusercontent.com/xyxxxxx/image/master/fdgoiyn3u6ojgrnkwht.PNG)





### 万维网的文档

**超文本标记语言HTML**

**可扩展标记语言XML**

**层叠样式表CSS**



通用网关接口CGI定义了动态文档应如何创建，万维网服务器中新增加的应用程序称为CGI程序：

![](https://raw.githubusercontent.com/xyxxxxx/image/master/gkpfohju46oiyrejdgowr.PNG)

CGI程序的正式名字是CGI**脚本(script)**，脚本指被另一个程序解释或执行的程序。有一些专门的脚本语言，如Perl，REXX，JavaScript等。



**活动文档(active document)**技术支持屏幕连续更新，即把所有的工作都转移给浏览器端。每当浏览器请求一个活动文档，服务器就返回一段活动文档程序副本，并在浏览器端运行。

![](https://raw.githubusercontent.com/xyxxxxx/image/master/fdionywkrgljnvjkw4t.PNG)





## 电子邮件

电子邮件最重要的两个标准即：SMTP和互联网文本报文格式.

一个电子邮件系统应具有三个主要组成构件：用户代理、邮件服务器以及邮件发送协议和邮件读取协议. POP3是**邮局协议**的版本3.

![](https://raw.githubusercontent.com/xyxxxxx/image/master/lafmkglwngj4ugtbhkwtg.PNG)

用户代理UA(User Agent)即用户与电子邮件系统的接口，在大多数情况下是运行在用户电脑中的一个程序。

发送步骤：

![](https://raw.githubusercontent.com/xyxxxxx/image/master/adopjgywpojgnjohw5y.PNG)

### SMTP

SMTP规定了两个相互通信的SMTP进程之间应如何交换信息，使用客户服务器方式。

### POP3和IMAP



### 基于万维网的电子邮件



### 通用互联网扩充MIME



## 动态主机配置协议DHCP

DHCP提供了**即插即用连网(plug-and-play networking)**机制，允许一台计算机加入新的网络和获取IP地址而不用手工参与。

<img src="C:\Users\Xiao Yuxuan\Documents\pic\fspiogjhyowrjt90tjgwroi.PNG" alt="fspiogjhyowrjt90tjgwroi" style="zoom:67%;" />

<img src="C:\Users\Xiao Yuxuan\Documents\pic\fpsojegionfsjbwy5isu.PNG" alt="fpsojegionfsjbwy5isu" style="zoom:67%;" />

<img src="C:\Users\Xiao Yuxuan\Documents\pic\nvsfon35yuognjgwr.PNG" alt="nvsfon35yuognjgwr" style="zoom:67%;" />

<img src="C:\Users\Xiao Yuxuan\Documents\pic\jh90jheoifsnoiwrtyongjwr.PNG" alt="jh90jheoifsnoiwrtyongjwr" style="zoom:67%;" />



## 简单网络管理协议SNMP



## 系统调用和应用编程接口API









## P2P应用

P2P应用即具有P2P体系结构的应用，而P2P体系结构即在网络应用中没有或有很少的服务器，绝大多数交互都是以对等方式进行的. P2P应用范围很广，例如文件分发、实时音频或视频会议、数据库系统、网络服务支持等

以P2P文件分发为例，其不需要使用集中式的媒体服务器，而所有的音频/视频文件都是在普通的互联网用户之间传输的. 

### 具有集中目录服务器的P2P工作方式

<img src="C:\Users\Xiao Yuxuan\Documents\pic\vfsionh3yjngjkwg.PNG" alt="vfsionh3yjngjkwg" style="zoom:67%;" />

### 具有全分布式结构的P2P文件共享程序

<img src="C:\Users\Xiao Yuxuan\Documents\pic\nfdbjklnhu64oynhtgjeht.PNG" alt="nfdbjklnhu64oynhtgjeht" style="zoom:67%;" />





# 安全外壳协议SSH

> 参考：
>
> [SSH原理与运用（一）：远程登录](http://www.ruanyifeng.com/blog/2011/12/ssh_remote_login.html)
>
> [SSH 教程](https://wangdoc.com/ssh/index.html)



## ssh

### 连接

```shell
# 连接服务器
$ ssh host

# 以指定用户名连接,默认使用ssh的当前用户名
$ ssh user@host
$ ssh -l user host

# 连接指定端口,默认为22端口
$ ssh -p port host
```



### 命令

```shell
# 执行远程命令
$ ssh user@host command

# 执行远程命令(启动互动式Shell环境)
$ ssh -t user@host command
```



### 密钥

```shell
# 生成密钥,使用RSA/DSA加密算法
$ ssh-keygen -t rsa/dsa

# 显示所有公钥
$ ls -l ~/.ssh/id_*.pub

# 自动上传公钥到服务器
$ ssh-copy-id -i key_file user@host

# 手动上传公钥到服务器
$ cat ~/.ssh/id_rsa.pub | ssh user@host "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"

# 删除服务器的公钥指纹,当服务器公钥变更时需要执行此操作
$ ssh-keygen -R host
```



## sshd

### 启动

```shell
# 启动/停止sshd
$ sudo sshd
$ sudo /usr/sbin/sshd     # 使用绝对路径以确保启动真正的sshd
$ sudo pkill sshd

# 启动/停止/重启sshd服务
$ sudo systemctl start sshd.service
$ sudo systemctl stop sshd.service
$ sudo systemctl restart sshd.service

# 设置开机自启动
$ sudo systemctl enable sshd.service
```



### 密码

```shell
# 修改连接密码
$ sudo passwd
```



### 密钥

```shell
# 对于每种当前主机密钥中不存在的密钥类型(rsa,dsa,ecdsa,ed25519),生成主机密钥,使用默认设置
$ sudo ssh-keygen -A

# 查看主机公钥的指纹
$ ssh-keygen -l -f /etc/ssh/ssh_host_rsa_key.pub
```



## 端口转发

