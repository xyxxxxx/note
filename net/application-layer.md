# 应用层

## 域名系统DNS

**DNS(Domain Name System)**是互联网使用的命名系统，用于将人们使用的主机名字转换为IP地址。DNS被设计为联机分布式数据库系统，DNS使大多数名字都在本地进行解析，仅少量解析需要在互联网上通信，因此效率很高

域名到IP地址的解析由分布在互联网上的许多**域名服务器**共同完成，解析过程要点如下：当某个应用进程需要把主机名解析为IP地址时，该进程就调用解析程序(resolver)，并称为DNS的一个客户，把待解析的域名放在DNS请求报文中，以UDP用户数据报方式发送至本地域名服务器. 本地域名服务器在查找域名后，把对应的IP地址放在回答报文中返回. 应用进程获得目的主机的IP地址后开始通信。

若本地域名服务器不能回答该请求，则其向其他域名服务器发送查询请求，直到找到能够回答该请求的域名服务器

DNS规定域名(domain name)中的标号都由英文字母和数字组成，每个标号不超过63个字符，也不区分大小写. 级别最低的域名写在最左边，顶级域名则写在最右边. 完整域名总共不超过255个字符. 各级域名由上一级的域名管理机构管理，顶级域名则由ICANN管理.

<img src="C:\Users\Xiao Yuxuan\Documents\pic\igno36ny3jkhtwgr.PNG" alt="igno36ny3jkhtwgr" style="zoom:67%;" />



**顶级域名TLD(Top Level Domain)** 已有326个，分为三类：

(1) 国家顶级域名nTLD	如中国cn，美国us，英国uk，日本jp等

(2) 通用顶级域名gTLD	如公司com，网络服务机构net，非营利性组织org，国际组织int，美国的教育机构edu，美国的政府部门gov，美国军事部门mil

(3) 基础结构域名，arpa

国家顶级域名下注册的二级域名由国家自行确定，例如日本的教育和企业机构的二级域名定为ac和co

<img src="C:\Users\Xiao Yuxuan\Documents\pic\dphom46kumlk3htnj.PNG" alt="dphom46kumlk3htnj" style="zoom:67%;" />

### 域名服务器

一个DNS服务器负责管辖的范围称为**区(zone)**：

![mgpomu4i6ohnejly35](C:\Users\Xiao Yuxuan\Documents\pic\mgpomu4i6ohnejly35.PNG)

域名服务器分为四种类型：

(1) **根域名服务器(root name server)** 是最高层次的域名服务器，所有根域名服务器都知道所有的顶级域名服务器的域名和IP地址。目前世界上有13组根域名服务器

<img src="C:\Users\Xiao Yuxuan\Documents\pic\fgpmomu4iohnejiy35.PNG" alt="fgpmomu4iohnejiy35" style="zoom:67%;" />

根域名服务器并非直接把待查询的域名转换为IP地址，而是告诉本地域名服务器下一步应该询问的顶级域名服务器

(2) **顶级域名服务器**负责管理在该顶级域名服务器注册的二级域名. 当收到DNS查询请求时就给出回答（IP地址，或下一步询问的域名服务器的IP地址）

(3) **权限域名服务器**负责一个区，如abc.com和y.abc.com各有一个权限域名服务器

(4) **本地域名服务器** 当一台主机发出DNS查询请求时，这个查询请求报文就发送到本地域名服务器，每一个ISP或单位都可以拥有一个本地域名服务器

为提高域名服务器的可靠性，DNS域名服务器把数据复制到几个域名服务器保存

<img src="C:\Users\Xiao Yuxuan\Documents\pic\903ykhgohihjeiongj.PNG" alt="903ykhgohihjeiongj" style="zoom:67%;" />



为了提高DNS查询效率，减轻根域名服务器的负荷，减少互联网上DNS查询报文的数量，在域名服务器中广泛采用了**高速缓存**，用来存放最近查询过的域名以及从何处获得域名映射信息的记录



## 文件传送协议FTP

**FTP(File Transfer Protocol)**是互联网上使用得最广泛的文件传送协议. FTP提供交互式的访问，允许客户知名文件的类型与格式，并允许文件具有存取权限

### 基本工作原理

FTP只提供文件传送的一些基本服务，其使用TCP可靠的运输服务，减少或消除在不同操作系统下处理文件的不兼容性

FTP使用客户服务器方式，一个FTP服务器进程可同时为多个客户进程提供服务。FTP服务器进程由两部分组成：主进程，负责接受新的请求；从属进程，负责处理单个请求. 其工作步骤如下：

![mkfolwrnhtjonjwrgnjhirwg](C:\Users\Xiao Yuxuan\Documents\pic\mkfolwrnhtjonjwrgnjhirwg.PNG)

<img src="C:\Users\Xiao Yuxuan\Documents\pic\i4590hteoi3joi35y.PNG" alt="i4590hteoi3joi35y" style="zoom:67%;" />

![iofdnho3j5ntj5oyujooewf](C:\Users\Xiao Yuxuan\Documents\pic\iofdnho3j5ntj5oyujooewf.PNG)



### 网络文件系统NFS

NFS允许应用进程打开一个远地文件，并在该文件的某一个位置上读写数据.



### 简单文件传送协议TFTP

TFTP也是用客户服务器方式，但使用UDP数据报，其主要优点是：

+ 可用于UDP环境
+ TFTP程序占用内存较小

TFTP的主要特点是：

<img src="C:\Users\Xiao Yuxuan\Documents\pic\jy829jgr2ionhtogrnoqj.PNG" alt="jy829jgr2ionhtogrnoqj" style="zoom:67%;" />



## 远程终端协议TELNET

TELNET提供远程服务，即用户通过TELNET即可使用主机通过TCP连接登录到远地的另一主机上，用户的击键传到远地主机，而远地主机的输出通过TCP连接返回到用户屏幕



## 万维网WWW

**万维网WWW(World Wide Web)** 是一个**大规模的、联机式的信息储藏所**，其用链接的方法能非常方便地从互联网上的一个站点访问另一个站点

万维网上的站点必须都连接在互联网上，每个站点都存放了许多文档

万维网以客户服务器方式工作，浏览器就是用户主机上的万维网客户程序，万维网文档所在的主机运行服务器程序. 客户程序向服务器程序发出请求，服务器程序向客户程序送回客户需要的万维网文档

### 统一资源定位符URL

统一资源定位符URL是互联网上的资源的地址；互联网上的所有资源都有唯一确定的URL. URL的一般形式由以下四部分组成：

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

![dfnklnhjklgnjkh5yq](C:\Users\Xiao Yuxuan\Documents\pic\dfnklnhjklgnjkh5yq.PNG)

### 超文本传送协议HTTP

HTTP是**面向事务的**应用层协议，是万维网上能够可靠地交换文件的重要基础

![gdmikou4n6okly35nuouhjgn](C:\Users\Xiao Yuxuan\Documents\pic\gdmikou4n6okly35nuouhjgn.PNG)

HTTP使用了面向连接的TCP作为运输层协议，保证了数据的可靠传输，但是HTTP协议本身是无连接的。HTTP协议是无状态的(stateless)，即每次访问同一个服务器上的页面时，服务器的响应相同。

<img src="C:\Users\Xiao Yuxuan\Documents\pic\etjiojy5oirgjkg25yg.PNG" alt="etjiojy5oirgjkg25yg" style="zoom: 80%;" />

HTTP/1.0的主要缺点是，每请求一个文档就需要两倍的RTT开销. HTTP/1.1协议则使用了持续连接(persistent connection)，有两种工作方式：非流水线方式(without pipelining)和流水线方式(with pipelining)



**代理服务器**

**代理服务器(proxy server)**是一种网络实体，又称为万维网高速缓存(Web cache)，其把最近的一些请求和响应暂存在本地磁盘中，当新请求到达时，若代理服务器发现这个请求与暂时存放的请求相同，则返回暂存的响应. 代理服务器可以在客户端或服务器端工作，也可在中间系统工作。

<img src="C:\Users\Xiao Yuxuan\Documents\pic\fdiony35mojgrkhwt.PNG" alt="fdiony35mojgrkhwt" style="zoom:67%;" />

![jy358iogrmwjlwgjnfewrg](C:\Users\Xiao Yuxuan\Documents\pic\jy358iogrmwjlwgjnfewrg.PNG)

![fdgoiyn3u6ojgrnkwht](C:\Users\Xiao Yuxuan\Documents\pic\fdgoiyn3u6ojgrnkwht.PNG)



**HTTP报文**

HTTP报文分为请求报文和响应报文：

![dgolkhmklgfnjkbgrkhwg](C:\Users\Xiao Yuxuan\Documents\pic\dgolkhmklgfnjkbgrkhwg.PNG)

HTTP是**面向文本的(text-oriented)**，报文中每个字段都是ASCII码串，字段长度也不固定. 完整的HTTP请求报文例如：

![gofeino42jtnjgnjkiqt4fe](C:\Users\Xiao Yuxuan\Documents\pic\gofeino42jtnjgnjkiqt4fe.PNG)

**状态码(Status-Code)**是三位数字，分为5大类，分别以不同数字开头：

+ 1xx表示通知信息
+ 2xx表示成功
+ 3xx表示重定向
+ 4xx表示客户的差错
+ 5xx表示服务器的差错

例如下面三种状态行：

<img src="C:\Users\Xiao Yuxuan\Documents\pic\ntroinhjygrkhwvsfs.PNG" alt="ntroinhjygrkhwvsfs" style="zoom:67%;" />



**Cookie**

Cookie是在HTTP服务器和客户之间传递的状态信息. 当用户A浏览某个使用Cookie的网站时，该网站的服务器就为A产生一个<u>唯一的识别码</u>，并以此为索引在服务器的<u>后端数据库中产生一个项目</u>，然后在给A的HTTP响应报文中添加首部行Set-cookie，例如

```
Set-cookie:31d4d96e407aad42
```

A收到响应报文后，浏览器在其管理的特定Cookie文件中添加一行，包括服务器主机名和Set-cookie后的识别码，并放到HTTP请求报文的Cookie首部行中

```
Cookie:31d4d96e407aad42
```

于是网站就能跟踪用户31d4d96e407aad42在该网站的活动

使用Cookie可能造成用户隐私的保护问题，网站服务器可能把用户的信息出卖给第三方



### 万维网的文档

**超文本标记语言HTML**

HTML(HyperText Markup Language)是万维网页面的标准语言. 

**可扩展标记语言XML**

**层叠样式表CSS**



通用网关接口CGI定义了动态文档应如何创建，万维网服务器中新增加的应用程序称为CGI程序：

<img src="C:\Users\Xiao Yuxuan\Documents\pic\gkpfohju46oiyrejdgowr.PNG" alt="gkpfohju46oiyrejdgowr" style="zoom:67%;" />

CGI程序的正式名字是CGI**脚本(script)**，脚本指被另一个程序解释或执行的程序. 有一些专门的脚本语言，如Perl，REXX，JavaScript等



**活动文档(active document)**技术支持屏幕连续更新，即把所有的工作都转移给浏览器端. 每当浏览器请求一个活动文档，服务器就返回一段活动文档程序副本，并在浏览器端运行

![fdionywkrgljnvjkw4t](C:\Users\Xiao Yuxuan\Documents\pic\fdionywkrgljnvjkw4t.PNG)



### 信息检索系统

在万维网中用来进行搜索的工具称为**搜索引擎(search engine)**.



![fgidondskln24iutej](C:\Users\Xiao Yuxuan\Documents\pic\fgidondskln24iutej.PNG)



![vjihpjmvdklsny35lgr](C:\Users\Xiao Yuxuan\Documents\pic\vjihpjmvdklsny35lgr.PNG)

**垂直搜索引擎(Vertical Search Engine)**，**元搜索引擎(Meta Search Engine)**



### 博客和微博



### 社交网站



## 电子邮件

电子邮件最重要的两个标准即：SMTP和互联网文本报文格式.

一个电子邮件系统应具有三个主要组成构件：用户代理、邮件服务器以及邮件发送协议和邮件读取协议. POP3是**邮局协议**的版本3.

<img src="C:\Users\Xiao Yuxuan\Documents\pic\lafmkglwngj4ugtbhkwtg.PNG" alt="lafmkglwngj4ugtbhkwtg" style="zoom:67%;" />

用户代理UA(User Agent)即用户与电子邮件系统的接口，在大多数情况下是运行在用户电脑中的一个程序

发送步骤：

![adopjgywpojgnjohw5y](C:\Users\Xiao Yuxuan\Documents\pic\adopjgywpojgnjohw5y.PNG)

### SMTP

SMTP规定了两个相互通信的SMTP进程之间应如何交换信息，使用客户服务器方式。

### POP3和IMAP



### 基于万维网的电子邮件



### 通用互联网扩充MIME



## 动态主机配置协议DHCP

DHCP提供了**即插即用连网(plug-and-play networking)**机制，允许一台计算机加入新的网络和获取IP地址而不用手工参与。

![fspiogjhyowrjt90tjgwroi](C:\Users\Xiao Yuxuan\Documents\pic\fspiogjhyowrjt90tjgwroi.PNG)

![fpsojegionfsjbwy5isu](C:\Users\Xiao Yuxuan\Documents\pic\fpsojegionfsjbwy5isu.PNG)

![nvsfon35yuognjgwr](C:\Users\Xiao Yuxuan\Documents\pic\nvsfon35yuognjgwr.PNG)

![jh90jheoifsnoiwrtyongjwr](C:\Users\Xiao Yuxuan\Documents\pic\jh90jheoifsnoiwrtyongjwr.PNG)



## 简单网络管理协议SNMP



## 系统调用和应用编程接口API









## P2P应用

P2P应用即具有P2P体系结构的应用，而P2P体系结构即在网络应用中没有或有很少的服务器，绝大多数交互都是以对等方式进行的. P2P应用范围很广，例如文件分发、实时音频或视频会议、数据库系统、网络服务支持等

以P2P文件分发为例，其不需要使用集中式的媒体服务器，而所有的音频/视频文件都是在普通的互联网用户之间传输的. 

### 具有集中目录服务器的P2P工作方式

![vfsionh3yjngjkwg](C:\Users\Xiao Yuxuan\Documents\pic\vfsionh3yjngjkwg.PNG)

### 具有全分布式结构的P2P文件共享程序

![nfdbjklnhu64oynhtgjeht](C:\Users\Xiao Yuxuan\Documents\pic\nfdbjklnhu64oynhtgjeht.PNG)

