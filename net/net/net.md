**计算机网络**由若干**节点**（node）和连接这些节点的**链路**（link）组成. 网络之间还可以通过路由器连接起来构成覆盖范围更大的计算机网络，称为**互联网**. 与网络相连的计算机称为主机（host）

**电路交换** 步骤为建立连接（占用通信资源）→通话（持续占用通信资源）→释放连接（归还通信资源）的交换方式；在通话的全部时间内，通话的双方始终占用端到端的通信资源

**分组交换** 

<img src="C:\Users\Xiao Yuxuan\Documents\pic\dgfkm4ju6khteb.PNG" alt="dgfkm4ju6khteb" style="zoom:80%;" />

分组（packet）又称为包，是互联网中传送的数据单元

<img src="C:\Users\Xiao Yuxuan\Documents\pic\iegtonjtkgvbnrhehj.PNG" alt="iegtonjtkgvbnrhehj" style="zoom:80%;" />

<img src="C:\Users\Xiao Yuxuan\Documents\pic\afidony4jhngvwjrbk.PNG" alt="afidony4jhngvwjrbk" style="zoom:80%;" />





计算机网络组成元素：

+ 主机
+ 交换节点
+ 通信链路
+ 拓扑结构
+ 通信软件



### 计算机网络分类

**按网络覆盖范围**

**个域网（PAN）**	蓝牙

**局域网（LAN）**	以太网

**无线局域网（WLAN）** WiFi

**城域网（MAN）**	WiMax

**广域网（WAN）**	蜂窝网络



**按网络传输技术**

**广播网络**只有一个通信信道，网络上所有主机/节点共享该信道通信

单播：一对一；组播：一对多；广播：一对全体

**点对点传输**



**按数据交换技术分类**

**电路交换**

**分组交换**



### 计算机网络性能

**速率**

网络中的速率指数据的传送速率，单位是bit/s（或bps）

**带宽bandwidth**

网络中的带宽表示某通道传送数据的能力，即某信道能通过的最高数据率，单位是bit/s

**吞吐量throughput**

吞吐量表示单位时间通过某个网络（或信道，接口）的实际数据量. 显然其受带宽或网络的额定速率的限制

**时延/延迟/delay/latency**

时延指数据从网络一端传送到另一端所需的时间，其由以下几部分组成：

(1) **发送时延** 主机或路由器发送数据帧所需要的时间，等于数据长度与发送速率之比

(2) **传播时延** 电磁波在信道中传播一定的距离需要花费的时间

(3) **处理时延** 路由器收到分组时要花费一定时间进行处理

(4) **排队时延** 分组进入路由器后要先在输入队列中排队等待处理

总时延即以上四项之和

**时延带宽积**

时延带宽积=传播时延×带宽，可以将链路比作管道，则时延带宽积表示管道的体积，也称为链路长度

**往返时间RTT**

双向交互一次需要的时间

**利用率**

信道或网络的利用率过高会产生非常大的时延



### 计算机网络体系结构

**网络协议**由语法，语义和同步三要素组成

复杂的计算机网络协议的结构应该是层次式的，例如：

<img src="C:\Users\Xiao Yuxuan\Documents\pic\geopjyrkgmwrjls.PNG" alt="geopjyrkgmwrjls" style="zoom:80%;" />

计算机网络的各层及其协议的集合就是网络的**体系结构**

<img src="C:\Users\Xiao Yuxuan\Documents\pic\opetgmkhltn6hju3bt.PNG" alt="opetgmkhltn6hju3bt" style="zoom:80%;" />

(1) **物理层physical layer**



(2) **链路层data link layer**



(3) **网络层network layer**

网络层负责为分组交换网上的不同主机提供通信服务。发送数据时，网络层把运输层产生的报文段或用户数据报封装成分组或包进行传送。TCP/IP体系中，由于网络层使用IP协议，因此分组也叫做IP数据报，或简称数据报。

(4) **运输层transport layer**

运输层负责向进程之间的通信提供<u>通用的数据传输</u>服务，主要使用以下两种协议：

+ 传输控制协议TCP（Transmission Control Protocol） 提供<u>面向连接的，可靠的</u>数据传输服务，其数据传输的单位是**报文段**；
+ 用户数据报协议UDP（User Datagram Protocol） 提供<u>无连接的，尽最大努力的</u>数据传输服务（不保证数据传输的可靠性），其数据传输单位是**用户数据报**

(5) **应用层application layer**

应用层通过进程间的交互来完成特定网络应用，应用层协议定义<u>进程间通信和交互的规则</u>，互联网中的应用层协议例如域名系统DNS，支持万维网的HTTP协议，支持电子邮件的SMTP协议等. 应用层交互的数据单元称为报文

<img src="C:\Users\Xiao Yuxuan\Documents\pic\svfdjonh4jkgr.PNG" alt="svfdjonh4jkgr" style="zoom:67%;" />

**实体**表示任何可以发送或接收信息的硬件或软件，**协议**是控制两个对等实体进行通信的规则的集合；在协议的控制下，两个对等实体的通信使得本层能够向上一层提供服务。服务是垂直的，协议是水平的



# TCP/IP体系

<img src="C:\Users\Xiao Yuxuan\Documents\pic\rfgiowjhyeionhetj.PNG" alt="rfgiowjhyeionhetj" style="zoom:80%;" />

<img src="C:\Users\Xiao Yuxuan\Documents\pic\iowrjgnyiognjrwwr.PNG" alt="iowrjgnyiognjrwwr" style="zoom:80%;" />

everything over IP & IP over everything

<img src="C:\Users\Xiao Yuxuan\Documents\pic\jiorwgjfiwoethnjlhet.PNG" alt="jiorwgjfiwoethnjlhet" style="zoom:80%;" />

