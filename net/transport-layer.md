# 运输层

## 概述

通信的真正端点不是主机，而是主机中的进程

![](https://i.loli.net/2020/08/12/tnBdPwMhJvzGY2x.png)

网络层为<u>主机之间</u>提供逻辑通信，而运输层为<u>应用进程之间</u>提供端到端的逻辑通信

运输层对收到的报文进行**差错检测**

运输层有两种运输协议：**面向连接的TCP**和**无连接的UDP**

### 主要协议

![](https://i.loli.net/2020/08/12/VlwuAUD8XMacO6s.png)

**用户数据报协议UDP(User Datagram Protocol)** 在传送数据之前<u>不需要先建立连接</u>，远地主机的运输层收到UDP报文后，不需要给出任何确认

**传输控制协议TCP(Transmission Control Protocol)** 提供<u>面向连接的服务</u>，在传送数据之前必须先建立连接，数据传送结束后要释放连接。由于TCP提供可靠的面向连接的运输服务，因此增加了许多开销，如确认、流量控制、计时器和连接管理等.

![](https://raw.githubusercontent.com/xyxxxxx/image/master/teiohuj47iujhmgklejhe.PNG)

运输层使用**协议端口号(protocol port number)**，简称为**端口(port)**，使用16位端口号（1~65535）来标志一个端口，端口号只具有本地意义，只标志本计算机应用层中各进程和运输层交互时的层间接口。端口号分为以下两大类：

+ 服务器端使用的端口号

  + 熟知端口号或系统端口号（0~1023）被指派给TCP/IP最重要的一些应用程序

  + 登记端口号（1024~49151），供没有熟知端口号的应用程序使用

+ 客户端使用的端口号（49152~65535）仅在客户进程运行时才动态选择，也称为短暂端口号

![](https://raw.githubusercontent.com/xyxxxxx/image/master/wti9yogj5io2hnjktb.PNG)



## UDP

### 概述

UDP只在IP的数据报服务的基础上增加了很少的功能，即复用和分用，差错检测的功能

UDP的主要特点：

+ **无连接的**，即发送数据之前不需要建立连接

+ **尽最大努力交付**，即不保证可靠交付、

+ **面向报文**，即对应用程序向下交付的报文，添加首部后就向下交付IP层

  ![](https://raw.githubusercontent.com/xyxxxxx/image/master/tjieoyjnoigrwno245ut.PNG)

+ **没有拥塞控制**，网络出现的拥塞不会使源主机的发送速率降低

+ 支持一对一，一对多，多对一和多对多的交互通信

+ 首部开销小，仅8个字节

### 首部格式

![](https://raw.githubusercontent.com/xyxxxxx/image/master/wrjiogtjoi56ynhejto.PNG)

![](https://raw.githubusercontent.com/xyxxxxx/image/master/r90y3j5hoitn3jh6u.PNG)

![](https://raw.githubusercontent.com/xyxxxxx/image/master/j3ih5ojuy5iohnwvjkwgt.PNG)

## TCP

### 概述

TCP的主要特点：

+ **面向连接的运输层协议**，即应用程序在使用TCP协议之前必须先建立TCP连接，传送数据完毕之后释放连接
+ 每条TCP连接只能有两个**端点(endpoint)**，即只能是点对点连接
+ 提供**可靠交付的服务**，即确保TCP连接传送的数据无差错，不丢失，不重复，按照顺序
+ 提供**全双工通信**，即允许双方进程在任何时候都能发送数据，TCP连接的两端都设有发送缓存和接收缓存
+ **面向字节流**

![](https://raw.githubusercontent.com/xyxxxxx/image/master/j3y5iojgtioenhetjkteg.PNG)



### TCP连接

TCP连接的端点是socket

### 可靠传输

**停止等待协议**

![](https://raw.githubusercontent.com/xyxxxxx/image/master/350ypmgrwknejlfeq.PNG)



+ A在发送完一个分组后，必须暂时保留已发送的分组的副本，收到相应的确认后才能清除
+ 分组和确认分组必须编号
+ 超时计时器设置的重传时间应比分组传输的平均往返时间更长一些 

![](https://raw.githubusercontent.com/xyxxxxx/image/master/k46okytepowrmg.PNG)

+ B收到重传的分组：不向上层交付，但向A发送确认
+ A收到重复的确认：收下即丢弃

上述机制实现了在不可靠的传输网络上实现可靠通信

**信道利用率**

![](https://raw.githubusercontent.com/xyxxxxx/image/master/iohj3yuiongjbeth.PNG)
$$
U=\frac{T_D}{T_D+RTT+T_A}
$$
![](https://raw.githubusercontent.com/xyxxxxx/image/master/xvnjknhtjkyehnjke5y3.PNG)

**连续ARQ协议**

![](https://raw.githubusercontent.com/xyxxxxx/image/master/y4j0i5jgrwiowh.PNG)

略



### 首部格式

![](https://raw.githubusercontent.com/xyxxxxx/image/master/04u6jkthgiotniy4tfw.PNG)

(1) **源端口和目的端口**，各2字节

(2) **序号**，4字节（0~2^32-1）. TCP连接中传送的字节流中的每一个字节都按顺序编号，序号字段指本报文段所发送的数据的第一个字节的序号

(3) **确认号**，4字节，是期望收到对方下一个报文段的第一个数据字节的序号. 若确认号=N，则序号N-1之前的所有数据都已正确收到

(4) **数据偏移**，4位，指出首部长度，单位为4字节

(5) **保留**，6位

(6) **紧急URG(URGent)** URG=1表示此报文段中有紧急数据，应尽快传送

(7) **确认ACK(ACKnowledgment)** 

(8) **推送PSH(PuSH)** 接收方TCP收到PSH=1的报文段，就尽快地交付接受应用进程

(9) **复位RST(ReSeT)** RST=1表示TCP连接出现严重差错，必须释放连接，再重新建立连接

(10) **同步SYN(SYNchronization)** SYN=1表示这是一个连接请求或连接接受报文

(11) **终止FIN(FINis)** FIN=1表示数据发送完毕，要求释放连接

(12) **窗口**，2字节，表示从本报文段首部中的确认号算起，接收方目前允许对方发送的数据量. 窗口为零时也可以发送紧急数据

(13) **检验和**，2字节，类似UDP数据报

(14) **紧急指针**，2字节，仅在URG=1时有意义，指出本报文段中紧急数据的字节数

(15) **选项**，最长40字节. **最大报文段长度MSS(Maximum Segment Size)**，是每一个TCP报文段中<u>数据字段的最大长度</u>，MSS应尽可能大，且满足在IP层传输时不需要再分片；**窗口扩大**；**时间戳**

### 可靠运输的实现

![](https://raw.githubusercontent.com/xyxxxxx/image/master/fipsojhu36ioyngljongwr.PNG)

![](https://raw.githubusercontent.com/xyxxxxx/image/master/89j5griwonhteiugrjnwrv.PNG)

![](https://raw.githubusercontent.com/xyxxxxx/image/master/hfwruighn3yinjibgrw.PNG)

![](https://raw.githubusercontent.com/xyxxxxx/image/master/jfspiojtki4ofndjvbht.PNG)

![](https://raw.githubusercontent.com/xyxxxxx/image/master/njvisniu5yb62i5uhg9wro.PNG)

**超时重传时间的选择**

略

**选择确认SACK**

略



### 流量控制

流量控制(flow control)即让发送方的发送速率不要太快，以免接收方来不及接收

**滑动窗口实现流量控制**

![](https://raw.githubusercontent.com/xyxxxxx/image/master/kxpoijyio53tjfuihetf.PNG)

**传输效率**

略

### 拥塞机制

计算机网络中的链路容量（带宽）、交换节点中的缓存和处理机等都是网络的资源. 在某段时间若对网络中某一资源的需求超过了该资源所能提供的可用部分，网络的性能就会下降，即造成**拥塞(congestion)**

拥塞控制就是防止过多的数据注入到网络中，这样可以使网络中的路由器或链路不致过载；流量控制往往是点对点通信量的控制

![](https://raw.githubusercontent.com/xyxxxxx/image/master/90y5wjygiorwjy25ugnwr.PNG)

**拥塞控制方法**

**慢开始(slow-start)**，**拥塞避免(congestion avoidance)**，**快重传(fast retransmit)**，**快恢复(fast recovery)**

略

**主动队列管理AQM**

略



### 连接管理

**连接建立**

TCP建立连接的过程称为**握手**，握手需要在客户和服务器之间交换三个TCP报文段：

![](https://raw.githubusercontent.com/xyxxxxx/image/master/sjigfoh3mu6klunjky5.PNG)

1. B的TCP服务器进程先创建传输控制块TCB，然后进入LISTEN状态，准备接受客户进程的连接请求
2. A的TCP客户进程先创建传输控制块TCB，然后希望建立TCP连接时向B发出<u>连接请求报文段</u>，其SYN=1，seq=x（规定SYN=1的报文段不能携带数据），TCP客户进程进入SYN-SENT状态
3. B收到连接请求报文段，同意建立连接，发送<u>确认报文段</u>，其SYN=1，ACK=1，seq=y，ack=x+1，TCP服务器进程进入SYN-RCVD状态
4. A收到确认报文段后，再次向B发送<u>确认报文段</u>，其ACK=1，seq=x+1，ack=y+1，该报文段可以携带数据，如果不携带数据则不消耗序号，A进入ESTABLISHED状态
5. B收到A的确认后，也进入ESTABLISHED状态



需要A再次确认的原因：防止A第一次发出的连接请求报文延误较长时间到达B，B同意则再次建立连接，一直等待A发送数据

**连接释放**

![](https://raw.githubusercontent.com/xyxxxxx/image/master/fjioh3nokwnhtjh35.PNG)

数据传输结束，A和B仍处于ESTABLISHED状态

1. A的应用进程通知TCP客户进程，发送<u>连接释放报文段</u>，并停止发送数据，其FIN=1，seq=u，进入FIN-WAIT-1状态（FIN报文消耗一个序号）
2. B收到连接释放报文后发出<u>确认报文段</u>，其ACK=1，seq=v，ack=u+1，TCP服务器进程通知高层应用进程，此时A→B的连接释放，进入CLOSE-WAIT状态
3. A收到确认报文段后进入FIN-WAIT-2状态
4. 当已经没有数据需要发送，B的应用进程通知TCP服务器进程，发送<u>连接释放报文段</u>，其FIN=1，ACK=1，seq=w，ack=u+1，进入LAST-ACK状态
5. A收到连接释放报文段后发送<u>确认报文段</u>，其ACK=1，seq=u+1，ack=w+1，进入TIME-WAIT状态，经过2MSL后进入CLOSED状态
6. B收到确认报文段，进入CLOSED状态

等待2MSL时间的理由：若A最后发送的确认报文段丢失，B会超时重传FIN+ACK报文段，A收到后会重传确认报文段，并重新计时；等待本连接产生的所有报文段从网络中消失，避免影响下一次连接







**TCP有限状态机**

![](https://raw.githubusercontent.com/xyxxxxx/image/master/tk3i5ojy36uiogejkeht.PNG)



