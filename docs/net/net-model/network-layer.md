网络层向上只提供简单灵活的、无连接的、尽最大努力交付的数据报服务：网络在发送分组时不需要先建立连接，每一个分组独立发送；网络层不提供服务质量的承诺，即传送的分组可能出错、丢失、重复或失序，也不保证分组交付的时限

![](https://raw.githubusercontent.com/xyxxxxx/image/master/giornh3yjhognvwrj.PNG)

# 网际协议IP

与IP协议配套使用的还有三个协议：

+ 地址解析协议ARP
+ 网际控制报文协议ICMP
+ 网际组管理协议IGMP

![](https://raw.githubusercontent.com/xyxxxxx/image/master/spekhtohnuhyjiwrf.PNG)

将网络连接起来需要使用一些中间设备，根据所在层次可分为以下四种：

1. 物理层使用的中间设备称为**转发器(repeater)**
2. 链路层……**桥接器(bridge)**
3. 网络层……**路由器(router)**
4. 网络层以上……**网关(gateway)**

**IP网** 使用IP协议的虚拟互联网络

![](https://raw.githubusercontent.com/xyxxxxx/image/master/etoihjgoihn4jtnhwrfjkg.PNG)

互联网由多种异构网络互连组成：

![](https://raw.githubusercontent.com/xyxxxxx/image/master/oeipthjephtomkebtoh.PNG)

# IP地址

整个互联网是一个单一的、抽象的网络。IP地址就是给互联网上每一台主机（或路由器）的每一个接口分配一个在全世界范围内唯一的32位标识符. 

## 分类的IP地址

该方法将IP地址划分为若干类，每一类由网络号(net-id)和主机号(host-id)组成，网络号在整个互联网范围内必须唯一，而主机号在网络号指明的网络范围内唯一，因此IP地址在整个互联网范围内唯一。

![](https://raw.githubusercontent.com/xyxxxxx/image/master/rwhbtkonh6jkbteg3.PNG)

近年来广泛使用无分类IP地址进行路由选择，上述分类已成为历史

![](https://raw.githubusercontent.com/xyxxxxx/image/master/sdtohnetjknsvfdfeq.PNG)

### A类地址

A类地址的网络号有7位可用，但除去2个： ①网络号全0，表示“本网络”；②网络号01111111(127)保留为本地软件环回测试本主机的进程之间的通信之用

主机号有3字节可用，但除去2个：①主机号全0，表示本主机连接到的网络地址（例如一主机的IP地址为5.6.7.8，则所在网络地址为5.0.0.0）②主机号全1，表示该网络上所有主机

### B类地址

B类地址的网络号有14位可用，网络地址128.0.0.0不指派

### C类地址

网络地址192.0.0.0不指派

![](https://raw.githubusercontent.com/xyxxxxx/image/master/erykhm64knkhjbetk.PNG)

## IP地址特点

+ IP地址是一种分等级的地址结构，这样：(1) IP地址管理机构分配IP地址时只分配网络号；(2) 路由器仅需根据目的主机所连接的网络号来分组
+ 当一台主机同时连接两个网络时，该主机必须同时具有两个网络号不同的IP地址，称为多归属主机，例如路由器
+ 一个网络是具有相同网络号的主机的集合. 具有不同网络号的局域网必须使用路由器互连
+ 互联网平等对待每一个IP地址

![](https://raw.githubusercontent.com/xyxxxxx/image/master/dfgjione5yjkfnwvcjwrg.PNG)

+ 同一个局域网上的主机或路由器的IP地址中的网络号必须一致
+ 网桥（链路层）互连的网段仍然是一个局域网
+ 路由器总是有两个或以上的IP地址
+ 两个路由器直接相连时，可以分配或不分配IP地址（如图中N1,N2,N3），未分配地址的称为无编号网络或无名网络

## IP地址和硬件地址

![](https://raw.githubusercontent.com/xyxxxxx/image/master/gekohj4ngvwrjkb35y.PNG)

![](https://raw.githubusercontent.com/xyxxxxx/image/master/eth24tmfokghjnkrwv.PNG)

IP层抽象的互联网上只能看到IP数据报，而屏蔽了下层的细节。由此在网络层上讨论问题，就能够使用统一的、抽象的IP地址研究主机或路由器之间的通信

## 地址解析协议ARP

ARP可以从网络层使用的IP地址中，解析出在链路层中使用的硬件地址，其工作原理是在主机ARP高速缓存中存放一个从IP地址到硬件地址的映射表，里面包含了本局域网上各主机和路由器，若缓存中没有查到，则：

![](https://raw.githubusercontent.com/xyxxxxx/image/master/sflgmhktelgrwklgnr.PNG)

再将主机B的映射写入高速缓存；

### ARP工作典型情况

![](https://raw.githubusercontent.com/xyxxxxx/image/master/sfoigmt3k5nhjk3.PNG)

## IP数据报

![](https://raw.githubusercontent.com/xyxxxxx/image/master/jteiksflngjkh532grf.PNG)

(1) **版本** 4位，IP协议的版本，目前广泛使用的版本号为4

(2) **首部长度** 4位，单位为32位/4字节. 首部长度最小为20字节，最高为60字节，但必须为4字节的整数倍，不足部分用填充字段填充

(3) **区分服务** 8位，一般不使用

(4) **总长度** 16位，数据报最大长度为65535字节. 但如果封装之后超过了链路层协议的**最大传送单元**，则必须将数据报分片处理

(5) **标识(identification)** 16位，超长的数据报拆分的各数据报片被赋予相同的标识字段的值

(6) **标志(flag)** 3位，最低位MF(more fragment)，=1时表示后面还有分片，=0时表示这是最后一个；中间位DF(don’t fragment)，=1时不能分片，=0时允许分片

(7) **片位移** 13位，表示该片在原数据报中的相对位置，单位为8字节

![](https://raw.githubusercontent.com/xyxxxxx/image/master/wkropmgketnhjt3h.PNG)

![](https://raw.githubusercontent.com/xyxxxxx/image/master/sfgktehmk5lnygj3vr.PNG)

(8) **生存时间** 8位，其目的是防止无法交付的数据报无限在互联网中兜圈子；TTL设置为跳数限制，路由器每次转发数据报之前把TTL值减1，当TTL减小到0则丢弃该数据报。若将TTL的值设置为1，就表示该数据报只能在本局域网中传送.

(9) **协议** 8位，指出该数据报携带的数据使用何种协议

(10) **首部检验和** 16位

![](https://raw.githubusercontent.com/xyxxxxx/image/master/etbkhlrhjnk35njky53g.PNG)

(11) **源地址** 32位

(12) **目的地址** 32位

## IP层转发分组的流程

![](https://raw.githubusercontent.com/xyxxxxx/image/master/adjokhn6jkbte.PNG)

![](https://raw.githubusercontent.com/xyxxxxx/image/master/fokhetnjkgrvwghgwy.PNG)

![](https://raw.githubusercontent.com/xyxxxxx/image/master/grwtncjkhtwhj7ik5.PNG)

# 子网和超网

## 划分子网

上述分类的IP地址的设计不够合理，造成IP地址空间利用率低、路由表过大、不够灵活的问题。在IP地址中增加“子网号字段”，使两级IP地址变为三级IP地址能够解决上述问题，称为**划分子网(subnetting)**：

+ 单位可将其所属的物理网络划分为若干个子网(subnet)，同时该网络对外仍表现为一个网络
+ 划分方法是从主机号中借用若干位作为子网号(subnet-id)，于是两级IP地址在单位内部变为三级IP地址：网络号、子网号和主机号

![](https://raw.githubusercontent.com/xyxxxxx/image/master/sdfmknjrynhj4o5g.PNG)

### 子网掩码

![](https://raw.githubusercontent.com/xyxxxxx/image/master/dfon64ujkhgrwjht.PNG)

所有网络必须使用子网掩码。如果一个网络不划分子网，那么就使用默认子网掩码：

![](https://raw.githubusercontent.com/xyxxxxx/image/master/worjphj36onhujk5gw.PNG)

路由器和相邻路由器交换路由信息时，必须告知自己所在网络/子网的子网掩码

B类网络子网划分示例：

![](https://raw.githubusercontent.com/xyxxxxx/image/master/norfnjgi36uoyig.PNG)

其中子网数等于$2^{子网位数}-2$，去掉的2个分别为全0和全1；但现在全0和全1也可以使用

## 分组转发

![](https://raw.githubusercontent.com/xyxxxxx/image/master/sfoklngjket5hnyj3k.PNG)

![](https://raw.githubusercontent.com/xyxxxxx/image/master/wrojghoiu36nyjgrw.PNG)

## 无分类编址CIDR

CIDR消除了传统的A类、B类和C类地址以及划分子网的概念，使IP地址回到无分类的两级编址；CIDR把网络前缀相同的连续IP地址组成一个**CIDR地址块**，例如：

![](https://raw.githubusercontent.com/xyxxxxx/image/master/wrjkgopmwnjhgn5.PNG)

CIDR使用32位的**地址掩码(address mask)**，其由一串1和一串0组成；地址掩码与子网掩码意义不同但使用方式相同，故也称为子网掩码；CIDR地址中斜线后面的数字就是地址掩码中1的个数.

路由表中利用CIDR地址块来查找目的网络，这种地址的聚合称为**路由聚合(route aggregation)**.

![](https://raw.githubusercontent.com/xyxxxxx/image/master/jwrgoinhbekjtn evbjk.PNG)

![](https://raw.githubusercontent.com/xyxxxxx/image/master/dgeitjhy3i5ohnb3jk.PNG)

### 最长前缀匹配

![](https://raw.githubusercontent.com/xyxxxxx/image/master/getionht3jk3hj5y.PNG)

### 二叉线索查找路由表

![](https://raw.githubusercontent.com/xyxxxxx/image/master/sfiogj364oiuhn3j5t.PNG)

# ICMP

![](https://raw.githubusercontent.com/xyxxxxx/image/master/sfgjiom46ukolngjrw.PNG)

## 种类

ICMP报文分为**ICMP差错报告报文**和**ICMP询问报文**:

![](https://raw.githubusercontent.com/xyxxxxx/image/master/dlhbmk5l34ymnlkh3t.PNG)

![](https://raw.githubusercontent.com/xyxxxxx/image/master/fsgjioethnjio5wrgv.PNG)

![](https://raw.githubusercontent.com/xyxxxxx/image/master/rjieojnu36ojlre.PNG)

## 应用

**PING(Packet InterNet Groper)**用来测试两台主机之间的连通性，其使用了ICMP回送请求和回答报文，而没有通过运输层的TCP或UDP.

**tracert**用来追踪一个分组从源点到终点的路径.

# 互联网的路由选择协议

理想的路由算法是正确的，公平的，计算简单，有自适应性，稳定性.

互联网采用分层次的路由选择协议，划分为许多较小的**自治系统（autonomous system）**. AS对其他AS表现出**单一的和一致的路由选择策略**.

互联网把路由选择协议分为两大类，即：

+ **内部网关协议IGP**(Interior Gateway Protocol)	AS内部使用的路由选择协议
+ **外部网关协议EGP**(External Gateway Protocol)     当数据报传到一个AS的边界时，就需要使用一种协议将路由选择信息传递到另一AS中

AS之间的路由选择称为**域间路由选择**，自治系统内部的路由选择称为**域内路由选择**，例如：

![](https://raw.githubusercontent.com/xyxxxxx/image/master/sfjviogtnhuyj35ntjg.PNG)

## IGP--RIP

RIP是一种分布式的基于距离向量的路由选择协议，其要求网络中每一个路由器都要维护从它到其他每一个目的网络的距离记录，**距离**也称为**跳数**，直接连接的距离为1，每经过一个路由器则距离+1. RIP允许一条路径最多包含15个路由器，因此只适用于小型互联网.

RIP协议中两个网络之间只能使用一条路由，即距离最短路由.

RIP协议的特点：

+ 仅和相邻路由器交换信息
+ 交换的信息的内容为当前本路由器所知道的全部信息
+ 按固定时间间隔交换路由信息

一般情况下，RIP协议可以收敛，最后AS中所有节点都得到了正确的路由选择信息

路由表中最重要的信息是到某个网络的距离以及应经过的下一条地址，更新的原则就是找出到每个网络的最短距离

### 距离向量算法

见例：

![](https://raw.githubusercontent.com/xyxxxxx/image/master/sdgijo3tn5hyjo35uy.PNG)

![](https://raw.githubusercontent.com/xyxxxxx/image/master/gon4oj6unj3ky3i.PNG)

![](https://raw.githubusercontent.com/xyxxxxx/image/master/sfhgnoj46u46hio4.PNG)

### RIP协议报文格式

![](https://raw.githubusercontent.com/xyxxxxx/image/master/3igtokhn3jkfwnjg.PNG)

RIP协议的优点是实现简单，开销较小，缺点是限制网络规模，路由表随网络规模扩大而增加，“坏消息传播得慢”：

![](https://raw.githubusercontent.com/xyxxxxx/image/master/vfsjion3y5ojngwjr.PNG)

## IGP--OSPF

OSPF(Open Shortest Path First)的名字显示了其采用了Dijkstra提出的最短路径算法，其采用的是分布式的**链路状态协议(link state protocol)**：

+ 向本AS中的所有路由器发送信息，使用洪泛法
+ 发送的信息是与本路由器相邻的所有路由器的链路状态，即本路由器和哪些路由器相邻，以及该链路的**度量**或**代价**. 度量可以用来表示费用、距离、时延、带宽等，因此较为灵活
+ 只有当链路状态发生变化时，路由器才向所有路由器用洪泛法发送此信息

所有的路由器最终都能建立一个**链路状态数据库**，即全网的拓扑结构图，该图在全网范围内一致. 

OSPF可以将AS划分为若干个更小的范围，即区域：

![](https://raw.githubusercontent.com/xyxxxxx/image/master/fsvnko3n5ylk43n6jhlo.PNG)

+ 洪泛法交换链路状态的范围局限于区域，路由器只知道本区域的完整网络拓扑
+ 主干区域0.0.0.0连通其他在下层的区域
+ R3, 4, 7是区域边界路由器，R3, 4, 5, 6, 7是主干路由器

OSPF的数据报直接使用IP数据报传送，结构如下：

![](https://raw.githubusercontent.com/xyxxxxx/image/master/sfgop46kuiohngwbjroh3.PNG)

OSPF协议的特点：

+ 允许管理员为每条路有指派不同的代价，对于不同类型的业务可计算出不同的路由
+ 如果到同一个目的网络有多条代价相同的路径，则将通信量分配给这几条路径，称为**负载平衡(load balancing)**
+ 所有分组都具有鉴别的功能
+ 支持可变长度的子网划分和CIDR
+ 由于网络的链路状态经常变化，OSPF让每一个链路状态都带上一个32位的序号，序号越大则越新

OSPF的分组类型：

![](https://raw.githubusercontent.com/xyxxxxx/image/master/sfgjio46juioynhjieh.PNG)

![](https://raw.githubusercontent.com/xyxxxxx/image/master/fsoin357unh3jk5tho.PNG)

OSPF规定每两个相邻路由器每隔10s要交换一次问候分组，以确知哪些邻站可达. 其他4种分组都是用来进行链路状态数据库的同步

![](https://raw.githubusercontent.com/xyxxxxx/image/master/svopk6muklhnebjtl.PNG)

洪泛法更新全网链路状态：

![](https://raw.githubusercontent.com/xyxxxxx/image/master/bhnejktjynjgrwojiy25.PNG)

## BGP

BGP采用**路径向量(path vector)路由选择协议**。配置BGP时，每个AS的管理员都需要选择至少一个路由器作为该AS的**BGP发言人**，一个BGP发言人要和其他AS的BGP发言人（即**邻站**）交换路由信息，要先建立TCP连接（端口号179），然后在此连接上交换BGP报文以建立BGP会话，利用其交换路由信息。

![](https://raw.githubusercontent.com/xyxxxxx/image/master/fsjighon35yjogwnujgy.PNG)

## 路由器

路由器是一种具有多个输入端口和多个输出端口的专用计算机，其任务是转发分组。从路由器某个输入端口收到的分组，按照分组要去的目的网络，把该分组从路由器某个合适的输出端口转发给下一跳路由器，结构如图：

![](https://raw.githubusercontent.com/xyxxxxx/image/master/sfipgoj64ionyjgwory2.PNG)

路由器结构分为**路由选择部分**和**分组转发部分**

# IPv6

IPv6的主要变化：

+ 更大的地址空间
+ 扩展的地址层次结构
+ 灵活的首部格式
+ 改进的选项
+ 允许协议继续扩充
+ 自动配置
+ 支持资源的预分配
+ 首部为8字节对齐

IPv6的数据报的结构如下：

![](https://raw.githubusercontent.com/xyxxxxx/image/master/snfjlkbnjk5ynjk5wgh.PNG)

## IPv6的基本首部

![](https://raw.githubusercontent.com/xyxxxxx/image/master/feijogjy35nojgnjkr2.PNG)

(1) **版本(version)** 4位，IPv6该字段是6

(2) **通信量类(traffic class)** 8位，区分不同数据报的类别或优先级

(3) **流标号(flow label)** 20位，支持流的资源预分配。**流(flow)**指网络上从特定源点到特定终点的一系列数据报（如语音通话或在线直播），而在流所流经的路径上的路由器都保证指明的服务质量. 对于非实时数据，流标号没有用处，直接置0

(4) **有效载荷长度(payload length)** 16位，最大值为65535字节(64KB)

(5) **下一个首部(next header)** 8位，若出现扩展首部，则标识下一个类型；否则标识数据应交付的IP层上的哪一个高层协议（TCP或UDP）

(6) **跳数限制(hop limit)** 8位，防止数据报的无限兜圈子

(7) **源地址**

(8) **目的地址**

**扩展首部**

略

## IPv6地址

3种基本类型：

+ **单播unicast** 
+ **多播multicast**
+ **任播anycast** 任播的终点是一组计算机，但数据报只交付给其中一个

使用**冒号十六进制记法**：

![](https://raw.githubusercontent.com/xyxxxxx/image/master/sdfgioij46uohngj.PNG)

![](https://raw.githubusercontent.com/xyxxxxx/image/master/vfjioj635oynhejthy.PNG)

![](https://raw.githubusercontent.com/xyxxxxx/image/master/dsjiogj46onhtjet42.PNG)

**本地链路单播地址(Link-Local Unicast Address)** 有些单位的内网使用TCP/IP协议，但没有连接到互联网. 内网上的主机可以使用这种本地地址进行通信

## IPv4至IPv6的过渡策略

**双协议栈(dual stack)**指一部分主机或路由器装有IPv4和IPv6双协议栈. 双协议栈主机在和IPv6主机通信时采用IPv6地址，而和IPv4主机通信时使用IPv4地址. 域名系统DNS若返回IPv4地址，则源主机使用IPv4地址. 

![](https://raw.githubusercontent.com/xyxxxxx/image/master/sjfiony35johnwjkrfwht.PNG)

**隧道技术(tunneling)** IPv6数据报进入IPv4网络时，封装位IPv4数据报

![](https://raw.githubusercontent.com/xyxxxxx/image/master/gdpoitju46ioyngjrwht.PNG)

## ICMPv6

略

# IP多播

> 网络直播，IPTV

![](https://raw.githubusercontent.com/xyxxxxx/image/master/sfjbiohj35yiongj2rw.PNG)

IP多播分为两种：本地局域网上进行硬件多播，在互联网范围多播

## 网际组管理协议IGMP

IGMP协议让连接在本地局域网上的多播路由器知道本局域网上是否有主机参加或推出了某个多播组

![](https://raw.githubusercontent.com/xyxxxxx/image/master/wgfpiy35nmkhb.PNG)

# 虚拟专用网VPN和网络地址转换NAT

**专用地址(private address)**只能用于机构的内部通信，而不能和互联网上的主机通信。互联网中的所有路由器，对目的地址是专用地址的数据报一律不进行转发。专用地址包括：

![](https://raw.githubusercontent.com/xyxxxxx/image/master/dfophj3on5hjkebtth.PNG)

采用专用IP地址的互连网络称为**专用互联网**或**本地互联网**

**虚拟专用网VPN(virtual private network)**用于机构内部的通信：

![](https://raw.githubusercontent.com/xyxxxxx/image/master/wrpojh3ionbjwk3h.PNG)

**网络地址转换NAT(Network Address Translation)**

![](https://raw.githubusercontent.com/xyxxxxx/image/master/fnsjgkonj5tyhniotejhyo.PNG)

通过NAT路由器的通信必须由专用网内的主机发起，因此专用网内部的主机不能作为服务器

![](https://raw.githubusercontent.com/xyxxxxx/image/master/etiyj35ioygrmjoihte6u.PNG)

# 多协议标记交换MPLS

