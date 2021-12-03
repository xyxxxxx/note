# 信息就是位+上下文

`hello`程序的生命周期从一个**源文件**开始，文件名是`hello.c`。源文件是由程序员通过编辑器创建并保存的文本文件，实际上是由值0和1组成的**位**序列。8个位被组织成一组，称为**字节**，每个字节表示程序中的某些文本字符。

大部分的现代计算机系统都是用ASCII标准表示文本字符，这种方式用一个唯一的单字节大小的整数值来表示字符。`hello.c`文件以字节序列的方式存储，每个字节都有一个整数值，对应于一个字符。像`hello.c`这样只由ASCII字符构成的文件称为**文本文件**，文本文件以外的其它文件称为**二进制文件**。

![Ascii Table](http://www.asciitable.com/index/asciifull.gif)

系统中所有的信息——包括磁盘文件，内存中的程序和用户数据，网络数据报都是由一串位序列表示的。区分不同数据对象的唯一方法是读取这些数据对象的**上下文**。





# 程序的编译

为了在系统上运行高级C语言程序，每条C语句都必须被其他程序转化为一系列的低级**机器语言**指令，然后这些指令按照一种称为**可执行目标文件**的格式打好包，并以二进制文件的形式存放起来。

在Unix系统上，从源文件到目标文件的转化是由**编译器驱动程序**完成的：

```
$ gcc -o hello hello.c
```

这里GCC编译器驱动程序读取源文件`hello.c`，并把它翻译成一个可执行目标文件`hello`。这个翻译过程可分为四个阶段完成，如图所示，执行这四个阶段的程序（**预处理器**，**编译器**，**汇编器**，**链接器**）共同构成了**编译系统（compilation system）**。



![](https://raw.githubusercontent.com/xyxxxxx/image/master/jgmefiloj356ykl357y35.PNG)

## 预处理阶段

预处理器（cpp）扩展源代码，插入所有用`#include`命令指定的文件，并扩展所有用`#define`声明指定的宏。



## 编译阶段

编译器（ccl）将文本文件`hello.i`翻译成文本文件`hello.s`，它包含一个**汇编程序语言**。汇编语言为不同的高级语言的不同编译器提供了通用的输出语言。



## 汇编阶段

汇编器（as）将`hello.s`翻译成机器语言指令，把这些指令打包成可重定位（未填入全局值的地址）目标程序的格式。得到的`hello.o`是一个二进制文件。



## 链接阶段

`hello`程序调用了`printf`函数，它是C**标准库**中的一个函数。`printf`函数存在于一个名为`printf.o`的单独的预编译好了的目标文件中，这个文件必须合并到`hello.o`程序中。链接器（ld）负责处理这种合并。得到的`hello`文件是一个可执行目标文件（简称**可执行文件**），可以被加载到内存中由系统执行。





# 处理器读并解释存储在内存中的指令

## shell

shell是一个命令行解释器，用于执行一个命令。如果该命令的第一个单词不是一个内置的shell命令，那么shell就会假设这是一个可执行文件的名字，于是加载并运行这个文件。



## 系统的硬件组成

![](https://raw.githubusercontent.com/xyxxxxx/image/master/sdkop24j6kiho3hj4y.PNG)



### 总线

**总线**是贯穿整个系统的一组电子管道，它负责在各个部件之间传递信息字节。通常总线被设计成传送定长的字节块，即**字（word）**。字中的字节数（即**字长**）是一个基本的系统参数，各个系统中都不尽相同。现在的大多数机器的字长或者是4个字节（32位），或者是8个字节（64位）。



### I/O设备

I/O设备是系统与外部世界的联系通道。这里的示例系统包括四个I/O设备：作为用户输入的键盘和鼠标，作为用户输出的显示器，用于长期存储数据和程序的磁盘驱动器。

每个I/O设备都通过一个**控制器**或**适配器**与I/O总线相连。控制器和适配器之间的区别在于它们的封装方式：控制器是I/O设备本身或者系统的**主板**上的芯片组，而适配器是一块插在主板插槽上的卡。



### 主存

**主存**是一个临时存储设备，在CPU执行程序时，用来存放程序和程序处理的数据。从物理上来说，主存是由一组**动态随机存取存储器（DRAM）**芯片组成的。从逻辑上来说，存储器是一个线性的字节数组，每个字节都有其唯一的地址，这些地址是从零开始的。



### CPU

**中央处理单元**（CPU），简称处理器，是执行存储在主存中指令的引擎。CPU的核心是一个大小为一个字的存储设备（即**寄存器**），称为**程序计数器（PC）**。在任何时刻，PC都指向主存中的某条机器语言指令。

CPU一直不断地执行程序计数器指向的指令，再更新程序计数器，使其指向下一条指令。CPU看上去是按照一个非常简单的指令执行模型来操作的，这个模型是由**指令集架构**决定的。在这个模型中，指令按照严格的顺序执行，执行一条指令包含执行一系列的步骤。CPU从程序处理器指向的内存储读取指令，解释指令中的位，执行该指令指示的简单操作，然后更新PC，使其指向下一条指令。

CPU的简单操作围绕着主存、**寄存器文件**（register file）和**算术逻辑单元**（ALU）进行。寄存器文件是一个小的存储设备，由一些单个字长的寄存器组成，每个寄存器都有唯一的名字。ALU计算新的数据和地址值。以下是一些简单操作的例子：

+ **加载**：从主存复制一个字节或者一个字到寄存器，覆盖寄存器原来的内容
+ **存储**：从寄存器复制一个字节或者一个字到主存的某个位置，覆盖该位置原来的内容
+ **操作**：把两个寄存器的内容复制到ALU，ALU对这两个字做算术运算，并将结果存放到一个寄存器中，覆盖该位置原来的内容
+ **跳转**：从指令本身中抽取一个字，将这个字复制到PC中，覆盖PC中原来的值

需要将处理器的指令集架构和处理器的**微体系结构**区分开来：<u>指令集架构描述的是每条机器代码指令的效果</u>，而<u>微体系结构描述的是处理器的实际实现</u>。





# 运行`hello`程序

初始时shell程序等待我们输入一个命令，当我们在键盘上输入字符串`./hello`后，shell程序将字符逐一读入寄存器，再把它存放到内存中。

![](https://raw.githubusercontent.com/xyxxxxx/image/master/skvgop3yjkio3557jykiop35gy.PNG)

当我们再键盘上敲回车时，shell程序知道我们已经结束了命令的输入，然后shell执行一系列指令来加载可执行的`hello`文件，这些指令将`hello`目标文件中的代码和数据从磁盘复制到主存。利用**直接存储器存取（DMA）**技术，数据可以不通过处理器而直接从磁盘到达主存。

![](https://raw.githubusercontent.com/xyxxxxx/image/master/asdfj8io24thjiog25h3n.PNG)

一旦目标文件`hello`中的代码和数据被加载到主存，处理器就开始执行`hello`程序和`main`程序中的机器语言指令。这些指令将`“hello, world\n”`字符串中的字节从主存复制到寄存器文件，再从寄存器文件中复制到显示设备，最终显示在屏幕上。

![](https://raw.githubusercontent.com/xyxxxxx/image/master/xfjio24tjiogwrnjgrefgh.PNG)





# 高速缓存至关重要

这个简单的示例揭示了一个重要的问题，即系统花费了大量时间把信息从一个地方移动到另一个地方。这些复制就是开销，减慢了程序真正的工作。

根据机械原理，较大的存储设备要比较小的存储设备运行得慢，而快速设备的造价远高于同类的低俗设备。针对这种CPU与主存之间的差异，系统设计者采用了更小更快的存储设备，称为**高速缓存存储器（cache memory，简称cache或高速缓存）**，作为暂时的集结区域，存放处理器近期可能需要的信息。位于处理器芯片上的**L1高速缓存**的容量可以达到数万字节，访问速度几乎和访问寄存器文件一样快。一个容量为数十万到数百万字节的更大的**L2高速缓存**通过一条特殊的总线连接到处理器。进程访问L2高速缓存的时间要比访问L1高速缓存的时间长5倍，但是仍然比访问主存的时间快5\~10倍。L1和L2高速缓存是一种叫做**静态随机访问存储器（SRAM）**的硬件技术实现的。比较新的、处理能力更强大的系统甚至有三级高速缓存。

高速缓存利用了**局部性原理**，即程序具有访问局部区域里的数据和代码的趋势。通过让高速缓存里存放可能经常访问的数据，大部分的内存操作都能在快速的高速缓存中完成。

![](https://raw.githubusercontent.com/xyxxxxx/image/master/dfsj8ui356u8oy356j3hy.PNG)

应用程序员能够利用高速缓存将程序的性能提高一个数量级。





# 存储设备形成层次结构

在处理器和一个较大较慢的设备（主存）之间插入一个更小更快的存储设备（高速缓存）的想法已经成为一个普遍的观念。实际上每个计算机系统的存储设备都被组织成了一个**存储器层次结构**。在这个层次结构中，从上到小，设备的访问速度越来越慢，容量越来越大，每字节的造价越来越便宜。

![](https://raw.githubusercontent.com/xyxxxxx/image/master/fgjio2345tjiowgrjiorwg.PNG)

存储器层次结构的主要思想是上一层的存储器作为低一层的存储器的高速缓存。





# 操作系统管理硬件

当shell加载和运行`hello`程序，以及`hello`程序输出自己的消息时，shell和`hello`程序都没有直接访问键盘、显示器、磁盘或主存，取而代之的是，它们依靠**操作系统**提供的服务。我们可以把操作系统看成是应用程序和硬件之间插入的一层软件，所有应用程序对硬件的操作尝试都必须通过操作系统。

操作系统有两个基本功能：（1）防止硬件被失控的应用程序滥用；（2）向应用程序提供简单一致的机制来控制复杂而又通常大不相同的低级硬件设备。操作系统通过几个基本的抽象概念（**进程**，**虚拟内存**和**文件**）来实现这两个功能。文件是对I/O设备的抽象表示，虚拟内存是对主存和磁盘I/O设备的抽象表示，进程则是对CPU、主存和I/O设备的抽象表示。

![](https://raw.githubusercontent.com/xyxxxxx/image/master/xcvjioj24tiojngsfviokgfh.PNG)

### 进程

像`hello`这样的程序在现代操作系统上运行时，操作系统会提供一种假象，就好像系统上只有这一个程序在运行。程序看上去是独占地使用CPU、内存和I/O设备。这些假象是通过进程的概念来实现的。

**进程**是操作系统对一个正在运行的程序的一种抽象。在一个系统上可以同事运行多个进程，而每个进程都好像在独占地使用硬件。**并发**运行，则是指一个进程的指令和另一个进程的指令是交错执行的。在大多数系统中，需要运行的进程数是多于可以运行它们的CPU个数。传统系统在一个时刻只能执行一个程序，而现今的**多核**处理器同时能够执行多个程序。无论在单核还是多核系统中，一个CPU看上去都像是在并发地执行多个进程，这是通过处理器在进程间切换来实现的。操作系统实现这种交错执行的机制称为上下文切换。

操作系统保持跟踪进程运行所需的所有状态信息，也就是**上下文**。上下文包括PC和寄存器文件的当前值，以及主存的内容。在任何一个时刻，单处理器系统都只能执行一个进程的代码。当操作系统决定要把控制权从当前进程转移到某个新进程时，就会进行**上下文切换**，即保存当前进程的上下文，恢复新进程的上下文，然后将控制权传递到新进程。新进程会从它上次停止的地方开始。

如图所示，示例情形中有两个并发的进程：shell进程和`hello`进程。起初，shell进程在等待命令行的输入。当我们让它运行`hello`程序时，shell通过一个系统调用来执行我们的请求，系统调用会将控制权传递给操作系统。操作系统保存shell进程的上下文，创建一个新的`hello`进程及其上下文，然后将控制权传给新的`hello`进程。`hello`进程终止后，操作系统恢复shell进程的上下文，并将控制权传回给它，shell进程等待下一个命令输入。

![](https://raw.githubusercontent.com/xyxxxxx/image/master/sfg35ipojghwrilonhtjaf.PNG)

从一个进程到另一个进程的转换由操作系统**内核（kernel）**管理，内核是操作系统代码常驻主存的部分。当应用程序需要操作系统的某些操作时，比如读写文件，它就执行一条特殊的**系统调用（system call）**指令，将控制权传递给内核。然后内核执行被请求的操作并返回应用程序。内核不是一个独立的进程，而是管理全部进程的所用代码和数据结构的集合。



### 线程

在现代操作系统中，一个进程实际上由多个称为**线程**的执行单元组成，每个线程都运行在进程的上下文中，并且共享同样的代码和全局数据。由于网络服务器中对并行处理的需求，线程成为越来越重要的编程模型，因为多线程比多进程之间更容易共享数据，而且线程一般而言都比进程更高效。



### 虚拟内存

**虚拟内存**为每个进程提供了一个假象，即每个进程都在独占地使用主存。每个进程看到的内存都是一致的，称为**虚拟地址空间**。如图所示的是Linux进程的虚拟地址空间。在Linux中，地址空间的顶部区域是保留给操作系统中的代码和数据的，底部区域存放用户进程定义的代码和数据。图中的地址从下往上增大。

![](https://raw.githubusercontent.com/xyxxxxx/image/master/cvjiok34ophjtoiwg.PNG)

每个进程看到的虚拟地址空间由大量准确的区构成，每个区有专门的功能。这里简单介绍每个区：

+ <u>程序代码和数据</u>：对所有进程来说，代码从同一固定地址开始，紧接的是和C全局变量相对应的数据位置。代码和数据区是直接按照可执行目标文件的内容初始化的。
+ <u>堆</u>：当调用`malloc`和`free`这样的C标准库函数时，**运行时堆**可以动态地扩展和收缩。
+ <u>共享库</u>：大约在地址空间的中间部分是一块用来存放像C标准库和数学库这样的共享库的代码和数据的区域。
+ <u>栈</u>：位于用户虚拟地址空间顶部的是**用户栈**，编译器用它来实现函数调用。和堆一样，用户栈在程序执行期间可以动态地扩展和收缩。每当我们调用一个函数时，栈就会增长；从一个函数返回时，栈就会收缩。
+ <u>内核虚拟内存</u>：地址空间顶部的区域为内核保留。不允许应用程序读写这个区域的内容或者直接调用内核代码定义的函数。相反，它们必须调用内核来执行这些操作。



### 文件

**文件**就是字节序列，仅此而已。每个I/O设备，包括磁盘、键盘、显示器，甚至网络，都可以看成是文件。系统中的所有输入输出都是通过使用一小组称为Unix I/O的系统函数调用读写文件来实现的。

文件这个简单而精致的概念是强大的，因为它向应用程序提供了一个统一的视角，来看待系统中可能含有的各种各样的I/O设备。





# 系统之间利用网络通信

从一个单独的系统来看，网络可视为一个I/O设备，如图所示。当系统从主存复制一串字节到网络适配器时，数据流经过网络到达另一台机器；相似地系统可以读取从其他机器发送来的数据，并把数据复制到自己的主存。

![](https://raw.githubusercontent.com/xyxxxxx/image/master/sdvcjkio3erjio345yjoy35.PNG)

随着Internet这样的全球网络的出现，从一台主机复制信息到另一台主机已经成为计算机系统最重要的用途之一。





# 重要主题

## 并发和并行

**并发（concurrency）**是一个通用的概念，指一个同时具有多个活动的系统；**并行（parallelism）**指用并发使一个系统运行得更快。并行可以在计算机系统的多个抽象层次上运用。

### 线程级并发

使用线程，我们可以在一个进程中执行多个控制流。传统意义上的并发只是<u>模拟</u>出来的，是通过一台计算机在它执行的进程间快速切换来实现的。这种配置称为**单处理器系统**。

**多核处理器**将多个CPU集成到一个集成电路芯片上。如图所示的是一个典型多核处理器的组织结构，其中微处理器芯片有4个CPU核，每个核都有自己的L1和L2高速缓存。这些核共享更高层次的高速缓存，以及到主存的接口。

![](https://raw.githubusercontent.com/xyxxxxx/image/master/xcvksiopjk356iopthyej.PNG)

**超线程（hyperthreading）**，或者称为**同时多线程（simultaneous multi-threading）**，是一项允许一个CPU执行多个控制流的技术。它涉及CPU某些硬件有多个备份，比如程序计数器和寄存器文件，而其它的硬件部分只有一份，比如执行浮点算术运算的单元。常规的处理器需要大约20 000个时钟周期做不同线程间的转换，而超线程的处理器可以在单个周期的基础上决定执行哪一个线程。这使得CPU能够更好地利用它的处理资源，例如假设一个线程必须等到某些数据被装载到高速缓存中，那CPU就可以去执行另一个线程。

多处理器的使用可以从两方面提高系统性能：首先，它减少了在执行多个任务时模拟并发的需要；其次，它可以使应用程序运行得更快，当然<u>前提是程序是以多线程方式编写的，这些线程可以并行地高效执行</u>。



### 指令级并行

在较低的抽象层次上，现代处理器可以同时执行多条指令的属性称为**指令级并行**。最近的处理器可以保持每个时钟周期2~4条指令的执行速率，但实际上每条指令从开始到结束需要长得多的时间，大约20个以上的周期，只是CPU使用了非常多的技巧来同时处理多达100条指令。在**流水线（pipeline）**中，执行一条指令所需要的活动被划分成不同的步骤，将CPU的硬件组织成一系列的阶段，每个阶段执行一个步骤。这些阶段可以并行地操作，用来处理不同指令的不同部分。

如果CPU可以达到比一个周期一条指令更快的执行速率，就称为**超标量（superscalar）**处理器。大多数现代处理器都支持超标量操作。



### 单指令、多数据并行

在最低层次上，许多现代CPU拥有特殊的硬件，允许一条指令产生多个可以并行执行的操作，这种方式称为**单指令、多数据**，即SIMD并行。例如较新几代的Intel和AMD处理器都具有并行地对8对单精度浮点数做加法的指令。



## 计算机系统的抽象

**抽象**的使用是计算机科学中最为重要的概念之一。例如为一组函数规定一个简单的应用程序接口（API）就是一个很好的编程习惯，程序员无需了解它内部的工作便可以使用这些代码。不同的编程语言提供不同形式和等级的抽象支持。

在处理器里，<u>**指令集架构**提供了对实际处理器硬件的抽象</u>。使用这个抽象，机器代码程序表现得就好像运行在一个一次只执行一条指令的处理器上。只要执行模型一样，不同的处理器实现也能执行同样的机器代码，而又提供不同的开销和性能。

除了文件、虚拟内存和进程，我们再增加一个新的抽象：**虚拟机**，它提供对整个计算机的抽象。

![](https://raw.githubusercontent.com/xyxxxxx/image/master/dsfxkop4tjiogndfuihet.PNG)