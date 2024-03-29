

计算机执行**机器代码**，用字节序列编码低级的操作，包括处理数据、管理内存、读写存储设备上的数据，以及网络通信。<u>编译器基于编程语言的规则、目标机器的指令集和操作系统遵循的惯例，经过一系列的阶段生成机器代码</u>。GCC C语言编译器以**汇编代码**的形式产生输出，<u>汇编代码是机器代码的文本表示</u>，给出程序中的每一条指令。然后GCC调用汇编器和链接器，根据汇编代码生成可执行的机器代码。

当我们用高级语言编程的时候，机器屏蔽了程序的细节，即机器级的实现。与此相反，当用汇编代码编程时，程序员必须指定程序用来执行计算的低级指令。高级语言提供的抽象级别比较高，大多数时候，在这种抽象级别上工作效率会更高，也更可靠。通常情况下，使用现代的优化编译器产生的代码至少与一个熟练的汇编语言程序员手工编写代码一样有效。最大的优点是，用高级语言编写的程序可以在很多不同的机器上编译和执行，而汇编代码则是与特定机器密切相关。

尽管如此，阅读和理解汇编代码仍然是一项很重要的技能。以适当的命令行选项调用编译器，编译器就会产生一个以汇编代码形式表示的输出文件。<u>通过阅读这些汇编代码，我们能够理解编译器的优化能力，并分析代码中隐含的低效率</u>。试图最大化一段关键代码性能的程序员，通常会尝试源代码的各种形式，每次编译并检查产生的汇编代码，从而了解程序将要运行的效率如何。也有些时候高级语言提供的抽象层会隐藏我们想要了解的程序的运行时行为。

x86-64是现在笔记本电脑和台式机中最常见处理器的机器语言，也是驱动大型数据中心和超级计算机的最常见处理器的机器语言。这种语言历史悠久，起源于Intel公司1978年的第一个16位处理器，然后扩展到32位，64位。

# 历史观点

Intel处理器系列俗称x86，经历了一个长期的、不断进化的发展过程。

Intel处理器模型介绍略。

近年来许多公司生产出了与Intel处理器兼容的处理器，能够运行完全相同的机器级程序，其中领头的是AMD。

# 程序编码

假设一个C程序有两个文件`p1.c`和`p2.c`，我们用Unix命令行编译这些代码：

```
$ gcc -Og -o p p1.c p2.c
```

命令`gcc`指的就是GCC C编译器，这是Linux上默认的编译器。编译选项`-Og`告诉编译器使用会生成符合原始C代码整体结构的机器代码的优化等级。使用更高级别优化（如`-O1`或`-O2`）产生的代码会严重变形，以至于产生的机器代码和初始源代码之间的关系非常难以理解，但从得到的程序的性能考虑，更高级别的优化是更好的选择。

## 机器级代码

对于机器级编程来说，两种抽象十分重要。第一种是由**指令集体系结构**或**指令集架构（ISA，Instruction Set Architecture）**来定义机器级程序的格式和行为，它包括了处理器状态、指令的格式，以及每条指令对状态的影响。大多数ISA，包括x86-64，将程序的行为描述成好像每条指令都是按照顺序执行的。处理器的硬件远比描述的精细复杂，它们并发地执行许多指令，但是可以采取措施保证整体行为与ISA指定的顺序执行的行为完全一致。第二种是机器级程序使用的内存地址是虚拟地址。

汇编代码表示非常接近于机器代码，与机器代码的<u>二进制格式</u>相比，汇编代码的主要特点是它用可读性更好的<u>文本格式</u>表示。<u>能够理解汇编代码以及它与原始C代码的联系，是理解计算机如何执行程序的关键一步</u>。

x86-64机器代码中，一些通常对C语言程序员隐藏的处理器状态都是可见的：

+ <u>程序计数器</u>给出将要执行的下一条指令在内存中的地址
+ <u>整数寄存器</u>文件包含16个命名的位置，分别存储64位的值。这些存储器可以存储地址或整数数据。有的寄存器被用来记录某些重要的程序状态，而其它的寄存器用来保存临时数据，例如过程的参数和局部变量，函数的返回值。
+ <u>条件码寄存器</u>保存最近执行的算术或逻辑指令的状态信息。它们用来实现控制或数据流中的条件变化，例如用来实现`if`和`while`语句。
+ 一组<u>向量寄存器</u>可以存放一个或多个整数或浮点数值。

<u>程序内存用虚拟地址来寻址</u>，在任意给定的时刻，只有有限的一部分虚拟地址被认为是合法的。例如x86-64的虚拟地址由64位的字来表示，在目前的实现中，这些地址的高16位必须设置为0，所以一个地址实际上能够指定的是 $2^{48}$ 或64TB范围内的一个字节。

一条机器指令只执行一个非常基本的操作，例如将存放在寄存器中的两个数字相加，在存储器和寄存器之间传递数据，或是条件分支转移到新的指令地址。编译器必须产生这些指令的序列，从而实现程序结构。

## 代码示例

假设我们编写了一个C语言代码文件`mstore.c`，包含如下的函数定义：

```c
long mult2(long, long);

void multstore(long x, long y, long *dest){
    long t = mult2(x, y);
    *dest = t;
}
```

使用`gcc -S mstore.c`产生汇编文件`mstore.s`，其中包括下面几行：

```
multstore:
	pushq   %rbx
	movq    %rdx, %rbx
	call    mult2
	movq    %rax, (%rbx)
	popq    %rbx
	ret
```

以上代码中每个缩进的行都对应于一条机器指令。

如果使用`gcc -c mstore.c`产生目标代码文件`mstore.o`。1368字节的该文件中有一段14字节的序列，其十六进制表示为：

```
53 48 89 d3 e8 00 00 00 00 48 89 03 5b c3
```

这就是上面列出的汇编指令对应的机器代码（目标代码）。从中我们知道，机器执行的程序只是一个字节序列，它是对一系列指令的编码。机器对产生这些指令的源代码几乎一无所知。

**反汇编器（disassembler）**程序可以根据机器代码产生一种类似汇编代码的格式，在Linux系统中使用`objdump -d mstore.o`反汇编得到如下结果：

```
0000000000000000 <multstore>:

0:   53                       push   %rbx
1:   48 89 d3                 mov    %rdx,%rbx
4:   e8 00 00 00 00           callq  9 <multstore+0x9>
9:   48 89 03                 mov    %rax,(%rbx)
c:   5b                       pop    %rbx
d:   c3                       retq
```

我们看到前面14字节的序列分成了若干组，每组都是一条指令，右边是等价的汇编语言。

其中一些关于机器代码和它的反汇编表示的特性值得注意：

+ x86-64的指令长度从1到15个字节不等。<u>常用的指令以及操作数较少的指令所需的字节数少</u>，而那些不太常用或操作数较多的指令所需的字节数多。
+ 设计指令格式的方式是，<u>从某个给定位置开始，可以将字节唯一解码成机器指令</u>，例如只有指令`push %rbx`是以字节值53开头的。
+ 反汇编器基于机器代码文件中的字节序列来确定汇编代码，它不需要访问程序的源代码或汇编代码，并且反汇编生成的汇编代码与GCC生成的汇编代码有些细微的差别

生成实际可执行的代码需要对一组目标代码文件运行链接器，而这一组目标代码文件中必须含有一个`main`函数。假设文件`main.c`如下：

```c
#include <stdio.h>

void multstore(long, long, long *)
    
int main() {
    long d;
    multstore(2,3,&d);
    printf("2 * 3 --> %ld\n", d);
    return 0;
}

long mult2(long a, long b) {
    long s = a * b
    return s
}
```

使用`GCC -o prog main.c mstore.c`生成可执行文件`prog`，该文件大小为8655个字节，因为它不仅包含了两个过程的代码，还包含了用来启动和终止程序的代码，以及用来和操作系统交互的代码。

如果使用`objdump -d prog`反汇编`prog`文件，反汇编器会抽取各种代码序列，包括下面这段：

```
0000000000400540 <multstore>:

400540:   53                       push   %rbx
400541:   48 89 d3                 mov    %rdx,%rbx
400544:   e8 42 00 00 00           callq  40058b <mult2>
400549:   48 89 03                 mov    %rax,(%rbx)
40054c:   5b                       pop    %rbx
40054d:   c3                       retq
40054e:   90                       nop
40054f:   90                       nop
```

这段代码与`mstore.o`反汇编产生的代码几乎一样，不同之处包括：左边列出的地址不同——链接器将这段代码的地址移到了一段不同的地址范围中；链接器填上了`callq`指令调用函数`mult2`需要使用的地址（链接器的任务之一就是为函数调用找到匹配的函数的可执行代码的位置）；多了两行代码——这两条指令的作用是使函数代码变为16字节，使得就存储器性能而言，能更好地放置下一个代码块。

## 关于格式的注解

汇编文件`mstore.s`的完整内容如下：

```
    .file   "010-mstore.c"
    .text
    .globl  multstore
    .type   multstore, @function

multstore:
	pushq   %rbx
	movq    %rdx, %rbx
	call    mult2
	movq    %rax, (%rbx)
	popq    %rbx
	ret
	.size   multstore, .-multstore
	.ident  "GCC: (Ubuntu 4.8.1-2ubuntu1~12.04) 4.8.1"
	.section        .note.GNU-stack,"",@progbits
```

所有以`.`开头的行都是指导汇编器和链接器工作的伪指令，我们通常忽略这些行。

为了更清楚地说明汇编代码，我们使用如图的格式表示汇编代码，它省略了大部分伪指令，但包括行号和解释性说明。

![](https://raw.githubusercontent.com/xyxxxxx/image/master/fsgjkiopkjm35kolmklgf.PNG)

# 数据格式

由于起初的体系结构为16位，Intel用“字（word）”表示16位数据类型，因此32位数据类型为“双字（double words）”，64位为“四字（quad words）”。如图所示C语言基本数据类型对应的x86-64表示。

![](https://raw.githubusercontent.com/xyxxxxx/image/master/sdnjkvnuio35jyhiok35nm.PNG)

大多数GCC生成的汇编代码指令都有一个字符的后缀，表明操作数的大小。例如数据传送指令有4个变种：`movb`，`movw`，`movl`，`movq`。后缀`l`既表示4字节整数也表示8字节双精度浮点数，但这不会产生歧义，因为浮点数使用一组完全不同的指令和寄存器。

# 访问数据

一个x86-64的CPU包含一组16个存储64位值的通用目的寄存器，这些寄存器用来存储整数数据和指针。如图显示了这16个寄存器，它们的名字都以`%r`开头，不过后面还跟着一些不同的命名规则的名字，这是由于指令集历史演化造成的。

![](https://raw.githubusercontent.com/xyxxxxx/image/master/djgfouij5io2jgiowrngjorh.PNG)

如图中嵌套的方框标明的，指令可以对这16个寄存器的低位字节中存放的不同大小的数据进行操作。字节级操作可以访问最低的字节，16位操作可以访问最低的2个字节，32位操作可以访问最低的4个字节，而64位操作可以访问整个寄存器。生成1字节和2字节数字的指令会保持剩下的字节不变；生成4字节数字的指令会把高位4个字节置0。

如图右边的解释说明的，在常见的程序里不同的寄存器扮演不同的角色。其中最特别的是栈指针`%rsp`，用来指明运行时栈的结束位置。另外15个寄存器的用法更灵活，只有少量指令会使用某些特定的寄存器。

## 操作数指示符

大多数指令有一个或多个**操作数（operand）**，指示出执行一个操作要使用的源数据值，以及放置结果的目的位置。源数据值可以以常数形式给出，或是从寄存器或内存中读出；结果可以存放在寄存器或内存中。因此各种不同的操作数的可能性分为三种类型：

1. **立即数（immediate）** 表示常数值。ATT汇编语言中，立即数的书写方式是`$`后面跟一个C语言表示的整数。不同的指令允许的立即数的范围不同，汇编器会自动选择最紧凑的方式进行数值编码。
2. **寄存器（register）** 表示某个寄存器，`R[]`表示寄存器的值。
3. **内存** 根据计算出来的地址（称为有效地址）访问某个内存位置。`M[]`表示内存的值。

如图所示，有多种不同的寻址模式，允许不同形式的内存引用。最常用的是 $Imm(r_b,r_i,s)$，这样的引用有四个组成部分：一个立即数偏移 $Imm$，一个基址寄存器 $r_b$，一个变址寄存器 $r_i$ 和一个比例因子s，这里s必须是1，2，4或8。基址寄存器和变址寄存器都必须是64位寄存器。有效地址计算为 $Imm+R[r_b]+R[r]\cdot s$。

![](https://raw.githubusercontent.com/xyxxxxx/image/master/sdcvjnuiojyh35ionmgjwfrkb.PNG)

## 数据传送指令

最频繁使用的指令是将数据从一个位置复制到另一个位置的指令。

我们把许多不同的指令划分成指令类，每一类中的指令执行相同的操作，只是操作数大小不同。

如图所示最简单形式的数据传送指令——MOV类，这些指令把数据从源位置复制到目的位置，不做任何变化。

![](https://raw.githubusercontent.com/xyxxxxx/image/master/vdhnuiojenthio35jhqrgw.PNG)

源操作数指定的值是一个立即数，存储在寄存器中或者内存中；目的操作数指定一个位置，是一个寄存器或者内存地址。x86-64加了一条限制，传送指令的两个操作数不能都指向内存位置，将一个值从一个内存位置复制到另一个内存位置需要两条指令和一个寄存器。

下图记录了两类数据移动指令，在将较小的源值复制到较大的目的时使用。所有这些指令都把数据从源（寄存器或内存）复制到目的寄存器。MOVZ类中的指令把目的中剩余的字节填充为0，而MOVS类中的指令通过符号扩展来填充，即复制源值的最高位。

![](https://raw.githubusercontent.com/xyxxxxx/image/master/sdfgjiok36mjklhetml.PNG)

## 数据传送示例

作为一个使用数据传送指令的代码示例，考虑如下的数据交换函数，既有C代码，又有GCC产生的汇编代码：

![](https://raw.githubusercontent.com/xyxxxxx/image/master/dgfjsionm25jklnmwgthr.PNG)

如图所示，函数`exchange`由三条指令实现：两个数据传送（`movq`），以及返回函数被调用点（`ret`）。当过程开始执行时，过程参数`xp`和`y`分别存储在寄存器`%rdi`和`%rsi`中，然后指令2从内存中读出`x`并存放到寄存器`%rax`中。稍后函数返回寄存器`%rax`的值，因而返回了`x`。指令3将`y`写入寄存器`$rdi`中`xp`指向的内存位置。

关于这段汇编代码有两点值得注意：首先C语言的指针其实就是地址，<u>使用指针就是将地址存放在一个寄存器中</u>，然后在内存引用中使用这个寄存器；其次像`x`这样的<u>局部变量通常是保存在寄存器中</u>而不是内存中，因为访问寄存器比访问内存要快得多。

## 压入和弹出栈数据

最后两个数据传送操作是将数据压入程序栈中，以及从程序栈中弹出数据。在x86-64中，程序栈存放在内存中某个区域，如图所示，栈向下增长，这样一来栈顶元素的地址是栈中所有元素地址中最低的。

![](https://raw.githubusercontent.com/xyxxxxx/image/master/sjcmvkioljnuio35yhjuiot3.PNG)

因为栈和程序代码以及其它形式的程序数据都是放在同一内存中，所以程序可以用标准的内存寻址方法访问栈内的任意位置，例如指令`movq 8(%rsp),%rdx`会将栈顶的第二个元素从栈中复制到寄存器`%rdx`。

> 将数据压栈通常是因为寄存器不足以防止局部变量。

# 算术和逻辑操作

如图所示x86-64的一些整数和逻辑操作。大多数操作都分成了指令类，这些指令类有各种带不同大小操作数的变种。这些操作被分为四组：加载有效地址、一元操作、二元操作和移位。

![](https://raw.githubusercontent.com/xyxxxxx/image/master/xfngvjinjir3hui245tty25.PNG)

## 加载有效地址

**加载有效地址（load effective address）**指令`leaq`实际上是`movq`指令的变形。该指令并没有真正去引用内存，而只是将内存地址复制到寄存器。编译器经常发现`leaq`的一些灵活用法，根本就与有效地址计算无关，例如如果寄存器`%rdi`的值为`x`，寄存器`%rsi`的值为`y`，那么指令`leaq 7(%rdi,%rsi,4),%rax`将设置寄存器`%rax`的值为`7+x+4y`。

## 一元操作和二元操作

一元操作只有一个操作数，既是源又是目的，操作数可以是寄存器或内存位置。`INC`类指令和`DEC`类指令对应C语言的`++`和`--`运算符。

二元操作的第二个操作数既是源又是目的，第一个操作数可以是立即数、寄存器或内存位置，第二个操作数可以是寄存器或内存位置。当第二个操作数为内存地址时，处理器必须从内存读出值，执行操作，再把结果写回内存。

## 移位操作

移位操作的第一个操作数是移位量，第二个操作数是要移位的数，可以进行算术和逻辑右移。移位量可以是立即数，或者放在单字节寄存器`%cl`中（只允许使用这个寄存器）。移位操作的目的数可以是一个寄存器或内存位置。原则上来说，1个字节的移位量使得移位量的编码范围可以达到255，x86-64中，移位操作对w位长的数据值进行操作，移位量是w除`%cl`的<u>余数</u>。因此，当寄存器`%cl`的十六进制值为`0xFF`时，指令`salb`会移7位，`salw`会移15位，`sall`会移31位，而`salq`会移63位。

左移指令`SAL`和`SHL`的效果是一样的。右移指令中，`SAR`执行算术右移，`SHR`执行逻辑右移。

## 讨论

以上算术和逻辑操作既可以用于无符号运算，也可以用于补码运算，只有右移操作要求区分有符号和无符号数。这一特性使得补码运算成为实现有符号整数运算的一种比较好的方法的原因之一。

## 特殊的算术操作

两个64位有符号或无符号整数相乘得到的乘积需要128位来表示。x86-64指令集对128位数的操作提供有限的支持。Intel把16字节的数称为**八字（oct word）**。如图所示支持产生两个64位数字的128位乘积以及整数除法的指令。

![](https://raw.githubusercontent.com/xyxxxxx/image/master/nvjkiowrhjio35jyio35.PNG)

指令`mulq`和`imulq`用于计算两个64位值的全128位乘积，这两条指令都要求一个参数必须在寄存器`%rax`中，而另一个作为指令的源操作数给出，最后乘积放在寄存器`%rdx`（高位）和`%rax`（低位）中。

下面这段C代码说明了如何从两个无符号64位数字`x`和`y`生成128位的乘积：

```c
#include <inttypes.h>

typedef unsigned __int128 uint128_t;

void store_uprod(uint128_t *dest, uint64_t x, uint64_t y){
    *dest = x * (uint128_t) y;
}
```

GCC生成的汇编代码如下：

![](https://raw.githubusercontent.com/xyxxxxx/image/master/bnuoifhni3ju5y35eth.PNG)

可以观察到，存储乘积需要两个`movq`指令。由于生成这段代码针对小端法机器，所以高位字节存储在大地址。

有符号除法指令`idivl`将寄存器`%rdx`（高64位）和`%rax`（低64位）中的128位数作为被除数，除数作为指令的操作数给出，最后的商存储在寄存器`%rax`中，余数存储在寄存器`%rdx`中。

下面这段C代码说明了如何计算两个有符号64位数`x`和`y`的商和余数：

```c
void remdiv(long x, long y, long *qp, long *rp) {
    long q = x/y;
    long r = x%y;
    *qp = q;
    *rp = r;
}
```

编译得到如下汇编代码：

![](https://raw.githubusercontent.com/xyxxxxx/image/master/vnujiwrnhjk35b6hikgr.PNG)

第4行将被除数符号扩展到`%rdx`。

# 控制

到目前为止，我们只考虑了直线代码的行为，即指令一条接一条地顺序执行。C语言中的某些结构，比如条件语句、循环语句和分支语句，要求有条件的执行，根据数据测试的结果决定操作执行的顺序。机器代码提供两种基本的底层机制来实现有条件的行为：测试数据值，然后根据测试结果改变控制流或数据流。

## 条件码

除了整数寄存器，CPU还维护着一组单个位的**条件码（condition code）**寄存器，它们描述了最近的算术或逻辑操作的属性，可以检测这些寄存器来执行条件分支指令。最常用的条件码有：

+ `CF`：进位标志：最近的操作使最高位产生了进位。可用来检查无符号操作的溢出
+ `ZF`：零标志：最近的操作得出的结果为0
+ `SF`：符号标志：最近的操作得到的结果为负数
+ `OF`：溢出标志：最近的操作导致一个补码溢出

例如用一条ADD指令完成`t=a+b`的功能，这里`a`，`b`，`t`都是整型的（无符号或有符号），然后可以根据下面的C表达式来设置条件码：

```
CF  (unsigned) t < (unsigned) a       //无符号整数，检查溢出
ZF  (t == 0)                          //零
SF  (t < 0)                           //负数
OF  (a<0 == b<0) && (t<0 != a<0)      //有符号整数，检查上溢和下溢
```

<u>算术操作中的所有指令</u>，除了`leaq`，<u>都会设置条件码</u>；对于<u>逻辑操作，进位标志和溢出标志会被置为0</u>；对于<u>移位操作，进位标志将置为最后一个被移出的位，而溢出标志置为0</u>；`INC`和`DEC`指令会设置溢出和零标志，但是不会改变进位标志。

此外，还有两类指令只设置条件码而不改变任何其它寄存器，如图所示：`CMP`指令根据两个操作数之差来设置条件码，其行为与`SUB`指令的行为类似，只是不更新目的寄存器；`TEST`指令行为与`AND`指令的行为类似。

![](https://raw.githubusercontent.com/xyxxxxx/image/master/xcvbnjrwkgn2jk45t2n45.PNG)

## 访问条件码

条件码常见的使用方法有三种：

1. 根据条件码的某种组合，将一个字节置为0或1；
2. 条件跳转到程序的某个其它的部分；
3. 有条件地传送数据；

将一个字节置为0或1的一整类指令称为SET指令，它们之间的区别就在于它们考虑的条件码的组合是什么，指令的后缀即是表示不同的条件。

![](https://raw.githubusercontent.com/xyxxxxx/image/master/nvcjkenio245jtoiqef.PNG)

一条SET指令的目的操作数是低位单字节寄存器元素之一，或者一个字节的内存位置，指令将这个字节置为0或1。为了得到一个32位或64位的结果，我们需要对高位清零。一个计算`a < b`的典型指令序列如下所示，这里`a`和`b`都是`long`类型：

![](https://raw.githubusercontent.com/xyxxxxx/image/master/cxnvjkdfnhio6jio3y53g.PNG)

注意`movzbl`会将寄存器`%rax`的高7个字节全部清零。

虽然所有的算术和逻辑操作都会设置条件码，但是各个SET命令的描述只是适用于：执行比较指令，根据计算`t=a-b`设置条件码，其中`a, b, t`都是补码形式的整数。其它情形需要比照这一情形。

例如`sete`，即”相等时设置（set when equal）“指令，当`a==b`时`t=0`，因此`ZF`置1就表示相等。`setl`，即”小于时设置（set when less）“指令，如果没有发生溢出（`OF=0`）， $a<b$ 时`a-b<0`，`SF`置1，而当 $a\ge b$ 时`SF`置0；如果发生溢出（`OF=1`），发生负溢出（ $a<b$ ）时`a-b>0`，`SF`置0，发生正溢出（ $a>b$ ）时`a-b<0`，`SF`置1；因此`SF ^ OF`就提供了 $a<b$ 是否为真的测试。

对于无符号数比较，当 $a<b$ 时，CMP指令会设置`CF`，因此无符号比较使用的是`CF`和`ZF`的组合。

## 跳转指令

正常执行的情况下，指令会按照它们出现的顺序一条一条地执行。**跳转（jump）**指令使执行切换到一个全新的位置。在汇编代码中，这些跳转的目的地通常用一个**标号（label）**指明。考虑下面的汇编代码序列：

![](https://raw.githubusercontent.com/xyxxxxx/image/master/zcvnjki2rbhityu24htyui.PNG)

指令`jmp .L1`会导致程序跳过`movq`指令，而从`popq`指令开始继续执行。在产生目标代码文件时，汇编器会确定所有带标号指令的地址，并将**跳转目标**编码为跳转指令的一部分。

下图列举了不同的跳转指令。`jmp`指令是无条件跳转，它可以是直接跳转，即跳转目标作为指令的一部分编码；也可以是间接跳转，即跳转目标从寄存器或内存位置中读出。汇编语言中，直接跳转给出一个标号作为跳转目标，例如上面代码中的标号`.L1`；间接跳转的写法是`*`后面跟一个操作数指示符，使用内存操作数格式中的一种。例如指令

```
jmp *%rax
```

从寄存器`%rax`读出跳转目标，而指令

```
jmp *(%rax)
```

以`%rax`中的值作为地址，从内存中读出跳转目标。

![](https://raw.githubusercontent.com/xyxxxxx/image/master/sdjgio25jiognsfrjkhtr.PNG)

表中所示的其它跳转指令都是**有条件的**——它们根据条件码的某种组合，或者跳转，或者继续执行序列的下一条指令。这些指令的名字和跳转条件和`SET`指令一致。条件跳转只能是直接跳转。

## 跳转指令的编码

理解跳转指令的目标如何编码对于研究链接非常重要，并且也能帮助理解反汇编器的输出。在汇编代码中，跳转目标用符号标号表示，汇编器和链接器会产生跳转目标的适当编码。跳转指令有几种不同的编码，最常用的是**PC相对的（PC-relative）**，也就是将目标指令的地址与跳转指令的下一条指令的地址（PC的值）之间的差作为编码，这样的地址偏移量可以编码为1, 2或4个字节；另一种编码方法是给出”绝对“地址，用4个字节直接指定目标，汇编器和链接器会选择适当的跳转目的编码。

下面是一个PC相对寻址的例子，它包含两个跳转：第2行的`jmp`指令向前跳转到更高的地址，而第7行的`jg`指令向后跳转到更低的地址：

```
    movq    %rdi, %rax
    jmp     .L2
  .L3:
    sarq    %rax
  .L2:
    testq   %rax, %rax
    jg      .L3
    rep; ret
```

将以上汇编代码汇编再反汇编得到：

```
    0:   48 89 f8			mov   %rdi,%rax
    3:	 eb 03              jmp   8 <loop+0x8>
    5:	 48 d1 f8           sar   %rax
    8:	 48 85 c0           test  %rax,%rax
    b:   7f f8              jg    5 <loop+0x5>
    d:	 f3 c3              repz  retq
```

反汇编器产生的注释中，第2行跳转指令的跳转目标指明为`0x8`，第5行的跳转目标为`0x5`。观察字节编码，第2行的目标为`03`，其加上下一条指令的地址`5`即为跳转目标地址`0x8`；第5行的目标为`f8`，即补码表示的十进制数-8，其加上下一条指令的地址`d`即为跳转目标地址`0x5`。

将以上代码链接再反汇编得到：

```
    4004d0:   48 89 f8			 mov   %rdi,%rax
    4004d3:   eb 03              jmp   4004d8 <loop+0x8>
    4004d5:	  48 d1 f8           sar   %rax
    4004d8:   48 85 c0           test  %rax,%rax
    4004db:   7f f8              jg    4004d5 <loop+0x5>
    4004dd:	  f3 c3              repz  retq
```

<u>跳转指令被重定位到不同的地址，但字节编码并没有改变</u>。PC相对的跳转目标编码可以使指令编码很简洁（只需要2个字节），并且不受具体内存地址的影响。

## 用条件控制实现条件分支

将C语言的条件语句翻译成机器代码，最常用的方式是结合有条件和无条件跳转。下面给出了计算两数之差的绝对值的函数的C代码和相应的汇编代码：

```c
long lt_cnt = 0;
long ge_cnt = 0;

long absdiff_se(long x, long y){
    long result;
    if (x < y) {
        lt_cnt++;      // 强制使用条件控制
        result = y-x;
    }
    else {
        ge_cnt++;
        result = x-y;
    }
    return result;
}
```

![](https://raw.githubusercontent.com/xyxxxxx/image/master/cvbnjk53bnhjky35hywrgf.PNG)

对于`if-else`语句，汇编实现通常会使用以下形式：

```c
	if (!test_expr)
		goto false;
	then-expr;
    goto done;
false:
	else-expr
done:        
```

## 用条件传送实现条件分支

实现条件分支的传统方法是使用*控制*的条件转移。当条件满足时，程序沿着一条执行路径执行，而当不满足时，就沿另一条路径。这种机制简单而通用，但在现代处理器上可能会非常低效。

一个替代的策略是使用*数据*的条件转移。这种方法计算一个条件的两种结果，然后根据条件是否满足从中选取一个结果。这种策略只有在一些受限制的情况中可行，但它只需要一条简单的*条件传送指令*实现，而条件传送指令更符合现代处理器的性能特性。

下面给出了可以用条件传送编译的C代码和相应的汇编代码：

```c
long absdiff(long x, long y){
    long result;
    if (x < y)
        result = y-x;
    else
        result = x-y;
    return result;
}
```

![](https://raw.githubusercontent.com/xyxxxxx/image/master/fsgjio05joi35rytnjh3tn.PNG)

可以看到，汇编代码中既计算了`y-x`，又计算了`x-y`，`cmpq`指令比较`x:y`，`cmovge`指令最终决定是否覆盖返回值。

> 为了理解为什么在性能上条件数据传送优于条件控制转移，我们必须了解一些现代处理器的知识。
>
> 处理器通过**流水线（pipeline）**机制获得高性能，但流水线要求待执行的指令序列能够事先确定。当机器遇到分支时，只有当分支条件求值完成后，才能决定分支往哪边走，因此处理器采用非常精密的分支预测逻辑来猜测每条跳转指令是否会执行。如果预测正确，则流水线中可以充满指令；如果预测错误，处理器就需要丢弃该跳转指令之后的所有指令已做的工作，然后从正确的指令位置开始重新填充流水线。<u>错误预测带来的惩罚十分严重，一般会浪费大约15~30个时钟周期</u>，导致程序性能严重下降。
>
> <u>条件数据传送则不依赖于测试数据，更容易保持流水线是满的</u>。

下图列举了x86-64上一些可用的条件传送指令，每条指令都有两个操作数：源寄存器或内存地址S，目的寄存器R。与SET和跳转指令一样，这些指令的结果取决于条件码的值。

源和目的的值可以是16位、32位、64位长。对于所有的操作数长度可以使用同一个指令，因为汇编器可以从目标寄存器的名字推断操作数长度。

![](https://raw.githubusercontent.com/xyxxxxx/image/master/cvnij24n56jkiy35gn.PNG)

比较条件控制转移和条件数据传送，考虑三元表达式`v = test-expr?then-expr:else-expr;`

条件控制转移的方法会编译这个表达式为

```c
	if (!test_expr)
		goto false;
	then-expr;
    goto done;
false:
	else-expr
done:  
```

条件数据传送的方法会编译这个表达式为

```c
v = then-expr;
ve = else-expr;
t = test-expr;
if (!t) v = ve;
```

> 尽管如此，条件数据传送也并不总是优于条件控制转移，这取决于浪费的计算与错误预测的惩罚之间的相对性能。但实际上，编译器并没有能力保证做出更好的决定。

## 循环

C语言提供了多种循环结构，汇编实现使用条件测试和跳转的组合。

### do-while循环

`do-while`语句的汇编实现通常为

```c
loop:
	body-expr;
	t = test-expr;
	if (t)
        goto loop;        
```

下面给出了使用`do-while`循环计算 $n!$ 的C代码和相应的汇编代码：

```c
long fact_do(long n){
    long result = 1;
    do {
        result *= n;
        n--;
    } while (n>1);
    return result;
}
```

![](https://raw.githubusercontent.com/xyxxxxx/image/master/xcvnjin4t2iubny25hi.PNG)

### while循环

`do-while`语句的第一种汇编实现为：

```c
	goto test;
loop:
	body-expr;
test:
	t = test-expr;
	if (t)
        goto loop;
        
```

下面给出了使用`while`循环计算 $n!$ 的C代码和相应的第一种汇编实现代码（使用`-Og`选项）：

```c
long fact_while(long n){
    long result = 1;
    while (n > 1) {
        result *= n;
        n--;
    }
    return result;
}
```

![](https://raw.githubusercontent.com/xyxxxxx/image/master/acjksioj1trio34jnfido.PNG)

第二种汇编实现称为guarded-do，使用条件分支和一个`do-while`循环：

```c
t = test-expr;
if (!t)
    goto done;
loop:
	body-expr;
	t = test-expr;
	if (t)
        goto loop;    
done:
```

下面给出了第一种汇编实现代码（使用`-O1`选项）：

![](https://raw.githubusercontent.com/xyxxxxx/image/master/vcjuio25htnuij2gn5r.PNG)

### for循环

根据C语言标准，`for`循环可以转换为`while`循环，因此汇编实现与`while`循环类似。

下面给出了使用`for`循环计算 $n!$ 的C代码和相应的第一种汇编实现代码（使用`-Og`选项）：

```c
long fact_for(long n){
    long i;
    long result = 1;
    for (i=2; i<=n; i++)
        result *= i;
    return result;
}
```

![](https://raw.githubusercontent.com/xyxxxxx/image/master/cvbnihj245btnuijghn.PNG)

## switch语句

`switch`语句根据一个整数索引值进行多重分支（multiway branching）。在处理具有多种可能结果的测试时，这种语句特别有用，它不仅提高了C代码的可读性，也通过**跳转表（jump table）**这种数据结构使得实现更加高效。<u>跳转表是一个数组，其中表项 $i$ 是一个代码段的地址，这个代码段存放当`switch`索引值等于 $i$ 时程序应该采取的动作</u>。与使用一组很长的`if-else`语句相比，使用跳转表的优点是执行`switch`语句的时间与分支的数量无关。GCC根据分支的数量和分支值的稀疏程度来翻译`switch`语句。，当分支数量较多并且值的范围跨度小时，就会使用跳转表。

下面是一个使用C语言`switch`语句的示例，这个示例中包含了`switch`语句中可能包含的各种情形：

```c
void switch_eg(long x, long n, long *dest){
    long val = x;
    switch(n){
    	case 100:
            val *= 13;
            break;
        case 102:
            val += 10;
            // fall through
        case 103:
            val += 11;
            break;
        case 104:
        case 106:
            val *= val;
            break;
        default:
            val = 0;
    }
    *dest = val;
}
```

编译该C语言产生的汇编代码，跳转表，以及描述其行为的C代码见下：

![](https://raw.githubusercontent.com/xyxxxxx/image/master/cvmjion245tyjkh3gtn.PNG)

![](https://raw.githubusercontent.com/xyxxxxx/image/master/dscvnmjoktngj5ki2g.PNG)

![](https://raw.githubusercontent.com/xyxxxxx/image/master/sfjgkioapj35ioyhju3.PNG)

汇编代码的第5行中，`jmp`指令的操作数有前缀*，表示这是一个间接跳转，由操作数指定内存位置。

# 过程

过程是软件的一种很重要的抽象，它提供了一种封装代码的方式，<u>用一组指定的参数和一个可选的返回值实现某种功能</u>。设计良好的软件用过程作为抽象机制，隐藏某个行为的具体实现，同时又提供清晰简洁的接口定义，说明要计算的是哪些值，过程对程序状态产生什么影响。不同编程语言中过程的形式多样：函数（function）、方法（method）、子例程（subroutine）、处理函数（handler）等，但是它们有一些共同的特性。

假设过程`P`调用过程`Q`，`Q`执行后返回到`P`，这些动作包括下面一些机制：

+ **传递控制** 在进入过程`Q`的时候，程序计数器必须被设置为`Q`的代码的起始地址；返回时要将程序计数器设置为`P`中调用`Q`后面那条指令的地址。
+ **传递数据** `P`必须能向`Q`提供一个或多个参数，`Q`必须能向`P`返回一个值。
+ **分配和释放内存** `Q`开始执行时，可能需要为局部变量分配空间，而在返回前需要释放这些空间。

x86-64的过程实现包括一组特殊的指令和一些对机器资源使用的约定规则。

## 运行时栈

C语言（和其它大部分语言）的过程调用机制的一个关键特性是使用了栈数据结构提供的后进先出的内存管理原则。在`P`调用`Q`的例子中，但`Q`在执行时，`P`以及调用链上所有其它过程都被暂时挂起。当`Q`运行时，它只需要为局部变量分配存储空间，或者设置到另一个过程的调用；当`Q`返回时，为它分配的局部存储空间将全部被释放。因此程序可以用栈来管理它的过程所需要的存储空间，栈和程序寄存器存放着传递控制和数据、分配内存所需要的信息。

x86-64的栈向低地址方向增长，而栈指针`&rsp`指向栈顶元素，可以使用`pushq`和`popq`指令将数据存入栈中或取出。减少栈指针可以分配空间，而增加栈指针会释放空间。

<u>当x86-64过程需要的存储空间超出寄存器能够存放的大小时，就需要在栈上分配空间，这个部分称为过程的**栈帧（stack frame）**</u>，如图所示。

![](https://raw.githubusercontent.com/xyxxxxx/image/master/xfjkiofnmjkogbhngwvrf.PNG)

　

## 转移控制

在x86-64机器中，将控制从函数`P`转移到函数`Q`由指令`call`实现，<u>该指令会把返回地址（即`call`指令后面那条指令的地址）压入栈中，并把PC设置为`Q`的起始地址</u>。相应地，<u>指令`ret`会从栈中弹出返回地址，并将PC设置为它</u>。

`call`指令有一个目标，即指明被调用过程的起始指令地址。同跳转一样，调用可以是直接的，也可以是间接的。

回想[前面代码](#代码示例)中`main`函数调用`multstore`函数时的`call`和`ret`指令的执行情况，下面给出了这两个函数的部分反汇编代码：

```
0000000000400540 <multstore>:
0x400540:   53                       push   %rbx
...
0x40054d:   c3                       retq
...

...
0x400563:   e8 d8 ff ff ff           callq  400540 <multstore>
0x400568:   48 8b 54 24 08           mov    0x8(%rsp),%rdx
```

可以看到，在`main`函数中，地址为`0x400563`的`call`指令调用函数`multstore`，状态如下图所示，其中指明了栈指针`%rsp`和程序计数器`%rip`的值。

![](https://raw.githubusercontent.com/xyxxxxx/image/master/vfbnufiojbgnjkotn14asd.PNG)

再来看一个更详细的说明过程间传递控制的例子，下图给出了函数`top`，`leaf`，`main`的反汇编代码，以及过程调用和返回的状态：

![](https://raw.githubusercontent.com/xyxxxxx/image/master/xcvzmjioktn242jkngfw.PNG)

![](https://raw.githubusercontent.com/xyxxxxx/image/master/cxvjnuoi24ntyuio45.PNG)

其中`main`调用`top(100)`，然后`top`调用`leaf(95)`。`leaf`向`top`返回97，`top`向`main`返回194。

## 数据传送

x86-64中，大部分过程间的数据传送通过寄存器实现，例如我们之前看到的，参数通过寄存器`%rdi`，`%rsi`等传递，返回值通过寄存器`%rax`传递。<u>x86-64中，可以通过寄存器每次最多传递6个整型参数</u>，并且寄存器使用的名字取决于要传递的数据类型的大小，如下图所示，会根据参数在参数列表中的顺序和参数大小为它们分配寄存器。

![](https://raw.githubusercontent.com/xyxxxxx/image/master/gvnjo2n45tjkgqgergr.PNG)

如果一个函数有大于6个整型参数，超出的部分就要通过栈来传递，如图3-25所示。通过栈传递参数时，所有的数据大小都必须是8位的整数倍。作为栈传递参数的示例，考虑以下C函数，该函数有8个参数，包括4种整数类型和4个指针：

```c
void proc(long  a1, long  *a1p,
          int   a2, int   *a2p,
          short a3, short *a3p,
          char  a4, char  *a4p){
    *a1p += a1;
    *a2p += a2;
    *a3p += a3;
    *a4p += a4;
}
```

![](https://raw.githubusercontent.com/xyxxxxx/image/master/juty8290gjkrfghrqef.PNG)

如图给出了`proc`函数生成的汇编代码，前面6个参数通过寄存器传递，后面2个通过栈传递，就如下图所示。可以看到针对不同长度的整数变量使用了`add`指令的不同版本。

![](https://raw.githubusercontent.com/xyxxxxx/image/master/cvmnjk245nhtuij142.PNG)

> 上图应为<u>函数proc的调用函数</u>的栈帧结构

## 栈上的局部存储

在许多情况下，局部数据必须存放在内存中：

+ 寄存器不足以存放所有的本地数据
+ 对一个局部变量使用地址运算符`&`，这会将局部变量存入内存并返回地址
+ <u>局部变量是数组或结构</u>

一般来说，过程通过减小栈指针在栈上分配空间，分配的结果作为栈帧的一部分，标号为局部变量，如图3-25所示。

来看一个处理地址运算符的例子：

```c
long swap_add(long *xp, long *yp){
    long x = *xp;
    long y = *yp;
    *xp = y;
    *yp = x;
    return x+y;
}

long caller(){
	long arg1 = 534;
    long arg2 = 1057;
    long sum = swap_add(&arg1, &arg2);
    long diff = arg1-arg2;
    return sum*diff;
}
```

![](https://raw.githubusercontent.com/xyxxxxx/image/master/t245jui9tjgriownjgwri.PNG)

`caller`的汇编代码在开始的时候将栈指针减掉16，也就是在栈上分配了16个字节，然后将`arg1,arg2`放置在栈指针偏移量0和8的位置，最后将栈指针加16以释放栈帧。通过这个例子可以看到，<u>运行时栈提供了一种简单的、在需要时分配、函数完成时释放局部存储的机制</u>。

下面是一个更复杂的例子：

```c
long call_proc(){
    long  x1=1; int  x2=2;
    short x3=3; char x4=4;
    proc(x1,&x1,x2,&x2,x3,&x3,x4,&x4);
    return (x1+x2)*(x3-x4);
}
```

![](https://raw.githubusercontent.com/xyxxxxx/image/master/jciovnj24ionutjort4.PNG)

`call_proc`的汇编代码中，第2~15行都在为调用`proc`做准备，其中包括<u>为局部变量和函数参数建立栈帧</u>，将函数参数加载至寄存器，栈帧如下图所示。调用过程`proc`时返回地址会被压栈，于是与图3-30一致。

![](https://raw.githubusercontent.com/xyxxxxx/image/master/kop134jtiog24rnh5uyi.PNG)

## 寄存器中的局部存储

<u>寄存器组是唯一被所有过程共享的资源，我们需要保证当一个过程调用另一个过程时，被调用者不会覆盖调用者之后需要使用的寄存器值</u>。为此x86-64采用了一组统一的寄存器使用惯例，所有的过程都必须遵循。

根据惯例，寄存器`%rbx,%rbp,%r12~%r15`被划分为<u>被调用者保存寄存器</u>。当过程`P`调用过程`Q`时，`Q`必须保存这些寄存器的值，保证当`Q`返回时这些寄存器的值不变。`Q`保存一个寄存器的值不变的方法，<u>或者是根本不去改变它，或者是将原始值压栈，在返回前从栈中弹出旧值</u>，<u>压入的寄存器值会在栈帧中创建标号为“保存的寄存器”的部分</u>，如图3-25所示，这样`P`就能将值安全地保存在被调用者保存寄存器中，而不用担心值被破坏。

其它的所有寄存器，除了栈指针`%rsp`，都分类为<u>调用者保存寄存器</u>。这意味着任何函数都能修改它们，而保存这些值就是`P`的责任。

来看下面的例子：

```c
long P(long x, long y){
    long u = Q(y);
    long v = Q(x);
    return u+v;
}
```

![](https://raw.githubusercontent.com/xyxxxxx/image/master/csmvio45jk90pje4uiot4.PNG)

由于函数`P`首先压栈`%rbp,%rbx`的值，因为需要使用寄存器`%rbp,%rbx`保存`x,Q(y)`的值，返回前再将其弹出。

## 递归过程

前面描述的寄存器和栈的惯例使得x86-64过程能够递归地调用自身，每个过程调用在栈中都有它自己的私有空间，因此多个未完成调用的局部变量不会互相影响。

下面给出了递归的阶乘函数C代码和生成的汇编代码：

```c
long rfact(long n){
    long result;
    if (n <= 1)
        result = 1;
    else
        result = n * rfact(n-1);
    return result;
}
```

![](https://raw.githubusercontent.com/xyxxxxx/image/master/5j36uiothj4uri25yhgr.PNG)

可以看到汇编代码使用寄存器`%rbx`来保存参数`n`，调用`rfact(n-1)`的结果保存在寄存器`%rax`中，相乘即得到期望的结果。

从这个例子我们看到，函数递归调用自身和调用其它函数是一样的。<u>栈规则提供了一种机制，每次函数调用都有它自己私有的状态信息（和局部变量）的存储空间</u>。栈分配和释放的规则也很自然地与函数调用-返回的顺序匹配。

# 数组分配和访问

C语言实现数组的方式非常简单，因此很容易翻译成机器代码。C语言的一个不同寻常的特点是可以产生指向数组中元素的指针，并对这些指针进行运算，在机器代码中这些指针会被翻译成地址计算。

## 基本原则

对于数据类型 $T$ 和整型常数 $N$，声明`T A[N];`有两个效果：首先它在内存中分配一个 $LN$ 字节的连续区域，这里 $L$ 是数据类型 $T$ 的大小；其次它引入标识符`A`，可以用`A`作为数组开头的指针，指针的值用 $x_A$ 表示。这样就可以用 $0～N-1$ 的整数索引来访问该数组元素。

作为示例，考虑以下声明：

```c
char    A[12];
char   *B[8];
int     C[6];
double *D[5];
```

这些声明会产生以下数组：

| 数组 | 元素大小 | 总大小 | 起始地址 | 元素 $i$  |
| ---- | -------- | ------ | -------- | ---------- |
| `A`  | 1        | 12     | $x_A$  | $x_A+i$  |
| `B`  | 8        | 64     | $x_B$  | $x_B+8i$ |
| `C`  | 4        | 24     | $x_C$  | $x_C+4i$ |
| `D`  | 8        | 40     | $x_D$  | $x_D+8i$ |

数组`A`由12个`char`元素（单字节）组成，`C`由6个`int`元素（4字节）组成，`B`和`D`都是指针数组，因此每个元素都是地址（8字节）。

x86-64的内存引用指令可以用来简化数组访问，例如`E`是`int`型数组，想要计算`E[i]`，其中`E`的地址存放在寄存器`%rdx`中，`i`存放在寄存器`%rcx`中，那么指令`movl (%rdx,%rcx,4),%eax`会计算地址 $x_E+4i$，读这个内存地址的值，并将结果存放到寄存器`%eax`中。

## 指针运算

C语言允许对指针进行运算，例如`p`是一个指向类型为 $T$ 的数据的指针，`p`的值为 $x_p$，那么表达式`p+i`的值为 $x_p+Li$，其中 $L$ 是数据类型 $T$ 的大小。

对于一个对象`Obj`，`&Obj`返回指向该对象地址的一个指针；对于一个地址`Adr`，`*Adr`返回该地址位置的值。

扩展之前的例子，假设整型数组`E`的起始地址和整数索引`i`分别存放在寄存器`%rdx`和`%rcx`中，下面给出了一些与`E`有关的表达式以及相应的汇编代码实现：

![](https://raw.githubusercontent.com/xyxxxxx/image/master/53ju89giwtrjnogthwn.PNG)

从这些例子中，可以看到返回数组值的操作类型为`int`，因此涉及4字节操作和寄存器`%eax`；返回指针的操作类型为`int*`，因此涉及8字节操作和寄存器`%rax`。最后一个例子表明可以计算同一个数组中的两个指针之差，结果的类型为`long`，<u>值为两个地址之差除以该数据类型的大小</u>。

## 二维数组

当我们创建数组的数组（即二维数组）时，数组分配和引用的一般原则仍成立。例如声明`int A[5][3];`，等价于声明

```c
typedef int row3_t[3];
row3_t A[5];
```

其中数据类型`row3_t`被定义为包含3个整数的数组，而数组`A`包含5个这样的数组，整个数组的大小就是 $4×5×3=60$ 字节。

数组`A`可以用`A[0][0]`到`A[4][2]`来引用，数组元素在内存中按照“行优先”的顺序排列，如图所示。

![](https://raw.githubusercontent.com/xyxxxxx/image/master/jiojykowgtrnjui2.PNG)

通常来说，对于声明`T D[R][C]`的数组，其数组元素`D[i][j]`的内存地址为 $x_D+L(C\cdot i+j)$，其中 $L$ 是数据类型 $T$ 的大小。作为一个示例，考虑前面的数组`A[5][3]`，假设 $x_A,i,j$ 分别在寄存器`%rdi,%rsi,%rdx`中，那么可以用以下代码将数组元素`A[i][j]`复制到寄存器`%eax`中：

![](https://raw.githubusercontent.com/xyxxxxx/image/master/42tjuigowrhjnuiwgbrnh.PNG)

可以看到，这段代码计算元素的地址为 $x_A+12i+4j$。

## 定长数组

C语言编译器能够优化定长多维数组上的操作代码。看下面的例子：

```c
#define N 16
typedef int fix_matrix[N][N];

int fix_prod_ele(fix_matrix A, fix_matrix B, long i, long k){
    long j;
    int result = 0;
    for (j=0; j<N; j++)
        result += A[i][j]*B[j][k];
    
    return result;
}
```

![](https://raw.githubusercontent.com/xyxxxxx/image/master/vjdsionj245iohymwfdbv.PNG)

上面的代码计算矩阵`A`的行`i`和`B`的列`k`的内积。将其编译再反编译，所有的数组引用都被替换为指针间接引用。

## 变长数组

历史上C语言只支持在编译时就能确定大小的多维数组，程序员需要变长数组时不得不用`malloc`或`calloc`这样的函数为数组分配存储空间，再显式地用行优先索引将多维数组映射到一维数组。ISO C99引入了新功能，允许数组的维度是表达式，在数组被分配的时候才计算出来。

例如要访问 $n\times n$ 数组的元素 $(i,j)$，可以写如下函数：

```CQL
int var_ele(long n, int A[n][n], long i, long j){
    return A[i][j];
}
```

GCC编译得到代码如下所示：

![](https://raw.githubusercontent.com/xyxxxxx/image/master/452u89fvjdfuiohb35tui.PNG)

如注释所示，这段代码计算元素 $(i,j)$ 的地址为 $x_A+4(n\cdot i)+4j$，这个计算类似于定长数组的地址计算，但不同点在于使用了乘法指令`imulq`计算 $n\cdot i$。动态的版本<u>必须用乘法指令</u>对`i`伸缩`n`倍，而不能用移位和加法指令，在一些处理器中乘法指令会招致严重的性能处罚，但是在这种情况中无法避免。

# 异质的数据结构

C语言提供了两种将不同对象组合到一起创建数据类型的机制：

+ **结构(structure)**，用关键字`struct`声明，将多个对象集合到一个对象中
+ **联合(union)**，用关键字`union`声明，允许用几种不同的类型来引用一个对象

## 结构

C语言的`struct`声明创建一个数据类型，<u>将不同类型的对象集合到一个对象中</u>，用名字来引用结构的各个部分。类似于数组的实现，<u>结构的所有组成部分都存放在内存中一段连续的区域内</u>，而<u>指向结构的指针就是结构的第一个字节的地址</u>。编译器维护关于每个结构类型的信息，指示每个字段(field)的字节偏移，并<u>以这些偏移作为内存引用指令中的位移</u>，从而产生对结构元素的引用。

考虑如下的结构声明：

```c
struct rec{
    int i;
    int j;
    int a[2];
    int *p;
};
```

这个结构包括4个字段：2个`int`，1个`int`数组和一个`int`指针，总共是24个字节：

![](https://raw.githubusercontent.com/xyxxxxx/image/master/ftiu2490jgieronbgwri.PNG)

可以观察到，数组`a`是嵌入到这个结构中的。

为了访问结构的字段，编译器产生的代码要将结构的地址加上适当的偏移。例如，假设`struct rec*`类型的变量`r`放在寄存器`%rdi`中，那么下面的代码将元素`r->i`复制到元素`r->j`：

![](https://raw.githubusercontent.com/xyxxxxx/image/master/cvnmjuio24nuignwtrgrw.PNG)

要产生一个指向结构内部对象的指针，只需要将结构的地址加上该字段的偏移量。例如下面的代码实现了语句`r->p=&r->a[r->i+r->j]`：

![](https://raw.githubusercontent.com/xyxxxxx/image/master/dfjvdjknjki245bng2.PNG)

## 联合

联合提供了一种方式，能够规避C语言的类型系统，允许以多种类型引用同一个对象。考虑下面的声明：

```c
struct S3{
    char c;
    int i[2];
    double v;
};
union U3{
    char c;
    int i[2];
    double v;
};
```

在一台x86-64Linux机器上编译时，字段的偏移量、数据类型`S3`和`U3`的完整大小如下：

| 类型 | `c`  | `i`  | `v`  | 大小 |
| ---- | ---- | ---- | ---- | ---- |
| `S3` | 0    | 4    | 16   | 24   |
| `U3` | 0    | 0    | 0    | 8    |

（下一节将解释为什么`i,v`的偏移量分别是4和16。）对于`union U3*`类型的指针`p`，`p->c, p->i[0], p->v`引用的都是联合的起始位置。一个联合的总的大小等于它的最大字段的大小。

联合在一些上下文中十分有用，但它也会引起一些错误，因为它们绕过了C语言类型系统提供的安全措施。一种应用情况是，我们事先知道对一个数据结构中的两个不同字段的使用是互斥的，那么将这两个字段声明为联合（而不是结构）的一部分，会减小分配空间的总量。

例如，假设我们想实现一个二叉树的数据结构，每个叶节点都有两个`double`类型的数据值，而每个内部节点都有两个指向子节点的指针，但是没有数据。如果声明：

```c
struct node_s{
    struct node_s *left;
    struct node_s *right;
    double data[2];
};
```

那么每个节点需要32个字节，每种类型的节点都要浪费一半的字节。相反，如果我们声明：

```c
union node_u{
    struct{
        union node_u *left;
        union node_u *right;
    }internal;
    double data[2];
};
```

那么每个节点只需要16个字节。如果`n`是一个指针，指向`union node_u`对象，那么可以用`n->data[0], n->data[1]`引用叶节点的数据，而用`n->internal.left, n->internal.right`来引用内部节点的子节点。

但是，如果这么编码，就没有办法确定一个给定的节点到底是叶节点还是内部节点。通常的方法是引入一个枚举类型，定义这个联合中可能的不同选择，然后再创建一个结构，包含一个标签字段和这个联合：

```c
typedef enum {N_LEAF, N_INTERNAL} nodetype_t;

struct node_t{
    nodetype_t type;
    union{
        struct{
            struct node_t *left;
            struct node_t *right;
        }internal;
        double data[2];
    }info;
};
```

这个结构需要24个字节：`type`4个字节，`info`16个字节，以及数据对齐需要4个字节。在这种情况下，使用联合带来的节省是很小的，但对于有较多字段的数据结构，这样的节省会更可观。

联合还可以用来访问不同数据类型的位模式。例如，假设我们使用简单的强制类型转换将一个`double`类型的值`d`转换为`unsigned long`类型的值`u`：

```c
unsigned long u = (unsigned long) d;
```

值`u`会是`d`的整数表示；`u`的位表示会与`d`的很不一样。再看下面这段代码：

```c
unsigned long double2ulong(double d){
    union{
        double d;
        unsigned long u;
    }temp;
    temp.d = d;
    return temp.u;
};
```

在这段代码中，我们<u>以一种数据类型来存储联合中的参数，而以另一种数据类型来访问它</u>，结果会是<u>`u`具有和`d`一样的位表示</u>。

当用联合将不同大小的数据类型结合到一起时，字节顺序的问题就变得重要。考虑下面这段代码：

```c
double uu2double(unsigned word0, unsigned word1){
    union{
        double d;
        unsigned u[2];
    }temp;
    temp.u[0]=word0;
    temp.u[1]=word1;
    return temp.d;
}
```

在x86-64这样的小端法机器上，`word0`是`d`的低位4个字节，而`word1`是高位4个字节。在大端法机器上则相反。

## 数据对齐

许多计算机系统对基本数据类型的合法地址做出了一些限制，要求某种类型对象的地址必须是某个值K（通常是2,4或8）的倍数，这种对齐限制简化了形成处理器和内存系统之间接口的硬件设计。例如，假设一个处理器总是从内存中取8个字节，则地址必须为8的倍数；如果我们能保证将所有的`double`或`long`类型的数据的地址对齐成8的倍数，那么就可以用一个内存操作来读写值。

无论数据是否对齐，x86-64硬件都能正常工作，不过Intel还是建议要对齐数据以提高内存系统的性能。对齐原则是<u>任何K字节的基本对象的地址必须是K的倍数</u>，参照这条原则会得到如下对齐：

| K    | 类型                  |
| ---- | --------------------- |
| 1    | `char`                |
| 2    | `short`               |
| 4    | `int, float`          |
| 8    | `long, double, char*` |

<u>确保每种类型的对象都满足它的对齐限制，就可以保证对齐</u>。编译器在汇编代码中放入命令，指明全局数据所需的对齐，例如[跳转表](#switch语句)的汇编代码声明在第2行包含命令`.align 8`，这就保证了每个跳转表项的起始地址是8的倍数（因为每个表项长8个字节）。

对于包含结构的代码，编译器可能需要在字段的分配中插入间隙，以保证每个结构元素都满足它的对齐要求，此外对于结构本身的起始地址也有一些对齐要求。例如，考虑下面的结构声明：

```c
struct S1{
    int  i;
    char c;
    int  j;      
};
```

假设编译器用最小的9字节分配：

![](https://raw.githubusercontent.com/xyxxxxx/image/master/dsgj890124jvnwqefgr.PNG)

它不满足字段`j`的4字节对齐要求。取而代之地，编译器在字段`c`和`j`之间插入一个3字节的间隙：

![](https://raw.githubusercontent.com/xyxxxxx/image/master/dscxjv90o25jyionwbvgfjk.PNG)

此外，编译器必须保证<u>任何`struct S1`对象的起始地址必须也是4的倍数</u>，这样每个字段都满足4字节对齐要求。

另外，编译器可能对结构末尾做一些填充，使结构数组的每个元素都满足它的对齐要求。例如，考虑下面的结构声明：

```c
struct S2{
    int  i;
    int  j;
    char c;
};
```

如果我们将这个结构打包成9个字节，只要保证结构的起始地址是4的倍数，我们仍然能够满足字段`i`和`j`的对齐要求。不过，考虑结构数组

```c
struct S2 d[4];
```

如果为结构分配9个字节，就不能满足`d`的每个元素的对齐要求。<u>实际上，编译器会为结构`S2`分配12个字节</u>，在最后插入一个3字节的间隙：

![](https://raw.githubusercontent.com/xyxxxxx/image/master/csvjio24jtionht3rgfqe.PNG)

这样所有的对齐限制就都满足了。

# 在机器级程序中结合控制与数据

## 理解指针

指针是C语言的核心特色之一，它们以一种统一的方式，对不同数据结构中的元素产生引用。对于编程新手来说，指针总是会带来很多困惑，但基本概念其实非常简单。在此，我们介绍一些指针和它们映射到机器代码的关键原则：

+ <u>每个指针都对应一个类型</u>，这个类型表明该指针指向的是哪一类对象。以下面的指针声明为例：

  ```c
  int *ip;
  char **cpp;
  ```

  变量`ip`是一个指向`int`类型对象的指针，而`cpp`指针指向一个指向`char`类型对象的指针。特殊的`void *`类型代表通用指针，比如`malloc`函数返回一个通用指针，然后通过显式或隐式强制类型转换，将它转换为一个有类型的指针。指针类型不是机器代码中的一部分，它们是C语言提供的一种抽象，帮助程序员避免寻址错误。

+ <u>每个指针都有一个值</u>，这个值是某个特定类型的对象的地址。特殊的`NULL`或`0x0`值表示该指针没有指向任何地方，即空指针。

+ <u>指针用`&`运算符创建</u>，这个运算符可以应用到任何`lvalue`类的C表达式上。`lvalue`（左值）指可以出现在赋值语句左边的表达式，包括变量、结构、联合和数组元素。

+ <u>`*`操作符用于间接引用指针</u>，其结果是一个值，类型与指针类型一致。间接引用是用内存引用来实现的，存取到一个指定的地址。

+ <u>数组与指针紧密联系</u>，一个数组的名字可以像一个指针变量一样引用；数组引用（例如`a[3]`）与指针运算和间接引用（例如`*(a+3)`）有一样的效果。数组引用和指针运算都需要用对象大小对偏移量进行伸缩。

+ <u>将指针强制转换类型不改变它的值</u>。强制转换类型的一个效果是改变指针运算的伸缩，例如，如果`p`是`char*`类型的指针，其值为p，那么表达式`(int*)p+7`计算为p+28，而`(int*)(p+7)`计算为p+7。

+ <u>指针也可以指向函数</u>，这提供了一个十分强大的存储和向代码传递引用的功能，这些引用可以被程序的某个其它部分调用。例如，定义如下函数：

  ```c
  int fun(int x, int *p);
  ```

  然后可以声明一个指针`fp`，将其赋值为这个函数：

  ```c
  int (*fp)(int, int*);
  fp = fun;
  ```

  然后用这个指针调用函数：

  ```c
  int y = 1;
  int result = fp(3, &y);
  ```

  函数指针的值是该函数机器代码的第一条指令的地址。

## 内存越界引用和缓冲区溢出

<u>C对于数组引用不进行任何边界检查，并且局部变量和状态信息（例如保存的寄存器值、参数信息、返回地址）都存放在栈中</u>。这两种情况结合到一起可能导致严重的程序错误，即<u>对越界的数组元素的写操作会破坏存储在栈中的状态信息</u>。当程序使用这个被破坏的状态，试图重新加载寄存器或执行`ret`指令时就会出现严重的错误。

一种特别常见的状态破坏称为**缓冲区溢出（buffer overflow）**。通常，在栈中分配某个字符数组来保存一个字符串，但是一旦字符串的长度超出了为数组分配的空间即发生缓冲区溢出。来看下面的程序示例：

```c
/* Implementation of library function gets() */
char *gets(char *s){
    int c;
    char *dest = s;
    while ((c = getchar()) != '\n' && c != EOF)
        *dest++ = c;
    if (c == EOF && dest == s)
        /* No characters read */
        return NULL;
    *dest++ = '\0';  /* Terminate string */
    return s;
}

/* Read input line and write it back */
void echo(){
    char buf[8];  /* Way too small! */
    gets(buf);
    puts(buf);
}

```

库函数`gets`从标准输入读入一行，在遇到换行或者EOF时停止。它将这个字符串复制到参数`s`指向的位置，并在字符串结尾加上NULL字符(`\0`)。然后在函数`echo`中，我们使用了`gets`和`puts`，这个函数只是简单地从标准输入中读入一行，再把它送回标准输出。

然而`gets`的问题在于它没有办法确定`buf`是否对于保存整个字符串有足够的空间。在`echo`示例中我们故意将缓冲区设置得非常小——只有8个字节，任何长度超过7个字符的字符串都会导致写越界。

检查GCC为`echo`产生的汇编代码，看看栈是如何组织的：

图

如下图所示，该程序把栈指针减去了24，即在栈上分配了24个字节。字符数组`buf`位于栈顶，`%rsp`被复制到`%rdi`作为调用`gets`和`puts`的参数，中间还有16字节未被使用。

图3-40

只要用户输入不超过7个字符，`gets`返回的字符串就能够放进为`buf`分配的空间里。不过，长一些的字符串就会导致`gets`覆盖栈上存储的某些信息；随着字符串变长，下列信息会被破坏：

表

字符串在23个字符之前都没有严重的后果，但一旦超过之后，返回地址的值以及更多可能的保存状态会被破坏。返回地址被破坏会导致`ret`  指令使程序跳转到一个完全意想不到的位置。如果只看C代码，根本就不可能看出会有上面这些行为；只有通过研究机器代码级别的程序才能理解像`gets`这样的函数进行的内存越界写的影响。

通常，使用`gets`或其它能导致缓冲区溢出的函数都是不好的编程习惯。但不幸的是，很多常用的库函数，包含`strcpy`、`strcat`和`sprintf`，都有一个属性，即不需要告诉它们目标缓冲区的大小，就写一个字节序列。

缓冲区溢出的一个更加致命的使用就是让程序执行它本来不愿意执行的函数。这是一种最常见的通过计算机网络攻击系统安全的方法。通常，输入给程序一个字符串，这个字符串中包含一些可执行代码的字节编码，称为攻击代码（exploit code），另外一些字节用一个指向攻击代码的指针覆盖返回地址。于是执行`ret`指令的效果就是跳转到攻击代码。

## 对抗缓冲区溢出攻击

现在的编译器和操作系统实现了很多机制以避免遭受缓冲区溢出攻击。本节将介绍一些Linux上最新GCC版本所提供的机制。

### 栈随机化

为了在系统中插入攻击代码，攻击着既要插入代码，又要插入指向这段代码的指针，这个指针也是攻击字符串的一部分。产生这个指针需要直到这个字符串放置的栈地址。在过去，程序的栈地址非常容易预测，对于所有运行同样程序和操作系统版本的系统来说，在不同的机器之间，栈的位置是相当固定的。因此如果攻击者可以确定一个常见的Web服务器所使用的栈空间，就可以设计一个在许多机器上都能实施的攻击。

栈随机化的思想使得栈的位置在程序每次运行时都有变化。实现方式是：程序开始时，在栈上分配一段0～n字节之间的随机大小的空间，例如使用分配函数`alloca`在栈上分配指定字节数量的空间。程序不适用这段空间，但是它会导致程序每次执行时后续的栈位置发生了变化。分配的范围n必须足够大以获得足够多的栈地址变化，同时又要足够小以不浪费程序太多空间。

……

