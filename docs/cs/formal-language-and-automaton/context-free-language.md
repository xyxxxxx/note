# 上下文无关文法

**定义** 如果所有的产生式都满足如下形式，文法 $G=(V,T,S,P)$ 称为**上下文无关（context-free）**文法

$$
A\to x
$$

其中 $A\in V，x\in (V\cup T)^*$。上下文无关文法生成的语言称为上下文无关语言。

正则文法都是上下文无关文法，因此正则语言也是上下文无关语言，正则语言族是上下文无关语言族的真子集。

线性文法也都是上下文无关文法。

@线性文法 $G=(\{S\},\{a,b\},S,P)$，产生式

$$
S\to aSa\\
S\to bSb\\
S\to \lambda
$$

生成的上下文无关语言为

$$
L(G)=\{ww^R:w\in\{a,b\}^*\}
$$

@证明语言 $L=\{a^nb^m:n\neq m\}$ 是上下文无关的，方法是构造产生这种语言的上下文无关文法：

$$
S\to AS_1|S_1B\\
S_1\to aS_1b|\lambda\\
A\to Aa|a\\
B\to Bb|b
$$

该文法不是线性文法，尽管对于该问题也能构造出线性文法。

## 最左推导和最右推导

在非线性的上下文无关文法中，推导可能包括由多个变量构成的句型。

**定义** 如果每次都替换句型中最左端的变量，那么这种推导称为**最左的（leftmost）**；如果每次都替换最右端的变量，那么这种推导称为**最右的（rightmost）**。

## 推导树

另一种表示推导过程的方式称为**推导树（derivation tree）**，如图所示
$$
A\to abABc
$$
![Screenshot from 2020-09-11 15-04-53.png](https://i.loli.net/2020/09/11/Ci4U6BhLEP7orcO.png)

**定义** 上下文无关文法的推导树……（略）

**定理** 设 $G=(V,T,S,P)$ 是上下文无关文法，那么 $G$ 的推导树的果与 $L(G)$ 中的符号串 $w$ 一一对应。

> 证明略。

# 分析

**分析（parsing）**指寻找 $w\in L(G)$ 的推导过程中使用的一系列产生式。

**穷举搜索分析（exhaustive search parsing）**由下例说明

@已知文法 $S\to SS|aSb|bSa|\lambda$ 和符号串 $w=aabb$，第一回合得到
$$
S\Rightarrow SS\\
S\Rightarrow aSb\\
S\Rightarrow bSa\\
S\Rightarrow \lambda
$$
推导1,2匹配，进入队列。第二回合得到
$$
S\Rightarrow SS\Rightarrow SSS  \\
S\Rightarrow SS\Rightarrow aSbS  \\
S\Rightarrow SS\Rightarrow bSaS  \\
S\Rightarrow SS\Rightarrow S  \\
$$
推导1,2匹配且不重复，进入队列。第三回合得到
$$
S\Rightarrow SS\Rightarrow SSS  \\
S\Rightarrow SS\Rightarrow SaSb  \\
S\Rightarrow SS\Rightarrow SbSa  \\
S\Rightarrow SS\Rightarrow S  \\
$$
推导2匹配且不重复，进入队列……最终从推导中得到目标符号串
$$
S\Rightarrow aSb \Rightarrow aaSbb \Rightarrow aabb
$$
因此 $w=aabb$ 属于该文法的生成语言

穷举搜索分析存在严重的缺陷，首先是效率低，其次是若推导没有得到目标符号串，那么分析可能永远都不会停止。

如果能够消除 $A\to \lambda$ 以及 $A\to B$ 形式的产生式，将能够在不影响文法表达能力的基础上确保返回结果并提高效率。

**定理** 假设 $G=(V,T,S,P)$ 是上下文无关文法，并且不具备下面形式的产生式
$$
A\to \lambda\\
A\to B
$$
其中 $A,B\in V$，那么穷举搜索分析法可以生成一个算法，它能对于任意 $w\in \Sigma^*$ 生成一种分析，或者返回不可能。

> 证明略。

尽管如此，穷举搜索分析法的时间复杂度依然是指数量级。对上下文无关文法更有效的分析方法参见编译原理，这里直接给出结论。

**定理** 每个上下文无关文法都存在一个算法，它分析任意 $w\in L(G)$ 的时间复杂度是 $\Theta(|w|^3)$ 

尽管此算法比指数时间复杂度的算法好些，但以此为基础的编译器对于中等长度的程序依然要花费大量的时间分析。我们想要线性时间的算法，但只有在上下文无关文法的特殊形式中才能找到这样的算法。

**定义** 上下文无关文法 $G=(V,T,S,P)$ 是**简单文法（simple grammar）**，如果它的产生式的形式都是
$$
A\to ax
$$
其中 $A\in V,a\in V,x\in V^*$，并且任何 $(A,a)$ 对至多出现一次。

@文法
$$
S\to aS|bSS|c
$$
是简单文法，而文法
$$
S\to aS|bSS|aSS|c
$$
不是简单文法，因为 $(S,a)$ 对既出现在 $S\to aS$ 中，又出现在 $S\to aSS$ 中。

如果 $G$ 是简单文法，那么 $L(G)$ 中的任何符号串 $|w|$ 都可以在 $\Theta(|w|)$ 的时间分析完。

## 二义性

**定义** 如果对某个 $w\in L(G)$ 至少存在两个不同的推导树，那么称上下文无关文法 $G$ 是**二义性的（ambiguous）**。即存在两个或两个以上的最左或最右推导。

@产生式为 $S\to aSb|SS|\lambda$ 的文法有二义性，因为句子 $aabb$ 有两棵推导树，如下图

![Screenshot from 2020-09-11 16-33-00.png](https://i.loli.net/2020/09/11/UW4XFOEm1tfd3rM.png)

在程序设计语言中，每条语句只应该有一个解释，即需要去除二义性，通常采用重写一个等价的、无二义性的文法来达到目的。

@已知文法 $G=(V,T,S,P)$，其中
$$
V=\{E,I\}\\
T=\{a,b,c,+,*,(,)\}
$$
产生式为
$$
E\to I\\
E\to E+E\\
E\to E*E\\
E\to(E)\\
I\to a|b|c
$$
这个文法存在二义性，例如符号串 $a+b*c$ 有两棵推导树

![Screenshot from 2020-09-11 16-39-11.png](https://i.loli.net/2020/09/11/MnbVuaF97UePjmX.png)

重写文法，使 $V=\{E,T,F,I\}$，产生式为
$$
E\to T\\
T\to F\\
F\to I\\
E\to E+T\\
T\to T*F\\
F\to (E)\\
I\to a|b|c
$$
这个文法无二义性，符号串 $a+b*c$ 的只有一棵推导树

![Screenshot from 2020-09-11 16-44-47.png](https://i.loli.net/2020/09/11/bW63VYRtyuAh48J.png)

**定义** **固有二义性（inherently ambiguous）**的上下文无关语言的每个生成文法都是二义性的，无二义性的上下文无关语言存在一个生成文法是非二义性的。

@证明语言
$$
L=\{a^nb^nc^m\}\cup\{a^nb^mc^m\}
$$
是固有二义性的上下文无关语言，其中 $n,m$ 是非负整数。

给出生成该语言的一个上下文无关文法
$$
S_1\to S_1c|A\\
A\to aAb|\lambda\\
S_2\to aS_2|B\\
B\to bBc|\lambda\\
S\to S_1|S_2
$$
这个文法是二义性的，因为符号串 $a^nb^nc^n$ 有两棵推导树，但不能因此推断 $L$ 是固有二义性的。这个命题的严格证明参见其它文献。

# 上下文无关文法和程序设计语言

形式语言理论的一个最重要的应用就是程序设计语言的定义和它们的解释器、编译器的构造。正则语言可以为简单的程序设计语言建模，而更复杂的情况需要上下文无关语言。

# 文法变换方法

首先考虑空串，在很多情形下空串表现为一种特殊情况，需要予以特别考虑，因此为了化简我们选择消除空串。

设上下文无关语言 $L$ 满足 $\lambda \notin L$，对应的上下文无关文法为 $G=(V,T,S,P)$，如果在 $V$ 中增加变量 $S_0$，使它作为开始变量，并增加产生式
$$
S_0\to S|\lambda
$$
这样得到的新文法 $\hat G$ 产生的语言即为 $L\cup \{\lambda\}$。鉴于此，在实际应用中，我们认为包含与不包含 $\lambda$ 的文法并没有区别。

## 代入简化

很多代入方法能够用于生成等价文法，这里将给出一种。**化简（simplification）**的粗略定义是消除某些不必要的产生式，但实际应用中并不一定导致产生式数量的减少。

**定理** 设上下文无关文法 $G=(V,T,S,P)$， $P$ 中包含一个形如
$$
A\to x_1Bx_2
$$
的产生式，其中 $A$ 与 $B$ 是不同的变量并且
$$
B\to y_1|y_2|\cdots|y_n
$$
是 $P$ 中所有以 $B$ 为左部的产生式的集合。现在替代 $P$ 中的产生式
$$
A\to x_1Bx_2 \Rightarrow A\to x_1y_1x_2|x_1y_2x_2|\cdots|x_1y_nx_2
$$
得到 $\hat P$，并构造上下文无关文法 $\hat G=(V,T,S,\hat P)$，那么
$$
L(\hat G)=L(G)
$$

> 证明略。

## 删除无用产生式

**定义** 设上下文无关文法 $G=(V,T,S,P)$，变量 $A\in V$ 是**有用的（useful）**，当且仅当至少存在一个 $w\in L(G)$ 使得
$$
S\Rightarrow^*xAy\Rightarrow^*w
$$
其中 $x,y\in (V\cup T)^*$。换言之，某个变量是有用的当且仅当它至少在一个符号串的推导中出现。如果某个产生式包含无用变量，则该产生式是无用的。

@变量无用的第一种情形为 $xAy\Rightarrow^*w$ 不存在，例如某个文法的产生式为
$$
S\to aSb|\lambda|A\\
A\to aA
$$
变量 $A$ 无法被消除，因而无法推导出任何符号串。变量 $A$ 是无用的，因此 $S\to A,A\to aA$ 也是无用的。

@变量无用的第二种情形为 $S\Rightarrow^*xAy$ 不存在，例如某个文法的产生式为

$$
S\to A\\
A\to aA|\lambda \\
B\to bA
$$
变量 $B$ 无法产生，因而没有任何作用。变量 $B$ 是无用的，因此 $B\to bA$ 也是无用的。

@消除文法 $G=(V,T,S,P)$ 中的无用符号与产生式，其中 $V=\{S,A,B,C\},T=\{a,b\}$，产生式 $P$ 如下
$$
S\to aS|A|C\\
A\to a\\
B\to aa\\
C\to aCb
$$
我们给出变量**依赖图（dependency graph）**来查看各变量之间的关系

图

可以看到变量 $B,C$ 是无用的，删除相关变量、终结符和产生式，得到最终文法 $\hat G=(\hat V,\hat T,S,\hat P)$，其中 $\hat V=\{S,A\},\hat T=\{a\},\hat P$ 如下
$$
S\to aS|A\\
A\to a
$$

**定理** 设上下文无关文法 $G=(V,T,S,P)$，则存在一个与之等价的文法 $\hat G=(\hat V,\hat T,S,\hat P)$，其不包含任何的无用符号或产生式。

> 证明略。

## 消除 $\lambda$ 产生式

**定义** 上下文无关文法中任意形如
$$
A\to \lambda
$$
的产生式称为** $\lambda$ 产生式（ $\lambda$ -production）**。任何存在推导
$$
A\Rightarrow^*\lambda
$$
的变量 $A$ 称为**可空的（nullable）**。

既然文法是否包含 $\lambda$ 已不重要，我们考虑消除所有的 $\lambda$ 产生式。

@考虑文法
$$
S\to aS_1b\\
S_1\to aS_1b|\lambda
$$
通过用 $\lambda$ 替换产生式右部的 $S_1$ 以替换产生式 $S_1\to \lambda$ 
$$
S\to aS_1b|ab\\
S_1\to aS_1b|ab
$$

**定理** 设 $G$ 是任意的上下文无关文法并且 $\lambda\notin L(G)$，则存在一个等价的不包含 $\lambda$ 产生式的文法 $\hat G$。

> 证明略。

## 消除单位产生式

**定义** 上下文无关文法中形如
$$
A\to B
$$
的产生式称为单位产生式，其中 $A,B\in V$。

使用代入简化方法即可消除单位产生式。

**定理** 设 $G=(V,T,S,P)$ 是一个任意的上下文无关文法且不包含 $\lambda$ 产生式，则存在一个上下文无关文法 $\hat G=(\hat V,\hat T,S,\hat P)$，它与 $G$ 等价且不存在单位产生式。

> 证明略。

@消除文法
$$
S\to Aa|B\\
A\to a|bc|B\\
B\to A|bb\\
$$
中的所有单位产生式。

原文法中的非单位产生式为
$$
S\to Aa\\
A\to a|bc\\
B\to bb
$$
代入原文法中的单位产生式
$$
S\to B\to bb\\
S\to B\to A\to a|bc\\
S\to B\to A\to B\to {\rm exist}\\
A\to B\to bb\\
A\to B\to A\to {\rm exist}\\
B\to A\to a|bc\\
B\to A\to B\to {\rm exist}
$$
合并得到
$$
S\to a|bb|bc|Aa\\
A\to a|bb|bc\\
B\to a|bb|bc
$$

综合前述的所有化简方法，我们有以下结论。

**定理** 设 $L$ 是不包含 $\lambda$ 的上下文无关语言，则存在一个产生 $L$ 的上下文无关文法，该文法不含无用产生式、 $\lambda$ 产生式或单位产生式。

# 乔姆斯基范式和格里巴克范式

**范式（normal form）**指一种受到限制但又具备足够表达能力的文法形式，任意文法都可以表示成相应的等价范式。

## 乔姆斯基范式

乔姆斯基范式严格限制产生式右部的符号数目。

**定义** 如果一个上下文无关文法的所有产生式都形如
$$
A\to BC\\
A\to a
$$
其中 $A,B,C\in V$， $a\in V$，则该文法属于乔姆斯基范式。

@文法
$$
S\to AS|a\\
A\to SA|b
$$
是乔姆斯基范式。

**定理** 对于任意的上下文无关文法 $G=(V,T,S,P)$， $\lambda\notin L(G)$，都存在一个等价的上下文无关文法 $\hat G=(\hat V,\hat T,S,\hat P)$ 属于乔姆斯基范式。

> 证明略。

@将具有产生式
$$
S\to ABa\\
A\to aab\\
B\to Ac
$$
的文法转换为乔姆斯基范式。

第一步，引入新的变量
$$
S\to ABB_a\\
A\to B_aB_aB_b\\
B\to AB_c\\
B_a\to a\\
B_b\to b\\
B_c\to c
$$
第二步，引入另外两个变量将前两个产生式转换为范式
$$
S\to AD_1\\
D_1\to BB_a\\
A\to B_aD_2\\
D_2\to B_aB_b\\
B\to AB_c\\
B_a\to a\\
B_b\to b\\
B_c\to c
$$

## 格里巴克范式

格里巴克范式限制终结符与变量可以出现的位置。

**定义** 如果一个上下文无关文法的所有产生式都形如
$$
A\to ax
$$
其中 $a\in T$ 且 $x\in V^*$，则该文法属于格里巴克范式。

@将具有产生式
$$
S\to abSb|aa
$$
的文法转换为格里巴克范式。
$$
S\to aBSB|aA\\
A\to a\\
B\to b
$$
@将具有产生式
$$
S\to AB\\
A\to aA|bB|b\\
B\to b\\
$$
的文法转换为格里巴克范式。

代入即得到
$$
S\to aAB|bBB|bB\\
A\to aA|bB|b\\
B\to b\\
$$

# 上下文无关文法的成员资格判定算法

我们之前已经指出，上下文无关文法的成员资格判定算法需要 $|w|^3$ 步才能完成符号串 $w$ 的分析，这里我们将使用CYK算法验证这一命题。

CYK算法的使用条件是上下文无关文法必须满足乔姆斯基范式。给定满足乔姆斯基范式的文法 $G=(V,T,S,P)$，对于符号串
$$
w=a_1a_2\cdots a_n
$$
我们定义其子串
$$
w_{ij}=a_i\cdots a_j
$$
以及 $V$ 的子集
$$
V_{ij}=\{A\in V:A\Rightarrow^* w_{ij} \}
$$
显然， $w\in L(G) \iff S\in V_{1n}$ 

注意到 $A\in V_{ii}$ 当且仅当 $G$ 中存在产生式 $A\to a_i$，因此通过检查产生式即可确定 $V_{ii}$。接下来注意到 $A\in V_{ij}$ 当且仅当存在产生式 $A\to BC$，存在 $i\le k<j$，有 $B\Rightarrow^*w_{ik}$ 且 $C\Rightarrow^*w_{k+1,j}$ ……因此CYK算法是动态规划的迭代方法，需要计算 $n(n+1)/2$ 个 $V_{ij}$ 集合，每个集合需要对每个 $k$ 的取值进行验证，即计算 $O(n)$ 次，总的时间复杂度为 $O(n^3)$。

@确定符号串 $w=aabbb$ 是否属于由文法
$$
S\to AB\\
A\to BB|a\\
B\to AB|b
$$
生成的语言。
$$
V_{11}=\{A\},V_{22}=\{A\},V_{33}=\{B\},V_{44}=\{B\},V_{55}=\{B\}\\
V_{12}=\varnothing,V_{23}=\{S,B\},V_{34}=\{A\},V_{45}=\{A\}\\
V_{13}=\{S,B\},V_{24}=\{A\},V_{35}=\{S,B\}\\
V_{14}=\{A\},V_{25}=\{S,B\}\\
V_{15}=\{S,B\}
$$
因此 $w\in L(G)$ 

# 上下文无关语言的性质

正则语言的泵引理为证明某些语言不是正则语言提供了一种有效手段。对其它语言族也存在着类似的泵引理。

## 上下文无关语言的泵引理

**定理** 设 $L$ 是一个无穷上下文无关语言，则存在一个正整数 $m$ 使得对于任意满足 $|w|\ge m$ 的 $w\in L$ 都能被分解为
$$
w=uvxyz
$$
其中
$$
|vxy|\le m\\
|vy|\ge 1
$$
则对于所有的 $i=0,1,2,\cdots$ 满足
$$
uv^ixy^iz\in L
$$

> 证明略。

@证明语言
$$
L=\{a^nb^nc^n:n\ge 0\}
$$
不是上下文相关的。

假设 $L$ 是上下文相关的，那么泵引理成立。取 $w=a^mb^mc^m$，若 $vxy=a^k,1\le k\le m$，那么 $vy=a^i,1\le i\le k$ 
$$
uxz=a^{m-i}b^mc^m\notin L
$$
若 $vxy=a^kb^l,k\ge 1,l\ge 1,k+l\le m$，那么 $vy=a^ib^j,1\le i+j\le k+l$ 
$$
uxz=a^{m-i}b^{m-j}c^m\notin L
$$
其它情形同理，因此假设不成立， $L$ 不是上下文相关的。

@证明语言
$$
L=\{ww:w\in\{a,b\}^*\}
$$
不是上下文相关的。

假设 $L$ 是上下文相关的，那么泵引理成立。取 $w=a^mb^ma^mb^m$，若 $vxy=a^k,1\le k\le m$，那么 $vy=a^i,1\le i\le k$ 
$$
uxz=a^{m-i}b^ma^mb^m\notin L
$$
若 $vxy=a^kb^l,k\ge 1,l\ge 1,k+l\le m$，那么 $vy=a^ib^j,1\le i+j\le k+l$ 
$$
uxz=a^{m-i}b^{m-j}a^mb^m\notin L
$$
其它情形同理，因此假设不成立， $L$ 不是上下文相关的。

## 线性语言的泵引理

**定义** 对于一个上下文无关语言，如果存在线性上下文无关文法 $G$ 满足 $L=L(G)$，则该语言称为线性语言。

**定理** 设 $L$ 是一个无穷线性语言，则存在一个正整数 $m$ 使得对于任意满足 $|w|\ge m$ 的 $w\in L$ 都能被分解为
$$
w=uvxyz
$$
其中
$$
|uvyz|\le m\\
|vy|\ge 1
$$
则对于所有的 $i=0,1,2,\cdots$ 满足
$$
uv^ixy^iz\in L
$$

> 证明略。

@证明语言
$$
L=\{w:n_a(w)=n_b(w)\}
$$
不是线性的。

假设 $L$ 是线性的，那么泵引理成立。取 $w=a^mb^{2m}a^m$，由于 $|uvyz|\le m$， $uvyz=a^k,0\le k\le m$，那么 $vy=a^{i+j},1\le i+j\le k$ 
$$
uxz=a^{m-i}b^{2m}a^{m-j}\notin L
$$
因此假设不成立， $L$ 不是线性的。

## 上下文无关语言的封闭性和判定算法

**定理** 上下文无关语言族对并运算、连接运算和闭包运算是封闭的。

> 证明略。

**定理** 上下文无关语言族对交运算和补运算是不封闭的。

> 证明略。

**定理** 设上下文无关语言 $L_1$，正则语言 $L_2$，则 $L_1\cap L_2$ 是一个上下文无关语言。

> 证明略。

@证明语言 $L=\{a^nb^n:n\ge 0,n\neq 100\}$ 是上下文无关的。
$$
上下文无关语言 L_1=L=\{a^nb^n:n\ge 0\}\\
正则语言L_2=\{a^{100}b^{100}\}\\
L=L_1\cap \overline{L}_2
$$
因此 $L$ 是上下文无关语言。

@证明语言 $L=\{w\in\{a,b,c\}^*:n_a(w)=n_b(w)=n_c(w) \}$ 不是上下文无关的。

假设 $L$ 是上下文无关的，则
$$
L\cap L(a^*b^*c^*)=\{a^nb^nc^n:n\ge 0\}
$$
也是上下文无关的，但我们已经使用泵引理证明其不是上下文无关的，因此假设不成立， $L$ 不是上下文无关的。

**定理** 给定一个上下文无关文法 $G=(V,T,S,P)$，存在判定 $L(G)$ 是否为空的算法。

**定理** 给定一个上下文无关文法 $G=(V,T,S,P)$，存在判定 $L(G)$ 是否有穷的算法。

> 证明略。