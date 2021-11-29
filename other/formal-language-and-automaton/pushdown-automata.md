# 非确定型下推自动机

![cvnmjkofndjidop35.PNG](https://i.loli.net/2020/09/13/7PURnvfCelihr4p.png)

下推自动机如图所示，其中控制部件的每次从输入文件中读入一个符号，改变栈的内容，并发生状态转移，每次状态转移由当前输入符号和当前栈顶符号共同决定。

**定义** **非确定型下推自动机（nondeterministic pushdown automata, NPDA）**由一个七元组定义
$$
M=(Q,\Sigma,\Gamma,\delta,q_0,z,F)
$$
其中

+ $Q$ 是控制部件内部状态的有穷集
+ $\Sigma$ 是输入字母表
+ $\Gamma$ 是要给有穷符号集，称为**栈字母表（stack alphabet）**
+ $\delta:Q\times (\Sigma\cup \{\lambda\})\times \Gamma \to Q\times \Gamma ^*$ 的有穷子集，称为状态转移函数
+ $q_0\in Q$ 是控制部件的初始状态
+ $z\in \Gamma$ 是**栈开始符（stack start symbol）**
+ $F\sube Q$ 是终止状态集合

$\delta$ 的参数是控制部件的当前状态，当前输入符号和当前栈顶符号， $\delta$ 的返回结果是 $(q,x)$ 对，其中 $q$ 是控制部件的下一个状态， $x$ 是对栈顶符号的更新。注意到 $\delta$ 的第二个参数可以是 $\lambda$，它表明自动机的一次迁移可以不需要输入符号，我们称为 $\lambda$ 转移。状态转移必须存在一个栈顶符号；栈顶符号为 $\lambda$ 时的状态转移函数没有定义。

设对于NPDA有三元组
$$
(q,w,u)
$$
其中 $q$ 是控制部件的状态， $w$ 是输入符号串的未读入部分， $u$ 是栈的当前内容，我们称该三元组是下推自动机的**瞬时描述（instantaneous description）**，从一个瞬时描述到另一个瞬时描述的迁移用符号 $\vdash$ 表示，这样
$$
(q_1,aw,bx)\vdash (q_2,w,yx) \iff (q_2,y)\in \delta(q_1,a,b)
$$



## 下推自动机接受的语言

**定义** 设NPDA $M=(Q,\Sigma,\Gamma,\delta,q_0,z,F)$，则 $M$ 接受的语言是集合
$$
L(M)=\{w\in \Sigma^*:(q_0,w,z)\vdash_M^*(p,\lambda,u),p\in F,u\in \Gamma^*\}
$$


@例如NPDA
$$
Q=\{q_0,q_1,q_2,q_3\}\\
\Sigma=\{a,b\}\\
\Gamma=\{0,1\}\\
z=0\\
f=\{q_3\}
$$
状态转移函数
$$
\delta(q_0,a,0)=\{(q_1,10),(q_3,\lambda)\}\\
\delta(q_0,\lambda,0)=\{(q_3,\lambda)\}\\
\delta(q_1,a,1)=\{(q_1,11)\}\\
\delta(q_1,b,1)=\{(q_2,\lambda)\}\\
\delta(q_2,b,1)=\{(q_2,\lambda)\}\\
\delta(q_2,\lambda,0)=\{(q_3,\lambda)\}\\
$$
注意到状态转移函数并没有对所有情形都做了定义，与NFA相同，此种情形的值为空集。

此NPDA的关键转移是
$$
\delta(q_1,a,1)=\{(q_1,11)\}
$$
它表示读到 $a$ 时在栈顶增加一个1，以及
$$
\delta(q_2,b,1)=\{(q_2,\lambda)\}
$$
它表示读到 $b$ 时在栈顶删除一个1。通过分析其它状态转移函数，我们看到NPDA接受的语言是
$$
L=\{a^nb^n:n\ge 0\}\cup\{a\}
$$


@构造一个NPDA能够接受语言
$$
L=\{w\in \{a,b\}^*:n_a(w)=n_b(w)\}
$$
思路是使用0和1分别为未配对的 $a,b$ 计数。方案是 $M=(\{q_0,q_f\},\{a,b\},\{0,1,z\},\delta,q_0,z,\{q_f\})$，其中 $\delta$ 为
$$
\delta(q_0,\lambda,z)=\{(q_f,z)\}\\
\delta(q_0,a,z)=\{(q_0,0z)\}\\
\delta(q_0,b,z)=\{(q_0,1z)\}\\
\delta(q_0,a,0)=\{(q_0,00)\}\\
\delta(q_0,b,0)=\{(q_0,\lambda)\}\\
\delta(q_0,a,1)=\{(q_0,\lambda)\}\\
\delta(q_0,b,1)=\{(q_0,11)\}\\
$$


@构造一个NPDA能够接受语言
$$
L=\{ww^R:w\in\{a,b\}^+ \}
$$
使用栈能够轻松解决对称匹配的问题，非确定性能够匹配 $w$ 和 $w^R$ 分界的每一个可能位置。方案是 $M=(Q,\Sigma,\Gamma,\delta,q_0,z,F)$，其中
$$
Q=\{q_0,q_1,q_2\}\\
\Sigma=\{a,b\}\\
\Gamma=\{a,b,z\}\\
F=\{q_2\}\\
压栈\quad\delta(q_0,a,a)=\{(q_0,aa)\}\\
\delta(q_0,b,a)=\{(q_0,ba)\}\\
\delta(q_0,a,b)=\{(q_0,ab)\}\\
\delta(q_0,b,b)=\{(q_0,bb)\}\\
\delta(q_0,a,z)=\{(q_0,az)\}\\
\delta(q_0,b,z)=\{(q_0,bz)\}\\
无条件转移到出栈\quad\delta(q_0,\lambda,a)=\{(q_1,a)\}\\
\delta(q_0,\lambda,b)=\{(q_1,b)\}\\
出栈\quad\delta(q_1,a,a)=\{(q_1,\lambda)\}\\
\quad\delta(q_1,b,b)=\{(q_1,\lambda)\}\\
无条件转移到终止状态\quad\delta(q_1,\lambda,z)=\{(q_2,z)\}\\
$$






# 下推自动机与上下文无关语言

**定理** 对于任何的上下文无关语言 $L$，存在一个NPDA $M$ 使得
$$
L=L(M)
$$

>  证明略。



@构造一个PDA能够接受具有如下产生式的文法生成的语言
$$
S\to aSbb|a
$$
将该文法转换为格里巴克范式，得到的新产生式为
$$
S\to aSA|a\\
A\to bB\\
B\to b
$$
首先将开始符号压栈
$$
\delta(q_0,\lambda,z)=\{(q_1,Sz)\}
$$
然后模拟产生式
$$
\delta(q_1,a,S)=\{(q_1,SA),(q_1,\lambda) \}\\
\delta(q_1,b,A)=\{(q_1,B)\}\\
\delta(q_1,b,B)=\{(q_1,\lambda)\}\\
$$
最后进入终态
$$
\delta(q_1,\lambda,z)=\{(q_2,\lambda)\}
$$




**定理** 对于任何一个NPDA $M$，如果 $L=L(M)$，则 $L$ 是上下文无关语言。

> 证明略。





# 确定型下推自动机和确定型上下文无关语言

**确定型下推自动机（deterministic pushdown automata, DPDA）**是没有迁移选择的下推自动机。

**定义** 一个下推自动机 $M=(Q,\Sigma,\Gamma,\delta,q_0,z,F)$ 称为确定型的，如果它在非确定型下推自动机的定义的基础上满足：对于任意 $q\in Q,a\in \Sigma\cup\{\lambda\},b\in\Gamma$ 有

1. $\delta(q,a,b)$ 最多包含一个元素
2. 如果 $\delta(q,\lambda,b)$ 非空，则对于每个 $c\in \Sigma$， $\delta(q,c,b)$ 都必须为空

第一个条件要求对于任意给定的输入符号与栈顶符号，最多只能执行一个转移；第二个条件说明某一格局如果存在无条件转移，则不能有读入输入符号的转移。即任何时候都至多存在一种可能的迁移。



**定义** 语言 $L$ 是确定型**上下文无关语言（deterministic context-free language）**当且仅当存在一个DPDA $M$ 满足 $L=L(M)$ 



@语言
$$
L=\{a^nb^n:n\ge 0\}
$$
是确定型上下文无关语言，因为DPDA $M=(\{q_0,q_1,q_2\},\{a,b\},\{0,1\},\delta,q_0,0,\{q_0\})$ 能够接受这个语言，其中 $M$ 的产生式为
$$
\delta(q_0,a,0)=\{(q_1,10)\}\\
\delta(q_1,a,1)=\{(q_1,11)\}\\
\delta(q_1,b,1)=\{(q_2,\lambda)\}\\
\delta(q_2,b,1)=\{(q_2,\lambda)\}\\
\delta(q_2,\lambda,0)=\{(q_0,\lambda)\}\\
$$





