# 命题

## 命题与连接词

**命题，真值，真命题，假命题，简单命题，复合命题**

| 非                            | $$\urcorner$$       | $$\urcorner p为真\Leftrightarrow p为假$$                    |
| ----------------------------- | ------------------- | ----------------------------------------------------------- |
| 合取，与，且，交，conjunction | $$\land$$           | $$p\land q为真\Leftrightarrow p为真,q为真$$                 |
| 析取，或，                    | $$\lor$$            | $$p\lor q为真\Leftrightarrow p,q有一个为真$$                |
| 蕴涵                          | →                   | $$p→q为假\Leftrightarrow p为真,q为假$$                      |
| 等价                          | $$\leftrightarrow$$ | $$p\leftrightarrow q为真\Leftrightarrow p,q同时为真或为假$$ |
| 与非                          | ↑                   | $$p↑q\Leftrightarrow \urcorner (p\land q)$$                 |
| 或非                          | ↓                   | $$p↓q\Leftrightarrow \urcorner (p\lor q)$$                  |

![jgfr02gj2nfvs](C:\Users\Xiao Yuxuan\Documents\pic\jgfr02gj2nfvs.PNG)



## 命题公式

**命题常项，命题变项，命题公式，赋值，成真赋值，成假赋值，重言式，矛盾式，可满足式，哑元**

**k层公式**

+ 单个命题变项为0层公式
+ A是n+1层公式当：(1) $$A=\urcorner B$$，B是n层公式；(2)$$A=B\land C|B\lor C|B→C|B\leftrightarrow C$$，B,C层数较大者为n

**真值表** 命题公式A在所有赋值下取值情况列成表

![jg3ri0gjnq3nv](C:\Users\Xiao Yuxuan\Documents\pic\jg3ri0gjnq3nv.PNG)

含n个命题变项的公式的真值表只有$$2^{2^n}$$种不同情况



# 命题逻辑

## 等值式

**等值** 命题公式A,B构成的等价式$$A\leftrightarrow B$$为重言式，则称A与B等值，记作$$A\Leftrightarrow B$$

**等值式模式**

+ 双重否定律 $$A\Leftrightarrow \urcorner \urcorner A$$
+ 幂等律 $$A\Leftrightarrow A\land A,~A\Leftrightarrow A\lor A$$
+ 交换律 $$A\lor B \Leftrightarrow B\lor A,~ A\land B \Leftrightarrow B\land A$$
+ 结合律 $$(A\lor B)\lor C\Leftrightarrow A\lor (B\lor C),~(A\land B)\land C\Leftrightarrow A\land (B\land C)$$
+ 分配律 $$A\lor (B\land C)\Leftrightarrow (A\lor B)\land (A \lor C),~A\land (B\lor C)\Leftrightarrow (A\land B)\lor (A \land C)$$
+ 德摩根律 $$\urcorner (A\lor B) \Leftrightarrow \urcorner A\land \urcorner B,~\urcorner (A\land B) \Leftrightarrow \urcorner A\lor \urcorner B$$
+ 吸收律 $$A\lor (A\land B)\Leftrightarrow A,~A\land(A\lor B)\Leftrightarrow A$$
+ 零律 $$A\lor 1 \Leftrightarrow 1,~A\land 0 \Leftrightarrow 0$$
+ 同一律 $$A\lor 0 \Leftrightarrow A,~A\land1\Leftrightarrow A$$
+ 排中律 $$A \lor \urcorner A \Leftrightarrow 1$$
+ 矛盾律 $$A\land \urcorner A \Leftrightarrow 0$$
+ 蕴涵等值式 $$A→B\Leftrightarrow \urcorner A\lor B$$
+ 等价等值式 $$A\leftrightarrow B\Leftrightarrow (A→B)\land (B→A)$$
+ 假言异位 $$A→B\Leftrightarrow \urcorner B→\urcorner A $$
+ 等价否定等值式 $$A\leftrightarrow B\Leftrightarrow \urcorner A \leftrightarrow \urcorner B$$
+ 归谬论 $$(A→B)\land(A→\urcorner B)\Leftrightarrow \urcorner A$$

**代入实例，等值演算**

**置换规则** 设命题公式$$\Phi(A)$$，$$\Phi(B)$$用B替换$$\Phi(A)$$中的A. 若$$B\Leftrightarrow A$$，则$$\Phi(A)\Leftrightarrow \Phi(B)$$



## 析取范式与合取范式

**文字** 命题变项及其否定统称为文字

**简单析取式** 由有限个文字构成的合取式称作简单析取式

**简单合取式** 由有限个文字构成的合取式称作简单合取式

**定理** (1)一个简单析取式是重言式当且仅当它同时包含某个命题变项及其否定；(2)一个简单合取式是矛盾式当且仅当它同时包含某个命题变项及其否定；

**析取范式** 有限个简单合取式的析取构成的命题公式

**合取范式** 有限个简单析取式的合取构成的命题公式

析取范式和合取范式统称**范式**

**定理** (1)一个析取范式是矛盾式当且仅当其每个简单合取式都是矛盾式；(2)一个合取范式是重言式当且仅当它的每个简单析取式都是重言式

**范式存在定理** 任一命题公式都存在与之等值的析取范式和合取范式

步骤：①消去$$→,\leftrightarrow$$；②消去双重否定符，内移否定符；③分配律

**极小项|极大项** 简单合取式|析取式中每个命题变项和其否定恰好出现一个且仅出现一次，且命题变项按照字典序排列，称作简单合取式|析取式的极小项|极大项

n个命题变项共可产生$$2^n$$个极小项，每个极小项有且仅有一个成真赋值，该成真赋值对应的十进制数记作i，极小项记作$$m_i$$；n个命题变项共可产生$$2^n$$个极大项，每个极大项有且仅有一个成假赋值，该成假赋值对应的十进制数记作i，极大项记作$$M_i$$

![fwejig1204nfqccdg](C:\Users\Xiao Yuxuan\Documents\pic\fwejig1204nfqccdg.PNG)

**定理** 设$$m_i和M_i$$是命题变项含$$p_1,p_2,\cdots,p_n$$的极小项和极大项，则$$\urcorner m_i\Leftrightarrow M_i$$

**主析取范式|主合取范式** 所有简单合取式|简单析取式都是极小项|极大项的析取范式|合取范式称为主析取范式|主合取范式；用于求公式的成真赋值|成假赋值，判断公式的类型，判断两个命题公式是否等值

**主范式唯一定理** 任何命题公式都存在与之等值的主析取范式和合取范式，并且是唯一的

**主析取范式与主合取范式** 
$$
A=m_{i_1}\lor m_{i_2}\lor \cdots \lor m_{i_s}\Leftrightarrow M_{j_1}\land M_{j_2}\land \cdots \land M_{j_t}\\
\{i\}\bigcup\{j\}=\{0,\cdots,2^n-1\},\{i\}\bigcap\{j\}=\empty
$$
矛盾式的主合取范式包含全部$$2^n$$个极大项，主析取范式无项，规定为0；重言式的主析取范式包含全部$$2^n$$个极小项，主合取范式无项，规定为1；

n个命题变项共可产生$$2^n$$个极小项，故可产生$$2^{2^n}$$个主析取范式|主合取范式



## 联结词的完备集

**n元真值函数** $$F:\{0,1\}^n→\{0,1\}$$

n元真值函数共有$$2^{2^n}$$个，每个真值函数与唯一的主析取范式|主合取范式等值

![sdfo024tjgin13r](C:\Users\Xiao Yuxuan\Documents\pic\sdfo024tjgin13r.PNG)

**联结词完备集** 任何n元真值函数都可以由仅含S中的联结词构成的公式表示

**定理** $$S=\{\urcorner, \land,\lor \}$$是联结词完备集；$$S_3=\{\urcorner, \land\},S_4=\{\urcorner, \lor\},S_5=\{\urcorner, →\}$$是联结词完备集

**定理** $$\{↑\},\{↓\}$$是联结词完备集

## 可满足性问题与消解法

**空简单析取式，补**

**消解式** 简单析取式$$C_1,C_2$$，$$C_1$$含文字$$l$$，$$C_2$$含文字$$l^c$$，从$$C_1$$中删去$$l$$，从$$C_2$$中删去$$l^c$$，再将所得到的结果析取为一个简单析取式，称其为$$C_1,C_2$$的消解式，记作$${\rm Res}(C_1,C_2)$$

**定理** $$C_1\land C_2\approx {\rm Res}(C_1,C_2)$$：等式两边具有相同的可满足性，但不一定等值

设S是一个合取范式，$$C_1,C_2,\cdots,C_n$$是简单析取式序列. 如果对每一个$$C_i$$是简单析取式或之前的两个简单析取式的消解式，则称此序列为S到处$$C_n$$的消解序列. 当$$C_n=\lambda$$时，此序列是S的一个**否证**. 如果S有否证，则S不可满足

**引理** 设S含有简单析取式$$l$$，从S中删去所有包含$$l$$的简单析取式，再从剩下的简单析取式中删去$$l^c$$，得到合取范式$$S'$$，则$$S\approx S'$$

**消解的完全性定理** 如果合取范式S不可满足，则S有否证



# 推理

**推理，前提，结论**
$$
推理~\Gamma \vdash B\\
正确推理~\Gamma \vDash B\quad 错误推理~\Gamma \nvDash B
$$
**正确推理** 只要不出现$$A_1\land A_2\land \cdots \land A_k=1,B=0$$的情形则推理有效或推理正确 

**定理** $$\{A_1,A_2,\cdots,A_k\}\vDash B$$当且仅当$$A_1\land A_2\land \cdots\land A_k\Rightarrow B$$（$$\Rightarrow$$表示蕴涵式为重言式）

判断推理正确的方法：①真值表法；②等值演算法；③主析取范式法

**推理定律**

1. $$A \Rightarrow (A\lor B)$$						                                   	附加律
2. $$(A\land B)\Rightarrow A$$                                                               化简律
3. $$(A→B)\land A\Rightarrow B$$                                                     假言推理
4. $$(A→B)\land \urcorner B\Rightarrow \urcorner A$$                                                 拒取式
5. $$(A\lor B)\land \urcorner B\Rightarrow A$$                                                     析取三段论
6. $$(A→B)\land (B→C) \Rightarrow (A→C)$$                           假言三段论
7. $$(A\leftrightarrow B)\land (B\leftrightarrow C) \Rightarrow (A\leftrightarrow C)$$                           等价三段论
8. $$(A→B)\land (C→D) \land (A \lor C) \Rightarrow (B\lor D)$$         构造性二难
9. $$(A→B)\land (C→D) \land (\urcorner B \lor \urcorner D) \Rightarrow (\urcorner A\lor \urcorner C)$$ 破坏性二难



# 自然推理系统P

**证明**

**形式系统I**由下面4部分组成：

1. 非空字母表$$A(I)$$
2. $$A(I)$$中符号构造的合式公式集$$E(I)$$
3. $$E(I)$$中一些特殊公式组成的公理集$$A_X(I)$$
4. 推理规则集$$R(I)$$

记$$I$$为4元组$$<A(I),E(I),A_X(I),R(I)>$$，其中$$<A(I),E(I)>$$是**形式语言系统**，$$<A_X(I),R(I)>$$是**形式演算系统**

形式系统分为**自然推理系统**和**公理推理系统**

自然推理系统从任意给定的前提出发，应用系统中的推理规则进行推理演算，最后得到的命题公式是推理的结论；公理推理系统只能从若干条给定的<u>公理</u>出发，因公用系统中的推理规则进行推理演算，得到的结论是系统中的<u>重言式</u>，称为系统中的**定理**

![fw013jtgnrvvwth](C:\Users\Xiao Yuxuan\Documents\pic\fw013jtgnrvvwth.PNG)

![f24i0gjnridnc2t4h](C:\Users\Xiao Yuxuan\Documents\pic\f24i0gjnridnc2t4h.PNG)

**附加前提证明法**

<img src="C:\Users\Xiao Yuxuan\Documents\pic\f2ko04jyhntbvf.PNG" alt="f2ko04jyhntbvf" style="zoom: 67%;" />

**归谬法**

<img src="C:\Users\Xiao Yuxuan\Documents\pic\kokn3i1otgrnvberwg.PNG" alt="kokn3i1otgrnvberwg" style="zoom:67%;" />

# 一阶逻辑

## 命题符号化

**个体词，个体变项，个体常项，个体域**

**谓词，谓词变项，谓词常项，n元谓词，0元谓词，特性谓词**

**量词，全称量词$$\forall$$，存在量词$$\exist$$**

<img src="C:\Users\Xiao Yuxuan\Documents\pic\13tgjfmgr2nbbjyk5oy.PNG" alt="13tgjfmgr2nbbjyk5oy" style="zoom:67%;" />

## 公式

**一阶语言，非逻辑符号，逻辑符号，项，公式**

![vfsnig2j408g2nb](C:\Users\Xiao Yuxuan\Documents\pic\vfsnig2j408g2nb.PNG)

![10jg204gdfsimvo](C:\Users\Xiao Yuxuan\Documents\pic\10jg204gdfsimvo.PNG)

**指导变元，辖域，约束出现，自由出现**

![ohp4yj3p6i8opuy42](C:\Users\Xiao Yuxuan\Documents\pic\ohp4yj3p6i8opuy42.PNG)

**闭式**

![f24u9gjvnfobnhkgr](C:\Users\Xiao Yuxuan\Documents\pic\f24u9gjvnfobnhkgr.PNG)

**解释**

![kofpgk5oyjgvfn34](C:\Users\Xiao Yuxuan\Documents\pic\kofpgk5oyjgvfn34.PNG)



**定理** 封闭的公式在任何解释下都变成命题

![gb0240tjtivkwqrfgrvb](C:\Users\Xiao Yuxuan\Documents\pic\gb0240tjtivkwqrfgrvb.PNG)

![fdso25yp24tgbhyj](C:\Users\Xiao Yuxuan\Documents\pic\fdso25yp24tgbhyj.PNG)

**定理** 重言式的代换实例都是永真式，矛盾式的代换实例都是矛盾式

> 证明可满足式：给出一个成真解释和一个成假解释
>
> 证明永真式|矛盾式：对于任意解释进行证明；证明原命题公式是重言式|矛盾式

## 等值式和置换规则

设A,B是一阶逻辑中任意两个公式，若$$A\leftrightarrow B$$是永真式，则称A与B等值，记作$$A\Leftrightarrow B$$

**等值式模式**

+ 命题逻辑中等值式模式的代换实例都是一阶逻辑的等值式

+ 消去量词等值式

  <img src="C:\Users\Xiao Yuxuan\Documents\pic\640jh3tin31nfevdb.PNG" alt="640jh3tin31nfevdb" style="zoom:67%;" />

+ 量词否定等值式

  <img src="C:\Users\Xiao Yuxuan\Documents\pic\25k0kbfmf2gitbgnmr.PNG" alt="25k0kbfmf2gitbgnmr" style="zoom:67%;" />

+ 量词辖域收缩与扩张等值式![jihto4jyo3n42rfgyj43yh](C:\Users\Xiao Yuxuan\Documents\pic\jihto4jyo3n42rfgyj43yh.PNG)

+ 量词分配等值式![kh35opttnhjbgrwfedcvf](C:\Users\Xiao Yuxuan\Documents\pic\kh35opttnhjbgrwfedcvf.PNG)

+ 置换规则：$$A\Leftrightarrow B \Rightarrow \Phi(A) \Leftrightarrow \Phi(B)$$

+ 换名规则![niojy83t2i2gghjyo4](C:\Users\Xiao Yuxuan\Documents\pic\niojy83t2i2gghjyo4.PNG)

+ 代替规则![kp13t24ygjrirtbyh](C:\Users\Xiao Yuxuan\Documents\pic\kp13t24ygjrirtbyh.PNG)



## 前束范式

**前束范式**

![qlrt2oy35hj75i5u64n](C:\Users\Xiao Yuxuan\Documents\pic\qlrt2oy35hj75i5u64n.PNG)

**前束范式存在定理** 一阶逻辑中的任何公式都存在等值的前束范式

> ![jm4ytn428y5ihtnbj](C:\Users\Xiao Yuxuan\Documents\pic\jm4ytn428y5ihtnbj.PNG)

## 推理理论

蕴涵式$$A_1\land A_2\land \cdots\land A_k→B$$若为永真式，则**推理正确**

**推理定律**

+ 命题逻辑中推理定律的代换实例
+ 基本等值式生成的推理定律
+ 常用的重要推理定律![hj6i4bfverophnu62rq](C:\Users\Xiao Yuxuan\Documents\pic\hj6i4bfverophnu62rq.PNG)
+ 量词消去与引入规则![j3ihngfvbnhroth3](C:\Users\Xiao Yuxuan\Documents\pic\j3ihngfvbnhroth3.PNG)

**自然推理系统**

![vnbhi3y5t2493rjifghn4](C:\Users\Xiao Yuxuan\Documents\pic\vnbhi3y5t2493rjifghn4.PNG)