## 元字符

| char                             | quantifiers          | position           |
| -------------------------------- | -------------------- | ------------------ |
| \d 数字                          | * 任意个             | ^ 起始             |
| \D 非数字                        |                      | $ 结束             |
| \w 字母或数字或汉字              | + 至少1个            | \b 单词边界        |
| \W 非字母或数字                  | ?  0个或1个          | \B 非单词边界      |
| \s 任意空白字符, 即[ \f\n\r\t\v] | {n} n个              | 单词: 连续的\w子串 |
| \S 任意非空白字符                | {min,max} min至max个 |                    |
| .    任意字符(除\n)              | {min,}               |                    |
|                                  | {,max}               |                    |
|                                  | ~？懒惰匹配后缀      |                    |



## 转义符号

|      |      |      |      |
| ---- | ---- | ---- | ---- |
| \\-  | \\_  | \\,  | \\.  |
| \\?  | \\!  | \\*  | \\+  |
| \\^  | \\$  | \\(  | \\[  |
| \\{  | \\\  |      |      |



## 字符类[]

```python
[0-9a-zA-Z\_] 				#1个数字、字母或者下划线
[0-9a-zA-Z\_]+				#1个或多个~
[a-zA-Z\_][0-9a-zA-Z\_]*	#首字符为字母或下划线,即py合法变量名

[^x]						#除x的任意字符
```



## 分枝条件|

```python
(P|p)ython					#python or Python
0\d{2}-\d{8}|0\d{3}-\d{7}	#027-87654321 or 0717-7654321

```



## 重复(){n}



## 分组

```
\b(\w+)\b\s+\1\b		#go go
#(\w+)将\w+捕获至分组1,之后的\1调用捕获内容

\b(?<Word>\w+)\b\s+\k<Word>\b	#自定义分组名


```

**分组语法**

| **捕获** | (exp)        | 匹配exp,并捕获文本到自动命名的组里             |
| -------- | ------------ | ---------------------------------------------- |
|          | (?<name>exp) | 匹配exp,并捕获文本到名称为name的组里           |
|          | (?:exp)      | 匹配exp,不捕获匹配的文本，也不给此分组分配组号 |
| 零宽断言 | (?=exp)      | 匹配exp前面的位置                              |
|          | (?<=exp)     | 匹配exp后面的位置                              |
|          | (?!exp)      | 匹配后面不是exp的位置                          |
|          | (?<!exp)     | 匹配前面不是exp的位置                          |
| 注释     | (?#comment)  |                                                |



## 零宽断言

### 正向零宽断言

```python
#后条件
\b\w+(?=ing\b)		#read, listen
^(?=.*\d)(?=.*[a-z])(?=.*[A-Z])[a-zA-Z0-9]{8,10}$
#起始位置之后必须是.*\d(任意字符+数字),.*[a-z]和.*[A-Z]

#前条件
(?<=\bmono)\w+\b	#rail, poly
```

### 负向零宽断言

```python
#后禁止
\d{3}(?!\d)

#前禁止
(?<![a-z])\d{7}
```



## 贪婪和懒惰匹配

```python
#*+为贪婪匹配
re.match(r'^(\d+)(0*)$', '102300').groups()
	#('102300', '')

#接?以懒惰匹配    
re.match(r'^(\d+?)(0*)$', '102300').groups()
	#('1023', '00')
    
```



## 处理选项

```python
IgnoreCase 忽略大小写
```



## 常用正则表达式

```python
\n\s*\r				#空白行
[\u4e00-\u9fa5]		#中文字符

((?:(?:25[0-5]|2[0-4]\\d|[01]?\\d?\\d)\\.){3}(?:25[0-5]|2[0-4]\\d|[01]?\\d?\\d))	#ip地址
\d{3}-\d{8}|\d{4}-\d{7}		   #国内电话号码
^(?=.*\d)(?=.*[a-z])(?=.*[A-Z])[a-zA-Z0-9]{8,10}$	#强密码(必须包含大小写字母,数字)

```



## Python re

```python
import re

#match 起始位置检查
if re.match(r'^\d{3,4}\-\d{3,8}$', '0716-8834387'):
    print('success')
else:
    print('failure')

#search 任意位置检查

#findall 检查所有匹配子串

#sub替换
tel = "2004-959-559"
tel = re.sub(r'\D', "", tel)

#split
re.split(r'[\s\,]+','a,b  c,   d')

#groups提取子串
>>> m = re.match(r"(\w+) (\w+)", "Isaac Newton, physicist")
>>> m.group(0)       # The entire match
'Isaac Newton'
>>> m.group(1)       # The first parenthesized subgroup.
'Isaac'
>>> m.group(2)       # The second parenthesized subgroup.
'Newton'
>>> m.group(1, 2)    # Multiple arguments give us a tuple.
('Isaac', 'Newton')
    
#compile对频繁使用的正则表达式预编译
re_tel = re.compile(r'^(\d{3,4})-(\d{3,8})$')
re_tel.match('010-12345').groups()

```
