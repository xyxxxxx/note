# 正则表达式

## 简易教程

### 元字符，限定符，位置符

| char                                 | quantifiers              | position           |
| ------------------------------------ | ------------------------ | ------------------ |
| `\d` 数字                            | `*` 任意个               | `^` 起始           |
| `\D` 非数字                          | `+` 至少1个              | `$` 结束           |
| `\w` 字母或数字或汉字或下划线        | `?`  0个或1个            | `\b` 单词边界      |
| `\W` 非字母或数字或汉字或下划线      | `{n}` n个                | `\B` 非单词边界    |
| `\s` 任意空白字符, 即`[ \f\n\r\t\v]` | `{min,max}` min至max个   | 单词: 连续的\w子串 |
| `\S` 任意非空白字符                  | `{min,}`       至少min个 |                    |
| `.`   任意字符(除\n)                 | `{,max}`       至多      |                    |
|                                      | `~?` 懒惰匹配后缀        |                    |

### 转义符号

| `\-`       | `\_` | `\,`       | `\.`       |
| ---------- | ---- | ---------- | ---------- |
| `\?`       | `\!` | `\*`       | `\+`       |
| `\^`       | `\$` | `\(`, `\)` | `\[`, `\]` |
| `\{`, `\}` | `\\` |            |            |

### 字符类[]

```
[aeiou]                     # 'a','e','i','o','u'中的任意一个
[0-9]                       # 等同于\d
[0-9a-zA-Z_]                # 1个数字、字母或者下划线
[0-9a-zA-Z_]+               # 1个或多个~
[a-zA-Z\_][0-9a-zA-Z\_]*    # 首字符为字母或下划线,即Python的合法变量名

[^x]                        # 除'x'的任意字符
[^aeiou]                    # 除'a','e','i','o','u'的任意字符
```

### 分枝条件|

#### 局部分枝

```
(P|p)ython                  # 'python' 或 'Python'
3\.(6|7|8)                  # '3.6', '3.7' 或 '3.8'
```

#### 全局分枝

```
0\d{2}-\d{8}|0\d{3}-\d{7}   # '027-87654321' 或 '0717-7654321'
```

#### 可选分枝

```
info(|rmation)              # 'info' 或 'information'
```

### 分组

```
(\d{1,3}\.){3}\d{1,3}           # 简单的IP地址匹配
((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)   # 完整的IP地址匹配

\b(\w+)\b\s+\1\b		          # go go
#(\w+)将\w+捕获至分组1,之后的\1调用捕获内容

\b(?<Word>\w+)\b\s+\k<Word>\b	# 自定义分组名
```

**分组语法**

| **捕获** | `(exp)`        | 匹配exp，并捕获文本到自动命名的组里             |
| -------- | -------------- | ----------------------------------------------- |
|          | `(?<name>exp)` | 匹配exp，并捕获文本到名称为name的组里           |
|          | `(?:exp)`      | 匹配exp，不捕获匹配的文本，也不给此分组分配组号 |
| 零宽断言 |                |                                                 |
|          | `(?<=exp)`     | 匹配前面是exp的位置                             |
|          | `(?=exp)`      | 匹配后面是exp的位置                             |
|          | `(?<!exp)`     | 匹配前面不是exp的位置                           |
|          | `(?!exp)`      | 匹配后面不是exp的位置                           |
| 注释     | `(?#comment)`  |                                                 |

### 零宽断言

零宽表示并不匹配任何字符，断言表示验证条件是否为真。

#### 正向零宽断言

```
# 后条件
\b\w+(?=ing\b)      # 从' reading listening '匹配'read', 'listen'

^(?=.*\d)(?=.*[a-z])(?=.*[A-Z])[a-zA-Z0-9]{8,10}$  # 强密码(必须包含大小写字母和数字的组合,不能使用特殊字符,长度在 8-10 之间)
# 起始位置之后的3个正向零宽断言分别验证包含了数字,小写字母和大写字母

# 前条件
(?<=\bmono)\w+\b	# 从' monorail monopoly '匹配'rail', 'poly'
```

类似于`(?<=A|B|C)`的正向零宽断言可以写为`(?:(?<=A)|(?<=B)|(?<=C))`，例如：

```python

```

#### 负向零宽断言

```
# 后禁止
\d{3}(?!\d)         # 匹配3位数字且之后不能是数字

# 前禁止
(?<![a-z])\d{7}     # 匹配3位数字且之前不能是小写字母 
```

类似于`(?<!A|B|C)`的负向零宽断言可以写为`(?<!A)|(?<!B)|(?<!C)`，例如：

```

```

#### 正向或负向零宽断言(look-around)

```python
>>> re.sub(r'\s+(?=[^\[\(]*\))|(?<=\()\s+', '', 'abcd (  ()e(e w  )f ) gh')
'abcd (()e(ew)f) gh'
```

### 贪婪和懒惰匹配

```python
# *+为贪婪匹配
>>> re.match(r'^(\d+)(0*)$', '102300').groups()
('102300', '')         # \d+直接匹配到结束
>>> re.match(r'^(\d*)(0*)$', '102300').groups()
('102300', '')

# *,+,?,{n,m}接?以懒惰匹配,即尽可能少匹配
>>> re.match(r'^(\d+?)(0*)$', '102300').groups()
('1023', '00')         # \d+?仅匹配'1023',剩余的'00'留给0*
    
```

### 处理选项

```python
IgnoreCase 忽略大小写
```

### 常用正则表达式

空白行

```
\n\s*\r
```

中文字符

```
[\u4e00-\u9fff]
```

ip地址

```
((?:(?:25[0-5]|2[0-4]\\d|[01]?\\d?\\d)\\.){3}(?:25[0-5]|2[0-4]\\d|[01]?\\d?\\d))
```

国内电话号码

```
\d{3}-\d{8}|\d{4}-\d{7}
```

强密码（必须包含大小写字母、数字）

```
^(?=.*\d)(?=.*[a-z])(?=.*[A-Z])[a-zA-Z0-9]{8,10}$
```

HTTP/HTTPS URL

```
https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)
```

## 常用工具

### Python 标准库：re

参见 [re——正则表达式操作](../../python/standard-library/re.md)。

## 练习

```python
>>> re.search('[\u4e00-\u9fff]', '中文')       # 匹配中文(中日韩统一表意文字)
<re.Match object; span=(0, 1), match='中'>

# 去掉
>>> 
'aa那就4哦撒01加上rw那就嗯'
```
