[toc]

# 数字和算术运算

## 整数

```python
width = 20   # int
i, j = 0, 1  # 多重赋值
# +-*
10 / 3       # 浮点数除法
10 // 3	     # 除法取整
10 % 3       # 除法取余
10 ** 3      # 10^3
```

```python
bin()	0b1100	# 二进制
oct()	0o1472	# 八进制
int()	1989
hex()	0xff00	# 十六进制
```



## 浮点数

```python
f = 1.2e9           # float, 1.2*10^9
PI = 3.14159265359	# 习惯用全大写字母表示常量
```





# 布尔值，布尔运算和比较运算

```python
# 比较运算 > < == >= <=
# 布尔运算 and or not
3 > 2			 # True
3 > 2 or 3 > 4	 # True
not 3 > 2    	 # False
A and not B or C # 优先级: not > and > or,因此等价于(A and (not B)) or C
a < b == c       # 两次比较

# 任意非零数值，非空字符串，非空数组都为True, 数值0为False
# and和or是短路运算符，它们从左至右解析参数，一旦可以确定结果解析就会停止
# 比较序列对象时使用字典序
```



比较操作符`in`和`not in`校验一个值是否在/不在一个序列里

```python
i = 1
list = [1,2,3,4]
if i in list
```



操作符`is`和`is not`比较两个对象是不是同一个对象：

```python
>>> a = 1
>>> b = 1
>>> a == b    # a,b的值相等
True
>>> a is b    # 且指向同一个对象
True
>>> id(a)
10914496
>>> id(b)
10914496
>>> a = 100000000
>>> b = 100000000
>>> a == b
True
>>> a is b    # 当a,b的值较大时也不再成立
False
```

```python
>>> a = []
>>> b = []
>>> a == b
True
>>> a is b
False
```





# 字符串

```python
print("abc")    # 可以使用双引号或单引号
print('abc')

print('"Yes," they said.')    # 单/双引号中的双/单引号无需转义
print("\"Yes,\" they said.")  # 单/双引号中的单/双引号需要转义

print(r'C:\some\name')        # 单引号前的r表示原始字符串方式，即不转义

print('包含中文的str')   # python3的字符串以Unicode编码，支持多语言

# 跨行输入
print('''line1
         line2
         line3''')
```



**转义字符**

| \\'  | ‘      | \\\  | \    |
| ---- | ------ | ---- | ---- |
| \\"  | “      | %%   | %    |
| \n   | 换行   |      |      |
| \t   | 制表符 |      |      |



## 字符串方法

### 示例

```python
>>> a = 3 * 'IN' + 'no' + 'Vation'
>>> a
'INININnoVation'
>>> len(a)                  # 字符串长度
14
>>> a[0]                    # 字符串可视作列表，进行索引和切片操作
'I'
>>> a[:6]
'INININ'
>>> a.replace('a', 'A')     # 替换字符
'INININnoVAtion'
>>> a.lower()               # 英文字符小写
'inininnovation'
>>> a.upper()               # 英文字符大写
'INININNOVATION'
>>> a.capitalize()          # 首个英文字符大写，后面小写
'Inininnovation'
>>> a.count('IN')           # 计数子串的出现次数
3
>>> a.startswith('INININ')  # 前缀比较
True
>>> a.endswith('tion')      # 后缀比较
True
>>> a.find('IN')            # 从前往后寻找子串的位置索引
0
>>> a.rfind('IN')           # 从后往前寻找子串的位置索引
4
>>> a.split('N')            # 拆分字符串
['I', 'I', 'I', 'noVation']
>>> 'N'.join(['I', 'I', 'I', 'noVation'])   # 拼接字符串
'INININnoVation'
```



## unicode编解码

Python3使用Unicode编码字符串，因此Python的字符串支持多语言：

```python
>>> print('English中文にほんご')
English中文にほんご
```

对于单个字符的编码，Python提供了`ord()`和`chr()`函数用于转换字符和Unicode编码：

```python
>>> hex(ord('A'))    # 字符 to 编码
'0x41'
>>> hex(ord('中'))
'0x4e2d'
>>> chr(0x03B1)      # 编码 to 字符
'α'
>>> chr(0x51EA)
'凪'
```

Unicode字符与转义的Unicode编码是等价的：

```python
>>> '\u4e2d文'
'中文'
```

Python字符串（`str`类型对象）在内存中以Unicode表示，如果要保存到磁盘上或者在网络上传输，就需要用ascii或者utf-8编码为字节序列。





# 空值

```python
None
```





# 类型转换

```python
int()	     # 转换为整数类型
float()    # 浮点
str()      # 字符串
```





