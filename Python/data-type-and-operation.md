# 数字和算术运算

## 整数

```python
width = 20   # int
i, j = 0, 1  # 多重赋值
# +-*
10 / 3    #浮点数除法
10 // 3	  #除法取整
10 % 3    #除法取余
10 ** 3   #10^3
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
PI = 3.14159265359	#习惯用全大写字母表示常量
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



操作符`is`和`is not`比较两个对象是不是同一个对象

```python
i = None
if i is None:
    print("none")
```





# 字符串

```python
print("abc")    #可以使用双引号或单引号
print('abc')

print('"Yes," they said.')   #单/双引号中的双/单引号无需转义
print("\"Yes,\" they said.") #单/双引号中的单/双引号需要转义

print(r'C:\some\name')       #单引号前的r表示原始字符串方式，即不转义

print('包含中文的str')	#python3的字符串以Unicode编码，支持多语言

#跨行输入
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

```python
a = 3 * 'IN' + 'no' #'INININno'
a = a 'Vation'      #'INININnoVation'
a.replace('a','A')	#'INININnoVAtion'
a.lower()			#'inininnovation'
a.upper()			#'INININNOVATION'
a.capitalize()		#'Inininnovation'
b = ('KZ.'
     a)             #'KZ.Inininnovation'

a[0]                # I
#字符串可视作列表，可切片，详见数据结构
len(a)              # 14
```



## unicode编解码

```python
#ord()字符→编码,chr()编码→字符
ord('A')	#65
ord('中')	#20013
chr(66)		#B
chr(25991)	#文
```





# 空值

```python
None
```





# 数据转换

```python
int()	#转换为整数类型
```





