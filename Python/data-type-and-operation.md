# 数据类型

## 整数

```python
bin()	0b1100	#二进制数
oct()	0o1472	#八进制数
int()	1989
hex()	0xff00	#十六进制数
```



## 浮点数

```python
1.2e9	#1.2*10^9
1.2e-5	#1.2*10^-5
```



## 字符串

```python
print('包含中文的str')	#python3的字符串以Unicode编码，支持多语言

#ord()字符→编码,chr()编码→字符
ord('A')	#65
ord('中')	#20013
chr(66)		#B
chr(25991)	#文

a='INnoVation'
a.replace('a','A')	#'INnoVAtion'
a.lower()			#'innovation'
a.upper()			#'INNOVATION'
a.capitalize()		#'Innovation'
```

### str和bytes转换



### 转义字符

| \\'  | ‘      | \\\  | \    |
| ---- | ------ | ---- | ---- |
| \\"  | “      | %%   | %    |
| \n   | 换行   |      |      |
| \t   | 制表符 |      |      |



```python
print('I\'m OK.')	#转义字符
print(r'\t\n\\')	#默认不转义，引号不可

print('''line1	#打印多行
line2
line3''')

```



## 布尔值和布尔运算

```python
3>2			#True
3>2 or 3>4	#True
not True	#False
#任意非零数值，非空字符串，非空数组都为True, 数值0为False
```



## 空值

```python
None
```



## 数据转换

```python
int()	#转换为整数类型
```



## 其他

### 赋值

```python
a=123
b='ABC'

```

### 常量

```python
PI=3.14159265359	#习惯用全大写字母表示常量
```





# 运算

## 赋值

```python
a=1		#无需声明类型，但必要时需要转换
b=int(input())

a,b,c=1,2,3	#多重赋值
a,b=b,a+b	#无需临时变量
```



## 算术运算

```python
# +-*/
10//3	#除法取整
10%3	#除法取余
10**3	#10^3
```

