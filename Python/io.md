# 交互窗口I/O

## 输出

```python
print('abc')	#输出字符串
print('abc','def','gh')	#输出多个字符串
print(100)	#输出整数
print(100+100)	#输出计算结果

print('Hi, %s, you have $%d.' % ('Michael', 1000000))	#输出变量
print('Hi, {0}, you have ${1}.' .format('Michael', 1000000))
```

| 占位符 | 类型         |
| ------ | ------------ |
| %d     | 整数         |
| %f     | 浮点数       |
| %s     | 字符串       |
| %x     | 十六进制整数 |



## 输入

```python
print('Please enter your name:')
name=input()	#输入字符串
print('Hello,',name)

```





# 文件读写

## Read

```python
f = open('/Users/michael/test.txt', 'r')	#'r'读
f.read()		#返回读取内容
f.close()

with open('/Users/michael/test.txt', 'r') as f:	#常用
    print(f.read())

f = open('/Users/michael/test.jpg', 'rb')	#rb读二进制
f = open('/Users/michael/gbk.txt', 'r', encoding='gbk')	#以gbk编码读取,默认为UTF-8


```

```python
#read函数
read()		#读取文件全部内容
read(size)	#一次读取size字节的内容
readline()	#一次读取一行内容
readlines()	#一次读取所有内容并按行返回list
```



## Write

```python
f = open('/Users/michael/test.txt', 'w')	#w写,a追加
f.write('Hello, world!')
f.close()

with open('/Users/michael/test.txt', 'w') as f:	#常用
    f.write('Hello, world!')
```





# 流

## StringIO

```python
from io import StringIO
f = StringIO()		#创建
f.write('hello')	#写入
f.write(' ')
f.write('world!')
print(f.getvalue())	#获取

#亦可用读取文件的方式读取字符串流
```



## BytesIO

```python
from io import BytesIO
f = BytesIO()
f.write('中文'.encode('utf-8'))	#传入编码
print(f.getvalue())

#亦可用读取文件的方式读取字节流
```





# 操作文件和目录





# 序列化



