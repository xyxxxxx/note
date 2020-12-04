# shell I/O

## 输出

```python
>>> print('abc')              # 输出字符串
abc
>>> print('abc','def','gh')   # 输出多个字符串,间隔一个空格
abc def gh
>>> print(100)                # 输出数
100
>>>
>>> print('Hi, %s, you have $%d.' % ('Michael', 1000000))         # C风格格式化输出
Hi, Michael, you have $1000000.
>>> print('Hi, {0}, you have ${1}.' .format('Michael', 1000000))  # 格式化输出(推荐)
Hi, Michael, you have $1000000.                                   # {0} 对应第0个参数,自动识别类型
>>>
>>> name = 'IBM'
>>> shares = 100
>>> price = 91.1
>>> print('{:s} {:d} {:.2f}'.format(name, shares, price))         # 格式化输出
IBM 100 91.10                                                     # {:s} 对应一个%s参数
>>> print(f'{name:>10s} {shares:>10d} {price:>10.2f}')            # f''格式化输出
       IBM        100      91.10
```

| 占位符 | 类型         |
| ------ | ------------ |
| %d     | 整数         |
| %b     | 二进制整数   |
| %x     | 十六进制整数 |
| %f     | 浮点数       |
| %s     | 字符串       |
| %c     | ASCII字符    |
|        |              |

| 位置调整 |                            |
| -------- | -------------------------- |
| `:>10d`  | 整数在10字符长的区间右对齐 |
| `:<10d`  | 整数在10字符长的区间左对齐 |
| `:^10d`  | 整数在10字符长的区间居中   |

```shell
# 格式化输出浮点数
>>> value = 42863.1
>>> print(value)
42863.1
>>> print('%0.4f' % value)            # C风格格式化输出
42863.1000
>>> print('%16.2f' % value)
        42863.10
>>>        
>>> print('{:0.4f}'.format(value))    # 格式化输出(推荐)
42863.1000
>>> print('{:16.2f}'.format(value))
        42863.10
>>> print('{:<16.2f}'.format(value))
42863.10        
>>> print('{:*>16.2f}'.format(value))
********42863.10
>>>
>>> print(f'{value:0.4f}')            # f''格式化输出
42863.1000
>>> print(f'{value:>16.2f}')
        42863.10
>>> print(f'{value:<16.2f}')
42863.10
>>> print(f'{value:*>16.2f}')
*******42,863.10


```







## 输入

```python
print('Please enter your name:')
name = input()	      # 输入字符串
print('Hello,',name)

```





# 文件读写

```python
f = open('foo.txt', 'rt')	# r for reading (text)
g = open('bar.txt', 'wt')	# w for write (text), a for append

data = f.read()		 # 返回读取文本
g.write('some text') # 写入文本

f.close()
g.close()

with open('foo.txt', 'rt') as f:	        # 代码块结束时将自动关闭该文件
    print(f.read())

with open('foo.txt', 'rt') as f:
    for line in f:                          # 逐行读取,实际更常用
        pass                                # 尤其是当读取文件较大，或每一行都需要单独处理时

with open('outfile', 'wt') as out:
    out.write('Hello World\n')    
    
with open('outfile', 'wt') as out:
    print('Hello World', file=out)          # 重定向print函数
    
f = open('test.jpg', 'rb')	                # rb读二进制
f = open('gbk.txt', 'r', encoding='gbk')	# 以gbk编码读取,默认为UTF-8
```

```python
# read函数
read()		# 读取文件全部内容
read(size)	# 一次读取size字节的内容
readline()	# 一次读取一行内容
readlines()	# 一次读取所有内容并按行返回list
```





# 流

## StandardIO

```python
sys.stdout  # 标准输出
sys.stderr  # 标准错误
sys.stdin   # 标准输入
```

默认地，`print`定向至`sys.stdout`，traceback和error定向至`sys.stderr`，输入从`sys.stdin`读取。

```shell
$ python3 prog.py > results.txt  # stdio连接到文件
```



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



