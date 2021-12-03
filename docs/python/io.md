[toc]

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

## 文本文件

读文本文件：

```python
# foo.txt:
# abcdefg
# 1234567
# 

>>> f = open('foo.txt', 'rt')   # 打开文件并返回文件描述符(文件句柄),标示符r表示read,t表示text
>>> f.read()                    # 一次读取所有文本内容
'abcdefg\n1234567\n'
>>> f.close()                   # 使用完毕后关闭文件
>>> f = open('notfound.txt', 'rt')
FileNotFoundError: [Errno 2] No such file or directory: 'notfound.txt'
```

写文本文件：

```python
>>> g = open('bar.txt', 'wt')	  # 标示符w表示write
>>> g.write('first line\n')
11
>>> g.close()                   # 调用close()方法之后才能保证数据全部写入磁盘,否则可能部分丢失
>>> g = open('bar.txt', 'wt')	  # 每次write写入都会覆盖文件,即丢弃原有的文件内容
>>> g.write('second line\n')
12
>>> g.close()
>>> g = open('bar.txt', 'at')	  # 标示符a表示append,append写入会在原有的文件内容末尾追加
>>> g.write('third line\n')
11
>>> g.close()

# bar.txt:
# second line
# third line
# 
```

上面的这种写法十分繁琐，并且经常容易忘记调用`f.close()`关闭文件，或者因为在读写过程中出错导致文件没有正确关闭，因此实践中通常使用上下文管理器来确保关闭文件：

```python
>>> with open('foo.txt', 'rt') as f:        # 代码块结束时将自动关闭该文件
...     print(f.read())
... 
abcdefg
1234567

>>> with open('foo.txt', 'rt') as f:
...     for line in f:                      # 逐行读取,实际更常用
...         print(line)                     # 尤其是当读取文件较大，或每一行都需要单独处理时
... 
abcdefg

1234567
 
>>> with open('bar.txt', 'wt') as out:
...     out.write('Hello World\n')
... 
12
>>> with open('bar.txt', 'wt') as out:
        print('Hello World', file=out)      # 重定向print函数
```

比较读文件的几种方式：

```python
# foo.txt:
# abcdefg
# 1234567
# 

>>> with open('foo.txt', 'rt') as f:
...     print(f.read())                   # 一次读取所有内容
... 
abcdefg
1234567

>>> with open('foo.txt', 'rt') as f:
...     line = f.readline()               # 一次读取一行内容,最节省内存,但花费时间最长
...     while line:
...         print(line)
...         line = f.readline()
... 
abcdefg

1234567

>>> with open('foo.txt', 'rt') as f:
...     lines = f.readlines()             # 一次读取所有内容并保存为一个列表,每行为一个元素,适用于逐行处理
...     for line in lines:
...         print(line)
... 
abcdefg

1234567

```



## 字符编码

```python
f = open('gbk.txt', 'r', encoding='gbk')	  # 以gbk编码读取,默认为UTF-8
```



## 二进制文件

```python
f = open('test.jpg', 'rb')	                # 标示符b表示binary
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

**序列化(serialization)**是指将数据结构或对象转换成字节序列的过程。反过来，将字节序列还原为数据结构的过程就是**反序列化(deserialization)**。

Python提供了`pickle`模块来实现Python对象结构的二进制序列化和反序列化。

```python
>>> import pickle
>>> d = dict(name='Bob', age=20, score=88)
>>> pickle.dumps(d)
b'\x80\x03}q\x00(X\x03\x00\x00\x00ageq\x01K\x14X\x05\x00\x00\x00scoreq\x02KXX\x04\x00\x00\x00nameq\x03X\x03\x00\x00\x00Bobq\x04u.'
>>> with open('foo', 'wb') as f:
...   pickle.dump(d, f)
... 
>>> with open('foo', 'rb') as f:
...   d1 = pickle.load(f)
... 
>>> d1
{'name': 'Bob', 'age': 20, 'score': 88}
```