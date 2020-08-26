# 导入模块

```python
import math	#导入math包
x=math.sin(math.pi/2)
```

> Python标准库https://docs.python.org/zh-cn/3/library/index.html





# 定义模块

```python
#!/usr/bin/env python3			#标准注释:py3文件
# -*- coding: utf-8 -*-			#标准注释:使用UTF-8编码

' a test module '				#文档注释

__author__ = 'Michael Liao'		#作者名

import sys						#正文

def test():
    args = sys.argv
    if len(args)==1:
        print('Hello, world!')
    elif len(args)==2:
        print('Hello, %s!' % args[1])
    else:
        print('Too many arguments!')

if __name__=='__main__':
    test()
```





# 作用域

```python
#abc		public变量
#_abc		public变量，但惯例不直接引用
#__abc		private变量，不可直接引用
#__abc__	特殊变量，可以直接引用

def _private_1(name):		#内部函数
    return 'Hello, %s' % name

def _private_2(name):		
    return 'Hi, %s' % name

def greeting(name):			#外部接口
    if len(name) > 3:
        return _private_1(name)
    else:
        return _private_2(name)
```


